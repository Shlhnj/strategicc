"""
strategicc/stockflow/aggregation.py -- v3.11
-------------------------------------------------
Aggregates per-iteration Stock & Flow outputs across iterations and by
state class, producing per-class-per-year stock and flow totals suitable
for Mode C SEEA-EA valuation.

For each timestep: load that stock type's raster from every iteration,
compute the per-cell MEDIAN across iterations, mask by the modal LULC
class for that timestep, sum within each class's cells.

Flow totals are aggregated from flow_log.csv (median total per flow_type
per year across iterations) -- flows are scalar aggregates already, not
per-cell rasters, so no class masking is needed (eligibility was already
class-gated by the engine at simulation time via FromStateClassId).

v3.11: aggregate_flow_by_class() now tolerates an empty-but-present
flow_log_by_class.csv (written unconditionally by engine._save_flow_log()
for any iteration with no by_class records) instead of crashing on
pd.errors.EmptyDataError.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from strategicc.io.csv_loader import StateClass


def aggregate_stock_by_class(
    iter_dirs:   list[Path],
    stock_types: list[str],
    classes:     dict[int, StateClass],
    modal_maps:  dict[int, np.ndarray],
    start_year:  int,
    n_timesteps: int,
) -> pd.DataFrame:
    """
    Aggregate per-iteration stock rasters into per-class-per-year totals.

    Returns
    -------
    DataFrame with columns: year, class_id, class_name, stock_type, total
    """
    rows = []
    total_steps = n_timesteps + 1

    for stock_type in stock_types:
        for t in range(total_steps):
            year = start_year + t
            if year not in modal_maps:
                continue
            modal = modal_maps[year]

            stack = []
            for d in iter_dirs:
                tif = d / "stocks" / stock_type / f"stock_{year}.tif"
                if tif.exists():
                    arr = np.array(Image.open(str(tif)), dtype=np.float32)
                    stack.append(arr)

            if not stack:
                continue

            cube   = np.stack(stack, axis=0)
            median = np.median(cube, axis=0)

            for cid, sc in classes.items():
                class_mask = (modal == cid)
                total = float(median[class_mask].sum())
                rows.append({
                    "year":       year,
                    "class_id":   cid,
                    "class_name": sc.name,
                    "stock_type": stock_type,
                    "total":      total,
                })

    return pd.DataFrame(rows)


def aggregate_flow_by_class(
    iter_dirs: list[Path],
) -> pd.DataFrame:
    """
    Aggregate per-iteration per-class flow logs into median
    total-per-class-per-flow-type-per-year.

    Reads flow_log_by_class.csv from each iteration directory.

    Returns
    -------
    DataFrame with columns: year, class_name, flow_type, from_stock,
    to_stock, total (median across iterations)
    """
    frames = []
    for d in iter_dirs:
        log_path = d / "flow_log_by_class.csv"
        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
            except pd.errors.EmptyDataError:
                # An iteration with no by_class flow records writes a
                # zero-column/empty file (engine._save_flow_log() always
                # writes it unconditionally) -- skip rather than crash.
                continue
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=[
            "year", "class_name", "flow_type", "from_stock", "to_stock", "total"
        ])

    combined = pd.concat(frames, ignore_index=True)
    group_cols = ["iteration", "year", "class_name", "flow_type", "from_stock", "to_stock"]
    grouped = (
        combined.groupby(group_cols)["amount"]
        .sum()
        .reset_index()
    )
    median_cols = ["year", "class_name", "flow_type", "from_stock", "to_stock"]
    median = (
        grouped.groupby(median_cols)["amount"]
        .median()
        .reset_index()
        .rename(columns={"amount": "total"})
    )
    return median


def aggregate_flow_by_type(
    iter_dirs: list[Path],
) -> pd.DataFrame:
    """
    Aggregate per-iteration flow logs into median total-per-flow-type-per-year
    (landscape-wide, no class breakdown). Kept for diagnostic/summary use;
    Mode C SEEA valuation uses aggregate_flow_by_class() instead.

    Returns
    -------
    DataFrame with columns: year, flow_type, total (median across iterations)
    """
    frames = []
    for d in iter_dirs:
        log_path = d / "flow_log.csv"
        if log_path.exists():
            frames.append(pd.read_csv(log_path))

    if not frames:
        return pd.DataFrame(columns=["year", "flow_type", "total"])

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["iteration", "year", "flow_type"])["total_amount"]
        .sum()
        .reset_index()
    )
    median = (
        grouped.groupby(["year", "flow_type"])["total_amount"]
        .median()
        .reset_index()
        .rename(columns={"total_amount": "total"})
    )
    return median


def build_asset_account(
    stock_df: pd.DataFrame,
    flow_df:  pd.DataFrame,
    stock_types: list[str],
    classes:     dict,
    start_year:  int,
    n_timesteps: int,
) -> pd.DataFrame:
    """
    Build a SEEA-EA-style asset account per stock type per class per year,
    following the standard structure:

        Opening balance
        + Additions   (all flows where this stock type is the TO side)
        - Reductions  (all flows where this stock type is the FROM side)
        = Closing balance (reconciled, i.e. Opening + Additions - Reductions)

    Year 1's Opening balance is the stock's initial (t=0) value. Each
    subsequent year's Opening balance equals the PRIOR year's reconciled
    Closing balance (the standard SEEA-EA rollforward).

    Because Additions/Reductions are derived from the MEDIAN of
    flow_log_by_class.csv across iterations, while the actual stock
    raster total (stock_df) is separately aggregated as its own MEDIAN,
    the two will not algebraically reconcile perfectly in a stochastic
    Monte Carlo setting (median of sums != sum of medians). Rather than
    silently picking one as "true", this function reports BOTH:

        closing_balance_reconciled : Opening + Additions - Reductions
        closing_balance_actual     : the real median stock_df total for
                                      that year (from the stock rasters)
        reconciliation_diff        : actual - reconciled

    A small reconciliation_diff is expected statistical noise from
    Monte Carlo aggregation; a LARGE one may indicate a real bug (e.g. a
    flow pathway not properly captured in flow_log_by_class.csv).

    Parameters
    ----------
    stock_df    : output of aggregate_stock_by_class()
                  (year, class_id, class_name, stock_type, total)
    flow_df     : output of aggregate_flow_by_class()
                  (year, class_name, flow_type, from_stock, to_stock, total)
    stock_types : list of stock type names to build accounts for
    classes     : dict[int, StateClass]
    start_year  : first simulation year
    n_timesteps : total number of timesteps

    Returns
    -------
    DataFrame with columns:
        stock_type, class_name, year,
        opening_balance, additions, reductions,
        closing_balance_reconciled, closing_balance_actual,
        reconciliation_diff
    """
    rows = []
    years = [start_year + t for t in range(n_timesteps + 1)]
    class_names = [sc.name for sc in classes.values()]

    for stock_type in stock_types:
        for class_name in class_names:

            opening_balance = None   # set from t=0 actual stock total

            for year in years:
                # Actual stock total for this year (from stock rasters)
                match = stock_df[
                    (stock_df["stock_type"] == stock_type)
                    & (stock_df["class_name"] == class_name)
                    & (stock_df["year"] == year)
                ]
                actual_closing = float(match["total"].sum()) if not match.empty else 0.0

                if opening_balance is None:
                    # First year: opening balance = initial actual stock
                    opening_balance = actual_closing
                    additions  = 0.0
                    reductions = 0.0
                else:
                    # Additions: flows where this stock is the TO side
                    add_match = flow_df[
                        (flow_df["class_name"] == class_name)
                        & (flow_df["to_stock"] == stock_type)
                        & (flow_df["year"] == year)
                    ]
                    additions = float(add_match["total"].sum()) if not add_match.empty else 0.0

                    # Reductions: flows where this stock is the FROM side
                    red_match = flow_df[
                        (flow_df["class_name"] == class_name)
                        & (flow_df["from_stock"] == stock_type)
                        & (flow_df["year"] == year)
                    ]
                    reductions = float(red_match["total"].sum()) if not red_match.empty else 0.0

                closing_reconciled = opening_balance + additions - reductions
                diff = actual_closing - closing_reconciled

                rows.append({
                    "stock_type":                  stock_type,
                    "class_name":                  class_name,
                    "year":                        year,
                    "opening_balance":             opening_balance,
                    "additions":                   additions,
                    "reductions":                  reductions,
                    "closing_balance_reconciled":  closing_reconciled,
                    "closing_balance_actual":      actual_closing,
                    "reconciliation_diff":         diff,
                })

                # Next year's opening = THIS year's reconciled closing
                # (standard SEEA-EA rollforward -- using the reconciled
                # value, not the actual, keeps the account internally
                # consistent year over year even though actual stock
                # totals are reported alongside for validation)
                opening_balance = closing_reconciled

    return pd.DataFrame(rows)


def stock_account_seea(
    stock_df:    pd.DataFrame,
    flow_df:     pd.DataFrame,
    stock_types: list[str],
    classes:     dict,
    start_year:  int,
    n_timesteps: int,
) -> dict[str, pd.DataFrame]:
    """
    Physical stock account in SEEA EA Table 13.3 layout (v3.20) — one
    table per stock type (a stock account is compiled per substance/
    reservoir, matching how Table 13.3 itself separates categories like
    Geocarbon/Biocarbon rather than mixing them), each with one block
    of rows per accounting period:

        Opening stock
        Additions to stock
        Reductions in stock
        Net carbon balance   (Additions - Reductions)
        Closing stock

    columns = classes + Total. This collapses Table 13.3's own
    sub-breakdown of Additions (Unmanaged/Managed expansion,
    Discoveries, Reclassifications, Imports) and Reductions (Unmanaged/
    Managed contraction, Reclassifications, Exports, Catastrophic
    losses) into single Additions/Reductions rows, since STRATEGICC's
    flow log has no source for that finer classification — the row set
    used here is the "basic form" the manual itself describes in
    para. 13.63 ("opening stock, additions, reductions and closing
    stock"), verified directly against table 13.3.

    Unlike build_asset_account() (the flat, database-style ledger this
    package already produced, kept as-is for its own diagnostic value —
    it carries closing_balance_actual/reconciliation_diff, useful for
    validation, that Table 13.3 itself has no equivalent for), this
    function is a presentation layer matching the standard's own row/
    column/period convention, the same one extent_account_seea() and
    monetary_asset_account_seea() already use for their own tables.

    Opening/Closing come directly from stock_df's actual per-year
    totals (never drift, since they're not rolled forward from a prior
    reconciled figure — each period's Opening/Closing is read fresh
    from the real stock rasters). Additions/Reductions come from
    flow_df, classified the same way build_asset_account() does
    (to_stock==stock_type -> addition, from_stock==stock_type ->
    reduction), looked up at the period's CLOSING year — matching
    build_asset_account()'s own (tested) year convention, not
    extent_account_seea()'s opening-year one; the two account builders
    use flow_df's `year` column differently and this follows
    build_asset_account()'s. Since v3.20's class-to-class stock
    carryover fix (see run_flows_for_timestep()), Additions/Reductions
    now include stock that moved between classes via land conversion,
    not just modeled flow pathways, so Net balance should track
    Closing - Opening closely for every class, not only ones with a
    modeled flow pathway of their own.

    Net balance is NOT forced to reconcile exactly against
    Closing - Opening (no residual row absorbs the difference, unlike
    monetary_asset_account_seea()'s Enhancement/Degradation) — any
    remaining gap is genuine Monte Carlo noise from aggregating medians
    across iterations (see build_asset_account()'s own docstring for
    the same caveat) and is left visible rather than hidden.

    Parameters
    ----------
    stock_df, flow_df, stock_types, classes, start_year, n_timesteps :
        same as build_asset_account().

    Returns
    -------
    dict[stock_type, DataFrame], each DataFrame with a MultiIndex
    (Period, Entry) on the rows and columns [class_name, ..., Total].
    Written to CSV/XLSX this reads as one block of rows per accounting
    period, in Table 13.3's own row order.
    """
    class_names = [sc.name for sc in classes.values()]
    years = [start_year + t for t in range(n_timesteps + 1)]
    if len(years) < 2:
        raise ValueError(
            "stock_account_seea() needs at least two years of stock "
            "data to form one accounting period."
        )

    result: dict[str, pd.DataFrame] = {}

    for stock_type in stock_types:

        def _stock_totals(year: int) -> pd.Series:
            match = stock_df[
                (stock_df["stock_type"] == stock_type)
                & (stock_df["year"] == year)
            ]
            s = match.groupby("class_name")["total"].sum()
            return s.reindex(class_names).fillna(0.0)

        def _flow_totals(year: int, side: str) -> pd.Series:
            # side: "to" (additions) or "from" (reductions), matching
            # build_asset_account()'s own to_stock/from_stock == stock_type
            # classification, looked up at the period's CLOSING year.
            col = "to_stock" if side == "to" else "from_stock"
            match = flow_df[
                (flow_df[col] == stock_type) & (flow_df["year"] == year)
            ]
            s = match.groupby("class_name")["total"].sum()
            return s.reindex(class_names).fillna(0.0)

        rows: list[tuple[tuple[str, str], dict]] = []
        for i in range(len(years) - 1):
            y0, y1 = years[i], years[i + 1]
            period = f"{y0}\u2013{y1}"

            opening    = _stock_totals(y0)
            closing    = _stock_totals(y1)
            additions  = _flow_totals(y1, "to")
            reductions = _flow_totals(y1, "from")
            net_balance = additions - reductions

            entries = [
                ("Opening stock",       opening),
                ("Additions to stock",  additions),
                ("Reductions in stock", reductions),
                ("Net carbon balance",  net_balance),
                ("Closing stock",       closing),
            ]

            for entry_name, series in entries:
                series = pd.Series(series, index=class_names).astype(float)
                row = series.to_dict()
                row["Total"] = float(series.sum())
                rows.append(((period, entry_name), row))

        index = pd.MultiIndex.from_tuples(
            [r[0] for r in rows], names=["Period", "Entry"]
        )
        result[stock_type] = pd.DataFrame(
            [r[1] for r in rows], index=index, columns=class_names + ["Total"]
        )

    return result
