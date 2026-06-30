"""
strategicc/stockflow/aggregation.py  —  v3.2
-------------------------------------------------
Aggregates per-iteration Stock & Flow outputs across iterations and by
state class, producing per-class-per-year stock and flow totals suitable
for Mode C SEEA-EA valuation.

For each timestep: load that stock type's raster from every iteration,
compute the per-cell MEDIAN across iterations, mask by the modal LULC
class for that timestep, sum within each class's cells.

Flow totals are aggregated from flow_log.csv (median total per flow_type
per year across iterations) — flows are scalar aggregates already, not
per-cell rasters, so no class masking is needed (eligibility was already
class-gated by the engine at simulation time via FromStateClassId).
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
            df = pd.read_csv(log_path)
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
                # (standard SEEA-EA rollforward — using the reconciled
                # value, not the actual, keeps the account internally
                # consistent year over year even though actual stock
                # totals are reported alongside for validation)
                opening_balance = closing_reconciled

    return pd.DataFrame(rows)
