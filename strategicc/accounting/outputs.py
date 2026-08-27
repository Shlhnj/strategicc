"""
strategicc/accounting/outputs.py  —  SEEA-EA output functions  v3.14
--------------------------------------------------------------------
Saves all ecosystem accounts as CSVs (and, since v3.19, Excel
workbooks with one sheet per year/period) and generates plots.

Functions
---------
save_all_accounts   — save all account tables to CSV
save_all_accounts_xlsx — (v3.19) save all account tables as .xlsx,
                          one sheet per year/period for tables that have
                          more than one row per year/period; a single
                          sheet for tables that are already one row per
                          year (splitting those further wouldn't help
                          readability — see _write_grouped_excel())
save_asset_account   — (v3.20) CSV + sheet-per-year .xlsx for the flat
                          asset account (stockflow.aggregation.
                          build_asset_account()) — previously CSV-only
                          and outside the save_all_accounts()/
                          save_all_accounts_xlsx() pair.
save_carbon_stock_account — (v3.20) CSV + sheet-per-period .xlsx, one
                          file per stock type, for the Table 13.3-shaped
                          physical stock account (stockflow.aggregation.
                          stock_account_seea()).
plot_monetary_flows — stacked area chart of total ecosystem value over time
plot_value_by_service — line chart per service type over time
plot_transition_heatmap — heatmap of transition matrix (area and value)
save_monetary_value_raster — (v3.12) Mode C genuine per-pixel valuation
                              raster from a simulated stock raster
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image

from strategicc.io.csv_loader import StateClass
from strategicc.accounting.csv_loader import EcosystemService
from strategicc.accounting.seea import SEEAAccount
from strategicc.io.raster import read_tiff, read_lulc, _TAG_TIE_POINT, _TAG_PIXEL_SCALE


# ── Color helpers ─────────────────────────────────────────────────────────────

def _class_colors(classes: dict[int, StateClass]) -> dict[str, tuple]:
    return {
        sc.name: (sc.color[1]/255, sc.color[2]/255, sc.color[3]/255)
        for sc in classes.values()
    }


# ── Save all account tables ───────────────────────────────────────────────────

def save_all_accounts(
    acct:    SEEAAccount,
    out_dir: Path,
) -> None:
    """Save all SEEA-EA account tables as CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    acct.extent_account().to_csv(out_dir / "seea_extent_account.csv")
    print(f"  Saved: seea_extent_account.csv")

    if acct.trans_df is not None and not acct.trans_df.empty:
        acct.extent_account_seea().to_csv(out_dir / "seea_extent_account_table4_1.csv")
        print(f"  Saved: seea_extent_account_table4_1.csv")
    else:
        print(f"  [Skipped] seea_extent_account_table4_1.csv — no trans_df "
              f"was provided to SEEAAccount (pass trans_df= to enable the "
              f"SEEA EA Table 4.1 formatted extent account)")

    acct.transition_matrix().to_csv(out_dir / "seea_transition_matrix_area.csv")
    print(f"  Saved: seea_transition_matrix_area.csv")

    acct.value_change_matrix().to_csv(out_dir / "seea_transition_matrix_value.csv")
    print(f"  Saved: seea_transition_matrix_value.csv")

    acct.monetary_flow_account().to_csv(out_dir / "seea_monetary_flow_account.csv")
    print(f"  Saved: seea_monetary_flow_account.csv")

    mon_seea = acct.monetary_flow_account_seea()
    mon_seea["supply"].to_csv(out_dir / "seea_monetary_flow_account_supply.csv")
    mon_seea["use"].to_csv(out_dir / "seea_monetary_flow_account_use.csv")
    print(f"  Saved: seea_monetary_flow_account_supply.csv")
    print(f"  Saved: seea_monetary_flow_account_use.csv")

    phys = acct.physical_flow_account()
    if phys is not None:
        phys.to_csv(out_dir / "seea_physical_flow_account.csv")
        print(f"  Saved: seea_physical_flow_account.csv")

    phys_seea = acct.physical_flow_account_seea()
    if phys_seea is not None:
        phys_seea["supply"].to_csv(out_dir / "seea_physical_flow_account_supply.csv")
        phys_seea["use"].to_csv(out_dir / "seea_physical_flow_account_use.csv")
        print(f"  Saved: seea_physical_flow_account_supply.csv")
        print(f"  Saved: seea_physical_flow_account_use.csv")

    acct.total_value_by_class().to_csv(out_dir / "seea_total_value_by_class.csv")
    print(f"  Saved: seea_total_value_by_class.csv")

    acct.change_in_value().to_csv(out_dir / "seea_change_in_value.csv")
    print(f"  Saved: seea_change_in_value.csv")

    if acct.trans_df is not None and not acct.trans_df.empty and acct.asset_valuation_params:
        acct.monetary_asset_account_seea().to_csv(out_dir / "seea_monetary_asset_account_table10_1.csv")
        print(f"  Saved: seea_monetary_asset_account_table10_1.csv")
    else:
        print(f"  [Skipped] seea_monetary_asset_account_table10_1.csv — requires "
              f"both trans_df and asset_valuation_params (pass "
              f"asset_valuation_params=load_asset_valuation_params(...) to enable "
              f"the SEEA EA Table 10.1 formatted monetary asset account)")

    unc_df = acct.uncertainty_summary()
    if unc_df is not None:
        unc_df.to_csv(out_dir / "seea_uncertainty_summary.csv", index=False)
        print(f"  Saved: seea_uncertainty_summary.csv")
    else:
        print(f"  [Skipped] seea_uncertainty_summary.csv — no per-iteration "
              f"area_df was provided to SEEAAccount (pass area_df= to enable)")


# ── Save all account tables as Excel workbooks, one sheet per year ────────────

_INVALID_SHEET_CHARS = re.compile(r"[\\/?*\[\]:]")

def _sanitize_sheet_name(name: object) -> str:
    """Excel sheet names: no \\/?*[]:  , max 31 chars, non-empty."""
    s = _INVALID_SHEET_CHARS.sub("_", str(name))
    return (s[:31] or "Sheet1")


def _write_grouped_excel(
    df:          pd.DataFrame,
    out_path:    Path,
    group_level: str,
    fallback_sheet_name: str = "All Years",
) -> None:
    """
    Write df to an .xlsx workbook. If df has an index level named
    group_level (e.g. "Year" or "Period") with more than one row per
    distinct value, splits into one sheet per value (the group level
    dropped from each sheet — it's redundant with the sheet name).
    Otherwise — no such level, or every value already has exactly one
    row (a flat year-indexed table like extent_account()) — writes the
    whole table to a single sheet instead; splitting an
    already-one-row-per-year table into one-row-per-sheet wouldn't
    help readability, it'd just be 30 sheets with one line each.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        names = list(df.index.names) if df.index.names else []
        if group_level in names:
            level_vals = df.index.get_level_values(group_level)
            groups = level_vals.unique()
            multi_row = any((level_vals == g).sum() > 1 for g in groups)
            if multi_row:
                for g in groups:
                    sub = df.xs(g, level=group_level)
                    sub.to_excel(writer, sheet_name=_sanitize_sheet_name(g))
                return
        df.to_excel(writer, sheet_name=fallback_sheet_name)


def _write_flat_grouped_excel(
    df:          pd.DataFrame,
    out_path:    Path,
    group_col:   str,
    fallback_sheet_name: str = "All Years",
) -> None:
    """
    Column-based counterpart to _write_grouped_excel() (v3.20), for flat
    (non-MultiIndex) tables like build_asset_account()'s output, where
    the split key (e.g. "year") is an ordinary column rather than an
    index level. If group_col is present and has more than one row per
    distinct value, splits into one sheet per value (the group column
    dropped from each sheet, same reasoning as _write_grouped_excel()).
    Otherwise writes the whole table to a single sheet.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if group_col in df.columns:
            groups = df[group_col].unique()
            multi_row = any((df[group_col] == g).sum() > 1 for g in groups)
            if multi_row:
                for g in groups:
                    sub = df[df[group_col] == g].drop(columns=[group_col])
                    sub.to_excel(writer, sheet_name=_sanitize_sheet_name(g), index=False)
                return
        df.to_excel(writer, sheet_name=fallback_sheet_name, index=False)


def save_asset_account(
    asset_account: pd.DataFrame,
    out_dir:       Path,
    write_csv:     bool = True,
    write_xlsx:    bool = True,
) -> None:
    """
    Save the flat SEEA-EA-style asset account (v3.3, from
    stockflow.aggregation.build_asset_account()) as CSV and, since
    v3.20, .xlsx split by year — matching the sheet-per-year treatment
    the other 14 accounts already get from save_all_accounts_xlsx(),
    which this table was previously excluded from (it's built and
    written directly by run.py, outside the save_all_accounts()/
    save_all_accounts_xlsx() pair, since it comes from
    stockflow.aggregation rather than SEEAAccount).

    write_csv/write_xlsx let a caller target csv/ and xlsx/ output
    folders separately (v3.20) by calling this twice, once per folder,
    with the other format switched off — rather than writing both
    formats into whichever single out_dir is passed.

    Kept as its own function (not folded into save_all_accounts[_xlsx])
    since callers who only have stock_df/flow_df output, not a full
    SEEAAccount, still need a place to write it (that's exactly how
    run.py calls it).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if write_csv:
        asset_account.to_csv(out_dir / "seea_asset_account.csv", index=False)
        print(f"  Saved: seea_asset_account.csv")

    if write_xlsx:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            print(f"  [Skipped] seea_asset_account.xlsx — requires openpyxl "
                  f"(pip install openpyxl, or strategicc[xlsx])")
        else:
            _write_flat_grouped_excel(
                asset_account, out_dir / "seea_asset_account.xlsx", "year"
            )
            print(f"  Saved: seea_asset_account.xlsx")


def save_carbon_stock_account(
    stock_account: dict[str, pd.DataFrame],
    out_dir:       Path,
    write_csv:     bool = True,
    write_xlsx:    bool = True,
) -> None:
    """
    Save the Table 13.3-shaped physical carbon stock account (v3.20,
    from stockflow.aggregation.stock_account_seea()) as CSV and .xlsx,
    one file per stock type (e.g. seea_carbon_stock_account_AGB.csv/
    .xlsx) — matching how other dict-returning accounts (e.g.
    monetary_flow_account_seea()'s supply/use pair) get one file per
    key. Each .xlsx gets one sheet per accounting period, via
    _write_grouped_excel() (stock_account_seea()'s tables are already
    Period-indexed, unlike the flat asset account above).

    write_csv/write_xlsx: see save_asset_account() — same
    separate-folder calling convention.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stock_type, df in stock_account.items():
        if write_csv:
            df.to_csv(out_dir / f"seea_carbon_stock_account_{stock_type}.csv")
            print(f"  Saved: seea_carbon_stock_account_{stock_type}.csv")

        if write_xlsx:
            try:
                import openpyxl  # noqa: F401
            except ImportError:
                print(f"  [Skipped] seea_carbon_stock_account_{stock_type}.xlsx "
                      f"— requires openpyxl (pip install openpyxl, or "
                      f"strategicc[xlsx])")
            else:
                _write_grouped_excel(
                    df, out_dir / f"seea_carbon_stock_account_{stock_type}.xlsx",
                    "Period",
                )
                print(f"  Saved: seea_carbon_stock_account_{stock_type}.xlsx")


def save_all_accounts_xlsx(
    acct:    SEEAAccount,
    out_dir: Path,
) -> None:
    """
    Save all SEEA-EA account tables as Excel workbooks (.xlsx), one
    workbook per account (matching save_all_accounts()'s CSV filenames,
    .xlsx instead of .csv), with one sheet per year for tables where a
    year has multiple rows (the flow-account supply/use pairs, Table
    4.1's Period blocks, Table 10.1's Period blocks) and a single sheet
    for tables that are already one row per year (extent_account(),
    monetary/physical_flow_account(), total_value_by_class(),
    change_in_value()) — see _write_grouped_excel() for why those
    aren't split further. Requires openpyxl (pip install openpyxl, or
    strategicc[xlsx]).

    Does not replace save_all_accounts()'s CSVs — call both if you want
    CSVs for scripting and workbooks for manual browsing; this function
    only adds the .xlsx files alongside whatever's already in out_dir.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError(
            "save_all_accounts_xlsx() requires openpyxl — "
            "install with `pip install openpyxl` or `pip install strategicc[xlsx]`."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    _write_grouped_excel(
        acct.extent_account(), out_dir / "seea_extent_account.xlsx", "Year"
    )
    print(f"  Saved: seea_extent_account.xlsx")

    if acct.trans_df is not None and not acct.trans_df.empty:
        _write_grouped_excel(
            acct.extent_account_seea(), out_dir / "seea_extent_account_table4_1.xlsx", "Period"
        )
        print(f"  Saved: seea_extent_account_table4_1.xlsx")

    with pd.ExcelWriter(out_dir / "seea_transition_matrix_area.xlsx", engine="openpyxl") as w:
        acct.transition_matrix().to_excel(w, sheet_name="Transition Matrix")
    print(f"  Saved: seea_transition_matrix_area.xlsx")

    with pd.ExcelWriter(out_dir / "seea_transition_matrix_value.xlsx", engine="openpyxl") as w:
        acct.value_change_matrix().to_excel(w, sheet_name="Value Change Matrix")
    print(f"  Saved: seea_transition_matrix_value.xlsx")

    _write_grouped_excel(
        acct.monetary_flow_account(), out_dir / "seea_monetary_flow_account.xlsx", "Year"
    )
    print(f"  Saved: seea_monetary_flow_account.xlsx")

    mon_seea = acct.monetary_flow_account_seea()
    _write_grouped_excel(
        mon_seea["supply"], out_dir / "seea_monetary_flow_account_supply.xlsx", "Year"
    )
    _write_grouped_excel(
        mon_seea["use"], out_dir / "seea_monetary_flow_account_use.xlsx", "Year"
    )
    print(f"  Saved: seea_monetary_flow_account_supply.xlsx")
    print(f"  Saved: seea_monetary_flow_account_use.xlsx")

    phys = acct.physical_flow_account()
    if phys is not None:
        _write_grouped_excel(
            phys, out_dir / "seea_physical_flow_account.xlsx", "Year"
        )
        print(f"  Saved: seea_physical_flow_account.xlsx")

    phys_seea = acct.physical_flow_account_seea()
    if phys_seea is not None:
        _write_grouped_excel(
            phys_seea["supply"], out_dir / "seea_physical_flow_account_supply.xlsx", "Year"
        )
        _write_grouped_excel(
            phys_seea["use"], out_dir / "seea_physical_flow_account_use.xlsx", "Year"
        )
        print(f"  Saved: seea_physical_flow_account_supply.xlsx")
        print(f"  Saved: seea_physical_flow_account_use.xlsx")

    _write_grouped_excel(
        acct.total_value_by_class(), out_dir / "seea_total_value_by_class.xlsx", "Year"
    )
    print(f"  Saved: seea_total_value_by_class.xlsx")

    _write_grouped_excel(
        acct.change_in_value(), out_dir / "seea_change_in_value.xlsx", "Year"
    )
    print(f"  Saved: seea_change_in_value.xlsx")

    if acct.trans_df is not None and not acct.trans_df.empty and acct.asset_valuation_params:
        _write_grouped_excel(
            acct.monetary_asset_account_seea(),
            out_dir / "seea_monetary_asset_account_table10_1.xlsx", "Period",
        )
        print(f"  Saved: seea_monetary_asset_account_table10_1.xlsx")

    unc_df = acct.uncertainty_summary()
    if unc_df is not None:
        with pd.ExcelWriter(out_dir / "seea_uncertainty_summary.xlsx", engine="openpyxl") as w:
            unc_df.to_excel(w, sheet_name="Uncertainty Summary", index=False)
        print(f"  Saved: seea_uncertainty_summary.xlsx")


# ── Plot: stacked area — total ecosystem value over time ──────────────────────

def plot_monetary_flows(
    acct:    SEEAAccount,
    classes: dict[int, StateClass],
    out_dir: Path,
    filename: str = "seea_monetary_flows.png",
) -> None:
    """
    Stacked area chart: total ecosystem service value per class over time.
    Shows which classes contribute most to total landscape value.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tv      = acct.total_value_by_class()
    colors  = _class_colors(classes)
    years   = tv.index.tolist()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ── Panel 1: stacked area by class ────────────────────────────────────────
    ax1    = axes[0]
    bottom = np.zeros(len(years))
    for col in tv.columns:
        vals  = tv[col].values
        color = colors.get(col, (0.5, 0.5, 0.5))
        ax1.fill_between(years, bottom, bottom + vals,
                         alpha=0.85, color=color, label=col)
        bottom += vals

    ax1.set_ylabel("Total ecosystem value (currency/yr)", fontsize=10)
    ax1.set_title("Total Ecosystem Service Value by Class", fontsize=11)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:,.0f}"
    ))
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: year-on-year change in total value ───────────────────────────
    ax2   = axes[1]
    delta = acct.change_in_value()["Total"].dropna()
    colors_bar = ["#2ecc71" if v >= 0 else "#e74c3c" for v in delta.values]
    ax2.bar(delta.index, delta.values, color=colors_bar, alpha=0.85, width=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Change in value (currency/yr)", fontsize=10)
    ax2.set_title("Year-on-Year Change in Total Ecosystem Value", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if abs(x) >= 1e6 else f"{x:,.0f}"
    ))
    ax2.grid(True, alpha=0.2, axis="y")

    # ── Uncertainty band on panel 1 ───────────────────────────────────────────
    unc = acct.uncertainty_summary()
    if unc is not None:
        unc = unc.set_index("Year")
        if not unc.empty:
            ax1.fill_between(
                unc.index, unc["Min value"], unc["Max value"],
                alpha=0.12, color="grey", label="Min–Max range"
            )
            ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot: line chart per service type ─────────────────────────────────────────

def plot_value_by_service(
    acct:    SEEAAccount,
    out_dir: Path,
    filename: str = "seea_value_by_service.png",
) -> None:
    """
    Line chart: total monetary value per service type over time.
    One line per service (Provisioning / Regulating / Cultural).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mf    = acct.monetary_flow_account()
    years = mf.index.tolist()

    # Aggregate by service type (top level of MultiIndex columns)
    type_totals: dict[str, list[float]] = {}
    for col in mf.columns:
        stype = col[0]
        type_totals.setdefault(stype, np.zeros(len(years)))
        type_totals[stype] += mf[col].values

    type_colors = {
        "Provisioning": "#e67e22",
        "Regulating":   "#27ae60",
        "Cultural":     "#8e44ad",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for stype, vals in type_totals.items():
        color = type_colors.get(stype, "#2c3e50")
        ax.plot(years, vals, color=color, linewidth=2.5, label=stype, zorder=3)
        ax.fill_between(years, 0, vals, color=color, alpha=0.08, zorder=2)

    # Also plot individual services as thin dashed lines
    for col in mf.columns:
        stype = col[0]
        sname = col[1]
        color = type_colors.get(stype, "#2c3e50")
        ax.plot(years, mf[col].values, color=color,
                linewidth=0.8, linestyle="--", alpha=0.5,
                label=f"  {sname}")

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Value (currency/yr)", fontsize=10)
    ax.set_title("Ecosystem Service Value by Service Type", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:,.0f}"
    ))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot: transition heatmap ──────────────────────────────────────────────────

def plot_transition_heatmap(
    acct:    SEEAAccount,
    out_dir: Path,
    filename: str = "seea_transition_heatmap.png",
) -> None:
    """
    Two-panel heatmap:
    Left  — area (ha) converted between classes (transition matrix)
    Right — monetary value change from those conversions
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tm = acct.transition_matrix()
    vm = acct.value_change_matrix()

    if tm.empty:
        print("  [Skip] transition matrix empty — no heatmap generated")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title, cmap, fmt in [
        (axes[0], tm, "Area converted (ha)",    "YlOrBr", ".1f"),
        (axes[1], vm, "Value change (currency)", "RdYlGn", ".0f"),
    ]:
        arr = data.values.astype(float)

        # Mask diagonal (no-change cells)
        mask_diag = np.eye(arr.shape[0], dtype=bool)
        arr_plot  = np.where(mask_diag, np.nan, arr)

        vmax = np.nanmax(np.abs(arr_plot)) if not np.all(np.isnan(arr_plot)) else 1
        vmin = -vmax if cmap == "RdYlGn" else 0

        im = ax.imshow(arr_plot, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)

        labels = list(data.index)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("To class", fontsize=9)
        ax.set_ylabel("From class", fontsize=9)
        ax.set_title(title, fontsize=10)

        # Annotate non-zero, non-diagonal cells
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not mask_diag[i, j] and arr[i, j] != 0:
                    ax.text(j, i, f"{arr[i,j]:{fmt}}",
                            ha="center", va="center", fontsize=7,
                            color="black")

    plt.suptitle("Ecosystem Transition Matrix", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode C — genuine per-pixel monetary valuation raster (v3.12)
# ─────────────────────────────────────────────────────────────────────────────

def save_monetary_value_raster(
    stock_raster_path: str | Path,
    lulc_raster_path:  str | Path,
    service:           EcosystemService,
    classes:           dict[int, StateClass],
    out_path:          str | Path,
    nodata_value:      float = -9999.0,
) -> Path:
    """
    Write a genuine per-pixel monetary valuation raster for ONE Mode C
    ecosystem service, for a single year/iteration.

    Distinct from Mode A/B valuation (SEEAAccount.monetary_flow_account()),
    which applies a single flat ValuePerUnitArea across an entire class's
    total area — a "price density map" with no real per-pixel variation.
    This function instead multiplies the ACTUAL simulated per-pixel stock
    value by the service's price-per-physical-unit, so two pixels of the
    same class with different simulated stock (e.g. different Biomass
    carbon due to age/history) get different valuations.

    v1 scope (flagged, not resolved)
    ---------------------------------
    - One service per call — no batch/multi-service raster in this version.
    - No manifest or engine changes — this reads already-saved stock
      rasters (engine.out_dir/.../stocks/{stock_type}/stock_{year}.tif)
      and an already-saved LULC raster for the same year/iteration; it
      does not hook into the run pipeline automatically.
    - Only Mode C (stockflow-sourced) services are valid input — Mode A/B
      services have no per-pixel physical quantity to draw from and will
      raise ValueError.

    Parameters
    ----------
    stock_raster_path : path to a per-pixel stock GeoTIFF (float32) for
                        the matching stock_type and year/iteration —
                        e.g. "{iter_dir}/stocks/Biomass/stock_2010.tif"
    lulc_raster_path  : path to the LULC class-id raster for the SAME
                        year/iteration, used to mask which pixels belong
                        to service.state_class (only those are priced;
                        everything else gets nodata_value)
    service           : EcosystemService with stockflow_source set
                        (Mode C) — value_per_unit_area is treated as a
                        price PER PHYSICAL UNIT, not per area, matching
                        SEEAAccount's Mode C convention.
    classes           : dict[int, StateClass] — used to resolve
                        service.state_class (a name, e.g. "Mangrove") to
                        the class id(s) that mask which pixels get priced.
    out_path          : output GeoTIFF path (float32, single-band)
    nodata_value      : value written for pixels not in service.state_class

    Returns
    -------
    Path actually written to
    """
    if not service.has_stockflow_source:
        raise ValueError(
            f"save_monetary_value_raster requires a Mode C service "
            f"(stockflow_source set) — service '{service.service_name}' "
            f"has no StockFlowSource. Use SEEAAccount.monetary_flow_account() "
            f"for Mode A/B flat per-class valuation instead."
        )

    stock_arr, _, _ = read_tiff(stock_raster_path)
    lulc_arr, _, _  = read_lulc(lulc_raster_path)

    if stock_arr.shape != lulc_arr.shape:
        raise ValueError(
            f"stock raster shape {stock_arr.shape} != "
            f"lulc raster shape {lulc_arr.shape} — must be the same "
            f"year/iteration/extent."
        )

    class_ids = [cid for cid, sc in classes.items() if sc.name == service.state_class]
    if not class_ids:
        raise ValueError(
            f"service.state_class '{service.state_class}' not found in "
            f"classes dict — available: {[sc.name for sc in classes.values()]}"
        )

    value_arr = np.full(stock_arr.shape, nodata_value, dtype=np.float32)
    mask = np.isin(lulc_arr, class_ids)
    value_arr[mask] = stock_arr[mask] * service.value_per_unit_area

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"compression": "lzw"}
    Image.fromarray(value_arr, mode="F").save(str(out_path), **save_kwargs)
    print(f"  Monetary value raster saved ('{service.service_name}', "
          f"Mode C): '{out_path}'")
    return out_path
