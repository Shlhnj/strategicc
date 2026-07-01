"""
strategicc/accounting/outputs.py  —  SEEA-EA output functions  v2.0
--------------------------------------------------------------------
Saves all ecosystem accounts as CSVs and generates plots.

Functions
---------
save_all_accounts   — save all account tables to CSV
plot_monetary_flows — stacked area chart of total ecosystem value over time
plot_value_by_service — line chart per service type over time
plot_transition_heatmap — heatmap of transition matrix (area and value)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from strategicc.io.csv_loader import StateClass
from strategicc.accounting.seea import SEEAAccount


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

    acct.transition_matrix().to_csv(out_dir / "seea_transition_matrix_area.csv")
    print(f"  Saved: seea_transition_matrix_area.csv")

    acct.value_change_matrix().to_csv(out_dir / "seea_transition_matrix_value.csv")
    print(f"  Saved: seea_transition_matrix_value.csv")

    acct.monetary_flow_account().to_csv(out_dir / "seea_monetary_flow_account.csv")
    print(f"  Saved: seea_monetary_flow_account.csv")

    phys = acct.physical_flow_account()
    if phys is not None:
        phys.to_csv(out_dir / "seea_physical_flow_account.csv")
        print(f"  Saved: seea_physical_flow_account.csv")

    acct.total_value_by_class().to_csv(out_dir / "seea_total_value_by_class.csv")
    print(f"  Saved: seea_total_value_by_class.csv")

    acct.change_in_value().to_csv(out_dir / "seea_change_in_value.csv")
    print(f"  Saved: seea_change_in_value.csv")

    acct.uncertainty_summary().to_csv(out_dir / "seea_uncertainty_summary.csv", index=False)
    print(f"  Saved: seea_uncertainty_summary.csv")


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
    unc    = acct.uncertainty_summary().set_index("Year")
    total  = acct.total_value_by_class().sum(axis=1)
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
