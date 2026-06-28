"""
strategicc/outputs.py  —  v1.1
-------------------------
Visualisation and tabular summary functions.

Per-iteration outputs are written by the engine directly.
This module aggregates across iterations and produces summary outputs.

Functions
---------
build_summary_tables    — aggregate area + transition CSVs across all iter dirs
plot_area_envelope      — area over time with median + min/max shaded band
plot_transition_envelope — transition count over time with uncertainty band
plot_lulc_maps          — LULC map panels for a single iteration (diagnostic)
plot_transition_maps    — transition event maps for a single iteration
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from strategicc.io.csv_loader import StateClass


# ─────────────────────────────────────────────────────────────────────────────
# Colormap helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_cmap(classes: dict[int, StateClass]) -> ListedColormap:
    max_id = max(classes.keys())
    colors = [(0.0, 0.0, 0.0)] + [
        (
            classes[i].color[1] / 255,
            classes[i].color[2] / 255,
            classes[i].color[3] / 255,
        ) if i in classes else (0.0, 0.0, 0.0)
        for i in range(1, max_id + 1)
    ]
    return ListedColormap(colors)


def _class_rgb(sc: StateClass) -> tuple[float, float, float]:
    return (sc.color[1] / 255, sc.color[2] / 255, sc.color[3] / 255)


def _legend_patches(classes: dict[int, StateClass]) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=_class_rgb(sc), label=sc.name)
        for sc in classes.values()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Aggregate across iterations
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_tables(
    iter_dirs: list[Path],
    summary_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate per-iteration area_table.csv and transition_log.csv files
    into two combined CSVs in summary_dir.

    Returns
    -------
    area_df        : concatenated area table (all iterations)
    transition_df  : concatenated transition log (all iterations)
    """
    summary_dir.mkdir(parents=True, exist_ok=True)

    area_frames, trans_frames = [], []
    for d in iter_dirs:
        a = d / "area_table.csv"
        t = d / "transition_log.csv"
        if a.exists():
            area_frames.append(pd.read_csv(a))
        if t.exists():
            trans_frames.append(pd.read_csv(t))

    area_df  = pd.concat(area_frames,  ignore_index=True) if area_frames  else pd.DataFrame()
    trans_df = pd.concat(trans_frames, ignore_index=True) if trans_frames else pd.DataFrame()

    area_df.to_csv( summary_dir / "area_all_iterations.csv",       index=False)
    trans_df.to_csv(summary_dir / "transitions_all_iterations.csv", index=False)

    print(f"  Summary tables saved to '{summary_dir}'")
    return area_df, trans_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Area envelope plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_area_envelope(
    area_df:    pd.DataFrame,
    classes:    dict[int, StateClass],
    out_dir:    Path,
    filename:   str = "area_envelope.png",
) -> None:
    """
    Line chart of state class area over time.
    Median across iterations = solid line.
    Min–max range across iterations = shaded band.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if area_df.empty:
        print("  [Skip] area_df is empty — no area envelope plot generated")
        return

    years      = sorted(area_df["year"].unique())
    class_ids  = sorted(classes.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    for cid in class_ids:
        sc     = classes[cid]
        color  = _class_rgb(sc)
        subset = area_df[area_df["class_id"] == cid]

        # Stats per year across iterations
        stats = (
            subset.groupby("year")["area_ha"]
            .agg(["median", "min", "max"])
            .reindex(years)
        )

        ax.plot(years, stats["median"], color=color, linewidth=2,
                label=sc.name, zorder=3)
        ax.fill_between(years, stats["min"], stats["max"],
                        color=color, alpha=0.20, zorder=2)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Area (ha)", fontsize=11)
    ax.set_title("State Class Area Over Time\n"
                 "(solid = median, band = min–max across iterations)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Area envelope plot saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transition count envelope plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_transition_envelope(
    trans_df:  pd.DataFrame,
    out_dir:   Path,
    filename:  str = "transition_envelope.png",
) -> None:
    """
    Bar/line chart of total transition count per year (all groups combined)
    with per-group stacked area + min–max uncertainty band.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if trans_df.empty:
        print("  [Skip] trans_df is empty — no transition envelope plot generated")
        return

    groups = sorted(trans_df["group"].unique())
    years  = sorted(trans_df["year"].unique())

    # Count transitions per iteration × year × group
    counts = (
        trans_df.groupby(["iteration", "year", "group"])
        .size()
        .reset_index(name="count")
    )

    # Color palette for groups (tab10)
    palette = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ── Panel 1: total transitions per year (all groups) ─────────────────────
    ax1 = axes[0]
    total = (
        counts.groupby(["iteration", "year"])["count"]
        .sum()
        .reset_index()
    )
    total_stats = (
        total.groupby("year")["count"]
        .agg(["median", "min", "max"])
        .reindex(years)
    )
    ax1.plot(years, total_stats["median"], color="black",
             linewidth=2, label="Median", zorder=3)
    ax1.fill_between(years, total_stats["min"], total_stats["max"],
                     color="grey", alpha=0.25, label="Min–Max", zorder=2)
    ax1.set_ylabel("Total transitions", fontsize=10)
    ax1.set_title("Total Transitions Per Year\n"
                  "(median ± min–max across iterations)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: per-group median transitions ─────────────────────────────────
    ax2 = axes[1]
    bottom = np.zeros(len(years))
    year_idx = {y: i for i, y in enumerate(years)}

    for g, color in zip(groups, palette):
        g_data = counts[counts["group"] == g]
        g_stats = (
            g_data.groupby("year")["count"]
            .agg(["median", "min", "max"])
            .reindex(years)
            .fillna(0)
        )
        medians = g_stats["median"].values
        ax2.bar(
            years, medians, bottom=bottom,
            color=color, label=g, alpha=0.85, width=0.7,
        )
        bottom += medians

    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Median transitions", fontsize=10)
    ax2.set_title("Transitions Per Year by Group (median across iterations)",
                  fontsize=11)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax2.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Transition envelope plot saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Single-iteration diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_lulc_maps(
    maps:       list[np.ndarray],
    classes:    dict[int, StateClass],
    start_year: int,
    out_dir:    Path,
    filename:   str = "lulc_maps.png",
) -> None:
    """Row of LULC map panels, one per timestep (diagnostic / single iter)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap   = _build_cmap(classes)
    max_id = max(classes.keys())
    n      = len(maps)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (ax, arr) in enumerate(zip(axes, maps)):
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=max_id, interpolation="nearest")
        ax.set_title(f"LULC {start_year + i}", fontsize=9)
        ax.axis("off")

    axes[-1].legend(
        handles=_legend_patches(classes),
        loc="lower left", fontsize=7,
        bbox_to_anchor=(1.02, 0), borderaxespad=0,
    )
    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  LULC maps saved to: {out_path}")


def plot_transition_maps(
    transitions: list[list],
    map_shape:   tuple[int, int],
    classes:     dict[int, StateClass],
    start_year:  int,
    out_dir:     Path,
    filename:    str = "transition_maps.png",
) -> None:
    """Row of transition event maps for a single iteration (diagnostic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap   = _build_cmap(classes)
    max_id = max(classes.keys())
    n      = len(transitions)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (ax, year_trans) in enumerate(zip(axes, transitions)):
        canvas = np.zeros(map_shape, dtype=np.uint8)
        for rec in year_trans:
            canvas[rec.row, rec.col] = rec.to_id
        ax.imshow(canvas, cmap=cmap, vmin=0, vmax=max_id, interpolation="nearest")
        ax.set_title(f"{start_year + i} → {start_year + i + 1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Transition maps saved to: {out_path}")
