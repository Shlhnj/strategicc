"""
strategicc/outputs.py  —  v3.8
--------------------------------
Visualisation and tabular summary functions.

Functions
---------
build_summary_tables     — aggregate area + transition CSVs across all iter dirs
aggregate_spatial        — modal class + uncertainty rasters per timestep  (v2.1)
plot_spatial_summary     — t=0 vs mid vs final modal map comparison plot   (v2.1)
plot_area_envelope       — area over time with median + min/max shaded band
plot_transition_envelope — transition count over time with uncertainty band
plot_lulc_maps           — LULC map panels for a single iteration (diagnostic)
plot_transition_maps     — transition event maps for a single iteration
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
from PIL import Image

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

    The area column name is detected automatically from the CSV header
    (area_ha, area_km2, or area_px) so this works regardless of AREA_UNIT.

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


def _area_col(df: pd.DataFrame) -> str:
    """Return the area column name from an area DataFrame (area_ha/km2/px)."""
    for col in df.columns:
        if col.startswith("area_"):
            return col
    raise ValueError(f"No area column found in DataFrame. Columns: {list(df.columns)}")


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
    Works with any AREA_UNIT (column name detected automatically).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if area_df.empty:
        print("  [Skip] area_df is empty — no area envelope plot generated")
        return

    acol       = _area_col(area_df)
    unit_label = acol.replace("area_", "")   # "ha", "km2", or "px"
    years      = sorted(area_df["year"].unique())
    class_ids  = sorted(classes.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    for cid in class_ids:
        sc     = classes[cid]
        color  = _class_rgb(sc)
        subset = area_df[area_df["class_id"] == cid]

        stats = (
            subset.groupby("year")[acol]
            .agg(["median", "min", "max"])
            .reindex(years)
        )

        ax.plot(years, stats["median"], color=color, linewidth=2,
                label=sc.name, zorder=3)
        ax.fill_between(years, stats["min"], stats["max"],
                        color=color, alpha=0.20, zorder=2)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(f"Area ({unit_label})", fontsize=11)
    ax.set_title("State Class Area Over Time\n"
                 "(solid = median, band = min–max across iterations)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Transition maps saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Spatial aggregation across iterations  (v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_spatial(
    iter_dirs:   list[Path],
    start_year:  int,
    n_timesteps: int,
    src_tags:    dict,
    summary_dir: Path,
    uncertainty: bool = True,
) -> dict[int, np.ndarray]:
    """
    For each timestep, load all iteration rasters, compute modal class
    per cell, and optionally the agreement % (uncertainty).

    Processes one timestep at a time to keep RAM usage bounded:
    only n_iterations rasters are in memory per loop.

    Parameters
    ----------
    iter_dirs   : list of per-iteration output directories
    start_year  : first simulation year
    n_timesteps : total number of timesteps (excluding t=0)
    src_tags    : GeoTIFF tag dict for georeferencing output rasters
    summary_dir : root summary directory; outputs go in summary_dir/spatial/
    uncertainty : if True, also save per-timestep agreement % rasters

    Returns
    -------
    modal_maps : dict mapping year → modal class array (uint8)
                 (useful for plot_spatial_summary without re-reading TIFs)
    """
    from strategicc.io.raster import _TAG_TIE_POINT, _TAG_PIXEL_SCALE

    spatial_dir = summary_dir
    spatial_dir.mkdir(parents=True, exist_ok=True)

    keep_tags = {k: src_tags[k]
                 for k in (_TAG_TIE_POINT, _TAG_PIXEL_SCALE, 34735, 34736, 34737)
                 if k in src_tags}

    modal_maps: dict[int, np.ndarray] = {}

    total_steps = n_timesteps + 1   # includes t=0
    print(f"  Aggregating {len(iter_dirs)} iterations × "
          f"{total_steps} timesteps spatially...")

    for t in range(total_steps):
        year = start_year + t

        # ── Load all iteration rasters for this timestep ──────────────────
        stack = []
        for d in iter_dirs:
            tif = d / f"lulc_{year}.tif"
            if tif.exists():
                arr = np.array(Image.open(str(tif)), dtype=np.uint8)
                stack.append(arr)

        if not stack:
            print(f"    [Skip] year {year} — no rasters found")
            continue

        cube = np.stack(stack, axis=0)   # (n_iter, rows, cols)

        # ── Modal class (most frequent per cell) ──────────────────────────
        from scipy import stats as scipy_stats
        mode_result = scipy_stats.mode(cube, axis=0, keepdims=False)
        modal = mode_result.mode.astype(np.uint8)
        modal_maps[year] = modal

        out_path = spatial_dir / f"lulc_mean_{year}.tif"
        save_kwargs = {"compression": "lzw"}
        if keep_tags:
            save_kwargs["tiffinfo"] = keep_tags
        Image.fromarray(modal, mode="L").save(str(out_path), **save_kwargs)

        # ── Uncertainty: fraction of iterations agreeing with modal ───────
        if uncertainty:
            n_iter      = cube.shape[0]
            matches     = (cube == modal[np.newaxis, :, :]).sum(axis=0)
            agreement   = (matches / n_iter * 100).astype(np.float32)

            unc_path = spatial_dir / f"uncertainty_{year}.tif"
            Image.fromarray(agreement.astype(np.uint8), mode="L").save(
                str(unc_path), **save_kwargs
            )

        print(f"    Year {year}: modal map saved"
              + (f" + uncertainty" if uncertainty else ""))

    print(f"  Spatial aggregation complete → '{spatial_dir}'")
    return modal_maps


# ─────────────────────────────────────────────────────────────────────────────
# 6. Spatial summary plot: t=0 vs mid vs final  (v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_summary(
    initial_lulc: np.ndarray,
    modal_maps:   dict[int, np.ndarray],
    classes:      dict[int, StateClass],
    start_year:   int,
    n_timesteps:  int,
    summary_dir:  Path,
    uncertainty:  bool = True,
    filename:     str  = "spatial_summary.png",
) -> None:
    """
    Side-by-side comparison of:
      - Col 1: initial LULC (t=0, actual)
      - Col 2: modal LULC at mid timestep
      - Col 3: modal LULC at final timestep

    If uncertainty=True, adds a second row showing agreement % maps
    for mid and final timesteps (t=0 has no uncertainty by definition).

    Parameters
    ----------
    initial_lulc : uint8 array from read_lulc() — the original t=0 raster
    modal_maps   : dict year → modal array, from aggregate_spatial()
    """
    spatial_dir = summary_dir
    spatial_dir.mkdir(parents=True, exist_ok=True)

    cmap   = _build_cmap(classes)
    max_id = max(classes.keys())

    mid_year   = start_year + n_timesteps // 2
    final_year = start_year + n_timesteps

    # Snap to nearest available year if exact year missing
    available = sorted(modal_maps.keys())
    def _nearest(target):
        return min(available, key=lambda y: abs(y - target))

    mid_year   = _nearest(mid_year)
    final_year = _nearest(final_year)

    n_rows = 2 if uncertainty else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # shape (1, 3)

    titles_top = [
        f"Initial LULC ({start_year})",
        f"Modal LULC — mid ({mid_year})",
        f"Modal LULC — final ({final_year})",
    ]
    maps_top = [
        initial_lulc,
        modal_maps.get(mid_year),
        modal_maps.get(final_year),
    ]

    # ── Row 1: LULC maps ──────────────────────────────────────────────────
    for col, (ax, arr, title) in enumerate(zip(axes[0], maps_top, titles_top)):
        if arr is None:
            ax.set_visible(False)
            continue
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=max_id,
                       interpolation="nearest")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Shared legend on first row
    axes[0, -1].legend(
        handles=_legend_patches(classes),
        loc="lower left", fontsize=7,
        bbox_to_anchor=(1.02, 0), borderaxespad=0,
    )

    # ── Row 2: uncertainty maps (if requested) ────────────────────────────
    if uncertainty:
        unc_titles = [
            "No uncertainty\n(initial raster)",
            f"Agreement % — mid ({mid_year})",
            f"Agreement % — final ({final_year})",
        ]
        unc_years  = [None, mid_year, final_year]

        for col, (ax, yr, title) in enumerate(
            zip(axes[1], unc_years, unc_titles)
        ):
            if yr is None:
                ax.text(0.5, 0.5, title, ha="center", va="center",
                        transform=ax.transAxes, fontsize=9,
                        color="grey")
                ax.axis("off")
                continue

            unc_path = spatial_dir / f"uncertainty_{yr}.tif"
            if not unc_path.exists():
                ax.set_visible(False)
                continue

            unc_arr = np.array(Image.open(str(unc_path)), dtype=np.float32)
            im = ax.imshow(unc_arr, cmap="RdYlGn", vmin=0, vmax=100,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.8,
                         label="Agreement (%)", orientation="vertical")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

    plt.suptitle(
        f"STRATEGICC Spatial Summary  |  "
        f"{len(list(spatial_dir.glob('lulc_mean_*.tif')))} timesteps aggregated",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    out_path = spatial_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Spatial summary plot saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Modal area table  (v2.2)
# ─────────────────────────────────────────────────────────────────────────────

def modal_to_area_table(
    modal_maps:  dict[int, np.ndarray],
    classes:     dict[int, StateClass],
    px_area:     float,
    area_unit:   str = "ha",
) -> pd.DataFrame:
    """
    Derive a per-class area table from modal LULC maps.

    This produces `area_modal_df` — the spatially-consistent area table
    used as input to SEEA-EA accounting.  It counts pixels of each class
    in each modal map and multiplies by `px_area` (already converted to
    the chosen unit).

    Parameters
    ----------
    modal_maps : dict year → uint8 modal class array, from aggregate_spatial()
    classes    : dict[int, StateClass]
    px_area    : pixel area in the chosen unit (engine.px_area)
    area_unit  : "ha" | "km2" | "px"  — used only for the column name

    Returns
    -------
    pd.DataFrame with columns:
        year, class_id, class_name, area_{unit}
    Schema matches area_df but with a single "iteration" = "modal" label.
    """
    unit_col = f"area_{area_unit}"
    rows = []
    for year in sorted(modal_maps.keys()):
        arr = modal_maps[year]
        for cid, sc in classes.items():
            rows.append({
                "year":       year,
                "class_id":   cid,
                "class_name": sc.name,
                unit_col:     float(np.sum(arr == cid)) * px_area,
            })
    return pd.DataFrame(rows)
