"""
strategicc/validation/hindcast.py  —  v3.12
--------------------------------------------
Orchestrates a full validation pass: run the calibrated model over a
historical window it wasn't fitted to reproduce cell-for-cell, and compare
against the real record.

Standalone step, NOT auto-wired into strategicc.run.main(). Sits between
calibration and the real scenario runs (BAU/Extraction/Conservation):

    1. calibration   (derive Transitions.csv, etc.)
    2. diagnostics   (eyeball LULC maps / statistics / probabilities)
    3. validation    (hindcast_run() -- this module)
    4. run           (real scenario runs)
    5. see result

Iteration count defaults to 20 (not the manifest's production ITERATIONS,
typically 100) -- hindcast_run() is a diagnostic gate, not a published
uncertainty estimate, so a cheaper/faster run is an acceptable trade-off
for a noisier band.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategicc import config
from strategicc.engine import StrategiccEngine
from strategicc import outputs
from strategicc.calibration.loader import LULCTimeSeries
from strategicc.outputs import _area_col, _class_rgb

from .extent import (
    compute_observed_extent,
    compare_extent_trajectories,
    spatial_agreement,
    attribute_extent_drift,
)


@dataclass
class HindcastResult:
    """Bundled output of hindcast_run()."""
    extent_comparison: pd.DataFrame
    spatial_agreement: dict[int, dict]                 # year -> spatial_agreement() dict
    drift:             dict[str, pd.DataFrame] = field(default_factory=dict)  # class_name -> drift df
    flagged_classes:   list[str]               = field(default_factory=list)
    plot_path:         Path | None              = None
    area_df:           pd.DataFrame | None      = None
    trans_df:          pd.DataFrame | None      = None


def hindcast_run(
    manifest_path:       str | Path,
    ts:                  LULCTimeSeries,
    n_iterations:        int = 20,
    start_year:          int | None = None,
    out_dir:             str | Path = "hindcast_output",
    cache_path:           str | Path | None = "calibration_result/validation_cache/ObservedExtent.csv",
    px_area_ha:           float | None = None,
    flag_threshold_pct:   float = 15.0,
) -> HindcastResult:
    """
    Run the calibrated model over the historical window and validate
    against the observed record.

    Parameters
    ----------
    manifest_path : path to the calibrated RunManifest.txt (all simulation
                    inputs -- Transitions.csv, TransitionMultipliers.csv,
                    etc. -- are read from here, same as a real run)
    ts            : LULCTimeSeries covering the hindcast window (from
                    calibration.load_lulc_timeseries())
    n_iterations  : iterations for THIS hindcast run (default 20, separate
                    from the manifest's production ITERATIONS)
    start_year    : first year of the hindcast window; defaults to
                    ts.years[0]. Must be within ts.years.
    out_dir       : output directory for this hindcast run (kept separate
                    from production run output)
    cache_path    : cache path passed to compute_observed_extent()
    px_area_ha    : pixel area in hectares; if None, taken from the engine
                    after load() (engine.px_area_ha)
    flag_threshold_pct : a class is flagged for attribute_extent_drift()
                    if abs(pct_diff) at the final shared year exceeds this

    Returns
    -------
    HindcastResult
    """
    manifest_path = Path(manifest_path)
    start_year = start_year if start_year is not None else ts.years[0]
    end_year   = ts.years[-1]
    n_timesteps = end_year - start_year
    if n_timesteps <= 0:
        raise ValueError(
            f"start_year={start_year} must be earlier than the last year "
            f"in ts ({end_year})"
        )

    # ── 1. Load manifest, override run-control settings for the hindcast ────
    config.load_manifest(str(manifest_path))
    config.reset_manifest_mode()

    initial_raster_path = Path(out_dir) / f"hindcast_initial_{start_year}.tif"
    _write_raster_from_stack(ts, start_year, initial_raster_path)

    config.LULC_PATH    = initial_raster_path
    config.START_YEAR   = start_year
    config.N_TIMESTEPS   = n_timesteps
    config.N_ITERATIONS  = n_iterations
    config.OUT_DIR       = Path(out_dir)

    # ── 2. Run the engine over the historical window ────────────────────────
    engine = StrategiccEngine.from_config()
    engine.load()
    engine.diagnostic()
    engine.run()

    if px_area_ha is None:
        px_area_ha = engine.px_area_ha

    # ── 3. Simulated summary tables + modal maps ─────────────────────────────
    summary_dir = engine.out_dir / "summary"
    area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)

    modal_maps = outputs.aggregate_spatial(
        iter_dirs   = engine.iter_dirs,
        start_year  = engine.start_year,
        n_timesteps = engine.n_timesteps,
        src_tags    = engine.src_tags,
        summary_dir = summary_dir,
        uncertainty = False,
    )

    area_modal_df = outputs.modal_to_area_table(
        modal_maps = modal_maps,
        classes    = engine.classes,
        px_area    = engine.px_area,
        area_unit  = engine.area_unit,
    )

    # ── 4. Observed extent + comparison ──────────────────────────────────────
    observed_df = compute_observed_extent(
        ts, engine.classes, px_area_ha, cache_path=cache_path
    )
    extent_comparison = compare_extent_trajectories(observed_df, area_modal_df)

    # ── 5. Spatial agreement for every shared year with a modal raster ───────
    spatial_results: dict[int, dict] = {}
    for year, sim_raster in modal_maps.items():
        if year not in ts.years:
            continue
        obs_raster = ts.stack[ts.year_index(year)]
        try:
            spatial_results[year] = spatial_agreement(sim_raster, obs_raster, engine.classes)
        except ValueError as e:
            print(f"  [Skip] spatial_agreement for year {year}: {e}")

    # ── 6. Flag diverging classes and drill into pathway drift ───────────────
    flagged_classes: list[str] = []
    drift: dict[str, pd.DataFrame] = {}

    final_year = extent_comparison["year"].max()
    final_rows = extent_comparison[extent_comparison["year"] == final_year]
    for _, row in final_rows.iterrows():
        if pd.notna(row["pct_diff"]) and abs(row["pct_diff"]) > flag_threshold_pct:
            flagged_classes.append(row["class_name"])

    name_to_id = {sc.name: cid for cid, sc in engine.classes.items()}
    for class_name in flagged_classes:
        cid = name_to_id.get(class_name)
        if cid is None:
            continue
        drift[class_name] = attribute_extent_drift(trans_df, cid, engine.classes)

    if flagged_classes:
        print(f"  [Flagged] {len(flagged_classes)} class(es) diverged >"
              f"{flag_threshold_pct}% by {final_year}: {flagged_classes}")
    else:
        print(f"  No class diverged beyond {flag_threshold_pct}% by {final_year}")

    # ── 7. Overlay plot: observed line on the simulated uncertainty band ─────
    plot_path = summary_dir / "hindcast_overlay.png"
    _plot_hindcast_overlay(area_df, observed_df, engine.classes, plot_path)

    return HindcastResult(
        extent_comparison = extent_comparison,
        spatial_agreement = spatial_results,
        drift             = drift,
        flagged_classes   = flagged_classes,
        plot_path         = plot_path,
        area_df           = area_df,
        trans_df          = trans_df,
    )


def _write_raster_from_stack(ts: LULCTimeSeries, year: int, out_path: Path) -> None:
    """Write ts.stack[year_index] to a GeoTIFF using ts.profile (rasterio)."""
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx = ts.year_index(year)
    profile = ts.profile.copy()
    profile.update(count=1, dtype="uint8")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ts.stack[idx].astype(np.uint8), 1)
    print(f"  Wrote hindcast initial raster for year {year}: '{out_path}'")


def _plot_hindcast_overlay(
    area_df:     pd.DataFrame,
    observed_df: pd.DataFrame,
    classes:     dict,
    out_path:    Path,
) -> None:
    """
    Area over time: simulated median (line) + min-max band (shaded), same
    as outputs.plot_area_envelope(), with the observed historical trajectory
    overlaid as a dashed line -- so it's visible whether history falls
    inside the model's own stochastic spread, rather than comparing only
    to the median.
    """
    if area_df.empty:
        print("  [Skip] area_df empty -- no hindcast overlay plot generated")
        return

    acol = _area_col(area_df)
    years = sorted(area_df["year"].unique())
    class_ids = sorted(classes.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    for cid in class_ids:
        sc = classes[cid]
        color = _class_rgb(sc)
        subset = area_df[area_df["class_id"] == cid]

        stats = (
            subset.groupby("year")[acol]
            .agg(["median", "min", "max"])
            .reindex(years)
        )
        ax.plot(years, stats["median"], color=color, linewidth=2,
                label=f"{sc.name} (sim median)", zorder=3)
        ax.fill_between(years, stats["min"], stats["max"],
                         color=color, alpha=0.20, zorder=2)

        obs_subset = observed_df[observed_df["class_name"] == sc.name]
        obs_stats = obs_subset.set_index("year")["area_ha"].reindex(years)
        ax.plot(years, obs_stats, color=color, linewidth=2, linestyle="--",
                marker="o", markersize=3, zorder=4)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Area (ha)", fontsize=11)
    ax.set_title(
        "Hindcast: Simulated (solid, band = min-max) vs. Observed (dashed)",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Hindcast overlay plot saved to: '{out_path}'")
