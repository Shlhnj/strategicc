"""
strategicc/calibration/age.py  —  v3.9
------------------------------------------
Backtrack a continuous (or binned) age-since-occupancy raster for ALL
classes simultaneously, given an annual LULC time series.

Generalises the original single-class script:
  - Computes age for every class present in the stack, not just one target
  - Defaults to continuous age in years (matches StrategiccEngine's age
    tracking design from v2.3)
  - Binning into discrete classes remains available via `age_bins=`

Algorithm
---------
For each class c and each cell, walk backwards from the baseline (most
recent) year. The age at the baseline year is the number of consecutive
prior years the cell held class c, ending at the baseline.

Cells whose class differs from c at baseline get age=NODATA for that
class's raster (age is class-specific and only meaningful where the
cell currently holds that class).

The combined output `age_combined` assigns each cell the age computed
for ITS OWN class at baseline — this is the raster to feed into
StrategiccEngine's AGE_RASTER_PATH.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from strategicc.calibration.loader import LULCTimeSeries


NODATA_AGE = 65535   # uint16 sentinel — matches engine.core.age uint16 dtype


@dataclass
class AgeRasterResult:
    """
    Output of compute_age_raster().

    Attributes
    ----------
    age_combined : uint16 array — age (in years, or bin label) for each
                   cell using ITS OWN class at baseline. This is the file
                   to pass as config.AGE_RASTER_PATH.
    age_per_class : dict[class_id, uint16 array] — per-class age raster,
                    NODATA_AGE where the cell is not that class at baseline.
    full_record_mask : bool array — True where the cell held its baseline
                    class for the ENTIRE time series (true age may exceed
                    what's observable — flagged for caution).
    baseline_year : int — the year age is computed relative to (latest
                    year in the stack).
    profile : georeferencing profile from the source rasters.
    """
    age_combined:     np.ndarray
    age_per_class:    dict[int, np.ndarray]
    full_record_mask: np.ndarray
    baseline_year:    int
    profile:          dict


def compute_age_raster(
    ts:           LULCTimeSeries,
    class_ids:    list[int] | None = None,
    age_bins:     list[tuple[int, int, int]] | None = None,
    baseline_year: int | None = None,
) -> AgeRasterResult:
    """
    Backtrack age-since-occupancy for all (or specified) classes.

    Parameters
    ----------
    ts            : LULCTimeSeries from load_lulc_timeseries()
    class_ids     : list of class IDs to compute age for. If None,
                    auto-detects all unique class values present in the
                    baseline year (excluding 0, treated as nodata).
    age_bins      : optional list of (age_min, age_max, bin_label) tuples
                    to bin continuous age into discrete classes — matches
                    the original script's AGE_CLASSES pattern. If None
                    (default), age is returned continuous in years.
    baseline_year : year to compute age relative to. Defaults to the
                    most recent year in the timeseries.

    Returns
    -------
    AgeRasterResult
    """
    if baseline_year is None:
        baseline_year = ts.years[-1]
    baseline_idx = ts.year_index(baseline_year)

    baseline_map = ts.stack[baseline_idx]
    rows, cols   = baseline_map.shape

    if class_ids is None:
        class_ids = sorted(int(c) for c in np.unique(baseline_map) if c != 0)
        print(f"  Auto-detected {len(class_ids)} class(es) at baseline "
              f"{baseline_year}: {class_ids}")

    age_per_class: dict[int, np.ndarray] = {}
    age_combined  = np.full((rows, cols), NODATA_AGE, dtype=np.uint16)
    full_record_mask = np.zeros((rows, cols), dtype=bool)

    n_years_back = baseline_idx + 1   # number of years available to walk back

    for cls in class_ids:
        is_target_at_baseline = (baseline_map == cls)
        n_baseline_cells = int(is_target_at_baseline.sum())
        if n_baseline_cells == 0:
            continue

        # streak_start_idx: the earliest index (walking back from baseline)
        # at which the cell is STILL the target class, contiguously.
        streak_start_idx = np.full((rows, cols), -1, dtype=np.int32)
        streak_start_idx[is_target_at_baseline] = baseline_idx

        for i in range(baseline_idx - 1, -1, -1):
            still_target = (ts.stack[i] == cls) & (streak_start_idx != -1)
            streak_start_idx[still_target] = i

        age_raw = np.full((rows, cols), NODATA_AGE, dtype=np.uint16)
        valid   = streak_start_idx != -1
        age_raw[valid] = (baseline_idx - streak_start_idx[valid]).astype(np.uint16)

        # Flag cells that were this class for the ENTIRE observable record
        oldest_possible = valid & (age_raw == (n_years_back - 1))
        full_record_mask |= oldest_possible

        age_per_class[cls] = age_raw
        age_combined[is_target_at_baseline] = age_raw[is_target_at_baseline]

        print(f"  Class {cls}: {n_baseline_cells:,} cells at baseline  "
              f"mean_age={age_raw[valid].mean():.1f}  "
              f"full_record={int(oldest_possible.sum()):,}")

    # ── Optional binning ───────────────────────────────────────────────────
    if age_bins is not None:
        print(f"  Binning into {len(age_bins)} discrete class(es)...")
        binned_combined = np.full((rows, cols), NODATA_AGE, dtype=np.uint16)
        binned_per_class: dict[int, np.ndarray] = {}

        for cls, age_raw in age_per_class.items():
            binned = np.full((rows, cols), NODATA_AGE, dtype=np.uint16)
            valid  = age_raw != NODATA_AGE
            for age_min, age_max, label in age_bins:
                mask = valid & (age_raw >= age_min) & (age_raw <= age_max)
                binned[mask] = label
            binned_per_class[cls] = binned
            binned_combined[baseline_map == cls] = binned[baseline_map == cls]

        age_per_class = binned_per_class
        age_combined  = binned_combined

    return AgeRasterResult(
        age_combined     = age_combined,
        age_per_class    = age_per_class,
        full_record_mask = full_record_mask,
        baseline_year    = baseline_year,
        profile          = ts.profile,
    )


def save_age_raster(
    result:   AgeRasterResult,
    out_path: str | Path | None = None,
    nodata:   int = NODATA_AGE,
) -> Path:
    """
    Write the combined age raster to disk as a GeoTIFF, preserving
    georeferencing from the source timeseries.

    Parameters
    ----------
    result   : AgeRasterResult from compute_age_raster()
    out_path : destination path; defaults to calibration_result/age.tif
    nodata   : nodata value for uint16 GeoTIFF (default 65535)

    Returns
    -------
    Path actually written to

    Requires rasterio.
    """
    import rasterio
    from strategicc.calibration.paths import AGE_RASTER

    out_path = Path(out_path) if out_path is not None else AGE_RASTER
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile  = result.profile.copy()
    profile.update(dtype="uint16", count=1, nodata=nodata)

    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(result.age_combined, 1)

    print(f"  Saved: {out_path}")
    return out_path
