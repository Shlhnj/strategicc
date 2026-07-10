"""
strategicc/core/spatial.py  —  1c. Spatial multipliers  (v3.15: time-varying)
-------------------------------------------------
Loads per-group spatial multiplier rasters (normalised 0–1, white = high
suitability / close to feature) and applies a power-sharpening curve to
concentrate transitions near feature edges.

Time-varying multipliers (v3.15)
---------------------------------
`TransitionSpatialMultipliers.csv` supports multiple `Timestep` rows per
group (e.g. a distance-to-Other_vegetation raster rebuilt for several
snapshot years). `Timestep` follows the same convention as
`TransitionTargets.csv` / `resolve_targets_per_timestep()`: a 0-based
simulated-timestep index (timestep 0 = first simulated year), NOT a
calendar year. A row with `timestep=None` (the pre-v3.15 convention: no
Timestep column, or a blank value) sorts before every explicit timestep
and applies from t=0 onward — this is what preserves old, single-raster
setups unchanged.

`load_spatial_multipliers()` now loads every entry (it no longer collapses
multiple rows for the same group into "whichever was read last") and
returns them sorted per group. `resolve_spatial_multipliers_per_timestep()`
forward-fills that per-group series into an explicit per-timestep lookup,
mirroring `targets.resolve_targets_per_timestep()`. Timesteps before a
group's earliest entry resolve to `None` (no raster loaded yet); the
engine treats `None` as "no effect" (ones array) rather than reaching
into the future for a later raster — using a later-vintage raster to
steer an earlier simulated year would be a look-ahead-bias risk (e.g. a
2022-derived distance-to-Other_vegetation raster steering a 2005
transition).

Assumptions
-----------
* Input rasters are already normalised to [0, 1] (white = close to feature).
* If a raster's max > 1 it is re-normalised automatically.
* Groups with no matching entry at all fall back to a uniform ones array
  (no effect), at every timestep.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from strategicc.io.raster import read_tiff, get_crs_info, assert_crs_consistent, CRSInfo
from strategicc.io.csv_loader import SpatialMultEntry
from strategicc import config


def load_spatial_multipliers(
    entries:      list[SpatialMultEntry],
    mult_dir:     str | Path,
    target_shape: tuple[int, int],
    reference_crs: CRSInfo | None = None,   # v3.6
) -> dict[str, list[tuple[int | None, np.ndarray]]]:
    """
    Load every spatial-multiplier raster entry and group them by
    TransitionGroupId, sorted by timestep (None-timestep entries first,
    matching `resolve_targets_per_timestep()`'s sort convention).

    Parameters
    ----------
    entries      : output of load_spatial_mult_index()
    mult_dir     : directory containing the multiplier TIF files
    target_shape : (rows, cols) — must match the LULC raster
    reference_crs : (v3.6) CRSInfo of the LULC raster, e.g.
                    engine's get_crs_info(engine.src_tags). If given,
                    every multiplier raster's CRS is checked against it
                    and a ValueError is raised on a confirmed mismatch —
                    a spatial multiplier on a different CRS/grid than the
                    LULC raster would be resampled (target_shape resize
                    below) as if it were aligned, silently corrupting
                    where transitions get concentrated. Pass None to skip
                    the check (e.g. in tests using synthetic rasters with
                    no georeferencing).

    Returns
    -------
    dict[group_name, list[(timestep_or_None, float32 ndarray in [0, 1])]],
    each group's list sorted ascending by (timestep is not None, timestep
    or 0) — i.e. a None-timestep entry (pre-v3.15 static convention) comes
    first, then explicit timesteps in increasing order. Pass this straight
    into `resolve_spatial_multipliers_per_timestep()`; don't index into it
    directly by group in new code, since a group may now have more than
    one entry.
    """
    mult_dir = Path(mult_dir)
    multipliers: dict[str, list[tuple[int | None, np.ndarray]]] = {}

    for entry in entries:
        group = entry.group
        fpath = mult_dir / entry.filename

        if not fpath.exists():
            print(f"  [Missing] '{group}' (timestep={entry.timestep}) → "
                  f"'{fpath}' not found — using ones")
            multipliers.setdefault(group, []).append(
                (entry.timestep, np.ones(target_shape, dtype=np.float32))
            )
            continue

        arr, _px_area_ha, tags = read_tiff(fpath)

        if reference_crs is not None:
            assert_crs_consistent(
                reference_crs, get_crs_info(tags),
                "LULC raster", f"spatial multiplier '{entry.filename}'",
            )

        # ── Resize to target if needed ────────────────────────────────────
        if arr.shape != target_shape:
            img_r = Image.fromarray(arr).resize(
                (target_shape[1], target_shape[0]), Image.NEAREST
            )
            arr = np.array(img_r, dtype=np.float32)

        # ── Normalise to [0, 1] ───────────────────────────────────────────
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max > 1.0 or arr_min < 0.0:
            arr = (arr - arr_min) / max(arr_max - arr_min, 1e-8)

        # ── Power sharpening ──────────────────────────────────────────────
        # Raises the surface to a power > 1 so only cells very close to
        # the feature retain meaningful probability.
        # e.g. power=6: 0.9→0.53, 0.5→0.016, 0.3→0.0007
        power = config.SHARPENING_POWER.get(group, config.DEFAULT_SHARPENING_POWER)
        arr   = np.power(arr, power).astype(np.float32)

        # Hard floor — anything below 1 % becomes exactly 0
        arr[arr < 0.01] = 0.0

        multipliers.setdefault(group, []).append((entry.timestep, arr))

        nonzero = int(np.count_nonzero(arr))
        pos_min = float(arr[arr > 0].min()) if nonzero else 0.0
        print(
            f"  Loaded [{group}] (timestep={entry.timestep}, ^{power}): "
            f"nonzero={nonzero:,}  min={pos_min:.4f}  max={arr.max():.4f}"
        )

    for group in multipliers:
        multipliers[group].sort(key=lambda pair: (pair[0] is not None, pair[0] or 0))
        n = len(multipliers[group])
        if n > 1:
            print(f"  [{group}] {n} time-varying entries loaded, "
                  f"timesteps={[t for t, _ in multipliers[group]]}")

    return multipliers


def resolve_spatial_multipliers_per_timestep(
    multipliers: dict[str, list[tuple[int | None, np.ndarray]]],
    n_timesteps: int,
) -> dict[int, dict[str, np.ndarray | None]]:
    """
    Forward-fill each group's timestep series into an explicit
    per-timestep lookup, mirroring `targets.resolve_targets_per_timestep()`.

    Parameters
    ----------
    multipliers : output of load_spatial_multipliers()
    n_timesteps : total number of simulation timesteps

    Returns
    -------
    dict[timestep_index, dict[group_name, array_or_None]]. `None` means no
    raster has become active yet for that group at that timestep (i.e. t
    is before the group's earliest entry) — the caller (get_multiplier)
    treats this as "no effect" (ones array), not as reaching backward to
    an unrelated group default or forward to a later raster.
    """
    resolved: dict[int, dict[str, np.ndarray | None]] = {
        t: {} for t in range(n_timesteps)
    }

    for group, raw_series in multipliers.items():
        # Sort defensively here too (not just in load_spatial_multipliers())
        # — this function's forward-fill scan assumes ascending order and
        # would silently mis-resolve on unsorted input otherwise, e.g. if
        # a caller ever builds this dict some other way than via
        # load_spatial_multipliers().
        series = sorted(raw_series, key=lambda pair: (pair[0] is not None, pair[0] or 0))
        current_arr: np.ndarray | None = None
        idx = 0

        for t in range(n_timesteps):
            while (
                idx < len(series)
                and (series[idx][0] is None or series[idx][0] <= t)
            ):
                current_arr = series[idx][1]
                idx += 1

            resolved[t][group] = current_arr

    return resolved


def get_multiplier(
    per_timestep: dict[int, dict[str, np.ndarray | None]],
    group: str,
    shape: tuple[int, int],
    t: int,
) -> np.ndarray:
    """
    Return the spatial multiplier for `group` at simulated timestep `t`,
    or a ones array if absent / not yet active at this timestep.

    Parameters
    ----------
    per_timestep : output of resolve_spatial_multipliers_per_timestep()
    group        : TransitionGroupId
    shape        : (rows, cols) fallback array shape
    t            : 0-based simulated timestep index
    """
    arr = per_timestep.get(t, {}).get(group)
    if arr is None:
        return np.ones(shape, dtype=np.float32)
    return arr
