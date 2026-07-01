"""
strategicc/core/spatial.py  —  1c. Spatial multipliers  (v3.6: CRS check)
-------------------------------------------------
Loads per-group spatial multiplier rasters (normalised 0–1, white = high
suitability / close to feature) and applies a power-sharpening curve to
concentrate transitions near feature edges.

Assumptions
-----------
* Input rasters are already normalised to [0, 1] (white = close to feature).
* If a raster's max > 1 it is re-normalised automatically.
* Groups with no matching file fall back to a uniform ones array (no effect).
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
) -> dict[str, np.ndarray]:
    """
    Build a dict mapping group_name → float32 multiplier array of shape
    `target_shape`, ready to be element-wise multiplied with effective
    transition probabilities in the engine.

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
    dict[group_name, float32 ndarray]  values in [0, 1]
    """
    mult_dir = Path(mult_dir)
    multipliers: dict[str, np.ndarray] = {}

    for entry in entries:
        group = entry.group
        fpath = mult_dir / entry.filename

        if not fpath.exists():
            print(f"  [Missing] '{group}' → '{fpath}' not found — using ones")
            multipliers[group] = np.ones(target_shape, dtype=np.float32)
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

        multipliers[group] = arr

        nonzero = int(np.count_nonzero(arr))
        pos_min = float(arr[arr > 0].min()) if nonzero else 0.0
        print(
            f"  Loaded [{group}] (^{power}): "
            f"nonzero={nonzero:,}  min={pos_min:.4f}  max={arr.max():.4f}"
        )

    return multipliers


def get_multiplier(
    multipliers: dict[str, np.ndarray],
    group: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Return the spatial multiplier for `group`, or a ones array if absent.
    Convenience wrapper used by the engine.
    """
    return multipliers.get(group, np.ones(shape, dtype=np.float32))
