"""
strategicc/core/age.py  —  v3.8
---------------------------------
Age tracking for the simulation engine.

Responsibilities
----------------
1. build_initial_age()     — construct t=0 age map from raster or CSV assumptions
2. update_age()            — increment age per cell, reset on transition
3. apply_age_gates()       — mask transitions by AgeMin / AgeMax per rule

Design
------
Age is tracked as a uint16 array (max 65,535 years — more than sufficient).
All operations are vectorised with NumPy; no per-cell Python loops.

Age reset behaviour (matches ST-Sim):
- If AgeReset=True (default): cell age → 0 on transition
- If AgeRelative is set:      cell age → AgeRelative on transition
- If AgeReset=False:          cell age continues incrementing (no reset)
"""

from __future__ import annotations
import numpy as np
from strategicc.io.csv_loader import InitialAgeRule
from strategicc.io.raster import get_crs_info, CRSInfo


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build initial age map
# ─────────────────────────────────────────────────────────────────────────────

def build_initial_age_from_raster(path: str) -> tuple[np.ndarray, CRSInfo]:
    """
    Load age raster from a GeoTIFF.

    Tries Pillow first (no extra dependency for the common case). Some
    16-bit unsigned GeoTIFFs written by rasterio (e.g. by
    strategicc.calibration.save_age_raster()) use a tag layout that
    Pillow's TIFF decoder cannot read ("unknown raw mode for given image
    mode") even though the file is perfectly valid — this is a known
    PIL/rasterio interoperability gap, not a corrupt file. In that case,
    fall back to rasterio if it's installed, since it's already a soft
    dependency for anyone using the calibration module that produces
    these files in the first place.

    Returns
    -------
    (arr, crs_info) : uint16 array shape (rows, cols), and a CRSInfo
                       (v3.6) describing the raster's CRS — pass this to
                       assert_crs_consistent() against the LULC raster's
                       CRSInfo to catch a mismatched age raster before it
                       silently corrupts area-derived accounting.
    """
    from PIL import Image
    try:
        img = Image.open(str(path))
        arr = np.array(img, dtype=np.uint16)
        crs_info = get_crs_info(img.tag_v2)
    except ValueError as e:
        if "unknown raw mode" not in str(e):
            raise
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                f"Could not read '{path}' with Pillow (likely a rasterio-"
                f"written 16-bit GeoTIFF using a tag layout Pillow's TIFF "
                f"decoder doesn't support). Install rasterio as a fallback "
                f"reader: pip install rasterio"
            ) from e
        with rasterio.open(str(path)) as src:
            arr = src.read(1).astype(np.uint16)
            crs_info = CRSInfo.from_rasterio_crs(src.crs)

    print(f"  Age raster loaded: shape={arr.shape}  "
          f"min={arr.min()}  max={arr.max()}  mean={arr.mean():.1f}")
    return arr, crs_info


def build_initial_age_from_rules(
    lulc_map:  np.ndarray,
    classes:   dict,
    rules:     list[InitialAgeRule],
    rng:       np.random.Generator,
) -> np.ndarray:
    """
    Build an initial age map by sampling from per-class truncated normal
    distributions defined in InitialAge.csv.

    Cells whose class is not in rules get age = 0.

    Parameters
    ----------
    lulc_map : uint8 array (rows, cols) — initial LULC at t=0
    classes  : dict[int, StateClass]
    rules    : list of InitialAgeRule from load_initial_age_rules()
    rng      : numpy Generator for reproducibility

    Returns
    -------
    uint16 array, shape (rows, cols)
    """
    rows, cols = lulc_map.shape
    age_map    = np.zeros((rows, cols), dtype=np.uint16)

    # Build class name → id lookup
    name_to_id = {sc.name: cid for cid, sc in classes.items()}
    # Also match "Mangrove:All" → "Mangrove"
    for cid, sc in classes.items():
        name_to_id[sc.full_name] = cid

    for rule in rules:
        cid = name_to_id.get(rule.state_class)
        if cid is None:
            print(f"  [Warning] InitialAge: unknown class '{rule.state_class}' — skipped")
            continue

        mask = (lulc_map == cid)
        n    = int(mask.sum())
        if n == 0:
            continue

        if rule.age_sd > 0:
            # Truncated normal: clip to [age_min, age_max]
            from scipy import stats as scipy_stats
            a = (rule.age_min - rule.age_mean) / rule.age_sd
            b = (rule.age_max - rule.age_mean) / rule.age_sd
            samples = scipy_stats.truncnorm.rvs(
                a, b, loc=rule.age_mean, scale=rule.age_sd,
                size=n, random_state=rng.integers(0, 2**31)
            ).astype(np.uint16)
        else:
            samples = np.full(n, int(rule.age_mean), dtype=np.uint16)

        samples = np.clip(samples, rule.age_min, rule.age_max).astype(np.uint16)
        age_map[mask] = samples

        print(f"  Class '{rule.state_class}': "
              f"n={n:,}  age mean={samples.mean():.1f}  "
              f"[{samples.min()}–{samples.max()}]")

    return age_map


# ─────────────────────────────────────────────────────────────────────────────
# 2. Update age each timestep
# ─────────────────────────────────────────────────────────────────────────────

def update_age(
    age_map:          np.ndarray,
    transition_fired: np.ndarray,
    fire_age_reset:   np.ndarray,
    fire_age_value:   np.ndarray,
) -> np.ndarray:
    """
    Update the age map for one timestep.

    Rules (matches ST-Sim):
    - Non-transitioned cells: age += 1
    - Transitioned + AgeReset=True:          age → 0
    - Transitioned + AgeRelative is set:     age → AgeRelative value
    - Transitioned + AgeReset=False:         age += 1 (no reset)

    Parameters
    ----------
    age_map          : current uint16 age array (rows, cols)
    transition_fired : bool array — True where any transition occurred
    fire_age_reset   : bool array — True where age should be reset to 0
    fire_age_value   : int16 array — specific age value to set (-1 = ignore)

    Returns
    -------
    new uint16 age array
    """
    new_age = age_map.copy()

    # All cells age by 1 (will be overridden for reset cells below)
    new_age = np.clip(new_age.astype(np.int32) + 1, 0, 65535).astype(np.uint16)

    # Reset age to 0 where flagged
    new_age[fire_age_reset] = 0

    # Set specific relative age where defined (overrides reset)
    has_relative = (fire_age_value >= 0)
    new_age[has_relative] = fire_age_value[has_relative].astype(np.uint16)

    return new_age


# ─────────────────────────────────────────────────────────────────────────────
# 3. Age gates — mask transition eligibility by AgeMin / AgeMax
# ─────────────────────────────────────────────────────────────────────────────

def age_gate_mask(
    age_map: np.ndarray,
    age_min: int | None,
    age_max: int | None,
) -> np.ndarray:
    """
    Return a boolean mask of cells that satisfy AgeMin / AgeMax constraints.

    Parameters
    ----------
    age_map : current uint16 age array
    age_min : minimum age (inclusive) — None = no lower bound
    age_max : maximum age (inclusive) — None = no upper bound

    Returns
    -------
    bool array — True where cell age satisfies the constraint
    """
    mask = np.ones(age_map.shape, dtype=bool)
    if age_min is not None:
        mask &= (age_map >= age_min)
    if age_max is not None:
        mask &= (age_map <= age_max)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save age raster
# ─────────────────────────────────────────────────────────────────────────────

def save_age_tif(
    age_map:  np.ndarray,
    year:     int,
    out_dir,
    src_tags: dict,
) -> None:
    """Save a uint16 age map as a GeoTIFF."""
    from pathlib import Path
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from strategicc.io.raster import _TAG_TIE_POINT, _TAG_PIXEL_SCALE
    keep_tags = {k: src_tags[k]
                 for k in (_TAG_TIE_POINT, _TAG_PIXEL_SCALE, 34735, 34736, 34737)
                 if k in src_tags}

    save_kwargs = {"compression": "lzw"}
    if keep_tags:
        save_kwargs["tiffinfo"] = keep_tags

    out_path = out_dir / f"age_{year}.tif"
    Image.fromarray(age_map, mode="I;16").save(str(out_path), **save_kwargs)
