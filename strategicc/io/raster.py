"""
strategicc/io/raster.py
-----------------------
Read and write GeoTIFF rasters using Pillow (no GDAL dependency).
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image


# ── Tag constants (GeoTIFF) ───────────────────────────────────────────────────
_TAG_TIE_POINT   = 33922   # ModelTiepointTag
_TAG_PIXEL_SCALE = 33550   # ModelPixelScaleTag

# ── Area unit conversion factors (from hectares) ──────────────────────────────
_UNIT_FACTORS: dict[str, float] = {
    "ha":  1.0,
    "km2": 0.01,
    "px":  None,   # special: ignore pixel size, return 1.0 per pixel
}

UNIT_LABELS: dict[str, str] = {
    "ha":  "ha",
    "km2": "km²",
    "px":  "pixels",
}


def _pixel_area_ha(tags: dict) -> float:
    """Return pixel area in hectares from GeoTIFF tags (assumes degree CRS)."""
    px_w = tags[_TAG_PIXEL_SCALE][0]
    px_h = tags[_TAG_PIXEL_SCALE][1]
    return (px_w * 111_000) * (px_h * 111_000) / 10_000


def get_pixel_area(px_area_ha: float, unit: str) -> float:
    """
    Convert px_area_ha to the target unit.

    Parameters
    ----------
    px_area_ha : pixel area in hectares (from read_lulc / read_tiff)
    unit       : one of "ha", "km2", "px"

    Returns
    -------
    area per pixel in the chosen unit
    """
    if unit not in _UNIT_FACTORS:
        raise ValueError(
            f"Unknown AREA_UNIT '{unit}'. Must be one of: "
            f"{list(_UNIT_FACTORS.keys())}"
        )
    if unit == "px":
        return 1.0
    return px_area_ha * _UNIT_FACTORS[unit]


def read_tiff(path: str | Path) -> tuple[np.ndarray, float, dict]:
    """
    Read any single-band GeoTIFF.

    Returns
    -------
    arr        : float32 ndarray, shape (rows, cols)
    px_area_ha : pixel area in hectares
    tags       : raw tag_v2 dict
    """
    img  = Image.open(str(path))
    arr  = np.array(img, dtype=np.float32)
    tags = img.tag_v2
    return arr, _pixel_area_ha(tags), tags


def read_lulc(path: str | Path) -> tuple[np.ndarray, float, dict]:
    """
    Read a LULC raster (uint8 class IDs).

    Returns
    -------
    arr        : uint8 ndarray, shape (rows, cols)
    px_area_ha : pixel area in hectares
    tags       : raw tag_v2 dict
    """
    img  = Image.open(str(path))
    arr  = np.array(img, dtype=np.uint8)
    tags = img.tag_v2
    return arr, _pixel_area_ha(tags), tags


def save_tifs(
    maps:       list[np.ndarray],
    start_year: int,
    src_tags:   dict,
    out_dir:    str | Path,
) -> None:
    """
    Save a list of LULC arrays as georeferenced GeoTIFFs.

    Parameters
    ----------
    maps       : list of uint8 arrays, one per timestep (index 0 = initial year)
    start_year : year label for the first map
    src_tags   : tag_v2 dict from the source raster (preserves georeferencing)
    out_dir    : output directory (created if absent)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_tags = {k: src_tags[k]
                 for k in (_TAG_TIE_POINT, _TAG_PIXEL_SCALE, 34735, 34736, 34737)
                 if k in src_tags}

    for t, arr in enumerate(maps):
        year     = start_year + t
        out_path = out_dir / f"lulc_{year}.tif"
        save_kwargs = {"compression": "lzw"}
        if keep_tags:
            save_kwargs["tiffinfo"] = keep_tags
        Image.fromarray(arr.astype(np.uint8), mode="L").save(
            str(out_path), **save_kwargs
        )
