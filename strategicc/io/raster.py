"""
strategicc/io/raster.py  —  v3.6
-----------------------
Read and write GeoTIFF rasters using Pillow (no GDAL dependency).

v3.6 changes
------------
* Pixel-area calculation is now CRS-aware. Previously, _pixel_area_ha()
  unconditionally treated the pixel scale as degrees (assumed a
  geographic CRS). Rasters in a projected CRS (e.g. UTM, pixel scale
  already in metres) were silently miscalculated -- often by several
  orders of magnitude. The GeoKeyDirectoryTag (34735) is now parsed to
  detect GTModelTypeGeoKey (geographic vs projected vs geocentric) and
  branch accordingly.
* Added CRSInfo + assert_crs_consistent() to let callers verify that
  multiple rasters used in the same run (state class/LULC, age, spatial
  multipliers) share a compatible CRS, and block the run with a clear
  error if they don't -- a silent CRS mismatch between rasters is just
  as capable of corrupting area-derived accounting as an unhandled
  projected CRS.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image


# ── Tag constants (GeoTIFF) ───────────────────────────────────────────────────
_TAG_TIE_POINT          = 33922   # ModelTiepointTag
_TAG_PIXEL_SCALE         = 33550   # ModelPixelScaleTag
_TAG_GEO_KEY_DIRECTORY   = 34735   # GeoKeyDirectoryTag

# ── GeoKey IDs (subset needed for CRS type / identity detection) ─────────────
_GEOKEY_GT_MODEL_TYPE       = 1024   # 1=Projected, 2=Geographic, 3=Geocentric
_GEOKEY_GEOGRAPHIC_TYPE     = 2048   # EPSG code when GTModelType == Geographic
_GEOKEY_PROJECTED_CS_TYPE   = 3072   # EPSG code when GTModelType == Projected
_GEOKEY_PROJ_LINEAR_UNITS   = 3076   # e.g. 9001 = metre

_GTMODELTYPE_PROJECTED  = 1
_GTMODELTYPE_GEOGRAPHIC = 2
_GTMODELTYPE_GEOCENTRIC = 3
_EPSG_LINEAR_UNIT_METRE = 9001

_MODEL_TYPE_LABELS = {
    _GTMODELTYPE_PROJECTED:  "projected",
    _GTMODELTYPE_GEOGRAPHIC: "geographic",
    _GTMODELTYPE_GEOCENTRIC: "geocentric",
}

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


def _parse_geokey(tags: dict, key_id: int) -> int | None:
    """
    Parse a single GeoKey's value out of GeoKeyDirectoryTag (34735).

    GeoKeyDirectoryTag layout: a 4-value header
    (KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys)
    followed by NumberOfKeys 4-tuples of (KeyID, TIFFTagLocation, Count,
    Value_Offset). Only keys with TIFFTagLocation == 0 (value stored
    inline as a SHORT) are resolved here -- sufficient for the
    model-type / EPSG-code keys this module needs; ASCII/DOUBLE-valued
    keys (TIFFTagLocation != 0) are not needed for CRS-type detection.

    Returns None if the tag is absent, malformed, or the key isn't found.
    """
    gk = tags.get(_TAG_GEO_KEY_DIRECTORY)
    if not gk or len(gk) < 4:
        return None
    try:
        n_keys = int(gk[3])
    except (TypeError, ValueError):
        return None
    for i in range(n_keys):
        base = 4 + i * 4
        if base + 3 >= len(gk):
            break
        key_id_i, tag_loc, _count, value = gk[base:base + 4]
        if int(key_id_i) == key_id and int(tag_loc) == 0:
            return int(value)
    return None


@dataclass(frozen=True)
class CRSInfo:
    """
    Lightweight, comparable summary of a raster's CRS, sufficient to
    detect (a) whether it's geographic/projected/geocentric, for correct
    pixel-area calculation, and (b) whether two rasters share the same
    CRS, for cross-raster consistency checks. Deliberately not a full
    CRS/projection object -- this module has no GDAL/pyproj dependency.
    """
    model_type: int | None   # 1=Projected, 2=Geographic, 3=Geocentric, None=unknown
    epsg:       int | None   # EPSG code if resolvable, else None
    source:     str = "tags" # "tags" (parsed from GeoKeyDirectoryTag) or "rasterio"

    @classmethod
    def from_tags(cls, tags: dict) -> "CRSInfo":
        model_type = _parse_geokey(tags, _GEOKEY_GT_MODEL_TYPE)
        if model_type == _GTMODELTYPE_PROJECTED:
            epsg = _parse_geokey(tags, _GEOKEY_PROJECTED_CS_TYPE)
        elif model_type == _GTMODELTYPE_GEOGRAPHIC:
            epsg = _parse_geokey(tags, _GEOKEY_GEOGRAPHIC_TYPE)
        else:
            epsg = None
        return cls(model_type=model_type, epsg=epsg, source="tags")

    @classmethod
    def from_rasterio_crs(cls, crs) -> "CRSInfo":
        """Build from a rasterio CRS object (used by rasterio-backed readers)."""
        if crs is None:
            return cls(model_type=None, epsg=None, source="rasterio")
        if crs.is_projected:
            model_type = _GTMODELTYPE_PROJECTED
        elif crs.is_geographic:
            model_type = _GTMODELTYPE_GEOGRAPHIC
        else:
            model_type = None
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
        return cls(model_type=model_type, epsg=epsg, source="rasterio")

    @property
    def known(self) -> bool:
        return self.model_type is not None

    def describe(self) -> str:
        if not self.known:
            return "unknown CRS (no GeoKeyDirectoryTag / CRS metadata found)"
        type_label = _MODEL_TYPE_LABELS.get(self.model_type, f"type={self.model_type}")
        epsg_label = f"EPSG:{self.epsg}" if self.epsg else "EPSG unknown"
        return f"{epsg_label} ({type_label})"

    def compare(self, other: "CRSInfo") -> tuple[str, str]:
        """
        Returns (status, reason). status is one of:
          "match"    — same model_type, and same epsg if both known
          "mismatch" — model_type differs, or both epsg known and differ
          "unknown"  — one or both sides have no usable CRS metadata,
                       so consistency cannot be verified either way
        """
        if not self.known or not other.known:
            return "unknown", "one or both rasters have no CRS metadata"
        if self.model_type != other.model_type:
            return "mismatch", (
                f"{_MODEL_TYPE_LABELS.get(self.model_type, self.model_type)} "
                f"vs {_MODEL_TYPE_LABELS.get(other.model_type, other.model_type)}"
            )
        if self.epsg and other.epsg and self.epsg != other.epsg:
            return "mismatch", f"EPSG:{self.epsg} vs EPSG:{other.epsg}"
        return "match", ""


def get_crs_info(tags: dict) -> CRSInfo:
    """Public wrapper — build a CRSInfo from a PIL tag_v2 dict."""
    return CRSInfo.from_tags(tags)


def assert_crs_consistent(
    reference: CRSInfo,
    other:     CRSInfo,
    reference_label: str,
    other_label:     str,
) -> None:
    """
    Raise ValueError if `other`'s CRS is confirmed to differ from
    `reference`'s. Prints a warning (does not raise) if consistency
    can't be verified because one or both sides lack CRS metadata --
    blocking on unknown-but-possibly-fine data would be too aggressive
    for rasters that simply don't carry full georeferencing tags.
    """
    status, reason = reference.compare(other)
    if status == "mismatch":
        raise ValueError(
            f"CRS mismatch between '{reference_label}' and '{other_label}': "
            f"{reason}. '{reference_label}' is {reference.describe()}; "
            f"'{other_label}' is {other.describe()}. All rasters used in the "
            f"same run (state class/LULC, age, spatial multipliers) must "
            f"share the same CRS -- pixel-area-derived accounting (SEEA-EA "
            f"valuation, transition areas, etc.) is computed once from the "
            f"LULC raster's pixel size and silently misapplied to any other "
            f"raster on a different grid/CRS. Reproject '{other_label}' to "
            f"match '{reference_label}' before running."
        )
    if status == "unknown":
        print(
            f"  [Warning] Could not verify CRS consistency between "
            f"'{reference_label}' ({reference.describe()}) and "
            f"'{other_label}' ({other.describe()}) -- proceeding, but this "
            f"is unverified."
        )


def _pixel_area_ha(tags: dict) -> float:
    """
    Return pixel area in hectares from GeoTIFF tags.

    CRS-aware (v3.6): branches on GTModelTypeGeoKey.
    - Projected CRS: pixel scale is assumed to already be in metres (the
      overwhelming common case for LULC work, e.g. UTM) -- area = w * h,
      converted m² -> ha directly, no degree conversion. Warns if the
      CRS's linear unit is present and isn't metres (EPSG:9001).
    - Geographic CRS (or CRS type undetermined): pixel scale is treated
      as degrees and converted via a flat 111,000 m/degree approximation
      -- this is the module's original (pre-3.6) behaviour, preserved
      unchanged for this case. Warns if the CRS type couldn't be
      determined at all, since the geographic assumption is then a
      guess, not a detection.
    """
    px_w = tags[_TAG_PIXEL_SCALE][0]
    px_h = tags[_TAG_PIXEL_SCALE][1]
    model_type = _parse_geokey(tags, _GEOKEY_GT_MODEL_TYPE)

    if model_type == _GTMODELTYPE_PROJECTED:
        linear_unit = _parse_geokey(tags, _GEOKEY_PROJ_LINEAR_UNITS)
        if linear_unit is not None and linear_unit != _EPSG_LINEAR_UNIT_METRE:
            print(
                f"  [Warning] Projected CRS reports linear unit code "
                f"{linear_unit}, not metres (EPSG:9001) -- pixel area "
                f"calculation assumes metres and will be wrong for this "
                f"raster."
            )
        return (px_w * px_h) / 10_000

    if model_type is None:
        print(
            "  [Warning] Could not determine raster CRS type from "
            "GeoKeyDirectoryTag -- assuming geographic (degrees), matching "
            "this function's historical behaviour. If this raster is "
            "actually in a projected CRS (e.g. UTM, metres), pixel area "
            "will be computed incorrectly."
        )

    # Geographic (model_type == 2), or undetermined (fallback, warned above)
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


def resolve_mult_dir(mult_dir: str | Path) -> Path:
    """
    Resolve a spatial multiplier directory path, transparently supporting
    a zipped multiplier set (v3.4).

    If `mult_dir` points at an existing .zip file, it is extracted ONCE
    to a sibling folder (same stem, no extension — e.g.
    "spatmult_uploads.zip" -> "spatmult_uploads/") and that folder's path
    is returned. If the sibling folder already exists with content,
    extraction is skipped (treated as already-extracted). If `mult_dir`
    is already a folder, it is returned unchanged — no zip handling
    needed.

    Parameters
    ----------
    mult_dir : path to either a folder of multiplier rasters, or a .zip
               file containing them

    Returns
    -------
    Path to a folder containing the multiplier rasters
    """
    import zipfile

    mult_dir = Path(mult_dir)

    if mult_dir.is_dir():
        return mult_dir

    if mult_dir.suffix.lower() == ".zip" and mult_dir.exists():
        sibling_dir = mult_dir.with_suffix("")

        if sibling_dir.is_dir() and any(sibling_dir.iterdir()):
            print(f"  [Cache hit] '{sibling_dir}' already extracted from "
                  f"'{mult_dir}' — skipping re-extraction.")
            return sibling_dir

        sibling_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(mult_dir, "r") as z:
            z.extractall(sibling_dir)

        # If the zip contained a single subfolder, descend into it
        entries = [p for p in sibling_dir.iterdir()
                   if not p.name.startswith("__MACOSX")]
        if len(entries) == 1 and entries[0].is_dir():
            sibling_dir = entries[0]

        print(f"  Extracted multiplier zip '{mult_dir}' -> '{sibling_dir}'")
        return sibling_dir

    # Neither an existing folder nor a zip — return as-is, let the
    # downstream loader raise a clear "not found" error for missing files.
    return mult_dir
