"""
tests/test_raster_crs.py  —  v3.6
Unit tests for CRS-aware pixel area calculation and cross-raster CRS
consistency checking (strategicc/io/raster.py).
"""

import numpy as np
import pytest
from PIL import Image

from strategicc.io.raster import (
    _pixel_area_ha, _parse_geokey, get_crs_info, assert_crs_consistent,
    CRSInfo, read_tiff,
)


# ── Synthetic GeoKeyDirectoryTag fixtures ────────────────────────────────────
# Real-world example (decoded from an actual project raster): geographic,
# WGS84 (EPSG:4326).
GEOGRAPHIC_WGS84_GEOKEYS = (
    1, 1, 0, 7,
    1024, 0, 1, 2,       # GTModelTypeGeoKey = 2 (Geographic)
    1025, 0, 1, 1,       # GTRasterTypeGeoKey
    2048, 0, 1, 4326,    # GeographicTypeGeoKey = EPSG:4326
    2049, 34737, 7, 0,   # GeogCitationGeoKey (ASCII, not inline)
    2054, 0, 1, 9102,    # GeogAngularUnitsGeoKey
    2057, 34736, 1, 1,   # GeogSemiMajorAxisGeoKey (DOUBLE, not inline)
    2059, 34736, 1, 0,   # GeogInvFlatteningGeoKey (DOUBLE, not inline)
)

# Synthetic projected example: UTM zone 48S, EPSG:32748, linear unit metres.
PROJECTED_UTM_GEOKEYS = (
    1, 1, 0, 4,
    1024, 0, 1, 1,       # GTModelTypeGeoKey = 1 (Projected)
    1025, 0, 1, 1,       # GTRasterTypeGeoKey
    3072, 0, 1, 32748,   # ProjectedCSTypeGeoKey = EPSG:32748
    3076, 0, 1, 9001,    # ProjLinearUnitsGeoKey = metre
)

# A second, different projected CRS (UTM zone 49S, EPSG:32749) for mismatch tests.
PROJECTED_UTM49_GEOKEYS = (
    1, 1, 0, 4,
    1024, 0, 1, 1,
    1025, 0, 1, 1,
    3072, 0, 1, 32749,
    3076, 0, 1, 9001,
)


def make_tags(pixel_scale, geokeys=None):
    tags = {33550: pixel_scale}
    if geokeys is not None:
        tags[34735] = geokeys
    return tags


# ── _pixel_area_ha: geographic ───────────────────────────────────────────────

def test_pixel_area_geographic_matches_historical_formula():
    """Geographic CRS: unchanged behaviour from pre-3.6 — degrees * 111,000."""
    px_scale = (0.00026949458523585647, 0.00026949458523585647, 0.0)
    tags = make_tags(px_scale, GEOGRAPHIC_WGS84_GEOKEYS)
    area_ha = _pixel_area_ha(tags)
    expected = (px_scale[0] * 111_000) * (px_scale[1] * 111_000) / 10_000
    assert area_ha == pytest.approx(expected)
    # Sanity: this is the real project raster's pixel — should be ~30m-ish
    assert area_ha == pytest.approx(0.0894, abs=0.001)


# ── _pixel_area_ha: projected (the actual bug fix) ──────────────────────────

def test_pixel_area_projected_uses_metres_directly():
    """
    Projected CRS: pixel scale is already in metres. Before the v3.6 fix,
    this would have gone through the degree conversion and been wrong by
    ~12 orders of magnitude (111,000 squared).
    """
    tags = make_tags((30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS)
    area_ha = _pixel_area_ha(tags)
    assert area_ha == pytest.approx(0.09)   # 30m x 30m = 900 m2 = 0.09 ha

def test_pixel_area_projected_non_square_pixel():
    tags = make_tags((10.0, 20.0, 0.0), PROJECTED_UTM_GEOKEYS)
    area_ha = _pixel_area_ha(tags)
    assert area_ha == pytest.approx(0.02)   # 10 * 20 = 200 m2 = 0.02 ha

def test_pixel_area_projected_warns_on_non_metre_linear_unit(capsys):
    geokeys = (
        1, 1, 0, 4,
        1024, 0, 1, 1,
        1025, 0, 1, 1,
        3072, 0, 1, 2229,
        3076, 0, 1, 9002,   # US survey foot, not metre
    )
    tags = make_tags((98.4, 98.4, 0.0), geokeys)
    _pixel_area_ha(tags)
    captured = capsys.readouterr()
    assert "not metres" in captured.out


# ── _pixel_area_ha: unknown CRS (no GeoKeyDirectoryTag) ─────────────────────

def test_pixel_area_unknown_crs_warns_and_falls_back_to_geographic(capsys):
    tags = make_tags((0.001, 0.001, 0.0))   # no 34735 tag at all
    area_ha = _pixel_area_ha(tags)
    expected = (0.001 * 111_000) * (0.001 * 111_000) / 10_000
    assert area_ha == pytest.approx(expected)
    captured = capsys.readouterr()
    assert "Could not determine raster CRS type" in captured.out


# ── CRSInfo / get_crs_info ───────────────────────────────────────────────────

def test_get_crs_info_geographic():
    info = get_crs_info(make_tags((0.0003, 0.0003, 0.0), GEOGRAPHIC_WGS84_GEOKEYS))
    assert info.known
    assert info.model_type == 2
    assert info.epsg == 4326

def test_get_crs_info_projected():
    info = get_crs_info(make_tags((30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS))
    assert info.known
    assert info.model_type == 1
    assert info.epsg == 32748

def test_get_crs_info_unknown():
    info = get_crs_info(make_tags((0.0003, 0.0003, 0.0)))
    assert not info.known


# ── CRSInfo.compare / assert_crs_consistent ─────────────────────────────────

def test_crs_compare_match_same_epsg():
    a = CRSInfo(model_type=1, epsg=32748, source="tags")
    b = CRSInfo(model_type=1, epsg=32748, source="tags")
    status, _ = a.compare(b)
    assert status == "match"

def test_crs_compare_mismatch_different_epsg():
    a = CRSInfo(model_type=1, epsg=32748, source="tags")
    b = CRSInfo(model_type=1, epsg=32749, source="tags")
    status, reason = a.compare(b)
    assert status == "mismatch"
    assert "32748" in reason and "32749" in reason

def test_crs_compare_mismatch_different_model_type():
    geographic = CRSInfo(model_type=2, epsg=4326, source="tags")
    projected  = CRSInfo(model_type=1, epsg=32748, source="tags")
    status, reason = geographic.compare(projected)
    assert status == "mismatch"

def test_crs_compare_unknown_when_either_side_unresolved():
    known   = CRSInfo(model_type=1, epsg=32748, source="tags")
    unknown = CRSInfo(model_type=None, epsg=None, source="tags")
    status, _ = known.compare(unknown)
    assert status == "unknown"

def test_assert_crs_consistent_raises_on_real_mismatch():
    lulc_crs = get_crs_info(make_tags((30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS))
    age_crs  = get_crs_info(make_tags((30.0, 30.0, 0.0), PROJECTED_UTM49_GEOKEYS))
    with pytest.raises(ValueError, match="CRS mismatch"):
        assert_crs_consistent(lulc_crs, age_crs, "LULC raster", "age raster")

def test_assert_crs_consistent_passes_silently_on_match(capsys):
    lulc_crs = get_crs_info(make_tags((30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS))
    same_crs = get_crs_info(make_tags((30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS))
    assert_crs_consistent(lulc_crs, same_crs, "LULC raster", "age raster")
    captured = capsys.readouterr()
    assert "Warning" not in captured.out

def test_assert_crs_consistent_warns_but_does_not_raise_when_unverifiable(capsys):
    lulc_crs    = get_crs_info(make_tags((30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS))
    unknown_crs = get_crs_info(make_tags((30.0, 30.0, 0.0)))   # no GeoKeys
    assert_crs_consistent(lulc_crs, unknown_crs, "LULC raster", "spatial multiplier 'x.tif'")
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "Could not verify" in captured.out


# ── End-to-end: real GeoTIFF files through read_tiff() ──────────────────────

def _write_geotiff(path, arr, pixel_scale, geokeys=None):
    img = Image.fromarray(arr)
    tiffinfo = {33550: pixel_scale}
    if geokeys is not None:
        tiffinfo[34735] = geokeys
    img.save(str(path), tiffinfo=tiffinfo)

def test_read_tiff_end_to_end_projected(tmp_path):
    p = tmp_path / "utm_raster.tif"
    arr = np.ones((4, 4), dtype=np.float32)
    _write_geotiff(p, arr, (30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS)
    _arr, px_area_ha, tags = read_tiff(p)
    assert px_area_ha == pytest.approx(0.09)
    assert get_crs_info(tags).epsg == 32748

def test_read_tiff_end_to_end_geographic(tmp_path):
    p = tmp_path / "wgs84_raster.tif"
    arr = np.ones((4, 4), dtype=np.float32)
    px_scale = (0.00026949458523585647, 0.00026949458523585647, 0.0)
    _write_geotiff(p, arr, px_scale, GEOGRAPHIC_WGS84_GEOKEYS)
    _arr, px_area_ha, tags = read_tiff(p)
    assert px_area_ha == pytest.approx(0.0894, abs=0.001)
    assert get_crs_info(tags).epsg == 4326


# ── build_initial_age_from_raster() — returns (arr, CRSInfo) as of v3.6 ─────

def test_build_initial_age_from_raster_returns_crs_info(tmp_path):
    from strategicc.core.age import build_initial_age_from_raster
    p = tmp_path / "age.tif"
    arr = np.full((4, 4), 5, dtype=np.uint16)
    _write_geotiff(p, arr, (30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS)
    age_arr, crs_info = build_initial_age_from_raster(str(p))
    assert age_arr.shape == (4, 4)
    assert crs_info.known
    assert crs_info.epsg == 32748


# ── load_spatial_multipliers() — reference_crs enforcement as of v3.6 ───────

def test_load_spatial_multipliers_blocks_on_crs_mismatch(tmp_path):
    from strategicc.core.spatial import load_spatial_multipliers
    from strategicc.io.csv_loader import SpatialMultEntry

    mult_dir = tmp_path
    p = mult_dir / "mangrove_mult.tif"
    arr = np.ones((4, 4), dtype=np.float32)
    _write_geotiff(p, arr, (30.0, 30.0, 0.0), PROJECTED_UTM49_GEOKEYS)  # different EPSG

    lulc_crs = get_crs_info(make_tags((30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS))
    entries = [SpatialMultEntry(group="Mangrove_recruitment", filename="mangrove_mult.tif")]

    with pytest.raises(ValueError, match="CRS mismatch"):
        load_spatial_multipliers(entries, mult_dir, (4, 4), reference_crs=lulc_crs)

def test_load_spatial_multipliers_passes_on_matching_crs(tmp_path):
    from strategicc.core.spatial import load_spatial_multipliers
    from strategicc.io.csv_loader import SpatialMultEntry

    mult_dir = tmp_path
    p = mult_dir / "mangrove_mult.tif"
    arr = np.ones((4, 4), dtype=np.float32)
    _write_geotiff(p, arr, (30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS)  # same EPSG

    lulc_crs = get_crs_info(make_tags((30.0, 30.0, 0.0), PROJECTED_UTM_GEOKEYS))
    entries = [SpatialMultEntry(group="Mangrove_recruitment", filename="mangrove_mult.tif")]

    result = load_spatial_multipliers(entries, mult_dir, (4, 4), reference_crs=lulc_crs)
    assert "Mangrove_recruitment" in result

def test_load_spatial_multipliers_skips_check_when_reference_crs_none(tmp_path):
    """Default (reference_crs=None) preserves pre-3.6 behaviour — no check."""
    from strategicc.core.spatial import load_spatial_multipliers
    from strategicc.io.csv_loader import SpatialMultEntry

    mult_dir = tmp_path
    p = mult_dir / "mangrove_mult.tif"
    arr = np.ones((4, 4), dtype=np.float32)
    _write_geotiff(p, arr, (30.0, 30.0, 0.0), PROJECTED_UTM49_GEOKEYS)

    entries = [SpatialMultEntry(group="Mangrove_recruitment", filename="mangrove_mult.tif")]
    result = load_spatial_multipliers(entries, mult_dir, (4, 4))   # no reference_crs
    assert "Mangrove_recruitment" in result
