"""
tests/test_core_age.py

Unit tests for strategicc/core/age.py — previously at 52% coverage.
Covers build_initial_age_from_raster() (Pillow path + rasterio fallback),
build_initial_age_from_rules() (including the unknown-class warning and
zero-SD branches), update_age(), age_gate_mask(), and save_age_tif().
"""

import numpy as np
import pytest

from strategicc.core.age import (
    build_initial_age_from_raster,
    build_initial_age_from_rules,
    update_age,
    age_gate_mask,
    save_age_tif,
)
from strategicc.io.csv_loader import StateClass, InitialAgeRule


@pytest.fixture
def classes():
    return {
        1: StateClass(id=1, name="Mangrove", full_name="Mangrove:All", color=(255, 0, 128, 0)),
        2: StateClass(id=2, name="Water_body", full_name="Water_body:All", color=(255, 0, 0, 255)),
    }


# ── build_initial_age_from_raster ───────────────────────────────────────────

def test_build_initial_age_from_raster_pillow_path(tmp_path):
    from PIL import Image
    arr = np.full((4, 4), 7, dtype=np.uint16)
    path = tmp_path / "age.tif"
    Image.fromarray(arr, mode="I;16").save(str(path))

    loaded, crs_info = build_initial_age_from_raster(str(path))
    assert loaded.dtype == np.uint16
    assert np.all(loaded == 7)


def test_build_initial_age_from_raster_rasterio_fallback(tmp_path, monkeypatch):
    """When Pillow raises the known 'unknown raw mode' ValueError, fall
    back to rasterio."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "age_rasterio.tif"
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    with rasterio.open(str(path), "w", driver="GTiff", dtype="uint16", count=1,
                        height=4, width=4, crs="EPSG:4326", transform=transform) as dst:
        dst.write(np.full((4, 4), 9, dtype=np.uint16), 1)

    import strategicc.core.age as age_mod

    class _FakeImage:
        @staticmethod
        def open(p):
            raise ValueError("unknown raw mode for given image mode")

    monkeypatch.setattr(age_mod, "Image", _FakeImage, raising=False)
    # build_initial_age_from_raster imports PIL.Image locally, so patch the
    # actual import target instead.
    import PIL.Image as real_pil_image
    def _raise_unknown_raw_mode(p):
        raise ValueError("unknown raw mode for given image mode")
    monkeypatch.setattr(real_pil_image, "open", _raise_unknown_raw_mode)

    loaded, crs_info = build_initial_age_from_raster(str(path))
    assert np.all(loaded == 9)
    assert crs_info is not None


def test_build_initial_age_from_raster_reraises_other_valueerror(tmp_path, monkeypatch):
    import PIL.Image as real_pil_image

    def _raise_other(p):
        raise ValueError("some other unrelated error")

    monkeypatch.setattr(real_pil_image, "open", _raise_other)

    with pytest.raises(ValueError, match="some other unrelated error"):
        build_initial_age_from_raster(str(tmp_path / "whatever.tif"))


# ── build_initial_age_from_rules ────────────────────────────────────────────

def test_build_initial_age_from_rules_fixed_age_zero_sd(classes):
    lulc = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    rules = [InitialAgeRule(state_class="Mangrove", age_mean=15, age_sd=0.0)]
    rng = np.random.default_rng(0)

    age_map = build_initial_age_from_rules(lulc, classes, rules, rng)
    assert np.all(age_map[lulc == 1] == 15)
    assert np.all(age_map[lulc == 2] == 0)   # unmatched class defaults to 0


def test_build_initial_age_from_rules_truncated_normal(classes):
    lulc = np.ones((20, 20), dtype=np.uint8)
    rules = [InitialAgeRule(state_class="Mangrove", age_mean=20, age_sd=5, age_min=5, age_max=50)]
    rng = np.random.default_rng(1)

    age_map = build_initial_age_from_rules(lulc, classes, rules, rng)
    assert age_map.min() >= 5
    assert age_map.max() <= 50
    assert age_map.dtype == np.uint16


def test_build_initial_age_from_rules_matches_full_name(classes):
    """Rule referencing the full ':All' name must also resolve."""
    lulc = np.ones((3, 3), dtype=np.uint8)
    rules = [InitialAgeRule(state_class="Mangrove:All", age_mean=8, age_sd=0.0)]
    rng = np.random.default_rng(0)
    age_map = build_initial_age_from_rules(lulc, classes, rules, rng)
    assert np.all(age_map == 8)


def test_build_initial_age_from_rules_unknown_class_warns(classes, capsys):
    lulc = np.ones((3, 3), dtype=np.uint8)
    rules = [InitialAgeRule(state_class="NotARealClass", age_mean=8, age_sd=0.0)]
    rng = np.random.default_rng(0)
    age_map = build_initial_age_from_rules(lulc, classes, rules, rng)
    assert np.all(age_map == 0)
    assert "unknown class" in capsys.readouterr().out


def test_build_initial_age_from_rules_zero_matching_cells_skipped(classes):
    """A rule for a class with zero cells in lulc_map should be skipped
    without error."""
    lulc = np.ones((3, 3), dtype=np.uint8)   # all class 1
    rules = [InitialAgeRule(state_class="Water_body", age_mean=8, age_sd=0.0)]
    rng = np.random.default_rng(0)
    age_map = build_initial_age_from_rules(lulc, classes, rules, rng)
    assert np.all(age_map == 0)


# ── update_age ───────────────────────────────────────────────────────────────

def test_update_age_increments_non_transitioned_cells():
    age_map = np.array([[5, 5], [5, 5]], dtype=np.uint16)
    fired = np.zeros((2, 2), dtype=bool)
    reset = np.zeros((2, 2), dtype=bool)
    relative = np.full((2, 2), -1, dtype=np.int16)

    new_age = update_age(age_map, fired, reset, relative)
    assert np.all(new_age == 6)


def test_update_age_resets_on_fire():
    age_map = np.array([[5, 5]], dtype=np.uint16)
    fired = np.array([[True, False]])
    reset = np.array([[True, False]])
    relative = np.full((1, 2), -1, dtype=np.int16)

    new_age = update_age(age_map, fired, reset, relative)
    assert new_age[0, 0] == 0
    assert new_age[0, 1] == 6


def test_update_age_relative_overrides_reset():
    age_map = np.array([[5]], dtype=np.uint16)
    fired = np.array([[True]])
    reset = np.array([[True]])
    relative = np.array([[42]], dtype=np.int16)

    new_age = update_age(age_map, fired, reset, relative)
    assert new_age[0, 0] == 42


def test_update_age_clips_at_max_uint16():
    age_map = np.array([[65535]], dtype=np.uint16)
    fired = np.array([[False]])
    reset = np.array([[False]])
    relative = np.full((1, 1), -1, dtype=np.int16)

    new_age = update_age(age_map, fired, reset, relative)
    assert new_age[0, 0] == 65535   # clipped, not wrapped


# ── age_gate_mask ────────────────────────────────────────────────────────────

def test_age_gate_mask_both_bounds():
    age_map = np.array([[1, 5, 10, 20]])
    mask = age_gate_mask(age_map, age_min=5, age_max=10)
    assert mask.tolist() == [[False, True, True, False]]


def test_age_gate_mask_no_bounds():
    age_map = np.array([[1, 5, 10]])
    mask = age_gate_mask(age_map, age_min=None, age_max=None)
    assert mask.all()


def test_age_gate_mask_only_min():
    age_map = np.array([[1, 5, 10]])
    mask = age_gate_mask(age_map, age_min=5, age_max=None)
    assert mask.tolist() == [[False, True, True]]


def test_age_gate_mask_only_max():
    age_map = np.array([[1, 5, 10]])
    mask = age_gate_mask(age_map, age_min=None, age_max=5)
    assert mask.tolist() == [[True, True, False]]


# ── save_age_tif ─────────────────────────────────────────────────────────────

def test_save_age_tif_creates_file(tmp_path):
    age_map = np.full((4, 4), 3, dtype=np.uint16)
    save_age_tif(age_map, year=2022, out_dir=tmp_path, src_tags={})
    out_path = tmp_path / "age_2022.tif"
    assert out_path.exists()

    from PIL import Image
    loaded = np.array(Image.open(str(out_path)))
    assert np.all(loaded == 3)


def test_save_age_tif_with_src_tags(tmp_path):
    """src_tags carrying real GeoTIFF georeferencing tags (as loaded from an
    actual raster) should be forwarded to the output file without error."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from PIL import Image

    src_path = tmp_path / "source.tif"
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    with rasterio.open(str(src_path), "w", driver="GTiff", dtype="uint8", count=1,
                        height=3, width=3, crs="EPSG:4326", transform=transform) as dst:
        dst.write(np.ones((3, 3), dtype=np.uint8), 1)

    src_tags = dict(Image.open(str(src_path)).tag_v2)

    age_map = np.full((3, 3), 1, dtype=np.uint16)
    save_age_tif(age_map, year=2023, out_dir=tmp_path, src_tags=src_tags)
    assert (tmp_path / "age_2023.tif").exists()
