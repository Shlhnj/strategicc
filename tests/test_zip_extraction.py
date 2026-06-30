"""
tests/test_zip_extraction.py  —  v3.4
Unit tests for:
  - extract_initial_state_class / extract_lulc_zip_to_folder (calibration)
  - resolve_mult_dir (io/raster.py) — spatial multiplier zip support
"""

import pytest
import numpy as np
import zipfile
from pathlib import Path

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from strategicc.calibration import (
    extract_initial_state_class, extract_lulc_zip_to_folder,
)
from strategicc.io.raster import resolve_mult_dir


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def lulc_zip(tmp_path):
    rows, cols = 8, 8
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}

    src_dir = tmp_path / "src_years"
    src_dir.mkdir()
    for year in [2020, 2021, 2022]:
        arr = np.full((rows, cols), year - 2019, dtype=np.uint8)
        with rasterio.open(str(src_dir / f"{year}.tif"), "w", **profile) as dst:
            dst.write(arr, 1)

    zip_path = tmp_path / "lulc_history.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for year in [2020, 2021, 2022]:
            zf.write(src_dir / f"{year}.tif", arcname=f"{year}.tif")

    return zip_path


# ── extract_lulc_zip_to_folder ────────────────────────────────────────────────

def test_extract_zip_extracts_all_years(tmp_path, lulc_zip):
    extract_dir = tmp_path / "lulc_annual"
    result = extract_lulc_zip_to_folder(lulc_zip, extract_dir)
    assert len(result) == 3
    assert set(result.keys()) == {2020, 2021, 2022}
    tifs = list(extract_dir.glob("*.tif"))
    assert len(tifs) == 3

def test_extract_zip_cache_hit_skips_reextraction(tmp_path, lulc_zip, capsys):
    extract_dir = tmp_path / "lulc_annual"
    extract_lulc_zip_to_folder(lulc_zip, extract_dir)
    capsys.readouterr()

    extract_lulc_zip_to_folder(lulc_zip, extract_dir)
    captured = capsys.readouterr()
    assert "Cache hit" in captured.out

def test_extract_zip_force_reextracts(tmp_path, lulc_zip, capsys):
    extract_dir = tmp_path / "lulc_annual"
    extract_lulc_zip_to_folder(lulc_zip, extract_dir)
    capsys.readouterr()

    extract_lulc_zip_to_folder(lulc_zip, extract_dir, force=True)
    captured = capsys.readouterr()
    assert "Cache hit" not in captured.out


# ── extract_initial_state_class ──────────────────────────────────────────────

def test_extract_initial_state_class_selects_correct_year(tmp_path, lulc_zip):
    extract_dir = tmp_path / "lulc_annual"
    selected = extract_initial_state_class(lulc_zip, year=2021, extract_dir=extract_dir)
    with rasterio.open(str(selected)) as src:
        data = src.read(1)
    assert np.all(data == 2)

def test_extract_initial_state_class_persists_all_years(tmp_path, lulc_zip):
    """Regression test: selecting one year must still leave ALL years on
    disk in the persistent folder, not just the selected one."""
    extract_dir = tmp_path / "lulc_annual"
    extract_initial_state_class(lulc_zip, year=2021, extract_dir=extract_dir)
    all_tifs = list(extract_dir.glob("*.tif"))
    assert len(all_tifs) == 3

def test_extract_initial_state_class_different_year_no_reextraction(tmp_path, lulc_zip, capsys):
    extract_dir = tmp_path / "lulc_annual"
    extract_initial_state_class(lulc_zip, year=2020, extract_dir=extract_dir)
    capsys.readouterr()

    selected = extract_initial_state_class(lulc_zip, year=2022, extract_dir=extract_dir)
    captured = capsys.readouterr()
    assert "Cache hit" in captured.out
    with rasterio.open(str(selected)) as src:
        data = src.read(1)
    assert np.all(data == 3)

def test_extract_initial_state_class_unknown_year_raises(tmp_path, lulc_zip):
    extract_dir = tmp_path / "lulc_annual"
    with pytest.raises(ValueError, match="not found"):
        extract_initial_state_class(lulc_zip, year=1999, extract_dir=extract_dir)


# ── resolve_mult_dir ──────────────────────────────────────────────────────────

def test_resolve_mult_dir_folder_passthrough(tmp_path):
    folder = tmp_path / "spatmult_uploads"
    folder.mkdir()
    (folder / "test.tif").write_text("dummy")
    result = resolve_mult_dir(folder)
    assert result == folder

def test_resolve_mult_dir_extracts_zip(tmp_path):
    zip_path = tmp_path / "spatmult_uploads.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("mult1.tif", "dummy1")
        zf.writestr("mult2.tif", "dummy2")

    result = resolve_mult_dir(zip_path)
    assert result.is_dir()
    assert result == zip_path.with_suffix("")
    files = list(result.glob("*.tif"))
    assert len(files) == 2

def test_resolve_mult_dir_cache_hit(tmp_path, capsys):
    zip_path = tmp_path / "spatmult_uploads.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("mult1.tif", "dummy1")

    resolve_mult_dir(zip_path)
    capsys.readouterr()

    resolve_mult_dir(zip_path)
    captured = capsys.readouterr()
    assert "Cache hit" in captured.out

def test_resolve_mult_dir_nonexistent_path_returned_unchanged(tmp_path):
    missing = tmp_path / "does_not_exist"
    result = resolve_mult_dir(missing)
    assert result == missing
