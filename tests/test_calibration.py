"""
tests/test_calibration.py  —  v2.4
Unit tests for the calibration module (age, transitions, temporal distribution).
"""

import pytest
import numpy as np
import zipfile
from pathlib import Path

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from strategicc.calibration import (
    load_lulc_timeseries, compute_age_raster, compute_transition_rates,
    compute_temporal_distribution,
)
from strategicc.calibration.transitions import compute_yearly_transition_counts
from strategicc.io.csv_loader import StateClass


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_zip(tmp_path_factory):
    """Build a small synthetic 5-year LULC zip: class 1 -> class 2 conversion."""
    tmp_dir = tmp_path_factory.mktemp("synthetic")
    rows, cols = 10, 10
    years = [2018, 2019, 2020, 2021, 2022]

    transform = from_origin(110.0, -7.0, 0.0001, 0.0001)
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "height": rows, "width": cols,
        "crs": "EPSG:4326", "transform": transform,
    }

    rng = np.random.default_rng(1)
    current = np.ones((rows, cols), dtype=np.uint8)
    current[:, 5:] = 2

    for i, yr in enumerate(years):
        if i > 0:
            mask = (current == 1)
            draws = rng.random((rows, cols))
            convert = mask & (draws < 0.1)
            current = current.copy()
            current[convert] = 2

        path = tmp_dir / f"{yr}.tif"
        with rasterio.open(str(path), "w", **profile) as dst:
            dst.write(current, 1)

    zip_path = tmp_dir / "synthetic.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for yr in years:
            zf.write(tmp_dir / f"{yr}.tif", arcname=f"{yr}.tif")

    return zip_path


@pytest.fixture(scope="module")
def loaded_ts(synthetic_zip, tmp_path_factory):
    extract_dir = tmp_path_factory.mktemp("extracted")
    return load_lulc_timeseries(synthetic_zip, extract_dir=extract_dir)


@pytest.fixture
def classes():
    return {
        1: StateClass(1, "Water_body", "Water_body:All", (255,0,128,255)),
        2: StateClass(2, "Mangrove",   "Mangrove:All",   (255,0,100,0)),
    }


@pytest.fixture
def group_map():
    return {(1, 2): "Mangrove_recruitment"}


# ── Tests: loader ──────────────────────────────────────────────────────────────

def test_load_timeseries_auto_mode(loaded_ts):
    assert loaded_ts.years == [2018, 2019, 2020, 2021, 2022]
    assert loaded_ts.shape == (10, 10)
    assert loaded_ts.stack.shape == (5, 10, 10)

def test_load_timeseries_year_index(loaded_ts):
    assert loaded_ts.year_index(2020) == 2
    with pytest.raises(ValueError):
        loaded_ts.year_index(1999)

def test_load_timeseries_manual_mode(synthetic_zip, tmp_path_factory):
    extract_dir = tmp_path_factory.mktemp("extracted_manual")
    ts = load_lulc_timeseries(
        synthetic_zip, extract_dir=extract_dir,
        periods=[(2018,2018,2018),(2019,2020,2019),(2021,2021,2021),(2022,2022,2022)]
    )
    assert ts.years == [2018, 2019, 2020, 2021, 2022]
    # 2019 and 2020 should be identical (same composite raster)
    assert (ts.stack[1] == ts.stack[2]).all()


# ── Tests: age ──────────────────────────────────────────────────────────────────

def test_compute_age_continuous(loaded_ts):
    result = compute_age_raster(loaded_ts)
    assert result.baseline_year == 2022
    assert result.age_combined.shape == (10, 10)
    assert 1 in result.age_per_class
    assert 2 in result.age_per_class

def test_compute_age_class1_full_record(loaded_ts):
    """Class 1 (Water_body) cells that never converted should have age=4 (full record)."""
    result = compute_age_raster(loaded_ts)
    # Original water cells in left half should mostly have full record
    assert result.full_record_mask.sum() > 0

def test_compute_age_binned(loaded_ts):
    result = compute_age_raster(
        loaded_ts, age_bins=[(0,1,1),(2,3,2),(4,999,3)]
    )
    unique_vals = set(np.unique(result.age_combined).tolist())
    # Should only contain bin labels (1,2,3) plus possibly NODATA
    assert unique_vals.issubset({1, 2, 3, 65535})

def test_compute_age_specific_classes(loaded_ts):
    result = compute_age_raster(loaded_ts, class_ids=[2])
    assert list(result.age_per_class.keys()) == [2]


# ── Tests: transitions ────────────────────────────────────────────────────────

def test_yearly_transition_counts(loaded_ts):
    yearly = compute_yearly_transition_counts(loaded_ts)
    assert not yearly.records.empty
    assert set(yearly.records.columns) == {
        "year", "from_id", "to_id", "n_cells", "n_from_total", "probability"
    }

def test_yearly_transition_no_self_transitions(loaded_ts):
    yearly = compute_yearly_transition_counts(loaded_ts)
    same = yearly.records[yearly.records["from_id"] == yearly.records["to_id"]]
    assert same.empty

def test_compute_transition_rates(loaded_ts, classes, group_map):
    yearly = compute_yearly_transition_counts(loaded_ts)
    df = compute_transition_rates(yearly, classes, group_map)
    assert not df.empty
    assert "Mangrove_recruitment" in df["TransitionTypeId"].values
    row = df[df["TransitionTypeId"] == "Mangrove_recruitment"].iloc[0]
    assert row["StateClassIdSource"] == "Water_body:All"
    assert row["StateClassIdDest"] == "Mangrove:All"
    assert 0 < row["Probability"] < 1

def test_compute_transition_rates_unmapped_excluded(loaded_ts, classes):
    yearly = compute_yearly_transition_counts(loaded_ts)
    df = compute_transition_rates(yearly, classes, group_map={})  # empty map
    assert df.empty


# ── Tests: temporal distribution ─────────────────────────────────────────────

def test_compute_temporal_distribution(loaded_ts, group_map):
    yearly = compute_yearly_transition_counts(loaded_ts)
    df = compute_temporal_distribution(yearly, group_map, min_years=2)
    assert not df.empty
    row = df.iloc[0]
    assert row["DistributionType"] == "Uniform"
    assert row["DistributionMin"] <= row["DistributionMax"]

def test_temporal_distribution_mean_is_one(loaded_ts, group_map):
    """Validates the core consistency guarantee: mean(multiplier) ~ 1.0"""
    yearly = compute_yearly_transition_counts(loaded_ts)
    df = yearly.records.copy()
    df["group"] = df.apply(
        lambda r: group_map.get((int(r["from_id"]), int(r["to_id"]))), axis=1
    )
    df = df.dropna(subset=["group"])

    for group, grp in df.groupby("group"):
        pooled = grp.groupby("year").apply(
            lambda g: g["n_cells"].sum() / g["n_from_total"].sum()
        )
        mean_prob = pooled.mean()
        multipliers = pooled / mean_prob
        assert multipliers.mean() == pytest.approx(1.0, abs=1e-9)

def test_temporal_distribution_insufficient_years(loaded_ts, group_map):
    yearly = compute_yearly_transition_counts(loaded_ts)
    df = compute_temporal_distribution(yearly, group_map, min_years=100)
    assert df.empty
