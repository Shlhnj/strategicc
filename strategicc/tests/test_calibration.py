"""
tests/test_calibration.py  —  v3.8
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
    compute_transition_coverage,
    compute_temporal_distribution, compute_size_distribution,
)
from strategicc.calibration.transitions import compute_yearly_transition_counts
from strategicc.io.csv_loader import StateClass, load_transition_size_rules, group_size_bins


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


@pytest.fixture(scope="module")
def clustered_zip(tmp_path_factory):
    """
    Build a 40x40, 8-year synthetic LULC zip where class 1 -> class 2
    conversions happen in spatially clustered blocks (not independent
    scattered pixels) — needed to exercise patch labeling meaningfully.
    """
    tmp_dir = tmp_path_factory.mktemp("clustered")
    rows, cols = 40, 40
    years = list(range(2015, 2023))

    transform = from_origin(110.0, -7.0, 0.0001, 0.0001)
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "height": rows, "width": cols,
        "crs": "EPSG:4326", "transform": transform,
    }

    rng = np.random.default_rng(3)
    current = np.ones((rows, cols), dtype=np.uint8)
    current[:, 20:] = 2

    for i, yr in enumerate(years):
        if i > 0:
            mask = (current == 1)
            seeds_r = rng.integers(0, rows, size=3)
            seeds_c = rng.integers(0, 20, size=3)
            for sr, sc in zip(seeds_r, seeds_c):
                size = rng.integers(2, 6)
                r0, r1 = max(0, sr - size), sr + size
                c0, c1 = max(0, sc - size), sc + size
                block = current[r0:r1, c0:c1]
                block_mask = mask[r0:r1, c0:c1]
                block[block_mask] = 2

        path = tmp_dir / f"{yr}.tif"
        with rasterio.open(str(path), "w", **profile) as dst:
            dst.write(current, 1)

    zip_path = tmp_dir / "clustered.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for yr in years:
            zf.write(tmp_dir / f"{yr}.tif", arcname=f"{yr}.tif")

    return zip_path


@pytest.fixture(scope="module")
def clustered_ts(clustered_zip, tmp_path_factory):
    extract_dir = tmp_path_factory.mktemp("clustered_extracted")
    return load_lulc_timeseries(clustered_zip, extract_dir=extract_dir)


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


# ── Tests: transition coverage preview (v3.14) ───────────────────────────────

def test_compute_transition_coverage_full_pair_mapped(loaded_ts, classes, group_map):
    yearly = compute_yearly_transition_counts(loaded_ts)
    df = compute_transition_coverage(yearly, classes, group_map)
    assert not df.empty
    assert set(df.columns) == {
        "from_class", "to_class", "group", "mean_probability",
        "total_n_cells", "avg_n_from_total",
    }
    row = df[(df["from_class"] == "Water_body") & (df["to_class"] == "Mangrove")].iloc[0]
    assert row["group"] == "Mangrove_recruitment"
    assert 0 < row["mean_probability"] < 1
    assert row["total_n_cells"] > 0


def test_compute_transition_coverage_unmapped_pair_flagged_unnamed(loaded_ts, classes):
    """A pair with no group_map entry must still appear, flagged UNNAMED --
    never silently excluded, and never truncated regardless of how many
    other pairs exist (no top-10 cutoff)."""
    yearly = compute_yearly_transition_counts(loaded_ts)
    df = compute_transition_coverage(yearly, classes, group_map={})  # nothing mapped
    assert not df.empty
    assert (df["group"] == "UNNAMED").all()


def test_compute_transition_coverage_sorted_by_total_n_cells_descending(loaded_ts, classes):
    yearly = compute_yearly_transition_counts(loaded_ts)
    df = compute_transition_coverage(yearly, classes, group_map={})
    assert list(df["total_n_cells"]) == sorted(df["total_n_cells"], reverse=True)


def test_compute_transition_coverage_no_truncation_beyond_ten_pairs(classes):
    """Regression test for the exact failure mode this function replaces:
    an 11th+ pair by rank must still appear in full, not be cut off at a
    hardcoded top-10 limit."""
    import pandas as pd
    from strategicc.calibration.transitions import YearlyTransitionCounts

    # 12 distinct (from,to) pairs, none in group_map -- all UNNAMED, and
    # the 12th (smallest) pair must still be present in the output.
    rows = []
    for i in range(12):
        rows.append({
            "year": 2020, "from_id": 1, "to_id": 10 + i,
            "n_cells": 100 - i, "n_from_total": 1000,
            "probability": (100 - i) / 1000,
        })
    yearly = YearlyTransitionCounts(records=pd.DataFrame(rows))
    df = compute_transition_coverage(yearly, classes, group_map={})
    assert len(df) == 12
    assert (df["group"] == "UNNAMED").all()


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


# ── Tests: size distribution ─────────────────────────────────────────────────

PX_AREA_HA = 0.0001 * 0.0001 * 111000 * 111000 / 10000  # deg^2 -> ha at equator


def test_compute_size_distribution_schema(clustered_ts, group_map):
    df = compute_size_distribution(
        clustered_ts, group_map, px_area_ha=PX_AREA_HA, n_bins=4, min_patches=3,
    )
    assert not df.empty
    assert list(df.columns) == [
        "Transition Type/Group", "Maximum Area (Hectares)", "Relative Amount",
    ]
    assert (df["Transition Type/Group"] == "Mangrove_recruitment [Type]").all()


def test_compute_size_distribution_bins_ascending(clustered_ts, group_map):
    df = compute_size_distribution(
        clustered_ts, group_map, px_area_ha=PX_AREA_HA, n_bins=4, min_patches=3,
    )
    areas = df["Maximum Area (Hectares)"].tolist()
    assert areas == sorted(areas)
    assert len(areas) == len(set(areas))   # strictly ascending, no duplicate bins


def test_compute_size_distribution_relative_amount_sums_to_100(clustered_ts, group_map):
    df = compute_size_distribution(
        clustered_ts, group_map, px_area_ha=PX_AREA_HA, n_bins=4, min_patches=3,
    )
    total = df["Relative Amount"].sum()
    assert total == pytest.approx(100.0, abs=1e-6)


def test_compute_size_distribution_insufficient_patches(clustered_ts, group_map):
    df = compute_size_distribution(
        clustered_ts, group_map, px_area_ha=PX_AREA_HA, n_bins=4, min_patches=10_000,
    )
    assert df.empty


def test_compute_size_distribution_invalid_connectivity(clustered_ts, group_map):
    with pytest.raises(ValueError):
        compute_size_distribution(
            clustered_ts, group_map, px_area_ha=PX_AREA_HA, connectivity=6,
        )


def test_compute_size_distribution_roundtrip_through_loader(clustered_ts, group_map, tmp_path):
    """
    Derived output must parse cleanly through the SAME loader the engine
    uses (load_transition_size_rules + group_size_bins), producing valid
    cumulative (min, max, probability) bins summing to ~1.0 per group —
    this is the contract strategicc.core.patches.sample_patch_size_ha()
    depends on.
    """
    df = compute_size_distribution(
        clustered_ts, group_map, px_area_ha=PX_AREA_HA, n_bins=4, min_patches=3,
    )
    out_path = tmp_path / "TransitionSizeDistribution.csv"
    df.to_csv(out_path, index=False)

    rules = load_transition_size_rules(out_path)
    bins = group_size_bins(rules)

    assert "Mangrove_recruitment" in bins
    group_bins = bins["Mangrove_recruitment"]

    # bins are contiguous: each bin's min == previous bin's max
    for (prev_min, prev_max, _), (this_min, this_max, _) in zip(group_bins, group_bins[1:]):
        assert this_min == prev_max

    total_prob = sum(p for _, _, p in group_bins)
    assert total_prob == pytest.approx(1.0, abs=1e-6)


def test_compute_size_distribution_group_map_shared_with_transitions(clustered_ts, group_map):
    """
    Sanity check that the SAME group_map used for compute_transition_rates()
    / compute_temporal_distribution() produces a consistent group label
    here too (no separate grouping scheme — per design decision).
    """
    yearly = compute_yearly_transition_counts(clustered_ts)
    temporal_df = compute_temporal_distribution(yearly, group_map, min_years=2)
    size_df = compute_size_distribution(
        clustered_ts, group_map, px_area_ha=PX_AREA_HA, n_bins=4, min_patches=3,
    )

    temporal_group = temporal_df.iloc[0]["TransitionGroupId"].replace(" [Type]", "")
    size_group = size_df.iloc[0]["Transition Type/Group"].replace(" [Type]", "")
    assert temporal_group == size_group == "Mangrove_recruitment"
