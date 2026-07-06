"""
tests/test_validation.py  —  v3.12
Unit tests for strategicc.validation (extent comparison, spatial agreement,
attribute drift, multiplier correction).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from strategicc.io.csv_loader import StateClass
from strategicc.calibration.loader import LULCTimeSeries
from strategicc.validation.extent import (
    compute_observed_extent,
    compare_extent_trajectories,
    spatial_agreement,
    attribute_extent_drift,
)
from strategicc.validation.correction import (
    compute_pathway_rate_ratios,
    correct_multipliers,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classes():
    return {
        1: StateClass(id=1, name="Mangrove", full_name="Mangrove:All", color=(255, 0, 128, 0)),
        2: StateClass(id=2, name="Water_body", full_name="Water_body:All", color=(255, 0, 0, 255)),
    }


@pytest.fixture
def synthetic_ts():
    """3-year synthetic stack: 10x10, half class 1 half class 2, class 1
    steadily converting to class 2 (mimics an inundation trend)."""
    rows, cols = 10, 10
    stack = []
    current = np.ones((rows, cols), dtype=np.uint8)
    current[:, 5:] = 2

    for i in range(3):
        if i > 0:
            current = current.copy()
            current[i, :5] = 2   # convert one row of class 1 -> class 2 per year
        stack.append(current.copy())

    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": rows, "width": cols}
    return LULCTimeSeries(stack=np.array(stack), years=[2020, 2021, 2022], profile=profile)


# ── compute_observed_extent ───────────────────────────────────────────────────

def test_compute_observed_extent_basic(synthetic_ts, classes):
    df = compute_observed_extent(synthetic_ts, classes, px_area_ha=1.0)
    assert set(df.columns) == {"year", "class_name", "area_ha"}
    assert set(df["year"].unique()) == {2020, 2021, 2022}
    # Mangrove area should shrink year over year as it converts to Water_body
    mangrove = df[df["class_name"] == "Mangrove"].set_index("year")["area_ha"]
    assert mangrove[2020] > mangrove[2021] > mangrove[2022]


def test_compute_observed_extent_cache_roundtrip(synthetic_ts, classes, tmp_path):
    cache_path = tmp_path / "ObservedExtent.csv"
    df1 = compute_observed_extent(synthetic_ts, classes, px_area_ha=2.0, cache_path=cache_path)
    assert cache_path.exists()

    # Mutate the underlying ts -- cache hit should return the OLD cached values,
    # not recompute, unless force_recompute=True.
    mutated_ts = LULCTimeSeries(
        stack=np.zeros_like(synthetic_ts.stack), years=synthetic_ts.years, profile=synthetic_ts.profile
    )
    df2 = compute_observed_extent(mutated_ts, classes, px_area_ha=2.0, cache_path=cache_path)
    pd.testing.assert_frame_equal(df1, df2)

    df3 = compute_observed_extent(
        mutated_ts, classes, px_area_ha=2.0, cache_path=cache_path, force_recompute=True
    )
    assert df3.empty or not df3["area_ha"].equals(df1["area_ha"])


# ── compare_extent_trajectories ───────────────────────────────────────────────

def test_compare_extent_trajectories():
    observed = pd.DataFrame({
        "year":       [2020, 2021],
        "class_name": ["Water_body", "Water_body"],
        "area_ha":    [50.0, 55.0],
    })
    simulated = pd.DataFrame({
        "year":       [2020, 2021],
        "class_name": ["Water_body", "Water_body"],
        "area_ha":    [50.0, 90.0],
    })
    result = compare_extent_trajectories(observed, simulated)
    row_2021 = result[result["year"] == 2021].iloc[0]
    assert row_2021["abs_diff"] == pytest.approx(35.0)
    assert row_2021["pct_diff"] == pytest.approx(100.0 * 35.0 / 55.0)


def test_compare_extent_trajectories_zero_observed_no_zerodiv():
    observed = pd.DataFrame({"year": [2020], "class_name": ["X"], "area_ha": [0.0]})
    simulated = pd.DataFrame({"year": [2020], "class_name": ["X"], "area_ha": [10.0]})
    result = compare_extent_trajectories(observed, simulated)
    assert np.isnan(result.iloc[0]["pct_diff"])


# ── spatial_agreement ─────────────────────────────────────────────────────────

def test_spatial_agreement_perfect_match(classes):
    raster = np.array([[1, 1, 2], [2, 2, 1]], dtype=np.uint8)
    result = spatial_agreement(raster, raster.copy(), classes)
    assert result["figure_of_merit"] == pytest.approx(100.0)
    assert result["quantity_disagreement"] == pytest.approx(0.0)
    assert result["allocation_disagreement"] == pytest.approx(0.0)
    assert result["kappa"] == pytest.approx(1.0)


def test_spatial_agreement_quantity_disagreement(classes):
    # obs: 3x class1, 3x class2 (balanced). sim: 6x class2 (all converted --
    # pure quantity disagreement, since sim has MORE class2 than obs everywhere)
    obs = np.array([1, 1, 1, 2, 2, 2], dtype=np.uint8).reshape(2, 3)
    sim = np.array([2, 2, 2, 2, 2, 2], dtype=np.uint8).reshape(2, 3)
    result = spatial_agreement(sim, obs, classes)
    assert result["quantity_disagreement"] > 0
    assert result["figure_of_merit"] < 100.0


def test_spatial_agreement_shape_mismatch_raises(classes):
    with pytest.raises(ValueError):
        spatial_agreement(np.zeros((2, 2)), np.zeros((3, 3)), classes)


# ── attribute_extent_drift ────────────────────────────────────────────────────

def test_attribute_extent_drift_basic(classes):
    trans_df = pd.DataFrame({
        "iteration":  [1, 1, 1, 1],
        "year":       [2020, 2020, 2021, 2021],
        "row":        [0, 1, 0, 1],
        "col":        [0, 1, 0, 1],
        "from_class": ["Mangrove", "Mangrove", "Mangrove", "Water_body"],
        "to_class":   ["Water_body", "Water_body", "Water_body", "Mangrove"],
        "group":      ["Inundation", "Erosion", "Inundation", "Mangrove_recruitment"],
    })
    drift = attribute_extent_drift(trans_df, class_id=2, classes=classes)  # Water_body

    incoming = drift[drift["direction"] == "incoming"]
    assert set(incoming["group"]) == {"Inundation", "Erosion"}
    inundation_row = incoming[incoming["group"] == "Inundation"].iloc[0]
    assert inundation_row["n_cells"] == 2
    # Two of three incoming cells are Inundation -> ~66.67%
    assert inundation_row["pct_of_class_total"] == pytest.approx(66.67, abs=0.1)

    outgoing = drift[drift["direction"] == "outgoing"]
    assert set(outgoing["group"]) == {"Mangrove_recruitment"}
    assert outgoing.iloc[0]["n_cells"] == 1


def test_attribute_extent_drift_empty_trans_df(classes):
    empty = pd.DataFrame(columns=["iteration", "year", "row", "col", "from_class", "to_class", "group"])
    result = attribute_extent_drift(empty, class_id=1, classes=classes)
    assert result.empty


# ── correction ─────────────────────────────────────────────────────────────────

def test_correct_multipliers_scaling(tmp_path):
    mult_path = tmp_path / "TransitionMultipliers.csv"
    mult_path.write_text(
        "TransitionGroupId,DistributionType,DistributionMin,DistributionMax,Amount\n"
        "Inundation [Type],Uniform,1.0,2.0,\n"
        "Agriculture_expansion [Type],Uniform,0.5,1.5,\n"
    )
    rate_ratios = pd.DataFrame({
        "group":          ["Inundation"],
        "observed_rate":  [0.05],
        "simulated_rate": [0.10],
        "ratio":          [0.5],
    })
    result = correct_multipliers(rate_ratios, mult_path, method="scaling")

    inundation_row = result[result["TransitionGroupId"] == "Inundation [Type]"].iloc[0]
    assert inundation_row["DistributionMin"] == pytest.approx(0.5)
    assert inundation_row["DistributionMax"] == pytest.approx(1.0)

    # Agriculture_expansion wasn't in rate_ratios -- must be untouched
    agri_row = result[result["TransitionGroupId"] == "Agriculture_expansion [Type]"].iloc[0]
    assert agri_row["DistributionMin"] == pytest.approx(0.5)
    assert agri_row["DistributionMax"] == pytest.approx(1.5)


def test_correct_multipliers_respects_bounds(tmp_path):
    mult_path = tmp_path / "TransitionMultipliers.csv"
    mult_path.write_text(
        "TransitionGroupId,DistributionType,DistributionMin,DistributionMax,Amount\n"
        "Inundation [Type],Uniform,1.0,2.0,\n"
    )
    rate_ratios = pd.DataFrame({
        "group": ["Inundation"], "observed_rate": [1.0],
        "simulated_rate": [0.001], "ratio": [1000.0],
    })
    result = correct_multipliers(rate_ratios, mult_path, method="scaling", bounds=(0.01, 50.0))
    row = result[result["TransitionGroupId"] == "Inundation [Type]"].iloc[0]
    assert row["DistributionMax"] <= 50.0


def test_correct_multipliers_optimize_not_implemented(tmp_path):
    mult_path = tmp_path / "TransitionMultipliers.csv"
    mult_path.write_text(
        "TransitionGroupId,DistributionType,DistributionMin,DistributionMax,Amount\n"
        "Inundation [Type],Uniform,1.0,2.0,\n"
    )
    rate_ratios = pd.DataFrame({"group": ["Inundation"], "ratio": [0.5]})
    with pytest.raises(NotImplementedError):
        correct_multipliers(rate_ratios, mult_path, method="optimize")


def test_compute_pathway_rate_ratios_basic(tmp_path):
    transitions_csv = tmp_path / "Transitions.csv"
    transitions_csv.write_text(
        "StateClassIdSource,StateClassIdDest,TransitionTypeId,Probability\n"
        "Mangrove:All,Water_body:All,Inundation [Type],0.05\n"
    )
    trans_df = pd.DataFrame({
        "iteration":  [1, 1],
        "year":       [2020, 2021],
        "row":        [0, 1],
        "col":        [0, 1],
        "from_class": ["Mangrove", "Mangrove"],
        "to_class":   ["Water_body", "Water_body"],
        "group":      ["Inundation", "Inundation"],
    })
    area_df = pd.DataFrame({
        "iteration":  [1, 1],
        "year":       [2020, 2020],
        "class_id":   [1, 2],
        "class_name": ["Mangrove", "Water_body"],
        "area_ha":    [100.0, 50.0],
    })
    result = compute_pathway_rate_ratios(trans_df, area_df, transitions_csv, n_timesteps=2)
    assert len(result) == 1
    assert result.iloc[0]["group"] == "Inundation"
    assert result.iloc[0]["observed_rate"] == pytest.approx(0.05)
    assert result.iloc[0]["simulated_rate"] > 0
