"""
tests/test_calibration_transitions.py

Unit tests for strategicc/calibration/transitions.py — previously at 71%
coverage. Covers edge-case branches in compute_yearly_transition_counts()
and compute_size_distribution() (non-contiguous years, class-0 skip,
empty-mask skip), compute_transition_rates() (empty input, min_probability
filter, unresolved class ids), normalize_transition_rates() end to end,
and the two CSV-saving helpers.
"""

import numpy as np
import pandas as pd
import pytest

from strategicc.calibration.loader import LULCTimeSeries
from strategicc.calibration.transitions import (
    compute_yearly_transition_counts,
    compute_transition_rates,
    normalize_transition_rates,
    save_transitions_csv,
    compute_size_distribution,
    save_size_distribution_csv,
    YearlyTransitionCounts,
)
from strategicc.io.csv_loader import StateClass


@pytest.fixture
def classes():
    return {
        1: StateClass(id=1, name="Mangrove", full_name="Mangrove:All", color=(255, 0, 128, 0)),
        2: StateClass(id=2, name="Aquaculture", full_name="Aquaculture:All", color=(255, 255, 0, 255)),
    }


def _profile():
    return {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": 4, "width": 4}


# ── compute_yearly_transition_counts ────────────────────────────────────────

def test_yearly_counts_skips_non_contiguous_year_gap():
    """A gap in ts.years (e.g. 2020 -> 2022, missing 2021) must not be
    treated as a one-year transition."""
    stack = np.array([
        np.ones((3, 3), dtype=np.uint8),
        np.full((3, 3), 2, dtype=np.uint8),
    ])
    ts = LULCTimeSeries(stack=stack, years=[2020, 2022], profile=_profile())
    counts = compute_yearly_transition_counts(ts)
    assert counts.records.empty


def test_yearly_counts_skips_class_zero():
    """Class 0 (nodata) as a from_id must be excluded from output."""
    map_from = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    map_to = np.array([[0, 2], [2, 1]], dtype=np.uint8)
    ts = LULCTimeSeries(stack=np.array([map_from, map_to]), years=[2020, 2021], profile=_profile())
    counts = compute_yearly_transition_counts(ts)
    assert 0 not in counts.records["from_id"].values


def test_yearly_counts_basic():
    map_from = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    map_to = np.array([[2, 2], [1, 1]], dtype=np.uint8)
    ts = LULCTimeSeries(stack=np.array([map_from, map_to]), years=[2020, 2021], profile=_profile())
    counts = compute_yearly_transition_counts(ts)
    row = counts.records.iloc[0]
    assert row["from_id"] == 1
    assert row["to_id"] == 2
    assert row["n_cells"] == 2
    assert row["n_from_total"] == 4
    assert row["probability"] == pytest.approx(0.5)


# ── compute_transition_rates ────────────────────────────────────────────────

def test_compute_transition_rates_empty_input_warns(classes, capsys):
    yearly = YearlyTransitionCounts(records=pd.DataFrame())
    result = compute_transition_rates(yearly, classes, group_map={})
    assert result.empty
    assert "No transition records" in capsys.readouterr().out


def test_compute_transition_rates_filters_below_min_probability(classes):
    records = pd.DataFrame({
        "year": [2020], "from_id": [1], "to_id": [2],
        "n_cells": [1], "n_from_total": [100000], "probability": [0.00001],
    })
    yearly = YearlyTransitionCounts(records=records)
    group_map = {(1, 2): "Aquaculture_expansion"}
    result = compute_transition_rates(yearly, classes, group_map, min_probability=0.001)
    assert result.empty


def test_compute_transition_rates_skips_unresolved_class_id(classes):
    """A pathway mapped in group_map but referencing a from/to id absent
    from `classes` must be silently dropped (not raise)."""
    records = pd.DataFrame({
        "year": [2020], "from_id": [1], "to_id": [99],
        "n_cells": [10], "n_from_total": [100], "probability": [0.1],
    })
    yearly = YearlyTransitionCounts(records=records)
    group_map = {(1, 99): "Ghost_transition"}
    result = compute_transition_rates(yearly, classes, group_map)
    assert result.empty


def test_compute_transition_rates_unmapped_pair_excluded_with_warning(classes, capsys):
    records = pd.DataFrame({
        "year": [2020], "from_id": [1], "to_id": [2],
        "n_cells": [10], "n_from_total": [100], "probability": [0.1],
    })
    yearly = YearlyTransitionCounts(records=records)
    result = compute_transition_rates(yearly, classes, group_map={})   # nothing mapped
    assert result.empty
    assert "unmapped" in capsys.readouterr().out


# ── normalize_transition_rates ──────────────────────────────────────────────

def test_normalize_transition_rates_empty_input_warns(classes, capsys):
    yearly = YearlyTransitionCounts(records=pd.DataFrame())
    result = normalize_transition_rates(pd.DataFrame(), yearly, {}, classes)
    assert result.empty
    assert "nothing to normalize" in capsys.readouterr().out


def test_normalize_transition_rates_rescales_for_dropped_mass(classes, capsys):
    """Class 1 has two observed destinations (2 and 3), but only (1,2) is
    in group_map. The kept pathway's probability should be scaled up to
    account for the excluded (1,3) mass."""
    records = pd.DataFrame({
        "year":         [2020, 2020],
        "from_id":      [1, 1],
        "to_id":        [2, 3],
        "n_cells":      [10, 5],
        "n_from_total": [100, 100],
        "probability":  [0.10, 0.05],
    })
    yearly = YearlyTransitionCounts(records=records)
    group_map = {(1, 2): "Aquaculture_expansion"}
    classes_3 = dict(classes)
    classes_3[3] = StateClass(id=3, name="Water_body", full_name="Water_body:All", color=(255, 0, 0, 255))

    transitions_df = compute_transition_rates(yearly, classes_3, group_map)
    result = normalize_transition_rates(transitions_df, yearly, group_map, classes_3)

    # raw_total = 0.15, sanctioned_total = 0.10 -> scale_factor = 1.5
    assert result.iloc[0]["Probability"] == pytest.approx(0.15, abs=1e-6)
    out = capsys.readouterr().out
    assert "rescaled" in out


def test_normalize_transition_rates_zero_sanctioned_total_left_at_one(classes, capsys):
    """If a source class has observed outgoing transitions but NONE are in
    group_map, scale_factor stays 1.0 and a warning is printed."""
    records = pd.DataFrame({
        "year": [2020], "from_id": [1], "to_id": [2],
        "n_cells": [10], "n_from_total": [100], "probability": [0.1],
    })
    yearly = YearlyTransitionCounts(records=records)
    # transitions_df has a row for class 1 but group_map has since been
    # emptied (simulating a source class with zero sanctioned pathways)
    transitions_df = pd.DataFrame({
        "StateClassIdSource": ["Mangrove:All"],
        "StateClassIdDest":   ["Aquaculture:All"],
        "TransitionTypeId":   ["Aquaculture_expansion"],
        "Probability":        [0.1],
    })
    result = normalize_transition_rates(transitions_df, yearly, group_map={}, classes=classes)
    assert result.iloc[0]["Probability"] == pytest.approx(0.1)
    assert "no sanctioned pathway" in capsys.readouterr().out


def test_normalize_transition_rates_unresolved_source_left_unscaled(classes, capsys):
    records = pd.DataFrame({
        "year": [2020], "from_id": [1], "to_id": [2],
        "n_cells": [10], "n_from_total": [100], "probability": [0.1],
    })
    yearly = YearlyTransitionCounts(records=records)
    transitions_df = pd.DataFrame({
        "StateClassIdSource": ["NotInClassesDict:All"],
        "StateClassIdDest":   ["Aquaculture:All"],
        "TransitionTypeId":   ["Aquaculture_expansion"],
        "Probability":        [0.2],
    })
    result = normalize_transition_rates(
        transitions_df, yearly, group_map={(1, 2): "Aquaculture_expansion"}, classes=classes
    )
    assert result.iloc[0]["Probability"] == pytest.approx(0.2)   # unscaled
    assert "did not resolve" in capsys.readouterr().out


# ── save_transitions_csv ────────────────────────────────────────────────────

def test_save_transitions_csv_writes_file(tmp_path):
    df = pd.DataFrame({
        "StateClassIdSource": ["Mangrove:All"],
        "StateClassIdDest":   ["Aquaculture:All"],
        "TransitionTypeId":   ["Aquaculture_expansion"],
        "Probability":        [0.1],
    })
    out_path = tmp_path / "Transitions.csv"
    result_path = save_transitions_csv(df, out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert pd.read_csv(out_path).equals(df)


def test_save_transitions_csv_default_path(tmp_path, monkeypatch):
    import strategicc.calibration.paths as paths_mod
    monkeypatch.setattr(paths_mod, "TRANSITIONS_CSV", tmp_path / "default" / "Transitions.csv")
    df = pd.DataFrame({"StateClassIdSource": ["A"], "StateClassIdDest": ["B"],
                        "TransitionTypeId": ["G"], "Probability": [0.1]})
    result_path = save_transitions_csv(df)
    assert result_path == tmp_path / "default" / "Transitions.csv"
    assert result_path.exists()


# ── compute_size_distribution edge cases ────────────────────────────────────

def test_size_distribution_skips_non_contiguous_years():
    map1 = np.ones((5, 5), dtype=np.uint8)
    map2 = np.full((5, 5), 2, dtype=np.uint8)
    ts = LULCTimeSeries(stack=np.array([map1, map2]), years=[2020, 2022], profile=_profile())
    group_map = {(1, 2): "G1"}
    df = compute_size_distribution(ts, group_map, px_area_ha=1.0, min_patches=1)
    assert df.empty


def test_size_distribution_skips_year_pair_with_no_mask_hits():
    """A group whose (from,to) pair never actually co-occurs in a given
    year-pair should just skip that pair (mask.any() is False) rather
    than error, and still work correctly for a later pair that does."""
    map1 = np.ones((5, 5), dtype=np.uint8)
    map2 = np.ones((5, 5), dtype=np.uint8)      # no change: 2020->2021
    map3 = np.full((5, 5), 2, dtype=np.uint8)   # change: 2021->2022
    ts = LULCTimeSeries(stack=np.array([map1, map2, map3]), years=[2020, 2021, 2022], profile=_profile())
    group_map = {(1, 2): "G1"}
    df = compute_size_distribution(ts, group_map, px_area_ha=1.0, n_bins=2, min_patches=1)
    assert not df.empty
    assert (df["Transition Type/Group"] == "G1 [Type]").all()


# ── save_size_distribution_csv ──────────────────────────────────────────────

def test_save_size_distribution_csv_writes_file(tmp_path):
    df = pd.DataFrame({
        "Transition Type/Group": ["G1 [Type]"],
        "Maximum Area (Hectares)": [1.0],
        "Relative Amount": [100.0],
    })
    out_path = tmp_path / "TransitionSizeDistribution.csv"
    result_path = save_size_distribution_csv(df, out_path)
    assert result_path == out_path
    assert out_path.exists()


def test_save_size_distribution_csv_default_path(tmp_path, monkeypatch):
    import strategicc.calibration.paths as paths_mod
    monkeypatch.setattr(paths_mod, "TRANS_SIZE_CSV", tmp_path / "default" / "TransitionSizeDistribution.csv")
    df = pd.DataFrame({
        "Transition Type/Group": ["G1 [Type]"],
        "Maximum Area (Hectares)": [1.0],
        "Relative Amount": [100.0],
    })
    result_path = save_size_distribution_csv(df)
    assert result_path == tmp_path / "default" / "TransitionSizeDistribution.csv"
    assert result_path.exists()
