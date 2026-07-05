"""
tests/test_calibration_manifest.py  —  v3.11
Unit tests for the v3.11 calibration.manifest additions:
  - save_calibration_manifest() now emits Section 7 (Stock & Flow) as TODO
    placeholders, and accepts distributions_path
  - fill_manifest_from_calibration() updates an existing RunManifest.txt
    in place, preserving everything else (including a hand-filled Section 7)
  - transitions.load_group_map_csv() builds a group_map dict from a
    Transitions.csv-schema CSV
"""

from pathlib import Path

import pytest

from strategicc.calibration.manifest import (
    save_calibration_manifest,
    fill_manifest_from_calibration,
)
from strategicc.calibration.transitions import load_group_map_csv
from strategicc.io.csv_loader import StateClass


@pytest.fixture
def classes():
    return {
        1: StateClass(1, "Water_body", "Water_body:All", (255, 0, 128, 255)),
        3: StateClass(3, "Mangrove",   "Mangrove:All",   (255, 0, 100, 0)),
        5: StateClass(5, "Aquaculture", "Aquaculture:All", (255, 255, 0, 255)),
    }


# ── save_calibration_manifest: Section 7 ──────────────────────────────────────

def test_save_calibration_manifest_includes_section_7(tmp_path):
    out = save_calibration_manifest(out_path=tmp_path / "RunManifest_calibrated.txt")
    text = out.read_text()
    assert "SECTION 7" in text
    assert "USE_STOCKFLOW" in text
    assert "STOCK_TYPE_CSV" in text
    assert "FLOW_PATHWAYS_CSV" in text
    assert "SEEA_VALUATION_MODE" in text


def test_save_calibration_manifest_section_7_is_todo_placeholder(tmp_path):
    """Section 7 fields are never calibration-derived, so they must always
    be TODO, never marked [calibration]."""
    out = save_calibration_manifest(out_path=tmp_path / "RunManifest_calibrated.txt")
    text = out.read_text()
    section_7 = text.split("SECTION 7")[1]
    assert "[calibration]" not in section_7
    assert "TODO" in section_7


def test_save_calibration_manifest_includes_distributions_csv(tmp_path):
    dist_path = tmp_path / "Distributions.csv"
    out = save_calibration_manifest(
        distributions_path=dist_path,
        out_path=tmp_path / "RunManifest_calibrated.txt",
    )
    text = out.read_text()
    assert "DISTRIBUTIONS_CSV" in text
    assert str(dist_path) in text
    assert "Distributions.csv" in text.split("Calibrated in this run:")[1].split("====")[0]


def test_save_calibration_manifest_distributions_csv_todo_when_absent(tmp_path):
    out = save_calibration_manifest(out_path=tmp_path / "RunManifest_calibrated.txt")
    text = out.read_text()
    dist_line = [l for l in text.splitlines() if l.startswith("DISTRIBUTIONS_CSV")][0]
    assert "TODO" in dist_line


# ── fill_manifest_from_calibration ────────────────────────────────────────────

EXISTING_MANIFEST = """\
#================================================================
#SECTION 1 - INITIAL CONDITIONS (SPATIAL)
#================================================================
StateClassFileName        = 2022.tif              #path
AgeFileName               =                        #path  (fill in)

#================================================================
#SECTION 2 - MULTI-ROW CSV INPUTS
#================================================================
STATE_CLASSES_CSV                   = inputs/State Class.csv                       #path
TRANSITIONS_CSV                     =                                              #path
TRANSITION_MULT_CSV                 =                                              #path
TRANSITION_SIZE_CSV                 =                                              #path
OUT_DIR                             = strategicc_output/                           #path

#================================================================
#SECTION 3 - RUN CONTROL
#================================================================
START_YEAR     = 2022   #int
N_TIMESTEPS    = 20     #int

#================================================================
#SECTION 7 - STOCK & FLOW
#================================================================
USE_STOCKFLOW                  = True    #bool  (hand-filled by the user)
STOCK_TYPE_CSV                 = inputs/StockType.csv                  #path
SEEA_VALUATION_MODE            = stock_flow    #str
"""


@pytest.fixture
def existing_manifest(tmp_path):
    p = tmp_path / "RunManifest.txt"
    p.write_text(EXISTING_MANIFEST)
    return p


def test_fill_manifest_updates_only_requested_fields(existing_manifest, tmp_path):
    out = fill_manifest_from_calibration(
        existing_manifest,
        transitions_path=tmp_path / "Transitions.csv",
        age_raster_path=tmp_path / "age.tif",
        out_path=tmp_path / "RunManifest_filled.txt",
    )
    text = out.read_text()
    assert f"AgeFileName               = {tmp_path / 'age.tif'}" in text
    assert str(tmp_path / "Transitions.csv") in text
    # Untouched: no path was supplied for these
    assert "TRANSITION_MULT_CSV                 =                                              #path" in text
    assert "TRANSITION_SIZE_CSV                 =                                              #path" in text


def test_fill_manifest_preserves_section_7_untouched(existing_manifest, tmp_path):
    out = fill_manifest_from_calibration(
        existing_manifest,
        transitions_path=tmp_path / "Transitions.csv",
        temporal_path=tmp_path / "TransitionMultipliers.csv",
        size_dist_path=tmp_path / "TransitionSizeDistribution.csv",
        age_raster_path=tmp_path / "age.tif",
        out_path=tmp_path / "RunManifest_filled.txt",
    )
    text = out.read_text()
    assert "USE_STOCKFLOW                  = True    #bool  (hand-filled by the user)" in text
    assert "STOCK_TYPE_CSV                 = inputs/StockType.csv                  #path" in text
    assert "SEEA_VALUATION_MODE            = stock_flow    #str" in text


def test_fill_manifest_preserves_other_sections(existing_manifest, tmp_path):
    out = fill_manifest_from_calibration(
        existing_manifest,
        transitions_path=tmp_path / "Transitions.csv",
        out_path=tmp_path / "RunManifest_filled.txt",
    )
    text = out.read_text()
    assert "StateClassFileName        = 2022.tif              #path" in text
    assert "START_YEAR     = 2022   #int" in text
    assert "OUT_DIR                             = strategicc_output/                           #path" in text


def test_fill_manifest_distributions_csv_appended_when_missing(existing_manifest, tmp_path):
    """The fixture manifest has no DISTRIBUTIONS_CSV line at all; requesting
    it should append a new line rather than silently doing nothing."""
    dist_path = tmp_path / "Distributions.csv"
    out = fill_manifest_from_calibration(
        existing_manifest,
        distributions_path=dist_path,
        out_path=tmp_path / "RunManifest_filled.txt",
    )
    text = out.read_text()
    assert "DISTRIBUTIONS_CSV" in text
    assert str(dist_path) in text
    # Should land in Section 2, right after TRANSITION_MULT_CSV
    lines = text.splitlines()
    mult_idx = next(i for i, l in enumerate(lines) if l.startswith("TRANSITION_MULT_CSV"))
    dist_idx = next(i for i, l in enumerate(lines) if l.startswith("DISTRIBUTIONS_CSV"))
    assert dist_idx == mult_idx + 1


def test_fill_manifest_defaults_to_overwriting_in_place(existing_manifest, tmp_path):
    fill_manifest_from_calibration(
        existing_manifest,
        transitions_path=tmp_path / "Transitions.csv",
    )
    text = existing_manifest.read_text()
    assert str(tmp_path / "Transitions.csv") in text


def test_fill_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fill_manifest_from_calibration(tmp_path / "nonexistent.txt", transitions_path=tmp_path / "x.csv")


def test_fill_manifest_no_args_leaves_file_unchanged(existing_manifest):
    original = existing_manifest.read_text()
    out = fill_manifest_from_calibration(existing_manifest)
    assert out.read_text() == original


# ── load_group_map_csv ────────────────────────────────────────────────────────

GROUP_MAP_CSV = """\
Iteration,Timestep,StratumIdSource,StateClassIdSource,StratumIdDest,StateClassIdDest,SecondaryStratumId,TertiaryStratumId,TransitionTypeId,Probability,Proportion,AgeMin,AgeMax,AgeRelative,AgeReset,TSTMin,TSTMax,TSTRelative
,,,Mangrove:All,,Aquaculture:All,,,Aquaculture_expansion [Type],,,,,,,,,
,,,Water_body:All,,Mangrove:All,,,Mangrove_recruitment [Type],,,,,,,,,
"""


@pytest.fixture
def group_map_csv(tmp_path):
    p = tmp_path / "group_map.csv"
    p.write_text(GROUP_MAP_CSV)
    return p


def test_load_group_map_csv_builds_expected_dict(group_map_csv, classes):
    group_map = load_group_map_csv(group_map_csv, classes)
    assert group_map == {
        (3, 5): "Aquaculture_expansion",
        (1, 3): "Mangrove_recruitment",
    }


def test_load_group_map_csv_ignores_blank_rows(tmp_path, classes):
    p = tmp_path / "group_map.csv"
    p.write_text(
        "StateClassIdSource,StateClassIdDest,TransitionTypeId,Probability\n"
        "Mangrove:All,Aquaculture:All,Aquaculture_expansion [Type],\n"
        ",,,\n"
    )
    group_map = load_group_map_csv(p, classes)
    assert group_map == {(3, 5): "Aquaculture_expansion"}


def test_load_group_map_csv_skips_unresolved_class_labels(tmp_path, classes, capsys):
    p = tmp_path / "group_map.csv"
    p.write_text(
        "StateClassIdSource,StateClassIdDest,TransitionTypeId,Probability\n"
        "Mangrove:All,Aquaculture:All,Aquaculture_expansion [Type],\n"
        "Unknown:All,Aquaculture:All,Aquaculture_expansion [Type],\n"
    )
    group_map = load_group_map_csv(p, classes)
    assert group_map == {(3, 5): "Aquaculture_expansion"}
    captured = capsys.readouterr()
    assert "Unknown:All" in captured.out


def test_load_group_map_csv_feeds_compute_transition_rates(group_map_csv, classes):
    """Round-trip: a group_map loaded from CSV should work identically to
    a hand-typed dict when passed into compute_transition_rates()."""
    from strategicc.calibration.transitions import (
        compute_yearly_transition_counts, compute_transition_rates,
        YearlyTransitionCounts,
    )
    import pandas as pd

    group_map = load_group_map_csv(group_map_csv, classes)
    yearly = YearlyTransitionCounts(records=pd.DataFrame([
        {"year": 2020, "from_id": 3, "to_id": 5, "n_cells": 10,
         "n_from_total": 100, "probability": 0.1},
    ]))
    result = compute_transition_rates(yearly, classes, group_map)
    assert len(result) == 1
    assert result.iloc[0]["TransitionTypeId"] == "Aquaculture_expansion"
