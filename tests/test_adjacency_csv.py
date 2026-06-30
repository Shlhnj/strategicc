"""
tests/test_adjacency_csv.py  —  v3.1
Unit tests for CSV-driven Transition Adjacency Setting/Multipliers.
"""

import pytest

from strategicc.io.csv_loader import (
    load_transition_adjacency_setting,
    load_transition_adjacency_multipliers,
    build_adjacency_strength_map,
    TransitionAdjacencyMultiplierRule,
)


# ── load_transition_adjacency_setting ─────────────────────────────────────────

def test_load_setting_basic(tmp_path):
    p = tmp_path / "Setting.csv"
    p.write_text(
        "TransitionGroupId,StateClassId,StateAttributeTypeId,"
        "NeighborhoodRadius,UpdateFrequency\n"
        "Aquaculture_expansion,,,,\n"
    )
    rules = load_transition_adjacency_setting(p)
    assert len(rules) == 1
    assert rules[0].group == "Aquaculture_expansion"
    assert rules[0].state_class is None
    assert rules[0].neighborhood_radius is None

def test_load_setting_skips_blank_group(tmp_path):
    p = tmp_path / "Setting.csv"
    p.write_text(
        "TransitionGroupId,StateClassId,StateAttributeTypeId,"
        "NeighborhoodRadius,UpdateFrequency\n"
        ",,,,\n"
    )
    rules = load_transition_adjacency_setting(p)
    assert len(rules) == 0

def test_load_setting_strips_type_suffix(tmp_path):
    p = tmp_path / "Setting.csv"
    p.write_text(
        "TransitionGroupId,StateClassId,StateAttributeTypeId,"
        "NeighborhoodRadius,UpdateFrequency\n"
        "Aquaculture_expansion [Type],,,,\n"
    )
    rules = load_transition_adjacency_setting(p)
    assert rules[0].group == "Aquaculture_expansion"

def test_load_setting_with_attribute_fields_populated(tmp_path):
    p = tmp_path / "Setting.csv"
    p.write_text(
        "TransitionGroupId,StateClassId,StateAttributeTypeId,"
        "NeighborhoodRadius,UpdateFrequency\n"
        "Fire,SomeState,Density,500,1\n"
    )
    rules = load_transition_adjacency_setting(p)
    assert rules[0].state_class == "SomeState"
    assert rules[0].state_attribute_type == "Density"
    assert rules[0].neighborhood_radius == 500.0
    assert rules[0].update_frequency == 1


# ── load_transition_adjacency_multipliers ─────────────────────────────────────

def test_load_multipliers_basic(tmp_path):
    p = tmp_path / "Mults.csv"
    p.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,AttributeValue,Amount,DistributionType,"
        "DistributionFrequencyId,DistributionSD,DistributionMin,DistributionMax\n"
        ",,,,,Aquaculture_expansion,,4.0,,,,,\n"
    )
    rules = load_transition_adjacency_multipliers(p)
    assert len(rules) == 1
    assert rules[0].group == "Aquaculture_expansion"
    assert rules[0].amount == 4.0
    assert rules[0].attribute_value is None

def test_load_multipliers_skips_missing_amount(tmp_path):
    p = tmp_path / "Mults.csv"
    p.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,AttributeValue,Amount,DistributionType,"
        "DistributionFrequencyId,DistributionSD,DistributionMin,DistributionMax\n"
        ",,,,,Aquaculture_expansion,,,,,,,\n"
    )
    rules = load_transition_adjacency_multipliers(p)
    assert len(rules) == 0

def test_load_multipliers_parses_attribute_value_rows(tmp_path, capsys):
    p = tmp_path / "Mults.csv"
    p.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,AttributeValue,Amount,DistributionType,"
        "DistributionFrequencyId,DistributionSD,DistributionMin,DistributionMax\n"
        ",,,,,Fire,0.5,2.0,,,,,\n"
    )
    rules = load_transition_adjacency_multipliers(p)
    assert len(rules) == 1
    assert rules[0].attribute_value == 0.5
    captured = capsys.readouterr()
    assert "AttributeValue" in captured.out


# ── build_adjacency_strength_map ──────────────────────────────────────────────

def test_build_strength_map_uses_blank_attribute_rows_only():
    rules = [
        TransitionAdjacencyMultiplierRule(group="Fire", amount=4.0, attribute_value=None),
        TransitionAdjacencyMultiplierRule(group="Flood", amount=2.0, attribute_value=0.5),
    ]
    strength_map = build_adjacency_strength_map(rules)
    assert strength_map == {"Fire": 4.0}
    assert "Flood" not in strength_map

def test_build_strength_map_last_blank_row_wins():
    rules = [
        TransitionAdjacencyMultiplierRule(group="Fire", amount=4.0, attribute_value=None),
        TransitionAdjacencyMultiplierRule(group="Fire", amount=8.0, attribute_value=None),
    ]
    strength_map = build_adjacency_strength_map(rules)
    assert strength_map["Fire"] == 8.0

def test_build_strength_map_empty_input():
    assert build_adjacency_strength_map([]) == {}
