"""
tests/test_stockflow_csv_loader.py

Unit tests for the previously-untested parts of strategicc/stockflow/csv_loader.py
(80% -> full coverage target): load_stock_groups, load_stock_group_membership,
load_state_attribute_types, State Attribute Values row-skip branch, Flow
Pathways row-skip branch, Flow Multiplier fallback-column + row-skip
branches, and validate_flow_pathways' individual warning branches.
"""

import pytest

from strategicc.stockflow.csv_loader import (
    load_stock_groups,
    load_stock_group_membership,
    load_state_attribute_types,
    load_state_attribute_values,
    load_flow_pathways,
    load_flow_multipliers,
    validate_flow_pathways,
    StateAttributeValueRule,
    FlowPathwayRule,
)
from strategicc.io.csv_loader import StateClass


# ── Simple name-list / mapping loaders ──────────────────────────────────────

def test_load_stock_groups(tmp_path):
    path = tmp_path / "StockGroup.csv"
    path.write_text("Name\nBiomass Pools [Group]\nSoil Pools [Group]\n")
    names = load_stock_groups(path)
    assert names == ["Biomass Pools", "Soil Pools"]


def test_load_stock_groups_skips_blank_names(tmp_path):
    path = tmp_path / "StockGroup.csv"
    path.write_text("Name\n\nBiomass\n")
    names = load_stock_groups(path)
    assert names == ["Biomass"]


def test_load_stock_group_membership(tmp_path):
    path = tmp_path / "Membership.csv"
    path.write_text(
        "StockTypeId,StockGroupId\n"
        "Biomass,Above Ground [Group]\n"
        "Soil,Above Ground [Group]\n"
        "Atmosphere,\n"
    )
    membership = load_stock_group_membership(path)
    assert membership == {"Above Ground": ["Biomass", "Soil"]}


def test_load_state_attribute_types(tmp_path):
    path = tmp_path / "StateAttributeType.csv"
    path.write_text("Name\nNPP\nEmission Rate\n")
    names = load_state_attribute_types(path)
    assert names == ["NPP", "Emission Rate"]


# ── State Attribute Values row-skip branch ──────────────────────────────────

def test_load_state_attribute_values_skips_rows_missing_attr_or_value(tmp_path):
    path = tmp_path / "StateAttributeValues.csv"
    path.write_text(
        "StateAttributeTypeId,StateClassId,AgeMin,AgeMax,Value\n"
        "NPP,Mangrove,0,10,5.0\n"
        ",Mangrove,0,10,5.0\n"          # missing attribute type -> skipped
        "NPP,Mangrove,0,10,\n"          # missing value -> skipped
    )
    rules = load_state_attribute_values(path)
    assert len(rules) == 1
    assert rules[0].attribute_type == "NPP"


# ── Flow Pathways row-skip branch ───────────────────────────────────────────

def test_load_flow_pathways_skips_incomplete_rows(tmp_path):
    header = (
        "FromStateClassId,FromAgeMin,FromStockTypeId,ToStateClassId,ToAgeMin,"
        "ToStockTypeId,TransitionGroupId,StateAttributeTypeId,FlowTypeId,"
        "TargetType,Multiplier\n"
    )
    path = tmp_path / "FlowPathways.csv"
    path.write_text(
        header
        + "Mangrove,,,,,Biomass,,,NPP [Type],Flow,1\n"          # missing FromStockTypeId -> skip
        + ",,Atmosphere,,,Biomass,,,,Flow,1\n"                   # missing FlowTypeId -> skip
        + ",,Atmosphere,,,Biomass,,,NPP [Type],Flow,\n"          # missing Multiplier -> skip
    )
    rules = load_flow_pathways(path)
    assert rules == []


def test_load_flow_pathways_target_type_variants(tmp_path):
    header = (
        "FromStateClassId,FromAgeMin,FromStockTypeId,ToStateClassId,ToAgeMin,"
        "ToStockTypeId,TransitionGroupId,StateAttributeTypeId,FlowTypeId,"
        "TargetType,Multiplier\n"
    )
    path = tmp_path / "FlowPathways.csv"
    path.write_text(
        header
        + ",,Atmosphere,,,Biomass,,,NPP [Type],To Stock,0.5\n"
        + ",,Biomass,,,Atmosphere,,,Emission [Type],From Stock,0.2\n"
        + ",,Biomass,,,Soil,,,Humification [Type],,0.1\n"   # blank -> defaults to Flow
    )
    rules = load_flow_pathways(path)
    assert [r.target_type for r in rules] == ["ToStock", "FromStock", "Flow"]


# ── Flow Multiplier fallback column + row-skip branch ───────────────────────

def test_load_flow_multipliers_fallback_column_and_skip(tmp_path):
    path = tmp_path / "FlowMultiplier.csv"
    path.write_text(
        "FlowGroupId,FlowMultiplierTypeId,DistributionType,DistributionMin,DistributionMax\n"
        ",NPP [Type],Uniform,0.5,1.5\n"                # FlowGroupId blank -> fallback to FlowMultiplierTypeId
        "Emission [Type],,Uniform,0.8,1.2\n"            # FlowGroupId present -> used directly
        ",,Uniform,0.5,1.5\n"                            # both blank -> skipped
        "Humification [Type],,,0.5,1.5\n"                # missing DistributionType -> skipped
    )
    rules = load_flow_multipliers(path)
    assert [r.flow_type for r in rules] == ["NPP", "Emission"]


# ── Initial Stock - Non Spatial ─────────────────────────────────────────────

def test_load_initial_stock_links(tmp_path):
    from strategicc.stockflow.csv_loader import load_initial_stock_links
    path = tmp_path / "InitialStockNonSpatial.csv"
    path.write_text(
        "StockTypeId,StateAttributeTypeId\n"
        "Biomass,NPP\n"
        ",NPP\n"          # missing stock -> skipped
        "Soil,\n"         # missing attribute -> skipped
    )
    links = load_initial_stock_links(path)
    assert links == {"Biomass": "NPP"}




@pytest.fixture
def classes():
    return {1: StateClass(id=1, name="Mangrove", full_name="Mangrove:All", color=(255, 0, 128, 0))}


def test_validate_flow_pathways_all_valid_prints_ok(classes, capsys):
    pathways = [FlowPathwayRule(
        from_state_class="Mangrove", from_age_min=None, from_stock_type="Atmosphere",
        to_state_class=None, to_age_min=None, to_stock_type="Biomass",
        transition_group=None, state_attribute="NPP", flow_type="NPP",
        target_type="Flow", multiplier=1.0,
    )]
    warnings = validate_flow_pathways(
        pathways, stock_types=["Atmosphere", "Biomass"], flow_types=["NPP"],
        state_attr_rules=[StateAttributeValueRule("NPP", None, None, None, 1.0)],
        classes=classes,
    )
    assert warnings == []
    assert "reference defined" in capsys.readouterr().out


def test_validate_flow_pathways_unknown_from_stock_type(classes):
    pathways = [FlowPathwayRule(
        from_state_class=None, from_age_min=None, from_stock_type="Ghost",
        to_state_class=None, to_age_min=None, to_stock_type="Biomass",
        transition_group=None, state_attribute=None, flow_type="NPP",
        target_type="Flow", multiplier=1.0,
    )]
    warnings = validate_flow_pathways(pathways, ["Biomass"], ["NPP"], [], classes)
    assert any("FromStockTypeId 'Ghost'" in w for w in warnings)


def test_validate_flow_pathways_unknown_to_stock_type(classes):
    pathways = [FlowPathwayRule(
        from_state_class=None, from_age_min=None, from_stock_type="Biomass",
        to_state_class=None, to_age_min=None, to_stock_type="Ghost",
        transition_group=None, state_attribute=None, flow_type="NPP",
        target_type="Flow", multiplier=1.0,
    )]
    warnings = validate_flow_pathways(pathways, ["Biomass"], ["NPP"], [], classes)
    assert any("ToStockTypeId 'Ghost'" in w for w in warnings)


def test_validate_flow_pathways_unknown_flow_type(classes):
    pathways = [FlowPathwayRule(
        from_state_class=None, from_age_min=None, from_stock_type="Biomass",
        to_state_class=None, to_age_min=None, to_stock_type="Biomass",
        transition_group=None, state_attribute=None, flow_type="Ghost",
        target_type="Flow", multiplier=1.0,
    )]
    warnings = validate_flow_pathways(pathways, ["Biomass"], ["NPP"], [], classes)
    assert any("FlowTypeId 'Ghost'" in w for w in warnings)


def test_validate_flow_pathways_unknown_state_attribute(classes):
    pathways = [FlowPathwayRule(
        from_state_class=None, from_age_min=None, from_stock_type="Biomass",
        to_state_class=None, to_age_min=None, to_stock_type="Biomass",
        transition_group=None, state_attribute="GhostAttr", flow_type="NPP",
        target_type="Flow", multiplier=1.0,
    )]
    warnings = validate_flow_pathways(pathways, ["Biomass"], ["NPP"], [], classes)
    assert any("StateAttributeTypeId 'GhostAttr'" in w for w in warnings)


def test_validate_flow_pathways_unknown_from_state_class(classes):
    pathways = [FlowPathwayRule(
        from_state_class="GhostClass", from_age_min=None, from_stock_type="Biomass",
        to_state_class=None, to_age_min=None, to_stock_type="Biomass",
        transition_group=None, state_attribute=None, flow_type="NPP",
        target_type="Flow", multiplier=1.0,
    )]
    warnings = validate_flow_pathways(pathways, ["Biomass"], ["NPP"], [], classes)
    assert any("FromStateClassId 'GhostClass'" in w for w in warnings)
