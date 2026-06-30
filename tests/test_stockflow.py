"""
tests/test_stockflow.py  —  v3.2
Unit tests for the Stock & Flow engine (strategicc/stockflow/).

Validated against Alongi (2020) mangrove carbon mass-balance reference
values where applicable (NPP ~18.4 Mg C/ha/yr combined AGB+BGB).
"""

import pytest
import numpy as np

from strategicc.stockflow.csv_loader import (
    FlowPathwayRule, StateAttributeValueRule, lookup_state_attribute,
    load_flow_pathways,
)
from strategicc.stockflow.engine import (
    init_stocks, run_flows_for_timestep, build_age_attribute_cache,
)
from strategicc.io.csv_loader import StateClass
from strategicc.core.transitions import TransitionRecord


# ── lookup_state_attribute ────────────────────────────────────────────────────

def test_lookup_state_attribute_basic():
    rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    assert lookup_state_attribute(rules, "NPP", age=10) == 18.4

def test_lookup_state_attribute_age_brackets():
    rules = [
        StateAttributeValueRule("NPP", None, 0, 10, 5.0),
        StateAttributeValueRule("NPP", None, 11, 20, 8.0),
        StateAttributeValueRule("NPP", None, 21, None, 12.0),
    ]
    assert lookup_state_attribute(rules, "NPP", age=5) == 5.0
    assert lookup_state_attribute(rules, "NPP", age=15) == 8.0
    assert lookup_state_attribute(rules, "NPP", age=50) == 12.0

def test_lookup_state_attribute_no_match_returns_none():
    rules = [StateAttributeValueRule("NPP", None, 0, 10, 5.0)]
    assert lookup_state_attribute(rules, "NPP", age=99) is None

def test_lookup_state_attribute_class_specific_wins_over_wildcard():
    rules = [
        StateAttributeValueRule("NPP", None, None, None, 5.0),
        StateAttributeValueRule("NPP", "Mangrove", None, None, 18.4),
    ]
    assert lookup_state_attribute(rules, "NPP", age=10, state_class="Mangrove") == 18.4
    assert lookup_state_attribute(rules, "NPP", age=10, state_class="Other") == 5.0


# ── build_age_attribute_cache ────────────────────────────────────────────────

def test_build_age_attribute_cache_vectorised():
    age_map = np.array([[10, 20], [30, 10]], dtype=np.uint16)
    rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    cache = build_age_attribute_cache(age_map, rules, "NPP")
    assert np.all(cache == 18.4)
    assert cache.shape == age_map.shape

def test_build_age_attribute_cache_unmatched_age_is_zero():
    age_map = np.array([[5]], dtype=np.uint16)
    rules = [StateAttributeValueRule("NPP", None, 10, 20, 18.4)]
    cache = build_age_attribute_cache(age_map, rules, "NPP")
    assert cache[0, 0] == 0.0


# ── load_flow_pathways: column-shift regression test ─────────────────────────

def test_load_flow_pathways_correct_column_alignment(tmp_path):
    """
    Regression test: real Flow Pathways rows have an empty StateClassId
    field between source class and stock-type columns. A naive
    hand-written CSV with too few commas silently shifts every
    subsequent field by one column — this caught real bugs during
    development and must never regress.
    """
    p = tmp_path / "fp.csv"
    p.write_text(
        "Iteration,Timestep,FromStratumId,FromSecondaryStratumId,FromTertiaryStratumId,"
        "FromStateClassId,FromAgeMin,FromStockTypeId,ToStratumId,ToStateClassId,ToAgeMin,"
        "ToStockTypeId,TransitionGroupId,StateAttributeTypeId,FlowTypeId,TargetType,"
        "Multiplier,TransferToStratumId,TransferToSecondaryStratumId,"
        "TransferToTertiaryStratumId,TransferToStateClassId,TransferToAgeMin\n"
        ",,,,,,,Atmosphere,,,,Biomass,,NPP,NPP,,1,,,,,\n"
    )
    rules = load_flow_pathways(p)
    assert len(rules) == 1
    assert rules[0].from_stock_type == "Atmosphere"
    assert rules[0].to_stock_type == "Biomass"
    assert rules[0].flow_type == "NPP"
    assert rules[0].state_attribute == "NPP"
    assert rules[0].multiplier == 1.0
    assert rules[0].transition_group is None

def test_load_flow_pathways_transition_triggered(tmp_path):
    p = tmp_path / "fp.csv"
    p.write_text(
        "Iteration,Timestep,FromStratumId,FromSecondaryStratumId,FromTertiaryStratumId,"
        "FromStateClassId,FromAgeMin,FromStockTypeId,ToStratumId,ToStateClassId,ToAgeMin,"
        "ToStockTypeId,TransitionGroupId,StateAttributeTypeId,FlowTypeId,TargetType,"
        "Multiplier,TransferToStratumId,TransferToSecondaryStratumId,"
        "TransferToTertiaryStratumId,TransferToStateClassId,TransferToAgeMin\n"
        ",,,,,,,Biomass,,,,Atmosphere,Aquaculture_expansion [Type],,Emission,,0.1,,,,,\n"
    )
    rules = load_flow_pathways(p)
    assert len(rules) == 1
    assert rules[0].transition_group == "Aquaculture_expansion"
    assert rules[0].multiplier == 0.1


# ── init_stocks ───────────────────────────────────────────────────────────────

def test_init_stocks_unlinked_starts_at_zero():
    stocks = init_stocks(
        stock_types=["Biomass"], shape=(3, 3),
        initial_links={}, state_attr_rules=[], age_map=None,
    )
    assert np.all(stocks["Biomass"] == 0)

def test_init_stocks_linked_uses_state_attribute():
    rules = [StateAttributeValueRule("Initial_biomass_carbon", None, None, None, 50.0)]
    stocks = init_stocks(
        stock_types=["Biomass"], shape=(2, 2),
        initial_links={"Biomass": "Initial_biomass_carbon"},
        state_attr_rules=rules, age_map=None,
    )
    assert np.all(stocks["Biomass"] == 50.0)


# ── run_flows_for_timestep: NPP accumulation (Alongi-grounded) ───────────────

@pytest.fixture
def classes():
    return {
        1: StateClass(1, "Mangrove",    "Mangrove:All",    (255,0,100,0)),
        2: StateClass(2, "Aquaculture", "Aquaculture:All", (255,255,0,255)),
    }

def test_npp_accumulation_matches_alongi_rate(classes):
    shape = (1, 1)
    age_map = np.array([[50]], dtype=np.uint16)
    lulc_map = np.array([[1]], dtype=np.uint8)
    attr_rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    stocks = {"Atmosphere": np.zeros(shape, dtype=np.float32),
              "Biomass": np.zeros(shape, dtype=np.float32)}
    pathway = FlowPathwayRule(
        from_state_class="Mangrove", from_age_min=None, from_stock_type="Atmosphere",
        to_state_class=None, to_age_min=None, to_stock_type="Biomass",
        transition_group=None, state_attribute="NPP", flow_type="NPP",
        target_type="Flow", multiplier=1.0,
    )
    for year in range(2022, 2032):
        stocks, _ = run_flows_for_timestep(
            stocks, [pathway], {"NPP": 1}, [], age_map, attr_rules,
            classes, year, {}, lulc_map=lulc_map,
        )
    assert abs(stocks["Biomass"][0, 0] - 10 * 18.4) < 0.01

def test_npp_state_attribute_source_not_clipped_by_zero_from_stock(classes):
    """
    Regression test: when a flow is sourced from a State Attribute, it
    must NOT be clipped to the From-Stock's current value — that stock
    is a notional/external source, not a depletable pool. This bug
    previously caused all NPP flows to silently produce zero.
    """
    shape = (1, 1)
    age_map = np.array([[50]], dtype=np.uint16)
    lulc_map = np.array([[1]], dtype=np.uint8)
    attr_rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    stocks = {"Atmosphere": np.zeros(shape, dtype=np.float32),
              "Biomass": np.zeros(shape, dtype=np.float32)}
    pathway = FlowPathwayRule(
        "Mangrove", None, "Atmosphere", None, None, "Biomass",
        None, "NPP", "NPP", "Flow", 1.0,
    )
    stocks, _ = run_flows_for_timestep(
        stocks, [pathway], {"NPP": 1}, [], age_map, attr_rules,
        classes, 2022, {}, lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(18.4, abs=0.01)


# ── from_state_class gating: critical regression tests ───────────────────────

def test_from_state_class_restricts_flow_to_matching_cells_only(classes):
    """
    Regression test: an automatic flow with FromStateClassId set must
    ONLY apply to cells currently in that class. Previously the gate
    was never applied at all, causing flows to fire on every cell
    regardless of land cover.
    """
    shape = (2, 2)
    lulc_map = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    age_map = np.full(shape, 30, dtype=np.uint16)
    attr_rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    stocks = {"Atmosphere": np.zeros(shape, dtype=np.float32),
              "Biomass": np.zeros(shape, dtype=np.float32)}
    pathway = FlowPathwayRule(
        from_state_class="Mangrove", from_age_min=None, from_stock_type="Atmosphere",
        to_state_class=None, to_age_min=None, to_stock_type="Biomass",
        transition_group=None, state_attribute="NPP", flow_type="NPP",
        target_type="Flow", multiplier=1.0,
    )
    stocks, _ = run_flows_for_timestep(
        stocks, [pathway], {"NPP": 1}, [], age_map, attr_rules,
        classes, 2022, {}, lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(18.4, abs=0.01)
    assert stocks["Biomass"][0, 1] == pytest.approx(18.4, abs=0.01)
    assert stocks["Biomass"][1, 0] == 0.0
    assert stocks["Biomass"][1, 1] == 0.0

def test_from_state_class_matches_full_name_label(classes):
    """
    Regression test: FromStateClassId in real CSVs uses the full
    ST-Sim label format ('Mangrove:All'), not just the short name
    ('Mangrove'). Previously only the short name was indexed, causing
    full-label matches to silently fail.
    """
    shape = (1, 1)
    lulc_map = np.array([[1]], dtype=np.uint8)
    age_map = np.array([[30]], dtype=np.uint16)
    attr_rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    stocks = {"Atmosphere": np.zeros(shape, dtype=np.float32),
              "Biomass": np.zeros(shape, dtype=np.float32)}
    pathway = FlowPathwayRule(
        from_state_class="Mangrove:All",
        from_age_min=None, from_stock_type="Atmosphere",
        to_state_class=None, to_age_min=None, to_stock_type="Biomass",
        transition_group=None, state_attribute="NPP", flow_type="NPP",
        target_type="Flow", multiplier=1.0,
    )
    stocks, _ = run_flows_for_timestep(
        stocks, [pathway], {"NPP": 1}, [], age_map, attr_rules,
        classes, 2022, {}, lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(18.4, abs=0.01)

def test_unmatched_from_state_class_yields_zero_eligible(classes):
    shape = (1, 1)
    lulc_map = np.array([[1]], dtype=np.uint8)
    age_map = np.array([[30]], dtype=np.uint16)
    attr_rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    stocks = {"Atmosphere": np.zeros(shape, dtype=np.float32),
              "Biomass": np.zeros(shape, dtype=np.float32)}
    pathway = FlowPathwayRule(
        from_state_class="NonexistentClass", from_age_min=None,
        from_stock_type="Atmosphere", to_state_class=None, to_age_min=None,
        to_stock_type="Biomass", transition_group=None, state_attribute="NPP",
        flow_type="NPP", target_type="Flow", multiplier=1.0,
    )
    stocks, records = run_flows_for_timestep(
        stocks, [pathway], {"NPP": 1}, [], age_map, attr_rules,
        classes, 2022, {}, lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == 0.0
    assert len(records) == 0


# ── Emission flow (transition-triggered) ──────────────────────────────────────

def test_emission_on_transition(classes):
    shape = (1, 1)
    age_map = np.array([[50]], dtype=np.uint16)
    lulc_map = np.array([[2]], dtype=np.uint8)
    stocks = {"Biomass": np.full(shape, 100.0, dtype=np.float32),
              "Atmosphere": np.zeros(shape, dtype=np.float32)}
    pathway = FlowPathwayRule(
        from_state_class=None, from_age_min=None, from_stock_type="Biomass",
        to_state_class=None, to_age_min=None, to_stock_type="Atmosphere",
        transition_group="Aquaculture_expansion", state_attribute=None,
        flow_type="Emission", target_type="Flow", multiplier=0.9,
    )
    year_transitions = [TransitionRecord(2032, 0, 0, 1, 2, "Aquaculture_expansion")]
    stocks, records = run_flows_for_timestep(
        stocks, [pathway], {"Emission": 1}, year_transitions, age_map,
        [], classes, 2032, {}, lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(10.0, abs=0.01)
    assert stocks["Atmosphere"][0, 0] == pytest.approx(90.0, abs=0.01)
    assert len(records) == 1
    assert records[0].flow_type == "Emission"

def test_emission_only_fires_on_cells_where_transition_occurred(classes):
    shape = (1, 2)
    age_map = np.full(shape, 50, dtype=np.uint16)
    lulc_map = np.array([[2, 1]], dtype=np.uint8)
    stocks = {"Biomass": np.full(shape, 100.0, dtype=np.float32),
              "Atmosphere": np.zeros(shape, dtype=np.float32)}
    pathway = FlowPathwayRule(
        None, None, "Biomass", None, None, "Atmosphere",
        "Aquaculture_expansion", None, "Emission", "Flow", 0.9,
    )
    year_transitions = [TransitionRecord(2032, 0, 0, 1, 2, "Aquaculture_expansion")]
    stocks, _ = run_flows_for_timestep(
        stocks, [pathway], {"Emission": 1}, year_transitions, age_map,
        [], classes, 2032, {}, lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(10.0, abs=0.01)
    assert stocks["Biomass"][0, 1] == 100.0


# ── Target Type modes ──────────────────────────────────────────────────────────

def test_to_stock_target_type(classes):
    shape = (1, 1)
    age_map = np.array([[10]], dtype=np.uint16)
    lulc_map = np.array([[1]], dtype=np.uint8)
    stocks = {"Atmosphere": np.zeros(shape, dtype=np.float32),
              "Biomass": np.full(shape, 100.0, dtype=np.float32)}
    pathway = FlowPathwayRule(
        None, None, "Atmosphere", None, None, "Biomass",
        None, None, "Growth", "ToStock", 1.5,
    )
    stocks, _ = run_flows_for_timestep(
        stocks, [pathway], {"Growth": 1}, [], age_map, [],
        classes, 2022, {}, lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(150.0, abs=0.01)

def test_from_stock_target_type_matches_doc_example(classes):
    shape = (1, 1)
    age_map = np.array([[10]], dtype=np.uint16)
    lulc_map = np.array([[1]], dtype=np.uint8)
    stocks = {"Biomass": np.full(shape, 100.0, dtype=np.float32),
              "Atmosphere": np.zeros(shape, dtype=np.float32)}
    pathway = FlowPathwayRule(
        None, None, "Biomass", None, None, "Atmosphere",
        None, None, "Decay", "FromStock", 0.8,
    )
    stocks, _ = run_flows_for_timestep(
        stocks, [pathway], {"Decay": 1}, [], age_map, [],
        classes, 2022, {}, lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(80.0, abs=0.01)


# ── Flow Order sequencing ──────────────────────────────────────────────────────

def test_flow_order_sequencing_npp_before_emission(classes):
    shape = (1, 1)
    age_map = np.array([[50]], dtype=np.uint16)
    lulc_map = np.array([[1]], dtype=np.uint8)
    attr_rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    stocks = {"Atmosphere": np.zeros(shape, dtype=np.float32),
              "Biomass": np.full(shape, 184.0, dtype=np.float32)}

    npp = FlowPathwayRule(
        "Mangrove", None, "Atmosphere", None, None, "Biomass",
        None, "NPP", "NPP", "Flow", 1.0,
    )
    emission = FlowPathwayRule(
        None, None, "Biomass", None, None, "Atmosphere",
        "Aquaculture_expansion", None, "Emission", "Flow", 0.9,
    )
    year_transitions = [TransitionRecord(2032, 0, 0, 1, 2, "Aquaculture_expansion")]

    stocks, _ = run_flows_for_timestep(
        stocks, [npp, emission], {"NPP": 1, "Emission": 2},
        year_transitions, age_map, attr_rules, classes, 2032, {},
        lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(20.24, abs=0.01)

def test_flow_order_reversed_gives_different_result(classes):
    shape = (1, 1)
    age_map = np.array([[50]], dtype=np.uint16)
    lulc_map = np.array([[1]], dtype=np.uint8)
    attr_rules = [StateAttributeValueRule("NPP", None, None, None, 18.4)]
    stocks = {"Atmosphere": np.zeros(shape, dtype=np.float32),
              "Biomass": np.full(shape, 184.0, dtype=np.float32)}

    npp = FlowPathwayRule(
        "Mangrove", None, "Atmosphere", None, None, "Biomass",
        None, "NPP", "NPP", "Flow", 1.0,
    )
    emission = FlowPathwayRule(
        None, None, "Biomass", None, None, "Atmosphere",
        "Aquaculture_expansion", None, "Emission", "Flow", 0.9,
    )
    year_transitions = [TransitionRecord(2032, 0, 0, 1, 2, "Aquaculture_expansion")]

    stocks, _ = run_flows_for_timestep(
        stocks, [npp, emission], {"NPP": 2, "Emission": 1},
        year_transitions, age_map, attr_rules, classes, 2032, {},
        lulc_map=lulc_map,
    )
    assert stocks["Biomass"][0, 0] == pytest.approx(36.8, abs=0.01)
