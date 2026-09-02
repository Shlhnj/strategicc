"""
tests/test_stockflow_engine.py

Unit tests for the remaining uncovered branches of strategicc/stockflow/engine.py
(77% -> target near-100%): init_stocks()'s no-age-map/class-scoped and
pure-wildcard branches, and run_flows_for_timestep()'s pathway-skip
branches (unknown stock type, no eligible cells, unresolved from-class,
age gate), ToStock/FromStock target types, per-class flow breakdown, and
the class-to-class stock carryover recording.
"""

import numpy as np
import pytest

from strategicc.stockflow.engine import (
    init_stocks,
    run_flows_for_timestep,
    build_age_attribute_cache,
)
from strategicc.stockflow.csv_loader import FlowPathwayRule, StateAttributeValueRule
from strategicc.io.csv_loader import StateClass
from strategicc.core.transitions import TransitionRecord


@pytest.fixture
def classes():
    return {
        1: StateClass(id=1, name="Mangrove", full_name="Mangrove:All", color=(255, 0, 128, 0)),
        2: StateClass(id=2, name="Aquaculture", full_name="Aquaculture:All", color=(255, 255, 0, 255)),
    }


# ── init_stocks: no age_map, WITH class_map (per-class lookup at age=0) ────

def test_init_stocks_no_age_map_with_class_map(classes):
    class_map = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    rules = [StateAttributeValueRule("NPP", "Mangrove:All", 0, 999, 5.0)]
    stocks = init_stocks(
        stock_types=["Biomass"], shape=(2, 2), initial_links={"Biomass": "NPP"},
        state_attr_rules=rules, age_map=None, class_map=class_map, classes=classes,
        px_area_ha=2.0,
    )
    assert np.all(stocks["Biomass"][class_map == 1] == 10.0)   # 5.0 * px_area_ha
    assert np.all(stocks["Biomass"][class_map == 2] == 0.0)    # no matching rule


def test_init_stocks_no_age_map_class_id_not_in_classes(classes):
    """A class_id present in class_map but absent from `classes` must be
    skipped (continue) rather than raising."""
    class_map = np.array([[1, 99]], dtype=np.uint8)   # 99 unknown
    rules = [StateAttributeValueRule("NPP", "Mangrove:All", 0, 999, 5.0)]
    stocks = init_stocks(
        stock_types=["Biomass"], shape=(1, 2), initial_links={"Biomass": "NPP"},
        state_attr_rules=rules, age_map=None, class_map=class_map, classes=classes,
        px_area_ha=1.0,
    )
    assert stocks["Biomass"][0, 0] == 5.0
    assert stocks["Biomass"][0, 1] == 0.0   # skipped, left at 0


# ── init_stocks: no age_map, no class_map (pure wildcard) ──────────────────

def test_init_stocks_no_age_no_class_map_wildcard():
    rules = [StateAttributeValueRule("NPP", None, 0, 999, 3.0)]   # wildcard rule
    stocks = init_stocks(
        stock_types=["Biomass"], shape=(2, 2), initial_links={"Biomass": "NPP"},
        state_attr_rules=rules, age_map=None, class_map=None, classes=None,
        px_area_ha=2.0,
    )
    assert np.all(stocks["Biomass"] == 6.0)   # 3.0 * px_area_ha


def test_init_stocks_no_age_no_class_map_no_matching_rule():
    stocks = init_stocks(
        stock_types=["Biomass"], shape=(2, 2), initial_links={"Biomass": "NPP"},
        state_attr_rules=[], age_map=None, class_map=None, classes=None,
        px_area_ha=1.0,
    )
    assert np.all(stocks["Biomass"] == 0.0)


# ── run_flows_for_timestep: pathway skip branches ───────────────────────────

def _rule(**overrides):
    base = dict(
        from_state_class=None, from_age_min=None, from_stock_type="Biomass",
        to_state_class=None, to_age_min=None, to_stock_type="Atmosphere",
        transition_group=None, state_attribute=None, flow_type="Emission",
        target_type="Flow", multiplier=0.5,
    )
    base.update(overrides)
    return FlowPathwayRule(**base)


def test_run_flows_skips_pathway_with_unknown_stock_type(classes):
    stocks = {"Biomass": np.ones((2, 2), dtype=np.float32)}
    rule = _rule(to_stock_type="GhostStock")
    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[rule], flow_order={}, year_transitions=[],
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={},
    )
    assert records == []


def test_run_flows_transition_triggered_no_fired_cells_skipped(classes):
    stocks = {"Biomass": np.ones((2, 2), dtype=np.float32),
              "Atmosphere": np.zeros((2, 2), dtype=np.float32)}
    rule = _rule(transition_group="Aquaculture_expansion")
    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[rule], flow_order={}, year_transitions=[],  # nothing fired
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={},
    )
    assert records == []


def test_run_flows_unresolved_from_state_class_yields_zero_eligible(classes):
    lulc_map = np.ones((2, 2), dtype=np.uint8)
    stocks = {"Biomass": np.ones((2, 2), dtype=np.float32),
              "Atmosphere": np.zeros((2, 2), dtype=np.float32)}
    rule = _rule(from_state_class="NotARealClass")
    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[rule], flow_order={}, year_transitions=[],
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={}, lulc_map=lulc_map,
    )
    assert records == []


def test_run_flows_age_gate_filters_cells(classes):
    lulc_map = np.ones((2, 2), dtype=np.uint8)
    age_map = np.array([[0, 10], [10, 10]], dtype=np.uint16)
    stocks = {"Biomass": np.full((2, 2), 4.0, dtype=np.float32),
              "Atmosphere": np.zeros((2, 2), dtype=np.float32)}
    rule = _rule(from_age_min=5, multiplier=0.5)
    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[rule], flow_order={}, year_transitions=[],
        age_map=age_map, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={}, lulc_map=lulc_map,
    )
    assert len(records) == 1
    # Cell (0,0) has age 0 < 5 -> excluded from the flow.
    assert updated["Biomass"][0, 0] == 4.0
    assert updated["Biomass"][0, 1] == pytest.approx(2.0)   # 4 - 4*0.5


# ── ToStock / FromStock target types ────────────────────────────────────────

def test_run_flows_to_stock_target_type(classes):
    stocks = {"Biomass": np.full((2, 2), 10.0, dtype=np.float32),
              "Soil": np.full((2, 2), 2.0, dtype=np.float32)}
    rule = _rule(from_stock_type="Biomass", to_stock_type="Soil",
                 target_type="ToStock", multiplier=1.5)
    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[rule], flow_order={}, year_transitions=[],
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={},
    )
    # target_to = 2.0 * 1.5 = 3.0; flow_amount = 3.0 - 2.0 = 1.0 added to Soil
    assert np.all(updated["Soil"] == pytest.approx(3.0))
    assert len(records) == 1
    assert records[0].flow_type == "Emission"


def test_run_flows_from_stock_target_type(classes):
    stocks = {"Biomass": np.full((2, 2), 10.0, dtype=np.float32),
              "Atmosphere": np.zeros((2, 2), dtype=np.float32)}
    rule = _rule(target_type="FromStock", multiplier=0.5)
    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[rule], flow_order={}, year_transitions=[],
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={},
    )
    # target_from = 10*0.5=5; flow_amount = 10-5=5
    assert np.all(updated["Biomass"] == pytest.approx(5.0))
    assert len(records) == 1


# ── Per-class flow breakdown (lulc_map present) ─────────────────────────────

def test_run_flows_records_per_class_breakdown(classes):
    lulc_map = np.array([[1, 2]], dtype=np.uint8)
    stocks = {"Biomass": np.array([[10.0, 20.0]], dtype=np.float32),
              "Atmosphere": np.zeros((1, 2), dtype=np.float32)}
    rule = _rule(multiplier=0.5)
    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[rule], flow_order={}, year_transitions=[],
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={}, lulc_map=lulc_map,
    )
    assert len(records) == 1
    assert records[0].by_class == {"Mangrove": pytest.approx(5.0), "Aquaculture": pytest.approx(10.0)}


# ── State-attribute-sourced flow with age_map (build_age_attribute_cache) ──

def test_run_flows_state_attribute_sourced_with_age_map(classes):
    lulc_map = np.ones((1, 2), dtype=np.uint8)
    age_map = np.array([[5, 15]], dtype=np.uint16)
    attr_rules = [StateAttributeValueRule("NPP", "Mangrove:All", 0, 10, 2.0),
                  StateAttributeValueRule("NPP", "Mangrove:All", 11, 999, 8.0)]
    stocks = {"Atmosphere": np.zeros((1, 2), dtype=np.float32),
              "Biomass": np.zeros((1, 2), dtype=np.float32)}
    rule = _rule(from_stock_type="Atmosphere", to_stock_type="Biomass",
                 state_attribute="NPP", multiplier=1.0)
    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[rule], flow_order={}, year_transitions=[],
        age_map=age_map, state_attr_rules=attr_rules, classes=classes, year=2022,
        flow_mult_sample={}, lulc_map=lulc_map, px_area_ha=1.0,
    )
    assert len(records) == 1
    assert updated["Biomass"][0, 0] == pytest.approx(2.0)
    assert updated["Biomass"][0, 1] == pytest.approx(8.0)


# ── Class-to-class stock carryover recording ────────────────────────────────

def test_run_flows_class_carryover_recorded(classes):
    lulc_map = np.array([[2]], dtype=np.uint8)   # already Aquaculture (post-transition)
    stocks = {"Biomass": np.array([[7.0]], dtype=np.float32)}
    year_transitions = [TransitionRecord(year=2022, row=0, col=0, from_id=1, to_id=2, group="Aquaculture_expansion")]

    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[], flow_order={}, year_transitions=year_transitions,
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={}, lulc_map=lulc_map,
    )
    flow_types = {(r.from_stock, r.to_stock) for r in records}
    assert ("Biomass", "__ClassTransferOut") in flow_types
    assert ("__ClassTransferIn", "Biomass") in flow_types
    out_rec = next(r for r in records if r.to_stock == "__ClassTransferOut")
    assert out_rec.by_class == {"Mangrove": pytest.approx(7.0)}
    in_rec = next(r for r in records if r.from_stock == "__ClassTransferIn")
    assert in_rec.by_class == {"Aquaculture": pytest.approx(7.0)}


def test_run_flows_class_carryover_skipped_when_zero_stock(classes):
    """A cell with zero stock at the transitioning location contributes
    nothing (nz mask all False -> continue)."""
    lulc_map = np.array([[2]], dtype=np.uint8)
    stocks = {"Biomass": np.array([[0.0]], dtype=np.float32)}
    year_transitions = [TransitionRecord(year=2022, row=0, col=0, from_id=1, to_id=2, group="Aquaculture_expansion")]

    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[], flow_order={}, year_transitions=year_transitions,
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={}, lulc_map=lulc_map,
    )
    assert records == []


def test_run_flows_no_class_change_no_carryover(classes):
    """A transition where from_id == to_id (self-transition) triggers no
    carryover recording."""
    lulc_map = np.array([[1]], dtype=np.uint8)
    stocks = {"Biomass": np.array([[7.0]], dtype=np.float32)}
    year_transitions = [TransitionRecord(year=2022, row=0, col=0, from_id=1, to_id=1, group="G")]

    updated, records = run_flows_for_timestep(
        stocks=stocks, pathways=[], flow_order={}, year_transitions=year_transitions,
        age_map=None, state_attr_rules=[], classes=classes, year=2022,
        flow_mult_sample={}, lulc_map=lulc_map,
    )
    assert records == []
