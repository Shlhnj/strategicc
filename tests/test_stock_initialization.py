"""
tests/test_stock_initialization.py  —  v3.18
------------------------------------------------
Regression tests for init_stocks()/build_age_attribute_cache(). Neither
had any direct test coverage before v3.18 — which is exactly how the
bug these tests guard against shipped silently: State Attribute
Values.csv rows scoped to a specific class (e.g. "Mangrove:All", the
normal way to write this table, since biomass legitimately differs by
ecosystem type) could never match during stock initialization, because
build_age_attribute_cache() always called lookup_state_attribute()
with state_class=None — which only matches WILDCARD rules (blank
StateClassId). Every real class-scoped rule was silently invisible,
and initial stocks came out all-zero regardless of what the CSV said.

Fixed by threading a per-cell class map (+ classes dict, for
full_name resolution — StateClassId in the CSV matches
StateClass.full_name, e.g. "Mangrove:All", not the short .name) through
init_stocks() -> build_age_attribute_cache() -> lookup_state_attribute().
"""

import pytest
import numpy as np

from strategicc.io.csv_loader import StateClass
from strategicc.stockflow.csv_loader import StateAttributeValueRule, FlowPathwayRule, validate_flow_pathways
from strategicc.stockflow.engine import build_age_attribute_cache, init_stocks


@pytest.fixture
def classes():
    return {
        1: StateClass(1, "Mangrove", "Mangrove:All", (0, 0, 0, 0)),
        2: StateClass(2, "Cropland", "Cropland:All", (0, 0, 0, 0)),
    }


@pytest.fixture
def class_scoped_rules():
    """Mirrors real-world State Attribute Values.csv usage: every rule
    is scoped to a specific class, none are wildcard (blank StateClassId).
    This is the normal, correct way to write this table — biomass
    legitimately differs by ecosystem type — and is exactly the pattern
    that triggered the bug."""
    return [
        StateAttributeValueRule("Initial_biomass_carbon", "Mangrove:All", 0, 20, 33.6),
        StateAttributeValueRule("Initial_biomass_carbon", "Mangrove:All", 21, 30, 48.28),
        StateAttributeValueRule("Initial_soil_carbon", "Mangrove:All", None, None, 26.0),
        StateAttributeValueRule("Initial_soil_carbon", "Cropland:All", None, None, 7.0),
    ]


# ── build_age_attribute_cache ─────────────────────────────────────────────────

def test_build_age_attribute_cache_without_class_map_only_matches_wildcards(class_scoped_rules):
    """Backward-compatible path (no class_map/classes given): only
    wildcard rules can match. With purely class-scoped rules (the
    real-world/bug-triggering case), this returns all zeros — this
    documents the OLD (broken, if your data is class-scoped) behavior,
    not an endorsement of it."""
    age_map = np.full((3, 3), 25, dtype=np.int32)
    out = build_age_attribute_cache(age_map, class_scoped_rules, "Initial_biomass_carbon")
    assert (out == 0).all()

def test_build_age_attribute_cache_with_class_map_resolves_class_scoped_rules(classes, class_scoped_rules):
    """The fix: with class_map/classes supplied, class-scoped rules
    actually match, using the correct age bracket per class."""
    class_map = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])  # left cols Mangrove, right col Cropland
    age_map = np.full((3, 3), 25, dtype=np.int32)  # falls in the 21-30 bracket

    out = build_age_attribute_cache(
        age_map, class_scoped_rules, "Initial_biomass_carbon",
        class_map=class_map, classes=classes,
    )
    assert np.allclose(out[class_map == 1], 48.28)
    # Cropland has no Initial_biomass_carbon rule at all -> 0, correctly
    assert (out[class_map == 2] == 0).all()

def test_build_age_attribute_cache_respects_age_brackets_per_class(classes, class_scoped_rules):
    class_map = np.full((2, 2), 1)  # all Mangrove
    age_map = np.array([[10, 25], [10, 25]])  # bracket 0-20 vs 21-30

    out = build_age_attribute_cache(
        age_map, class_scoped_rules, "Initial_biomass_carbon",
        class_map=class_map, classes=classes,
    )
    assert out[0, 0] == pytest.approx(33.6)   # age 10 -> 0-20 bracket
    assert out[0, 1] == pytest.approx(48.28)  # age 25 -> 21-30 bracket

def test_build_age_attribute_cache_unknown_class_id_skipped(classes, class_scoped_rules):
    """A class ID present in class_map but absent from `classes` (e.g.
    stale raster) shouldn't crash — those cells just stay 0."""
    class_map = np.array([[1, 99]])  # 99 not in `classes`
    age_map = np.array([[25, 25]])
    out = build_age_attribute_cache(
        age_map, class_scoped_rules, "Initial_biomass_carbon",
        class_map=class_map, classes=classes,
    )
    assert out[0, 0] == pytest.approx(48.28)
    assert out[0, 1] == 0.0


# ── init_stocks ────────────────────────────────────────────────────────────────

def test_init_stocks_class_scoped_rules_produce_nonzero_stock(classes, class_scoped_rules):
    """The actual regression test for the reported bug: with realistic
    class-scoped State Attribute Values rules, initial stocks must NOT
    come out all-zero when class_map/classes are supplied."""
    class_map = np.array([[1, 1, 2], [1, 1, 2]])
    age_map = np.full((2, 3), 25, dtype=np.int32)
    initial_links = {"AGB": "Initial_biomass_carbon", "Soil": "Initial_soil_carbon"}

    stocks = init_stocks(
        stock_types=["AGB", "Soil"], shape=(2, 3),
        initial_links=initial_links, state_attr_rules=class_scoped_rules,
        age_map=age_map, class_map=class_map, classes=classes,
    )
    assert not (stocks["AGB"] == 0).all(), "AGB stock is all-zero — the bug is back"
    assert np.allclose(stocks["AGB"][class_map == 1], 48.28)
    assert np.allclose(stocks["Soil"][class_map == 1], 26.0)
    assert np.allclose(stocks["Soil"][class_map == 2], 7.0)

def test_init_stocks_without_class_map_stays_backward_compatible(class_scoped_rules):
    """Callers that don't pass class_map/classes (old call signature)
    must not crash — they just keep the old wildcard-only behavior."""
    age_map = np.full((2, 2), 25, dtype=np.int32)
    stocks = init_stocks(
        stock_types=["AGB"], shape=(2, 2),
        initial_links={"AGB": "Initial_biomass_carbon"},
        state_attr_rules=class_scoped_rules, age_map=age_map,
    )
    assert (stocks["AGB"] == 0).all()

def test_init_stocks_no_age_map_still_resolves_class_scoped_rules(classes, class_scoped_rules):
    """Age-independent rule (Initial_soil_carbon has no age_min/age_max
    for Cropland) resolved per-class even without age tracking at all."""
    class_map = np.array([[1, 2]])
    stocks = init_stocks(
        stock_types=["Soil"], shape=(1, 2),
        initial_links={"Soil": "Initial_soil_carbon"},
        state_attr_rules=class_scoped_rules, age_map=None,
        class_map=class_map, classes=classes,
    )
    assert stocks["Soil"][0, 0] == pytest.approx(26.0)  # Mangrove
    assert stocks["Soil"][0, 1] == pytest.approx(7.0)   # Cropland

def test_init_stocks_unlinked_stock_type_stays_zero(classes, class_scoped_rules):
    class_map = np.array([[1, 1]])
    age_map = np.full((1, 2), 25, dtype=np.int32)
    stocks = init_stocks(
        stock_types=["Atmosphere"], shape=(1, 2),
        initial_links={},  # no link for Atmosphere
        state_attr_rules=class_scoped_rules, age_map=age_map,
        class_map=class_map, classes=classes,
    )
    assert (stocks["Atmosphere"] == 0).all()


# ── validate_flow_pathways ────────────────────────────────────────────────────

@pytest.fixture
def valid_pathway(classes):
    return FlowPathwayRule(
        from_state_class="Mangrove:All", from_age_min=None, from_stock_type="Atmosphere",
        to_state_class=None, to_age_min=None, to_stock_type="AGB",
        transition_group=None, state_attribute="NPP", flow_type="GPP",
        target_type="Flow", multiplier=0.7,
    )

def test_validate_flow_pathways_all_valid_no_warnings(classes, class_scoped_rules, valid_pathway, capsys):
    rules_with_npp = class_scoped_rules + [
        StateAttributeValueRule("NPP", None, None, None, 5.0)
    ]
    warnings = validate_flow_pathways(
        [valid_pathway], stock_types=["AGB", "Atmosphere"], flow_types=["GPP"],
        state_attr_rules=rules_with_npp, classes=classes,
    )
    assert warnings == []
    assert "all validation" not in capsys.readouterr().out.lower()

def test_validate_flow_pathways_catches_dangling_state_attribute(classes, class_scoped_rules, valid_pathway):
    """The actual regression test for the real-world bug: a
    StateAttributeTypeId that doesn't exist anywhere must be flagged."""
    bad_pathway = FlowPathwayRule(**{**valid_pathway.__dict__, "state_attribute": "NPP_AG"})
    warnings = validate_flow_pathways(
        [bad_pathway], stock_types=["AGB", "Atmosphere"], flow_types=["GPP"],
        state_attr_rules=class_scoped_rules,  # only has Initial_*/etc, no NPP_AG
        classes=classes,
    )
    assert len(warnings) == 1
    assert "NPP_AG" in warnings[0]
    assert "State Attribute Values.csv" in warnings[0]

def test_validate_flow_pathways_catches_dangling_stock_type(classes, class_scoped_rules, valid_pathway):
    bad_pathway = FlowPathwayRule(**{**valid_pathway.__dict__, "to_stock_type": "Typo"})
    warnings = validate_flow_pathways(
        [bad_pathway], stock_types=["AGB", "Atmosphere"], flow_types=["GPP"],
        state_attr_rules=class_scoped_rules + [StateAttributeValueRule("NPP", None, None, None, 5.0)],
        classes=classes,
    )
    assert any("Typo" in w and "Stock Type.csv" in w for w in warnings)

def test_validate_flow_pathways_catches_dangling_flow_type(classes, class_scoped_rules, valid_pathway):
    bad_pathway = FlowPathwayRule(**{**valid_pathway.__dict__, "flow_type": "Typo"})
    warnings = validate_flow_pathways(
        [bad_pathway], stock_types=["AGB", "Atmosphere"], flow_types=["GPP"],
        state_attr_rules=class_scoped_rules + [StateAttributeValueRule("NPP", None, None, None, 5.0)],
        classes=classes,
    )
    assert any("Typo" in w and "Flow Type.csv" in w for w in warnings)

def test_validate_flow_pathways_catches_dangling_state_class(classes, class_scoped_rules, valid_pathway):
    bad_pathway = FlowPathwayRule(**{**valid_pathway.__dict__, "from_state_class": "Typo:All"})
    warnings = validate_flow_pathways(
        [bad_pathway], stock_types=["AGB", "Atmosphere"], flow_types=["GPP"],
        state_attr_rules=class_scoped_rules + [StateAttributeValueRule("NPP", None, None, None, 5.0)],
        classes=classes,
    )
    assert any("Typo:All" in w and "State Class.csv" in w for w in warnings)
