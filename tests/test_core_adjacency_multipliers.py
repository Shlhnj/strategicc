"""
tests/test_core_adjacency_multipliers.py

Unit tests for strategicc.core.adjacency.compute_neighbor_fractions and
strategicc.core.multipliers.sample_transition_multipliers /
describe_multiplier_rules — both were previously untested.
"""

import numpy as np
import pytest

from strategicc.core.adjacency import compute_neighbor_fractions
from strategicc.core.multipliers import (
    sample_transition_multipliers,
    describe_multiplier_rules,
    _sample,
    _sample_empirical,
)
from strategicc.io.csv_loader import TransitionMultiplierRule, DistributionEntry


# ── compute_neighbor_fractions ─────────────────────────────────────────────

def test_neighbor_fractions_uniform_map_all_class_one():
    """A map that is entirely class 1 should give every cell a fraction of
    1.0 for class 1 and 0.0 for every other class."""
    lulc = np.ones((5, 5), dtype=np.uint8)
    fracs = compute_neighbor_fractions(lulc, n_classes=3)

    assert fracs.shape == (5, 5, 4)
    assert np.allclose(fracs[:, :, 1], 1.0)
    assert np.allclose(fracs[:, :, 2], 0.0)
    assert np.allclose(fracs[:, :, 3], 0.0)


def test_neighbor_fractions_center_cell_checkerboard():
    """3x3 map, center is class 1, all 8 neighbours are class 2:
    the center cell's fraction for class 2 must be 1.0."""
    lulc = np.array([
        [2, 2, 2],
        [2, 1, 2],
        [2, 2, 2],
    ], dtype=np.uint8)
    fracs = compute_neighbor_fractions(lulc, n_classes=2)

    assert fracs[1, 1, 2] == pytest.approx(1.0)
    assert fracs[1, 1, 1] == pytest.approx(0.0)


def test_neighbor_fractions_corner_cell_has_three_neighbors():
    """A corner cell in a 3x3 grid only has 3 valid neighbours; fractions
    must still sum to <= 1 and reflect only those 3 neighbours."""
    lulc = np.array([
        [1, 2, 2],
        [2, 1, 1],
        [1, 1, 2],
    ], dtype=np.uint8)
    fracs = compute_neighbor_fractions(lulc, n_classes=2)

    top_left = fracs[0, 0, :]
    # Corner (0,0) neighbours are (0,1)=2, (1,0)=2, (1,1)=1 → 3 neighbours
    assert top_left[1:].sum() == pytest.approx(1.0)
    assert top_left[1] == pytest.approx(1 / 3)
    assert top_left[2] == pytest.approx(2 / 3)


def test_neighbor_fractions_sum_never_exceeds_one():
    rng = np.random.default_rng(42)
    lulc = rng.integers(1, 4, size=(10, 10)).astype(np.uint8)
    fracs = compute_neighbor_fractions(lulc, n_classes=3)
    totals = fracs[:, :, 1:].sum(axis=2)
    assert np.all(totals <= 1.0 + 1e-6)
    assert np.all(totals >= 0.0)


# ── multipliers: literal Uniform ────────────────────────────────────────────

def test_sample_uniform_within_bounds():
    rule = TransitionMultiplierRule(group="G1", distribution="Uniform", dist_min=0.5, dist_max=1.5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        val = _sample(rule, rng)
        assert 0.5 <= val <= 1.5


def test_sample_uniform_case_insensitive():
    rule = TransitionMultiplierRule(group="G1", distribution="UNIFORM", dist_min=1.0, dist_max=1.0)
    rng = np.random.default_rng(0)
    assert _sample(rule, rng) == pytest.approx(1.0)


# ── multipliers: named empirical distribution ───────────────────────────────

def test_sample_empirical_respects_weights():
    entry = DistributionEntry(name="G1 Distribution", values=[1.0, 2.0], weights=[0.0, 1.0])
    rng = np.random.default_rng(0)
    # weight on 1.0 is zero, so every draw must be 2.0
    for _ in range(20):
        assert _sample_empirical(entry, rng) == 2.0


def test_sample_empirical_zero_total_weight_raises():
    entry = DistributionEntry(name="G1 Distribution", values=[1.0, 2.0], weights=[0.0, 0.0])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no positive relative"):
        _sample_empirical(entry, rng)


def test_sample_named_distribution_lookup():
    rule = TransitionMultiplierRule(group="G1", distribution="G1 Distribution", dist_min=0, dist_max=0)
    distributions = {
        "G1 Distribution": DistributionEntry(name="G1 Distribution", values=[3.0], weights=[1.0]),
    }
    rng = np.random.default_rng(0)
    assert _sample(rule, rng, distributions) == 3.0


def test_sample_unsupported_distribution_raises():
    rule = TransitionMultiplierRule(group="G1", distribution="Normal", dist_min=0, dist_max=1)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unsupported DistributionType"):
        _sample(rule, rng, distributions=None)


def test_sample_named_distribution_missing_from_table_raises():
    rule = TransitionMultiplierRule(group="G1", distribution="Missing Distribution", dist_min=0, dist_max=1)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unsupported DistributionType"):
        _sample(rule, rng, distributions={})


# ── sample_transition_multipliers ───────────────────────────────────────────

def test_sample_transition_multipliers_one_per_group():
    rules = [
        TransitionMultiplierRule(group="A", distribution="Uniform", dist_min=1.0, dist_max=1.0),
        TransitionMultiplierRule(group="B", distribution="Uniform", dist_min=2.0, dist_max=2.0),
    ]
    rng = np.random.default_rng(0)
    result = sample_transition_multipliers(rules, rng)
    assert result == {"A": 1.0, "B": 2.0}


def test_sample_transition_multipliers_empty_rules():
    rng = np.random.default_rng(0)
    assert sample_transition_multipliers([], rng) == {}


# ── describe_multiplier_rules ───────────────────────────────────────────────

def test_describe_multiplier_rules_empty(capsys):
    describe_multiplier_rules([])
    out = capsys.readouterr().out
    assert "No transition multiplier rules loaded" in out


def test_describe_multiplier_rules_prints_each_rule(capsys):
    rules = [
        TransitionMultiplierRule(group="Agriculture_expansion", distribution="Uniform", dist_min=0.5, dist_max=1.5),
    ]
    describe_multiplier_rules(rules)
    out = capsys.readouterr().out
    assert "Agriculture_expansion" in out
    assert "Uniform" in out
