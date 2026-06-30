"""
tests/test_targets.py  —  v3.1
Unit tests for Transition Targets (core/targets.py + io/csv_loader.py).
"""

import pytest
import numpy as np

from strategicc.core.targets import (
    resolve_targets_per_timestep,
    scale_probability_to_target,
    target_to_patch_budget,
)
from strategicc.core.patches import grow_patches_for_group
from strategicc.io.csv_loader import TransitionTargetRule, load_transition_targets


# ── resolve_targets_per_timestep: forward-fill semantics ─────────────────────

def test_basic_forward_fill():
    rules = [
        TransitionTargetRule(group="Fire", timestep=0, amount=50.0),
        TransitionTargetRule(group="Fire", timestep=5, amount=100.0),
    ]
    resolved = resolve_targets_per_timestep(rules, n_timesteps=10)
    assert resolved[0]["Fire"] == 50.0
    assert resolved[4]["Fire"] == 50.0
    assert resolved[5]["Fire"] == 100.0
    assert resolved[9]["Fire"] == 100.0

def test_target_turned_off():
    rules = [
        TransitionTargetRule(group="Fire", timestep=0, amount=50.0),
        TransitionTargetRule(group="Fire", timestep=5, amount=None),
    ]
    resolved = resolve_targets_per_timestep(rules, n_timesteps=10)
    assert resolved[4]["Fire"] == 50.0
    assert resolved[5]["Fire"] is None
    assert resolved[9]["Fire"] is None

def test_blank_timestep_applies_from_start():
    rules = [TransitionTargetRule(group="Fire", timestep=None, amount=75.0)]
    resolved = resolve_targets_per_timestep(rules, n_timesteps=5)
    for t in range(5):
        assert resolved[t]["Fire"] == 75.0

def test_no_rules_for_group_means_no_target():
    resolved = resolve_targets_per_timestep([], n_timesteps=5)
    for t in range(5):
        assert resolved[t] == {}

def test_multiple_groups_independent():
    rules = [
        TransitionTargetRule(group="Fire",  timestep=0, amount=50.0),
        TransitionTargetRule(group="Flood", timestep=2, amount=20.0),
    ]
    resolved = resolve_targets_per_timestep(rules, n_timesteps=5)
    assert resolved[0]["Fire"] == 50.0
    assert resolved[0].get("Flood") is None
    assert resolved[2]["Flood"] == 20.0

def test_out_of_order_rules_still_sort_correctly():
    rules = [
        TransitionTargetRule(group="Fire", timestep=5, amount=100.0),
        TransitionTargetRule(group="Fire", timestep=0, amount=50.0),
    ]
    resolved = resolve_targets_per_timestep(rules, n_timesteps=10)
    assert resolved[0]["Fire"] == 50.0
    assert resolved[5]["Fire"] == 100.0


# ── scale_probability_to_target ────────────────────────────────────────────

def test_scale_matches_official_doubling_example():
    """target=100ha, expected=50ha -> probabilities x2."""
    shape = (10, 10)
    p_eff = np.full(shape, 0.5, dtype=np.float32)
    eligible = np.ones(shape, dtype=bool)
    scaled = scale_probability_to_target(p_eff, eligible, target_area=100.0, px_area=1.0)
    assert scaled[eligible].sum() == pytest.approx(100.0, rel=0.01)

def test_scale_clips_to_valid_probability_range():
    p_eff = np.full((5, 5), 0.01, dtype=np.float32)
    eligible = np.ones((5, 5), dtype=bool)
    scaled = scale_probability_to_target(p_eff, eligible, target_area=1000.0, px_area=1.0)
    assert scaled.max() <= 1.0
    assert scaled.min() >= 0.0

def test_scale_zero_expected_returns_unchanged():
    p_eff = np.zeros((5, 5), dtype=np.float32)
    eligible = np.ones((5, 5), dtype=bool)
    scaled = scale_probability_to_target(p_eff, eligible, target_area=50.0, px_area=1.0)
    assert np.array_equal(scaled, p_eff)

def test_scale_no_eligible_cells_returns_unchanged():
    p_eff = np.full((5, 5), 0.5, dtype=np.float32)
    eligible = np.zeros((5, 5), dtype=bool)
    scaled = scale_probability_to_target(p_eff, eligible, target_area=50.0, px_area=1.0)
    assert np.array_equal(scaled, p_eff)


# ── target_to_patch_budget ─────────────────────────────────────────────────

def test_target_to_patch_budget_basic():
    assert target_to_patch_budget(target_area=50.0, px_area=2.0) == 25.0

def test_target_to_patch_budget_zero_px_area():
    assert target_to_patch_budget(target_area=50.0, px_area=0.0) == 0.0

def test_target_to_patch_budget_negative_target_clips_zero():
    assert target_to_patch_budget(target_area=-10.0, px_area=1.0) == 0.0


# ── grow_patches_for_group: budget_override integration ──────────────────────

def test_patches_budget_override_replaces_p_eff_budget():
    rng = np.random.default_rng(5)
    shape = (40, 40)
    p_eff = np.full(shape, 0.001, dtype=np.float32)
    eligible = np.ones(shape, dtype=bool)
    bins = [(0, 3, 0.5), (3, 8, 0.5)]

    fired = grow_patches_for_group(
        p_eff, eligible, bins, px_area_ha=1.0, rng=rng,
        budget_override=50.0,
    )
    assert fired.sum() > 10

def test_patches_zero_p_eff_with_budget_override_still_fires():
    """Critical edge case: zero base probability, purely target-driven."""
    rng = np.random.default_rng(9)
    shape = (30, 30)
    p_eff = np.zeros(shape, dtype=np.float32)
    eligible = np.ones(shape, dtype=bool)
    bins = [(0, 5, 1.0)]

    fired = grow_patches_for_group(
        p_eff, eligible, bins, px_area_ha=1.0, rng=rng,
        budget_override=20.0,
    )
    assert fired.sum() > 0, (
        "Patches should still grow via uniform seed fallback when p_eff is "
        "all-zero but a target budget is supplied"
    )
    assert abs(fired.sum() - 20.0) <= 20.0 * 0.3

def test_patches_without_override_uses_p_eff_as_before():
    """Backward compatibility: omitting budget_override preserves v2.5 behaviour."""
    rng = np.random.default_rng(7)
    shape = (50, 50)
    p_eff = np.full(shape, 0.05, dtype=np.float32)
    eligible = np.ones(shape, dtype=bool)
    bins = [(0, 3, 0.5), (3, 10, 0.5)]

    fired = grow_patches_for_group(p_eff, eligible, bins, 1.0, rng)
    budget = p_eff[eligible].sum()
    assert abs(fired.sum() - budget) <= budget * 0.2


# ── load_transition_targets CSV parsing ───────────────────────────────────────

def test_load_transition_targets_basic(tmp_path):
    p = tmp_path / "TransitionTargets.csv"
    p.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,Amount,DistributionType,DistributionFrequencyId,"
        "DistributionSD,DistributionMin,DistributionMax\n"
        ",0,,,,Aquaculture_expansion [Type],50,,,,,\n"
    )
    rules = load_transition_targets(p)
    assert len(rules) == 1
    assert rules[0].group == "Aquaculture_expansion"
    assert rules[0].timestep == 0
    assert rules[0].amount == 50.0

def test_load_transition_targets_blank_amount_means_off(tmp_path):
    p = tmp_path / "TransitionTargets.csv"
    p.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,Amount,DistributionType,DistributionFrequencyId,"
        "DistributionSD,DistributionMin,DistributionMax\n"
        ",3,,,,Fire,,,,,,\n"
    )
    rules = load_transition_targets(p)
    assert len(rules) == 1
    assert rules[0].amount is None

def test_load_transition_targets_skips_blank_group(tmp_path):
    p = tmp_path / "TransitionTargets.csv"
    p.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,Amount,DistributionType,DistributionFrequencyId,"
        "DistributionSD,DistributionMin,DistributionMax\n"
        ",0,,,,,50,,,,,\n"
    )
    rules = load_transition_targets(p)
    assert len(rules) == 0
