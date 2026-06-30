"""
tests/test_patches.py  —  v2.5
Unit tests for the patch-growing mechanic (core/patches.py).
"""

import pytest
import numpy as np
from scipy import ndimage

from strategicc.core.patches import (
    sample_patch_size_ha, grow_patch, grow_patches_for_group,
)


# ── sample_patch_size_ha ──────────────────────────────────────────────────────

def test_sample_patch_size_distribution_shape():
    rng = np.random.default_rng(42)
    bins = [(0, 1, 0.3), (1, 5, 0.4), (5, 20, 0.2), (20, 50, 0.1)]
    samples = np.array([sample_patch_size_ha(bins, rng) for _ in range(2000)])
    assert samples.min() >= 0
    assert samples.max() <= 50
    assert 0.20 < (samples <= 1).mean() < 0.40
    assert 0.05 < (samples > 20).mean() < 0.18

def test_sample_patch_size_single_bin():
    rng = np.random.default_rng(1)
    bins = [(5, 5, 1.0)]
    size = sample_patch_size_ha(bins, rng)
    assert size == 5


# ── grow_patch ────────────────────────────────────────────────────────────────

def test_grow_patch_reaches_target_in_open_field():
    eligible = np.ones((20, 20), dtype=bool)
    claimed  = ~eligible
    patch = grow_patch(10, 10, target_cells=15, eligible=eligible, claimed=claimed)
    assert len(patch) == 15

def test_grow_patch_respects_eligible_boundary():
    eligible = np.zeros((10, 10), dtype=bool)
    eligible[3:7, 3:7] = True
    claimed = ~eligible
    patch = grow_patch(5, 5, target_cells=100, eligible=eligible, claimed=claimed)
    assert len(patch) <= 16
    for r, c in patch:
        assert eligible[r, c]

def test_grow_patch_avoids_claimed_cells():
    eligible = np.ones((10, 10), dtype=bool)
    claimed  = ~eligible
    claimed[5, 6] = True
    patch = grow_patch(5, 5, target_cells=5, eligible=eligible, claimed=claimed)
    assert (5, 6) not in patch

def test_grow_patch_single_cell_island():
    eligible = np.zeros((5, 5), dtype=bool)
    eligible[2, 2] = True
    claimed = ~eligible
    patch = grow_patch(2, 2, target_cells=10, eligible=eligible, claimed=claimed)
    assert patch == [(2, 2)]


# ── grow_patches_for_group: budget accounting (regression for the fix) ───────

def test_budget_matches_independent_cell_expectation():
    """
    Core regression test: total fired cells should approximate
    sum(p_eff over eligible cells) — same expectation independent-cell
    Bernoulli firing would produce. Prior to fixing this, budget was
    tracked in probability-mass units instead of cell-count units,
    causing massive over-firing (e.g. 10000 cells fired against a
    budget of 500).
    """
    rng = np.random.default_rng(7)
    shape = (100, 100)
    p_eff = np.full(shape, 0.05, dtype=np.float32)
    eligible = np.ones(shape, dtype=bool)
    bins = [(0, 3, 0.5), (3, 10, 0.5)]

    fired = grow_patches_for_group(p_eff, eligible, bins, px_area_ha=1.0, rng=rng)
    budget = p_eff[eligible].sum()
    n_fired = fired.sum()

    assert n_fired < eligible.sum() * 0.5, (
        "Over-firing regression: patches consumed most of the eligible "
        "region despite a small probability budget"
    )
    assert abs(n_fired - budget) <= budget * 0.20, (
        f"Fired count {n_fired} too far from expected budget {budget:.1f}"
    )

def test_budget_accuracy_across_seeds():
    shape = (50, 50)
    bins  = [(0, 2, 0.4), (2, 8, 0.4), (8, 20, 0.2)]
    for seed in [1, 5, 42, 100]:
        rng   = np.random.default_rng(seed)
        p_eff = np.random.default_rng(seed + 1).uniform(0.01, 0.1, shape).astype(np.float32)
        eligible = np.ones(shape, dtype=bool)

        fired  = grow_patches_for_group(p_eff, eligible, bins, 1.0, rng)
        budget = p_eff[eligible].sum()
        pct_diff = abs(fired.sum() - budget) / budget * 100
        assert pct_diff < 20, f"seed={seed}: budget mismatch {pct_diff:.1f}%"


# ── grow_patches_for_group: spatial clustering ────────────────────────────────

def test_fired_cells_form_clusters_not_scatter():
    """Patches should be spatially clustered, distinguishing them from
    independent-cell firing which produces scattered single pixels."""
    rng = np.random.default_rng(3)
    shape = (60, 60)
    p_eff = np.full(shape, 0.3, dtype=np.float32)
    eligible = np.ones(shape, dtype=bool)
    bins = [(0, 5, 0.5), (5, 15, 0.5)]

    fired = grow_patches_for_group(p_eff, eligible, bins, 1.0, rng)
    labeled, n_components = ndimage.label(fired, structure=np.ones((3, 3)))
    n_fired = fired.sum()

    assert n_components < n_fired, "Expected fewer clusters than fired cells"
    avg_patch_size = n_fired / max(n_components, 1)
    assert avg_patch_size > 1.5, "Patches should average more than 1 cell"


# ── grow_patches_for_group: boundary and edge cases ───────────────────────────

def test_respects_eligible_boundary_at_group_level():
    eligible = np.zeros((20, 20), dtype=bool)
    eligible[5:10, 5:10] = True
    p_eff = np.full((20, 20), 0.8, dtype=np.float32)
    bins  = [(0, 100, 1.0)]

    fired = grow_patches_for_group(p_eff, eligible, bins, px_area_ha=1.0,
                                   rng=np.random.default_rng(9))
    assert fired.sum() <= 25
    for r in range(20):
        for c in range(20):
            if fired[r, c]:
                assert eligible[r, c]

def test_empty_eligible_returns_no_fire():
    shape = (30, 30)
    p_eff = np.full(shape, 0.5, dtype=np.float32)
    eligible = np.zeros(shape, dtype=bool)
    fired = grow_patches_for_group(p_eff, eligible, [(0, 10, 1.0)], 1.0,
                                   np.random.default_rng(1))
    assert fired.sum() == 0

def test_zero_probability_returns_no_fire():
    shape = (30, 30)
    p_eff = np.zeros(shape, dtype=np.float32)
    eligible = np.ones(shape, dtype=bool)
    fired = grow_patches_for_group(p_eff, eligible, [(0, 10, 1.0)], 1.0,
                                   np.random.default_rng(1))
    assert fired.sum() == 0

def test_max_patches_safety_cap():
    shape = (20, 20)
    p_eff = np.full(shape, 1e-6, dtype=np.float32)
    eligible = np.ones(shape, dtype=bool)
    bins = [(0, 1, 1.0)]
    fired = grow_patches_for_group(
        p_eff, eligible, bins, px_area_ha=1.0,
        rng=np.random.default_rng(1), max_patches=5,
    )
    assert fired.sum() >= 0
