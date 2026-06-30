"""
strategicc/core/patches.py  —  v2.5
--------------------------------------
Patch-growing mechanic for transition groups with a historical size
distribution (TransitionSizeDistribution.csv).

Replaces independent-cell probability firing with discrete spatial events:
each "patch" is grown via 8-connected BFS from a seed cell until a sampled
target size is reached, consuming budget from the group's total effective
probability mass for the timestep.

Design
------
- total_budget = sum(p_eff) over all eligible, unfired cells for this group
  this timestep — same probability mass the independent-cell mechanic would
  have consumed, just reshaped into clusters instead of scattered pixels.
- Seed selection is weighted by p_eff (which already encodes spatial
  multiplier + adjacency), so patches preferentially originate in
  high-suitability areas — consistent with how independent firing already
  favours those cells.
- Growth is constrained to STAY within eligible, unfired cells — a patch
  cannot grow into cells that don't qualify for this transition.
- If a patch cannot reach its sampled size (boundary or eligibility limits
  growth), it fires whatever it could grow and the shortfall is simply lost
  for that event (matches real-world disturbance behaviour: a fire can't
  burn into water).
"""

from __future__ import annotations
import numpy as np
from collections import deque


def sample_patch_size_ha(
    bins: list[tuple[float, float, float]],
    rng:  np.random.Generator,
) -> float:
    """
    Sample one patch size in hectares from a group's cumulative bin table.

    Parameters
    ----------
    bins : list of (min_area_ha, max_area_ha, probability) from
           group_size_bins() — probabilities should sum to ~1.0
    rng  : numpy Generator

    Returns
    -------
    sampled area in hectares (uniform within the chosen bin)
    """
    probs = np.array([b[2] for b in bins])
    probs = probs / probs.sum()
    idx   = rng.choice(len(bins), p=probs)
    lo, hi, _ = bins[idx]
    return float(rng.uniform(lo, hi)) if hi > lo else lo


def grow_patch(
    seed_r:       int,
    seed_c:       int,
    target_cells: int,
    eligible:     np.ndarray,
    claimed:      np.ndarray,
) -> list[tuple[int, int]]:
    """
    Grow a single patch via 8-connected BFS from a seed cell.

    Parameters
    ----------
    seed_r, seed_c : starting cell coordinates
    target_cells   : number of cells to grow to (patch stops early if it
                     runs out of eligible neighbours)
    eligible       : bool array — cells that qualify for this transition
    claimed        : bool array — cells unavailable to grow into (mutated
                     in place: True = unavailable, set True for claimed cells)

    Returns
    -------
    list of (row, col) tuples belonging to the grown patch
    """
    rows, cols = eligible.shape
    patch: list[tuple[int, int]] = [(seed_r, seed_c)]
    claimed[seed_r, seed_c] = True

    frontier = deque([(seed_r, seed_c)])
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while frontier and len(patch) < target_cells:
        r, c = frontier.popleft()
        for dr, dc in directions:
            if len(patch) >= target_cells:
                break
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if claimed[nr, nc] or not eligible[nr, nc]:
                continue
            claimed[nr, nc] = True
            patch.append((nr, nc))
            frontier.append((nr, nc))

    return patch


def grow_patches_for_group(
    p_eff:           np.ndarray,
    eligible:        np.ndarray,
    size_bins:       list[tuple[float, float, float]],
    px_area_ha:      float,
    rng:             np.random.Generator,
    max_patches:     int = 10_000,
    budget_override: float | None = None,
) -> np.ndarray:
    """
    Grow patches for one transition group until the budget is exhausted.

    Parameters
    ----------
    p_eff            : effective probability array (base x t_mult x adj x
                        spatial) for this group, this timestep — still used
                        for seed-selection weighting even when
                        budget_override is supplied
    eligible         : bool array — cells eligible for this transition
    size_bins        : group's cumulative size bins from group_size_bins()
    px_area_ha       : area per pixel in hectares
    rng              : numpy Generator
    max_patches      : safety cap to prevent runaway loops
    budget_override  : if supplied (v3.1 — Transition Targets), use this
                        cell-count budget directly instead of deriving it
                        from sum(p_eff over eligible cells). Used when a
                        target area is set for a group that also has a
                        size distribution — the target REPLACES the
                        probability-derived budget rather than scaling it,
                        matching the size-distribution target algorithm.

    Returns
    -------
    bool array, shape matching p_eff — True where a cell was claimed by
    a grown patch and should fire this transition.

    Budget semantics
    ----------------
    By default, `total_budget = sum(p_eff over eligible cells)` is the
    EXPECTED number of cells that independent-cell firing would produce
    (since each cell fires with probability p_eff[r,c], the expectation
    of the sum of Bernoulli draws is exactly sum(p_eff)). Patch-growing
    reshapes WHERE that same expected cell count lands spatially, so the
    budget must be consumed in units of CELLS FIRED, not probability mass.
    When budget_override is supplied, that value is used as the cell-count
    budget directly, bypassing the p_eff-derived calculation entirely.
    """
    shape = p_eff.shape
    fired = np.zeros(shape, dtype=bool)

    if budget_override is not None:
        total_budget = float(budget_override)
    else:
        total_budget = float(p_eff[eligible].sum())   # expected cell count

    if total_budget <= 0 or not eligible.any():
        return fired


    consumed = 0.0   # cells fired so far
    claimed  = ~eligible   # True = unavailable to grow into

    n_patches = 0
    while consumed < total_budget and n_patches < max_patches:
        available_mask = eligible & ~claimed
        if not available_mask.any():
            break

        avail_r, avail_c = np.where(available_mask)
        weights = p_eff[avail_r, avail_c]
        w_sum   = weights.sum()
        if w_sum <= 0:
            if budget_override is not None:
                # Purely target-driven group with no underlying p_eff
                # signal (e.g. a managed target with no base probability
                # defined) — fall back to uniform random seed selection
                # among eligible cells rather than refusing to grow.
                weights = np.ones(len(avail_r), dtype=np.float64)
                w_sum = weights.sum()
            else:
                break
        weights = weights / w_sum

        seed_idx = rng.choice(len(avail_r), p=weights)
        seed_r, seed_c = int(avail_r[seed_idx]), int(avail_c[seed_idx])

        patch_size_ha = sample_patch_size_ha(size_bins, rng)
        target_cells  = max(1, int(round(patch_size_ha / px_area_ha)))

        # Don't grow past the remaining budget
        remaining_cells = total_budget - consumed
        target_cells    = min(target_cells, max(1, int(round(remaining_cells))))

        patch = grow_patch(seed_r, seed_c, target_cells, eligible, claimed)

        rs = [r for r, c in patch]
        cs = [c for r, c in patch]
        fired[rs, cs] = True

        consumed += len(patch)   # budget consumed in CELL COUNT, not probability mass
        n_patches += 1

    return fired
