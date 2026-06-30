"""
strategicc/core/targets.py  —  v3.1
--------------------------------------
Transition Targets: area-based overrides that replace or scale a
transition group's probability-derived budget for a given timestep.

Two mechanics, branching on whether the group also has a size
distribution (patch-growing, from v2.5):

  No size distribution:
    Probabilities are scaled so the EXPECTED area matches the target:
        scale = target_area / expected_area
        p_eff_scaled = p_eff * scale
    This preserves stochastic variance - the realised area in any one
    iteration will vary around the target, matching independent-cell
    firing behaviour with a renormalised probability.

  With size distribution:
    The target area becomes the patch-growing BUDGET directly (replacing
    sum(p_eff) as used in v2.5's grow_patches_for_group), so the model
    grows patches until as close to the target as possible - much lower
    variance than the probability-scaling approach, by design.

Persistence
-----------
A target set at timestep T applies from T onward for all subsequent
timesteps, until a new record for the same group appears at a later
timestep. A row with amount=None explicitly turns the target off from
that timestep onward (group reverts to its normal probability-derived
budget). resolve_targets_per_timestep() pre-computes this forward-fill
once per run so the engine's per-timestep loop does a simple dict lookup.
"""

from __future__ import annotations
import numpy as np
from strategicc.io.csv_loader import TransitionTargetRule


def resolve_targets_per_timestep(
    rules:       list[TransitionTargetRule],
    n_timesteps: int,
) -> dict[int, dict[str, float | None]]:
    """
    Forward-fill target rules into an explicit per-timestep lookup.

    Parameters
    ----------
    rules       : output of load_transition_targets()
    n_timesteps : total number of simulation timesteps

    Returns
    -------
    dict[timestep_index, dict[group_name, target_amount_or_None]]
    target_amount is None if no target is active for that group at that
    timestep (either never set, or explicitly turned off via a blank-amount
    row). Timestep 0 is the first simulated timestep.
    """
    by_group: dict[str, list[TransitionTargetRule]] = {}
    for r in rules:
        by_group.setdefault(r.group, []).append(r)

    for group in by_group:
        by_group[group].sort(
            key=lambda r: (r.timestep is not None, r.timestep or 0)
        )

    resolved: dict[int, dict[str, float | None]] = {
        t: {} for t in range(n_timesteps)
    }

    for group, group_rules in by_group.items():
        current_amount: float | None = None
        rule_idx = 0

        for t in range(n_timesteps):
            while (
                rule_idx < len(group_rules)
                and (group_rules[rule_idx].timestep is None
                     or group_rules[rule_idx].timestep <= t)
            ):
                current_amount = group_rules[rule_idx].amount
                rule_idx += 1

            resolved[t][group] = current_amount

    return resolved


def scale_probability_to_target(
    p_eff:       np.ndarray,
    eligible:    np.ndarray,
    target_area: float,
    px_area:     float,
) -> np.ndarray:
    """
    Scale p_eff so the EXPECTED fired area matches target_area.

    Parameters
    ----------
    p_eff       : effective probability array for this group, this timestep
    eligible    : bool array - cells eligible for this transition
    target_area : desired expected area (in the engine's configured AREA_UNIT)
    px_area     : area per pixel in the same unit

    Returns
    -------
    scaled p_eff array. If expected area is zero, returns the array
    unchanged - there is nothing to scale, and the caller's eligibility
    check will naturally produce zero fires.
    """
    expected_cells = float(p_eff[eligible].sum())
    if expected_cells <= 0:
        return p_eff

    expected_area = expected_cells * px_area
    if expected_area <= 0:
        return p_eff

    target_cells = target_area / px_area
    scale = target_cells / expected_cells

    scaled = p_eff * scale
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def target_to_patch_budget(
    target_area: float,
    px_area:     float,
) -> float:
    """
    Convert a target area directly into a patch-growing cell-count budget,
    for groups that also have a size distribution (v2.5 patch mechanic).

    This REPLACES sum(p_eff) as the budget passed to grow_patches_for_group,
    rather than scaling p_eff.

    Parameters
    ----------
    target_area : desired area (in the engine's configured AREA_UNIT)
    px_area     : area per pixel in the same unit

    Returns
    -------
    target cell count (float).
    """
    if px_area <= 0:
        return 0.0
    return max(0.0, target_area / px_area)
