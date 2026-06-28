"""
strategicc/core/multipliers.py  —  v1.1  Stochastic transition multipliers
---------------------------------------------------------------------
Samples one scalar multiplier per group per timestep from the distribution
defined in TransitionMultipliers.csv, then returns a lookup dict used by
the engine to scale base probabilities before firing transitions.

Design
------
* Sampling happens once per timestep (not per cell), so all cells sharing a
  group see the same temporal multiplier that step — matching ST-Sim behaviour.
* The RNG passed in is the same generator used for cell-level draws, ensuring
  full reproducibility from a single seed per iteration.
* Supported distributions (v1.1): Uniform.
  Extend _sample() for Normal, Beta, etc. in future versions.

Usage (inside engine)
---------------------
    group_mults = sample_transition_multipliers(rules, rng)
    # → {"Agriculture_expansion": 0.41, "Inundation": 0.87, ...}
    # then: p_eff = base_prob * adj_mult * sp_mult * group_mults.get(group, 1.0)
"""

from __future__ import annotations
import numpy as np
from strategicc.io.csv_loader import TransitionMultiplierRule


# ── Supported distributions ───────────────────────────────────────────────────
_SUPPORTED = {"uniform"}


def _sample(rule: TransitionMultiplierRule, rng: np.random.Generator) -> float:
    """Draw one scalar from the rule's distribution."""
    dist = rule.distribution.lower()
    if dist == "uniform":
        return float(rng.uniform(rule.dist_min, rule.dist_max))
    # Placeholder for future distributions
    raise ValueError(
        f"Unsupported DistributionType '{rule.distribution}' for group "
        f"'{rule.group}'.  Supported: {_SUPPORTED}"
    )


def sample_transition_multipliers(
    rules: list[TransitionMultiplierRule],
    rng:   np.random.Generator,
) -> dict[str, float]:
    """
    Sample one multiplier per group for a single timestep.

    Parameters
    ----------
    rules : output of load_transition_multipliers()
    rng   : numpy Generator (shared with cell-level draws for reproducibility)

    Returns
    -------
    dict[group_name, sampled_scalar]
    Groups not present in rules get an implicit multiplier of 1.0 in the engine.
    """
    result: dict[str, float] = {}
    for rule in rules:
        result[rule.group] = _sample(rule, rng)
    return result


def describe_multiplier_rules(rules: list[TransitionMultiplierRule]) -> None:
    """Print a human-readable summary of loaded multiplier rules."""
    if not rules:
        print("  No transition multiplier rules loaded.")
        return
    print("  Transition multiplier rules:")
    for r in rules:
        print(
            f"    {r.group:30s}  {r.distribution}("
            f"{r.dist_min}, {r.dist_max})"
        )
