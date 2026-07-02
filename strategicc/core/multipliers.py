"""
strategicc/core/multipliers.py  —  v1.2  Stochastic transition multipliers
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
* Supported distributions:
    - "Uniform" (v1.1)                — draws from (DistributionMin, DistributionMax)
    - Any other name (v1.2 / v3.6.1)  — treated as a reference into a
      Distributions.csv-derived empirical table (see DistributionEntry in
      strategicc.io.csv_loader). Value is drawn discretely with probability
      proportional to ValueDistributionRelativeFrequency, matching ST-Sim's
      "Iteration and Timestep" frequency-distribution behaviour.
  Extend _sample() for continuous Normal, Beta, etc. in future versions.

Usage (inside engine)
---------------------
    group_mults = sample_transition_multipliers(rules, rng, distributions)
    # → {"Agriculture_expansion": 0.41, "Inundation": 0.87, ...}
    # then: p_eff = base_prob * adj_mult * sp_mult * group_mults.get(group, 1.0)
"""

from __future__ import annotations
import numpy as np
from strategicc.io.csv_loader import TransitionMultiplierRule, DistributionEntry


# ── Distribution kinds handled literally (case-insensitive) ───────────────────
_LITERAL = {"uniform"}


def _sample_empirical(entry: DistributionEntry, rng: np.random.Generator) -> float:
    """Draw one value from a named empirical distribution, weighted by
    ValueDistributionRelativeFrequency."""
    weights = np.asarray(entry.weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError(
            f"Distribution '{entry.name}' has no positive relative "
            "frequency weights to sample from."
        )
    probs = weights / total
    idx = rng.choice(len(entry.values), p=probs)
    return float(entry.values[idx])


def _sample(
    rule: TransitionMultiplierRule,
    rng: np.random.Generator,
    distributions: dict[str, DistributionEntry] | None = None,
) -> float:
    """Draw one scalar from the rule's distribution.

    'Uniform' is handled literally via (dist_min, dist_max). Any other
    DistributionType is looked up by name in `distributions` (loaded from
    Distributions.csv) and sampled as a discrete empirical distribution.
    """
    dist = rule.distribution.lower()
    if dist in _LITERAL:
        return float(rng.uniform(rule.dist_min, rule.dist_max))

    if distributions and rule.distribution in distributions:
        return _sample_empirical(distributions[rule.distribution], rng)

    raise ValueError(
        f"Unsupported DistributionType '{rule.distribution}' for group "
        f"'{rule.group}': not 'Uniform' and no matching named entry was "
        "found in Distributions.csv. Supported literal types: "
        f"{_LITERAL}. Check that DISTRIBUTIONS_CSV is configured and that "
        f"'{rule.distribution}' exists as a DistributionTypeId in it."
    )


def sample_transition_multipliers(
    rules: list[TransitionMultiplierRule],
    rng:   np.random.Generator,
    distributions: dict[str, DistributionEntry] | None = None,
) -> dict[str, float]:
    """
    Sample one multiplier per group for a single timestep.

    Parameters
    ----------
    rules         : output of load_transition_multipliers()
    rng           : numpy Generator (shared with cell-level draws for reproducibility)
    distributions : output of load_distributions(), or None if Distributions.csv
                    is not configured (only literal 'Uniform' rules will work
                    in that case)

    Returns
    -------
    dict[group_name, sampled_scalar]
    Groups not present in rules get an implicit multiplier of 1.0 in the engine.
    """
    result: dict[str, float] = {}
    for rule in rules:
        result[rule.group] = _sample(rule, rng, distributions)
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
