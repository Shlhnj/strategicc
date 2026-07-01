# `strategicc.core`

The transition-firing mechanics used internally by `StrategiccEngine` each timestep. Most users interact with these only indirectly through `StrategiccEngine`'s toggles, this page is for understanding how a toggle changes behaviour, or for building a custom engine on top of the same primitives.

```python
from strategicc.core import (
    compute_neighbor_fractions, build_initial_age_from_raster, update_age,
    grow_patches_for_group, resolve_targets_per_timestep, build_transition_index,
)
```

## Transitions (`core/transitions.py`)

`build_transition_index()` converts a flat list of `TransitionRule` (from `Transitions.csv`) into a per-source-class lookup dict the engine iterates over each timestep, `{from_id: [(to_id, base_probability, group), ...]}`.

`TransitionRecord` is the dataclass logged every time a transition actually fires on a cell: `year`, `row`, `col`, `from_id`, `to_id`, `group`.

## Adjacency (`core/adjacency.py`)

`compute_neighbor_fractions(lulc_map, n_classes)` computes, for every cell, what fraction of its 8 neighbours belong to each class, frozen at the start of each timestep before any transitions fire that step. This is the basis for `use_adjacency`: groups in `STRICT_EXPANSION_GROUPS` require at least one neighbour of the target class to fire at all; other groups get a baseline-1.0 boost proportional to neighbour fraction.

## Age (`core/age.py`)

| Function | Purpose |
|---|---|
| `build_initial_age_from_raster(path)` | Load a pre-existing age raster (e.g. from calibration) |
| `build_initial_age_from_rules(lulc, classes, rules, rng)` | Sample initial age per cell from `InitialAge.csv` truncated-normal assumptions, when no raster is available |
| `update_age(age_map, fired, reset_mask, relative_values)` | Increment age by 1 for non-transitioned cells; reset to 0 or a specified value for transitioned cells, per the `AgeReset`/`AgeRelative` columns in `Transitions.csv` |
| `age_gate_mask(age_map, age_min, age_max)` | Boolean mask of cells whose age satisfies a transition's `AgeMin`/`AgeMax` gate |

This module is what makes `use_age=True` meaningful, without it, transitions can't be age-gated and Stock & Flow's age-indexed NPP lookup has no age to index against.

## Patches (`core/patches.py`)

The mechanic behind groups listed in `TransitionSizeDistribution.csv`, instead of independent-cell firing, transitions cluster into discrete spatial events matching a historical size distribution.

```python
fired = grow_patches_for_group(p_eff, eligible, size_bins, px_area_ha, rng)
```

`grow_patches_for_group()` repeatedly seeds a patch (weighted by `p_eff`, which already encodes adjacency + spatial suitability) and grows it via 8-connected Breadth-First Search (BFS) until either the sampled patch size is reached or the budget (`sum(p_eff)` over eligible cells, matching what independent-cell firing would have produced in expectation) is exhausted. An optional `budget_override` parameter lets Transition Targets replace that budget with an explicit area target.

## Targets (`core/targets.py`)

The mechanic behind `TransitionTargets.csv`, area-based overrides that replace or scale a group's normal probability-derived budget for a timestep, matching ST-Sim's official target-normalization algorithm.

| Function | Purpose |
|---|---|
| `resolve_targets_per_timestep(rules, n_timesteps)` | Forward-fills target rules into an explicit `{timestep: {group: amount}}` lookup (a target persists until a new record or an explicit blank-amount row turns it off) |
| `scale_probability_to_target(p_eff, eligible, target_area, px_area)` | For groups without a size distribution: scales `p_eff` so the expected fired area matches the target, preserving stochastic variance |
| `target_to_patch_budget(target_area, px_area)` | For groups with a size distribution: converts the target directly into a patch-growing cell-count budget |

## Multipliers (`core/multipliers.py`)

`sample_transition_multipliers(rules, rng)` draws one stochastic scalar per transition group per timestep from a `Uniform(min, max)` distribution defined in `TransitionMultipliers.csv`, this is what `use_trans_multiplier` controls.

## How it all combines

Each timestep, for each `(from_class, to_class, group)` pathway, the engine computes:

```
p_eff = base_probability * temporal_multiplier * adjacency_multiplier * spatial_multiplier
```

then branches: if the group has a `TransitionAdjacencySetting`/`Multipliers` entry, the adjacency strength comes from that CSV instead of the global `ADJACENCY_STRENGTH` constant; if the group has a size distribution, `p_eff` becomes the patch-growing seed-weighting and budget instead of an independent per-cell draw; if the group has an active target, the target either scales `p_eff` or replaces the patch budget entirely.
