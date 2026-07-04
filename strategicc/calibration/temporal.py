"""
strategicc/calibration/temporal.py  v3.10
------------------------------------------------
Derive the temporal (stochastic) transition multiplier distribution from
the SAME year-by-year transition counts used by compute_transition_rates().

For each transition GROUP, the empirical multiplier set is built from every
individual (from_id, to_id) PATHWAY mapped to that group, each normalized to
its OWN historical mean probability:

    multiplier_year_pathway = probability_year_pathway / mean_probability_pathway

Pathways are normalized independently (not pooled by summing cell counts
first) so that a group made of several from/to pairs preserves each
pathway's own year-to-year variability, rather than diluting it into one
blended series. A pathway needs at least `min_years` distinct observed
years to contribute; pathways with fewer are dropped rather than pooled in
as sparse/duplicate-zero noise (mirrors the existing group-level min_years
guard, just applied per pathway).

By construction this guarantees mean(multiplier) ~ 1.0 per pathway, so the
pooled empirical set for a group is centered around 1.0 as well.

Two outputs are produced from the same pooled empirical set:
  - TransitionMultipliers.csv row (one per group): DistributionType is set
    to a NAMED distribution ("{group} Distribution") rather than the
    literal "Uniform". DistributionMin/Max are still filled with the
    pooled set's min/max — this is required only because
    load_transition_multipliers() skips rows where these are None, even
    though the engine ignores them once a named distribution is used
    (see strategicc.core.multipliers._sample()).
  - Distributions.csv rows (one per group): one row per DISTINCT observed
    multiplier value, with duplicate values collapsed into
    ValueDistributionRelativeFrequency counts — matching the ST-Sim course's
    own Distributions Datafeed convention (Exercise 4, Task 4).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc.calibration.transitions import YearlyTransitionCounts


#: Full ST-Sim Distributions.csv column order (only DistributionTypeId,
#: Value, and ValueDistributionRelativeFrequency are populated here; the
#: rest are left blank, matching how load_distributions() only reads
#: those three columns).
_DISTRIBUTIONS_CSV_COLUMNS = [
    "Iteration", "Timestep", "StratumId", "SecondaryStratumId",
    "TertiaryStratumId", "DistributionTypeId", "ExternalVariableTypeId",
    "ExternalVariableMin", "ExternalVariableMax", "Value",
    "ValueDistributionTypeId", "ValueDistributionFrequency",
    "ValueDistributionSD", "ValueDistributionMin", "ValueDistributionMax",
    "ValueDistributionRelativeFrequency",
]


def compute_temporal_distribution(
    yearly:     YearlyTransitionCounts,
    group_map:  dict[tuple[int, int], str],
    min_years:  int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive a named empirical multiplier distribution per transition group
    from year-by-year, PER-PATHWAY probability variability.

    Parameters
    ----------
    yearly     : output of compute_yearly_transition_counts() — the SAME
                 object passed to compute_transition_rates(), ensuring
                 consistency between Transitions.csv and this output.
    group_map  : dict[(from_id, to_id), group_name] — same mapping used
                 for compute_transition_rates().
    min_years  : a (from_id, to_id) PATHWAY observed in fewer than this
                 many distinct years is dropped from the group's empirical
                 set entirely (not pooled in), rather than skipping the
                 whole group. A group is only skipped if none of its
                 pathways clear this threshold.

    Returns
    -------
    (temporal_df, distributions_df) :
        temporal_df       — ST-Sim TransitionMultipliers.csv schema:
                             TransitionGroupId, DistributionType,
                             DistributionMin, DistributionMax
                             One row per transition GROUP. DistributionType
                             is a named reference ("{group} Distribution"),
                             not the literal "Uniform".
        distributions_df  — ST-Sim Distributions.csv schema (full column
                             set; only DistributionTypeId, Value, and
                             ValueDistributionRelativeFrequency populated).
                             One row per distinct observed multiplier value
                             per group, pooled across all of that group's
                             qualifying pathways.
    """
    df = yearly.records.copy()
    if df.empty:
        print("  [Warning] No yearly records — cannot derive temporal distribution")
        return pd.DataFrame(), pd.DataFrame(columns=_DISTRIBUTIONS_CSV_COLUMNS)

    df["group"] = df.apply(
        lambda r: group_map.get((int(r["from_id"]), int(r["to_id"]))), axis=1
    )
    df = df.dropna(subset=["group"])

    if df.empty:
        print("  [Warning] No mapped transitions — check group_map")
        return pd.DataFrame(), pd.DataFrame(columns=_DISTRIBUTIONS_CSV_COLUMNS)

    mult_rows = []
    dist_rows = []
    skipped_pathways = []
    skipped_groups   = []

    for group, grp in df.groupby("group"):
        pooled_multipliers: list[float] = []
        n_pathways_used = 0

        for (from_id, to_id), path_df in grp.groupby(["from_id", "to_id"]):
            n_years_observed = path_df["year"].nunique()
            if n_years_observed < min_years:
                skipped_pathways.append((group, int(from_id), int(to_id), n_years_observed))
                continue

            mean_prob = path_df["probability"].mean()
            if mean_prob <= 0:
                skipped_pathways.append((group, int(from_id), int(to_id), n_years_observed))
                continue

            multipliers = path_df["probability"] / mean_prob
            pooled_multipliers.extend(multipliers.tolist())
            n_pathways_used += 1

        if not pooled_multipliers:
            skipped_groups.append(group)
            continue

        arr = np.asarray(pooled_multipliers, dtype=float)
        dist_name = f"{group} Distribution"

        mult_rows.append({
            "TransitionGroupId": f"{group} [Type]",
            "DistributionType":  dist_name,
            "DistributionMin":   round(float(arr.min()), 4),
            "DistributionMax":   round(float(arr.max()), 4),
        })

        values, counts = np.unique(np.round(arr, 6), return_counts=True)
        for value, count in zip(values, counts):
            row = {col: None for col in _DISTRIBUTIONS_CSV_COLUMNS}
            row["DistributionTypeId"] = dist_name
            row["Value"] = float(value)
            row["ValueDistributionRelativeFrequency"] = int(count)
            dist_rows.append(row)

        print(f"  {group}: n_pathways={n_pathways_used}  n_values={len(arr)}  "
              f"({len(values)} distinct)  "
              f"multiplier_range=[{arr.min():.3f}, {arr.max():.3f}]")

    if skipped_pathways:
        print(f"  [Info] {len(skipped_pathways)} pathway(s) excluded from their "
              f"group's empirical set (insufficient years):")
        for group, from_id, to_id, n in skipped_pathways:
            print(f"      {group}  ({from_id} -> {to_id})  "
                  f"(n_years={n}, min_required={min_years})")

    if skipped_groups:
        print(f"  [Info] {len(skipped_groups)} group(s) skipped entirely "
              f"(no pathway met min_years, recommend fixed multiplier=1.0):")
        for group in skipped_groups:
            print(f"      {group}")

    temporal_df = pd.DataFrame(mult_rows)
    distributions_df = pd.DataFrame(dist_rows, columns=_DISTRIBUTIONS_CSV_COLUMNS)

    print(f"  TransitionMultipliers.csv: {len(temporal_df)} group(s) derived")
    print(f"  Distributions.csv: {len(distributions_df)} value row(s) across "
          f"{distributions_df['DistributionTypeId'].nunique() if not distributions_df.empty else 0} "
          f"named distribution(s)")

    return temporal_df, distributions_df


def save_temporal_distribution_csv(
    df:       pd.DataFrame,
    out_path: str | Path | None = None,
) -> Path:
    """Save derived temporal distribution in ST-Sim TransitionMultipliers.csv format.

    Parameters
    ----------
    df       : temporal_df from compute_temporal_distribution()
    out_path : destination path; defaults to calibration_result/TransitionMultipliers.csv

    Returns
    -------
    Path actually written to
    """
    from strategicc.calibration.paths import TRANS_MULT_CSV
    out_path = Path(out_path) if out_path is not None else TRANS_MULT_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return out_path


def save_distributions_csv(
    df:       pd.DataFrame,
    out_path: str | Path | None = None,
) -> Path:
    """Save derived empirical distributions in ST-Sim Distributions.csv format.

    Parameters
    ----------
    df       : distributions_df from compute_temporal_distribution()
    out_path : destination path; defaults to calibration_result/Distributions.csv

    Returns
    -------
    Path actually written to
    """
    from strategicc.calibration.paths import TRANS_DIST_CSV
    out_path = Path(out_path) if out_path is not None else TRANS_DIST_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return out_path
