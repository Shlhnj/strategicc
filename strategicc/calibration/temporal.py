"""
strategicc/calibration/temporal.py  —  v3.9
------------------------------------------------
Derive the temporal (stochastic) transition multiplier distribution from
the SAME year-by-year transition counts used by compute_transition_rates().

For each transition group, the yearly multiplier is:
    multiplier_year = yearly_probability / mean_probability

By construction this guarantees mean(multiplier_year) ~ 1.0 — sampling
Uniform(min(multiplier), max(multiplier)) during simulation reproduces the
same average behaviour as the static Transitions.csv probability, while
preserving the historical year-to-year variability as the uncertainty range.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc.calibration.transitions import YearlyTransitionCounts


def compute_temporal_distribution(
    yearly:     YearlyTransitionCounts,
    group_map:  dict[tuple[int, int], str],
    min_years:  int = 3,
) -> pd.DataFrame:
    """
    Derive a Uniform(min, max) multiplier distribution per transition group
    from year-by-year probability variability.

    Parameters
    ----------
    yearly     : output of compute_yearly_transition_counts() — the SAME
                 object passed to compute_transition_rates(), ensuring
                 consistency between Transitions.csv and this output.
    group_map  : dict[(from_id, to_id), group_name] — same mapping used
                 for compute_transition_rates().
    min_years  : groups observed in fewer than this many distinct years
                 are skipped and a fixed multiplier of 1.0 is recommended.

    Returns
    -------
    DataFrame in ST-Sim TransitionMultipliers.csv schema:
        TransitionGroupId, DistributionType, DistributionMin, DistributionMax
    One row per transition GROUP (multiple from/to pairs sharing a group
    are pooled together).
    """
    df = yearly.records.copy()
    if df.empty:
        print("  [Warning] No yearly records — cannot derive temporal distribution")
        return pd.DataFrame()

    df["group"] = df.apply(
        lambda r: group_map.get((int(r["from_id"]), int(r["to_id"]))), axis=1
    )
    df = df.dropna(subset=["group"])

    if df.empty:
        print("  [Warning] No mapped transitions — check group_map")
        return pd.DataFrame()

    rows = []
    skipped = []

    for group, grp in df.groupby("group"):
        pooled = (
            grp.groupby("year")
            .apply(lambda g: g["n_cells"].sum() / g["n_from_total"].sum())
            .rename("group_probability")
        )

        n_years_observed = len(pooled)
        if n_years_observed < min_years:
            skipped.append((group, n_years_observed))
            continue

        mean_prob = pooled.mean()
        if mean_prob <= 0:
            skipped.append((group, n_years_observed))
            continue

        multipliers = pooled / mean_prob

        rows.append({
            "TransitionGroupId": f"{group} [Type]",
            "DistributionType":  "Uniform",
            "DistributionMin":   round(float(multipliers.min()), 4),
            "DistributionMax":   round(float(multipliers.max()), 4),
        })

        print(f"  {group}: n_years={n_years_observed}  "
              f"mean_prob={mean_prob:.5f}  "
              f"multiplier_range=[{multipliers.min():.3f}, {multipliers.max():.3f}]")

    if skipped:
        print(f"  [Info] {len(skipped)} group(s) skipped "
              f"(insufficient years, recommend fixed multiplier=1.0):")
        for group, n in skipped:
            print(f"      {group}  (n_years={n}, min_required={min_years})")

    result = pd.DataFrame(rows)
    print(f"  TransitionMultipliers.csv: {len(result)} group(s) derived")
    return result


def save_temporal_distribution_csv(
    df:       pd.DataFrame,
    out_path: str | Path | None = None,
) -> Path:
    """Save derived temporal distribution in ST-Sim TransitionMultipliers.csv format.

    Parameters
    ----------
    df       : DataFrame from compute_temporal_distribution()
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
