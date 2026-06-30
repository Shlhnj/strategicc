"""
strategicc/calibration/transitions.py  —  v2.4
--------------------------------------------------
Derive transition rates from a historical LULC time series.

Computes year-by-year transition counts ONCE, then exposes:
  - mean annual probability per (from_class, to_class) pathway
    → feeds Transitions.csv
  - the underlying year-by-year probability series
    → feeds compute_temporal_distribution() for TransitionMultipliers.csv

Computing both from the same year-by-year counts (rather than separately)
guarantees mean(yearly_multiplier_samples) ≈ 1.0 — the two output files
stay mathematically consistent with each other.

A `TransitionGroupMap` must be supplied to assign each observed
(from_class, to_class) pair to a named transition group (e.g.
"Mangrove_recruitment"), since raw class-to-class pairs in satellite
time series carry no group label on their own.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc.calibration.loader import LULCTimeSeries
from strategicc.io.csv_loader import StateClass


@dataclass
class YearlyTransitionCounts:
    """
    Year-by-year transition pixel counts.

    Attributes
    ----------
    records : DataFrame with columns:
        year, from_id, to_id, n_cells, n_from_total, probability
        (probability = n_cells / n_from_total for that year)
    """
    records: pd.DataFrame


def compute_yearly_transition_counts(
    ts: LULCTimeSeries,
) -> YearlyTransitionCounts:
    """
    Compute year-over-year transition pixel counts for every
    (from_class, to_class) pair observed in the time series.

    This is the single source of truth — both the mean Transitions.csv
    and the TransitionMultipliers.csv distribution are derived from this.

    Returns
    -------
    YearlyTransitionCounts
    """
    rows = []

    for i in range(len(ts.years) - 1):
        year_from = ts.years[i]
        year_to   = ts.years[i + 1]
        if year_to != year_from + 1:
            continue

        map_from = ts.stack[i]
        map_to   = ts.stack[i + 1]

        from_classes = np.unique(map_from)
        for from_id in from_classes:
            if from_id == 0:
                continue
            from_mask    = (map_from == from_id)
            n_from_total = int(from_mask.sum())
            if n_from_total == 0:
                continue

            to_values, counts = np.unique(map_to[from_mask], return_counts=True)
            for to_id, n_cells in zip(to_values, counts):
                if to_id == from_id or to_id == 0:
                    continue
                rows.append({
                    "year":         year_from,
                    "from_id":      int(from_id),
                    "to_id":        int(to_id),
                    "n_cells":      int(n_cells),
                    "n_from_total": n_from_total,
                    "probability":  n_cells / n_from_total,
                })

    df = pd.DataFrame(rows)
    print(f"  Computed yearly transitions: {len(df)} (year, from, to) records "
          f"across {len(ts.years)-1} year-pairs")
    return YearlyTransitionCounts(records=df)


def compute_transition_rates(
    yearly:          YearlyTransitionCounts,
    classes:         dict[int, StateClass],
    group_map:       dict[tuple[int, int], str],
    min_probability: float = 1e-5,
) -> pd.DataFrame:
    """
    Compute mean annual transition probability per pathway, in
    ST-Sim Transitions.csv format.

    Parameters
    ----------
    yearly          : output of compute_yearly_transition_counts()
    classes         : dict[int, StateClass] — for class name lookup
    group_map       : dict[(from_id, to_id), group_name] — assigns each
                      observed class pair to a named transition group.
                      Pairs not in this dict are EXCLUDED from output
                      (treated as noise/unmapped, printed as a warning).
    min_probability : pathways with mean probability below this are
                      dropped (filters out single-pixel classification
                      noise from the historical record).

    Returns
    -------
    DataFrame in Transitions.csv schema:
        StateClassIdSource, StateClassIdDest, TransitionTypeId, Probability
    """
    df = yearly.records
    if df.empty:
        print("  [Warning] No transition records to summarise")
        return pd.DataFrame()

    summary = (
        df.groupby(["from_id", "to_id"])["probability"]
        .mean()
        .reset_index()
        .rename(columns={"probability": "mean_probability"})
    )

    rows = []
    unmapped = []
    for _, row in summary.iterrows():
        from_id, to_id = int(row["from_id"]), int(row["to_id"])
        prob = float(row["mean_probability"])

        if prob < min_probability:
            continue

        key = (from_id, to_id)
        if key not in group_map:
            unmapped.append((from_id, to_id, prob))
            continue

        from_sc = classes.get(from_id)
        to_sc   = classes.get(to_id)
        if from_sc is None or to_sc is None:
            continue

        rows.append({
            "StateClassIdSource": from_sc.full_name,
            "StateClassIdDest":   to_sc.full_name,
            "TransitionTypeId":   group_map[key],
            "Probability":        round(prob, 6),
        })

    if unmapped:
        print(f"  [Warning] {len(unmapped)} unmapped (from,to) pair(s) "
              f"excluded — add to group_map if these are real transitions:")
        for from_id, to_id, prob in sorted(unmapped, key=lambda x: -x[2])[:10]:
            fn = classes[from_id].name if from_id in classes else from_id
            tn = classes[to_id].name   if to_id   in classes else to_id
            print(f"      {fn} → {tn}  (mean_prob={prob:.5f})")

    result = pd.DataFrame(rows)
    print(f"  Transitions.csv: {len(result)} pathway(s) derived "
          f"(min_probability={min_probability})")
    return result


def save_transitions_csv(df: pd.DataFrame, out_path: str | Path) -> None:
    """Save derived transition rates in ST-Sim Transitions.csv format."""
    out_path = Path(out_path)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
