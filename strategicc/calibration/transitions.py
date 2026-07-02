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
from scipy import ndimage

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


def compute_size_distribution(
    ts:           LULCTimeSeries,
    group_map:    dict[tuple[int, int], str],
    px_area_ha:   float,
    n_bins:       int = 5,
    connectivity: int = 8,
    min_patches:  int = 10,
) -> pd.DataFrame:
    """
    Derive a historical patch-size distribution per transition group, in
    ST-Sim TransitionSizeDistribution.csv format.

    For each year-pair and each transition GROUP (pooling all (from_id,
    to_id) pairs mapped to that group — same pooling behaviour as
    compute_temporal_distribution()), builds a binary "did this cell
    undergo this group's transition this year" mask, labels 8- (or 4-)
    connected patches within it via scipy.ndimage, and converts patch
    pixel counts to hectares using px_area_ha. Patch sizes are pooled
    across ALL year-pairs for that group, then binned into an
    equal-frequency (quantile) cumulative histogram matching the
    TransitionSizeRule schema consumed by strategicc.core.patches.

    Parameters
    ----------
    ts           : LULCTimeSeries from load_lulc_timeseries()
    group_map    : dict[(from_id, to_id), group_name] — SAME mapping used
                   for compute_transition_rates() / compute_temporal_
                   distribution(), so group definitions stay consistent
                   across Transitions.csv, TransitionMultipliers.csv, and
                   this output.
    px_area_ha   : area per pixel in hectares (from engine's CRS-aware
                   _pixel_area_ha(), NOT assumed).
    n_bins       : number of equal-frequency (quantile) bins per group.
                   Auto quantile binning — no manual bin_edges_ha needed.
    connectivity : 4 or 8 — neighbour connectivity for patch labeling.
                   8 (default) matches core.patches.grow_patch(), which
                   also grows patches via 8-connected BFS.
    min_patches  : groups with fewer than this many observed patches
                   (pooled across all years) are skipped and a printed
                   warning recommends omitting them from
                   TransitionSizeDistribution.csv (independent-cell
                   firing will be used for that group instead).

    Returns
    -------
    DataFrame in TransitionSizeDistribution.csv schema:
        Transition Type/Group, Maximum Area (Hectares), Relative Amount
    Bins are cumulative and ascending per group, "Relative Amount" as a
    percentage (sums to ~100 per group) — matches load_transition_size_
    rules()'s expected format exactly.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)

    # group -> list of (from_id, to_id) pairs mapped to it
    pairs_by_group: dict[str, list[tuple[int, int]]] = {}
    for (from_id, to_id), group in group_map.items():
        pairs_by_group.setdefault(group, []).append((from_id, to_id))

    sizes_by_group: dict[str, list[float]] = {g: [] for g in pairs_by_group}

    for i in range(len(ts.years) - 1):
        year_from = ts.years[i]
        year_to   = ts.years[i + 1]
        if year_to != year_from + 1:
            continue

        map_from = ts.stack[i]
        map_to   = ts.stack[i + 1]

        for group, pairs in pairs_by_group.items():
            mask = np.zeros(map_from.shape, dtype=bool)
            for from_id, to_id in pairs:
                mask |= (map_from == from_id) & (map_to == to_id)

            if not mask.any():
                continue

            labels, n_patches = ndimage.label(mask, structure=structure)
            counts = ndimage.sum(mask, labels, index=range(1, n_patches + 1))
            sizes_by_group[group].extend((np.asarray(counts) * px_area_ha).tolist())

    rows = []
    skipped = []

    for group, sizes in sizes_by_group.items():
        n_observed = len(sizes)
        if n_observed < min_patches:
            skipped.append((group, n_observed))
            continue

        sizes_arr = np.asarray(sizes)
        quantile_points = np.linspace(0, 100, n_bins + 1)[1:]  # drop the 0th percentile
        max_areas = np.unique(np.percentile(sizes_arr, quantile_points))

        prev_max = 0.0
        for max_area in max_areas:
            in_bin = ((sizes_arr > prev_max) & (sizes_arr <= max_area))
            relative_amount = 100.0 * in_bin.sum() / n_observed
            rows.append({
                "Transition Type/Group":   f"{group} [Type]",
                "Maximum Area (Hectares)": round(float(max_area), 4),
                "Relative Amount":         round(relative_amount, 4),
            })
            prev_max = max_area

        print(f"  {group}: n_patches={n_observed}  "
              f"size_range=[{sizes_arr.min():.4f}, {sizes_arr.max():.4f}] ha  "
              f"bins={len(max_areas)}")

    if skipped:
        print(f"  [Info] {len(skipped)} group(s) skipped "
              f"(insufficient patches, recommend omitting from "
              f"TransitionSizeDistribution.csv — independent-cell firing "
              f"will apply):")
        for group, n in skipped:
            print(f"      {group}  (n_patches={n}, min_required={min_patches})")

    result = pd.DataFrame(rows)
    print(f"  TransitionSizeDistribution.csv: {len(result)} bin row(s) "
          f"across {result['Transition Type/Group'].nunique() if not result.empty else 0} group(s)")
    return result


def save_size_distribution_csv(df: pd.DataFrame, out_path: str | Path) -> None:
    """Save derived patch-size distribution in ST-Sim TransitionSizeDistribution.csv format."""
    out_path = Path(out_path)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
