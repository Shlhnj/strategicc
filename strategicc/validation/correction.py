"""
strategicc/validation/correction.py  —  v3.12
--------------------------------------------------
Auto-correction of Transition Multipliers, given a hindcast run's observed
vs. simulated divergence, broken down by pathway (attribute_extent_drift()).

Two methods, selected via `method=`:

  "scaling"  (default, v3.12) — for each flagged pathway, compute a single
             ratio: observed_rate / simulated_rate over the hindcast window,
             then rescale that pathway's existing Transition Multiplier
             range by that ratio. Closed-form, no re-running the engine.

  "optimize" — iterative re-run + scipy optimization to fit multipliers
             against the observed trajectory directly. NOT YET IMPLEMENTED:
             ships as a selectable path so callers can switch methods
             later without an API change, but calling it raises
             NotImplementedError until the open design questions (multiplier
             bounds, convergence criteria, re-run budget) are resolved.
             See v3.12 scoping notes.

Adjustment target
------------------
Corrections are applied to Transition Multipliers (DistributionMin /
DistributionMax), NOT to the calibrated base Transitions.csv probabilities
-- the empirical baseline stays untouched; only the temporal-variability
range layered on top of it is rescaled.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc.io.csv_loader import _strip_type_suffix


# ─────────────────────────────────────────────────────────────────────────────
# Pathway rate ratios (observed vs. simulated), from a hindcast run
# ─────────────────────────────────────────────────────────────────────────────

def compute_pathway_rate_ratios(
    trans_df:              pd.DataFrame,
    area_df:                pd.DataFrame,
    transitions_csv_path:   str | Path,
    n_timesteps:            int,
) -> pd.DataFrame:
    """
    Approximate mean annual transition probability per group from a
    simulated hindcast run, and compare it against the calibrated
    (observed) rate for that group in Transitions.csv.

    Approximation (flagged, not resolved)
    --------------------------------------
    The "eligible source pool" for a group is approximated as the total
    area of every class that appears as a `from_class` for that group in
    `trans_df`, measured at the FIRST simulated year, held constant across
    the whole hindcast window. In reality the pool shrinks/grows every
    year as cells convert -- this is a simplification that will understate
    or overstate the ratio for groups with large area swings during the
    hindcast window. Flagging rather than silently building the full
    year-by-year depleting-pool version (which would mirror
    calibration.transitions.compute_yearly_transition_counts more closely
    but needs per-iteration, per-year eligible-pool bookkeeping).

    Parameters
    ----------
    trans_df    : concatenated transition_log.csv from the hindcast run
                  (iteration, year, row, col, from_class, to_class, group)
    area_df     : concatenated area_table.csv from the hindcast run
                  (iteration, year, class_id, class_name, area_{unit})
    transitions_csv_path : path to the calibrated Transitions.csv
                  (StateClassIdSource, StateClassIdDest, TransitionTypeId,
                  Probability) -- observed rates, one row per group
                  (TransitionTypeId IS the group column here).
    n_timesteps : number of years simulated in the hindcast window

    Returns
    -------
    DataFrame: group, observed_rate, simulated_rate, ratio
    (ratio = observed_rate / simulated_rate; NaN where simulated_rate is 0)
    """
    if trans_df.empty:
        return pd.DataFrame(columns=["group", "observed_rate", "simulated_rate", "ratio"])

    acol = next((c for c in area_df.columns if c.startswith("area_")), None)
    if acol is None:
        raise ValueError(f"area_df has no area_* column. Got: {list(area_df.columns)}")

    # Observed rate per group: mean Probability across all pairs mapped to
    # that group in the calibrated Transitions.csv.
    obs_df = pd.read_csv(transitions_csv_path)
    obs_df["group"] = obs_df["TransitionTypeId"].apply(_strip_type_suffix)
    observed_rates = obs_df.groupby("group")["Probability"].mean().to_dict()

    n_iterations = trans_df["iteration"].nunique() or 1

    rows = []
    for group, grp_trans in trans_df.groupby("group"):
        n_transitioned = len(grp_trans)

        source_classes = grp_trans["from_class"].unique().tolist()
        first_year = area_df["year"].min()
        pool = area_df[
            (area_df["year"] == first_year) & (area_df["class_name"].isin(source_classes))
        ][acol].sum()
        # area_{unit} values are per-iteration area; average pool across
        # iterations at the first year for a single representative pool size.
        n_pool_iters = area_df[area_df["year"] == first_year]["iteration"].nunique() or 1
        pool = pool / n_pool_iters

        eligible_cell_years = pool * n_timesteps * n_iterations
        simulated_rate = (n_transitioned / eligible_cell_years) if eligible_cell_years else np.nan

        observed_rate = observed_rates.get(group, np.nan)
        ratio = (
            observed_rate / simulated_rate
            if simulated_rate and not np.isnan(simulated_rate) and simulated_rate > 0
            else np.nan
        )

        rows.append({
            "group":          group,
            "observed_rate":  observed_rate,
            "simulated_rate": simulated_rate,
            "ratio":          ratio,
        })

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Multiplier correction
# ─────────────────────────────────────────────────────────────────────────────

def correct_multipliers(
    rate_ratios:              pd.DataFrame,
    transition_mult_csv_path: str | Path,
    method:                   str = "scaling",
    bounds:                   tuple[float, float] = (0.01, 100.0),
) -> pd.DataFrame:
    """
    Rescale Transition Multiplier ranges for flagged pathways, using the
    observed/simulated rate ratios from compute_pathway_rate_ratios().

    Parameters
    ----------
    rate_ratios : output of compute_pathway_rate_ratios()
                  (group, observed_rate, simulated_rate, ratio)
    transition_mult_csv_path : path to the existing TransitionMultipliers.csv
                  to correct. Read and returned in its ORIGINAL column
                  schema -- all columns and rows for unaffected groups are
                  preserved untouched; only DistributionMin/DistributionMax
                  (and Amount, if set) are rescaled for matched groups.
    method      : "scaling" (default, v3.12) or "optimize" (not yet
                  implemented -- raises NotImplementedError)
    bounds      : (min, max) clamp applied to corrected DistributionMin/
                  DistributionMax values, to keep the optimizer/scaler from
                  proposing physically nonsensical multiplier ranges
                  (e.g. negative or absurdly large). Default is a wide
                  permissive range; tighten per-project as needed.

    Returns
    -------
    DataFrame in TransitionMultipliers.csv schema (same columns as the
    input file). Does NOT overwrite the file -- caller decides whether to
    save it (consistent with flag-don't-silently-apply).
    """
    if method == "optimize":
        raise NotImplementedError(
            "method='optimize' (iterative re-run + scipy optimization) is "
            "not yet implemented -- open design questions (multiplier "
            "bounds, convergence criteria, re-run budget per fit attempt) "
            "have not been resolved. Use method='scaling' for now."
        )
    if method != "scaling":
        raise ValueError(f"Unknown method '{method}'. Use 'scaling' or 'optimize'.")

    df = pd.read_csv(transition_mult_csv_path)
    if "TransitionGroupId" not in df.columns:
        raise ValueError(
            f"'{transition_mult_csv_path}' has no TransitionGroupId column -- "
            f"got columns: {list(df.columns)}"
        )

    ratio_by_group = {
        row["group"]: row["ratio"]
        for _, row in rate_ratios.iterrows()
        if pd.notna(row["ratio"])
    }

    lo, hi = bounds
    n_corrected = 0

    for idx, row in df.iterrows():
        group = _strip_type_suffix(str(row.get("TransitionGroupId", "")))
        ratio = ratio_by_group.get(group)
        if ratio is None:
            continue

        for col in ("DistributionMin", "DistributionMax", "Amount"):
            if col in df.columns and pd.notna(row.get(col)):
                new_val = float(row[col]) * ratio
                new_val = min(max(new_val, lo), hi)
                df.at[idx, col] = new_val

        n_corrected += 1

    print(f"  correct_multipliers: {n_corrected} row(s) rescaled "
          f"(method='scaling', bounds={bounds})")

    return df
