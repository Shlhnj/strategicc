"""
strategicc/validation/correction.py  —  v3.14
--------------------------------------------------
Auto-correction of Transition Multipliers, given a hindcast run's observed
vs. simulated divergence, broken down by pathway (attribute_extent_drift()).

Two methods, selected via `method=`:

  "scaling"  (default) — for each flagged pathway, compute a single ratio:
             observed_rate / simulated_rate over the hindcast window, then
             rescale that pathway's Transition Multiplier by that ratio.
             Closed-form, no re-running the engine.

             IMPORTANT (fixed in this revision): DistributionMin/Max in
             TransitionMultipliers.csv are only actually used by the engine
             when DistributionType == "Uniform" (see core/multipliers.py).
             For a group whose DistributionType is a NAMED reference (e.g.
             calibration.temporal's empirical output), the engine ignores
             DistributionMin/Max entirely and samples from Distributions.csv
             instead. Scaling only DistributionMin/Max for such a group is
             therefore a no-op at simulation time. This revision detects
             which case applies per group and scales the right file --
             Distributions.csv's Value column for named distributions,
             DistributionMin/Max for literal "Uniform" rows.

  "optimize" — per flagged group, a bounded 1-D search (scipy
             minimize_scalar) over a scale factor, using cheap trial
             hindcast re-runs (hindcast_run(..., lightweight=True)) as the
             objective function. Resolved design parameters (this
             revision):
               - bounds: same as "scaling", (0.01, 100.0) by default
               - stopping rule: fixed max iterations (maxiter passed to
                 the bounded search)
               - re-run budget: max_reruns trial re-runs per group
                 (default 8) -- the binding constraint; maxiter is capped
                 to it so the search can't exceed the affordable re-run
                 budget even if the bounded method's internal iteration
                 count would otherwise run more evaluations.

Adjustment target
------------------
Corrections are applied to Transition Multipliers, NOT to the calibrated
base Transitions.csv probabilities -- the empirical baseline stays
untouched; only the temporal-variability layer on top of it is rescaled.

NOTE on scope vs. strategicc.calibration.transitions.normalize_transition_rates()
(v3.13): that function is the one that DOES touch the baseline -- it
rescales Transitions.csv itself, upstream at calibration time, to correct
for probability mass lost to unmapped pathways excluded by group_map. It
runs before correct_multipliers() ever sees the pipeline and is
independent of it; "the empirical baseline stays untouched" above is a
statement about what THIS module does, not a whole-pipeline guarantee.

v3.14 -- compute_pathway_rate_ratios() redesign
------------------------------------------------
Full redesign, not a patch. Previously this function pooled every
(from_class, to_class) pair sharing one TransitionTypeId "group" together
before computing a rate for that group. That pooling hid two compounding
bugs at once for any group spanning more than one pair:

  1. Unit mismatch -- the simulated side's denominator (mean_pool) was
     taken straight from area_df's hectare column, never converted to a
     pixel count via px_area_ha, while the numerator (mean_n_transitioned)
     was always a genuine pixel count. Every simulated_rate (and
     therefore every ratio) this function ever produced was off by a
     constant factor of 1/px_area_ha.
  2. Weighting mismatch -- observed_rate was an UNWEIGHTED mean across a
     group's pairs (each pair's calibrated Probability counted equally),
     while simulated_rate pooled all those pairs' source-class areas into
     one shared, pool-size-weighted denominator. For "Agriculture_expansion"
     (Aquaculture->Cropland + Other_vegetation->Cropland), Aquaculture's
     much larger pool dominated the pooled simulated_rate while
     observed_rate gave both pairs equal weight -- so the pooled ratio
     didn't represent either pathway's real performance, and hid that
     Aquaculture->Cropland (not Other_vegetation->Cropland) was the
     dominant, correctable, ~116x-under-simulated pathway.

Neither bug can recur in this version, because the pooling code path that
caused them is gone, not patched. The function now computes a rate at the
(from_class, to_class) PAIR level as the atomic unit and never pools
pairs together during calculation:

  * `group` is attached to each pair AFTER the rate is computed, as a
    display/reporting label only (read from Transitions.csv's
    TransitionTypeId for that pair) -- it is never a grouping key in the
    math.
  * Pairs with no group_map entry (i.e. absent from Transitions.csv but
    present in the simulated transition log) still appear in the output,
    labeled "unnamed", instead of being silently excluded.
  * `avg_pool_area_ha` and `total_ha_transitioned_*` columns are added so
    a reader can judge whether a given ratio matters in absolute area
    terms, rather than trusting a dimensionless ratio in isolation.
  * `px_area_ha` is now a required parameter, used to convert the
    hectare-denominated pool into a pixel count before dividing, fixing
    bug #1 above.
  * Source-class conditioning (denominator = eligible SOURCE-class pool)
    is confirmed as the only correct convention -- a transition
    probability is "fraction of source-class cells converting this year",
    never destination-class composition of inflow.

This is a breaking signature change (px_area_ha is now required, and the
output schema gained from_class/to_class/avg_pool_area_ha/
total_ha_transitioned_observed/total_ha_transitioned_simulated columns).
correct_multipliers(method="scaling") still keys its scale factors by
`group`, so a group spanning multiple pairs with differing ratios will
have only its last pair's ratio applied (dict collision) -- this is
flagged explicitly there rather than silently corrupting the multiplier
file. The recommended fix for such groups is to split them into separate,
independently-calibrated TransitionTypeId rows (e.g.
"Agriculture_expansion_OV" / "Agriculture_expansion_Aqua") at the
group_map/calibration stage, so each group maps to exactly one pair again.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc.io.csv_loader import _strip_type_suffix

_LITERAL_DISTRIBUTIONS = {"uniform"}


# ─────────────────────────────────────────────────────────────────────────────
# Pathway rate ratios (observed vs. simulated), from a hindcast run
# ─────────────────────────────────────────────────────────────────────────────

_RATE_RATIO_COLUMNS = [
    "from_class", "to_class", "group",
    "observed_rate", "simulated_rate", "ratio", "n_years_used",
    "avg_pool_area_ha",
    "total_ha_transitioned_observed", "total_ha_transitioned_simulated",
]


def compute_pathway_rate_ratios(
    trans_df:              pd.DataFrame,
    area_df:                pd.DataFrame,
    transitions_csv_path:   str | Path,
    px_area_ha:             float,
    n_timesteps:            int | None = None,
) -> pd.DataFrame:
    """
    Mean annual transition probability per (from_class, to_class) PAIR
    from a simulated hindcast run, compared against the calibrated
    (observed) rate for that same pair in Transitions.csv.

    v3.14 -- full redesign, pair-level (see module docstring for the full
    rationale and the two bugs this eliminates). The atomic unit of
    computation is now a single (from_class, to_class) pair; `group` is
    attached afterward purely as a display label and never participates
    in the rate calculation. Pairs are never pooled together.

    Per-year, per-pair rate calculation
    ------------------------------------
    For each YEAR separately: the eligible source-class pool is that
    year's actual area (not a single pool measured once and held
    constant), converted from hectares to a pixel count via px_area_ha
    (this conversion is the fix for the unit-mismatch bug -- the
    numerator, cells transitioned, is always a pixel count, so the
    denominator must be one too). The per-year rate is then averaged
    across years -- an unweighted mean, matching the convention
    calibration.transitions.compute_transition_rates() uses on the
    observed side. Since the simulation has multiple stochastic
    iterations, each year's rate is itself an average across iterations
    before being averaged across years.

    Parameters
    ----------
    trans_df    : concatenated transition_log.csv from the hindcast run
                  (iteration, year, row, col, from_class, to_class, group)
    area_df     : concatenated area_table.csv from the hindcast run
                  (iteration, year, class_id, class_name, area_{unit} in
                  hectares -- an "area_" prefixed column)
    transitions_csv_path : path to the calibrated Transitions.csv
                  (StateClassIdSource, StateClassIdDest, TransitionTypeId,
                  Probability) -- one row per (from, to) pair. Used
                  directly, without any cross-pair averaging, to look up
                  each pair's observed_rate and group label.
    px_area_ha  : pixel area in hectares (e.g. engine.px_area_ha) --
                  REQUIRED to convert area_df's hectare pool into a pixel
                  count comparable to trans_df's pixel-count transition
                  totals.
    n_timesteps : unused, kept only for call-site backward compatibility
                  with pre-v3.14 code; rates are computed per-year
                  regardless of this value.

    Returns
    -------
    DataFrame, one row per (from_class, to_class) pair observed in EITHER
    the calibrated Transitions.csv OR the simulated transition log
    (union, not intersection -- a pair calibrated but never simulated
    still appears, with simulated_rate showing exactly how often it
    fired: 0.0 if the pool existed but nothing converted, NaN if the
    pool never existed at all in area_df). Columns:

        from_class, to_class  : the pair (atomic unit; never pooled)
        group                 : TransitionTypeId for this pair from
                                 Transitions.csv, or the literal "unnamed"
                                 if the pair has no entry there (simulated
                                 but not calibrated -- a genuine, surfaced
                                 gap rather than a silent exclusion or a
                                 borrowed label from the simulated log)
        observed_rate         : this pair's own calibrated Probability
                                 (no averaging across other pairs sharing
                                 its group)
        simulated_rate        : this pair's per-year, source-pool-
                                 conditioned simulated rate (see above)
        ratio                 : observed_rate / simulated_rate; NaN if
                                 simulated_rate is 0, NaN, or observed_rate
                                 is NaN
        n_years_used          : how many of the window's years actually
                                 had a positive source-class pool for
                                 this pair (treat ratios backed by very
                                 few years with appropriate caution)
        avg_pool_area_ha      : mean source-class pool (hectares) across
                                 the years used
        total_ha_transitioned_observed : observed_rate * avg_pool_area_ha
                                 * n_years_used -- an ESTIMATE (this
                                 function has no separate historical pool
                                 series, only the mean calibrated
                                 Probability, so the simulated window's
                                 own average pool is used as the best
                                 available stand-in). NaN if inputs are
                                 missing.
        total_ha_transitioned_simulated : sum, across years used, of
                                 (mean cells transitioned that year) *
                                 px_area_ha -- the actual simulated area
                                 moved by this pair, not an estimate.

    Sorted by total_ha_transitioned_simulated descending (materiality --
    NaN treated as 0 for sorting only), so pairs that moved the most
    simulated area surface first.
    """
    if trans_df.empty and area_df.empty:
        return pd.DataFrame(columns=_RATE_RATIO_COLUMNS)

    acol = next((c for c in area_df.columns if c.startswith("area_")), None)
    if acol is None:
        raise ValueError(f"area_df has no area_* column. Got: {list(area_df.columns)}")

    obs_df = pd.read_csv(transitions_csv_path)
    obs_df["group"]      = obs_df["TransitionTypeId"].apply(_strip_type_suffix)
    obs_df["from_class"] = obs_df["StateClassIdSource"].astype(str).str.split(":").str[0]
    obs_df["to_class"]   = obs_df["StateClassIdDest"].astype(str).str.split(":").str[0]

    obs_map: dict[tuple[str, str], tuple[str, float]] = {}
    dup_pairs = 0
    for _, r in obs_df.iterrows():
        key = (r["from_class"], r["to_class"])
        if key in obs_map:
            dup_pairs += 1
        obs_map[key] = (r["group"], float(r["Probability"]))
    if dup_pairs:
        print(f"  [Warning] {dup_pairs} (from,to) pair(s) appear more than "
              f"once in {transitions_csv_path} -- only the last row's "
              f"group/Probability was kept per pair (Transitions.csv is "
              f"expected to have exactly one row per pair).")

    # Pairs the simulator logged (used only to build the pair universe --
    # NOT as a fallback source of a group label; a pair absent from
    # Transitions.csv is always labeled "unnamed", regardless of whatever
    # group the simulated log happens to carry for it, so a real gap in
    # calibration can't hide behind a borrowed label).
    sim_pairs: set[tuple[str, str]] = set()
    if not trans_df.empty:
        sim_pairs = set(
            trans_df[["from_class", "to_class"]].drop_duplicates().itertuples(index=False, name=None)
        )

    all_pairs = sorted(set(obs_map) | sim_pairs)
    if not all_pairs:
        return pd.DataFrame(columns=_RATE_RATIO_COLUMNS)

    if not trans_df.empty:
        all_iterations = sorted(trans_df["iteration"].unique())
    else:
        all_iterations = sorted(area_df["iteration"].unique())

    n_unnamed = 0
    rows = []
    for from_class, to_class in all_pairs:
        obs_group, observed_rate = obs_map.get((from_class, to_class), (None, np.nan))
        if obs_group is not None:
            group = obs_group
        else:
            group = "unnamed"
            n_unnamed += 1

        if not trans_df.empty:
            pair_trans = trans_df[
                (trans_df["from_class"] == from_class) & (trans_df["to_class"] == to_class)
            ]
        else:
            pair_trans = trans_df  # empty

        years = sorted(area_df.loc[area_df["class_name"] == from_class, "year"].unique())

        yearly_rates: list[float] = []
        pool_samples_ha: list[float] = []
        total_ha_simulated = 0.0

        for year in years:
            grp_year = pair_trans[pair_trans["year"] == year] if not pair_trans.empty else pair_trans
            n_by_iter = (
                grp_year.groupby("iteration").size().reindex(all_iterations, fill_value=0)
                if not grp_year.empty else pd.Series(0, index=all_iterations)
            )
            mean_n_transitioned = n_by_iter.mean()

            year_area = area_df[
                (area_df["year"] == year) & (area_df["class_name"] == from_class)
            ]
            pool_by_iter = year_area.groupby("iteration")[acol].sum()
            if pool_by_iter.empty:
                continue
            mean_pool_ha = pool_by_iter.reindex(all_iterations, fill_value=0.0).mean()
            if not mean_pool_ha or mean_pool_ha <= 0:
                continue

            pool_samples_ha.append(mean_pool_ha)
            mean_pool_px = mean_pool_ha / px_area_ha
            yearly_rates.append(mean_n_transitioned / mean_pool_px)
            total_ha_simulated += mean_n_transitioned * px_area_ha

        simulated_rate = float(np.mean(yearly_rates)) if yearly_rates else np.nan
        n_years_used   = len(yearly_rates)
        avg_pool_area_ha = float(np.mean(pool_samples_ha)) if pool_samples_ha else np.nan

        ratio = (
            observed_rate / simulated_rate
            if (n_years_used and not np.isnan(simulated_rate)
                and simulated_rate > 0 and not np.isnan(observed_rate))
            else np.nan
        )

        total_ha_observed = (
            observed_rate * avg_pool_area_ha * n_years_used
            if (n_years_used and not np.isnan(observed_rate) and not np.isnan(avg_pool_area_ha))
            else np.nan
        )

        rows.append({
            "from_class":      from_class,
            "to_class":        to_class,
            "group":           group,
            "observed_rate":   observed_rate,
            "simulated_rate":  simulated_rate,
            "ratio":           ratio,
            "n_years_used":    n_years_used,
            "avg_pool_area_ha": avg_pool_area_ha,
            "total_ha_transitioned_observed":  total_ha_observed,
            "total_ha_transitioned_simulated": total_ha_simulated if n_years_used else np.nan,
        })

    result = pd.DataFrame(rows, columns=_RATE_RATIO_COLUMNS)
    result = result.sort_values(
        by="total_ha_transitioned_simulated",
        key=lambda s: s.fillna(0.0),
        ascending=False,
    ).reset_index(drop=True)

    print(f"  compute_pathway_rate_ratios: {len(result)} (from,to) pair(s) "
          f"({n_unnamed} 'unnamed' -- simulated but absent from "
          f"Transitions.csv)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Shared: apply a per-group scale factor to whichever file actually governs
# that group's sampled multiplier (Uniform -> TransitionMultipliers.csv
# min/max; named -> Distributions.csv Value column)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_group_scales(
    scale_by_group:           dict[str, float],
    transition_mult_csv_path: str | Path,
    distributions_csv_path:   str | Path | None,
    bounds:                   tuple[float, float],
) -> dict[str, pd.DataFrame]:
    """
    Apply scale_by_group[group] to each group's governing file.

    Returns dict with keys "transition_multipliers" (always present) and
    "distributions" (present only if distributions_csv_path was supplied
    and exists).
    """
    mult_df = pd.read_csv(transition_mult_csv_path)
    if "TransitionGroupId" not in mult_df.columns:
        raise ValueError(
            f"'{transition_mult_csv_path}' has no TransitionGroupId column -- "
            f"got columns: {list(mult_df.columns)}"
        )

    lo, hi = bounds
    dist_df = None
    if distributions_csv_path is not None and Path(distributions_csv_path).exists():
        dist_df = pd.read_csv(distributions_csv_path)

    n_uniform_corrected = 0
    n_named_corrected = 0
    n_skipped_no_distfile = 0

    for idx, row in mult_df.iterrows():
        group = _strip_type_suffix(str(row.get("TransitionGroupId", "")))
        scale = scale_by_group.get(group)
        if scale is None:
            continue

        dist_type = str(row.get("DistributionType", "")).strip()
        is_uniform = dist_type.lower() in _LITERAL_DISTRIBUTIONS

        if is_uniform:
            for col in ("DistributionMin", "DistributionMax", "Amount"):
                if col in mult_df.columns and pd.notna(row.get(col)):
                    new_val = float(row[col]) * scale
                    new_val = min(max(new_val, lo), hi)
                    mult_df.at[idx, col] = new_val
            n_uniform_corrected += 1
        else:
            # Named distribution -- DistributionMin/Max are inert at
            # sample time (see core.multipliers._sample); the real
            # correction target is Distributions.csv's Value column for
            # this DistributionTypeId.
            if dist_df is None:
                print(f"  [Warning] Group '{group}' uses named distribution "
                      f"'{dist_type}', but no Distributions.csv was supplied "
                      f"-- scaling DistributionMin/Max would be a no-op "
                      f"(engine ignores them for non-'Uniform' rows), so "
                      f"this group was left UNCORRECTED. Pass "
                      f"distributions_csv_path to correct it.")
                n_skipped_no_distfile += 1
                continue

            mask = dist_df["DistributionTypeId"].astype(str).str.strip() == dist_type
            if not mask.any():
                print(f"  [Warning] DistributionTypeId '{dist_type}' (group "
                      f"'{group}') not found in Distributions.csv -- skipped.")
                continue

            new_vals = dist_df.loc[mask, "Value"].astype(float) * scale
            new_vals = new_vals.clip(lower=lo, upper=hi)
            dist_df.loc[mask, "Value"] = new_vals
            n_named_corrected += 1

    print(f"  correct_multipliers: {n_uniform_corrected} 'Uniform' row(s) "
          f"rescaled in TransitionMultipliers.csv, {n_named_corrected} named "
          f"distribution(s) rescaled in Distributions.csv"
          + (f", {n_skipped_no_distfile} group(s) left uncorrected "
             f"(no Distributions.csv supplied)" if n_skipped_no_distfile else "")
          + f"  (bounds={bounds})")

    result = {"transition_multipliers": mult_df}
    if dist_df is not None:
        result["distributions"] = dist_df
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multiplier correction
# ─────────────────────────────────────────────────────────────────────────────

def correct_multipliers(
    rate_ratios:              pd.DataFrame,
    transition_mult_csv_path: str | Path,
    method:                   str = "scaling",
    bounds:                   tuple[float, float] = (0.01, 100.0),
    distributions_csv_path:   str | Path | None = None,
    # -- method="optimize" only --
    manifest_path:            str | Path | None = None,
    ts=None,
    target_groups:            list[str] | None = None,
    n_iterations_per_trial:   int = 5,
    max_reruns:               int = 8,
) -> dict[str, pd.DataFrame]:
    """
    Correct Transition Multipliers for flagged pathways.

    Parameters
    ----------
    rate_ratios : output of compute_pathway_rate_ratios()
                  (group, observed_rate, simulated_rate, ratio). Used
                  directly as the scale factor for method="scaling";
                  used only to pick which groups to search over for
                  method="optimize" (the ratio itself is just the
                  optimizer's starting guess there).
    transition_mult_csv_path : path to the existing TransitionMultipliers.csv.
                  Read and returned in its ORIGINAL column schema -- rows
                  for unaffected groups are preserved untouched.
    method      : "scaling" (default) or "optimize"
    bounds      : (min, max) clamp on the scale factor applied to
                  DistributionMin/Max (Uniform groups) or Distributions.csv
                  Value entries (named-distribution groups). Default is a
                  wide, PERMISSIVE safety valve, not a validated "safe"
                  range for your project -- it exists so the function
                  can't emit negative or runaway values, not because
                  those bounds are known to keep transition dynamics
                  realistic. Check your own TransitionMultipliers.csv's
                  typical range and tighten this per-project.
    distributions_csv_path : path to Distributions.csv, REQUIRED to
                  actually correct any group whose DistributionType is a
                  named reference rather than literal "Uniform" (see
                  module docstring -- DistributionMin/Max are inert for
                  those groups). If omitted, such groups are flagged and
                  left uncorrected rather than silently doing nothing.

    method="optimize" only
    -----------------------
    manifest_path : calibrated RunManifest.txt (same one hindcast_run()
                  used) -- required, since each trial needs to re-run the
                  engine.
    ts            : LULCTimeSeries covering the hindcast window --
                  required, same reason.
    target_groups : which groups to search over; defaults to every group
                  in rate_ratios with a non-NaN ratio.
    n_iterations_per_trial : engine iterations per trial re-run (default
                  5 -- cheaper than hindcast_run()'s own default of 20,
                  since this runs up to max_reruns times PER group).
    max_reruns    : max trial re-runs per group (default 8). This is the
                  binding budget constraint -- the bounded search's
                  internal maxiter is capped to this value.

    Returns
    -------
    dict with key "transition_multipliers" (always) and "distributions"
    (only if distributions_csv_path was supplied and exists). Neither is
    written to disk -- caller decides.
    """
    if method not in ("scaling", "optimize"):
        raise ValueError(f"Unknown method '{method}'. Use 'scaling' or 'optimize'.")

    if method == "scaling":
        # v3.14: rate_ratios is now pair-level (compute_pathway_rate_ratios()
        # redesign) -- a group spanning multiple (from,to) pairs can appear
        # more than once here with DIFFERING ratios. TransitionMultipliers.csv
        # is keyed by group, not by pair, so only one ratio per group can be
        # applied; a naive dict build would silently keep whichever row
        # happened to iterate last. Detect and flag this explicitly instead.
        valid = rate_ratios[pd.notna(rate_ratios["ratio"])]
        group_ratio_counts = valid.groupby("group")["ratio"].nunique()
        collided = group_ratio_counts[group_ratio_counts > 1]
        if not collided.empty:
            print(
                f"  [Warning] {len(collided)} group(s) span multiple (from,to) "
                f"pairs with DIFFERING ratios -- TransitionMultipliers.csv can "
                f"only apply one scale per group, so only the LAST pair's "
                f"ratio is used for each: {sorted(collided.index)}. "
                f"Recommended fix: split these into separately-calibrated "
                f"TransitionTypeId groups (one pair per group) at the "
                f"group_map stage so each group maps to exactly one ratio."
            )
        scale_by_group = {
            row["group"]: row["ratio"]
            for _, row in valid.iterrows()
        }
        return _apply_group_scales(
            scale_by_group, transition_mult_csv_path, distributions_csv_path, bounds
        )

    # ── method == "optimize" ──────────────────────────────────────────────
    if manifest_path is None or ts is None:
        raise ValueError(
            "method='optimize' requires manifest_path and ts (each trial "
            "needs to re-run the engine over the hindcast window)."
        )

    from scipy.optimize import minimize_scalar
    from .hindcast import hindcast_run
    import tempfile

    groups = target_groups or [
        g for g in rate_ratios.loc[pd.notna(rate_ratios["ratio"]), "group"]
    ]

    scale_by_group: dict[str, float] = {}

    for group in groups:
        starting_ratio = float(
            rate_ratios.loc[rate_ratios["group"] == group, "ratio"].iloc[0]
        )
        n_evals = 0

        def objective(scale: float) -> float:
            nonlocal n_evals
            n_evals += 1
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                trial = _apply_group_scales(
                    {group: scale}, transition_mult_csv_path,
                    distributions_csv_path, bounds,
                )
                trial_mult_path = tmp / "TransitionMultipliers.csv"
                trial["transition_multipliers"].to_csv(trial_mult_path, index=False)

                trial_dist_path = None
                if "distributions" in trial:
                    trial_dist_path = tmp / "Distributions.csv"
                    trial["distributions"].to_csv(trial_dist_path, index=False)

                result = hindcast_run(
                    manifest_path=manifest_path, ts=ts,
                    n_iterations=n_iterations_per_trial,
                    out_dir=tmp / "trial_run",
                    transition_mult_csv_override=trial_mult_path,
                    distributions_csv_override=trial_dist_path,
                    lightweight=True,
                )
                ec = result.extent_comparison
                final_year = ec["year"].max()
                sse = (ec.loc[ec["year"] == final_year, "abs_diff"] ** 2).sum()
                return float(sse)

        opt_result = minimize_scalar(
            objective, bounds=bounds, method="bounded",
            options={"maxiter": max_reruns, "xatol": 1e-2},
        )
        print(f"  [optimize] group='{group}': best_scale={opt_result.x:.4f} "
              f"(started from ratio={starting_ratio:.4f}, "
              f"{n_evals} re-run(s), converged={opt_result.success})")
        scale_by_group[group] = float(opt_result.x)

    return _apply_group_scales(
        scale_by_group, transition_mult_csv_path, distributions_csv_path, bounds
    )
