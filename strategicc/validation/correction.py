"""
strategicc/validation/correction.py  —  v3.12
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
    hindcast window.

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
        scale_by_group = {
            row["group"]: row["ratio"]
            for _, row in rate_ratios.iterrows()
            if pd.notna(row["ratio"])
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
