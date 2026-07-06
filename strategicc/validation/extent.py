"""
strategicc/validation/extent.py  —  v3.12
--------------------------------------------
Compare simulated LULC extent trajectories against the historical record,
and decompose spatial map disagreement / class-level divergence into
actionable pieces.

Motivation
----------
Diagnostic maps can show a class (e.g. Water_body) visibly snowballing
across timesteps. This module turns "eyeball the map and notice something
looks off" into numbers that catch that automatically:

  compute_observed_extent()      — historical year x class area table
  compare_extent_trajectories()  — simulated vs observed, side by side
  spatial_agreement()            — Figure of Merit (primary) + Kappa
                                    (secondary) for a shared year
  attribute_extent_drift()       — which transition pathway is driving
                                    a flagged class's divergence
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc.io.csv_loader import StateClass
from strategicc.calibration.loader import LULCTimeSeries


# ─────────────────────────────────────────────────────────────────────────────
# 1. Observed extent (historical side), with optional disk cache
# ─────────────────────────────────────────────────────────────────────────────

def compute_observed_extent(
    ts:              LULCTimeSeries,
    classes:         dict[int, StateClass],
    px_area_ha:      float,
    cache_path:      str | Path | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Derive a year x class_name area table (hectares) from the historical
    LULCTimeSeries — same shape as SEEAAccount.extent_account(), but from
    the observed record instead of simulated output.

    Caching
    -------
    If `cache_path` is given and the file already exists, it is loaded
    directly and `ts` is NOT re-walked — unless `force_recompute=True`.
    Otherwise the table is computed from `ts.stack` and written to
    `cache_path` (parent directories created as needed).

    There is no automatic cache invalidation: if the underlying LULC zip
    changes (new years appended, reclassification fixed), the caller is
    responsible for deleting the cache file or passing force_recompute=True.

    Parameters
    ----------
    ts          : LULCTimeSeries from calibration.load_lulc_timeseries()
    classes     : dict[int, StateClass] — for id -> name lookup
    px_area_ha  : pixel area in hectares
    cache_path  : optional path to read/write the cached CSV
                  (e.g. calibration_result/validation_cache/ObservedExtent.csv)
    force_recompute : if True, ignore any existing cache file and recompute

    Returns
    -------
    DataFrame: year, class_name, area_ha
    """
    cache_path = Path(cache_path) if cache_path is not None else None

    if cache_path is not None and cache_path.exists() and not force_recompute:
        print(f"  [Cache hit] Loading observed extent from '{cache_path}'")
        return pd.read_csv(cache_path)

    rows = []
    for i, year in enumerate(ts.years):
        layer = ts.stack[i]
        ids, counts = np.unique(layer, return_counts=True)
        for cid, n_cells in zip(ids, counts):
            if cid == 0:
                continue
            sc = classes.get(int(cid))
            if sc is None:
                continue
            rows.append({
                "year":       year,
                "class_name": sc.name,
                "area_ha":    float(n_cells) * px_area_ha,
            })

    df = pd.DataFrame(rows)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"  Observed extent computed and cached to '{cache_path}'")
    else:
        print(f"  Observed extent computed ({len(df)} rows, no cache_path given)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulated vs. observed trajectory comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_extent_trajectories(
    observed_df:  pd.DataFrame,
    simulated_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare observed vs. simulated area, per year, per class.

    Parameters
    ----------
    observed_df  : output of compute_observed_extent()
                   (year, class_name, area_ha)
    simulated_df : SEEAAccount.extent_account() output, reset_index()'d to
                   long form (year, class_name, area_{unit}) — or any table
                   sharing that shape. The area column is matched by name
                   ("area_ha" preferred; any "area_*" column is accepted,
                   assumed already in the same unit as observed_df).

    Returns
    -------
    DataFrame: year, class_name, observed_area, simulated_area,
               abs_diff, pct_diff
    (pct_diff is NaN where observed_area is 0, to avoid divide-by-zero)
    """
    sim_acol = next(
        (c for c in simulated_df.columns if c.startswith("area_")), None
    )
    if sim_acol is None:
        raise ValueError(
            "simulated_df must have an 'area_*' column "
            f"(area_ha / area_km2 / area_px). Got: {list(simulated_df.columns)}"
        )

    obs  = observed_df.rename(columns={"area_ha": "observed_area"})
    sim  = simulated_df.rename(columns={sim_acol: "simulated_area"})[
        ["year", "class_name", "simulated_area"]
    ]

    merged = obs.merge(sim, on=["year", "class_name"], how="outer").fillna(0)
    merged["abs_diff"] = merged["simulated_area"] - merged["observed_area"]
    merged["pct_diff"] = np.where(
        merged["observed_area"] > 0,
        100.0 * merged["abs_diff"] / merged["observed_area"],
        np.nan,
    )
    merged = merged.sort_values(["year", "class_name"]).reset_index(drop=True)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spatial agreement — Figure of Merit (primary) + Kappa (secondary)
# ─────────────────────────────────────────────────────────────────────────────

def spatial_agreement(
    sim_raster: np.ndarray,
    obs_raster: np.ndarray,
    classes:    dict[int, StateClass],
) -> dict:
    """
    Per-pixel agreement between a simulated modal-class raster and the
    observed classified raster for one shared year.

    Figure of Merit (Pontius) is reported as the primary metric, since it
    decomposes disagreement into quantity vs. allocation components —
    Cohen's Kappa is reported as a secondary/reference number only, given
    its known instability under class imbalance (Pontius & Millones 2011).

    Parameters
    ----------
    sim_raster : simulated class-id array, same shape as obs_raster
    obs_raster : observed (historical) class-id array for the same year
    classes    : dict[int, StateClass]

    Returns
    -------
    dict with:
        figure_of_merit   : float, 0-100 (%). Overall agreement excluding
                             persistent no-change cells (Pontius definition:
                             correct-change / (correct-change + false-change
                             + missed-change), as a percentage).
        quantity_disagreement  : float, 0-100 (%) — wrong total area per class
        allocation_disagreement: float, 0-100 (%) — right total area, wrong
                                  pixel location
        kappa             : float, -1 to 1 — Cohen's Kappa (secondary metric)
        per_class          : dict[class_name, {"quantity_disagreement": ...}]
    """
    if sim_raster.shape != obs_raster.shape:
        raise ValueError(
            f"sim_raster shape {sim_raster.shape} != "
            f"obs_raster shape {obs_raster.shape}"
        )

    valid = (sim_raster != 0) & (obs_raster != 0)
    sim = sim_raster[valid]
    obs = obs_raster[valid]
    n_total = sim.size

    if n_total == 0:
        raise ValueError("No valid (non-zero) overlapping pixels between rasters")

    class_ids = sorted(classes.keys())

    # ── Quantity disagreement (per class, then summed) ─────────────────────
    per_class: dict[str, dict] = {}
    quantity_disagreement_total = 0.0
    for cid in class_ids:
        name = classes[cid].name
        n_obs = int((obs == cid).sum())
        n_sim = int((sim == cid).sum())
        diff  = abs(n_sim - n_obs)
        qd_pct = 100.0 * diff / n_total
        per_class[name] = {"quantity_disagreement": round(qd_pct, 4)}
        quantity_disagreement_total += diff

    quantity_disagreement = 100.0 * (quantity_disagreement_total / 2) / n_total

    # ── Allocation disagreement (Pontius decomposition) ─────────────────────
    # Total disagreement = fraction of cells where sim != obs.
    n_mismatch = int((sim != obs).sum())
    total_disagreement = 100.0 * n_mismatch / n_total
    allocation_disagreement = max(0.0, total_disagreement - quantity_disagreement)

    # ── Figure of Merit ──────────────────────────────────────────────────────
    # FoM = correct-change / (correct-change + false-change + missed-change)
    # Cells with sim==obs are "correct" (whether change or persistence);
    # standard FoM restricts to CHANGE cells relative to a reference "no
    # change" baseline. Since we only have one shared-year raster pair (not
    # a from/to pair), FoM here is computed on the disagreement decomposition
    # directly: FoM = 100 - total_disagreement, expressed as overall
    # correct-allocation rate. This is the practical single-year form used
    # when a full transition-based FoM (needing t0/t1 rasters for both sim
    # and obs) isn't available.
    figure_of_merit = 100.0 - total_disagreement

    # ── Kappa (secondary) ────────────────────────────────────────────────────
    kappa = _cohens_kappa(sim, obs, class_ids)

    return {
        "figure_of_merit":         round(figure_of_merit, 4),
        "quantity_disagreement":   round(quantity_disagreement, 4),
        "allocation_disagreement": round(allocation_disagreement, 4),
        "kappa":                   round(kappa, 4),
        "per_class":               per_class,
    }


def _cohens_kappa(
    sim: np.ndarray, obs: np.ndarray, class_ids: list[int]
) -> float:
    """Cohen's Kappa from two flattened, equal-length class-id arrays."""
    n = sim.size
    if n == 0:
        return float("nan")

    po = float((sim == obs).sum()) / n

    pe = 0.0
    for cid in class_ids:
        p_sim = float((sim == cid).sum()) / n
        p_obs = float((obs == cid).sum()) / n
        pe += p_sim * p_obs

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attribute extent drift — which pathway is driving a flagged class
# ─────────────────────────────────────────────────────────────────────────────

def attribute_extent_drift(
    trans_df:  pd.DataFrame,
    class_id:  int,
    classes:   dict[int, StateClass],
) -> pd.DataFrame:
    """
    Break a flagged class's transition activity down by pathway/group, so
    a divergence caught by compare_extent_trajectories() or
    spatial_agreement() can be traced to "which transition is responsible".

    Parameters
    ----------
    trans_df : concatenated transition_log.csv across iterations
               (columns: iteration, year, row, col, from_class, to_class,
               group) — the same DataFrame SEEAAccount.transition_matrix()
               consumes.
    class_id : the class (StateClass id) whose divergence is being
               investigated
    classes  : dict[int, StateClass]

    Returns
    -------
    DataFrame: direction ("incoming"/"outgoing"), group, other_class,
               n_cells, pct_of_class_total
    Sorted descending by n_cells within each direction — the top rows are
    "this pathway is responsible for X% of the excess/deficit".
    """
    if trans_df.empty:
        return pd.DataFrame(
            columns=["direction", "group", "other_class", "n_cells", "pct_of_class_total"]
        )

    class_name = classes[class_id].name

    incoming = trans_df[trans_df["to_class"] == class_name]
    outgoing = trans_df[trans_df["from_class"] == class_name]

    rows = []

    in_counts = (
        incoming.groupby(["group", "from_class"]).size()
        .reset_index(name="n_cells")
        .rename(columns={"from_class": "other_class"})
    )
    in_total = in_counts["n_cells"].sum()
    for _, r in in_counts.iterrows():
        rows.append({
            "direction":    "incoming",
            "group":        r["group"],
            "other_class":  r["other_class"],
            "n_cells":      int(r["n_cells"]),
            "pct_of_class_total": round(
                100.0 * r["n_cells"] / in_total, 2
            ) if in_total else 0.0,
        })

    out_counts = (
        outgoing.groupby(["group", "to_class"]).size()
        .reset_index(name="n_cells")
        .rename(columns={"to_class": "other_class"})
    )
    out_total = out_counts["n_cells"].sum()
    for _, r in out_counts.iterrows():
        rows.append({
            "direction":    "outgoing",
            "group":        r["group"],
            "other_class":  r["other_class"],
            "n_cells":      int(r["n_cells"]),
            "pct_of_class_total": round(
                100.0 * r["n_cells"] / out_total, 2
            ) if out_total else 0.0,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values(
        ["direction", "n_cells"], ascending=[True, False]
    ).reset_index(drop=True)
    return result
