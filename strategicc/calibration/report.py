"""
strategicc/calibration/report.py  —  v3.8
-------------------------------------------
Calibration result summary: printed table, age raster map,
age histogram, and returned dict for programmatic access.

Functions
---------
calibration_summary  — display full calibration result after a run
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from strategicc.calibration.age import AgeRasterResult, NODATA_AGE


# ── helpers ───────────────────────────────────────────────────────────────────

def _divider(char: str = "═", width: int = 58) -> str:
    return char * width


def _tick(label: str, detail: str = "") -> str:
    return f"  ✓  {label:<32s}{detail}"


def _cross(label: str, detail: str = "") -> str:
    return f"  ✗  {label:<32s}{detail}"  # noqa: RUF001


def _fmt_prob(series: pd.Series) -> str:
    lo, hi = series.min(), series.max()
    if lo == hi:
        return f"{lo:.4f}"
    return f"{lo:.4f} – {hi:.4f}"


# ── main function ─────────────────────────────────────────────────────────────

def calibration_summary(
    transitions_df:    pd.DataFrame | None = None,
    temporal_df:       pd.DataFrame | None = None,
    size_dist_df:      pd.DataFrame | None = None,
    age_result:        AgeRasterResult | None = None,
    transitions_path:  Path | None = None,
    temporal_path:     Path | None = None,
    size_dist_path:    Path | None = None,
    age_raster_path:   Path | None = None,
    manifest_path:     Path | None = None,
    plot_out:          Path | None = None,
) -> dict:
    """
    Print a calibration run summary, plot the age raster and histogram,
    and return a dict of key statistics for programmatic access.

    Parameters
    ----------
    transitions_df   : DataFrame from compute_transition_rates()
    temporal_df      : DataFrame from compute_temporal_distribution()
    size_dist_df     : DataFrame from compute_size_distribution()
    age_result       : AgeRasterResult from compute_age_raster()
    transitions_path : path where Transitions.csv was saved
    temporal_path    : path where TransitionMultipliers.csv was saved
    size_dist_path   : path where TransitionSizeDistribution.csv was saved
    age_raster_path  : path where age.tif was saved
    manifest_path    : path where RunManifest_calibrated.txt was saved
    plot_out         : destination for the summary plot PNG;
                       defaults to calibration_result/calibration_summary.png

    Returns
    -------
    dict with keys:
        transitions, temporal, size_distribution, age, manifest, plot_path
    """
    from strategicc.calibration.paths import CALIBRATION_DIR

    if plot_out is None:
        plot_out = CALIBRATION_DIR / "calibration_summary.png"
    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "transitions":      None,
        "temporal":         None,
        "size_distribution": None,
        "age":              None,
        "manifest":         None,
        "plot_path":        None,
    }

    # ── Printed summary ───────────────────────────────────────────────────────
    print(_divider())
    print("  STRATEGICC Calibration Summary")
    print(_divider())

    # Transitions.csv
    if transitions_df is not None and not transitions_df.empty:
        n   = len(transitions_df)
        rng = _fmt_prob(transitions_df["Probability"])
        print(_tick("Transitions.csv",
                    f"{n} pathway(s)  prob: {rng}"))
        result["transitions"] = {
            "n_pathways": n,
            "prob_min":   float(transitions_df["Probability"].min()),
            "prob_max":   float(transitions_df["Probability"].max()),
            "path":       str(transitions_path) if transitions_path else None,
            "data":       transitions_df,
        }
    else:
        print(_cross("Transitions.csv", "not calibrated"))

    # TransitionMultipliers.csv
    if temporal_df is not None and not temporal_df.empty:
        n    = temporal_df["TransitionGroupId"].nunique()
        lo   = temporal_df["DistributionMin"].min()
        hi   = temporal_df["DistributionMax"].max()
        print(_tick("TransitionMultipliers.csv",
                    f"{n} group(s)  mult: {lo:.2f} – {hi:.2f}"))
        result["temporal"] = {
            "n_groups":   n,
            "mult_min":   float(lo),
            "mult_max":   float(hi),
            "path":       str(temporal_path) if temporal_path else None,
            "data":       temporal_df,
        }
    else:
        print(_cross("TransitionMultipliers.csv", "not calibrated"))

    # TransitionSizeDistribution.csv
    if size_dist_df is not None and not size_dist_df.empty:
        n_groups = size_dist_df["Transition Type/Group"].nunique()
        lo = size_dist_df["Maximum Area (Hectares)"].min()
        hi = size_dist_df["Maximum Area (Hectares)"].max()
        print(_tick("TransitionSizeDistribution.csv",
                    f"{n_groups} group(s)  patch: {lo:.2f} – {hi:.2f} ha"))
        result["size_distribution"] = {
            "n_groups":      n_groups,
            "patch_min_ha":  float(lo),
            "patch_max_ha":  float(hi),
            "path":          str(size_dist_path) if size_dist_path else None,
            "data":          size_dist_df,
        }
    else:
        print(_cross("TransitionSizeDistribution.csv", "not calibrated"))

    # Age raster
    if age_result is not None:
        valid = age_result.age_combined[
            age_result.age_combined != NODATA_AGE
        ]
        n_classes = len(age_result.age_per_class)
        print(_tick("Age raster",
                    f"{n_classes} class(es)  age: {int(valid.min())} – "
                    f"{int(valid.max())} yrs  → {age_raster_path or 'not saved'}"))
        result["age"] = {
            "n_classes":      n_classes,
            "age_min":        int(valid.min()),
            "age_max":        int(valid.max()),
            "age_mean":       float(valid.mean()),
            "n_full_record":  int(age_result.full_record_mask.sum()),
            "path":           str(age_raster_path) if age_raster_path else None,
        }
    else:
        print(_cross("Age raster", "not calibrated"))

    # Manifest
    if manifest_path is not None and Path(manifest_path).exists():
        print(_tick("RunManifest_calibrated.txt",
                    f"pre-filled → {manifest_path}"))
        result["manifest"] = str(manifest_path)
    else:
        print(_cross("RunManifest_calibrated.txt", "not generated"))

    # TODO section
    print(_divider("─"))
    print("  TODO (manual):")
    todos = [
        ("ECOSYSTEM_SERVICES_CSV",          "service values per class"),
        ("N_TIMESTEPS, N_ITERATIONS",        "run control"),
        ("TransitionSpatialMultipliers.csv", "not derivable from rasters alone"),
        ("STATE_CLASSES_CSV",               "if not yet prepared"),
    ]
    for field, note in todos:
        print(f"  ✗  {field:<32s}# {note}")
    print(_divider())
    print(f"  All calibration outputs → {CALIBRATION_DIR}/")
    print(_divider())

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_calibration_summary(
        age_result      = age_result,
        transitions_df  = transitions_df,
        temporal_df     = temporal_df,
        size_dist_df    = size_dist_df,
        out_path        = plot_out,
    )
    result["plot_path"] = str(plot_out)

    return result


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_calibration_summary(
    age_result:     AgeRasterResult | None,
    transitions_df: pd.DataFrame | None,
    temporal_df:    pd.DataFrame | None,
    size_dist_df:   pd.DataFrame | None,
    out_path:       Path,
) -> None:
    """
    Multi-panel calibration summary figure:
      Row 1 (if age):   age raster map  |  age histogram (max 10 bins shown)
      Row 2 (if trans): transition probability bar chart
      Row 3 (if temp):  temporal multiplier range bar chart
      Row 4 (if size):  patch size distribution bar chart

    Saves to out_path as PNG.
    """
    has_age   = age_result is not None
    has_trans = transitions_df is not None and not transitions_df.empty
    has_temp  = temporal_df   is not None and not temporal_df.empty
    has_size  = size_dist_df  is not None and not size_dist_df.empty

    n_rows = int(has_age) + int(has_trans) + int(has_temp) + int(has_size)
    if n_rows == 0:
        return

    # age row is double-wide (map + histogram), so counts as 1 row with 2 cols
    fig = plt.figure(figsize=(14, 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    row = 0

    # ── Age: raster map + histogram ───────────────────────────────────────────
    if has_age:
        arr   = age_result.age_combined.astype(float)
        nodata = float(NODATA_AGE)
        arr[arr == nodata] = np.nan

        # Panel A: spatial map
        ax_map = fig.add_subplot(gs[row, 0])
        im = ax_map.imshow(arr, cmap="YlOrBr", aspect="auto",
                           interpolation="nearest")
        plt.colorbar(im, ax=ax_map, shrink=0.8, label="Age (years)")
        ax_map.set_title(
            f"Age Raster  (baseline {age_result.baseline_year})",
            fontsize=10, fontweight="bold",
        )
        ax_map.axis("off")

        # Panel B: histogram (max 10 bins → table-style bar chart)
        ax_hist = fig.add_subplot(gs[row, 1])
        valid   = arr[~np.isnan(arr)].flatten()

        n_bins  = min(10, int(np.ptp(valid)) + 1) if valid.size > 0 else 5
        counts, edges = np.histogram(valid, bins=n_bins)
        bin_labels = [
            f"{int(edges[i])}–{int(edges[i+1])}"
            for i in range(len(edges) - 1)
        ]
        # limit to 10 rows (already capped by n_bins ≤ 10)
        ax_hist.barh(bin_labels, counts, color="#e67e22", alpha=0.85)
        ax_hist.set_xlabel("Cell count", fontsize=9)
        ax_hist.set_title("Age Distribution (≤10 bins)", fontsize=10,
                           fontweight="bold")
        ax_hist.invert_yaxis()
        ax_hist.grid(True, axis="x", alpha=0.25)

        # Print the age table (max 10 rows) to console
        table_rows = list(zip(bin_labels, counts))[:10]
        print("\n  Age distribution table:")
        print(f"  {'Age range (yrs)':<20s}  {'Cell count':>10s}  {'%':>6s}")
        print("  " + "-" * 42)
        total = counts.sum()
        for label, cnt in table_rows:
            pct = 100.0 * cnt / total if total > 0 else 0
            print(f"  {label:<20s}  {cnt:>10,d}  {pct:>5.1f}%")

        row += 1

    # ── Transition probabilities ───────────────────────────────────────────────
    if has_trans:
        ax = fig.add_subplot(gs[row, :])
        groups = transitions_df["TransitionTypeId"].tolist()
        probs  = transitions_df["Probability"].tolist()
        colors = plt.cm.tab10(np.linspace(0, 0.6, len(groups)))
        bars = ax.barh(groups, probs, color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
        ax.set_xlabel("Mean annual probability", fontsize=9)
        ax.set_title("Calibrated Transition Probabilities", fontsize=10,
                     fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.25)
        row += 1

    # ── Temporal multiplier range ─────────────────────────────────────────────
    if has_temp:
        ax    = fig.add_subplot(gs[row, :])
        grps  = temporal_df["TransitionGroupId"].str.replace(
            r" \[.*\]$", "", regex=True
        ).tolist()
        lo    = temporal_df["DistributionMin"].tolist()
        hi    = temporal_df["DistributionMax"].tolist()
        y     = np.arange(len(grps))
        mid   = [(l + h) / 2 for l, h in zip(lo, hi)]
        err   = [(h - l) / 2 for l, h in zip(lo, hi)]
        ax.barh(y, [h - l for l, h in zip(lo, hi)], left=lo,
                color="#27ae60", alpha=0.75, label="Multiplier range")
        ax.scatter(mid, y, color="#1a5c38", zorder=3, s=40, label="Midpoint")
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--",
                   label="mean = 1.0")
        ax.set_yticks(y)
        ax.set_yticklabels(grps, fontsize=8)
        ax.set_xlabel("Temporal multiplier", fontsize=9)
        ax.set_title("Temporal Multiplier Range (Uniform distribution)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.25)
        row += 1

    # ── Patch size distribution ───────────────────────────────────────────────
    if has_size:
        ax = fig.add_subplot(gs[row, :])
        for group, grp_df in size_dist_df.groupby("Transition Type/Group"):
            label = group.replace(" [Type]", "")
            ax.plot(
                grp_df["Maximum Area (Hectares)"].tolist(),
                grp_df["Relative Amount"].cumsum().tolist(),
                marker="o", linewidth=1.8, label=label,
            )
        ax.set_xlabel("Maximum patch area (ha)", fontsize=9)
        ax.set_ylabel("Cumulative relative amount", fontsize=9)
        ax.set_title("Transition Size Distribution (cumulative)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.suptitle("STRATEGICC Calibration Summary", fontsize=13, y=1.01,
                 fontweight="bold")
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")
