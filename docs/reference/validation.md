# `strategicc.validation`

Hindcast validation and calibration-correction tools: run the engine over a
historical period with known outcomes, compare simulated vs. observed
land cover, and derive corrections to close the gap. No reference page
existed for this subpackage before v3.13 despite it being a real,
load-bearing part of the calibration workflow — this page fills that gap.

```python
from strategicc.validation.hindcast import hindcast_run
from strategicc.validation.correction import compute_pathway_rate_ratios, correct_multipliers
from strategicc.validation.extent import spatial_agreement
```

## Running a hindcast

```python
from strategicc.calibration import load_lulc_timeseries

ts = load_lulc_timeseries("annual_lulc_2000_2022.zip")
result = hindcast_run("RunManifest.txt", ts, n_iterations=20)
```

`hindcast_run()` reloads configuration from the given manifest path itself
(it calls `config.load_manifest()` internally) — it does not read whatever
`strategicc.config` currently holds, even if you've already configured a
session via direct attribute assignment. Always pass a real manifest file
path; there's currently no direct-config equivalent for this entry point.

Returns a `HindcastResult`:

| Field | Type | Contents |
|---|---|---|
| `extent_comparison` | `DataFrame` | Per-class simulated vs. observed area, by year |
| `spatial_agreement` | `dict[int, dict]` | year → Pontius Figure-of-Merit decomposition (see below) |
| `drift` | `dict[str, DataFrame]` | class name → drift diagnostics, where populated |
| `flagged_classes` | `list[str]` | Classes whose simulated/observed extent diverged beyond an internal threshold |
| `plot_path` | `Path \| None` | Saved comparison figure, if one was generated |
| `area_df` | `DataFrame \| None` | Per-iteration area table — feeds `compute_pathway_rate_ratios()` below |
| `trans_df` | `DataFrame \| None` | Per-iteration transition log — feeds `compute_pathway_rate_ratios()` below |

## Spatial agreement metrics

```python
metrics = spatial_agreement(sim_raster, obs_raster, classes)
```

Reports Pontius's Figure of Merit decomposition for one shared year:
`figure_of_merit`, `quantity_disagreement`, `allocation_disagreement`, and
`kappa` (Cohen's Kappa, included as a secondary/reference number only —
it's known to be unstable under class imbalance, per Pontius & Millones
2011, so Figure of Merit is the primary metric to read). Also returns
`per_class`, a nested per-class breakdown.

## Correcting calibration from hindcast results

```python
rate_ratios = compute_pathway_rate_ratios(
    result.trans_df, result.area_df, "calibration_result/Transitions.csv", n_timesteps=22,
)
corrected = correct_multipliers(rate_ratios, "TransitionMultipliers.csv")
```

**Version-specific behavior — read before trusting output from this
function.** As of v3.13, `compute_pathway_rate_ratios()`:

- Operates at the **group** (`TransitionTypeId`) level, not at the
  `(from_class, to_class)` pair level. A group spanning multiple pairs
  gets an unweighted mean on the observed side, but a pool-weighted mean
  (all source classes pooled into one shared denominator) on the
  simulated side — these two ways of aggregating don't measure the same
  thing, and the mismatch can hide which specific pair within a group is
  actually miscalibrated.
- Takes **no `px_area_ha` parameter** — the simulated rate is derived
  directly from `area_ha` without converting to a pixel count first,
  which scales every returned ratio by `1/px_area_ha` relative to the
  true probability.

Both are fixed in a later version (pair-level atomicity, explicit
`px_area_ha` correction) but **not in v3.13** — don't treat ratios from
this version as directly comparable across pixel sizes or as reliable at
the individual-pair level within a multi-pair group.

`correct_multipliers()` returns a dict:

| Key | Present when |
|---|---|
| `"transition_multipliers"` | Always |
| `"distributions"` | Only if a `distributions_csv_path` was supplied and matching entries were found |

`bounds` (default `(0.01, 100.0)`) clamps the **resulting value after
multiplication** (`min(max(value * scale, lo), hi)`), not the scale factor
itself — confirmed by reading `_apply_group_scales()` directly, since this
distinction isn't obvious from the parameter name alone and matters for
interpreting what a clamped result actually represents.

Neither function writes to disk — both return DataFrames for the caller
to inspect and save explicitly, consistent with the rest of the package's
"corrections return data, the caller decides whether to persist it"
convention.
