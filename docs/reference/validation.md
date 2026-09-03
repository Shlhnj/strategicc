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

**Version history — this section was stale.** The two caveats below described v3.13's behavior and were still shown as current in this doc; checking against the installed `strategicc==3.22` source shows both have since been fixed:

- **Grouping.** As of v3.16 (reverting a brief v3.14 change to pair-level output), `compute_pathway_rate_ratios()` pools rates at the `TransitionTypeId` **group** level consistently on both the observed and simulated sides — the v3.13-era mismatch (unweighted mean on one side, pool-weighted on the other) no longer applies. Pair-level rates can still be computed directly from `trans_df`/`area_df` if needed.
- **Units.** `px_area_ha` is now a **required** parameter (added at v3.14, kept since): the hectare-denominated `area_df` pool is converted to a pixel count via `px_area_ha` before dividing, so numerator and denominator are in the same units. Pre-v3.14 output (lacking this parameter) was off by a constant factor of `1/px_area_ha`.

```python
rate_ratios = compute_pathway_rate_ratios(
    result.trans_df, result.area_df, "calibration_result/Transitions.csv",
    px_area_ha=engine.px_area_ha,   # required
    n_timesteps=22,                 # accepted for call-site compatibility but currently unused —
                                     # rates are computed per-year regardless of this value
)
corrected = correct_multipliers(rate_ratios, "TransitionMultipliers.csv")
```

`correct_multipliers()` also now accepts several parameters not shown in the minimal call above — `method`, `distributions_csv_path`, `manifest_path`, `ts`, `target_groups`, `n_iterations_per_trial`, `max_reruns` — suggesting an iterative rerun-based correction mode beyond simple scaling exists; this page doesn't yet cover that mode in detail, check the function's own docstring (`method="scaling"` is the default and matches everything described below).

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
