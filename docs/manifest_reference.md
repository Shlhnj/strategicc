# `RunManifest.txt` Field Reference

`RunManifest.txt` is an alternative to setting `strategicc.config` attributes directly; it lets you define an entire scenario in one plain-text file, loaded with:

```python
import strategicc.config as cfg
cfg.load_manifest("RunManifest.txt")
```

See [`config` reference](reference/config.md) for the rules around manifest mode vs direct mode (they're mutually exclusive within a session).

## Format

```
Variable = Value          #type  (optional description)
```

Lines starting with `#` are comments and ignored. The `#type` hint is for human reference only, the parser uses a fixed internal schema, not the hint text, to decide how to cast each value. Recognised types: `int`, `float`, `bool` (accepts `True`/`False` or `Yes`/`No`), `path`, `str`.

Documentation/example content (the file's own header explanation, and an appendix of filled CSV examples) is wrapped in triple-backtick fences and skipped entirely during parsing, this means example text can safely say things like `Variable = Value` without being mistaken for a real config line.

A full example file with every section filled in and a worked CSV example appendix ships in `inputs/RunManifest.txt`.

## Section 1 Initial Conditions (Spatial)

| Variable | Maps to | Type |
|---|---|---|
| `StateClassFileName` | `LULC_PATH` | path |
| `AgeFileName` | `AGE_RASTER_PATH` | path |
| `StratumFileName` / `SecondaryStratumFileName` / `TertiaryStratumFileName` | (reserved, not yet implemented) | path |
| `FetchInitialStateClassFromZip` | `FETCH_INITIAL_SC_FROM_ZIP` | bool |
| `LULCZipPath` | `LULC_ZIP_PATH` | path |
| `InitialStateClassYear` | `INITIAL_SC_YEAR` | int |

If `FetchInitialStateClassFromZip=True`, `StateClassFileName` is ignored and the initial raster is instead extracted from `LULCZipPath` for the year `InitialStateClassYear`, see [calibration reference](reference/calibration.md).

## Section 2 Multi-row CSV Inputs

| Variable | Maps to |
|---|---|
| `STATE_CLASSES_CSV` | `STATE_CLASSES_CSV` |
| `TRANSITIONS_CSV` | `TRANSITIONS_CSV` |
| `SPATIAL_MULT_CSV` | `SPATIAL_MULT_CSV` |
| `TRANSITION_MULT_CSV` | `TRANSITION_MULT_CSV` |
| `TRANSITION_TYPE_CSV` | `TRANSITION_TYPE_CSV` |
| `ECOSYSTEM_SERVICES_CSV` | `ECOSYSTEM_SERVICES_CSV` |
| `AGE_INITIAL_CSV` | `AGE_INITIAL_CSV` |
| `TRANSITION_SIZE_CSV` | `TRANSITION_SIZE_CSV` |
| `TRANSITION_ADJACENCY_SETTING_CSV` / `TRANSITION_ADJACENCY_MULT_CSV` | same names |
| `TERMINOLOGY_CSV` | `TERMINOLOGY_CSV` |
| `MULT_DIR` | `MULT_DIR` — folder or `.zip` (auto-extracted once to a sibling folder) |
| `OUT_DIR` | `OUT_DIR` |

All `path` type.

## Section 3 Run Control

| Variable | Maps to | Type | Notes |
|---|---|---|---|
| `START_YEAR` | `START_YEAR` | int | |
| `N_TIMESTEPS` | `N_TIMESTEPS` | int | |
| `N_ITERATIONS` | `N_ITERATIONS` | int | |
| `RNG_SEED` | `RNG_SEED` | int | |
| `AREA_UNIT` | `AREA_UNIT` | str | `ha`, `km2`, or `px` |

## Section 4 Feature Toggles

| Variable | Maps to |
|---|---|
| `USE_ADJACENCY` | `USE_ADJACENCY` |
| `USE_SPATIAL_MULT` | `USE_SPATIAL_MULT` |
| `USE_TRANS_MULTIPLIER` | `USE_TRANS_MULTIPLIER` |
| `USE_SEEA` | `USE_SEEA` |
| `USE_AGE` | `USE_AGE` |
| `SAVE_AGE_RASTERS` | `SAVE_AGE_RASTERS` |

All `bool` type.

## Section 5 Output Options (Non-Spatial)

| Variable | Maps to | Type |
|---|---|---|
| `SummaryOutputSC` | `SUMMARY_OUTPUT_SC` | bool |
| `SummaryOutputSCTimesteps` | `SUMMARY_OUTPUT_SC_TIMESTEPS` | int |
| `SummaryOutputTR` | `SUMMARY_OUTPUT_TR` | bool |
| `SummaryOutputTRTimesteps` | `SUMMARY_OUTPUT_TR_TIMESTEPS` | int |

Controls whether `area_table.csv` / `transition_log.csv` are saved per iteration, and at what timestep stride.

## Section 6 Output Options (Spatial)

| Variable | Maps to | Type |
|---|---|---|
| `RasterOutputSC` / `RasterOutputSCTimesteps` | `RASTER_OUTPUT_SC` / `_TIMESTEPS` | bool / int |
| `RasterOutputAge` / `RasterOutputAgeTimesteps` | `RASTER_OUTPUT_AGE` / `_TIMESTEPS` | bool / int |
| `RasterOutputTransitionEvents` / `RasterOutputTransitionEventTimesteps` | `RASTER_OUTPUT_TRANSITION_EVENTS` / `_TIMESTEPS` | bool / int |

`*Timesteps` fields control the save stride (e.g. `RasterOutputSCTimesteps=2`) saves every second timestep's LULC raster instead of every one (the final timestep is always saved regardless of stride).

## Section 7 Stock & Flow

| Variable | Maps to | Type |
|---|---|---|
| `USE_STOCKFLOW` | `USE_STOCKFLOW` | bool |
| `STOCK_TYPE_CSV` | `STOCK_TYPE_CSV` | path |
| `STOCK_GROUP_CSV` | `STOCK_GROUP_CSV` | path |
| `STOCK_GROUP_MEMBERSHIP_CSV` | `STOCK_GROUP_MEMBERSHIP_CSV` | path |
| `FLOW_TYPE_CSV` | `FLOW_TYPE_CSV` | path |
| `FLOW_ORDER_CSV` | `FLOW_ORDER_CSV` | path |
| `FLOW_PATHWAYS_CSV` | `FLOW_PATHWAYS_CSV` | path |
| `FLOW_MULTIPLIER_CSV` | `FLOW_MULTIPLIER_CSV` | path |
| `STATE_ATTRIBUTE_TYPE_CSV` | `STATE_ATTRIBUTE_TYPE_CSV` | path |
| `STATE_ATTRIBUTE_VALUES_CSV` | `STATE_ATTRIBUTE_VALUES_CSV` | path |
| `INITIAL_STOCK_NON_SPATIAL_CSV` | `INITIAL_STOCK_NON_SPATIAL_CSV` | path |
| `SAVE_STOCK_RASTERS` | `SAVE_STOCK_RASTERS` | bool |
| `SEEA_VALUATION_MODE` | `SEEA_VALUATION_MODE` | str — `area` or `stock_flow` |

See [Guide 3](guides/03_stockflow_full.md) and [`stockflow` reference](reference/stockflow.md) for what each CSV actually contains.

## Minimal example

```
StateClassFileName = 2022.tif #path
AgeFileName = #path

STATE_CLASSES_CSV = inputs/StateClasses.csv #path
TRANSITIONS_CSV = inputs/Transitions.csv #path
ECOSYSTEM_SERVICES_CSV = inputs/EcosystemServices.csv #path
OUT_DIR = output/ #path

START_YEAR = 2022 #int
N_TIMESTEPS = 10 #int
N_ITERATIONS = 20 #int

USE_ADJACENCY = True #bool
USE_SEEA = True #bool
```

Any field not present in the file simply keeps whatever default `strategicc/config.py` already has — you don't need to fill in every section, only the parts relevant to your scenario.
