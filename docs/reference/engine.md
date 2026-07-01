# `strategicc.engine` & `StrategiccEngine`

The core simulation class. Loads inputs, runs N stochastic Monte Carlo iterations of the spatial state-and-transition simulation, and writes per-iteration outputs (LULC rasters, transition logs, age rasters, stock/flow rasters) to disk.

```python
from strategicc import StrategiccEngine
```

## Constructing an engine

Two ways to build an engine, pick one:

**From `strategicc.config`** (recommended; reads all settings from the config module, which user can populate via `config.load_manifest()` or by setting attributes directly):

```python
engine = StrategiccEngine.from_config()
```

**Directly**, passing every parameter explicitly:

```python
engine = StrategiccEngine(
    lulc_path              = "2022.tif",
    state_classes_csv      = "inputs/StateClasses.csv",
    transitions_csv        = "inputs/Transitions.csv",
    spatial_mult_csv       = "inputs/TransitionSpatialMultipliers.csv",
    trans_mult_csv         = "inputs/TransitionMultipliers.csv",
    ecosystem_services_csv = "inputs/EcosystemServices.csv",
    mult_dir               = "spatmult_uploads/",
    out_dir                = "output/",
    start_year             = 2022,
    n_timesteps            = 10,
    n_iterations           = 20,
    rng_seed               = 42,
    use_adjacency          = True,
    use_spatial_mult       = True,
    use_trans_multiplier   = True,
    use_seea               = True,
    use_age                = False,
    use_stockflow          = False,
)
```

Most parameters have sensible defaults, only `lulc_path`, `state_classes_csv`, `transitions_csv`, and `out_dir` are strictly required to get something running. CSVs that don't exist on disk are silently skipped with a printed note (e.g. no `TransitionSizeDistribution.csv` -> all groups use independent-cell firing instead of patch growth).

## The three-step lifecycle

```python
engine.load()         # read all rasters and CSVs
engine.diagnostic()   # optional but recommended -- prints expected transitions per pathway
engine.run()          # run all n_iterations, write outputs to out_dir
```

`load()` must be called before `run()`. `diagnostic()` is optional but catches misconfigurations early, it prints, for every transition pathway, the number of source cells and the expected number of fires given the current probability and multipliers, before committing to a potentially slow Monte Carlo run.

## What `run()` produces

For each iteration `i`, a subfolder `{out_dir}/iter_{i:03d}/` containing:

- `lulc_{year}.tif` --> one LULC raster per timestep
- `transition_log.csv` --> every fired transition (year, row, col, from/to class, group)
- `area_table.csv` --> area per class per timestep
- `age/age_{year}.tif` --> if `use_age=True` and `save_age_rasters=True`
- `stocks/{stock_type}/stock_{year}.tif` --> if `use_stockflow=True` and `cfg.SAVE_STOCK_RASTERS=True`
- `flow_log.csv` / `flow_log_by_class.csv` --> if `use_stockflow=True`
- `transition_events/events_{year}.tif`--> if `RASTER_OUTPUT_TRANSITION_EVENTS` is enabled

`engine.iter_dirs` is a list of `Path` objects pointing at each iteration's output folder,  pass this directly to `strategicc.outputs.aggregate_spatial()` and similar aggregation functions.

## Key attributes after `load()`

| Attribute | Type | Description |
|---|---|---|
| `engine.classes` | `dict[int, StateClass]` | Loaded state classes, keyed by class ID |
| `engine.px_area_ha` / `engine.px_area` | `float` | Pixel area in hectares / in the configured `AREA_UNIT` |
| `engine.ecosystem_services` | `list[EcosystemService]` | Loaded ecosystem service rows |
| `engine._stock_types` | `list[str]` | Loaded stock type names, if `use_stockflow=True` |
| `engine._initial_lulc` | `np.ndarray` | The t=0 LULC raster as a numpy array |

## Feature toggles

All of these default sensibly and can be combined freely:

| Toggle | Enables |
|---|---|
| `use_adjacency` | Neighbour-class weighting of transition probability |
| `use_spatial_mult` | Spatial multiplier rasters (suitability surfaces) |
| `use_trans_multiplier` | Stochastic per-timestep multipliers (temporal variability) |
| `use_seea` | Loading `EcosystemServices.csv` for downstream SEEA-EA accounting |
| `use_age` | Per-cell age tracking, age-gated transitions |
| `use_stockflow` | Carbon (or other material) Stock & Flow tracking |

See [Guide 2](../guides/02_calibration_stsm.md) and [Guide 3](../guides/03_stockflow_full.md) for these in context, and [`core`](core.md) for how each toggle actually affects the firing mechanic.
