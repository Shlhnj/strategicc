# `strategicc.outputs`

Aggregating across iterations and plotting, the bridge between `StrategiccEngine.run()`'s raw per-iteration outputs and `SEEAAccount`. This is a standalone module, not a subpackage, so it's always accessed as `strategicc.outputs`:

```python
from strategicc import outputs
```

## The aggregation pipeline

These three are typically called in sequence after `engine.run()`:

```python
area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)

modal_maps = outputs.aggregate_spatial(
    iter_dirs=engine.iter_dirs, start_year=engine.start_year,
    n_timesteps=engine.n_timesteps, src_tags=engine.src_tags,
    summary_dir=summary_dir, uncertainty=True,
)

area_modal_df = outputs.modal_to_area_table(
    modal_maps=modal_maps, classes=engine.classes,
    px_area=engine.px_area, area_unit=engine.area_unit,
)
```

| Function | Purpose |
|---|---|
| `build_summary_tables(iter_dirs, summary_dir)` | Concatenates every iteration's raw `area_table.csv` and `transition_log.csv` into two combined DataFrames |
| `aggregate_spatial(iter_dirs, ..., uncertainty=True)` | Loads each timestep's LULC raster across all iterations, computes the modal (most frequent) class per cell, and optionally an agreement-percentage uncertainty raster. Processes one timestep at a time to keep memory bounded, important at 100+ iterations |
| `modal_to_area_table(modal_maps, classes, px_area, area_unit)` | Converts the modal rasters into an area table (same schema as `area_df`, but spatially consistent, this is what `SEEAAccount` actually uses) |

`area_df` (raw, all iterations) and `area_modal_df` (derived from the modal maps) serve different purposes: `area_df` powers the uncertainty band in `plot_area_envelope()` and `SEEAAccount.uncertainty_summary()`; `area_modal_df` is the spatially-consistent input every other SEEA-EA account is computed from.

## Plots

| Function | Produces |
|---|---|
| `plot_area_envelope(area_df, classes, summary_dir)` | Area per class over time, median line + min/max uncertainty band |
| `plot_transition_envelope(trans_df, summary_dir)` | Total transitions per year, with a per-group stacked breakdown |
| `plot_lulc_maps(maps, classes, start_year, out_dir)` | A row of LULC map panels, one per timestep, diagnostic, single iteration |
| `plot_transition_maps(transitions, shape, classes, start_year, out_dir)` | A row of transition-event maps, one per timestep, diagnostic, single iteration |
| `plot_spatial_summary(initial_lulc, modal_maps, classes, ..., uncertainty=True)` | t=0 vs mid vs final modal maps, with an uncertainty row below |

## Output files

`aggregate_spatial()` writes `lulc_mean_{year}.tif` (the modal raster) and, if `uncertainty=True`, `uncertainty_{year}.tif` (0-100, percent of iterations agreeing with the modal class) into `{summary_dir}/spatial/`, these are the files `strategicc.animate()` reads to build its map panel, so they need to exist before calling `animate()`.
