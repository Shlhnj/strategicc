# Guide 4 — Visualization

STRATEGICC produces all plots as PNG files saved directly to disk — no interactive window is shown, so they work correctly in Colab and headless environments. This guide covers every plot the package generates, where to find the output files, and how to display them inline in a notebook.

## Simulation outputs (`strategicc.outputs`)

These are typically called right after `engine.run()` as part of the aggregation pipeline. See [Guide 2](02_calibration_stsm.md) for the full sequence.

### Area envelope

```python
from strategicc import outputs

summary_dir = engine.out_dir / "summary"
area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)

outputs.plot_area_envelope(area_df, engine.classes, summary_dir)
# Saves to: summary_dir/area_envelope.png
```

Shows area per class over time — one line per class (median across iterations) with a shaded min/max uncertainty band. This is the quickest way to see whether your simulated landscape is actually changing and how much spread there is across iterations.

### Transition envelope

```python
outputs.plot_transition_envelope(trans_df, summary_dir)
# Saves to: summary_dir/transition_envelope.png
```

Total transitions per year, stacked by group. Useful for checking whether transition rates are stable, accelerating, or running out of source cells over time.

### Spatial summary

```python
modal_maps = outputs.aggregate_spatial(
    iter_dirs=engine.iter_dirs, start_year=engine.start_year,
    n_timesteps=engine.n_timesteps, src_tags=engine.src_tags,
    summary_dir=summary_dir, uncertainty=True,
)

outputs.plot_spatial_summary(
    initial_lulc=engine._initial_lulc, modal_maps=modal_maps,
    classes=engine.classes, start_year=engine.start_year,
    n_timesteps=engine.n_timesteps, summary_dir=summary_dir,
    uncertainty=True,
)
# Saves to: summary_dir/spatial/spatial_summary.png
```

A three-column grid: t=0, mid-simulation, and final year modal maps. If `uncertainty=True`, a second row shows the agreement raster (0-100%, how many iterations agreed with the modal class per cell) for each of the three snapshots — low agreement areas are where the simulation is genuinely uncertain about the outcome.

### Diagnostic maps (single iteration)

```python
# One LULC panel per timestep, from a single iteration
outputs.plot_lulc_maps(maps_iter1, engine.classes, engine.start_year, diag_dir)
# Saves to: diag_dir/lulc_maps.png

# One transition-event panel per timestep, showing where fires occurred
outputs.plot_transition_maps(transitions_iter1, lulc.shape, engine.classes,
                              engine.start_year, diag_dir)
# Saves to: diag_dir/transition_maps.png
```

These are diagnostic tools for inspecting one iteration's trajectory in detail — useful for verifying that transitions are happening in plausible spatial patterns before committing to a full multi-iteration run.

## SEEA-EA plots (`strategicc.accounting`)

These require `SEEAAccount` to be built first. See [Guide 2](02_calibration_stsm.md) or [Guide 3](03_stockflow_full.md).

### Monetary flows

```python
from strategicc.accounting import (
    SEEAAccount, save_all_accounts,
    plot_monetary_flows, plot_value_by_service, plot_transition_heatmap,
)

seea_dir = engine.out_dir / "seea"
acct = SEEAAccount(...)

plot_monetary_flows(acct, engine.classes, seea_dir)
# Saves to: seea_dir/seea_monetary_flows.png
```

Stacked area chart of total monetary value per class over time, plus a year-on-year change panel below it. The stacked area shows both absolute value and composition; the change panel immediately surfaces which years had the biggest value loss or gain.

### Value by service

```python
plot_value_by_service(acct, seea_dir)
# Saves to: seea_dir/seea_value_by_service.png
```

One line per ecosystem service, total value across the landscape over time. Useful for comparing which services dominate and how their relative contributions shift as the landscape changes.

### Transition heatmap

```python
plot_transition_heatmap(acct, seea_dir)
# Saves to: seea_dir/seea_transition_heatmap.png
```

Two heatmaps side by side: area converted from each class to each class (from `transition_matrix()`), and the implied monetary value change of those conversions. Cells on the diagonal represent no change. Bright off-diagonal cells indicate large conversions — the value-change heatmap immediately shows whether those conversions were economically significant.

## Saving all CSVs and plots

```python
save_all_accounts(acct, seea_dir)          # writes all 7 CSV accounts
plot_monetary_flows(acct, engine.classes, seea_dir)
plot_value_by_service(acct, seea_dir)
plot_transition_heatmap(acct, seea_dir)
```

`save_all_accounts()` saves CSVs only — the three plot functions are separate calls, giving you control over which ones to generate. A common pattern is to always save the CSVs and only generate the plots you actually need for a given figure.

## Displaying plots inline in Colab or Jupyter

All plot functions save to disk and close the figure — nothing is displayed automatically. To show a plot inline after saving:

```python
from IPython.display import Image, display

outputs.plot_area_envelope(area_df, engine.classes, summary_dir)
display(Image(str(summary_dir / "area_envelope.png")))

plot_monetary_flows(acct, engine.classes, seea_dir)
display(Image(str(seea_dir / "seea_monetary_flows.png")))
```

## Changing export resolution

All plots are saved at 150 DPI by default. To change this, edit the `dpi=150` argument in `fig.savefig(...)` directly in:

- `strategicc/outputs.py` — 5 occurrences (simulation plots)
- `strategicc/accounting/outputs.py` — 3 occurrences (SEEA-EA plots)

300 DPI is recommended for publication-quality figures.

## Animation

For an animated view of the landscape changing over time (combined LULC map + valuation statistics), see the [`animate` reference](../reference/animate.md). The animation reads from the same `lulc_mean_{year}.tif` files that `plot_spatial_summary()` uses — so `outputs.aggregate_spatial()` must be run first.
