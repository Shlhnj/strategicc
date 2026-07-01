# Guide 2 — Calibration, Spatial Simulation, and SEEA-EA

**Complexity:** Intermediate
**Full script:** `strategicc_examples/example2_calibration_stsm_seea.py`

This guide builds on [Guide 1](01_simple_seea.md) by introducing the full simulation pipeline: instead of valuing one static map, you project how the landscape might change over the next N years and value that projected future.

You'll learn to:

1. **Calibrate** transition rates from a historical LULC time series, instead of guessing probabilities by hand
2. **Simulate** a spatially explicit Monte Carlo STSM, using a spatial multiplier raster to bias *where* transitions occur
3. **Aggregate** outcomes across iterations
4. Run **SEEA-EA** on the simulated future

## Step 1 — Calibrate from historical data

If you have several years of classified imagery (a zip of yearly GeoTIFFs, auto-detected by year in the filename), STRATEGICC can derive transition rates directly from the observed land cover changes:

```python
from strategicc.calibration import (
    load_lulc_timeseries, compute_transition_rates, compute_temporal_distribution,
    compute_yearly_transition_counts, save_transitions_csv,
    save_temporal_distribution_csv,
)
from strategicc.io import load_state_classes

ts = load_lulc_timeseries("annual_lulc_2015_2022.zip", extract_dir="extracted_hist")
classes = load_state_classes("inputs/StateClasses.csv")

yearly = compute_yearly_transition_counts(ts)
```

`compute_yearly_transition_counts()` is computed once and feeds both outputs below — this guarantees `Transitions.csv` and `TransitionMultipliers.csv` stay mathematically consistent with each other (the mean of the sampled multipliers will equal exactly 1.0).

You must map observed `(from_class_id, to_class_id)` pixel transitions to a named transition group — pairs not listed are excluded as classification noise:

```python
group_map = {(1, 3): "Aquaculture_expansion"}   # Mangrove(1) -> Aquaculture(3)

transitions_df = compute_transition_rates(yearly, classes, group_map, min_probability=1e-5)
save_transitions_csv(transitions_df, "inputs/Transitions.csv")

temporal_df = compute_temporal_distribution(yearly, group_map, min_years=3)
save_temporal_distribution_csv(temporal_df, "inputs/TransitionMultipliers.csv")
```

## Step 2 — Build a spatial multiplier raster

Spatial multipliers bias *where* a transition is more or less likely — e.g. cells closer to existing aquaculture ponds are more likely to convert. STRATEGICC expects a 0-1 normalised raster (1.0 = highest suitability):

```python
from scipy.ndimage import distance_transform_edt

aqua_mask = (final_year_lulc == 3)
dist = distance_transform_edt(~aqua_mask)
dist_norm = 1.0 - (dist / dist.max())   # invert: close = high value
```

Save it and point `TransitionSpatialMultipliers.csv` at it:

```python
with open("inputs/TransitionSpatialMultipliers.csv", "w") as f:
    f.write('''Iteration,Timestep,TransitionGroupId,TransitionMultiplierTypeId,MultiplierFileName
,,Aquaculture_expansion [Type],,aquaculture_distance.tif
''')
```

## Step 3 — Run the simulation

```python
from strategicc import StrategiccEngine

engine = StrategiccEngine(
    lulc_path              = "2022.tif",
    state_classes_csv      = "inputs/StateClasses.csv",
    transitions_csv        = "inputs/Transitions.csv",
    spatial_mult_csv       = "inputs/TransitionSpatialMultipliers.csv",
    trans_mult_csv         = "inputs/TransitionMultipliers.csv",
    ecosystem_services_csv = "inputs/EcosystemServices.csv",
    mult_dir                = "spatmult_uploads/",
    out_dir                  = "output/",
    start_year   = 2022,
    n_timesteps  = 10,
    n_iterations = 20,
    use_adjacency        = True,
    use_spatial_mult      = True,    # turns the spatial driver on
    use_trans_multiplier   = True,
    use_seea                 = True,
)
engine.load()
engine.diagnostic()   # prints expected transitions before running, useful sanity check
engine.run()
```

`engine.diagnostic()` is worth always calling before `run()` — it prints the expected number of transitions per pathway given your inputs, so you can catch a misconfigured probability or missing multiplier before spending time on a full Monte Carlo run.

## Step 4 — Aggregate across iterations

Each of the 20 iterations is stochastic and slightly different. `outputs.aggregate_spatial()` collapses them into a single modal map per timestep (the most frequent class per cell across all iterations) plus an uncertainty raster (% agreement):

```python
from strategicc import outputs

summary_dir = engine.out_dir / "summary"
area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)
outputs.plot_area_envelope(area_df, engine.classes, summary_dir)

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

## Step 5 — SEEA-EA on the projected future

Same `SEEAAccount` as Guide 1, but now `area_modal_df` covers every simulated year, and `trans_df` actually has transition data — so the transition matrix and change-in-value accounts become meaningful:

```python
from strategicc.accounting import SEEAAccount, save_all_accounts, plot_monetary_flows

acct = SEEAAccount(
    area_modal_df = area_modal_df,
    trans_df      = trans_df,
    services      = engine.ecosystem_services,
    classes       = engine.classes,
    px_area       = engine.px_area,
    area_df       = area_df,   # raw per-iteration data, powers the uncertainty summary
)
save_all_accounts(acct, engine.out_dir / "seea")

monetary = acct.monetary_flow_account()
total_by_year = monetary.sum(axis=1)
print(f"2022: {total_by_year.loc[2022]:,.0f} IDR")
print(f"2032: {total_by_year.loc[2032]:,.0f} IDR")
```

## A note on modal aggregation and visible change

If your transition probabilities are low relative to the number of iterations and timesteps, the modal class per cell may not flip even though individual iterations show real transitions — this is statistically correct (no cell converts in a majority of iterations), not a bug. If your projected value change looks flat, check the raw `area_df` envelope plot first; if individual iterations show clear trends but the modal map doesn't, you likely need a stronger calibrated signal, more timesteps, or fewer iterations relative to your landscape size.

## What this doesn't cover

This guide uses static (Mode A/B) ecosystem service values — carbon is priced at a flat rate per hectare, not derived from an actual simulated carbon cycle. For that, continue to [Guide 3](03_stockflow_full.md).
