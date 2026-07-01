# Guide 3 — Full Pipeline: Age, Stock & Flow, and Dynamic Valuation

**Complexity:** Advanced
**Full script:** `strategicc_examples/example3_full_stockflow_seea.py`

This is the full STRATEGICC pipeline, building on [Guide 2](02_calibration_stsm.md) by adding carbon Stock & Flow accounting. The carbon valuation responds to age structure, transition dynamics, and stochastic variation — not a flat per-hectare number.

You'll learn to:

1. **Calibrate** transition rates *and* a continuous age raster from historical data
2. **Simulate** with age tracking, so carbon flows can be age-indexed
3. Track **Stock & Flow**: NPP grows carbon every year (automatic, age-indexed), Emission releases it on conversion (transition-triggered)
4. Run **SEEA-EA Mode C** valuation — carbon priced from the actual simulated quantity, plus a full asset account

## Step 1 — Calibrate transitions *and* age

```python
from strategicc.calibration import (
    load_lulc_timeseries, compute_age_raster, save_age_raster,
    compute_yearly_transition_counts, compute_transition_rates, save_transitions_csv,
    compute_temporal_distribution, save_temporal_distribution_csv,
)

ts = load_lulc_timeseries("annual_lulc_2010_2022.zip", extract_dir="extracted_hist")

age_result = compute_age_raster(ts)   # continuous age, backtracked from the whole record
save_age_raster(age_result, "inputs/age.tif")
```

A longer historical record (here, 13 years) gives the age backtracking more to work with — `compute_age_raster()` walks backward from the baseline year counting how long each cell has continuously held its current class.

## Step 2 — Define the carbon cycle

Three new CSVs define the Stock & Flow mechanic. Stock types are the pools material moves between:

```python
with open("inputs/StockType.csv", "w") as f:
    f.write("Name,Description\nAtmosphere,Notional carbon source/sink\nBiomass,Living mangrove carbon\n")

with open("inputs/FlowType.csv", "w") as f:
    f.write("Name,Description\nNPP,Net primary production\nEmission,Release on conversion\n")

with open("inputs/FlowOrder.csv", "w") as f:
    f.write("Iteration,Timestep,FlowTypeId,Order\n,,NPP,1\n,,Emission,2\n")
```

`FlowOrder.csv` matters: NPP (order=1) is computed before Emission (order=2) each timestep, so emission acts on the post-growth biomass total, not the prior year's value.

Flow pathways define how stocks connect. A pathway with no `TransitionGroupId` is automatic (fires every timestep); one with a `TransitionGroupId` only fires when that transition occurs on a cell that timestep:

```python
fp_rows = [
    # NPP: automatic, age-indexed (StateAttributeTypeId="NPP"), Mangrove only
    {"FromStateClassId": "Mangrove:All", "FromStockTypeId": "Atmosphere",
     "ToStockTypeId": "Biomass", "StateAttributeTypeId": "NPP",
     "FlowTypeId": "NPP", "Multiplier": "1"},
    # Emission: triggered by Aquaculture_expansion, releases 90% of biomass
    {"FromStockTypeId": "Biomass", "ToStockTypeId": "Atmosphere",
     "TransitionGroupId": "Aquaculture_expansion [Type]",
     "FlowTypeId": "Emission", "Multiplier": "0.9"},
]
```

Because the NPP pathway specifies `StateAttributeTypeId="NPP"`, its flow quantity comes from an age-bracketed lookup table rather than a flat rate — younger mangrove sequesters less carbon per year than mature mangrove:

```python
with open("inputs/StateAttributeValues.csv", "w") as f:
    f.write('''Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,StateClassId,StateAttributeTypeId,AgeMin,AgeMax,TSTGroupId,TSTMin,TSTMax,Value,DistributionType,DistributionFrequencyId,DistributionSD,DistributionMin,DistributionMax
,,,,,,NPP,0,10,,,,5.1,,,,,
,,,,,,NPP,11,20,,,,11.0,,,,,
,,,,,,NPP,21,999,,,,18.4,,,,,
''')
```

These rates (5.1, 11.0, 18.4 Mg C/ha/yr) are drawn from real published mangrove carbon literature ([Alongi 2020](https://doi.org/10.3390/jmse8100767)) — the package's own Stock & Flow engine has been validated against the same source.

## Step 3 — Mode C ecosystem services

Instead of a static `PhysicalValuePerHa`, set `StockFlowSource` to pull the physical quantity directly from the simulated flow or stock:

```python
with open("inputs/EcosystemServices.csv", "w") as f:
    f.write('''StateClassId,ServiceName,ServiceType,ValuePerHa,Currency,PhysicalUnit,PhysicalValuePerHa,StockFlowSource
Mangrove,Carbon Sequestration,Regulating,75000,IDR,MgC,,flow:NPP
Mangrove,Carbon Storage,Regulating,75000,IDR,MgC,,stock:Biomass
Mangrove,Coastal Protection,Regulating,15000000,IDR,,,
Aquaculture,Aquaculture Fishery,Provisioning,45000000,IDR,kg/ha,800,
''')
```

`flow:NPP` and `stock:Biomass` are genuinely different things: flow values the annual carbon sequestration service (a recurring rate, good for something like a carbon credit payment); stock values the carbon currently stored (a standing asset, good for asset valuation). In Mode C, `ValuePerHa` is reinterpreted as price per physical unit, not per hectare.

## Step 4 — Run with age tracking and Stock & Flow enabled

```python
from strategicc import StrategiccEngine

engine = StrategiccEngine(
    lulc_path              = "2022.tif",
    state_classes_csv      = "inputs/StateClasses.csv",
    transitions_csv        = "inputs/Transitions.csv",
    trans_mult_csv         = "inputs/TransitionMultipliers.csv",
    ecosystem_services_csv = "inputs/EcosystemServices.csv",
    out_dir                = "output/",
    start_year             = 2022,
    n_timesteps            = 10,
    n_iterations           = 15,
    use_adjacency          = True,
    use_trans_multiplier   = True,
    use_seea               = True,
    use_age                = True,
    age_raster_path        = "inputs/age.tif",
    save_age_rasters       = True,
    use_stockflow          = True,
)
```

The Stock & Flow CSVs themselves are configured via `strategicc.config` rather than `StrategiccEngine`'s constructor — set them before calling `engine.load()`:

```python
import strategicc.config as cfg
cfg.STOCK_TYPE_CSV              = "inputs/StockType.csv"
cfg.FLOW_TYPE_CSV               = "inputs/FlowType.csv"
cfg.FLOW_ORDER_CSV              = "inputs/FlowOrder.csv"
cfg.FLOW_PATHWAYS_CSV           = "inputs/FlowPathways.csv"
cfg.STATE_ATTRIBUTE_VALUES_CSV  = "inputs/StateAttributeValues.csv"

engine.load()
engine.run()
```

## Step 5 — Aggregate Stock & Flow outputs

```python
from strategicc.stockflow import aggregate_stock_by_class, aggregate_flow_by_class, build_asset_account

stock_df = aggregate_stock_by_class(
    iter_dirs=engine.iter_dirs, stock_types=engine._stock_types,
    classes=engine.classes, modal_maps=modal_maps,
    start_year=engine.start_year, n_timesteps=engine.n_timesteps,
)
flow_df = aggregate_flow_by_class(engine.iter_dirs)
```

## Step 6 — SEEA-EA asset account

A full standard asset account (opening balance, additions, reductions, closing balance) for each stock type per class per year:

```python
asset_account = build_asset_account(
    stock_df=stock_df, flow_df=flow_df, stock_types=engine._stock_types,
    classes=engine.classes, start_year=engine.start_year,
    n_timesteps=engine.n_timesteps,
)
```

`asset_account` reports both a reconciled closing balance (`opening + additions - reductions`) and the actual closing balance from the stock rasters, plus their difference. A small difference is expected statistical noise from Monte Carlo aggregation (median-of-sums vs sum-of-medians); a large one may indicate a missing flow pathway.

## Step 7 — Mode C SEEA-EA accounting

```python
from strategicc.accounting import SEEAAccount, save_all_accounts

acct = SEEAAccount(
    area_modal_df = area_modal_df,
    trans_df      = trans_df,
    services      = engine.ecosystem_services,
    classes       = engine.classes,
    px_area       = engine.px_area,
    area_df       = area_df,
    stock_df      = stock_df,   # enables Mode C
    flow_df       = flow_df,    # enables Mode C
)
save_all_accounts(acct, engine.out_dir / "seea")
```

The "Carbon Storage" service in `monetary_flow_account()` will now show growing value as mangrove matures and accumulates biomass, and the "Carbon Sequestration" service shows the annual flow value — both genuinely responsive to the simulated age structure and transition dynamics, not a flat assumption.

## Where to go from here

This is the full pipeline. For animating the result, see the [`animate` reference](../reference/animate.md). For the complete field-by-field `RunManifest.txt` reference (an alternative to setting `strategicc.config` attributes directly), see [manifest_reference.md](../manifest_reference.md).
