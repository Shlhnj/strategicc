# `strategicc.stockflow`

Per-cell, per-timestep tracking of material quantities (typically carbon) moving between pools (Stock Types) via Flow Pathways — either automatically every timestep (age-driven) or triggered by transitions. See [Guide 3](../guides/03_stockflow_full.md) for a complete worked example.

```python
from strategicc.stockflow import (
    run_flows_for_timestep, init_stocks, aggregate_stock_by_class,
    aggregate_flow_by_class, build_asset_account,
)
```

This module is wired into `StrategiccEngine` automatically when `use_stockflow=True` — you typically don't call `run_flows_for_timestep()` directly, but the aggregation and asset-account functions below are meant to be called after `engine.run()` completes.

## The flow mechanic

A flow pathway connects two stock types. Whether it fires automatically or only on a transition, and where its quantity comes from, depends on which CSV columns are set:

- No `TransitionGroupId` -> automatic, invoked every timestep
- `TransitionGroupId` set -> only invoked on cells where that transition group fired this timestep
- `StateAttributeTypeId` set -> flow quantity comes from an age-bracketed lookup table (`StateAttributeValues.csv`), not the current stock total
- `StateAttributeTypeId` blank -> flow quantity is `Multiplier` x the current From-Stock total

`FlowOrder.csv` determines the sequence flows are computed in each timestep — this matters when one flow's output feeds another flow's input within the same step (e.g. NPP growing biomass before an emission flow releases a fraction of it).

Three `TargetType` modes control how the flow amount is calculated: `Flow` (the default — `source_quantity x Multiplier`), `ToStock` (multiplier is the target proportion of pre-flow To-Stock), and `FromStock` (multiplier is the target proportion of pre-flow From-Stock).

## Aggregating across iterations

```python
stock_df = aggregate_stock_by_class(
    iter_dirs=engine.iter_dirs, stock_types=engine._stock_types,
    classes=engine.classes, modal_maps=modal_maps,
    start_year=engine.start_year, n_timesteps=engine.n_timesteps,
)
flow_df = aggregate_flow_by_class(engine.iter_dirs)
```

`aggregate_stock_by_class()` takes the median stock raster across iterations for each timestep, then masks it by the modal LULC class for that timestep — consistent with how the rest of STRATEGICC's spatial aggregation works. `aggregate_flow_by_class()` reads the per-class flow breakdown logged during simulation and takes the median total per class per flow type per year.

## Building a SEEA-EA asset account

```python
asset_account = build_asset_account(
    stock_df=stock_df, flow_df=flow_df, stock_types=engine._stock_types,
    classes=engine.classes, start_year=engine.start_year, n_timesteps=engine.n_timesteps,
)
```

Produces the full standard SEEA-EA structure — `opening_balance`, `additions`, `reductions`, `closing_balance_reconciled` (`opening + additions - reductions`) — per stock type per class per year. Additions/reductions are inferred automatically from flow direction (any flow where the stock is the `to_stock` counts as an addition, `from_stock` as a reduction).

The output also includes `closing_balance_actual` (the real stock-raster median for that year) and `reconciliation_diff` (the difference). These won't always match exactly: medians of sums and sums of medians aren't algebraically identical in a stochastic Monte Carlo setting, so a small `reconciliation_diff` is expected statistical noise. A large one is worth investigating — it usually means a flow pathway isn't being captured correctly.

## CSV loaders

Same pattern as `strategicc.io` — `load_stock_types`, `load_flow_types`, `load_flow_order`, `load_flow_pathways`, `load_flow_multipliers`, `load_state_attribute_values`, `load_initial_stock_links` each parse their respective CSV into a list of dataclasses. `lookup_state_attribute(rules, attribute_type, age, state_class=None)` performs the age-bracket lookup directly, useful for inspecting what value a given age would resolve to before running a full simulation.
