# `strategicc.accounting`

SEEA-EA ecosystem accounts: extent, transition matrix, physical/monetary flow, change-in-value, uncertainty summary. The central class is `SEEAAccount`; see all three guides for it in context.

```python
from strategicc.accounting import (
    SEEAAccount, load_ecosystem_services, EcosystemService,
    save_all_accounts, plot_monetary_flows, plot_value_by_service, plot_transition_heatmap,
)
```

## `EcosystemServices.csv` and the three valuation modes

`load_ecosystem_services()` parses `EcosystemServices.csv` into a list of `EcosystemService` rows. Required columns: `StateClassId`, `ServiceName`, `ServiceType`, `ValuePerUnitArea`, `Currency` (all `ValuePerUnitArea`/`PhysicalValuePerUnitArea` figures are always interpreted as *per hectare*, regardless of the run's `AREA_UNIT`). The legacy column names `ValuePerHa` / `PhysicalValuePerHa` are still accepted for backward compatibility, but `ValuePerUnitArea` / `PhysicalValuePerUnitArea` are the current names. Each row supports one of three modes, distinguished by which optional columns are filled:

| Mode | Columns set | Behaviour |
|---|---|---|
| A | `ValuePerUnitArea` only | `value = ValuePerUnitArea x area` |
| B | + `PhysicalUnit`, `PhysicalValuePerUnitArea` | Adds a static physical flow account alongside Mode A's monetary one |
| C | + `StockFlowSource` (`"flow:<Type>"` or `"stock:<Type>"`) | Physical quantity comes from the actual simulated Stock & Flow output; `ValuePerUnitArea` is reinterpreted as price per physical unit |

Modes can be mixed freely within the same file, different rows (even for the same class) can use different modes. Two further optional columns, `UserType` and `UserShare`, attribute a service's use to a named beneficiary group for the supply/use tables below; services without a `UserType` are folded into a single `"Unspecified"` user at `UserShare=1.0`, so existing `EcosystemServices.csv` files work unchanged. See [Guide 3](../guides/03_stockflow_full.md) for Mode C in detail, including the distinction between `flow:` (an annual service rate) and `stock:` (a standing asset value).

## `SEEAAccount`

```python
acct = SEEAAccount(
    area_modal_df = area_modal_df,   # from outputs.modal_to_area_table()
    trans_df      = trans_df,        # from outputs.build_summary_tables()
    services      = services,
    classes       = classes,
    px_area       = px_area,
    px_area_ha    = px_area_ha,      # optional; pixel area in hectares, required for correct
                                      # valuation whenever the run's AREA_UNIT isn't "ha"
    area_df       = area_df,         # optional, raw per-iteration data for uncertainty
    stock_df      = stock_df,        # optional, required for Mode C stock-kind services
    flow_df       = flow_df,         # optional, required for Mode C flow-kind services
    asset_valuation_params = asset_valuation_params,  # optional, required for monetary_asset_account_seea()
)
```

`area_modal_df` (derived from the modal map across iterations) is what every account is actually computed from, ensuring the spatial output and the tabular accounts stay consistent with each other. `area_df` (raw, per-iteration) is used only for `uncertainty_summary()`, it never feeds the other accounts.

If `px_area_ha` is omitted and the run's `AREA_UNIT` isn't `"ha"`, a factor of 1.0 is assumed and a warning is printed, since valuation would otherwise be silently wrong. `asset_valuation_params` comes from `strategicc.accounting.csv_loader.load_asset_valuation_params()`, keyed by `StateClassId` (with `"ALL"` as the fallback default), it's only needed if you call `monetary_asset_account_seea()`.

### Methods

Two families: the original summary methods (collapsed across class), and the newer `_seea` methods (added in strategicc 3.17 / accounting v3.4-3.5) that reproduce the actual SEEA EA table layouts.

| Method | Returns |
|---|---|
| `extent_account()` | Area per class per year (flat time series) |
| `transition_matrix()` | Median area converted from each class to each class, summed across all timesteps |
| `value_change_matrix()` | Monetary value change implied by `transition_matrix()` |
| `physical_flow_account()` | Total physical units supplied per service per year (Mode B/C only; `None` if no service has a physical unit) |
| `monetary_flow_account()` | Total monetary value per service per year, the most commonly used output |
| `total_value_by_class()` | Total value per class per year (sum across all that class's services) |
| `change_in_value()` | Year-on-year change in total value, per class and overall |
| `uncertainty_summary()` | Median/min/max value range across iterations, reported once (not per-account) |
| `extent_account_seea(managed_groups=None)` | Ecosystem extent account in **SEEA EA Table 4.1** layout: one block per accounting period with Opening extent, Additions, Reductions, Net change, Closing extent, per class plus a Total column |
| `physical_flow_account_seea()` | `{"supply": DataFrame, "use": DataFrame}` matching **SEEA EA Tables 7.1a/7.1b**, supply by (year, class), use by (year, user_type), built from the `UserType`/`UserShare` columns. `None` under the same precondition as `physical_flow_account()` |
| `monetary_flow_account_seea()` | `{"supply": DataFrame, "use": DataFrame}` matching **SEEA EA Tables 9.1a/9.1b**, same shape as `physical_flow_account_seea()`, in monetary terms. `supply.sum() == use.sum()` per (year, service) by construction |
| `monetary_asset_account_seea(catastrophic_groups=None)` | Monetary ecosystem asset account in **SEEA EA Table 10.1** layout: Opening value, Ecosystem enhancement, Ecosystem degradation, Ecosystem conversions, Other changes in volume, Revaluations, Net change, Closing value, per class plus Total. Requires `asset_valuation_params` to have been passed to the constructor. `Reappraisals` is always reported as `0.0`, STRATEGICC has no mechanism to generate a genuine methodology-change entry, and Enhancement/Degradation is a residual needed to make Net change reconcile exactly, a documented approximation rather than SEEA EA's condition-attributed split |

## Saving everything at once

```python
save_all_accounts(acct, out_dir)
```

Writes every applicable account above to CSV in `out_dir`: `seea_extent_account.csv`, `seea_extent_account_table4_1.csv`, `seea_transition_matrix_area.csv`, `seea_transition_matrix_value.csv`, `seea_monetary_flow_account.csv`, `seea_monetary_flow_account_supply.csv`, `seea_monetary_flow_account_use.csv`, `seea_physical_flow_account.csv` (+ `_supply`/`_use`), `seea_total_value_by_class.csv`, `seea_change_in_value.csv`, `seea_monetary_asset_account_table10_1.csv`, `seea_uncertainty_summary.csv`. The `_table4_1`/`_table10_1` and physical-flow/uncertainty files are each skipped (with a printed note) when their required input, `trans_df`, `asset_valuation_params`, or `area_df`, wasn't passed to `SEEAAccount`, or when no service has a physical unit.

## Plots

```python
plot_monetary_flows(acct, classes, out_dir)      # stacked area + year-on-year change
plot_value_by_service(acct, out_dir)              # line chart per service type
plot_transition_heatmap(acct, out_dir)            # area + value-change matrices
```
