# `strategicc.accounting`

SEEA-EA ecosystem accounts: extent, transition matrix, physical/monetary flow, change-in-value, uncertainty summary. The central class is `SEEAAccount`; see all three guides for it in context.

```python
from strategicc.accounting import (
    SEEAAccount, load_ecosystem_services, EcosystemService,
    save_all_accounts, plot_monetary_flows, plot_value_by_service, plot_transition_heatmap,
)
```

## `EcosystemServices.csv` and the three valuation modes

`load_ecosystem_services()` parses `EcosystemServices.csv` into a list of `EcosystemService` rows. Each row supports one of three modes, distinguished by which columns are filled:

| Mode | Columns set | Behaviour |
|---|---|---|
| A | `ValuePerHa` only | `value = ValuePerHa x area` |
| B | + `PhysicalUnit`, `PhysicalValuePerHa` | Adds a static physical flow account alongside Mode A's monetary one |
| C | + `StockFlowSource` (`"flow:<Type>"` or `"stock:<Type>"`) | Physical quantity comes from the actual simulated Stock & Flow output; `ValuePerHa` is reinterpreted as price per physical unit |

Modes can be mixed freely within the same file — different rows (even for the same class) can use different modes. See [Guide 3](../guides/03_stockflow_full.md) for Mode C in detail, including the distinction between `flow:` (an annual service rate) and `stock:` (a standing asset value).

## `SEEAAccount`

```python
acct = SEEAAccount(
    area_modal_df = area_modal_df,   # from outputs.modal_to_area_table()
    trans_df      = trans_df,        # from outputs.build_summary_tables()
    services      = services,
    classes       = classes,
    px_area       = px_area,
    area_df       = area_df,         # optional, raw per-iteration data for uncertainty
    stock_df      = stock_df,        # optional, required for Mode C
    flow_df       = flow_df,         # optional, required for Mode C
)
```

`area_modal_df` (derived from the modal map across iterations) is what every account is actually computed from, ensuring the spatial output and the tabular accounts stay consistent with each other. `area_df` (raw, per-iteration) is used only for `uncertainty_summary()` — it never feeds the other accounts.

### Methods

| Method | Returns |
|---|---|
| `extent_account()` | Area per class per year |
| `transition_matrix()` | Median area converted from each class to each class, summed across all timesteps |
| `value_change_matrix()` | Monetary value change implied by `transition_matrix()` |
| `physical_flow_account()` | Total physical units supplied per service per year (Mode B/C only; `None` if no service has a physical unit) |
| `monetary_flow_account()` | Total monetary value per service per year — the most commonly used output |
| `total_value_by_class()` | Total value per class per year (sum across all that class's services) |
| `change_in_value()` | Year-on-year change in total value, per class and overall |
| `uncertainty_summary()` | Median/min/max value range across iterations, reported once (not per-account) |

## Saving everything at once

```python
save_all_accounts(acct, out_dir)
```

Writes every account above to CSV in `out_dir`, named `seea_extent_account.csv`, `seea_monetary_flow_account.csv`, etc.

## Plots

```python
plot_monetary_flows(acct, classes, out_dir)      # stacked area + year-on-year change
plot_value_by_service(acct, out_dir)              # line chart per service type
plot_transition_heatmap(acct, out_dir)            # area + value-change matrices
```
