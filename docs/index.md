# STRATEGICC Documentation

**STRATEGICC**: State and Transition Integrated Economic-Environmental Accounting

Python package implementing State-and-Transition Simulation Models (STSM, [Daniel et al. 2016](https://doi.org/10.1111/2041-210X.12597)) integrated with the UN's System of Environmental-Economic Accounting - Ecosystem Accounting ([SEEA-EA](https://seea.un.org/ecosystem-accounting)).

The package can simulates how a landscape's land cover changes over time under stochastic, spatially explicit transition probabilities, then translates that simulated future into ecosystem service value, such as a carbon Stock & Flow accounting.

## Where to start

To understand the package, user must have familiarity with STSM workflow concept, then work through the guides in order which each examples builds on the previous one's concepts:

1. **[Getting Started](guides/01_simple_seea.md)** -> a single LULC raster, no simulation, just SEEA-EA valuation of a snapshot
2. **[Calibration + Simulation](guides/02_calibration_stsm.md)**  ->derive transition rates from historical data, run a spatial Monte Carlo simulation, value the simulated future
3. **[Full Pipeline with Stock & Flow](guides/03_stockflow_full.md)** -> age-indexed carbon flows, transition-triggered emissions, and dynamic (not static) ecosystem valuation
4. **[Visualization](guides/04_visualization.md)** — all plots the package produces, output file locations, inline display in Colab, and export resolution

For exact function signatures and parameters, see the **API Reference** below.

## Installation

See [installation.md](installation.md).

## Configuration

STRATEGICC can be configured either by editing `strategicc/config.py` directly, by setting attributes on the `config` module at runtime (`cfg.N_TIMESTEPS = 20`), or via a single master `RunManifest.txt` file. See [manifest_reference.md](manifest_reference.md) for the full field list.

## API Reference

The package is organized into subpackages, each documented separately:

| Subpackage | Purpose |
|---|---|
| [`engine`](reference/engine.md) | `StrategiccEngine` — the core simulation class |
| [`config`](reference/config.md) | Runtime configuration, `RunManifest.txt` loader |
| [`core`](reference/core.md) | Transition firing mechanics: adjacency, age, patch growth, targets |
| [`io`](reference/io.md) | Raster and CSV reading/writing |
| [`calibration`](reference/calibration.md) | Derive inputs from a historical LULC time series |
| [`stockflow`](reference/stockflow.md) | Carbon (or other material) Stock & Flow accounting |
| [`accounting`](reference/accounting.md) | SEEA-EA ecosystem accounts |
| [`outputs`](reference/outputs.md) | Aggregation across iterations, plots |
| [`animate`](reference/animate.md) | Two-panel LULC + valuation GIF/MP4 |

## Worked examples

Three complete, runnable scripts at increasing complexity live in `strategicc_examples/`:

| Example | Demonstrates |
|---|---|
| `example1_simple_seea.py` | Single raster → SEEA-EA snapshot valuation |
| `example2_calibration_stsm_seea.py` | Calibration → spatial Monte Carlo simulation → SEEA-EA on a projected future |
| `example3_full_stockflow_seea.py` | Calibration with age → Stock & Flow carbon cycle → Mode C dynamic valuation → asset account |

## Testing

The package ships with 162 tests covering every module, including literature-grounded validation (Alongi 2020 mangrove carbon mass-balance) and regression tests for several real bugs caught during development. Run with:

```bash
pip install -e ".[dev]"
pytest tests/
```
