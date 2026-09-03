# STRATEGICC Documentation

**STRATEGICC**: State and Transition Integrated Economic-Environmental Accounting

Python package implementing State-and-Transition Simulation Models (STSM, [Daniel et al. 2016](https://doi.org/10.1111/2041-210X.12597)) integrated with the UN's System of Environmental-Economic Accounting - Ecosystem Accounting ([SEEA-EA](https://seea.un.org/ecosystem-accounting)).

The package can simulates how a landscape's land cover changes over time under stochastic, spatially explicit transition probabilities, then translates that simulated future into ecosystem service value, such as a carbon Stock & Flow accounting.

## Where to start

To understand the package, user must have familiarity with STSM workflow concept, then work through the guides in order which each examples builds on the previous one's concepts:

1. **[Getting Started](guides/01_simple_seea.md)**  --> A single LULC raster, no simulation, just SEEA-EA valuation of a snapshot
2. **[Calibration + Simulation](guides/02_calibration_stsm.md)**  --> Derive transition rates from historical data, run a spatial Monte Carlo simulation, value the simulated future
3. **[Full Pipeline with Stock & Flow](guides/03_stockflow_full.md)** --> Age-indexed carbon flows, transition-triggered emissions, and dynamic (not static) ecosystem valuation
4. **[Visualization](guides/04_visualization.md)** --> All plots the package produces, output file locations, inline display in Colab, and export resolution

For exact function signatures and parameters, see the **API Reference** below.

## Installation

See [installation.md](installation.md).

## Configuration

STRATEGICC can be configured either by editing `strategicc/config.py` directly, by setting attributes on the `config` module at runtime (`cfg.N_TIMESTEPS = 20`), or via a single master `RunManifest.txt` file. See [manifest_reference.md](manifest_reference.md) for the full field list.

## API Reference

The package is organized into subpackages, each documented separately:

| Subpackage | Purpose |
|---|---|
| [`engine`](reference/engine.md) | `StrategiccEngine`, the core simulation class |
| [`config`](reference/config.md) | Runtime configuration, `RunManifest.txt` loader |
| [`core`](reference/core.md) | Transition firing mechanics: adjacency, age, patch growth, targets |
| [`io`](reference/io.md) | Raster and CSV reading/writing |
| [`calibration`](reference/calibration.md) | Derive inputs from a historical LULC time series |
| [`validation`](reference/validation.md) | Hindcast validation, spatial agreement metrics, calibration correction |
| [`stockflow`](reference/stockflow.md) | Carbon (or other material) Stock & Flow accounting |
| [`accounting`](reference/accounting.md) | SEEA-EA ecosystem accounts |
| [`outputs`](reference/outputs.md) | Aggregation across iterations, plots |
| [`animate`](reference/animate.md) | Two-panel LULC + valuation GIF/MP4 |

## QGIS plugin

A QGIS dock-panel plugin wraps the full pipeline (setup, run, calibration,
hindcast/correction, SEEA-EA) as file-picker-driven forms and background
tasks, so scenarios can be built and run without touching Python or a
manifest file directly. Not part of this repository, ask the plugin's
maintainer for the current build.

## Worked examples

**Note: `strategicc_examples/` doesn't exist in this repository**, I checked the full repo tree (`tests/`, `inputs/`, `strategicc/`, and everything else at root) and there's no such directory, nor any trace of `example1_simple_seea.py` / `example2_calibration_stsm_seea.py` / `example3_full_stockflow_seea.py` anywhere, including in `CHANGELOG.md`. Either these scripts were never added or were removed without updating this page. Until they exist, the closest equivalents are the runnable code blocks in [Guide 1](guides/01_simple_seea.md), [Guide 2](guides/02_calibration_stsm.md), and [Guide 3](guides/03_stockflow_full.md), which cover the same three complexity levels described below:

| Example | Demonstrates |
|---|---|
| *(would be)* `example1_simple_seea.py` | Single raster → SEEA-EA snapshot valuation, see Guide 1 |
| *(would be)* `example2_calibration_stsm_seea.py` | Calibration → spatial Monte Carlo simulation → SEEA-EA on a projected future, see Guide 2 |
| *(would be)* `example3_full_stockflow_seea.py` | Calibration with age → Stock & Flow carbon cycle → Mode C dynamic valuation → asset account, see Guide 3 |

## Testing

The package ships with 436 tests across 31 files in `tests/` (plus `strategicc/calibration/test_calibration.py` inside the package itself), covering every module, including literature-grounded validation (Alongi 2020 mangrove carbon mass-balance) and regression tests for several real bugs caught during development. Run with:

```bash
pip install -e ".[dev]"
pytest tests/
```
