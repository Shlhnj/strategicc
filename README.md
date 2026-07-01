# STRATEGICC

**State and Transition Integrated Economic-Environmental Accounting**

A Python package implementing spatially explicit State-and-Transition Simulation Models ([Daniel et al. 2016](https://doi.org/10.1111/2041-210X.12597)) integrated with the UN's System of Environmental-Economic Accounting — Ecosystem Accounting ([SEEA-EA](https://seea.un.org/ecosystem-accounting)).

## Install
Install using pip
```
pip install strategicc
```
or using github repo:

```bash
pip install git+https://github.com/Shlhnj/strategicc.git
```

## Quick start

```python
import strategicc.config as cfg
from strategicc import StrategiccEngine, outputs
from strategicc.accounting import SEEAAccount, save_all_accounts
```

There are 2 methods of runnign strategicc; either with a manifest file or using a direct python command.
See the docs for more info.

```
# Load everything from a single manifest file
cfg.load_manifest("RunManifest.txt")

engine = StrategiccEngine.from_config()
engine.load()
engine.run()

# Aggregate and account
summary_dir = engine.out_dir / "summary"
area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)
modal_maps = outputs.aggregate_spatial(engine.iter_dirs, engine.start_year,
    engine.n_timesteps, engine.src_tags, summary_dir)
area_modal_df = outputs.modal_to_area_table(modal_maps, engine.classes,
    engine.px_area, engine.area_unit)

acct = SEEAAccount(area_modal_df, trans_df, engine.ecosystem_services,
    engine.classes, engine.px_area)
save_all_accounts(acct, engine.out_dir / "seea")
```

## Documentation

Full documentation lives in [`docs/`](docs/index.md):
- **[Installation](docs/installation.md)**
- **[Guide 1: Simple SEEA-EA from a single raster](docs/guides/01_simple_seea.md)**
- **[Guide 2: Calibration + Simulation + SEEA-EA](docs/guides/02_calibration_stsm.md)**
- **[Guide 3: Full pipeline with Stock & Flow](docs/guides/03_stockflow_full.md)**
- **[API Reference](docs/index.md#api-reference)**
- **[RunManifest.txt field reference](docs/manifest_reference.md)**

---
Muhammad Shulhan Jihadi. 
2026. 07. 01.
