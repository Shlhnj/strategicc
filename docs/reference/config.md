# `strategicc.config`

Runtime configuration for everything in the package, file paths, run control, feature toggles, and tuning constants. There is no `Config` class; `config` is a plain module whose attributes are read by `StrategiccEngine.from_config()` and other parts of the package.

```python
import strategicc.config as cfg
```

## Two configuration modes

**Direct mode**: set attributes on the module:

```python
import strategicc.config as cfg
from pathlib import Path

cfg.LULC_PATH       = Path("2022.tif")
cfg.N_TIMESTEPS     = 20
cfg.N_ITERATIONS    = 50
cfg.USE_AGE         = True
```

**Manifest mode**: load every setting at once from a single `RunManifest.txt` file:

```python
cfg.load_manifest("RunManifest.txt")
```

These two modes are mutually exclusive within the same session. Once `load_manifest()` has been called, any subsequent direct `cfg.X = value` assignment raises `ManifestModeError`, this is enforced at the module level, not just documented as a convention. Call `cfg.reset_manifest_mode()` to lift the restriction if you want to switch modes (e.g. running one scenario via manifest, then a second via direct assignment, in the same notebook session):

```python
cfg.load_manifest("RunManifest.txt")
# ... run a scenario ...

cfg.reset_manifest_mode()
cfg.N_ITERATIONS = 10   # now allowed again
```

See [manifest_reference.md](../manifest_reference.md) for the complete field-by-field reference.

## Commonly set attributes

| Category | Attributes |
|---|---|
| Paths | `LULC_PATH`, `STATE_CLASSES_CSV`, `TRANSITIONS_CSV`, `SPATIAL_MULT_CSV`, `TRANSITION_MULT_CSV`, `ECOSYSTEM_SERVICES_CSV`, `MULT_DIR`, `OUT_DIR` |
| Run control | `START_YEAR`, `N_TIMESTEPS`, `N_ITERATIONS`, `RNG_SEED`, `AREA_UNIT` |
| Toggles | `USE_ADJACENCY`, `USE_SPATIAL_MULT`, `USE_TRANS_MULTIPLIER`, `USE_SEEA`, `USE_AGE`, `USE_STOCKFLOW` |
| Age | `AGE_RASTER_PATH`, `AGE_INITIAL_CSV`, `SAVE_AGE_RASTERS` |
| Stock & Flow | `STOCK_TYPE_CSV`, `FLOW_TYPE_CSV`, `FLOW_ORDER_CSV`, `FLOW_PATHWAYS_CSV`, `STATE_ATTRIBUTE_VALUES_CSV`, `SAVE_STOCK_RASTERS` |
| Adjacency tuning | `ADJACENCY_STRENGTH`, `STRICT_EXPANSION_GROUPS` |
| Output gating | `SUMMARY_OUTPUT_SC`, `SUMMARY_OUTPUT_TR`, `RASTER_OUTPUT_SC`, `RASTER_OUTPUT_AGE`, `RASTER_OUTPUT_TRANSITION_EVENTS` (and their `*_TIMESTEPS` stride controls) |
| LULC zip fetch | `FETCH_INITIAL_SC_FROM_ZIP`, `LULC_ZIP_PATH`, `INITIAL_SC_YEAR` |

Note: Stock & Flow CSV paths are always configured via `strategicc.config`, not as `StrategiccEngine` constructor arguments, see [Guide 3](../guides/03_stockflow_full.md) for why and how.

## Reading the current configuration

Since `config` is a plain module, you can always inspect current values directly:

```python
print(cfg.N_TIMESTEPS, cfg.USE_AGE, cfg.AREA_UNIT)
```

This works identically whether values were set via direct assignment or loaded from a manifest.
