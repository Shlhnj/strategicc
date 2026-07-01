# `strategicc.io`

Raster and CSV reading/writing — the foundation everything else builds on. Most of these are used internally by `StrategiccEngine`, but the loaders are useful standalone (e.g. for [Guide 1](../guides/01_simple_seea.md)'s snapshot valuation, which calls them directly without a full simulation).

```python
from strategicc.io import (
    read_lulc, save_tifs, get_pixel_area, resolve_mult_dir,
    load_state_classes, load_transitions,
)
```

## Raster I/O (`io/raster.py`)

| Function | Purpose |
|---|---|
| `read_lulc(path)` | Read a GeoTIFF as a uint8 class-ID array, returning `(array, pixel_area_ha, tags)` |
| `read_tiff(path)` | Generic single-band GeoTIFF reader (any dtype), same return shape |
| `save_tifs(maps, start_year, src_tags, out_dir)` | Save a list of LULC arrays as `lulc_{year}.tif`, preserving georeferencing |
| `get_pixel_area(px_area_ha, unit)` | Convert hectares to the configured `AREA_UNIT` (`"ha"`, `"km2"`, or `"px"`) |
| `resolve_mult_dir(mult_dir)` | If `mult_dir` is a `.zip`, auto-extract once to a sibling folder and return that folder's path; if it's already a folder, return unchanged |

`read_lulc()` requires real GeoTIFF georeferencing tags (ModelPixelScaleTag, ModelTiepointTag) — a plain `PIL.Image.save()` without those tags will fail to read back. Use `rasterio` to write synthetic test rasters if you need them (see any of the worked examples for the pattern).

## CSV loaders (`io/csv_loader.py`)

One loader per ST-Sim-format input file:

| Loader | Reads |
|---|---|
| `load_state_classes` | `StateClasses.csv` -> `dict[int, StateClass]` |
| `load_transitions` | `Transitions.csv` -> `list[TransitionRule]` |
| `load_spatial_mult_index` | `TransitionSpatialMultipliers.csv` -> `list[SpatialMultEntry]` |
| `load_transition_multipliers` | `TransitionMultipliers.csv` -> `list[TransitionMultiplierRule]` |
| `load_initial_age_rules` | `InitialAge.csv` -> `list[InitialAgeRule]` |
| `load_transition_size_rules` + `group_size_bins` | `TransitionSizeDistribution.csv` -> per-group cumulative bins |
| `load_transition_targets` | `TransitionTargets.csv` -> `list[TransitionTargetRule]` |
| `load_transition_adjacency_setting` / `_multipliers` + `build_adjacency_strength_map` | `TransitionAdjacencySetting/Multipliers.csv` -> `{group: strength}` |

Each loader returns a list of small dataclasses (e.g. `TransitionRule`, `StateClass`) mirroring the CSV's columns, with malformed or incomplete rows skipped and a printed warning rather than raising — a single bad row in a large hand-edited CSV won't crash an otherwise-valid run.

## Example: loading classes and a raster directly

```python
from strategicc.io import load_state_classes, read_lulc

classes = load_state_classes("inputs/StateClasses.csv")
lulc_arr, px_area_ha, src_tags = read_lulc("2024.tif")

for cid, sc in classes.items():
    print(cid, sc.name, sc.color)
```
