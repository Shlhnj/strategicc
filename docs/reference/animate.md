# `strategicc.animate`

Renders a two-panel animation: the modal LULC map on the left, a synced statistics panel on the right, one frame per timestep. Called manually after a simulation has completed and been aggregated, not part of the standard `run.py` pipeline.

```python
from strategicc import animate

path = animate(out_dir="strategicc_output/", panel="value_per_class")
```

## Requirements before calling

`animate()` reads already-generated outputs, not raw simulation data, `outputs.aggregate_spatial()` must have already been run (it needs `lulc_mean_{year}.tif` files under `{out_dir}/summary/spatial/`), and `strategicc.config.STATE_CLASSES_CSV` must still point at a valid `StateClasses.csv` (used for the legend and colours).

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `out_dir` | required | The simulation's output directory |
| `panel` | `"value_per_class"` | Right-panel content — see options below, or `None` for a map-only animation |
| `start_year` / `end_year` | `None` (full range) | Restrict the animated year range; can extend into `historical_ts` years |
| `frame_rate` | `2` | Frames per second |
| `output_format` | `"gif"` | `"gif"` or `"mp4"` (MP4 requires the `ffmpeg` binary) |
| `output_path` | `None` (defaults to `{out_dir}/animation.{format}`) | Where to save |
| `historical_ts` | `None` | An `LULCTimeSeries` (from `strategicc.calibration.load_lulc_timeseries()`) whose years are prepended before the simulated timeline |

## Right panel options

| `panel` value | Shows |
|---|---|
| `"value_per_class"` | One line per class, total monetary value (from `seea_total_value_by_class.csv`) |
| `"value_total"` | A single line, landscape-wide total monetary value |
| `"area_per_class"` | One line per class, area (from `area_modal.csv`) |
| `"transitions_out"` | Median transitions OUT of each class per year |
| `"transitions_in"` | Median transitions INTO each class per year |
| `None` | No right panel, map only |

The right panel draws the full history as faint lines immediately, with a moving vertical cursor showing the current frame's year — not an animated redraw of the line itself.

## Including historical years

If you've already run `strategicc.calibration.load_lulc_timeseries()` on a historical zip, pass the result directly — its years before the simulation's start year are prepended, producing one continuous past-to-future sequence:

```python
from strategicc.calibration import load_lulc_timeseries

ts = load_lulc_timeseries("annual_lulc_2015_2022.zip", extract_dir="extracted/")

path = animate(out_dir="strategicc_output/", panel=None, historical_ts=ts)
```

Historical-only frames (before the simulation's first year) are labelled "(historical)" in the map panel title and have no right-panel data, since there's no simulated value/transition data for years that predate the simulation.

## Validation

`animate()` raises `ValueError` for an unknown `panel` string, an unsupported `output_format`, or an empty year range after filtering, and `RuntimeError` if `output_format="mp4"` is requested but `ffmpeg` isn't installed.
