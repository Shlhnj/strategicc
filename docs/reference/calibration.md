# `strategicc.calibration`

Derive STRATEGICC inputs from a historical LULC time series, instead of hand-tuning probabilities. Requires the optional `rasterio` dependency (`pip install rasterio`).

```python
from strategicc.calibration import (
    load_lulc_timeseries, compute_age_raster, compute_transition_rates,
    compute_temporal_distribution, compute_yearly_transition_counts,
)
```

See [Guide 2](../guides/02_calibration_stsm.md) and [Guide 3](../guides/03_stockflow_full.md) for these in a complete workflow.

## Loading a time series

```python
ts = load_lulc_timeseries("annual_lulc_2015_2022.zip", extract_dir="extracted/")
```

Accepts a zip of yearly GeoTIFFs. In AUTO mode (the default), filenames are scanned for a 4-digit year — `"2010.tif"`, `"lulc_2010.tif"`, `"2010_classified.tif"` all work. Missing years between the earliest and latest detected are forward-filled from the prior year with a printed warning. For multi-year composites (e.g. a 5-year Landsat mosaic that should repeat across several annual slots), pass `periods=[(start_year, end_year, filename_year), ...]` to skip auto-detection entirely.

Returns an `LULCTimeSeries` dataclass with `.stack` (the annual raster stack), `.years` (list of years), and `.profile` (georeferencing, for writing outputs later).

## Deriving an age raster

```python
age_result = compute_age_raster(ts)
save_age_raster(age_result, "inputs/age.tif")
```

Backtracks, for every class, how many consecutive years each cell has continuously held that class — walking backward from the most recent (baseline) year. Defaults to continuous age in years; pass `age_bins=[(min, max, label), ...]` to bin into discrete classes instead. Cells that held their class for the entire observable record are flagged separately (`.full_record_mask`), since their true age may exceed what's visible in the time series.

## Deriving transition rates

```python
yearly = compute_yearly_transition_counts(ts)
```

Computes year-over-year pixel transition counts once — this is the single source of truth that both `compute_transition_rates()` (below) and `compute_temporal_distribution()` derive from, which guarantees the two outputs stay mathematically consistent with each other.

```python
group_map = {(1, 3): "Aquaculture_expansion"}   # (from_class_id, to_class_id) -> group name

transitions_df = compute_transition_rates(yearly, classes, group_map, min_probability=1e-5)
save_transitions_csv(transitions_df, "inputs/Transitions.csv")
```

`group_map` is required — raw class-ID pairs carry no group label on their own, and any pair not listed is excluded from the output (treated as classification noise) rather than silently guessed. `min_probability` filters out pathways below a threshold, useful for dropping single-pixel noise from a large historical record.

## Deriving the temporal multiplier distribution

```python
temporal_df = compute_temporal_distribution(yearly, group_map, min_years=3)
save_temporal_distribution_csv(temporal_df, "inputs/TransitionMultipliers.csv")
```

For each group, computes `multiplier_year = yearly_probability / mean_probability` across all observed years, then reports `Uniform(min(multiplier), max(multiplier))` — by construction, `mean(multiplier) ~= 1.0`, so sampling this distribution during simulation reproduces the same average behaviour as the static `Transitions.csv` probability while preserving the historical year-to-year variability as uncertainty. Groups observed in fewer than `min_years` distinct years are skipped (insufficient data for a reliable range).

## Fetching an initial state class from a zip

If you already have a historical LULC zip, you can also use it as the source for your simulation's initial (t=0) raster, instead of maintaining a separate standalone file:

```python
from strategicc.calibration import extract_initial_state_class

initial_path = extract_initial_state_class(
    zip_path="annual_lulc_2015_2022.zip", year=2022, extract_dir="inputs/lulc_annual",
)
```

This extracts the entire zip to a persistent folder (not just the requested year) — so re-running with a different `year` later reuses the cache instead of re-extracting. The same behaviour is available via the manifest fields `FetchInitialStateClassFromZip`, `LULCZipPath`, `InitialStateClassYear` — see [manifest_reference.md](../manifest_reference.md).
