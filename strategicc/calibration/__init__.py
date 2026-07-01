"""
strategicc/calibration  —  v3.4
---------------------------------
Tools to derive STRATEGICC inputs from a historical LULC time series
(supplied as a zip of yearly GeoTIFFs).

Modules
-------
loader        — extract + load a zip of yearly LULC rasters into a stack
age           — backtrack continuous (or binned) age-since-transition per class
transitions   — derive mean annual transition probabilities (Transitions.csv)
temporal      — derive year-by-year multiplier distribution (TransitionMultipliers.csv)

These tools require the optional `rasterio` dependency:
    pip install rasterio
"""

from .loader import (
    load_lulc_timeseries, LULCTimeSeries,
    extract_initial_state_class, extract_lulc_zip_to_folder,
)
from .age import compute_age_raster, save_age_raster
from .transitions import (
    compute_yearly_transition_counts,
    compute_transition_rates,
    save_transitions_csv,
)
from .temporal import compute_temporal_distribution, save_temporal_distribution_csv

__all__ = [
    "load_lulc_timeseries",
    "LULCTimeSeries",
    "extract_initial_state_class",
    "extract_lulc_zip_to_folder",
    "compute_age_raster",
    "save_age_raster",
    "compute_yearly_transition_counts",
    "compute_transition_rates",
    "save_transitions_csv",
    "compute_temporal_distribution",
    "save_temporal_distribution_csv",
]
