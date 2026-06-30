"""
strategicc/calibration  —  v2.4
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
    pip install strategicc[calibration]
"""

from .loader import load_lulc_timeseries, LULCTimeSeries
from .age import compute_age_raster
from .transitions import compute_transition_rates
from .temporal import compute_temporal_distribution

__all__ = [
    "load_lulc_timeseries",
    "LULCTimeSeries",
    "compute_age_raster",
    "compute_transition_rates",
    "compute_temporal_distribution",
]
