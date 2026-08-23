"""
strategicc/calibration  v3.11
---------------------------------
Tools to derive STRATEGICC inputs from a historical LULC time series
(supplied as a zip of yearly GeoTIFFs).

Modules
-------
loader -- extract + load a zip of yearly LULC rasters into a stack
age -- backtrack continuous (or binned) age-since-transition per class
transitions -- derive mean annual transition probabilities (Transitions.csv)
                and historical patch-size distributions
                (TransitionSizeDistribution.csv); also loads a persisted
                group_map from a Transitions.csv-schema CSV
                (load_group_map_csv, v3.11)
temporal -- derive per-pathway empirical multiplier distributions,
                emitting both TransitionMultipliers.csv (named
                DistributionType references) and Distributions.csv (the
                empirical value/frequency tables those names resolve to)
manifest -- write a brand-new RunManifest_calibrated.txt
                (save_calibration_manifest) or update an existing,
                hand-filled RunManifest.txt in place
                (fill_manifest_from_calibration, v3.11)

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
    compute_size_distribution,
    save_size_distribution_csv,
    load_group_map_csv,
)
from .temporal import (
    compute_temporal_distribution,
    save_temporal_distribution_csv,
    save_distributions_csv,
)
from .manifest import save_calibration_manifest, fill_manifest_from_calibration
from .report import calibration_summary

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
    "load_group_map_csv",
    "compute_temporal_distribution",
    "save_temporal_distribution_csv",
    "save_distributions_csv",
    "compute_size_distribution",
    "save_size_distribution_csv",
    "save_calibration_manifest",
    "fill_manifest_from_calibration",
    "calibration_summary",
]
