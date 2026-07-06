"""
strategicc/validation  v3.12
-------------------------------
Compare simulated LULC/ecosystem output against the historical record,
diagnose class-level divergence, and optionally correct Transition
Multipliers to close the gap.

Modules
-------
extent      -- compute_observed_extent, compare_extent_trajectories,
               spatial_agreement (Figure of Merit primary, Kappa
               secondary), attribute_extent_drift
hindcast    -- hindcast_run() -- orchestrates a full validation pass:
               runs the calibrated model over a historical window and
               bundles the comparison + spatial agreement + drift results
correction  -- compute_pathway_rate_ratios, correct_multipliers
               (method="scaling", default; method="optimize" not yet
               implemented)

Workflow position
------------------
Standalone step between calibration and the real scenario runs:
    1. calibration -> 2. diagnostics (eyeball) -> 3. validation (this
    module) -> 4. run -> 5. see result
Not auto-wired into strategicc.run.main().
"""

from .extent import (
    compute_observed_extent,
    compare_extent_trajectories,
    spatial_agreement,
    attribute_extent_drift,
)
from .hindcast import hindcast_run, HindcastResult
from .correction import compute_pathway_rate_ratios, correct_multipliers

__all__ = [
    "compute_observed_extent",
    "compare_extent_trajectories",
    "spatial_agreement",
    "attribute_extent_drift",
    "hindcast_run",
    "HindcastResult",
    "compute_pathway_rate_ratios",
    "correct_multipliers",
]
