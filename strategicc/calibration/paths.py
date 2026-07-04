"""
strategicc/calibration/paths.py  —  v3.10
------------------------------------------
Predefined output paths for all calibration artefacts.

All outputs are written to calibration_result/ relative to the
current working directory — keeping calibration outputs isolated
from user-managed inputs and simulation outputs.

Importing this module does NOT create any directories; directories
are created on first write by the individual save_* functions.
"""

from pathlib import Path

#: Root output folder for all calibration artefacts
CALIBRATION_DIR = Path("calibration_result")

#: Predefined output paths (relative to cwd)
TRANSITIONS_CSV     = CALIBRATION_DIR / "Transitions.csv"
TRANS_MULT_CSV      = CALIBRATION_DIR / "TransitionMultipliers.csv"
TRANS_SIZE_CSV      = CALIBRATION_DIR / "TransitionSizeDistribution.csv"
TRANS_DIST_CSV      = CALIBRATION_DIR / "Distributions.csv"
AGE_RASTER          = CALIBRATION_DIR / "age.tif"
RUN_MANIFEST        = CALIBRATION_DIR / "RunManifest_calibrated.txt"
