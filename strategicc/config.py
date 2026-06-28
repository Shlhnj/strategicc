"""
strategicc/config.py
--------------------
Runtime configuration: file paths, toggles, and tuning constants.
Edit this file (or override at runtime) to point at your data.
"""

from pathlib import Path

# ── Input paths ───────────────────────────────────────────────────────────────
LULC_PATH                = Path("2022.tif")
MULT_DIR                 = Path("spatmult_uploads/")
OUT_DIR                  = Path("strategicc_output/")

# ST-Sim–format CSV inputs
STATE_CLASSES_CSV        = Path("inputs/StateClasses.csv")
TRANSITIONS_CSV          = Path("inputs/Transitions.csv")
SPATIAL_MULT_CSV         = Path("inputs/TransitionSpatialMultipliers.csv")
TRANSITION_MULT_CSV      = Path("inputs/TransitionMultipliers.csv")

# SEEA-EA input (v2.0)
ECOSYSTEM_SERVICES_CSV   = Path("inputs/EcosystemServices.csv")

# ── Simulation settings ───────────────────────────────────────────────────────
START_YEAR   = 2022
N_TIMESTEPS  = 10
N_ITERATIONS = 10
RNG_SEED     = 42

# ── Feature toggles ───────────────────────────────────────────────────────────
USE_ADJACENCY        = True
USE_SPATIAL_MULT     = True
USE_TRANS_MULTIPLIER = True
USE_SEEA             = True    # v2.0 — run ecosystem accounting after simulation

# ── Adjacency tuning ──────────────────────────────────────────────────────────
ADJACENCY_STRENGTH = 4.0
STRICT_EXPANSION_GROUPS: set[str] = {
    "Agriculture_expansion",
    "Aquaculture_expansion",
    "Mangrove_recruitment",
    "Sedimentation",
    "Urbanization",
}

# ── Spatial multiplier tuning ─────────────────────────────────────────────────
SHARPENING_POWER: dict[str, int] = {
    "Agriculture_expansion": 6,
    "Aquaculture_expansion": 6,
    "Inundation":            8,
    "Mangrove_recruitment":  4,
    "Sedimentation":         4,
    "Urbanization":          6,
}
DEFAULT_SHARPENING_POWER = 4
