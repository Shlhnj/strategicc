"""
strategicc/config.py  —  v3.0
------------------------------
Runtime configuration: file paths, toggles, and tuning constants.

Two ways to configure a run:

  (1) Direct mode — edit this file, or assign cfg.XXX = ... in Python
      (e.g. in a notebook) before calling StrategiccEngine.from_config().

  (2) Manifest mode — call config.load_manifest("RunManifest.txt") to
      populate every setting from a single external file. In this mode,
      do NOT also assign cfg.XXX = ... afterward in the same run — the
      two mechanisms are mutually exclusive (load_manifest() enforces
      this with a runtime guard).

Both modes populate the exact same module-level attributes below.
"""

from __future__ import annotations
from pathlib import Path


# ── Input paths (Direct-mode defaults) ─────────────────────────────────────────
LULC_PATH                = Path("2022.tif")
MULT_DIR                 = Path("spatmult_uploads/")
OUT_DIR                  = Path("strategicc_output/")

STATE_CLASSES_CSV        = Path("inputs/StateClasses.csv")
TRANSITIONS_CSV          = Path("inputs/Transitions.csv")
SPATIAL_MULT_CSV         = Path("inputs/TransitionSpatialMultipliers.csv")
TRANSITION_MULT_CSV      = Path("inputs/TransitionMultipliers.csv")
TRANSITION_TYPE_CSV      = Path("inputs/TransitionType.csv")

ECOSYSTEM_SERVICES_CSV   = Path("inputs/EcosystemServices.csv")

# ── Initial state class from historical LULC zip (v3.4) ──────────────────────
# When enabled, the initial LULC raster (normally LULC_PATH / StateClassFileName)
# is instead extracted from a historical LULC time-series zip — the same format
# the calibration module consumes (strategicc.calibration.load_lulc_timeseries).
# The extracted year is cached to disk under inputs/ so subsequent runs skip
# re-extraction.
FETCH_INITIAL_SC_FROM_ZIP = False
LULC_ZIP_PATH             = Path("inputs/annual_lulc_1985_2022.zip")
INITIAL_SC_YEAR           = 2022

# ── Age tracking ───────────────────────────────────────────────────────────────
USE_AGE              = True
AGE_RASTER_PATH      = Path("inputs/age.tif")
AGE_INITIAL_CSV      = Path("inputs/InitialAge.csv")
SAVE_AGE_RASTERS     = True

# ── Transition size distribution ────────────────────────────────────────────────
TRANSITION_SIZE_CSV  = Path("inputs/TransitionSizeDistribution.csv")

# ── Transition targets (v3.1) ─────────────────────────────────────────────────
# Area-based overrides that replace or scale a group's probability-derived
# budget per timestep. Groups absent from this file (or the file itself
# missing) keep their normal probability-only behaviour.
TRANSITION_TARGETS_CSV = Path("inputs/TransitionTargets.csv")

# ── Transition adjacency (CSV-driven, optional — falls back to the scalar
#    ADJACENCY_STRENGTH / STRICT_EXPANSION_GROUPS below if not provided) ────────
TRANSITION_ADJACENCY_SETTING_CSV = Path("inputs/TransitionAdjacencySetting.csv")
TRANSITION_ADJACENCY_MULT_CSV    = Path("inputs/TransitionAdjacencyMultipliers.csv")

TERMINOLOGY_CSV = Path("inputs/Terminology.csv")

# ── Simulation settings ───────────────────────────────────────────────────────
START_YEAR   = 2022
N_TIMESTEPS  = 10
N_ITERATIONS = 10
RNG_SEED     = 42

# ── Area unit ──────────────────────────────────────────────────────────────────
#   "ha"  — hectares          (default; px_area_ha × 1.0)
#   "km2" — square kilometres (px_area_ha × 0.01)
#   "px"  — raw pixel count   (1 pixel = 1 unit, ignores pixel size)
AREA_UNIT = "ha"

# ── Feature toggles ───────────────────────────────────────────────────────────
USE_ADJACENCY        = True
USE_SPATIAL_MULT     = True
USE_TRANS_MULTIPLIER = True
USE_SEEA             = True

# ── Adjacency tuning (scalar fallback, used when no CSV is provided) ──────────
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

# ── Output options (non-spatial) ──────────────────────────────────────────────
SUMMARY_OUTPUT_SC            = True
SUMMARY_OUTPUT_SC_TIMESTEPS  = 1
SUMMARY_OUTPUT_TR            = True
SUMMARY_OUTPUT_TR_TIMESTEPS  = 1

# ── Output options (spatial) ──────────────────────────────────────────────────
RASTER_OUTPUT_SC                          = True
RASTER_OUTPUT_SC_TIMESTEPS                = 1
RASTER_OUTPUT_AGE                         = True
RASTER_OUTPUT_AGE_TIMESTEPS               = 1
RASTER_OUTPUT_TRANSITION_EVENTS           = True
RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS  = 1

# ── Stock & Flow (v3.2) ────────────────────────────────────────────────────────
# Set USE_STOCKFLOW = True to enable per-cell carbon (or other material)
# stock and flow tracking. Requires all CSVs below to be present.
USE_STOCKFLOW = False

STOCK_TYPE_CSV                = Path("inputs/StockType.csv")
STOCK_GROUP_CSV                = Path("inputs/StockGroup.csv")
STOCK_GROUP_MEMBERSHIP_CSV      = Path("inputs/StockTypeGroupMembership.csv")
FLOW_TYPE_CSV                    = Path("inputs/FlowType.csv")
FLOW_ORDER_CSV                    = Path("inputs/FlowOrder.csv")
FLOW_PATHWAYS_CSV                  = Path("inputs/FlowPathways.csv")
FLOW_MULTIPLIER_CSV                 = Path("inputs/FlowMultiplier.csv")
STATE_ATTRIBUTE_TYPE_CSV             = Path("inputs/StateAttributeType.csv")
STATE_ATTRIBUTE_VALUES_CSV            = Path("inputs/StateAttributeValues.csv")
INITIAL_STOCK_NON_SPATIAL_CSV          = Path("inputs/InitialStockNonSpatial.csv")

# Whether to save per-timestep per-stock-type rasters to disk
SAVE_STOCK_RASTERS = True

# SEEA-EA valuation mode (v3.2):
#   "area"        — current v2.2 behaviour: ValuePerUnitArea x area_ha (static)
#   "stock_flow"  — pull carbon quantity from the Stock & Flow engine's
#                   flow output instead of static PhysicalValuePerUnitArea
SEEA_VALUATION_MODE = "area"


# ═════════════════════════════════════════════════════════════════════════════
# MANIFEST LOADER  (v3.0)
# ═════════════════════════════════════════════════════════════════════════════

# Fixed schema: manifest Variable name -> (config.py attribute name, type)
# This is the single source of truth for how RunManifest.txt rows are parsed
# and which config.py attribute each one populates. Manifest variable names
# that differ from their config.py counterpart (e.g. "StateClassFileName"
# vs "LULC_PATH") are mapped explicitly here.
_MANIFEST_SCHEMA: dict[str, tuple[str, str]] = {
    # Section 1 — Initial Conditions (spatial)
    "StateClassFileName":          ("LULC_PATH", "path"),
    "AgeFileName":                 ("AGE_RASTER_PATH", "path"),
    "StratumFileName":             ("PRIMARY_STRATUM_PATH", "path"),
    "SecondaryStratumFileName":    ("SECONDARY_STRATUM_PATH", "path"),
    "TertiaryStratumFileName":     ("TERTIARY_STRATUM_PATH", "path"),

    # Section 1b — Initial state class from historical LULC zip (v3.4)
    "FetchInitialStateClassFromZip": ("FETCH_INITIAL_SC_FROM_ZIP", "bool"),
    "LULCZipPath":                    ("LULC_ZIP_PATH", "path"),
    "InitialStateClassYear":           ("INITIAL_SC_YEAR", "int"),

    # Section 2 — Multi-row CSV inputs
    "STATE_CLASSES_CSV":                  ("STATE_CLASSES_CSV", "path"),
    "TRANSITIONS_CSV":                    ("TRANSITIONS_CSV", "path"),
    "SPATIAL_MULT_CSV":                   ("SPATIAL_MULT_CSV", "path"),
    "TRANSITION_MULT_CSV":                ("TRANSITION_MULT_CSV", "path"),
    "TRANSITION_TYPE_CSV":                ("TRANSITION_TYPE_CSV", "path"),
    "ECOSYSTEM_SERVICES_CSV":             ("ECOSYSTEM_SERVICES_CSV", "path"),
    "AGE_INITIAL_CSV":                    ("AGE_INITIAL_CSV", "path"),
    "TRANSITION_SIZE_CSV":                ("TRANSITION_SIZE_CSV", "path"),
    "TRANSITION_TARGETS_CSV":             ("TRANSITION_TARGETS_CSV", "path"),
    "TRANSITION_ADJACENCY_SETTING_CSV":   ("TRANSITION_ADJACENCY_SETTING_CSV", "path"),
    "TRANSITION_ADJACENCY_MULT_CSV":      ("TRANSITION_ADJACENCY_MULT_CSV", "path"),
    "TERMINOLOGY_CSV":                    ("TERMINOLOGY_CSV", "path"),
    "MULT_DIR":                           ("MULT_DIR", "path"),
    "OUT_DIR":                            ("OUT_DIR", "path"),

    # Section 3 — Run control
    "START_YEAR":    ("START_YEAR", "int"),
    "N_TIMESTEPS":   ("N_TIMESTEPS", "int"),
    "N_ITERATIONS":  ("N_ITERATIONS", "int"),
    "RNG_SEED":      ("RNG_SEED", "int"),
    "AREA_UNIT":     ("AREA_UNIT", "str"),

    # Section 4 — Feature toggles
    "USE_ADJACENCY":         ("USE_ADJACENCY", "bool"),
    "USE_SPATIAL_MULT":      ("USE_SPATIAL_MULT", "bool"),
    "USE_TRANS_MULTIPLIER":  ("USE_TRANS_MULTIPLIER", "bool"),
    "USE_SEEA":              ("USE_SEEA", "bool"),
    "USE_AGE":               ("USE_AGE", "bool"),
    "SAVE_AGE_RASTERS":      ("SAVE_AGE_RASTERS", "bool"),

    # Section 5 — Output options (non-spatial)
    "SummaryOutputSC":           ("SUMMARY_OUTPUT_SC", "bool"),
    "SummaryOutputSCTimesteps":  ("SUMMARY_OUTPUT_SC_TIMESTEPS", "int"),
    "SummaryOutputTR":           ("SUMMARY_OUTPUT_TR", "bool"),
    "SummaryOutputTRTimesteps":  ("SUMMARY_OUTPUT_TR_TIMESTEPS", "int"),

    # Section 6 — Output options (spatial)
    "RasterOutputSC":                       ("RASTER_OUTPUT_SC", "bool"),
    "RasterOutputSCTimesteps":               ("RASTER_OUTPUT_SC_TIMESTEPS", "int"),
    "RasterOutputAge":                        ("RASTER_OUTPUT_AGE", "bool"),
    "RasterOutputAgeTimesteps":                ("RASTER_OUTPUT_AGE_TIMESTEPS", "int"),
    "RasterOutputTransitionEvents":             ("RASTER_OUTPUT_TRANSITION_EVENTS", "bool"),
    "RasterOutputTransitionEventTimesteps":      ("RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS", "int"),

    # Section 7 — Stock & Flow (v3.2)
    "USE_STOCKFLOW":                  ("USE_STOCKFLOW", "bool"),
    "STOCK_TYPE_CSV":                 ("STOCK_TYPE_CSV", "path"),
    "STOCK_GROUP_CSV":                ("STOCK_GROUP_CSV", "path"),
    "STOCK_GROUP_MEMBERSHIP_CSV":     ("STOCK_GROUP_MEMBERSHIP_CSV", "path"),
    "FLOW_TYPE_CSV":                  ("FLOW_TYPE_CSV", "path"),
    "FLOW_ORDER_CSV":                 ("FLOW_ORDER_CSV", "path"),
    "FLOW_PATHWAYS_CSV":              ("FLOW_PATHWAYS_CSV", "path"),
    "FLOW_MULTIPLIER_CSV":            ("FLOW_MULTIPLIER_CSV", "path"),
    "STATE_ATTRIBUTE_TYPE_CSV":       ("STATE_ATTRIBUTE_TYPE_CSV", "path"),
    "STATE_ATTRIBUTE_VALUES_CSV":     ("STATE_ATTRIBUTE_VALUES_CSV", "path"),
    "INITIAL_STOCK_NON_SPATIAL_CSV":  ("INITIAL_STOCK_NON_SPATIAL_CSV", "path"),
    "SAVE_STOCK_RASTERS":             ("SAVE_STOCK_RASTERS", "bool"),
    "SEEA_VALUATION_MODE":            ("SEEA_VALUATION_MODE", "str"),
}

# Tracks whether load_manifest() has been called this session — used to
# block subsequent direct cfg.XXX = ... assignments via _ManifestGuard.
_manifest_loaded: bool = False


class ManifestModeError(RuntimeError):
    """Raised when direct cfg.XXX = ... assignment is attempted after
    load_manifest() has already been called in the same session."""
    pass


def _cast_value(raw: str, type_hint: str, variable: str):
    """Cast a raw manifest string value according to its fixed type hint."""
    raw = raw.strip()

    if type_hint == "path":
        return Path(raw) if raw else None
    if type_hint == "int":
        if not raw:
            raise ValueError(f"Manifest variable '{variable}' requires an "
                              f"int value but was left blank")
        return int(raw)
    if type_hint == "float":
        if not raw:
            raise ValueError(f"Manifest variable '{variable}' requires a "
                              f"float value but was left blank")
        return float(raw)
    if type_hint == "bool":
        normalised = raw.lower()
        if normalised in ("true", "yes", "1"):
            return True
        if normalised in ("false", "no", "0", ""):
            return False
        raise ValueError(f"Manifest variable '{variable}': cannot parse "
                          f"'{raw}' as bool")
    if type_hint == "str":
        return raw

    raise ValueError(f"Unknown type hint '{type_hint}' for variable '{variable}'")


def _parse_manifest_lines(path: str | Path) -> dict[str, str]:
    """
    Parse RunManifest.txt into a flat dict of {Variable: raw_value_string}.

    Recognises lines of the form:  Variable = Value   # comment

    Only lines OUTSIDE fenced code blocks (```...```) are treated as live
    configuration rows. Everything inside a fenced block is documentation
    or example content and is always skipped, regardless of what it
    contains — this is how the header format explanation and the
    Appendix's example CSV rows are kept out of the parsed result without
    relying on content-based heuristics.

    Ignores blank lines and lines starting with '#' or '='.
    """
    path = Path(path)
    raw_values: dict[str, str] = {}
    in_code_block = False

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()

            # Fenced blocks (```...```) mark documentation/example text —
            # never parsed as live config, regardless of content. This
            # covers the header explanation (which references things like
            # "Variable = Value" and "cfg.XXX = ...") and the Appendix
            # example CSV rows, without needing to guess from formatting.
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            if not stripped or stripped.startswith("#") or stripped.startswith("="):
                continue
            if "=" not in stripped:
                continue   # section headers, dashed lines, etc.

            # Split on first '=' only (paths or comments may contain '=')
            var_part, rest = stripped.split("=", 1)
            variable = var_part.strip()
            if not variable:
                continue

            # Strip trailing inline comment (starts with '#')
            if "#" in rest:
                rest = rest.split("#", 1)[0]
            value = rest.strip()

            raw_values[variable] = value

    return raw_values


def load_manifest(path: str | Path) -> None:
    """
    Load all configuration from a RunManifest.txt file, overwriting the
    corresponding module-level attributes in config.py.

    Only lines OUTSIDE fenced code blocks (```...```) are parsed as live
    configuration. Everything inside a fenced block — the header format
    explanation and the Appendix's example CSV content — is documentation
    and is always skipped, regardless of its content.

    Enforces CSV-mode exclusivity: once called, any subsequent direct
    cfg.XXX = ... assignment in the same session will raise
    ManifestModeError. Call reset_manifest_mode() to lift this guard
    (e.g. between independent runs in the same notebook session).

    Parameters
    ----------
    path : path to RunManifest.txt

    Raises
    ------
    FileNotFoundError : if the manifest file does not exist
    ValueError         : if a required value is malformed or missing
    """
    global _manifest_loaded

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    raw_values = _parse_manifest_lines(path)

    applied = []
    skipped_unknown = []

    for variable, raw_value in raw_values.items():
        if variable not in _MANIFEST_SCHEMA:
            skipped_unknown.append(variable)
            continue

        attr_name, type_hint = _MANIFEST_SCHEMA[variable]
        casted = _cast_value(raw_value, type_hint, variable)

        globals()[attr_name] = casted
        applied.append((variable, attr_name, casted))

    _manifest_loaded = True

    print(f"  Manifest loaded from '{path}': {len(applied)} variable(s) applied")
    if skipped_unknown:
        print(f"  [Warning] {len(skipped_unknown)} unrecognised variable(s) "
              f"in manifest, skipped: {skipped_unknown}")


def reset_manifest_mode() -> None:
    """
    Lift the manifest-mode guard, allowing direct cfg.XXX = ... assignments
    again. Use this between independent configuration passes in the same
    Python session (e.g. running one scenario via manifest, then a second
    scenario via direct Python assignment).
    """
    global _manifest_loaded
    _manifest_loaded = False


# ── Real enforcement of the CSV-mode / Direct-mode mutual exclusion ──────────
#
# Python module attributes have no built-in way to intercept external
# `cfg.X = value` assignments (e.g. from a notebook doing
# `import strategicc.config as cfg; cfg.N_ITERATIONS = 50`). To make the
# "manifest mode XOR direct mode" rule a real, enforced guarantee rather
# than just a documented convention, this module replaces its own class
# in sys.modules with a subclass of ModuleType that overrides __setattr__.
# This is a standard, documented pattern (see PEP 562) and is transparent
# to normal usage — reads (cfg.N_TIMESTEPS) and the in-module assignments
# above (at import time, before the guard is installed) are unaffected.
#
# Once load_manifest() has been called, any further `cfg.X = value` from
# OUTSIDE this module raises ManifestModeError immediately. Assignments
# performed internally by load_manifest() itself (via globals()[...] = ...)
# bypass __setattr__ entirely since they write straight to the module's
# __dict__, so the loader is unaffected by its own guard.

import sys as _sys
import types as _types


class _GuardedConfigModule(_types.ModuleType):
    def __setattr__(self, name: str, value) -> None:
        if globals().get("_manifest_loaded", False) and not name.startswith("_"):
            raise ManifestModeError(
                f"Cannot set 'cfg.{name} = ...' directly — a manifest was "
                f"already loaded via load_manifest() in this session. "
                f"Manifest mode and direct mode cannot be mixed in the same "
                f"run. Call config.reset_manifest_mode() first if you "
                f"intend to switch to direct configuration mode."
            )
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _GuardedConfigModule
