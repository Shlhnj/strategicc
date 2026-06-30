"""
tests/test_manifest.py  —  v3.0
Unit tests for config.load_manifest() and the CSV-mode / Direct-mode guard.
"""

import pytest
from pathlib import Path

import strategicc.config as cfg


MANIFEST_CONTENT = """\
```
================================================================
STRATEGICC - RunManifest.txt (test fixture)
================================================================
This block is documentation and must never be parsed as config,
even though it contains lines like "Variable = Value" below.
  Variable = Value
```

SECTION 1 - INITIAL CONDITIONS (SPATIAL)
StateClassFileName = 2022.tif #path
AgeFileName = 2022_age.tif #path

SECTION 2 - MULTI-ROW CSV INPUTS
STATE_CLASSES_CSV = inputs/State Class.csv #path
TRANSITIONS_CSV = inputs/Transitions.csv #path
ECOSYSTEM_SERVICES_CSV = inputs/EcosystemServices.csv #path
OUT_DIR = my_output/ #path

SECTION 3 - RUN CONTROL
START_YEAR = 2021 #int
N_TIMESTEPS = 15 #int
N_ITERATIONS = 25 #int
RNG_SEED = 7 #int
AREA_UNIT = km2 #str

SECTION 4 - FEATURE TOGGLES
USE_ADJACENCY = True #bool
USE_AGE = False #bool
USE_SEEA = Yes #bool

SECTION 5 - OUTPUT OPTIONS (NON-SPATIAL)
SummaryOutputSC = Yes #bool
SummaryOutputSCTimesteps = 5 #int

SECTION 6 - OUTPUT OPTIONS (SPATIAL)
RasterOutputAge = No #bool

```
APPENDIX - FILLED EXAMPLE FOR EACH REFERENCED CSV
----------------------------------------------------------------
State Class.csv
----------------------------------------------------------------
Name,StateLabelXId,StateLabelYId,Id,Color,Legend,Description,IsAutoName
Mangrove:All,Mangrove,All,3,"255,0,100,0",,,No
```
"""


@pytest.fixture(autouse=True)
def reset_config_state():
    """Ensure manifest mode is reset before and after every test."""
    cfg.reset_manifest_mode()
    yield
    cfg.reset_manifest_mode()


@pytest.fixture
def manifest_file(tmp_path):
    p = tmp_path / "RunManifest.txt"
    p.write_text(MANIFEST_CONTENT)
    return p


# ── Parsing and type casting ──────────────────────────────────────────────────

def test_load_manifest_applies_path_values(manifest_file):
    cfg.load_manifest(manifest_file)
    assert cfg.LULC_PATH == Path("2022.tif")
    assert cfg.AGE_RASTER_PATH == Path("2022_age.tif")
    assert cfg.OUT_DIR == Path("my_output/")

def test_load_manifest_applies_int_values(manifest_file):
    cfg.load_manifest(manifest_file)
    assert cfg.START_YEAR == 2021
    assert isinstance(cfg.START_YEAR, int)
    assert cfg.N_TIMESTEPS == 15
    assert cfg.N_ITERATIONS == 25
    assert cfg.RNG_SEED == 7

def test_load_manifest_applies_str_values(manifest_file):
    cfg.load_manifest(manifest_file)
    assert cfg.AREA_UNIT == "km2"

def test_load_manifest_applies_bool_true_false(manifest_file):
    cfg.load_manifest(manifest_file)
    assert cfg.USE_ADJACENCY is True
    assert cfg.USE_AGE is False

def test_load_manifest_applies_bool_yes_no(manifest_file):
    """Yes/No should be accepted as bool equivalents to True/False."""
    cfg.load_manifest(manifest_file)
    assert cfg.USE_SEEA is True
    assert cfg.SUMMARY_OUTPUT_SC is True
    assert cfg.RASTER_OUTPUT_AGE is False

def test_load_manifest_section5_and_6(manifest_file):
    cfg.load_manifest(manifest_file)
    assert cfg.SUMMARY_OUTPUT_SC_TIMESTEPS == 5

def test_load_manifest_ignores_appendix_section(manifest_file):
    """Lines after 'APPENDIX' (example CSV content) must not be parsed
    as manifest variables, even though they contain commas/values."""
    cfg.load_manifest(manifest_file)
    assert not hasattr(cfg, "Name")
    assert not hasattr(cfg, "StateLabelXId")

def test_load_manifest_ignores_fenced_documentation_lines(manifest_file, capsys):
    """
    Regression test: the literal documentation line "Variable = Value"
    inside a fenced (```...```) header block must be completely ignored —
    not applied, and not even reported as an unrecognised variable.
    Prior to fencing, this required a hardcoded special-case string
    match, which was fragile and is no longer needed.
    """
    cfg.load_manifest(manifest_file)
    captured = capsys.readouterr()
    assert "Variable" not in captured.out

def test_load_manifest_fenced_block_not_applied_as_config(tmp_path):
    """Anything inside ``` fences is skipped outright, even if it looks
    exactly like a valid Variable = Value row."""
    p = tmp_path / "fenced.txt"
    p.write_text(
        "```\n"
        "N_TIMESTEPS = 999 #int\n"
        "```\n"
        "N_TIMESTEPS = 5 #int\n"
    )
    cfg.load_manifest(p)
    assert cfg.N_TIMESTEPS == 5   # only the unfenced row was applied


# ── Mode exclusivity guard ────────────────────────────────────────────────────

def test_direct_assignment_blocked_after_manifest_load(manifest_file):
    cfg.load_manifest(manifest_file)
    with pytest.raises(cfg.ManifestModeError):
        cfg.N_ITERATIONS = 999

def test_direct_assignment_works_before_manifest_load():
    cfg.N_ITERATIONS = 11
    assert cfg.N_ITERATIONS == 11

def test_reset_manifest_mode_lifts_guard(manifest_file):
    cfg.load_manifest(manifest_file)
    with pytest.raises(cfg.ManifestModeError):
        cfg.N_ITERATIONS = 999
    cfg.reset_manifest_mode()
    cfg.N_ITERATIONS = 999
    assert cfg.N_ITERATIONS == 999

def test_guard_does_not_block_reads(manifest_file):
    cfg.load_manifest(manifest_file)
    _ = cfg.N_TIMESTEPS
    _ = cfg.AREA_UNIT
    _ = cfg.USE_ADJACENCY


# ── Error handling ─────────────────────────────────────────────────────────────

def test_load_manifest_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        cfg.load_manifest("/nonexistent/path/RunManifest.txt")

def test_load_manifest_unparseable_int_raises(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("N_TIMESTEPS = notanumber #int\n")
    with pytest.raises(ValueError):
        cfg.load_manifest(bad)

def test_load_manifest_blank_required_int_raises(tmp_path):
    bad = tmp_path / "bad2.txt"
    bad.write_text("N_TIMESTEPS = #int\n")
    with pytest.raises(ValueError):
        cfg.load_manifest(bad)

def test_load_manifest_unknown_variable_warns_not_raises(tmp_path, capsys):
    p = tmp_path / "unknown.txt"
    p.write_text(
        "N_TIMESTEPS = 5 #int\n"
        "SOME_UNKNOWN_VARIABLE = whatever #str\n"
    )
    cfg.load_manifest(p)
    assert cfg.N_TIMESTEPS == 5
    captured = capsys.readouterr()
    assert "unrecognised" in captured.out.lower() or "SOME_UNKNOWN_VARIABLE" in captured.out


# ── Blank path values ──────────────────────────────────────────────────────────

def test_load_manifest_blank_path_becomes_none(tmp_path):
    p = tmp_path / "blankpath.txt"
    p.write_text("AgeFileName = #path\n")
    cfg.load_manifest(p)
    assert cfg.AGE_RASTER_PATH is None
