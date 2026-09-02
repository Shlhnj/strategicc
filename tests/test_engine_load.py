"""
tests/test_engine_load.py

Unit/integration tests for strategicc.engine.StrategiccEngine.from_config()
and .load() — targeting the many optional-feature branches (spatial mult,
transition multipliers, named distributions, SEEA/ecosystem services, asset
valuation params, age tracking, transition size/targets/adjacency, and
Stock & Flow) that were previously untested.

Reuses the real, schema-correct example CSVs shipped in inputs/ rather
than hand-building fixtures for every optional file, since they already
match the loaders' expected columns.
"""

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

import strategicc.config as cfg
from strategicc.engine import StrategiccEngine

INPUTS = Path(__file__).resolve().parent.parent / "inputs"


@pytest.fixture(autouse=True)
def reset_config_state():
    cfg.reset_manifest_mode()
    yield
    cfg.reset_manifest_mode()


def _write_lulc(tmp_path):
    rows, cols = 6, 6
    lulc = np.ones((rows, cols), dtype=np.uint8)
    lulc[:, 3:] = 2
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}
    path = tmp_path / "lulc.tif"
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(lulc, 1)
    return path


def _base_kwargs(tmp_path, missing):
    return dict(
        lulc_path=_write_lulc(tmp_path),
        state_classes_csv=INPUTS / "State Class.csv",
        transitions_csv=INPUTS / "Transitions.csv",
        spatial_mult_csv=missing,
        trans_mult_csv=missing,
        ecosystem_services_csv=missing,
        mult_dir=tmp_path / "m",
        out_dir=tmp_path / "out",
        start_year=2022, n_timesteps=2, n_iterations=1, rng_seed=1,
        use_adjacency=False, use_spatial_mult=False, use_trans_multiplier=False,
        use_seea=False, use_age=False,
    )


@pytest.fixture
def missing(tmp_path):
    return tmp_path / "does_not_exist.csv"


# ── from_config() ────────────────────────────────────────────────────────────

def test_from_config_builds_engine_from_manifest(tmp_path):
    manifest = tmp_path / "RunManifest.txt"
    manifest.write_text(f"""\
StateClassFileName = {_write_lulc(tmp_path)} #path
STATE_CLASSES_CSV = {INPUTS / "State Class.csv"} #path
TRANSITIONS_CSV = {INPUTS / "Transitions.csv"} #path
OUT_DIR = {tmp_path / "out"}/ #path
START_YEAR = 2022 #int
N_TIMESTEPS = 3 #int
N_ITERATIONS = 1 #int
RNG_SEED = 5 #int
USE_ADJACENCY = False #bool
USE_SPATIAL_MULT = False #bool
USE_TRANS_MULTIPLIER = False #bool
USE_SEEA = False #bool
USE_AGE = False #bool
""")
    cfg.load_manifest(manifest)
    engine = StrategiccEngine.from_config()

    assert engine.start_year == 2022
    assert engine.n_timesteps == 3
    assert engine.n_iterations == 1
    assert engine.rng_seed == 5
    assert engine.use_seea is False


# ── use_trans_multiplier branches ───────────────────────────────────────────

def test_load_trans_multiplier_uniform_only(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["trans_mult_csv"] = tmp_path / "TransMult.csv"
    kwargs["trans_mult_csv"].write_text(
        "TransitionGroupId,DistributionType,DistributionMin,DistributionMax\n"
        "Mangrove_recruitment [Type],Uniform,0.5,1.5\n"
    )
    kwargs["use_trans_multiplier"] = True
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert len(engine.trans_mult_rules) == 1
    assert engine.distributions == {}


def test_load_trans_multiplier_named_distribution_with_table(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["trans_mult_csv"] = INPUTS / "Transition Multipliers.csv"
    kwargs["distributions_csv"] = INPUTS / "Distributions.csv"
    kwargs["use_trans_multiplier"] = True
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert len(engine.trans_mult_rules) > 0
    assert len(engine.distributions) > 0


def test_load_trans_multiplier_named_distribution_missing_table(tmp_path, missing, capsys):
    """Named DistributionType rules but DISTRIBUTIONS_CSV not set/found ->
    warns but does not raise."""
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["trans_mult_csv"] = INPUTS / "Transition Multipliers.csv"
    kwargs["distributions_csv"] = None
    kwargs["use_trans_multiplier"] = True
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert "Warning" in capsys.readouterr().out


def test_load_trans_multiplier_missing_file_disables(tmp_path, missing, capsys):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["trans_mult_csv"] = missing
    kwargs["use_trans_multiplier"] = True
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert engine.use_trans_multiplier is False
    assert "not found" in capsys.readouterr().out


# ── use_seea / ecosystem services branches ──────────────────────────────────

def test_load_seea_with_ecosystem_services(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["ecosystem_services_csv"] = INPUTS / "EcosystemServices.csv"
    kwargs["use_seea"] = True
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert len(engine.ecosystem_services) > 0


def test_load_seea_missing_ecosystem_services_disables(tmp_path, missing, capsys):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["ecosystem_services_csv"] = missing
    kwargs["use_seea"] = True
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert engine.use_seea is False
    assert "not found" in capsys.readouterr().out


# ── asset valuation params branches ─────────────────────────────────────────

def test_load_asset_valuation_params_present(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["ecosystem_services_csv"] = INPUTS / "EcosystemServices.csv"
    kwargs["use_seea"] = True
    avp = tmp_path / "AssetValuationParams.csv"
    avp.write_text(
        "StateClassId,DiscountRate,AssetLifeYears\n"
        "ALL,0.02,20\n"
    )
    kwargs["asset_valuation_params_csv"] = avp
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert "ALL" in engine.asset_valuation_params


def test_load_asset_valuation_params_missing_warns(tmp_path, missing, capsys):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["ecosystem_services_csv"] = INPUTS / "EcosystemServices.csv"
    kwargs["use_seea"] = True
    kwargs["asset_valuation_params_csv"] = missing
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert engine.asset_valuation_params == {}
    assert "will be omitted" in capsys.readouterr().out


# ── use_age branches ─────────────────────────────────────────────────────────

def test_load_age_from_initial_csv(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["use_age"] = True
    kwargs["age_initial_csv"] = INPUTS / "InitialAge.csv"
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert engine._initial_age is not None
    assert engine._initial_age.shape == engine._initial_lulc.shape


def test_load_age_defaults_to_zero_with_warning(tmp_path, missing, capsys):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["use_age"] = True
    kwargs["age_initial_csv"] = None
    kwargs["age_raster_path"] = None
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert np.all(engine._initial_age == 0)
    assert "No age raster or InitialAge.csv" in capsys.readouterr().out


def test_load_age_from_raster(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["use_age"] = True
    age_path = tmp_path / "age.tif"
    rows, cols = 6, 6
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint16", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}
    with rasterio.open(str(age_path), "w", **profile) as dst:
        dst.write(np.full((rows, cols), 5, dtype=np.uint16), 1)
    kwargs["age_raster_path"] = age_path
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert np.all(engine._initial_age == 5)


# ── transition size / targets / adjacency branches ──────────────────────────

def test_load_transition_size_distribution(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["transition_size_csv"] = INPUTS / "TransitionSizeDistribution.csv"
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert len(engine._size_bins) > 0


def test_load_transition_targets(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["transition_targets_csv"] = INPUTS / "TransitionTargets.csv"
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert len(engine._targets_by_timestep) > 0


def test_load_transition_adjacency_csv_driven(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    setting = tmp_path / "AdjSetting.csv"
    setting.write_text(
        "TransitionGroupId,StateClassId,StateAttributeTypeId,NeighborhoodRadius,UpdateFrequency\n"
        "Mangrove_recruitment,Mangrove,,1,1\n"
    )
    mult = tmp_path / "AdjMult.csv"
    mult.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,AttributeValue,Amount,DistributionType,"
        "DistributionFrequencyId,DistributionSD,DistributionMin,DistributionMax\n"
        ",,,,,Mangrove_recruitment,,1.5,,,,,\n"
    )
    kwargs["transition_adjacency_setting_csv"] = setting
    kwargs["transition_adjacency_mult_csv"] = mult
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert "Mangrove_recruitment" in engine._adjacency_groups


# ── Stock & Flow branches ────────────────────────────────────────────────────

def test_load_stockflow_missing_required_files_disables(tmp_path, missing, capsys, monkeypatch):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["use_stockflow"] = True
    monkeypatch.setattr(cfg, "STOCK_TYPE_CSV", missing)
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert engine.use_stockflow is False
    assert "missing required" in capsys.readouterr().out


def test_load_stockflow_full_success(tmp_path, missing, monkeypatch):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["use_stockflow"] = True
    monkeypatch.setattr(cfg, "STOCK_TYPE_CSV", INPUTS / "StockType.csv")
    monkeypatch.setattr(cfg, "FLOW_TYPE_CSV", INPUTS / "FlowType.csv")
    monkeypatch.setattr(cfg, "FLOW_ORDER_CSV", INPUTS / "FlowOrder.csv")
    monkeypatch.setattr(cfg, "FLOW_PATHWAYS_CSV", INPUTS / "FlowPathways.csv")
    monkeypatch.setattr(cfg, "STATE_ATTRIBUTE_VALUES_CSV", INPUTS / "StateAttributeValues.csv")
    monkeypatch.setattr(cfg, "FLOW_MULTIPLIER_CSV", INPUTS / "Flow Multipliers.csv")
    monkeypatch.setattr(cfg, "INITIAL_STOCK_NON_SPATIAL_CSV", INPUTS / "InitialStockNonSpatial.csv")
    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert engine.use_stockflow is True
    assert len(engine._stock_types) > 0
    assert len(engine._flow_pathways) > 0
    assert len(engine._flow_mult_rules) > 0
    # The shipped example InitialStockNonSpatial.csv has a header but no
    # data rows; asserting the type confirms the loader branch still ran.
    assert isinstance(engine._initial_stock_links, dict)
