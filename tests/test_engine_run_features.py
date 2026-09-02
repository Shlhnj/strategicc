"""
tests/test_engine_run_features.py

Focused tests exercising StrategiccEngine.run()'s remaining uncovered
feature branches: age gating + age-raster saving, adjacency (both the
global scalar and CSV-driven strength map), transition multipliers
sampled during a real run, patch-growing with and without an active
target, target-only (non-patch) scaling, and the transition-event raster
save stride skip.

Each test enables ONE feature combination at a time against a tiny
synthetic 2-class raster, keeping cause and effect easy to trace if a
test ever fails.
"""

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

import strategicc.config as cfg
from strategicc.engine import StrategiccEngine


@pytest.fixture(autouse=True)
def reset_config_state():
    cfg.reset_manifest_mode()
    yield
    cfg.reset_manifest_mode()
    # Restore the RASTER_OUTPUT_* defaults these tests may have flipped,
    # since config is a shared module and other test files assume them.
    cfg.RASTER_OUTPUT_AGE = True
    cfg.RASTER_OUTPUT_AGE_TIMESTEPS = 1
    cfg.RASTER_OUTPUT_TRANSITION_EVENTS = True
    cfg.RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS = 1


def _write_lulc(path, rows=12, cols=12):
    lulc = np.ones((rows, cols), dtype=np.uint8)
    lulc[:, cols // 2:] = 2
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(lulc, 1)


def _write_state_classes(path):
    path.write_text(
        "Name,StateLabelXId,StateLabelYId,Id,Color,Legend,Description,IsAutoName\n"
        "Mangrove:All,Mangrove,All,1,\"255,0,100,0\",,,No\n"
        "Aquaculture:All,Aquaculture,All,2,\"255,255,0,255\",,,No\n"
    )


def _write_transitions(path, *, age_min=None, age_max=None, prob=0.3):
    age_min_s = "" if age_min is None else str(age_min)
    age_max_s = "" if age_max is None else str(age_max)
    path.write_text(
        "Iteration,Timestep,StratumIdSource,StateClassIdSource,StratumIdDest,"
        "StateClassIdDest,SecondaryStratumId,TertiaryStratumId,TransitionTypeId,"
        "Probability,Proportion,AgeMin,AgeMax,AgeRelative,AgeReset,TSTMin,TSTMax,TSTRelative\n"
        f",,,Mangrove:All,,Aquaculture:All,,,Aquaculture_expansion,{prob},,"
        f"{age_min_s},{age_max_s},,,,,,\n"
    )


def _base_kwargs(tmp_path, missing):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    lulc_path = tmp_path / "lulc.tif"
    _write_lulc(lulc_path)
    sc_path = inputs_dir / "StateClasses.csv"
    _write_state_classes(sc_path)
    trans_path = inputs_dir / "Transitions.csv"
    _write_transitions(trans_path)
    return dict(
        lulc_path=lulc_path,
        state_classes_csv=sc_path,
        transitions_csv=trans_path,
        spatial_mult_csv=missing,
        trans_mult_csv=missing,
        ecosystem_services_csv=missing,
        mult_dir=tmp_path / "m",
        out_dir=tmp_path / "out",
        start_year=2022, n_timesteps=3, n_iterations=1, rng_seed=1,
        use_adjacency=False, use_spatial_mult=False, use_trans_multiplier=False,
        use_seea=False, use_age=False,
    )


@pytest.fixture
def missing(tmp_path):
    return tmp_path / "does_not_exist.csv"


# ── Age gating + age raster saving ──────────────────────────────────────────

def test_run_with_age_gate_and_age_raster_output(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    inputs_dir = tmp_path / "inputs"
    trans_path = inputs_dir / "Transitions.csv"
    # AgeMin=3 gates the transition: only cells with age >= 3 are eligible.
    _write_transitions(trans_path, age_min=3, prob=0.5)   # overwrite AFTER _base_kwargs

    age_csv = inputs_dir / "InitialAge.csv"
    age_csv.write_text("StateClassId,AgeMean,AgeSD,AgeMin,AgeMax\nMangrove,10,0,0,20\n")

    kwargs["use_age"] = True
    kwargs["age_initial_csv"] = age_csv
    kwargs["save_age_rasters"] = True

    cfg.RASTER_OUTPUT_AGE = True
    cfg.RASTER_OUTPUT_AGE_TIMESTEPS = 1

    engine = StrategiccEngine(**kwargs)
    engine.load()
    engine.run()

    iter_dir = engine.iter_dirs[0]
    assert (iter_dir / "age").exists()
    assert any((iter_dir / "age").glob("age_*.tif"))


def test_run_with_age_gate_blocks_young_cells(tmp_path, missing):
    """AgeMin higher than any cell's starting age -> zero transitions fire."""
    kwargs = _base_kwargs(tmp_path, missing)
    inputs_dir = tmp_path / "inputs"
    trans_path = inputs_dir / "Transitions.csv"
    _write_transitions(trans_path, age_min=999, prob=0.9)   # overwrite AFTER _base_kwargs

    age_csv = inputs_dir / "InitialAge.csv"
    age_csv.write_text("StateClassId,AgeMean,AgeSD,AgeMin,AgeMax\nMangrove,1,0,0,5\n")

    kwargs["use_age"] = True
    kwargs["age_initial_csv"] = age_csv

    engine = StrategiccEngine(**kwargs)
    engine.load()
    engine.run()

    log_path = engine.iter_dirs[0] / "transition_log.csv"
    # A run with zero fired transitions still writes the file (as an
    # empty DataFrame, which serializes to a single blank line), so
    # check content rather than assuming a populated, parseable CSV.
    content = log_path.read_text() if log_path.exists() else ""
    assert content.strip() == ""


# ── Adjacency: global scalar + CSV-driven strength map ──────────────────────

def test_run_with_adjacency_and_csv_driven_strength(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["use_adjacency"] = True

    setting = tmp_path / "AdjSetting.csv"
    setting.write_text(
        "TransitionGroupId,StateClassId,StateAttributeTypeId,NeighborhoodRadius,UpdateFrequency\n"
        "Aquaculture_expansion,Mangrove,,1,1\n"
    )
    mult = tmp_path / "AdjMult.csv"
    mult.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,AttributeValue,Amount,DistributionType,"
        "DistributionFrequencyId,DistributionSD,DistributionMin,DistributionMax\n"
        ",,,,,Aquaculture_expansion,,2.0,,,,,\n"
    )
    kwargs["transition_adjacency_setting_csv"] = setting
    kwargs["transition_adjacency_mult_csv"] = mult

    engine = StrategiccEngine(**kwargs)
    engine.load()
    engine.run()   # must not raise
    assert engine.iter_dirs


# ── Transition multipliers sampled during a real run ────────────────────────

def test_run_with_transition_multiplier_sampled(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["use_trans_multiplier"] = True
    mult_csv = tmp_path / "TransMult.csv"
    mult_csv.write_text(
        "TransitionGroupId,DistributionType,DistributionMin,DistributionMax\n"
        "Aquaculture_expansion [Type],Uniform,0.5,1.5\n"
    )
    kwargs["trans_mult_csv"] = mult_csv

    engine = StrategiccEngine(**kwargs)
    engine.load()
    engine.run()

    out = (engine.iter_dirs[0]).parent  # just confirm no crash + output exists
    assert engine.iter_dirs[0].exists()


# ── Patch growing: with and without an active target ────────────────────────

def _size_dist_csv(path):
    path.write_text(
        "Transition Type/Group,Maximum Area (Hectares),Relative Amount\n"
        "Aquaculture_expansion [Type],5,50\n"
        "Aquaculture_expansion [Type],50,50\n"
    )


def test_run_with_patch_growing_no_target(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    size_csv = tmp_path / "SizeDist.csv"
    _size_dist_csv(size_csv)
    kwargs["transition_size_csv"] = size_csv

    engine = StrategiccEngine(**kwargs)
    engine.load()
    assert "Aquaculture_expansion" in engine._size_bins
    engine.run()
    assert engine.iter_dirs


def test_run_with_patch_growing_and_target(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    size_csv = tmp_path / "SizeDist.csv"
    _size_dist_csv(size_csv)
    kwargs["transition_size_csv"] = size_csv

    targets_csv = tmp_path / "Targets.csv"
    targets_csv.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,Amount,DistributionType,DistributionFrequencyId,"
        "DistributionSD,DistributionMin,DistributionMax\n"
        ",0,,,,Aquaculture_expansion,2,,,,,\n"
    )
    kwargs["transition_targets_csv"] = targets_csv

    engine = StrategiccEngine(**kwargs)
    engine.load()
    engine.run()   # must not raise (exercises the has_target + patch branch)
    assert engine.iter_dirs


def test_run_with_target_only_no_patch_growing(tmp_path, missing):
    """A target with NO transition_size_csv exercises the
    scale_probability_to_target() (non-patch) branch instead."""
    kwargs = _base_kwargs(tmp_path, missing)
    targets_csv = tmp_path / "Targets.csv"
    targets_csv.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "TransitionGroupId,Amount,DistributionType,DistributionFrequencyId,"
        "DistributionSD,DistributionMin,DistributionMax\n"
        ",0,,,,Aquaculture_expansion,2,,,,,\n"
    )
    kwargs["transition_targets_csv"] = targets_csv

    engine = StrategiccEngine(**kwargs)
    engine.load()
    engine.run()
    assert engine.iter_dirs


# ── Transition-event raster save: stride skip branch ────────────────────────

def test_run_transition_event_raster_stride_skip(tmp_path, missing):
    kwargs = _base_kwargs(tmp_path, missing)
    kwargs["n_timesteps"] = 4
    cfg.RASTER_OUTPUT_TRANSITION_EVENTS = True
    cfg.RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS = 2   # skip every other timestep

    engine = StrategiccEngine(**kwargs)
    engine.load()
    engine.run()

    events_dir = engine.iter_dirs[0] / "transition_events"
    saved = sorted(p.name for p in events_dir.glob("*.tif"))
    # With stride=2 over 4 timesteps (t=0..3) plus always-keep-last,
    # fewer than 4 files should be saved.
    assert 0 < len(saved) < 4
