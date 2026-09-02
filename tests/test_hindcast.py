"""
tests/test_hindcast.py

Integration tests for strategicc/validation/hindcast.py's hindcast_run()
(previously at 20% coverage) — both the full (non-lightweight) path and
the lightweight fast-path used by the multiplier-correction optimizer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

import strategicc.config as cfg
from strategicc.calibration.loader import LULCTimeSeries
from strategicc.validation.hindcast import hindcast_run, HindcastResult


@pytest.fixture(autouse=True)
def reset_config_state():
    cfg.reset_manifest_mode()
    yield
    cfg.reset_manifest_mode()


@pytest.fixture
def synthetic_ts():
    """4-year synthetic stack: 10x10, class 1 steadily converting to class 2
    (mimics Mangrove -> Aquaculture expansion), with real georeferencing so
    hindcast_run()'s rasterio write step succeeds."""
    rows, cols = 10, 10
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}

    stack = []
    current = np.ones((rows, cols), dtype=np.uint8)
    current[:, 5:] = 2
    for i in range(4):
        if i > 0:
            current = current.copy()
            current[i, :5] = 2   # convert one more row of class 1 -> class 2 each year
        stack.append(current.copy())

    return LULCTimeSeries(stack=np.array(stack), years=[2019, 2020, 2021, 2022], profile=profile)


def _write_manifest(tmp_path):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    sc_path = inputs_dir / "StateClasses.csv"
    sc_path.write_text(
        "Name,StateLabelXId,StateLabelYId,Id,Color,Legend,Description,IsAutoName\n"
        "Mangrove:All,Mangrove,All,1,\"255,0,100,0\",,,No\n"
        "Aquaculture:All,Aquaculture,All,2,\"255,255,0,255\",,,No\n"
    )
    trans_path = inputs_dir / "Transitions.csv"
    trans_path.write_text(
        "Iteration,Timestep,StratumIdSource,StateClassIdSource,StratumIdDest,"
        "StateClassIdDest,SecondaryStratumId,TertiaryStratumId,TransitionTypeId,"
        "Probability,Proportion,AgeMin,AgeMax,AgeRelative,AgeReset,TSTMin,TSTMax,TSTRelative\n"
        ",,,Mangrove:All,,Aquaculture:All,,,Aquaculture_expansion,0.1,,,,,,,,\n"
    )
    # A placeholder StateClassFileName -- hindcast_run() overrides LULC_PATH
    # with its own initial raster before engine.load() ever reads it.
    dummy_raster = tmp_path / "placeholder.tif"
    rows, cols = 10, 10
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    with rasterio.open(str(dummy_raster), "w", driver="GTiff", dtype="uint8",
                        count=1, height=rows, width=cols, crs="EPSG:4326",
                        transform=transform) as dst:
        dst.write(np.ones((rows, cols), dtype=np.uint8), 1)

    manifest = tmp_path / "RunManifest.txt"
    manifest.write_text(f"""\
StateClassFileName = {dummy_raster} #path
STATE_CLASSES_CSV = {sc_path} #path
TRANSITIONS_CSV = {trans_path} #path
OUT_DIR = {tmp_path / "prod_out"}/ #path
START_YEAR = 2022 #int
N_TIMESTEPS = 10 #int
N_ITERATIONS = 5 #int
RNG_SEED = 3 #int
USE_ADJACENCY = False #bool
USE_SPATIAL_MULT = False #bool
USE_TRANS_MULTIPLIER = False #bool
USE_SEEA = False #bool
USE_AGE = False #bool
""")
    return manifest


# ── Full (non-lightweight) hindcast ─────────────────────────────────────────

def test_hindcast_run_full_pipeline(tmp_path, synthetic_ts, capsys):
    manifest = _write_manifest(tmp_path)
    out_dir = tmp_path / "hindcast_out"
    cache_path = tmp_path / "cache" / "ObservedExtent.csv"

    result = hindcast_run(
        manifest_path=manifest,
        ts=synthetic_ts,
        n_iterations=2,
        out_dir=out_dir,
        cache_path=cache_path,
        flag_threshold_pct=1.0,   # low threshold -> likely flags something
    )

    out = capsys.readouterr().out
    assert isinstance(result, HindcastResult)
    assert not result.extent_comparison.empty
    assert result.area_df is not None and not result.area_df.empty
    assert result.trans_df is not None
    assert result.plot_path is not None
    assert result.plot_path.exists()
    assert "diverged" in out or "No class diverged" in out
    # spatial_agreement should have an entry for at least one shared year
    assert isinstance(result.spatial_agreement, dict)


def test_hindcast_run_no_flagged_classes_with_high_threshold(tmp_path, synthetic_ts, capsys):
    manifest = _write_manifest(tmp_path)
    result = hindcast_run(
        manifest_path=manifest,
        ts=synthetic_ts,
        n_iterations=2,
        out_dir=tmp_path / "hindcast_out2",
        cache_path=tmp_path / "cache2" / "ObservedExtent.csv",
        flag_threshold_pct=1000.0,   # effectively unreachable -> no flags
    )
    out = capsys.readouterr().out
    assert result.flagged_classes == []
    assert result.drift == {}
    assert "No class diverged" in out


# ── Lightweight fast path ────────────────────────────────────────────────────

def test_hindcast_run_lightweight_skips_expensive_steps(tmp_path, synthetic_ts):
    manifest = _write_manifest(tmp_path)
    result = hindcast_run(
        manifest_path=manifest,
        ts=synthetic_ts,
        n_iterations=2,
        out_dir=tmp_path / "hindcast_light",
        cache_path=tmp_path / "cache3" / "ObservedExtent.csv",
        lightweight=True,
    )
    assert not result.extent_comparison.empty
    assert result.spatial_agreement == {}
    assert result.drift == {}
    assert result.flagged_classes == []
    assert result.plot_path is None


# ── start_year validation ────────────────────────────────────────────────────

def test_hindcast_run_invalid_start_year_raises(tmp_path, synthetic_ts):
    manifest = _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="must be earlier than"):
        hindcast_run(
            manifest_path=manifest,
            ts=synthetic_ts,
            start_year=synthetic_ts.years[-1],   # == end_year -> n_timesteps <= 0
            out_dir=tmp_path / "hindcast_bad",
        )


# ── override CSV paths ──────────────────────────────────────────────────────

def test_hindcast_run_transition_mult_override(tmp_path, synthetic_ts):
    manifest = _write_manifest(tmp_path)
    override_csv = tmp_path / "OverrideMult.csv"
    override_csv.write_text(
        "TransitionGroupId,DistributionType,DistributionMin,DistributionMax\n"
        "Aquaculture_expansion [Type],Uniform,0.5,1.5\n"
    )
    dist_override = tmp_path / "OverrideDist.csv"
    dist_override.write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,"
        "DistributionTypeId,ExternalVariableTypeId,ExternalVariableMin,"
        "ExternalVariableMax,Value,ValueDistributionTypeId,"
        "ValueDistributionFrequency,ValueDistributionSD,ValueDistributionMin,"
        "ValueDistributionMax,ValueDistributionRelativeFrequency\n"
    )

    result = hindcast_run(
        manifest_path=manifest,
        ts=synthetic_ts,
        n_iterations=2,
        out_dir=tmp_path / "hindcast_override",
        cache_path=tmp_path / "cache4" / "ObservedExtent.csv",
        transition_mult_csv_override=override_csv,
        distributions_csv_override=dist_override,
        lightweight=True,
    )
    assert cfg.TRANSITION_MULT_CSV == override_csv
    assert cfg.DISTRIBUTIONS_CSV == dist_override
    assert not result.extent_comparison.empty
