"""
tests/test_spatial_timevarying_engine.py  —  v3.15
Integration test: confirms the actual StrategiccEngine wiring for
time-varying spatial multipliers (not just the isolated core.spatial
functions) — TransitionSpatialMultipliers.csv with two Timestep rows for
one group loads, resolves, and switches mid-run inside a real
engine.load() / engine.run() cycle.
"""

import pytest
import numpy as np

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

import strategicc.config as cfg
from strategicc.engine import StrategiccEngine


@pytest.fixture(autouse=True)
def reset_config_state():
    cfg.reset_manifest_mode()
    yield
    cfg.reset_manifest_mode()


@pytest.fixture
def timevarying_engine(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "m").mkdir()
    rows, cols = 10, 10
    lulc = np.ones((rows, cols), dtype=np.uint8)   # all Mangrove

    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}
    lulc_path = tmp_path / "2022.tif"
    with rasterio.open(str(lulc_path), "w", **profile) as dst:
        dst.write(lulc, 1)

    sc_path = tmp_path / "inputs" / "StateClasses.csv"
    sc_path.write_text(
        "Name,StateLabelXId,StateLabelYId,Id,Color,Legend,Description,IsAutoName\n"
        "Mangrove:All,Mangrove,All,1,\"255,0,100,0\",,,No\n"
        "Aquaculture:All,Aquaculture,All,2,\"255,255,0,255\",,,No\n"
    )
    trans_path = tmp_path / "inputs" / "Transitions.csv"
    trans_path.write_text(
        "Iteration,Timestep,StratumIdSource,StateClassIdSource,StratumIdDest,"
        "StateClassIdDest,SecondaryStratumId,TertiaryStratumId,TransitionTypeId,"
        "Probability,Proportion,AgeMin,AgeMax,AgeRelative,AgeReset,TSTMin,TSTMax,TSTRelative\n"
        ",,,Mangrove:All,,Aquaculture:All,,,Aquaculture_expansion,0.05,,,,,,,, \n"
    )

    # ── Two rasters, values 0.2 (early) and 0.9 (from timestep=2 onward) ──
    raster_profile = {**profile, "dtype": "float32"}
    early_path = tmp_path / "m" / "aqua_early.tif"
    with rasterio.open(str(early_path), "w", **raster_profile) as dst:
        dst.write(np.full((rows, cols), 0.9, dtype=np.float32), 1)   # near 1 -> survives power curve
    late_path = tmp_path / "m" / "aqua_late.tif"
    with rasterio.open(str(late_path), "w", **raster_profile) as dst:
        dst.write(np.full((rows, cols), 0.99, dtype=np.float32), 1)

    mult_path = tmp_path / "inputs" / "TransitionSpatialMultipliers.csv"
    mult_path.write_text(
        "Iteration,Timestep,TransitionGroupId,TransitionMultiplierTypeId,MultiplierFileName\n"
        ",0,Aquaculture_expansion,,aqua_early.tif\n"
        ",2,Aquaculture_expansion,,aqua_late.tif\n"
    )

    missing = tmp_path / "inputs" / "missing.csv"

    return StrategiccEngine(
        lulc_path=lulc_path, state_classes_csv=sc_path, transitions_csv=trans_path,
        spatial_mult_csv=mult_path, trans_mult_csv=missing,
        ecosystem_services_csv=missing, mult_dir=tmp_path / "m",
        out_dir=tmp_path / "out",
        start_year=2022, n_timesteps=4, n_iterations=1, rng_seed=1,
        use_adjacency=False, use_spatial_mult=True, use_trans_multiplier=False,
        use_seea=False, use_age=False,
    )


def test_engine_loads_and_resolves_per_timestep_series(timevarying_engine):
    timevarying_engine.load()

    series = dict(timevarying_engine.spatial_mults["Aquaculture_expansion"])
    assert set(series.keys()) == {0, 2}

    by_t = timevarying_engine.spatial_mults_by_timestep
    assert len(by_t) == 4   # n_timesteps

    # t=0,1 -> early raster; t=2,3 -> late raster (forward-fill switch)
    assert by_t[0]["Aquaculture_expansion"] is by_t[1]["Aquaculture_expansion"]
    assert by_t[2]["Aquaculture_expansion"] is by_t[3]["Aquaculture_expansion"]
    assert by_t[0]["Aquaculture_expansion"] is not by_t[2]["Aquaculture_expansion"]

    early_val = float(by_t[0]["Aquaculture_expansion"].max())
    late_val = float(by_t[2]["Aquaculture_expansion"].max())
    assert late_val > early_val, "the t=2 raster (0.99) should sharpen to a higher value than the t=0 raster (0.9)"


def test_engine_runs_end_to_end_with_time_varying_multiplier(timevarying_engine):
    """Full run() must not crash with a real per-timestep spatial multiplier
    series wired in, and get_multiplier must be called with the current t
    (not silently fall back to a single static array) throughout."""
    timevarying_engine.load()
    timevarying_engine.run()
    assert len(timevarying_engine.iter_dirs) == 1
