"""
tests/test_output_options.py  —  v3.2
Integration tests for Output Options gating (SUMMARY_OUTPUT_*, RASTER_OUTPUT_*).
"""

import pytest
import numpy as np
from pathlib import Path

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

import strategicc.config as cfg
from strategicc.engine import StrategiccEngine


@pytest.fixture(autouse=True)
def reset_config_state():
    cfg.reset_manifest_mode()
    yield
    cfg.reset_manifest_mode()
    cfg.SUMMARY_OUTPUT_SC = True
    cfg.SUMMARY_OUTPUT_TR = True
    cfg.RASTER_OUTPUT_SC = True
    cfg.RASTER_OUTPUT_SC_TIMESTEPS = 1
    cfg.RASTER_OUTPUT_AGE = True
    cfg.RASTER_OUTPUT_AGE_TIMESTEPS = 1
    cfg.RASTER_OUTPUT_TRANSITION_EVENTS = True
    cfg.RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS = 1


@pytest.fixture
def minimal_engine(tmp_path):
    (tmp_path / "inputs").mkdir()
    rows, cols = 15, 15
    lulc = np.ones((rows, cols), dtype=np.uint8)
    lulc[:, 7:] = 2

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

    missing = tmp_path / "inputs" / "missing.csv"

    return StrategiccEngine(
        lulc_path=lulc_path, state_classes_csv=sc_path, transitions_csv=trans_path,
        spatial_mult_csv=missing, trans_mult_csv=missing,
        ecosystem_services_csv=missing, mult_dir=tmp_path / "m",
        out_dir=tmp_path / "out",
        start_year=2022, n_timesteps=4, n_iterations=1, rng_seed=1,
        use_adjacency=False, use_spatial_mult=False, use_trans_multiplier=False,
        use_seea=False, use_age=False,
    )


def _run(engine):
    engine.load()
    engine.run()
    return engine.iter_dirs[0]


# ── SUMMARY_OUTPUT gating ──────────────────────────────────────────────────────

def test_summary_output_sc_false_skips_area_table(minimal_engine):
    cfg.SUMMARY_OUTPUT_SC = False
    cfg.SUMMARY_OUTPUT_TR = True
    iter_dir = _run(minimal_engine)
    assert not (iter_dir / "area_table.csv").exists()

def test_summary_output_tr_false_skips_transition_log(minimal_engine):
    cfg.SUMMARY_OUTPUT_SC = True
    cfg.SUMMARY_OUTPUT_TR = False
    iter_dir = _run(minimal_engine)
    assert not (iter_dir / "transition_log.csv").exists()

def test_summary_outputs_true_both_present(minimal_engine):
    cfg.SUMMARY_OUTPUT_SC = True
    cfg.SUMMARY_OUTPUT_TR = True
    iter_dir = _run(minimal_engine)
    assert (iter_dir / "area_table.csv").exists()
    assert (iter_dir / "transition_log.csv").exists()


# ── RASTER_OUTPUT_SC gating ────────────────────────────────────────────────────

def test_raster_output_sc_false_skips_lulc_tifs(minimal_engine):
    cfg.RASTER_OUTPUT_SC = False
    iter_dir = _run(minimal_engine)
    assert len(list(iter_dir.glob("lulc_*.tif"))) == 0

def test_raster_output_sc_stride(minimal_engine):
    cfg.RASTER_OUTPUT_SC = True
    cfg.RASTER_OUTPUT_SC_TIMESTEPS = 2
    iter_dir = _run(minimal_engine)
    tifs = sorted(p.name for p in iter_dir.glob("lulc_*.tif"))
    assert "lulc_2022.tif" in tifs
    assert "lulc_2024.tif" in tifs
    assert "lulc_2026.tif" in tifs


# ── RASTER_OUTPUT_TRANSITION_EVENTS ───────────────────────────────────────────

def test_transition_events_disabled_no_dir(minimal_engine):
    cfg.RASTER_OUTPUT_TRANSITION_EVENTS = False
    iter_dir = _run(minimal_engine)
    assert not (iter_dir / "transition_events").exists()

def test_transition_events_enabled_creates_rasters(minimal_engine):
    cfg.RASTER_OUTPUT_TRANSITION_EVENTS = True
    cfg.RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS = 1
    iter_dir = _run(minimal_engine)
    events_dir = iter_dir / "transition_events"
    assert events_dir.exists()
    assert len(list(events_dir.glob("events_*.tif"))) > 0

def test_transition_event_raster_values_match_destination_class(minimal_engine):
    cfg.RASTER_OUTPUT_TRANSITION_EVENTS = True
    cfg.RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS = 1
    iter_dir = _run(minimal_engine)
    import pandas as pd
    log = pd.read_csv(iter_dir / "transition_log.csv")
    if log.empty:
        pytest.skip("No transitions fired in this synthetic run")
    first_year = log["year"].min()
    arr = np.array(rasterio.open(
        str(iter_dir / "transition_events" / f"events_{first_year}.tif")
    ).read(1))
    year_log = log[log["year"] == first_year]
    for _, row in year_log.iterrows():
        assert arr[row["row"], row["col"]] == 2


# ── Backward compatibility ────────────────────────────────────────────────────

def test_default_output_options_save_everything(minimal_engine):
    iter_dir = _run(minimal_engine)
    assert (iter_dir / "area_table.csv").exists()
    assert (iter_dir / "transition_log.csv").exists()
    assert len(list(iter_dir.glob("lulc_*.tif"))) == 5
