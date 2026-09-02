"""
tests/test_run.py

End-to-end integration tests for strategicc/run.py — main() and
seea_only() — previously at 0% coverage. Uses a tiny synthetic 2-class
scenario (small raster, minimal CSVs) so the full pipeline runs fast,
rather than the full example inputs/ dataset.
"""

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

import strategicc.config as cfg
from strategicc.engine import StrategiccEngine
from strategicc.run import main, seea_only, RunNotFoundError

INPUTS = Path(__file__).resolve().parent.parent / "inputs"


@pytest.fixture(autouse=True)
def reset_config_state():
    cfg.reset_manifest_mode()
    yield
    cfg.reset_manifest_mode()


def _write_lulc(path):
    rows, cols = 8, 8
    lulc = np.ones((rows, cols), dtype=np.uint8)
    lulc[:, 4:] = 2
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


def _write_transitions(path):
    path.write_text(
        "Iteration,Timestep,StratumIdSource,StateClassIdSource,StratumIdDest,"
        "StateClassIdDest,SecondaryStratumId,TertiaryStratumId,TransitionTypeId,"
        "Probability,Proportion,AgeMin,AgeMax,AgeRelative,AgeReset,TSTMin,TSTMax,TSTRelative\n"
        ",,,Mangrove:All,,Aquaculture:All,,,Aquaculture_expansion,0.1,,,,,,,,\n"
    )


def _write_ecosystem_services(path):
    path.write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,"
        "PhysicalUnit,PhysicalValuePerUnitArea,StockFlowSource\n"
        "Mangrove,Carbon Storage,Regulating,1000,USD,,,\n"
        "Aquaculture,Provisioning,Provisioning,500,USD,,,\n"
    )


def _configure_direct_mode(tmp_path, *, use_seea, out_dir=None, use_stockflow=False):
    """Populate strategicc.config with a tiny, fast, valid scenario using
    direct-mode attribute assignment (mirrors what a notebook does)."""
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    lulc_path = tmp_path / "2022.tif"
    _write_lulc(lulc_path)
    sc_path = inputs_dir / "StateClasses.csv"
    _write_state_classes(sc_path)
    trans_path = inputs_dir / "Transitions.csv"
    _write_transitions(trans_path)
    missing = inputs_dir / "missing.csv"

    cfg.LULC_PATH = lulc_path
    cfg.STATE_CLASSES_CSV = sc_path
    cfg.TRANSITIONS_CSV = trans_path
    cfg.SPATIAL_MULT_CSV = missing
    cfg.TRANSITION_MULT_CSV = missing
    cfg.DISTRIBUTIONS_CSV = missing
    cfg.MULT_DIR = tmp_path / "mult"
    cfg.OUT_DIR = out_dir or (tmp_path / "out")
    cfg.START_YEAR = 2022
    cfg.N_TIMESTEPS = 2
    cfg.N_ITERATIONS = 2
    cfg.RNG_SEED = 1
    cfg.AREA_UNIT = "ha"
    cfg.USE_ADJACENCY = False
    cfg.USE_SPATIAL_MULT = False
    cfg.USE_TRANS_MULTIPLIER = False
    cfg.USE_AGE = False
    cfg.AGE_RASTER_PATH = None
    cfg.AGE_INITIAL_CSV = None
    cfg.SAVE_AGE_RASTERS = False
    cfg.TRANSITION_SIZE_CSV = missing
    cfg.TRANSITION_TARGETS_CSV = missing
    cfg.TRANSITION_ADJACENCY_SETTING_CSV = missing
    cfg.TRANSITION_ADJACENCY_MULT_CSV = missing
    cfg.ASSET_VALUATION_PARAMS_CSV = missing
    cfg.FETCH_INITIAL_SC_FROM_ZIP = False

    if use_stockflow:
        cfg.USE_STOCKFLOW = True
        cfg.STOCK_TYPE_CSV = INPUTS / "StockType.csv"
        cfg.FLOW_TYPE_CSV = INPUTS / "FlowType.csv"
        cfg.FLOW_ORDER_CSV = INPUTS / "FlowOrder.csv"
        cfg.FLOW_PATHWAYS_CSV = INPUTS / "FlowPathways.csv"
        cfg.STATE_ATTRIBUTE_VALUES_CSV = INPUTS / "StateAttributeValues.csv"
        cfg.FLOW_MULTIPLIER_CSV = INPUTS / "Flow Multipliers.csv"
        cfg.INITIAL_STOCK_NON_SPATIAL_CSV = missing
    else:
        cfg.USE_STOCKFLOW = False

    if use_seea:
        eco_path = inputs_dir / "EcosystemServices.csv"
        _write_ecosystem_services(eco_path)
        cfg.ECOSYSTEM_SERVICES_CSV = eco_path
        cfg.USE_SEEA = True
    else:
        cfg.ECOSYSTEM_SERVICES_CSV = missing
        cfg.USE_SEEA = False

    return cfg.OUT_DIR


# ── main() — full pipeline, no SEEA/stockflow ──────────────────────────────

def test_main_runs_full_pipeline_without_seea(tmp_path, capsys):
    out_dir = _configure_direct_mode(tmp_path, use_seea=False)
    main()

    out = capsys.readouterr().out
    assert "[OK] Done." in out

    summary_dir = out_dir / "summary"
    assert (summary_dir / "area_all_iterations.csv").exists()
    assert (summary_dir / "transitions_all_iterations.csv").exists()
    assert (summary_dir / "area_envelope.png").exists()
    assert (summary_dir / "transition_envelope.png").exists()
    assert (summary_dir / "area_modal.csv").exists()
    assert (summary_dir / "spatial_summary.png").exists()
    assert (summary_dir / "diagnostic_iter1" / "lulc_maps.png").exists()


# ── main() — full pipeline including SEEA ───────────────────────────────────

def test_main_runs_full_pipeline_with_seea(tmp_path, capsys):
    out_dir = _configure_direct_mode(tmp_path, use_seea=True)
    main()

    out = capsys.readouterr().out
    assert "[19] Running SEEA-EA ecosystem accounting" in out
    assert "[OK] Done." in out

    seea_csv_dir = out_dir / "seea" / "csv"
    assert seea_csv_dir.exists()
    assert any(seea_csv_dir.glob("*.csv"))


# ── main() — full pipeline including Stock & Flow ───────────────────────────

def test_main_runs_full_pipeline_with_stockflow_and_seea(tmp_path, capsys):
    out_dir = _configure_direct_mode(tmp_path, use_seea=True, use_stockflow=True)
    main()

    out = capsys.readouterr().out
    assert "[18] Aggregating Stock & Flow outputs by class..." in out
    assert "[18b] Building SEEA-EA asset account" in out
    assert "[18c] Building SEEA EA Table 13.3 carbon stock account" in out
    assert "[OK] Done." in out

    summary_dir = out_dir / "summary"
    assert (summary_dir / "stock_by_class.csv").exists()
    assert (summary_dir / "flow_by_class.csv").exists()

    seea_csv_dir = out_dir / "seea" / "csv"
    assert any(seea_csv_dir.glob("*asset_account*"))
    assert any(seea_csv_dir.glob("*carbon_stock*"))


# ── seea_only() — recompute over a stockflow-enabled run ───────────────────

def test_seea_only_skip_stock_accounts_flag(tmp_path):
    """include_stock_accounts=False skips the asset/carbon-stock account
    build+save step while still aggregating stock/flow totals."""
    out_dir = _configure_direct_mode(tmp_path, use_seea=True, use_stockflow=True)
    main()

    engine = StrategiccEngine.from_config()
    engine.load()
    engine.iter_dirs = sorted(out_dir.glob("iter_*"))

    seea_dir = out_dir / "seea"
    for f in (seea_dir / "csv").glob("*asset_account*"):
        f.unlink()
    for f in (seea_dir / "csv").glob("*carbon_stock*"):
        f.unlink()

    seea_only(engine=engine, include_stock_accounts=False, generate_plots=False)

    assert not any((seea_dir / "csv").glob("*asset_account*"))
    assert not any((seea_dir / "csv").glob("*carbon_stock*"))




def test_seea_only_recomputes_from_completed_run(tmp_path, capsys):
    """seea_only(engine=...) reruns accounting from an already-completed
    run's iter_* dirs without re-simulating."""
    out_dir = _configure_direct_mode(tmp_path, use_seea=True)
    main()   # produce a completed run to recompute from

    # Build a fresh engine pointed at the same completed output.
    engine = StrategiccEngine.from_config()
    engine.load()
    engine.iter_dirs = sorted(out_dir.glob("iter_*"))

    acct = seea_only(engine=engine)
    out = capsys.readouterr().out

    assert acct is not None
    assert "seea_only() done" in out
    assert (out_dir / "seea" / "csv").exists()


def test_seea_only_zip_output(tmp_path):
    out_dir = _configure_direct_mode(tmp_path, use_seea=True)
    main()

    engine = StrategiccEngine.from_config()
    engine.load()
    engine.iter_dirs = sorted(out_dir.glob("iter_*"))

    seea_only(engine=engine, zip_output=True)
    archive = out_dir.parent / f"{out_dir.name}_seea.zip"
    assert archive.exists()


def test_seea_only_raises_when_no_completed_iterations(tmp_path):
    out_dir = _configure_direct_mode(tmp_path, use_seea=False, out_dir=tmp_path / "empty_out")
    engine = StrategiccEngine.from_config()
    engine.load()
    engine.iter_dirs = []   # never ran

    with pytest.raises(RunNotFoundError, match="No iter_\\* folders found"):
        seea_only(engine=engine)


def test_seea_only_via_manifest_path(tmp_path):
    """seea_only(manifest_path=...) loads a manifest and builds its own
    engine internally, rather than being handed one."""
    out_dir = _configure_direct_mode(tmp_path, use_seea=True)
    main()

    manifest = tmp_path / "RunManifest.txt"
    manifest.write_text(f"""\
StateClassFileName = {cfg.LULC_PATH} #path
STATE_CLASSES_CSV = {cfg.STATE_CLASSES_CSV} #path
TRANSITIONS_CSV = {cfg.TRANSITIONS_CSV} #path
ECOSYSTEM_SERVICES_CSV = {cfg.ECOSYSTEM_SERVICES_CSV} #path
OUT_DIR = {out_dir}/ #path
START_YEAR = {cfg.START_YEAR} #int
N_TIMESTEPS = {cfg.N_TIMESTEPS} #int
N_ITERATIONS = {cfg.N_ITERATIONS} #int
RNG_SEED = {cfg.RNG_SEED} #int
USE_ADJACENCY = False #bool
USE_SPATIAL_MULT = False #bool
USE_TRANS_MULTIPLIER = False #bool
USE_SEEA = True #bool
USE_AGE = False #bool
""")
    cfg.reset_manifest_mode()
    acct = seea_only(manifest_path=manifest, generate_plots=False, include_stock_accounts=False)
    assert acct is not None
