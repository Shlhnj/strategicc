"""
tests/test_animate.py  —  v3.12.4
Integration tests for strategicc.animate().
"""

import pytest
import numpy as np
from pathlib import Path

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

import strategicc.config as cfg
from strategicc.engine import StrategiccEngine
from strategicc import outputs
from strategicc import animate


@pytest.fixture(autouse=True)
def reset_config_state():
    cfg.reset_manifest_mode()
    yield
    cfg.reset_manifest_mode()


@pytest.fixture
def ready_engine(tmp_path):
    (tmp_path / "inputs").mkdir()
    rows, cols = 15, 15
    lulc = np.ones((rows, cols), dtype=np.uint8)
    lulc[:, 8:] = 2

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
        ",,,Mangrove:All,,Aquaculture:All,,,Aquaculture_expansion,0.03,,,,,,,, \n"
    )

    missing = tmp_path / "inputs" / "missing.csv"
    out_dir = tmp_path / "out"

    engine = StrategiccEngine(
        lulc_path=lulc_path, state_classes_csv=sc_path, transitions_csv=trans_path,
        spatial_mult_csv=missing, trans_mult_csv=missing,
        ecosystem_services_csv=missing, mult_dir=tmp_path / "m",
        out_dir=out_dir,
        start_year=2022, n_timesteps=3, n_iterations=2, rng_seed=1,
        use_adjacency=False, use_spatial_mult=False, use_trans_multiplier=False,
        use_seea=False, use_age=False,
    )
    engine.load()
    engine.run()

    summary_dir = out_dir / "summary"
    area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)
    modal_maps = outputs.aggregate_spatial(
        iter_dirs=engine.iter_dirs, start_year=engine.start_year,
        n_timesteps=engine.n_timesteps, src_tags=engine.src_tags,
        summary_dir=summary_dir, uncertainty=False,
    )
    area_modal_df = outputs.modal_to_area_table(
        modal_maps=modal_maps, classes=engine.classes,
        px_area=engine.px_area, area_unit=engine.area_unit,
    )
    area_modal_df.to_csv(summary_dir / "area_modal.csv", index=False)

    cfg.STATE_CLASSES_CSV = sc_path
    return engine


# ── Basic rendering ────────────────────────────────────────────────────────────

def test_animate_gif_panel_none(ready_engine, tmp_path):
    path = animate(
        out_dir=ready_engine.out_dir, panel=None, output_format="gif",
        output_path=tmp_path / "test1.gif",
    )
    assert path.exists()
    assert path.stat().st_size > 0

def test_animate_gif_area_per_class(ready_engine, tmp_path):
    path = animate(
        out_dir=ready_engine.out_dir, panel="area_per_class", output_format="gif",
        output_path=tmp_path / "test2.gif",
    )
    assert path.exists()

def test_animate_gif_transitions_out(ready_engine, tmp_path):
    path = animate(
        out_dir=ready_engine.out_dir, panel="transitions_out", output_format="gif",
        output_path=tmp_path / "test3.gif",
    )
    assert path.exists()

def test_animate_gif_transitions_in(ready_engine, tmp_path):
    path = animate(
        out_dir=ready_engine.out_dir, panel="transitions_in", output_format="gif",
        output_path=tmp_path / "test4.gif",
    )
    assert path.exists()

def test_animate_default_output_path(ready_engine):
    path = animate(out_dir=ready_engine.out_dir, panel=None, output_format="gif")
    assert path == ready_engine.out_dir / "animation.gif"
    assert path.exists()


# ── Year range filtering ───────────────────────────────────────────────────────

def test_animate_start_end_year_filter(ready_engine, tmp_path):
    path = animate(
        out_dir=ready_engine.out_dir, panel=None, start_year=2023, end_year=2024,
        output_format="gif", output_path=tmp_path / "filtered.gif",
    )
    assert path.exists()

def test_animate_invalid_year_range_raises(ready_engine, tmp_path):
    with pytest.raises(ValueError, match="No years available"):
        animate(
            out_dir=ready_engine.out_dir, panel=None,
            start_year=1900, end_year=1901, output_format="gif",
            output_path=tmp_path / "bad.gif",
        )


# ── Validation ──────────────────────────────────────────────────────────────────

def test_animate_invalid_panel_raises(ready_engine):
    with pytest.raises(ValueError, match="Unknown panel"):
        animate(out_dir=ready_engine.out_dir, panel="not_a_real_panel")

def test_animate_invalid_format_raises(ready_engine):
    with pytest.raises(ValueError, match="output_format"):
        animate(out_dir=ready_engine.out_dir, panel=None, output_format="avi")

def test_animate_missing_spatial_dir_raises(tmp_path):
    empty_out = tmp_path / "empty_out"
    empty_out.mkdir()
    with pytest.raises(ValueError, match="No lulc_mean"):
        animate(out_dir=empty_out, panel=None)


# ── Historical merge ────────────────────────────────────────────────────────────

def test_animate_with_historical_ts(ready_engine, tmp_path):
    import zipfile
    rows, cols = 15, 15
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}

    hist_dir = tmp_path / "hist_src"
    hist_dir.mkdir()
    for year in [2019, 2020, 2021]:
        arr = np.ones((rows, cols), dtype=np.uint8)
        with rasterio.open(str(hist_dir / f"{year}.tif"), "w", **profile) as dst:
            dst.write(arr, 1)

    hist_zip = tmp_path / "historical.zip"
    with zipfile.ZipFile(hist_zip, "w") as zf:
        for year in [2019, 2020, 2021]:
            zf.write(hist_dir / f"{year}.tif", arcname=f"{year}.tif")

    from strategicc.calibration import load_lulc_timeseries
    ts = load_lulc_timeseries(hist_zip, extract_dir=tmp_path / "hist_extract")

    path = animate(
        out_dir=ready_engine.out_dir, panel=None, historical_ts=ts,
        output_format="gif", output_path=tmp_path / "merged.gif",
    )
    assert path.exists()

def test_animate_historical_years_excluded_outside_range(ready_engine, tmp_path):
    import zipfile
    rows, cols = 15, 15
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}

    hist_dir = tmp_path / "hist_src2"
    hist_dir.mkdir()
    arr = np.ones((rows, cols), dtype=np.uint8)
    with rasterio.open(str(hist_dir / "2020.tif"), "w", **profile) as dst:
        dst.write(arr, 1)

    hist_zip = tmp_path / "historical2.zip"
    with zipfile.ZipFile(hist_zip, "w") as zf:
        zf.write(hist_dir / "2020.tif", arcname="2020.tif")

    from strategicc.calibration import load_lulc_timeseries
    ts = load_lulc_timeseries(hist_zip, extract_dir=tmp_path / "hist_extract2")

    path = animate(
        out_dir=ready_engine.out_dir, panel=None, historical_ts=ts,
        start_year=2022, output_format="gif",
        output_path=tmp_path / "sim_only.gif",
    )
    assert path.exists()


# ── v3.12.4: historical overlay on the right panel ──────────────────────────────

def _make_historical_ts(tmp_path, years, rows=15, cols=15, folder_name="hist_src3"):
    import zipfile
    transform = from_origin(110.0, -7.0, 0.001, 0.001)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
               "height": rows, "width": cols,
               "crs": "EPSG:4326", "transform": transform}

    hist_dir = tmp_path / folder_name
    hist_dir.mkdir()
    for year in years:
        arr = np.ones((rows, cols), dtype=np.uint8)
        arr[:, 8:] = 2
        with rasterio.open(str(hist_dir / f"{year}.tif"), "w", **profile) as dst:
            dst.write(arr, 1)

    hist_zip = tmp_path / f"{folder_name}.zip"
    with zipfile.ZipFile(hist_zip, "w") as zf:
        for year in years:
            zf.write(hist_dir / f"{year}.tif", arcname=f"{year}.tif")

    from strategicc.calibration import load_lulc_timeseries
    return load_lulc_timeseries(hist_zip, extract_dir=tmp_path / f"{folder_name}_extract")


def test_animate_area_per_class_historical_overlay(ready_engine, tmp_path):
    ts = _make_historical_ts(tmp_path, [2019, 2020, 2021])
    path = animate(
        out_dir=ready_engine.out_dir, panel="area_per_class", historical_ts=ts,
        px_area_ha=ready_engine.px_area_ha,
        output_format="gif", output_path=tmp_path / "area_hist.gif",
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_animate_area_per_class_historical_requires_px_area_ha(
    ready_engine, tmp_path
):
    ts = _make_historical_ts(tmp_path, [2019, 2020, 2021], folder_name="hist_src4")
    with pytest.raises(ValueError, match="requires 'px_area_ha'"):
        animate(
            out_dir=ready_engine.out_dir, panel="area_per_class", historical_ts=ts,
            output_format="gif", output_path=tmp_path / "area_hist_missing.gif",
        )


def test_animate_non_area_panel_historical_warns(ready_engine, tmp_path, capsys):
    ts = _make_historical_ts(tmp_path, [2019, 2020, 2021], folder_name="hist_src5")
    path = animate(
        out_dir=ready_engine.out_dir, panel="value_per_class", historical_ts=ts,
        output_format="gif", output_path=tmp_path / "value_hist.gif",
    )
    assert path.exists()
    captured = capsys.readouterr()
    assert "no historical equivalent" in captured.out
