"""
tests/test_outputs.py

Unit tests for strategicc/outputs.py: build_summary_tables, the plotting
functions (area/transition envelope, spatial summary, LULC/transition
maps), aggregate_spatial, and modal_to_area_table.

Plots are checked by asserting the expected file exists and is non-empty
(matplotlib output content is not asserted pixel-by-pixel).
"""

import numpy as np
import pandas as pd
import pytest

from strategicc import outputs
from strategicc.io.csv_loader import StateClass
from strategicc.core.transitions import TransitionRecord


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def classes():
    return {
        1: StateClass(id=1, name="Forest", full_name="Forest:All", color=(255, 0, 128, 0)),
        2: StateClass(id=2, name="Water",  full_name="Water:All",  color=(255, 0, 0, 255)),
    }


@pytest.fixture
def area_df():
    rows = []
    for it in range(3):
        for year in (2020, 2021):
            for cid, area in ((1, 100.0 + it), (2, 50.0 + it)):
                rows.append({"iteration": it, "year": year, "class_id": cid, "area_ha": area})
    return pd.DataFrame(rows)


@pytest.fixture
def trans_df():
    rows = []
    for it in range(3):
        for year in (2020, 2021):
            rows.append({"iteration": it, "year": year, "group": "G1", "row": 0, "col": 0})
            rows.append({"iteration": it, "year": year, "group": "G2", "row": 1, "col": 1})
    return pd.DataFrame(rows)


# ── build_summary_tables ─────────────────────────────────────────────────────

def test_build_summary_tables_concatenates_iterations(tmp_path):
    iter_dirs = []
    for i in range(2):
        d = tmp_path / f"iter_{i}"
        d.mkdir()
        pd.DataFrame({"year": [2020], "class_id": [1], "area_ha": [10.0 + i]}).to_csv(d / "area_table.csv", index=False)
        pd.DataFrame({"year": [2020], "group": ["G1"]}).to_csv(d / "transition_log.csv", index=False)
        iter_dirs.append(d)

    summary_dir = tmp_path / "summary"
    area_df, trans_df = outputs.build_summary_tables(iter_dirs, summary_dir)

    assert len(area_df) == 2
    assert len(trans_df) == 2
    assert (summary_dir / "area_all_iterations.csv").exists()
    assert (summary_dir / "transitions_all_iterations.csv").exists()


def test_build_summary_tables_missing_files_are_skipped(tmp_path):
    iter_dirs = [tmp_path / "iter_empty"]
    iter_dirs[0].mkdir()
    summary_dir = tmp_path / "summary"

    area_df, trans_df = outputs.build_summary_tables(iter_dirs, summary_dir)
    assert area_df.empty
    assert trans_df.empty


def test_area_col_detects_unit_column():
    df = pd.DataFrame({"year": [2020], "class_id": [1], "area_km2": [1.0]})
    assert outputs._area_col(df) == "area_km2"


def test_area_col_raises_when_missing():
    df = pd.DataFrame({"year": [2020]})
    with pytest.raises(ValueError, match="No area column found"):
        outputs._area_col(df)


# ── plot_area_envelope ───────────────────────────────────────────────────────

def test_plot_area_envelope_creates_file(tmp_path, area_df, classes):
    outputs.plot_area_envelope(area_df, classes, tmp_path)
    out = tmp_path / "area_envelope.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_area_envelope_empty_df_skips(tmp_path, classes, capsys):
    outputs.plot_area_envelope(pd.DataFrame(), classes, tmp_path)
    assert not (tmp_path / "area_envelope.png").exists()
    assert "Skip" in capsys.readouterr().out


# ── plot_transition_envelope ─────────────────────────────────────────────────

def test_plot_transition_envelope_creates_file(tmp_path, trans_df):
    outputs.plot_transition_envelope(trans_df, tmp_path)
    out = tmp_path / "transition_envelope.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_transition_envelope_empty_df_skips(tmp_path, capsys):
    outputs.plot_transition_envelope(pd.DataFrame(), tmp_path)
    assert not (tmp_path / "transition_envelope.png").exists()
    assert "Skip" in capsys.readouterr().out


# ── plot_lulc_maps ───────────────────────────────────────────────────────────

def test_plot_lulc_maps_multiple_panels(tmp_path, classes):
    maps = [np.ones((4, 4), dtype=np.uint8), np.full((4, 4), 2, dtype=np.uint8)]
    outputs.plot_lulc_maps(maps, classes, start_year=2020, out_dir=tmp_path)
    out = tmp_path / "lulc_maps.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_lulc_maps_single_panel(tmp_path, classes):
    """n == 1 triggers the axes-wrapping branch (axes = [axes])."""
    maps = [np.ones((3, 3), dtype=np.uint8)]
    outputs.plot_lulc_maps(maps, classes, start_year=2020, out_dir=tmp_path)
    assert (tmp_path / "lulc_maps.png").exists()


# ── plot_transition_maps ─────────────────────────────────────────────────────

def test_plot_transition_maps_multiple_panels(tmp_path, classes):
    recs_year1 = [TransitionRecord(year=2020, row=0, col=0, from_id=1, to_id=2, group="G1")]
    recs_year2 = [TransitionRecord(year=2021, row=1, col=1, from_id=2, to_id=1, group="G1")]
    outputs.plot_transition_maps(
        [recs_year1, recs_year2], map_shape=(3, 3), classes=classes,
        start_year=2020, out_dir=tmp_path,
    )
    out = tmp_path / "transition_maps.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_transition_maps_single_panel(tmp_path, classes):
    recs = [TransitionRecord(year=2020, row=0, col=0, from_id=1, to_id=2, group="G1")]
    outputs.plot_transition_maps(
        [recs], map_shape=(2, 2), classes=classes, start_year=2020, out_dir=tmp_path,
    )
    assert (tmp_path / "transition_maps.png").exists()


# ── aggregate_spatial ────────────────────────────────────────────────────────

def _write_tif(path, arr):
    from PIL import Image
    Image.fromarray(arr, mode="L").save(str(path))


def test_aggregate_spatial_modal_and_uncertainty(tmp_path):
    iter_dirs = []
    # 3 iterations, 2 of which agree on class 1 for every cell, 1 disagrees.
    for i, val in enumerate([1, 1, 2]):
        d = tmp_path / f"iter_{i}"
        d.mkdir()
        _write_tif(d / "lulc_2020.tif", np.full((2, 2), val, dtype=np.uint8))
        iter_dirs.append(d)

    summary_dir = tmp_path / "summary"
    modal_maps = outputs.aggregate_spatial(
        iter_dirs=iter_dirs, start_year=2020, n_timesteps=0,
        src_tags={}, summary_dir=summary_dir, uncertainty=True,
    )

    assert set(modal_maps.keys()) == {2020}
    assert np.all(modal_maps[2020] == 1)
    assert (summary_dir / "lulc_mean_2020.tif").exists()
    assert (summary_dir / "uncertainty_2020.tif").exists()


def test_aggregate_spatial_skips_missing_year(tmp_path, capsys):
    d = tmp_path / "iter_0"
    d.mkdir()
    # only lulc_2020.tif exists; timestep for 2021 has nothing
    _write_tif(d / "lulc_2020.tif", np.ones((2, 2), dtype=np.uint8))

    summary_dir = tmp_path / "summary"
    modal_maps = outputs.aggregate_spatial(
        iter_dirs=[d], start_year=2020, n_timesteps=1,
        src_tags={}, summary_dir=summary_dir, uncertainty=False,
    )
    assert set(modal_maps.keys()) == {2020}
    assert "Skip" in capsys.readouterr().out


def test_aggregate_spatial_no_uncertainty_no_file(tmp_path):
    d = tmp_path / "iter_0"
    d.mkdir()
    _write_tif(d / "lulc_2020.tif", np.ones((2, 2), dtype=np.uint8))
    summary_dir = tmp_path / "summary"

    outputs.aggregate_spatial(
        iter_dirs=[d], start_year=2020, n_timesteps=0,
        src_tags={}, summary_dir=summary_dir, uncertainty=False,
    )
    assert not (summary_dir / "uncertainty_2020.tif").exists()


# ── modal_to_area_table ──────────────────────────────────────────────────────

def test_modal_to_area_table_basic(classes):
    modal_maps = {2020: np.array([[1, 1], [2, 2]], dtype=np.uint8)}
    df = outputs.modal_to_area_table(modal_maps, classes, px_area=0.5, area_unit="ha")

    assert list(df.columns) == ["year", "class_id", "class_name", "area_ha"]
    row1 = df[(df["year"] == 2020) & (df["class_id"] == 1)].iloc[0]
    row2 = df[(df["year"] == 2020) & (df["class_id"] == 2)].iloc[0]
    assert row1["area_ha"] == pytest.approx(1.0)   # 2 px * 0.5
    assert row2["area_ha"] == pytest.approx(1.0)
    assert row1["class_name"] == "Forest"


def test_modal_to_area_table_multiple_years_sorted(classes):
    modal_maps = {
        2021: np.array([[1]], dtype=np.uint8),
        2020: np.array([[2]], dtype=np.uint8),
    }
    df = outputs.modal_to_area_table(modal_maps, classes, px_area=1.0, area_unit="km2")
    assert df["year"].tolist()[:2] == [2020, 2020]   # 2020 sorted before 2021


# ── plot_spatial_summary ─────────────────────────────────────────────────────

def test_plot_spatial_summary_creates_file(tmp_path, classes):
    initial = np.ones((2, 2), dtype=np.uint8)
    modal_maps = {
        2020: initial,
        2022: np.full((2, 2), 2, dtype=np.uint8),
    }
    outputs.plot_spatial_summary(
        initial_lulc=initial, modal_maps=modal_maps, classes=classes,
        start_year=2020, n_timesteps=2, summary_dir=tmp_path, uncertainty=False,
    )
    out = tmp_path / "spatial_summary.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_spatial_summary_with_uncertainty_files(tmp_path, classes):
    initial = np.ones((2, 2), dtype=np.uint8)
    mid = np.full((2, 2), 2, dtype=np.uint8)
    final = np.full((2, 2), 1, dtype=np.uint8)
    modal_maps = {2020: initial, 2021: mid, 2022: final}

    # aggregate_spatial normally writes these; write directly for this test
    from PIL import Image
    Image.fromarray(np.full((2, 2), 50, dtype=np.uint8), mode="L").save(str(tmp_path / "uncertainty_2021.tif"))
    Image.fromarray(np.full((2, 2), 80, dtype=np.uint8), mode="L").save(str(tmp_path / "uncertainty_2022.tif"))

    outputs.plot_spatial_summary(
        initial_lulc=initial, modal_maps=modal_maps, classes=classes,
        start_year=2020, n_timesteps=2, summary_dir=tmp_path, uncertainty=True,
    )
    assert (tmp_path / "spatial_summary.png").exists()


def test_plot_spatial_summary_missing_map_hides_panel(tmp_path, classes):
    """When a modal map for mid/final year is missing, the top-row panel
    is hidden (ax.set_visible(False)) rather than raising."""
    initial = np.ones((2, 2), dtype=np.uint8)
    modal_maps = {2020: initial}   # no mid or final year present

    outputs.plot_spatial_summary(
        initial_lulc=initial, modal_maps=modal_maps, classes=classes,
        start_year=2020, n_timesteps=2, summary_dir=tmp_path, uncertainty=False,
    )
    assert (tmp_path / "spatial_summary.png").exists()


def test_plot_spatial_summary_missing_uncertainty_tif_hides_panel(tmp_path, classes):
    """When uncertainty=True but the uncertainty_<year>.tif file is
    missing on disk, that bottom-row panel is hidden rather than raising."""
    initial = np.ones((2, 2), dtype=np.uint8)
    mid = np.full((2, 2), 2, dtype=np.uint8)
    final = np.full((2, 2), 1, dtype=np.uint8)
    modal_maps = {2020: initial, 2021: mid, 2022: final}
    # deliberately do not write uncertainty_2021.tif / uncertainty_2022.tif

    outputs.plot_spatial_summary(
        initial_lulc=initial, modal_maps=modal_maps, classes=classes,
        start_year=2020, n_timesteps=2, summary_dir=tmp_path, uncertainty=True,
    )
    assert (tmp_path / "spatial_summary.png").exists()
