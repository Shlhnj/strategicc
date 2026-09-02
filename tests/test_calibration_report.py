"""
tests/test_calibration_report.py

Unit tests for strategicc/calibration/report.py's calibration_summary()
and its internal _plot_calibration_summary(), covering every combination
of present/absent inputs (transitions, temporal multipliers, size
distribution, age raster, manifest) — previously at 9% coverage.
"""

import numpy as np
import pandas as pd
import pytest

from strategicc.calibration.report import calibration_summary
from strategicc.calibration.age import AgeRasterResult, NODATA_AGE


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def transitions_df():
    return pd.DataFrame({
        "TransitionTypeId": ["Aquaculture_expansion", "Urbanization"],
        "Probability": [0.05, 0.02],
    })


@pytest.fixture
def temporal_df():
    return pd.DataFrame({
        "TransitionGroupId": ["Aquaculture_expansion [Type]", "Urbanization [Type]"],
        "DistributionType": ["Aquaculture_expansion Distribution", "Urbanization Distribution"],
        "DistributionMin": [0.5, 0.6],
        "DistributionMax": [1.5, 1.4],
    })


@pytest.fixture
def distributions_df():
    return pd.DataFrame({
        "DistributionTypeId": ["Aquaculture_expansion Distribution"] * 3,
        "Value": [0.8, 1.0, 1.2],
        "ValueDistributionRelativeFrequency": [1, 2, 1],
    })


@pytest.fixture
def size_dist_df():
    return pd.DataFrame({
        "Transition Type/Group": ["Aquaculture_expansion [Type]"] * 3,
        "Maximum Area (Hectares)": [0.05, 0.2, 1.0],
        "Relative Amount": [30.0, 40.0, 30.0],
    })


@pytest.fixture
def age_result():
    rows, cols = 5, 5
    age_combined = np.full((rows, cols), 10, dtype=np.uint16)
    age_combined[0, 0] = NODATA_AGE
    full_record_mask = np.ones((rows, cols), dtype=bool)
    full_record_mask[0, 0] = False
    return AgeRasterResult(
        age_combined=age_combined,
        age_per_class={1: age_combined.copy()},
        full_record_mask=full_record_mask,
        baseline_year=2022,
        profile={},
    )


# ── Full summary, everything present ────────────────────────────────────────

def test_calibration_summary_all_inputs_present(
    tmp_path, transitions_df, temporal_df, distributions_df, size_dist_df, age_result, capsys
):
    manifest = tmp_path / "RunManifest_calibrated.txt"
    manifest.write_text("# fake manifest")
    plot_out = tmp_path / "summary.png"

    result = calibration_summary(
        transitions_df=transitions_df,
        temporal_df=temporal_df,
        distributions_df=distributions_df,
        size_dist_df=size_dist_df,
        age_result=age_result,
        transitions_path=tmp_path / "Transitions.csv",
        temporal_path=tmp_path / "TransitionMultipliers.csv",
        distributions_path=tmp_path / "Distributions.csv",
        size_dist_path=tmp_path / "TransitionSizeDistribution.csv",
        age_raster_path=tmp_path / "age.tif",
        manifest_path=manifest,
        plot_out=plot_out,
    )

    out = capsys.readouterr().out
    assert "STRATEGICC Calibration Summary" in out
    assert "Age distribution table" in out

    assert result["transitions"]["n_pathways"] == 2
    assert result["temporal"]["n_groups"] == 2
    assert result["distributions"]["n_distributions"] == 1
    assert result["size_distribution"]["n_groups"] == 1
    assert result["age"]["n_classes"] == 1
    assert result["age"]["age_min"] == 10
    assert result["manifest"] == str(manifest)
    assert result["plot_path"] == str(plot_out)
    assert plot_out.exists()
    assert plot_out.stat().st_size > 0


# ── Nothing calibrated: every "cross" branch, and the empty-plot early return ──

def test_calibration_summary_nothing_calibrated(tmp_path, capsys):
    plot_out = tmp_path / "summary.png"
    result = calibration_summary(plot_out=plot_out)

    out = capsys.readouterr().out
    assert out.count("not calibrated") == 5   # transitions/temporal/dist/size/age
    assert "not generated" in out             # manifest

    for key in ("transitions", "temporal", "distributions", "size_distribution", "age", "manifest"):
        assert result[key] is None
    # n_rows == 0 -> _plot_calibration_summary returns early, no file written
    assert not plot_out.exists()


def test_calibration_summary_empty_dataframes_treated_as_absent(tmp_path, capsys):
    """Empty (not None) DataFrames must also take the 'not calibrated' branch."""
    result = calibration_summary(
        transitions_df=pd.DataFrame(),
        temporal_df=pd.DataFrame(),
        distributions_df=pd.DataFrame(),
        size_dist_df=pd.DataFrame(),
        plot_out=tmp_path / "summary.png",
    )
    out = capsys.readouterr().out
    assert out.count("not calibrated") == 5  # transitions/temporal/dist/size/age
    assert result["transitions"] is None


# ── Default plot_out (CALIBRATION_DIR) ──────────────────────────────────────

def test_calibration_summary_default_plot_out(tmp_path, monkeypatch, transitions_df):
    """When plot_out isn't given, it defaults to CALIBRATION_DIR / ...png."""
    import strategicc.calibration.paths as paths_mod
    monkeypatch.setattr(paths_mod, "CALIBRATION_DIR", tmp_path / "calibration_result")

    result = calibration_summary(transitions_df=transitions_df)
    expected = tmp_path / "calibration_result" / "calibration_summary.png"
    assert result["plot_path"] == str(expected)
    assert expected.exists()


# ── Individual plot rows in isolation ───────────────────────────────────────

def test_calibration_summary_age_only(tmp_path, age_result):
    plot_out = tmp_path / "age_only.png"
    result = calibration_summary(age_result=age_result, plot_out=plot_out)
    assert plot_out.exists()
    assert result["age"]["n_classes"] == 1


def test_calibration_summary_temporal_only(tmp_path, temporal_df):
    plot_out = tmp_path / "temporal_only.png"
    result = calibration_summary(temporal_df=temporal_df, plot_out=plot_out)
    assert plot_out.exists()
    assert result["temporal"]["n_groups"] == 2


def test_calibration_summary_size_only(tmp_path, size_dist_df):
    plot_out = tmp_path / "size_only.png"
    result = calibration_summary(size_dist_df=size_dist_df, plot_out=plot_out)
    assert plot_out.exists()
    assert result["size_distribution"]["n_groups"] == 1


def test_calibration_summary_transitions_only(tmp_path, transitions_df):
    plot_out = tmp_path / "trans_only.png"
    result = calibration_summary(transitions_df=transitions_df, plot_out=plot_out)
    assert plot_out.exists()
    assert result["transitions"]["prob_min"] == pytest.approx(0.02)
    assert result["transitions"]["prob_max"] == pytest.approx(0.05)
