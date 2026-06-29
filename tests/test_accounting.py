"""
tests/test_accounting.py  —  v2.2
Unit tests for SEEA-EA accounting module.
Uses area_modal_df (from modal maps) as primary input.
"""

import pytest
import pandas as pd
import numpy as np
from strategicc.accounting.csv_loader import EcosystemService
from strategicc.accounting.seea import SEEAAccount
from strategicc.io.csv_loader import StateClass
from strategicc.outputs import modal_to_area_table, _area_col


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_classes():
    return {
        1: StateClass(1, "Forest",   "Forest:All",   (255,0,100,0)),
        2: StateClass(2, "Cropland", "Cropland:All", (255,255,255,0)),
    }

def make_services():
    return [
        EcosystemService("Forest",   "Carbon",  "Regulating",  10_000, "IDR", "MgC/ha", 100),
        EcosystemService("Forest",   "Tourism", "Cultural",     5_000, "IDR", None,      None),
        EcosystemService("Cropland", "Crops",   "Provisioning", 8_000, "IDR", "kg/ha",   500),
    ]

def make_modal_maps():
    """2 years × 2×3 grid: mostly Forest, some Cropland."""
    return {
        2022: np.array([[1,1,2],[1,2,2]], dtype=np.uint8),   # 3 Forest, 3 Cropland
        2023: np.array([[1,1,2],[2,2,2]], dtype=np.uint8),   # 2 Forest, 4 Cropland
        2024: np.array([[1,2,2],[2,2,2]], dtype=np.uint8),   # 1 Forest, 5 Cropland
    }

def make_area_modal_df():
    return modal_to_area_table(make_modal_maps(), make_classes(),
                               px_area=0.01, area_unit="ha")

def make_raw_area_df():
    """Raw per-iteration area (for uncertainty summary)."""
    rows = []
    for it in [1, 2]:
        for yr in [2022, 2023, 2024]:
            rows.append({"iteration":it,"year":yr,"class_id":1,
                         "class_name":"Forest",  "area_ha": 0.03 - (yr-2022)*0.01})
            rows.append({"iteration":it,"year":yr,"class_id":2,
                         "class_name":"Cropland","area_ha": 0.03 + (yr-2022)*0.01})
    return pd.DataFrame(rows)

def make_trans_df():
    rows = []
    for it in [1, 2]:
        rows.append({"iteration":it,"year":2022,"row":0,"col":2,
                     "from_class":"Forest","to_class":"Cropland",
                     "group":"Agriculture_expansion"})
    return pd.DataFrame(rows)

def make_acct(**kwargs):
    defaults = dict(
        area_modal_df = make_area_modal_df(),
        trans_df      = make_trans_df(),
        services      = make_services(),
        classes       = make_classes(),
        px_area       = 0.01,
        area_df       = make_raw_area_df(),
    )
    defaults.update(kwargs)
    return SEEAAccount(**defaults)


# ── Tests: modal_to_area_table ────────────────────────────────────────────────

def test_modal_to_area_table_ha():
    df = modal_to_area_table(make_modal_maps(), make_classes(),
                             px_area=0.09, area_unit="ha")
    assert _area_col(df) == "area_ha"
    assert len(df) == 6   # 3 years × 2 classes

def test_modal_to_area_table_km2():
    df = modal_to_area_table(make_modal_maps(), make_classes(),
                             px_area=0.0009, area_unit="km2")
    assert _area_col(df) == "area_km2"

def test_modal_to_area_table_px():
    df = modal_to_area_table(make_modal_maps(), make_classes(),
                             px_area=1.0, area_unit="px")
    assert _area_col(df) == "area_px"
    # 2022: 3 Forest pixels × 1.0 = 3.0
    row = df[(df["year"]==2022) & (df["class_name"]=="Forest")]
    assert row["area_px"].values[0] == pytest.approx(3.0)

def test_modal_area_values():
    df = modal_to_area_table(make_modal_maps(), make_classes(),
                             px_area=0.01, area_unit="ha")
    # 2022: 3 Forest × 0.01 = 0.03 ha
    row = df[(df["year"]==2022) & (df["class_name"]=="Forest")]
    assert row["area_ha"].values[0] == pytest.approx(0.03)
    # 2024: 1 Forest × 0.01 = 0.01 ha
    row = df[(df["year"]==2024) & (df["class_name"]=="Forest")]
    assert row["area_ha"].values[0] == pytest.approx(0.01)


# ── Tests: SEEAAccount ────────────────────────────────────────────────────────

def test_extent_account_shape():
    ea = make_acct().extent_account()
    assert ea.shape == (3, 2)   # 3 years × 2 classes
    assert "Forest" in ea.columns
    assert "Cropland" in ea.columns

def test_extent_account_values():
    ea = make_acct().extent_account()
    assert ea.loc[2022, "Forest"]   == pytest.approx(0.03)
    assert ea.loc[2024, "Forest"]   == pytest.approx(0.01)
    assert ea.loc[2024, "Cropland"] == pytest.approx(0.05)

def test_monetary_flow_account():
    mf = make_acct().monetary_flow_account()
    assert not mf.empty
    assert mf.sum().sum() > 0

def test_physical_flow_account():
    pf = make_acct().physical_flow_account()
    assert pf is not None
    assert not pf.empty

def test_transition_matrix():
    tm = make_acct().transition_matrix()
    assert tm.loc["Forest", "Cropland"] > 0
    assert tm.loc["Cropland", "Forest"] == 0

def test_value_change_matrix():
    vm = make_acct().value_change_matrix()
    # Forest→Cropland: Cropland(8000) - Forest(15000) = -7000 → negative
    assert vm.loc["Forest", "Cropland"] < 0

def test_change_in_value_total():
    cv = make_acct().change_in_value()
    # Forest shrinking → losing high-value land → total should decline
    assert cv["Total"].iloc[1] < 0

def test_uncertainty_summary_columns():
    unc = make_acct().uncertainty_summary()
    assert unc is not None
    assert "Year" in unc.columns
    assert "Range (%)" in unc.columns

def test_uncertainty_summary_none_when_no_raw():
    acct = make_acct(area_df=None)
    assert acct.uncertainty_summary() is None

def test_mode_a_no_physical():
    """Mode A services (no physical unit) still produce monetary account."""
    services_a = [
        EcosystemService("Forest", "Tourism", "Cultural", 5000, "IDR", None, None),
    ]
    acct = make_acct(services=services_a)
    assert acct.physical_flow_account() is None
    assert not acct.monetary_flow_account().empty

def test_unit_label_in_extent_account():
    """Extent account attrs should carry the unit label."""
    ea = make_acct().extent_account()
    assert ea.attrs.get("unit") == "ha"
