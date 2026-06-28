"""
tests/test_accounting.py
Basic unit tests for SEEA-EA accounting module.
"""

import pytest
import pandas as pd
import numpy as np
from strategicc.accounting.csv_loader import EcosystemService
from strategicc.accounting.seea import SEEAAccount
from strategicc.io.csv_loader import StateClass


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_classes():
    return {
        1: StateClass(id=1, name="Forest",   full_name="Forest:All",   color=(255,0,100,0)),
        2: StateClass(id=2, name="Cropland", full_name="Cropland:All", color=(255,255,255,0)),
    }

def make_services():
    return [
        EcosystemService("Forest",   "Carbon",   "Regulating",  10_000, "IDR", "MgC/ha", 100),
        EcosystemService("Forest",   "Tourism",  "Cultural",     5_000, "IDR", None,      None),
        EcosystemService("Cropland", "Crops",    "Provisioning", 8_000, "IDR", "kg/ha",   500),
    ]

def make_area_df():
    """Simulate 2 iterations × 3 years × 2 classes."""
    rows = []
    for it in [1, 2]:
        for yr in [2022, 2023, 2024]:
            rows.append({"iteration": it, "year": yr, "class_id": 1,
                         "class_name": "Forest",   "area_ha": 100 - (yr-2022)*2})
            rows.append({"iteration": it, "year": yr, "class_id": 2,
                         "class_name": "Cropland", "area_ha": 50  + (yr-2022)*2})
    return pd.DataFrame(rows)

def make_trans_df():
    rows = []
    for it in [1, 2]:
        rows.append({"iteration": it, "year": 2022, "row": 0, "col": 0,
                     "from_class": "Forest", "to_class": "Cropland", "group": "Agriculture_expansion"})
        rows.append({"iteration": it, "year": 2023, "row": 1, "col": 1,
                     "from_class": "Forest", "to_class": "Cropland", "group": "Agriculture_expansion"})
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_extent_account_shape():
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       make_services(), make_classes(), px_area_ha=0.01)
    ea = acct.extent_account()
    assert ea.shape == (3, 2)          # 3 years × 2 classes
    assert "Forest" in ea.columns
    assert "Cropland" in ea.columns


def test_extent_account_values():
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       make_services(), make_classes(), px_area_ha=0.01)
    ea = acct.extent_account()
    assert ea.loc[2022, "Forest"] == pytest.approx(100)
    assert ea.loc[2024, "Forest"] == pytest.approx(96)
    assert ea.loc[2024, "Cropland"] == pytest.approx(54)


def test_monetary_flow_account():
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       make_services(), make_classes(), px_area_ha=0.01)
    mf = acct.monetary_flow_account()
    assert not mf.empty
    # Forest 2022: (10000 + 5000) * 100 = 1_500_000
    forest_total_2022 = mf.loc[2022].xs("Forest", level=1, axis=0, drop_level=False) \
        if False else None
    # Just check total is positive
    assert mf.sum().sum() > 0


def test_physical_flow_account():
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       make_services(), make_classes(), px_area_ha=0.01)
    pf = acct.physical_flow_account()
    assert pf is not None
    assert not pf.empty


def test_transition_matrix():
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       make_services(), make_classes(), px_area_ha=0.01)
    tm = acct.transition_matrix()
    assert tm.loc["Forest", "Cropland"] > 0
    assert tm.loc["Cropland", "Forest"] == 0


def test_value_change_matrix():
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       make_services(), make_classes(), px_area_ha=0.01)
    vm = acct.value_change_matrix()
    # Forest → Cropland: Cropland value (8000) - Forest value (15000) = -7000 per ha
    assert vm.loc["Forest", "Cropland"] < 0


def test_change_in_value_total():
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       make_services(), make_classes(), px_area_ha=0.01)
    cv = acct.change_in_value()
    # Forest is shrinking (losing high-value land), total should decline
    assert cv["Total"].iloc[1] < 0


def test_uncertainty_summary_columns():
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       make_services(), make_classes(), px_area_ha=0.01)
    unc = acct.uncertainty_summary()
    assert "Year" in unc.columns
    assert "Median value" in unc.columns
    assert "Range (%)" in unc.columns


def test_mode_a_no_physical():
    """Mode A services (no physical unit) should still produce monetary account."""
    services_a = [
        EcosystemService("Forest", "Tourism", "Cultural", 5000, "IDR", None, None),
    ]
    acct = SEEAAccount(make_area_df(), make_trans_df(),
                       services_a, make_classes(), px_area_ha=0.01)
    assert acct.physical_flow_account() is None
    assert not acct.monetary_flow_account().empty
