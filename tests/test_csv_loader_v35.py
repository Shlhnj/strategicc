"""
tests/test_csv_loader_v35.py
-----------------------------
Unit tests for csv_loader.py's v3.5 additions: UserType/UserShare on
EcosystemServices.csv, and AssetValuationParams.csv.
"""

import pytest
import tempfile
from pathlib import Path
from strategicc.accounting.csv_loader import (
    load_ecosystem_services, load_asset_valuation_params,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_no_usertype_column_defaults_unspecified(tmp_path):
    csv = _write(tmp_path, "es.csv",
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency\n"
        "Mangrove,Ecotourism,Cultural,12500000,IDR\n")
    svcs = load_ecosystem_services(csv)
    assert len(svcs) == 1
    assert svcs[0].user_type == "Unspecified"
    assert svcs[0].has_explicit_user is False

def test_multi_row_same_service_no_usertype_is_additive_not_flagged(tmp_path, capsys):
    """Regression test for the real-world Carbon Storage = stock:AGB +
    stock:Soil pattern: two rows, same service_name, no UserType at
    all — must NOT trigger the UserShare-sum warning."""
    csv = _write(tmp_path, "es.csv",
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,PhysicalUnit,PhysicalValuePerUnitArea,StockFlowSource\n"
        "Mangrove,Carbon Storage,Regulating,238333,IDR,MgC,,stock:AGB\n"
        "Mangrove,Carbon Storage,Regulating,238333,IDR,MgC,,stock:Soil\n")
    svcs = load_ecosystem_services(csv)
    assert len(svcs) == 2
    assert all(not s.has_explicit_user for s in svcs)
    out = capsys.readouterr().out
    assert "UserShare entries sum to" not in out

def test_explicit_usertype_split_reconciles_no_warning(tmp_path, capsys):
    csv = _write(tmp_path, "es.csv",
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,UserType,UserShare\n"
        "Cropland,Crops,Provisioning,8000,IDR,Households,0.6\n"
        "Cropland,Crops,Provisioning,8000,IDR,Agriculture,0.4\n")
    svcs = load_ecosystem_services(csv)
    assert all(s.has_explicit_user for s in svcs)
    out = capsys.readouterr().out
    assert "UserShare entries sum to" not in out

def test_explicit_usertype_split_bad_shares_warns(tmp_path, capsys):
    csv = _write(tmp_path, "es.csv",
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,UserType,UserShare\n"
        "Cropland,Crops,Provisioning,8000,IDR,Households,0.6\n"
        "Cropland,Crops,Provisioning,8000,IDR,Agriculture,0.6\n")
    load_ecosystem_services(csv)
    out = capsys.readouterr().out
    assert "UserShare entries sum to 1.2000" in out

def test_load_asset_valuation_params_basic(tmp_path):
    csv = _write(tmp_path, "avp.csv",
        "StateClassId,DiscountRate,AssetLifeYears,PriceGrowthRate,ConditionProxy,ConditionReferenceLevel\n"
        "Mangrove,0.02,100,0.00,Biomass,180\n"
        "Cropland,0.02,50,0.01,,\n"
        "ALL,0.02,100,0.00,,\n")
    params = load_asset_valuation_params(csv)
    assert set(params.keys()) == {"Mangrove", "Cropland", "ALL"}
    assert params["Mangrove"].condition_proxy == "Biomass"
    assert params["Mangrove"].condition_reference_level == 180
    assert params["Cropland"].condition_proxy is None
    assert params["Cropland"].price_growth_rate == pytest.approx(0.01)

def test_load_asset_valuation_params_missing_required_skipped(tmp_path, capsys):
    csv = _write(tmp_path, "avp.csv",
        "StateClassId,DiscountRate,AssetLifeYears\n"
        "Mangrove,notanumber,100\n"
        "Cropland,0.02,50\n")
    params = load_asset_valuation_params(csv)
    assert "Mangrove" not in params
    assert "Cropland" in params
    assert "invalid DiscountRate" in capsys.readouterr().out
