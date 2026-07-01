"""
tests/test_seea_stockflow.py  —  v3.2
Unit tests for Mode C (stock_flow-linked) SEEA-EA valuation and the
stockflow aggregation module.
"""

import pytest
import numpy as np
import pandas as pd

from strategicc.accounting.csv_loader import EcosystemService, load_ecosystem_services
from strategicc.accounting.seea import SEEAAccount
from strategicc.io.csv_loader import StateClass
from strategicc.stockflow.aggregation import (
    aggregate_stock_by_class, aggregate_flow_by_class, aggregate_flow_by_type,
)


# ── EcosystemService Mode C parsing ───────────────────────────────────────────

def test_ecosystem_service_stockflow_kind_and_type():
    svc = EcosystemService(
        "Mangrove", "Carbon Seq", "Regulating", 75000, "IDR",
        "MgC", None, stockflow_source="flow:NPP",
    )
    assert svc.has_stockflow_source
    assert svc.stockflow_kind == "flow"
    assert svc.stockflow_type_name == "NPP"

def test_ecosystem_service_stock_kind():
    svc = EcosystemService(
        "Mangrove", "Carbon Storage", "Regulating", 75000, "IDR",
        "MgC", None, stockflow_source="stock:Biomass",
    )
    assert svc.stockflow_kind == "stock"
    assert svc.stockflow_type_name == "Biomass"

def test_ecosystem_service_no_stockflow_source():
    svc = EcosystemService("Mangrove", "Tourism", "Cultural", 5000, "IDR", None, None)
    assert not svc.has_stockflow_source
    assert svc.stockflow_kind is None
    assert svc.stockflow_type_name is None

def test_load_ecosystem_services_mode_c(tmp_path):
    p = tmp_path / "EcosystemServices.csv"
    p.write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,"
        "PhysicalUnit,PhysicalValuePerUnitArea,StockFlowSource\n"
        "Mangrove,Carbon Sequestration,Regulating,75000,IDR,MgC,,flow:NPP\n"
    )
    services = load_ecosystem_services(p)
    assert len(services) == 1
    assert services[0].has_stockflow_source
    assert services[0].stockflow_kind == "flow"
    assert services[0].stockflow_type_name == "NPP"
    assert services[0].physical_per_unit_area is None

def test_load_ecosystem_services_legacy_column_names_still_work(tmp_path, capsys):
    """Old ValuePerHa/PhysicalValuePerHa headers (pre-v3.3) still parse."""
    p = tmp_path / "EcosystemServices.csv"
    p.write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerHa,Currency,"
        "PhysicalUnit,PhysicalValuePerHa\n"
        "Mangrove,Carbon,Regulating,75000,IDR,MgC/ha,1300\n"
    )
    services = load_ecosystem_services(p)
    assert len(services) == 1
    assert services[0].value_per_unit_area == 75000
    assert services[0].physical_per_unit_area == 1300
    captured = capsys.readouterr()
    assert "legacy column" in captured.out

def test_load_ecosystem_services_intermediate_v33_column_names_still_work(tmp_path, capsys):
    """Short-lived v3.3 ValuePerUnit/PhysicalValuePerUnit headers still parse."""
    p = tmp_path / "EcosystemServices.csv"
    p.write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerUnit,Currency,"
        "PhysicalUnit,PhysicalValuePerUnit\n"
        "Mangrove,Carbon,Regulating,75000,IDR,MgC/ha,1300\n"
    )
    services = load_ecosystem_services(p)
    assert len(services) == 1
    assert services[0].value_per_unit_area == 75000
    assert services[0].physical_per_unit_area == 1300
    captured = capsys.readouterr()
    assert "legacy column" in captured.out

def test_load_ecosystem_services_invalid_stockflow_source_warns(tmp_path, capsys):
    p = tmp_path / "EcosystemServices.csv"
    p.write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,"
        "PhysicalUnit,PhysicalValuePerUnitArea,StockFlowSource\n"
        "Mangrove,Bad,Regulating,75000,IDR,,,invalidformat\n"
    )
    services = load_ecosystem_services(p)
    assert len(services) == 1
    assert not services[0].has_stockflow_source
    captured = capsys.readouterr()
    assert "invalid StockFlowSource" in captured.out

def test_load_ecosystem_services_mode_a_b_still_work(tmp_path):
    p = tmp_path / "EcosystemServices.csv"
    p.write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,"
        "PhysicalUnit,PhysicalValuePerUnitArea\n"
        "Mangrove,Tourism,Cultural,5000,IDR,,\n"
        "Mangrove,Carbon,Regulating,75000,IDR,MgC/ha,1300\n"
    )
    services = load_ecosystem_services(p)
    assert len(services) == 2
    assert not services[0].has_physical
    assert not services[0].has_stockflow_source
    assert services[1].has_physical
    assert services[1].physical_per_unit_area == 1300


# ── SEEAAccount Mode C valuation ──────────────────────────────────────────────

@pytest.fixture
def classes():
    return {1: StateClass(1, "Mangrove", "Mangrove:All", (255, 0, 100, 0))}

@pytest.fixture
def area_modal_df():
    return pd.DataFrame([
        {"year": 2022, "class_id": 1, "class_name": "Mangrove", "area_ha": 100.0},
        {"year": 2023, "class_id": 1, "class_name": "Mangrove", "area_ha": 95.0},
    ])

def test_mode_c_flow_monetary_value(classes, area_modal_df):
    services = [
        EcosystemService(
            "Mangrove", "Carbon Sequestration", "Regulating", 75000, "IDR",
            "MgC", None, stockflow_source="flow:NPP",
        ),
    ]
    flow_df = pd.DataFrame([
        {"year": 2022, "class_name": "Mangrove", "flow_type": "NPP", "total": 1840.0},
        {"year": 2023, "class_name": "Mangrove", "flow_type": "NPP", "total": 1748.0},
    ])
    acct = SEEAAccount(
        area_modal_df=area_modal_df, trans_df=pd.DataFrame(), services=services,
        classes=classes, px_area=1.0, flow_df=flow_df,
    )
    mon = acct.monetary_flow_account()
    assert mon.loc[2022].values[0] == pytest.approx(1840.0 * 75000)
    assert mon.loc[2023].values[0] == pytest.approx(1748.0 * 75000)

def test_mode_c_differs_from_area_based_calculation(classes, area_modal_df):
    services = [
        EcosystemService(
            "Mangrove", "Carbon Sequestration", "Regulating", 75000, "IDR",
            "MgC", None, stockflow_source="flow:NPP",
        ),
    ]
    flow_df = pd.DataFrame([
        {"year": 2022, "class_name": "Mangrove", "flow_type": "NPP", "total": 1840.0},
    ])
    acct = SEEAAccount(
        area_modal_df=area_modal_df, trans_df=pd.DataFrame(), services=services,
        classes=classes, px_area=1.0, flow_df=flow_df,
    )
    mon = acct.monetary_flow_account()
    area_based_value = 100.0 * 75000
    assert mon.loc[2022].values[0] != pytest.approx(area_based_value)

def test_mode_c_stock_kind(classes, area_modal_df):
    services = [
        EcosystemService(
            "Mangrove", "Carbon Storage", "Regulating", 75000, "IDR",
            "MgC", None, stockflow_source="stock:Biomass",
        ),
    ]
    stock_df = pd.DataFrame([
        {"year": 2022, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 18400.0},
    ])
    acct = SEEAAccount(
        area_modal_df=area_modal_df, trans_df=pd.DataFrame(), services=services,
        classes=classes, px_area=1.0, stock_df=stock_df,
    )
    mon = acct.monetary_flow_account()
    assert mon.loc[2022].values[0] == pytest.approx(18400.0 * 75000)

def test_mode_c_missing_aggregation_df_returns_zero(classes, area_modal_df):
    services = [
        EcosystemService(
            "Mangrove", "Carbon Sequestration", "Regulating", 75000, "IDR",
            "MgC", None, stockflow_source="flow:NPP",
        ),
    ]
    acct = SEEAAccount(
        area_modal_df=area_modal_df, trans_df=pd.DataFrame(), services=services,
        classes=classes, px_area=1.0,
    )
    mon = acct.monetary_flow_account()
    assert mon.loc[2022].values[0] == 0.0

def test_mode_a_b_unaffected_by_mode_c_presence(classes, area_modal_df):
    services = [
        EcosystemService("Mangrove", "Tourism", "Cultural", 5000, "IDR", None, None),
        EcosystemService(
            "Mangrove", "Carbon Sequestration", "Regulating", 75000, "IDR",
            "MgC", None, stockflow_source="flow:NPP",
        ),
    ]
    flow_df = pd.DataFrame([
        {"year": 2022, "class_name": "Mangrove", "flow_type": "NPP", "total": 1840.0},
    ])
    acct = SEEAAccount(
        area_modal_df=area_modal_df, trans_df=pd.DataFrame(), services=services,
        classes=classes, px_area=1.0, flow_df=flow_df,
    )
    mon = acct.monetary_flow_account()
    tourism_value = mon.loc[2022, ("Cultural", "Tourism")]
    carbon_value  = mon.loc[2022, ("Regulating", "Carbon Sequestration")]
    assert tourism_value == pytest.approx(100.0 * 5000)
    assert carbon_value  == pytest.approx(1840.0 * 75000)


# ── v3.3: area-unit conversion in valuation ─────────────────────────────────

def test_mode_a_valuation_correct_when_area_unit_is_km2(classes):
    """
    ValuePerUnitArea is hectare-denominated. When area_modal_df is expressed in
    km2 (AREA_UNIT="km2"), SEEAAccount must convert back to hectares using
    px_area_ha before applying the price — 1 km2 = 100 ha.
    """
    area_modal_df = pd.DataFrame([
        {"year": 2022, "class_id": 1, "class_name": "Mangrove", "area_km2": 1.0},
    ])
    services = [
        EcosystemService("Mangrove", "Tourism", "Cultural", 1000, "IDR", None, None),
    ]
    # 1 pixel = 0.01 ha = 0.0001 km2  →  px_area_ha / px_area = 100 (ha per km2)
    acct = SEEAAccount(
        area_modal_df=area_modal_df, trans_df=pd.DataFrame(), services=services,
        classes=classes, px_area=0.0001, px_area_ha=0.01,
    )
    mon = acct.monetary_flow_account()
    # 1 km2 = 100 ha  →  value = 1000 IDR/ha * 100 ha = 100,000 IDR
    assert mon.loc[2022].values[0] == pytest.approx(1000 * 100.0)

def test_mode_a_valuation_without_px_area_ha_warns_when_not_ha(classes, capsys):
    """
    Omitting px_area_ha when the area unit isn't hectares should warn,
    since valuation would otherwise silently treat km2/px as if hectares.
    """
    area_modal_df = pd.DataFrame([
        {"year": 2022, "class_id": 1, "class_name": "Mangrove", "area_km2": 1.0},
    ])
    services = [
        EcosystemService("Mangrove", "Tourism", "Cultural", 1000, "IDR", None, None),
    ]
    SEEAAccount(
        area_modal_df=area_modal_df, trans_df=pd.DataFrame(), services=services,
        classes=classes, px_area=0.0001,   # px_area_ha intentionally omitted
    )
    captured = capsys.readouterr()
    assert "px_area_ha" in captured.out


# ── Aggregation functions ──────────────────────────────────────────────────────

def test_aggregate_flow_by_type_median_across_iterations(tmp_path):
    for i in [1, 2, 3]:
        d = tmp_path / f"iter_{i:03d}"
        d.mkdir()
        pd.DataFrame([
            {"iteration": i, "year": 2022, "flow_type": "NPP",
             "from_stock": "Atmosphere", "to_stock": "Biomass",
             "transition_group": "", "total_amount": 100.0 * i},
        ]).to_csv(d / "flow_log.csv", index=False)

    result = aggregate_flow_by_type([tmp_path / f"iter_{i:03d}" for i in [1,2,3]])
    assert len(result) == 1
    assert result.iloc[0]["total"] == 200.0

def test_aggregate_flow_by_class_reads_per_class_breakdown(tmp_path):
    for i in [1, 2]:
        d = tmp_path / f"iter_{i:03d}"
        d.mkdir()
        pd.DataFrame([
            {"iteration": i, "year": 2022, "flow_type": "NPP",
             "from_stock": "Atmosphere", "to_stock": "Biomass",
             "class_name": "Mangrove", "amount": 1000.0 * i},
        ]).to_csv(d / "flow_log_by_class.csv", index=False)

    result = aggregate_flow_by_class([tmp_path / f"iter_{i:03d}" for i in [1,2]])
    assert len(result) == 1
    assert result.iloc[0]["class_name"] == "Mangrove"
    assert result.iloc[0]["total"] == 1500.0

def test_aggregate_flow_by_class_empty_when_no_files(tmp_path):
    d = tmp_path / "iter_001"
    d.mkdir()
    result = aggregate_flow_by_class([d])
    assert result.empty

def test_aggregate_stock_by_class_masks_by_modal_class(tmp_path):
    from PIL import Image
    classes = {
        1: StateClass(1, "Mangrove",    "Mangrove:All",    (255,0,100,0)),
        2: StateClass(2, "Aquaculture", "Aquaculture:All", (255,255,0,255)),
    }
    iter_dir = tmp_path / "iter_001"
    stock_dir = iter_dir / "stocks" / "Biomass"
    stock_dir.mkdir(parents=True)

    arr = np.array([[10.0, 10.0], [5.0, 5.0]], dtype=np.float32)
    Image.fromarray(arr, mode="F").save(str(stock_dir / "stock_2022.tif"))

    modal_maps = {2022: np.array([[1, 1], [2, 2]], dtype=np.uint8)}

    result = aggregate_stock_by_class(
        iter_dirs=[iter_dir], stock_types=["Biomass"], classes=classes,
        modal_maps=modal_maps, start_year=2022, n_timesteps=0,
    )
    mangrove_total = result[
        (result["class_name"]=="Mangrove") & (result["year"]==2022)
    ]["total"].values[0]
    aqua_total = result[
        (result["class_name"]=="Aquaculture") & (result["year"]==2022)
    ]["total"].values[0]
    assert mangrove_total == pytest.approx(20.0)
    assert aqua_total == pytest.approx(10.0)


# ── build_asset_account (v3.3) ─────────────────────────────────────────────────

from strategicc.stockflow.aggregation import build_asset_account


def test_asset_account_no_emission_pure_accumulation():
    classes = {1: StateClass(1, "Mangrove", "Mangrove:All", (255, 0, 100, 0))}
    stock_df = pd.DataFrame([
        {"year": 2022, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 0.0},
        {"year": 2023, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 1840.0},
    ])
    flow_df = pd.DataFrame([
        {"year": 2023, "class_name": "Mangrove", "flow_type": "NPP",
         "from_stock": "Atmosphere", "to_stock": "Biomass", "total": 1840.0},
    ])
    account = build_asset_account(
        stock_df=stock_df, flow_df=flow_df, stock_types=["Biomass"],
        classes=classes, start_year=2022, n_timesteps=1,
    )
    row = account[account["year"] == 2023].iloc[0]
    assert row["opening_balance"] == 0.0
    assert row["additions"] == 1840.0
    assert row["reductions"] == 0.0
    assert row["closing_balance_reconciled"] == 1840.0
    assert row["closing_balance_actual"] == 1840.0
    assert row["reconciliation_diff"] == pytest.approx(0.0, abs=1e-6)

def test_asset_account_with_additions_and_reductions():
    classes = {1: StateClass(1, "Mangrove", "Mangrove:All", (255, 0, 100, 0))}
    stock_df = pd.DataFrame([
        {"year": 2022, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 184.0},
        {"year": 2023, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 20.24},
    ])
    flow_df = pd.DataFrame([
        {"year": 2023, "class_name": "Mangrove", "flow_type": "NPP",
         "from_stock": "Atmosphere", "to_stock": "Biomass", "total": 18.4},
        {"year": 2023, "class_name": "Mangrove", "flow_type": "Emission",
         "from_stock": "Biomass", "to_stock": "Atmosphere", "total": 182.16},
    ])
    account = build_asset_account(
        stock_df=stock_df, flow_df=flow_df, stock_types=["Biomass"],
        classes=classes, start_year=2022, n_timesteps=1,
    )
    row = account[account["year"] == 2023].iloc[0]
    assert row["opening_balance"] == 184.0
    assert row["additions"] == 18.4
    assert row["reductions"] == 182.16
    assert row["closing_balance_reconciled"] == pytest.approx(20.24, abs=1e-6)
    assert row["reconciliation_diff"] == pytest.approx(0.0, abs=1e-6)

def test_asset_account_rollforward_opening_equals_prior_closing():
    """Year N's opening balance must equal year N-1's reconciled closing."""
    classes = {1: StateClass(1, "Mangrove", "Mangrove:All", (255, 0, 100, 0))}
    stock_df = pd.DataFrame([
        {"year": y, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 100.0 * i}
        for i, y in enumerate([2022, 2023, 2024])
    ])
    flow_df = pd.DataFrame([
        {"year": 2023, "class_name": "Mangrove", "flow_type": "NPP",
         "from_stock": "Atmosphere", "to_stock": "Biomass", "total": 100.0},
        {"year": 2024, "class_name": "Mangrove", "flow_type": "NPP",
         "from_stock": "Atmosphere", "to_stock": "Biomass", "total": 100.0},
    ])
    account = build_asset_account(
        stock_df=stock_df, flow_df=flow_df, stock_types=["Biomass"],
        classes=classes, start_year=2022, n_timesteps=2,
    )
    row_2023 = account[account["year"] == 2023].iloc[0]
    row_2024 = account[account["year"] == 2024].iloc[0]
    assert row_2024["opening_balance"] == row_2023["closing_balance_reconciled"]

def test_asset_account_surfaces_discrepancy_not_hidden():
    """
    Regression / design test: when Additions/Reductions (derived from
    flow_log medians) don't perfectly reconcile with the actual stock
    raster median, the discrepancy must be reported explicitly via
    reconciliation_diff, never silently swallowed or forced to zero.
    """
    classes = {1: StateClass(1, "Mangrove", "Mangrove:All", (255, 0, 100, 0))}
    stock_df = pd.DataFrame([
        {"year": 2022, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 0.0},
        {"year": 2023, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 100.0},
    ])
    flow_df = pd.DataFrame([
        {"year": 2023, "class_name": "Mangrove", "flow_type": "NPP",
         "from_stock": "Atmosphere", "to_stock": "Biomass", "total": 50.0},
    ])
    account = build_asset_account(
        stock_df=stock_df, flow_df=flow_df, stock_types=["Biomass"],
        classes=classes, start_year=2022, n_timesteps=1,
    )
    row = account[account["year"] == 2023].iloc[0]
    assert row["closing_balance_reconciled"] == 50.0
    assert row["closing_balance_actual"] == 100.0
    assert row["reconciliation_diff"] == 50.0

def test_asset_account_multiple_classes_independent():
    classes = {
        1: StateClass(1, "Mangrove",    "Mangrove:All",    (255,0,100,0)),
        2: StateClass(2, "Aquaculture", "Aquaculture:All", (255,255,0,255)),
    }
    stock_df = pd.DataFrame([
        {"year": 2022, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 0.0},
        {"year": 2022, "class_id": 2, "class_name": "Aquaculture",
         "stock_type": "Biomass", "total": 0.0},
        {"year": 2023, "class_id": 1, "class_name": "Mangrove",
         "stock_type": "Biomass", "total": 100.0},
        {"year": 2023, "class_id": 2, "class_name": "Aquaculture",
         "stock_type": "Biomass", "total": 0.0},
    ])
    flow_df = pd.DataFrame([
        {"year": 2023, "class_name": "Mangrove", "flow_type": "NPP",
         "from_stock": "Atmosphere", "to_stock": "Biomass", "total": 100.0},
    ])
    account = build_asset_account(
        stock_df=stock_df, flow_df=flow_df, stock_types=["Biomass"],
        classes=classes, start_year=2022, n_timesteps=1,
    )
    mangrove_row = account[
        (account["year"]==2023) & (account["class_name"]=="Mangrove")
    ].iloc[0]
    aqua_row = account[
        (account["year"]==2023) & (account["class_name"]=="Aquaculture")
    ].iloc[0]
    assert mangrove_row["additions"] == 100.0
    assert aqua_row["additions"] == 0.0   # no flow for Aquaculture this year
