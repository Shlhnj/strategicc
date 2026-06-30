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
        "StateClassId,ServiceName,ServiceType,ValuePerHa,Currency,"
        "PhysicalUnit,PhysicalValuePerHa,StockFlowSource\n"
        "Mangrove,Carbon Sequestration,Regulating,75000,IDR,MgC,,flow:NPP\n"
    )
    services = load_ecosystem_services(p)
    assert len(services) == 1
    assert services[0].has_stockflow_source
    assert services[0].stockflow_kind == "flow"
    assert services[0].stockflow_type_name == "NPP"
    assert services[0].physical_per_ha is None

def test_load_ecosystem_services_invalid_stockflow_source_warns(tmp_path, capsys):
    p = tmp_path / "EcosystemServices.csv"
    p.write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerHa,Currency,"
        "PhysicalUnit,PhysicalValuePerHa,StockFlowSource\n"
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
        "StateClassId,ServiceName,ServiceType,ValuePerHa,Currency,"
        "PhysicalUnit,PhysicalValuePerHa\n"
        "Mangrove,Tourism,Cultural,5000,IDR,,\n"
        "Mangrove,Carbon,Regulating,75000,IDR,MgC/ha,1300\n"
    )
    services = load_ecosystem_services(p)
    assert len(services) == 2
    assert not services[0].has_physical
    assert not services[0].has_stockflow_source
    assert services[1].has_physical
    assert services[1].physical_per_ha == 1300


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
