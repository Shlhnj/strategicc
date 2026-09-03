"""
example1_simple_seea.py

Guide 1: Getting Started — SEEA-EA from a single LULC raster.

The simplest possible use of STRATEGICC: no simulation at all. You have one
classified land-cover raster (e.g. this year's satellite classification) and
want to know what the landscape is worth, broken down by ecosystem service.

Requires:
    - A classified LULC raster: "2024.tif" (integer class IDs, real GeoTIFF
      georeferencing tags — see strategicc.io.read_lulc)

See docs/guides/01_simple_seea.md for the full walkthrough.
"""

from pathlib import Path

import pandas as pd

from strategicc.io import load_state_classes, read_lulc
from strategicc.accounting import load_ecosystem_services, SEEAAccount


INPUTS_DIR = Path("inputs")
LULC_RASTER = Path("2024.tif")


def write_state_classes(path: Path) -> None:
    """StateClasses.csv — one row per land-cover class."""
    path.write_text(
        "Name,StateLabelXId,StateLabelYId,Id,Color,Legend,Description,IsAutoName\n"
        'Water_body:All,Water_body,All,1,"255,0,128,255",,,No\n'
        'Mangrove:All,Mangrove,All,2,"255,0,100,0",,,No\n'
        'Aquaculture:All,Aquaculture,All,3,"255,255,0,255",,,No\n'
        'Cropland:All,Cropland,All,4,"255,255,255,0",,,No\n'
    )


def write_ecosystem_services(path: Path) -> None:
    """
    EcosystemServices.csv — static per-hectare value (Mode A/B): a price per
    hectare, not derived from any simulation. ValuePerUnitArea is the price
    per hectare per year; set PhysicalUnit/PhysicalValuePerUnitArea too if you
    also know the physical quantity supplied (e.g. carbon sequestered) — this
    produces a physical flow account alongside the monetary one.
    """
    path.write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,"
        "PhysicalUnit,PhysicalValuePerUnitArea\n"
        "Mangrove,Carbon Sequestration,Regulating,25000000,IDR,MgC/ha,350\n"
        "Mangrove,Coastal Protection,Regulating,15000000,IDR,,\n"
        "Mangrove,Fishery Nursery,Provisioning,8000000,IDR,,\n"
        "Aquaculture,Aquaculture Fishery,Provisioning,45000000,IDR,kg/ha,800\n"
        "Cropland,Crop Provisioning,Provisioning,30000000,IDR,kg/ha,5000\n"
    )


def main() -> None:
    INPUTS_DIR.mkdir(exist_ok=True)

    state_classes_csv = INPUTS_DIR / "StateClasses.csv"
    ecosystem_services_csv = INPUTS_DIR / "EcosystemServices.csv"
    write_state_classes(state_classes_csv)
    write_ecosystem_services(ecosystem_services_csv)

    if not LULC_RASTER.exists():
        raise FileNotFoundError(
            f"'{LULC_RASTER}' not found. Point LULC_RASTER at a real "
            f"classified GeoTIFF (integer class IDs matching StateClasses.csv) "
            f"before running this example."
        )

    classes = load_state_classes(state_classes_csv)
    services = load_ecosystem_services(ecosystem_services_csv)

    # There's no StrategiccEngine here since there's no simulation — build the
    # area table by hand from the raster, then feed it straight to SEEAAccount.
    lulc_arr, px_area_ha, src_tags = read_lulc(LULC_RASTER)

    rows = []
    for cid, sc in classes.items():
        area_ha = float((lulc_arr == cid).sum()) * px_area_ha
        rows.append({"year": 2024, "class_id": cid, "class_name": sc.name, "area_ha": area_ha})
    area_modal_df = pd.DataFrame(rows)

    acct = SEEAAccount(
        area_modal_df=area_modal_df,
        trans_df=pd.DataFrame(),  # no transitions — single snapshot
        services=services,
        classes=classes,
        px_area=px_area_ha,
    )

    monetary = acct.monetary_flow_account()
    print(monetary)

    total_value = monetary.sum(axis=1).iloc[0]
    print(f"Total landscape value: {total_value:,.0f} IDR")


if __name__ == "__main__":
    main()
