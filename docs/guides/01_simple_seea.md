# Guide 1 — Getting Started: SEEA-EA from a Single LULC Raster

**Complexity:** Beginner

**Full script:** `strategicc_examples/example1_simple_seea.py`

This is the simplest possible use of STRATEGICC — no simulation at all. You have one land cover raster (e.g. this year's classified satellite image) and want to know what the landscape is worth, broken down by ecosystem service.

Use this to do a quick analysis about **snapshot valuation** of a landscape. No timeseries. No modelling. 

## What you need

- A classified LULC raster (GeoTIFF, with integer class IDs)
- A `StateClasses.csv` defining your classes
- An `EcosystemServices.csv` defining economic value per class per hectare

## Step 1 — Define your classes
User can define their land cover class using csv file or directly in python.

```python
with open("inputs/StateClasses.csv", "w") as f:
    f.write('''Name,StateLabelXId,StateLabelYId,Id,Color,Legend,Description,IsAutoName
Water_body:All,Water_body,All,1,"255,0,128,255",,,No
Mangrove:All,Mangrove,All,2,"255,0,100,0",,,No
Aquaculture:All,Aquaculture,All,3,"255,255,0,255",,,No
Cropland:All,Cropland,All,4,"255,255,255,0",,,No
''')
```

## Step 2 — Define ecosystem service values

This is Mode A/B valuation — a static value per hectare, not derived from any simulation:
User can define their land cover class monetary value or service value of each land cover value using csv file or directly in python.

```python
with open("inputs/EcosystemServices.csv", "w") as f:
    f.write('''StateClassId,ServiceName,ServiceType,ValuePerHa,Currency,PhysicalUnit,PhysicalValuePerHa
Mangrove,Carbon Sequestration,Regulating,25000000,IDR,MgC/ha,350
Mangrove,Coastal Protection,Regulating,15000000,IDR,,
Mangrove,Fishery Nursery,Provisioning,8000000,IDR,,
Aquaculture,Aquaculture Fishery,Provisioning,45000000,IDR,kg/ha,800
Cropland,Crop Provisioning,Provisioning,30000000,IDR,kg/ha,5000
''')
```

`ValuePerHa` is the price per hectare per year. If user also know the physical quantity supplied (e.g. carbon sequestered), set `PhysicalUnit` and `PhysicalValuePerHa` too — this produces a physical flow account alongside the monetary one.

## Step 3 — Compute extent + valuation directly

There's no `StrategiccEngine` here since there's no simulation — you build the area table by hand from the raster, then hand it straight to `SEEAAccount`:

```python
from strategicc.io import load_state_classes, read_lulc
from strategicc.accounting import load_ecosystem_services, SEEAAccount

classes  = load_state_classes("inputs/StateClasses.csv")
services = load_ecosystem_services("inputs/EcosystemServices.csv")

lulc_arr, px_area_ha, src_tags = read_lulc("2024.tif")

rows = []
for cid, sc in classes.items():
    area_ha = float((lulc_arr == cid).sum()) * px_area_ha
    rows.append({"year": 2024, "class_id": cid, "class_name": sc.name, "area_ha": area_ha})
area_modal_df = pd.DataFrame(rows)

acct = SEEAAccount(
    area_modal_df = area_modal_df,
    trans_df      = pd.DataFrame(),   # no transitions, single snapshot
    services      = services,
    classes       = classes,
    px_area       = px_area_ha,
)
```

`trans_df` is empty here because `SEEAAccount` was designed around the full simulation pipeline — `trans_df` normally feeds the transition matrix account, which doesn't apply to a single static map.

## Step 4 — Read the results

```python
monetary = acct.monetary_flow_account()
print(monetary)

total_value = monetary.sum(axis=1).iloc[0]
print(f"Total landscape value: {total_value:,.0f} IDR")
```

## What this doesn't cover

This guide intentionally skips the simulation engine, calibration, spatial multipliers, age tracking, and Stock & Flow — it's a pure accounting exercise on one map. If you want to project how the landscape might change and value that projected future, continue to [Guide 2](02_calibration_stsm.md).
