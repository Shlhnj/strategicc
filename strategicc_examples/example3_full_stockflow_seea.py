"""
example3_full_stockflow_seea.py

Guide 3: Full Pipeline — Age, Stock & Flow, and Dynamic Valuation.

Builds on example2 by adding carbon Stock & Flow accounting. Carbon valuation
responds to age structure, transition dynamics, and stochastic variation —
not a flat per-hectare number.

Steps:
    1. Calibrate transitions *and* a continuous age raster from history
    2. Define the carbon cycle (Stock Types, Flow Types, Flow Pathways)
    3. Configure Mode C ecosystem services (physical quantity from the sim)
    4. Run with age tracking and Stock & Flow enabled
    5. Aggregate Stock & Flow outputs
    6. Build a SEEA-EA asset account
    7. Run Mode C SEEA-EA accounting

Requires:
    - A longer historical zip (more years = better age backtracking):
      "annual_lulc_2010_2022.zip"
    - The current-year raster: "2022.tif"

See docs/guides/03_stockflow_full.md for the full walkthrough.
"""

from pathlib import Path

import pandas as pd

import strategicc.config as cfg
from strategicc import StrategiccEngine
from strategicc.io import load_state_classes
from strategicc.calibration import (
    load_lulc_timeseries,
    compute_age_raster,
    save_age_raster,
    compute_yearly_transition_counts,
    compute_transition_rates,
    save_transitions_csv,
    compute_temporal_distribution,
    save_temporal_distribution_csv,
)
from strategicc import outputs
from strategicc.stockflow import aggregate_stock_by_class, aggregate_flow_by_class, build_asset_account
from strategicc.accounting import SEEAAccount, save_all_accounts


INPUTS_DIR = Path("inputs")
HISTORICAL_ZIP = Path("annual_lulc_2010_2022.zip")
CURRENT_RASTER = Path("2022.tif")
OUT_DIR = Path("output")

GROUP_MAP = {(2, 3): "Aquaculture_expansion"}  # Mangrove(2) -> Aquaculture(3)


def step1_calibrate_transitions_and_age(classes):
    ts = load_lulc_timeseries(str(HISTORICAL_ZIP), extract_dir="extracted_hist")

    age_result = compute_age_raster(ts)  # continuous age, backtracked from the whole record
    save_age_raster(age_result, INPUTS_DIR / "age.tif")

    yearly = compute_yearly_transition_counts(ts)
    transitions_df = compute_transition_rates(yearly, classes, GROUP_MAP, min_probability=1e-5)
    save_transitions_csv(transitions_df, INPUTS_DIR / "Transitions.csv")

    temporal_df, distributions_df = compute_temporal_distribution(yearly, GROUP_MAP, min_years=3)
    save_temporal_distribution_csv(temporal_df, INPUTS_DIR / "TransitionMultipliers.csv")
    distributions_df.to_csv(INPUTS_DIR / "Distributions.csv", index=False)


def step2_define_carbon_cycle() -> None:
    """Stock types are the pools material moves between."""
    (INPUTS_DIR / "StockType.csv").write_text(
        "Name,Description\n"
        "Atmosphere,Notional carbon source/sink\n"
        "Biomass,Living mangrove carbon\n"
    )
    (INPUTS_DIR / "FlowType.csv").write_text(
        "Name,Description\n"
        "NPP,Net primary production\n"
        "Emission,Release on conversion\n"
    )
    # FlowOrder.csv matters: NPP (order=1) is computed before Emission (order=2)
    # each timestep, so emission acts on the post-growth biomass total.
    (INPUTS_DIR / "FlowOrder.csv").write_text(
        "Iteration,Timestep,FlowTypeId,Order\n,,NPP,1\n,,Emission,2\n"
    )

    # Flow pathways define how stocks connect. No TransitionGroupId -> automatic
    # (fires every timestep). TransitionGroupId set -> fires only on cells where
    # that transition occurs this timestep.
    fp_rows = [
        # NPP: automatic, age-indexed (StateAttributeTypeId="NPP"), Mangrove only
        {
            "FromStateClassId": "Mangrove:All", "FromStockTypeId": "Atmosphere",
            "ToStockTypeId": "Biomass", "StateAttributeTypeId": "NPP",
            "FlowTypeId": "NPP", "Multiplier": "1",
        },
        # Emission: triggered by Aquaculture_expansion, releases 90% of biomass
        {
            "FromStockTypeId": "Biomass", "ToStockTypeId": "Atmosphere",
            "TransitionGroupId": "Aquaculture_expansion [Type]",
            "FlowTypeId": "Emission", "Multiplier": "0.9",
        },
    ]
    pd.DataFrame(fp_rows).to_csv(INPUTS_DIR / "FlowPathways.csv", index=False)

    # NPP's flow quantity comes from an age-bracketed lookup table rather than a
    # flat rate — younger mangrove sequesters less carbon per year than mature
    # mangrove. Rates (5.1, 11.0, 18.4 Mg C/ha/yr) are from Alongi 2020
    # (https://doi.org/10.3390/jmse8100767); the package's own Stock & Flow
    # engine has been validated against the same source.
    (INPUTS_DIR / "StateAttributeValues.csv").write_text(
        "Iteration,Timestep,StratumId,SecondaryStratumId,TertiaryStratumId,StateClassId,"
        "StateAttributeTypeId,AgeMin,AgeMax,TSTGroupId,TSTMin,TSTMax,Value,DistributionType,"
        "DistributionFrequencyId,DistributionSD,DistributionMin,DistributionMax\n"
        ",,,,,,NPP,0,10,,,,5.1,,,,,\n"
        ",,,,,,NPP,11,20,,,,11.0,,,,,\n"
        ",,,,,,NPP,21,999,,,,18.4,,,,,\n"
    )


def step3_mode_c_services() -> None:
    """
    flow:NPP and stock:Biomass are genuinely different things: flow values the
    annual carbon sequestration service (a recurring rate, e.g. a carbon credit
    payment); stock values the carbon currently stored (a standing asset). In
    Mode C, ValuePerUnitArea is reinterpreted as price per physical unit.
    """
    (INPUTS_DIR / "EcosystemServices.csv").write_text(
        "StateClassId,ServiceName,ServiceType,ValuePerUnitArea,Currency,PhysicalUnit,"
        "PhysicalValuePerUnitArea,StockFlowSource\n"
        "Mangrove,Carbon Sequestration,Regulating,75000,IDR,MgC,,flow:NPP\n"
        "Mangrove,Carbon Storage,Regulating,75000,IDR,MgC,,stock:Biomass\n"
        "Mangrove,Coastal Protection,Regulating,15000000,IDR,,,\n"
        "Aquaculture,Aquaculture Fishery,Provisioning,45000000,IDR,kg/ha,800,\n"
    )


def step4_run_with_age_and_stockflow() -> StrategiccEngine:
    engine = StrategiccEngine(
        lulc_path=str(CURRENT_RASTER),
        state_classes_csv=str(INPUTS_DIR / "StateClasses.csv"),
        transitions_csv=str(INPUTS_DIR / "Transitions.csv"),
        # This example doesn't use a spatial multiplier raster, but spatial_mult_csv
        # and mult_dir are still required constructor arguments (no defaults) —
        # point them somewhere and turn the feature off explicitly.
        spatial_mult_csv=str(INPUTS_DIR / "TransitionSpatialMultipliers.csv"),
        trans_mult_csv=str(INPUTS_DIR / "TransitionMultipliers.csv"),
        ecosystem_services_csv=str(INPUTS_DIR / "EcosystemServices.csv"),
        mult_dir="spatmult_uploads/",
        out_dir=str(OUT_DIR),
        start_year=2022,
        n_timesteps=10,
        n_iterations=15,
        use_adjacency=True,
        use_spatial_mult=False,  # not used in this example
        use_trans_multiplier=True,
        use_seea=True,
        use_age=True,
        age_raster_path=str(INPUTS_DIR / "age.tif"),
        save_age_rasters=True,
        use_stockflow=True,
    )

    # Stock & Flow CSVs are configured via strategicc.config, not the constructor.
    cfg.STOCK_TYPE_CSV = str(INPUTS_DIR / "StockType.csv")
    cfg.FLOW_TYPE_CSV = str(INPUTS_DIR / "FlowType.csv")
    cfg.FLOW_ORDER_CSV = str(INPUTS_DIR / "FlowOrder.csv")
    cfg.FLOW_PATHWAYS_CSV = str(INPUTS_DIR / "FlowPathways.csv")
    cfg.STATE_ATTRIBUTE_VALUES_CSV = str(INPUTS_DIR / "StateAttributeValues.csv")

    engine.load()
    engine.run()
    return engine


def step5_aggregate_stockflow(engine: StrategiccEngine, modal_maps):
    stock_df = aggregate_stock_by_class(
        iter_dirs=engine.iter_dirs,
        stock_types=engine._stock_types,
        classes=engine.classes,
        modal_maps=modal_maps,
        start_year=engine.start_year,
        n_timesteps=engine.n_timesteps,
    )
    flow_df = aggregate_flow_by_class(engine.iter_dirs)
    return stock_df, flow_df


def step6_asset_account(engine: StrategiccEngine, stock_df, flow_df):
    asset_account = build_asset_account(
        stock_df=stock_df,
        flow_df=flow_df,
        stock_types=engine._stock_types,
        classes=engine.classes,
        start_year=engine.start_year,
        n_timesteps=engine.n_timesteps,
    )
    print(asset_account.head())
    return asset_account


def step7_mode_c_seea(engine, area_df, trans_df, area_modal_df, stock_df, flow_df):
    acct = SEEAAccount(
        area_modal_df=area_modal_df,
        trans_df=trans_df,
        services=engine.ecosystem_services,
        classes=engine.classes,
        px_area=engine.px_area,
        px_area_ha=engine.px_area_ha,
        area_df=area_df,
        stock_df=stock_df,  # enables Mode C
        flow_df=flow_df,    # enables Mode C
    )
    save_all_accounts(acct, engine.out_dir / "seea")


def main() -> None:
    INPUTS_DIR.mkdir(exist_ok=True)

    from example1_simple_seea import write_state_classes
    write_state_classes(INPUTS_DIR / "StateClasses.csv")
    classes = load_state_classes(INPUTS_DIR / "StateClasses.csv")

    for required in (HISTORICAL_ZIP, CURRENT_RASTER):
        if not required.exists():
            raise FileNotFoundError(
                f"'{required}' not found — point it at real data before running this example."
            )

    step1_calibrate_transitions_and_age(classes)
    step2_define_carbon_cycle()
    step3_mode_c_services()
    engine = step4_run_with_age_and_stockflow()

    summary_dir = engine.out_dir / "summary"
    area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)
    modal_maps = outputs.aggregate_spatial(
        iter_dirs=engine.iter_dirs, start_year=engine.start_year,
        n_timesteps=engine.n_timesteps, src_tags=engine.src_tags,
        summary_dir=summary_dir, uncertainty=True,
    )
    area_modal_df = outputs.modal_to_area_table(
        modal_maps=modal_maps, classes=engine.classes,
        px_area=engine.px_area, area_unit=engine.area_unit,
    )

    stock_df, flow_df = step5_aggregate_stockflow(engine, modal_maps)
    step6_asset_account(engine, stock_df, flow_df)
    step7_mode_c_seea(engine, area_df, trans_df, area_modal_df, stock_df, flow_df)


if __name__ == "__main__":
    main()
