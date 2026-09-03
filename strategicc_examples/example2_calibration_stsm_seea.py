"""
example2_calibration_stsm_seea.py

Guide 2: Calibration, Spatial Simulation, and SEEA-EA.

Builds on example1 by introducing the full simulation pipeline: instead of
valuing one static map, project how the landscape might change over the next
N years and value that projected future.

Steps:
    1. Calibrate transition rates from a historical LULC time series
    2. Build a spatial multiplier raster to bias *where* transitions occur
    3. Run a spatially explicit Monte Carlo STSM
    4. Aggregate outcomes across iterations
    5. Run SEEA-EA on the simulated future

Requires:
    - A zip of yearly classified GeoTIFFs: "annual_lulc_2015_2022.zip"
      (filenames auto-detected by 4-digit year, e.g. "2015.tif", "lulc_2015.tif")
    - The current-year raster: "2022.tif"

See docs/guides/02_calibration_stsm.md for the full walkthrough.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from strategicc import StrategiccEngine, outputs
from strategicc.io import load_state_classes, read_lulc
from strategicc.calibration import (
    load_lulc_timeseries,
    compute_yearly_transition_counts,
    compute_transition_rates,
    save_transitions_csv,
    compute_temporal_distribution,
    save_temporal_distribution_csv,
)
from strategicc.accounting import SEEAAccount, save_all_accounts


INPUTS_DIR = Path("inputs")
HISTORICAL_ZIP = Path("annual_lulc_2015_2022.zip")
CURRENT_RASTER = Path("2022.tif")
OUT_DIR = Path("output")

# (from_class_id, to_class_id) -> named transition group.
# Mangrove(2) -> Aquaculture(3). Extend this for every real pathway you expect.
GROUP_MAP = {(2, 3): "Aquaculture_expansion"}


def step1_calibrate(classes) -> None:
    """Derive Transitions.csv and TransitionMultipliers.csv from history."""
    ts = load_lulc_timeseries(str(HISTORICAL_ZIP), extract_dir="extracted_hist")

    # Computed once, feeds both outputs below — guarantees Transitions.csv and
    # TransitionMultipliers.csv stay mathematically consistent with each other.
    yearly = compute_yearly_transition_counts(ts)

    transitions_df = compute_transition_rates(yearly, classes, GROUP_MAP, min_probability=1e-5)
    save_transitions_csv(transitions_df, INPUTS_DIR / "Transitions.csv")

    # Returns TWO DataFrames: temporal_df (TransitionMultipliers.csv schema)
    # and distributions_df (Distributions.csv schema) — the multiplier is a
    # named empirical distribution, not a simple Uniform(min, max).
    temporal_df, distributions_df = compute_temporal_distribution(
        yearly, GROUP_MAP, min_years=3
    )
    save_temporal_distribution_csv(temporal_df, INPUTS_DIR / "TransitionMultipliers.csv")
    distributions_df.to_csv(INPUTS_DIR / "Distributions.csv", index=False)


def step2_spatial_multiplier(classes) -> None:
    """
    Spatial multipliers bias *where* a transition is more likely (e.g. cells
    closer to existing aquaculture ponds are more likely to convert).
    STRATEGICC expects a 0-1 normalised raster (1.0 = highest suitability).
    """
    final_year_lulc, _, src_tags = read_lulc(CURRENT_RASTER)

    aqua_mask = final_year_lulc == 3
    dist = distance_transform_edt(~aqua_mask)
    dist_norm = 1.0 - (dist / dist.max())  # invert: close = high value

    mult_dir = Path("spatmult_uploads")
    mult_dir.mkdir(exist_ok=True)
    from PIL import Image
    Image.fromarray((dist_norm * 255).astype(np.uint8), mode="L").save(
        mult_dir / "aquaculture_distance.tif"
    )

    (INPUTS_DIR / "TransitionSpatialMultipliers.csv").write_text(
        "Iteration,Timestep,TransitionGroupId,TransitionMultiplierTypeId,MultiplierFileName\n"
        ",,Aquaculture_expansion [Type],,aquaculture_distance.tif\n"
    )
    return mult_dir


def step3_run(mult_dir: Path) -> StrategiccEngine:
    engine = StrategiccEngine(
        lulc_path=str(CURRENT_RASTER),
        state_classes_csv=str(INPUTS_DIR / "StateClasses.csv"),
        transitions_csv=str(INPUTS_DIR / "Transitions.csv"),
        spatial_mult_csv=str(INPUTS_DIR / "TransitionSpatialMultipliers.csv"),
        trans_mult_csv=str(INPUTS_DIR / "TransitionMultipliers.csv"),
        ecosystem_services_csv=str(INPUTS_DIR / "EcosystemServices.csv"),
        mult_dir=str(mult_dir),
        out_dir=str(OUT_DIR),
        start_year=2022,
        n_timesteps=10,
        n_iterations=20,
        use_adjacency=True,
        use_spatial_mult=True,  # turns the spatial driver on
        use_trans_multiplier=True,
        use_seea=True,
    )
    engine.load()
    engine.diagnostic()  # expected transitions per pathway — sanity check before a full run
    engine.run()
    return engine


def step4_aggregate(engine: StrategiccEngine):
    summary_dir = engine.out_dir / "summary"
    area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)
    outputs.plot_area_envelope(area_df, engine.classes, summary_dir)

    modal_maps = outputs.aggregate_spatial(
        iter_dirs=engine.iter_dirs,
        start_year=engine.start_year,
        n_timesteps=engine.n_timesteps,
        src_tags=engine.src_tags,
        summary_dir=summary_dir,
        uncertainty=True,
    )
    area_modal_df = outputs.modal_to_area_table(
        modal_maps=modal_maps,
        classes=engine.classes,
        px_area=engine.px_area,
        area_unit=engine.area_unit,
    )
    return area_df, trans_df, area_modal_df


def step5_seea(engine: StrategiccEngine, area_df, trans_df, area_modal_df) -> None:
    acct = SEEAAccount(
        area_modal_df=area_modal_df,
        trans_df=trans_df,
        services=engine.ecosystem_services,
        classes=engine.classes,
        px_area=engine.px_area,
        px_area_ha=engine.px_area_ha,
        area_df=area_df,  # raw per-iteration data, powers the uncertainty summary
    )
    save_all_accounts(acct, engine.out_dir / "seea")

    monetary = acct.monetary_flow_account()
    total_by_year = monetary.sum(axis=1)
    print(f"{engine.start_year}: {total_by_year.loc[engine.start_year]:,.0f} IDR")
    last_year = engine.start_year + engine.n_timesteps
    print(f"{last_year}: {total_by_year.loc[last_year]:,.0f} IDR")


def main() -> None:
    INPUTS_DIR.mkdir(exist_ok=True)

    # Reuse example1's StateClasses.csv / EcosystemServices.csv, or write your own here.
    from example1_simple_seea import write_state_classes, write_ecosystem_services
    write_state_classes(INPUTS_DIR / "StateClasses.csv")
    write_ecosystem_services(INPUTS_DIR / "EcosystemServices.csv")
    classes = load_state_classes(INPUTS_DIR / "StateClasses.csv")

    for required in (HISTORICAL_ZIP, CURRENT_RASTER):
        if not required.exists():
            raise FileNotFoundError(
                f"'{required}' not found — point it at real data before running this example."
            )

    step1_calibrate(classes)
    mult_dir = step2_spatial_multiplier(classes)
    engine = step3_run(mult_dir)
    area_df, trans_df, area_modal_df = step4_aggregate(engine)
    step5_seea(engine, area_df, trans_df, area_modal_df)


if __name__ == "__main__":
    main()
