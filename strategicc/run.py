"""
strategicc/run.py  —  v2.2
---------------------------
Entry point.

    python -m strategicc.run

or:

    from strategicc.run import main
    main()

Workflow
--------
1.  load()               — read rasters + CSVs
2.  diagnostic()         — print expected transitions
3.  run()                — simulate N iterations → per-iter TIFs + CSVs
4.  build_summary_tables — concatenate raw area_df + trans_df
5.  area envelope plot   — uncertainty band from raw area_df
6.  transition envelope  — uncertainty band from raw trans_df
7.  aggregate_spatial    — modal class per cell per timestep → lulc_mean_YYYY.tif
                           + optional uncertainty_YYYY.tif
8.  modal_to_area_table  — area_modal_df from modal maps (SEEA input)
9.  spatial_summary plot — t=0 vs mid vs final modal maps
10. SEEA-EA accounting   — all accounts from area_modal_df
11. Diagnostic iter1 map
"""

from pathlib import Path
from strategicc import StrategiccEngine
from strategicc import outputs
from strategicc.accounting.seea import SEEAAccount
from strategicc.accounting import outputs as seea_outputs


def main() -> None:
    # ── 1. Build & load ───────────────────────────────────────────────────────
    engine = StrategiccEngine.from_config()
    engine.load()
    engine.diagnostic()

    # ── 2. Run all iterations ─────────────────────────────────────────────────
    engine.run()

    summary_dir = engine.out_dir / "summary"

    # ── 3. Raw tabular summaries (for uncertainty band) ───────────────────────
    print("\n[7] Building raw summary tables...")
    area_df, trans_df = outputs.build_summary_tables(
        engine.iter_dirs, summary_dir
    )

    print("\n[8] Generating area + transition envelope plots...")
    outputs.plot_area_envelope(area_df, engine.classes, summary_dir)
    outputs.plot_transition_envelope(trans_df, summary_dir)

    # ── 4. Spatial aggregation → modal maps ───────────────────────────────────
    print("\n[9] Aggregating spatial outputs (modal class per cell)...")
    modal_maps = outputs.aggregate_spatial(
        iter_dirs   = engine.iter_dirs,
        start_year  = engine.start_year,
        n_timesteps = engine.n_timesteps,
        src_tags    = engine.src_tags,
        summary_dir = summary_dir,
        uncertainty = True,
    )

    # ── 5. Modal area table (SEEA input) ──────────────────────────────────────
    print("\n[10] Deriving area table from modal maps...")
    area_modal_df = outputs.modal_to_area_table(
        modal_maps = modal_maps,
        classes    = engine.classes,
        px_area    = engine.px_area,
        area_unit  = engine.area_unit,
    )
    area_modal_df.to_csv(summary_dir / "area_modal.csv", index=False)
    print(f"  area_modal.csv saved ({len(area_modal_df)} rows)")

    # ── 6. Spatial summary plot ───────────────────────────────────────────────
    print("\n[11] Generating spatial summary plot...")
    outputs.plot_spatial_summary(
        initial_lulc = engine._initial_lulc,
        modal_maps   = modal_maps,
        classes      = engine.classes,
        start_year   = engine.start_year,
        n_timesteps  = engine.n_timesteps,
        summary_dir  = summary_dir,
        uncertainty  = True,
    )

    # ── 7. SEEA-EA accounting from modal area ─────────────────────────────────
    if engine.use_seea and engine.ecosystem_services:
        seea_dir = engine.out_dir / "seea"
        print("\n[12] Running SEEA-EA ecosystem accounting (modal input)...")

        acct = SEEAAccount(
            area_modal_df = area_modal_df,   # ← modal, spatially consistent
            trans_df      = trans_df,
            services      = engine.ecosystem_services,
            classes       = engine.classes,
            px_area       = engine.px_area,
            area_df       = area_df,          # ← raw, for uncertainty summary
        )

        print("  Saving account tables...")
        seea_outputs.save_all_accounts(acct, seea_dir)

        print("  Generating SEEA plots...")
        seea_outputs.plot_monetary_flows(acct, engine.classes, seea_dir)
        seea_outputs.plot_value_by_service(acct, seea_dir)
        seea_outputs.plot_transition_heatmap(acct, seea_dir)
    else:
        print("\n[12] SEEA-EA skipped — set USE_SEEA=True and "
              "provide EcosystemServices.csv")

    # ── 8. Diagnostic map (iter 1) ────────────────────────────────────────────
    print("\n[13] Generating diagnostic maps (iteration 1)...")
    from strategicc.io.raster import read_lulc
    maps_iter1 = []
    for t in range(engine.n_timesteps + 1):
        tif = engine.iter_dirs[0] / f"lulc_{engine.start_year + t}.tif"
        if tif.exists():
            arr, *_ = read_lulc(tif)
            maps_iter1.append(arr)

    if maps_iter1:
        diag_dir = summary_dir / "diagnostic_iter1"
        diag_dir.mkdir(exist_ok=True)
        outputs.plot_lulc_maps(
            maps_iter1, engine.classes, engine.start_year, diag_dir
        )

    print(f"\n✓ Done.  Outputs in: {engine.out_dir.resolve()}")


if __name__ == "__main__":
    main()
