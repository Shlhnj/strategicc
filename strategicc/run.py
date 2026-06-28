"""
strategicc/run.py  —  v2.1
---------------------------
Entry point.

    python -m strategicc.run

or:

    from strategicc.run import main
    main()
"""

from pathlib import Path
from strategicc import STSMEngine
from strategicc import outputs
from strategicc.accounting.seea import SEEAAccount
from strategicc.accounting import outputs as seea_outputs


def main() -> None:
    # ── 1. Build & load ───────────────────────────────────────────────────────
    engine = STSMEngine.from_config()
    engine.load()
    engine.diagnostic()

    # ── 2. Run all iterations ─────────────────────────────────────────────────
    engine.run()

    # ── 3. Aggregate tabular outputs ──────────────────────────────────────────
    summary_dir = engine.out_dir / "summary"
    print("\n[7] Building LULC summary tables...")
    area_df, trans_df = outputs.build_summary_tables(
        engine.iter_dirs, summary_dir
    )

    print("\n[8] Generating LULC summary plots...")
    outputs.plot_area_envelope(area_df, engine.classes, summary_dir)
    outputs.plot_transition_envelope(trans_df, summary_dir)

    # ── 4. Spatial aggregation (v2.1) ─────────────────────────────────────────
    print("\n[9] Aggregating spatial outputs across iterations...")
    modal_maps = outputs.aggregate_spatial(
        iter_dirs   = engine.iter_dirs,
        start_year  = engine.start_year,
        n_timesteps = engine.n_timesteps,
        src_tags    = engine.src_tags,
        summary_dir = summary_dir,
        uncertainty = True,      # set False to skip uncertainty rasters
    )

    print("\n[10] Generating spatial summary plot...")
    outputs.plot_spatial_summary(
        initial_lulc = engine._initial_lulc,
        modal_maps   = modal_maps,
        classes      = engine.classes,
        start_year   = engine.start_year,
        n_timesteps  = engine.n_timesteps,
        summary_dir  = summary_dir,
        uncertainty  = True,
    )

    # ── 5. SEEA-EA accounting ─────────────────────────────────────────────────
    if engine.use_seea and engine.ecosystem_services:
        seea_dir = engine.out_dir / "seea"
        print("\n[11] Running SEEA-EA ecosystem accounting...")

        acct = SEEAAccount(
            area_df    = area_df,
            trans_df   = trans_df,
            services   = engine.ecosystem_services,
            classes    = engine.classes,
            px_area_ha = engine.px_area_ha,
        )

        print("\n  Saving account tables...")
        seea_outputs.save_all_accounts(acct, seea_dir)

        print("\n  Generating SEEA plots...")
        seea_outputs.plot_monetary_flows(acct, engine.classes, seea_dir)
        seea_outputs.plot_value_by_service(acct, seea_dir)
        seea_outputs.plot_transition_heatmap(acct, seea_dir)

    # ── 6. Diagnostic maps (iter 1) ───────────────────────────────────────────
    print("\n[12] Generating diagnostic maps (iteration 1)...")
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
