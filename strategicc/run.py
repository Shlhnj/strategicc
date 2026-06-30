"""
strategicc/run.py  —  v3.2
---------------------------
Entry point.

    python -m strategicc.run

or:

    from strategicc.run import main
    main()

Workflow
--------
1.  load()               — read rasters + CSVs (incl. Stock & Flow, v3.2)
2.  diagnostic()         — print expected transitions
3.  run()                — simulate N iterations -> per-iter TIFs + CSVs
                            (incl. stock rasters + flow logs if enabled)
4.  build_summary_tables — concatenate raw area_df + trans_df
5.  area envelope plot   — uncertainty band from raw area_df
6.  transition envelope  — uncertainty band from raw trans_df
7.  aggregate_spatial    — modal class per cell per timestep
8.  modal_to_area_table  — area_modal_df from modal maps (SEEA input)
9.  spatial_summary plot — t=0 vs mid vs final modal maps
10. Stock & Flow aggregation (v3.2) — per-class stock/flow totals, used
    for Mode C SEEA-EA valuation
11. SEEA-EA accounting   — all accounts from area_modal_df (+ stock/flow_df)
12. Diagnostic iter1 map
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
    print("\n[13] Building raw summary tables...")
    area_df, trans_df = outputs.build_summary_tables(
        engine.iter_dirs, summary_dir
    )

    print("\n[14] Generating area + transition envelope plots...")
    outputs.plot_area_envelope(area_df, engine.classes, summary_dir)
    outputs.plot_transition_envelope(trans_df, summary_dir)

    # ── 4. Spatial aggregation -> modal maps ──────────────────────────────────
    print("\n[15] Aggregating spatial outputs (modal class per cell)...")
    modal_maps = outputs.aggregate_spatial(
        iter_dirs   = engine.iter_dirs,
        start_year  = engine.start_year,
        n_timesteps = engine.n_timesteps,
        src_tags    = engine.src_tags,
        summary_dir = summary_dir,
        uncertainty = True,
    )

    # ── 5. Modal area table (SEEA input) ──────────────────────────────────────
    print("\n[16] Deriving area table from modal maps...")
    area_modal_df = outputs.modal_to_area_table(
        modal_maps = modal_maps,
        classes    = engine.classes,
        px_area    = engine.px_area,
        area_unit  = engine.area_unit,
    )
    area_modal_df.to_csv(summary_dir / "area_modal.csv", index=False)
    print(f"  area_modal.csv saved ({len(area_modal_df)} rows)")

    # ── 6. Spatial summary plot ───────────────────────────────────────────────
    print("\n[17] Generating spatial summary plot...")
    outputs.plot_spatial_summary(
        initial_lulc = engine._initial_lulc,
        modal_maps   = modal_maps,
        classes      = engine.classes,
        start_year   = engine.start_year,
        n_timesteps  = engine.n_timesteps,
        summary_dir  = summary_dir,
        uncertainty  = True,
    )

    # ── 7. Stock & Flow aggregation (v3.2) + Asset Account (v3.3) ─────────────
    stock_df = None
    flow_df  = None
    if engine.use_stockflow and engine._stock_types:
        print("\n[18] Aggregating Stock & Flow outputs by class...")
        from strategicc.stockflow.aggregation import (
            aggregate_stock_by_class, aggregate_flow_by_class, build_asset_account,
        )
        stock_df = aggregate_stock_by_class(
            iter_dirs   = engine.iter_dirs,
            stock_types = engine._stock_types,
            classes     = engine.classes,
            modal_maps  = modal_maps,
            start_year  = engine.start_year,
            n_timesteps = engine.n_timesteps,
        )
        flow_df = aggregate_flow_by_class(engine.iter_dirs)
        stock_df.to_csv(summary_dir / "stock_by_class.csv", index=False)
        flow_df.to_csv(summary_dir / "flow_by_class.csv", index=False)
        print(f"  stock_by_class.csv saved ({len(stock_df)} rows)")
        print(f"  flow_by_class.csv saved ({len(flow_df)} rows)")

        print("\n[18b] Building SEEA-EA asset account (v3.3)...")
        asset_account = build_asset_account(
            stock_df    = stock_df,
            flow_df     = flow_df,
            stock_types = engine._stock_types,
            classes     = engine.classes,
            start_year  = engine.start_year,
            n_timesteps = engine.n_timesteps,
        )
        asset_dir = engine.out_dir / "seea"
        asset_dir.mkdir(parents=True, exist_ok=True)
        asset_account.to_csv(asset_dir / "seea_asset_account.csv", index=False)
        print(f"  seea_asset_account.csv saved ({len(asset_account)} rows)")

        max_diff = asset_account["reconciliation_diff"].abs().max()
        if max_diff > 0:
            pct_of_stock = (
                max_diff / asset_account["closing_balance_actual"].abs().clip(lower=1).max()
                * 100
            )
            print(f"  Max reconciliation diff: {max_diff:.4f} "
                  f"(~{pct_of_stock:.2f}% of largest stock total) — expected "
                  f"Monte Carlo noise from median-of-sums vs sum-of-medians; "
                  f"large values may indicate a missing flow pathway.")
    else:
        print("\n[18] Stock & Flow aggregation skipped — USE_STOCKFLOW=False")

    # ── 8. SEEA-EA accounting from modal area (+ stock/flow for Mode C) ──────
    if engine.use_seea and engine.ecosystem_services:
        seea_dir = engine.out_dir / "seea"
        print("\n[19] Running SEEA-EA ecosystem accounting (modal input)...")

        acct = SEEAAccount(
            area_modal_df = area_modal_df,
            trans_df      = trans_df,
            services      = engine.ecosystem_services,
            classes       = engine.classes,
            px_area       = engine.px_area,
            area_df       = area_df,
            stock_df      = stock_df,   # v3.2 — Mode C
            flow_df       = flow_df,    # v3.2 — Mode C
        )

        print("  Saving account tables...")
        seea_outputs.save_all_accounts(acct, seea_dir)

        print("  Generating SEEA plots...")
        seea_outputs.plot_monetary_flows(acct, engine.classes, seea_dir)
        seea_outputs.plot_value_by_service(acct, seea_dir)
        seea_outputs.plot_transition_heatmap(acct, seea_dir)
    else:
        print("\n[19] SEEA-EA skipped — set USE_SEEA=True and "
              "provide EcosystemServices.csv")

    # ── 9. Diagnostic map (iter 1) ────────────────────────────────────────────
    print("\n[20] Generating diagnostic maps (iteration 1)...")
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

    print(f"\n[OK] Done.  Outputs in: {engine.out_dir.resolve()}")


if __name__ == "__main__":
    main()
