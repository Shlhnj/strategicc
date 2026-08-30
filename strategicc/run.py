"""
strategicc/run.py  —  v3.21
----------------------------
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

seea_only() (v3.21)
--------------------
Recomputes SEEA-EA accounting (steps 10-11 above) from an already-completed
run, without re-simulating. See its own docstring below — this is for the
common case of rerunning accounting after fixing an input CSV (e.g. a
StateAttributeValues correction, or adding AssetValuationParams) without
paying for a full re-simulation.
"""

from pathlib import Path
from strategicc import StrategiccEngine
from strategicc import config
from strategicc import outputs
from strategicc.accounting.seea import SEEAAccount
from strategicc.accounting import outputs as seea_outputs


class RunNotFoundError(Exception):
    """
    Raised by seea_only() (v3.21) when no completed iteration output can
    be found under engine.out_dir. Distinguishes "wrong OUT_DIR / run
    never finished" from a bare AssertionError, and names the exact
    directory that was checked so the fix is obvious from the message
    alone.
    """


def _run_stockflow_and_seea(
    engine,
    area_df,
    trans_df,
    modal_maps,
    area_modal_df,
    summary_dir: Path,
    include_stock_accounts: bool = True,
    generate_plots: bool = True,
) -> "SEEAAccount | None":
    """
    Shared by main() and seea_only() (v3.21) — Stock & Flow aggregation,
    the physical asset account / Table 13.3 carbon stock account, and
    SEEA-EA ecosystem accounting, all from already-simulated output.
    Extracted out of main() so seea_only() doesn't duplicate this logic
    (and so a future fix here doesn't need to be made in two places).

    include_stock_accounts / generate_plots default True to match
    main()'s own always-on behavior — main() calls this with its
    defaults unchanged; seea_only() exposes both as its own parameters
    so a caller iterating quickly on input CSVs can skip the slower
    parts.

    Returns the built SEEAAccount, or None if SEEA didn't run (USE_SEEA
    False, or no ecosystem services loaded).
    """
    stock_df = flow_df = None
    if engine.use_stockflow and engine._stock_types:
        print("\n[18] Aggregating Stock & Flow outputs by class...")
        from strategicc.stockflow.aggregation import (
            aggregate_stock_by_class, aggregate_flow_by_class,
            build_asset_account, stock_account_seea,
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

        # v3.21 — promoted from something a caller had to remember to
        # paste into their own script (see the manual BaU notebook
        # pattern) to a permanent, unconditional check here. A stock
        # account that's silently all-zero is exactly the kind of thing
        # that otherwise produces plausible-looking but wrong output.
        if (stock_df["total"] == 0).all():
            print("\n  [WARNING] Every stock total in stock_df is 0 — any "
                  "ecosystem service priced from a stock "
                  "(StockFlowSource=stock:...) will value at 0 in the "
                  "accounts below. Check init_stocks / StateAttributeValues "
                  "before trusting the carbon figures.\n")

        if include_stock_accounts:
            print("\n[18b] Building SEEA-EA asset account (v3.3)...")
            asset_account = build_asset_account(
                stock_df    = stock_df,
                flow_df     = flow_df,
                stock_types = engine._stock_types,
                classes     = engine.classes,
                start_year  = engine.start_year,
                n_timesteps = engine.n_timesteps,
            )
            seea_dir = engine.out_dir / "seea"
            csv_dir  = seea_dir / "csv"    # v3.20 — csv/xlsx now in separate folders
            xlsx_dir = seea_dir / "xlsx"
            seea_outputs.save_asset_account(asset_account, csv_dir,  write_csv=True,  write_xlsx=False)
            seea_outputs.save_asset_account(asset_account, xlsx_dir, write_csv=False, write_xlsx=True)

            print("\n[18c] Building SEEA EA Table 13.3 carbon stock account (v3.20)...")
            carbon_stock_account = stock_account_seea(
                stock_df    = stock_df,
                flow_df     = flow_df,
                stock_types = engine._stock_types,
                classes     = engine.classes,
                start_year  = engine.start_year,
                n_timesteps = engine.n_timesteps,
            )
            seea_outputs.save_carbon_stock_account(carbon_stock_account, csv_dir,  write_csv=True,  write_xlsx=False)
            seea_outputs.save_carbon_stock_account(carbon_stock_account, xlsx_dir, write_csv=False, write_xlsx=True)

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

    acct = None
    if engine.use_seea and engine.ecosystem_services:
        seea_dir = engine.out_dir / "seea"
        csv_dir  = seea_dir / "csv"    # v3.20 — csv/xlsx now in separate folders
        xlsx_dir = seea_dir / "xlsx"
        print("\n[19] Running SEEA-EA ecosystem accounting (modal input)...")

        acct = SEEAAccount(
            area_modal_df = area_modal_df,
            trans_df      = trans_df,
            services      = engine.ecosystem_services,
            classes       = engine.classes,
            px_area       = engine.px_area,
            px_area_ha    = engine.px_area_ha,   # v3.3 — required for correct valuation when AREA_UNIT != "ha"
            area_df       = area_df,
            stock_df      = stock_df,   # v3.2 — Mode C
            flow_df       = flow_df,    # v3.2 — Mode C
            asset_valuation_params = engine.asset_valuation_params,  # v3.21
        )

        print("  Saving account tables (csv)...")
        seea_outputs.save_all_accounts(acct, csv_dir)

        print("  Saving account tables (xlsx)...")
        seea_outputs.save_all_accounts_xlsx(acct, xlsx_dir)

        if generate_plots:
            print("  Generating SEEA plots...")
            seea_outputs.plot_monetary_flows(acct, engine.classes, seea_dir)
            seea_outputs.plot_value_by_service(acct, seea_dir)
            seea_outputs.plot_transition_heatmap(acct, seea_dir)
    else:
        print("\n[19] SEEA-EA skipped — set USE_SEEA=True and "
              "provide EcosystemServices.csv")

    return acct


def seea_only(
    manifest_path: str | Path | None = None,
    engine:        "StrategiccEngine | None" = None,
    include_stock_accounts: bool = True,
    generate_plots:         bool = True,
    zip_output:             bool = False,
) -> "SEEAAccount | None":
    """
    Recompute SEEA-EA accounting from an already-completed run, without
    re-simulating (v3.21).

    Replaces the manual rebuild-iter_dirs / aggregate / build-SEEAAccount
    pipeline that was previously copy-pasted per notebook (see e.g. the
    BaU rerun script) with a single call:

        from strategicc.run import seea_only
        seea_only(manifest_path="RunManifest_BaU.txt")

    or, if a manifest is already loaded and an engine already built in
    the current session (e.g. earlier in the same notebook):

        seea_only(engine=engine)

    Parameters
    ----------
    manifest_path : path to a RunManifest.txt to load. Ignored if
                    engine= is given. If both are None, uses whatever
                    manifest config.load_manifest() already loaded in
                    this session (same as main()).
    engine        : an already-loaded StrategiccEngine (e.g. reused from
                    an earlier cell). If given, manifest_path is not
                    used — pass one or the other, not both.
    include_stock_accounts : (default True, matching main()) also build
                    and save the v3.3 flat asset account and v3.20
                    Table 13.3 carbon stock account, not just the
                    ecosystem-service SEEA tables. Set False to skip
                    these for a faster rerun while iterating on inputs.
    generate_plots : (default True, matching main()) also generate the
                    monetary-flow / value-by-service / transition-heatmap
                    plots. Set False to skip for a faster rerun.
    zip_output    : (default False) zip engine.out_dir/"seea" to
                    <out_dir parent>/<out_dir name>_seea.zip when done,
                    matching the manual script's shutil.make_archive
                    step.

    Returns
    -------
    The built SEEAAccount, or None if SEEA didn't actually run this call
    (USE_SEEA=False, or no ecosystem services loaded) — check engine
    print output in that case, same diagnostics as main() step [19].

    Raises
    ------
    RunNotFoundError : if no iter_* folders are found under
                    engine.out_dir — wrong OUT_DIR in the manifest, or
                    the run never completed.
    """
    if engine is None:
        if manifest_path is not None:
            config.load_manifest(manifest_path)
        engine = StrategiccEngine.from_config()
        engine.load()

    if not engine.iter_dirs:
        engine.iter_dirs = sorted(engine.out_dir.glob("iter_*"))
    if not engine.iter_dirs:
        raise RunNotFoundError(
            f"No iter_* folders found under {engine.out_dir.resolve()} — "
            f"check OUT_DIR in the manifest, or that the run actually "
            f"finished before this was called."
        )
    print(f"Found {len(engine.iter_dirs)} completed iteration dir(s) "
          f"under {engine.out_dir.resolve()}")

    summary_dir = engine.out_dir / "summary"
    area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)

    modal_maps = outputs.aggregate_spatial(
        iter_dirs   = engine.iter_dirs,
        start_year  = engine.start_year,
        n_timesteps = engine.n_timesteps,
        src_tags    = engine.src_tags,
        summary_dir = summary_dir,
        uncertainty = True,
    )
    area_modal_df = outputs.modal_to_area_table(
        modal_maps = modal_maps,
        classes    = engine.classes,
        px_area    = engine.px_area,
        area_unit  = engine.area_unit,
    )
    area_modal_df.to_csv(summary_dir / "area_modal.csv", index=False)

    acct = _run_stockflow_and_seea(
        engine, area_df, trans_df, modal_maps, area_modal_df, summary_dir,
        include_stock_accounts = include_stock_accounts,
        generate_plots         = generate_plots,
    )

    if zip_output:
        import shutil
        seea_dir = engine.out_dir / "seea"
        archive_base = engine.out_dir.parent / f"{engine.out_dir.name}_seea"
        archive_path = shutil.make_archive(str(archive_base), "zip", seea_dir)
        print(f"  Zipped: {archive_path}")

    print(f"\n[OK] seea_only() done. SEEA output in: "
          f"{(engine.out_dir / 'seea').resolve()}")
    return acct


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

    # ── 7-8. Stock & Flow aggregation + Asset Account + SEEA-EA accounting ────
    # (v3.21 — factored into _run_stockflow_and_seea(), shared with seea_only())
    _run_stockflow_and_seea(
        engine, area_df, trans_df, modal_maps, area_modal_df, summary_dir,
    )

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
