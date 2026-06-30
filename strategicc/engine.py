"""
strategicc/engine.py  —  v2.5
------------------------------
StrategiccEngine: main simulation class with multi-iteration support.

Usage
-----
    from strategicc import StrategiccEngine
    engine = StrategiccEngine.from_config()
    engine.load()
    engine.run()    # runs all iterations, writes per-iter outputs to disk
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc import config
from strategicc.io.raster     import read_lulc, save_tifs, get_pixel_area, UNIT_LABELS, resolve_mult_dir
from strategicc.io.csv_loader import (
    load_state_classes,
    load_transitions,
    load_spatial_mult_index,
    load_transition_multipliers,
    load_initial_age_rules,
    load_transition_size_rules,
    group_size_bins,
    load_transition_targets,
    load_transition_adjacency_setting,
    load_transition_adjacency_multipliers,
    build_adjacency_strength_map,
)
from strategicc.core.transitions import build_transition_index, TransitionRecord
from strategicc.core.adjacency   import compute_neighbor_fractions
from strategicc.core.spatial     import load_spatial_multipliers, get_multiplier
from strategicc.core.multipliers import (
    sample_transition_multipliers,
    describe_multiplier_rules,
)
from strategicc.core.age import (
    build_initial_age_from_raster,
    build_initial_age_from_rules,
    update_age,
    age_gate_mask,
    save_age_tif,
)
from strategicc.core.patches import grow_patches_for_group
from strategicc.core.targets import (
    resolve_targets_per_timestep,
    scale_probability_to_target,
    target_to_patch_budget,
)
from strategicc.stockflow import (
    load_stock_types, load_flow_types, load_flow_order,
    load_stock_groups, load_stock_group_membership,
    load_state_attribute_types, load_state_attribute_values,
    load_flow_pathways, load_flow_multipliers,
    load_initial_stock_links,
    init_stocks, run_flows_for_timestep, sample_flow_multipliers,
)
from strategicc.accounting.csv_loader import load_ecosystem_services, EcosystemService


class StrategiccEngine:
    """
    STRATEGICC engine  v2.0

    Attributes set after load()
    ---------------------------
    classes          : dict[int, StateClass]
    trans_index      : TransitionIndex
    spatial_mults    : dict[str, np.ndarray]
    trans_mult_rules : list[TransitionMultiplierRule]
    ecosystem_services: list[EcosystemService]   ← v2.0

    Attributes set after run()
    --------------------------
    iter_dirs : list[Path]  — one output folder per completed iteration
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        lulc_path:            str | Path,
        state_classes_csv:    str | Path,
        transitions_csv:      str | Path,
        spatial_mult_csv:     str | Path,
        trans_mult_csv:       str | Path,
        ecosystem_services_csv: str | Path,
        mult_dir:             str | Path,
        out_dir:              str | Path,
        start_year:           int  = 2022,
        n_timesteps:          int  = 10,
        n_iterations:         int  = 10,
        rng_seed:             int  = 42,
        use_adjacency:        bool = True,
        use_spatial_mult:     bool = True,
        use_trans_multiplier: bool = True,
        use_seea:             bool = True,
        area_unit:            str  = "ha",
        use_age:              bool = False,   # v2.3
        age_raster_path       = None,         # v2.3 Path | None
        age_initial_csv       = None,         # v2.3 Path | None
        save_age_rasters:     bool = False,   # v2.3
        transition_size_csv   = None,         # v2.5 Path | None
        transition_targets_csv = None,        # v3.1 Path | None
        transition_adjacency_setting_csv = None,    # v3.1 Path | None
        transition_adjacency_mult_csv    = None,    # v3.1 Path | None
        use_stockflow:        bool = False,   # v3.2
    ) -> None:
        self.lulc_path               = Path(lulc_path)
        self.state_classes_csv       = Path(state_classes_csv)
        self.transitions_csv         = Path(transitions_csv)
        self.spatial_mult_csv        = Path(spatial_mult_csv)
        self.trans_mult_csv          = Path(trans_mult_csv)
        self.ecosystem_services_csv  = Path(ecosystem_services_csv)
        self.mult_dir                = Path(mult_dir)
        self.out_dir                 = Path(out_dir)
        self.start_year              = start_year
        self.n_timesteps             = n_timesteps
        self.n_iterations            = n_iterations
        self.rng_seed                = rng_seed
        self.use_adjacency           = use_adjacency
        self.use_spatial_mult        = use_spatial_mult
        self.use_trans_multiplier    = use_trans_multiplier
        self.use_seea                = use_seea
        self.area_unit               = area_unit   # v2.2
        self.use_age                 = use_age              # v2.3
        self.age_raster_path         = Path(age_raster_path) if age_raster_path else None
        self.age_initial_csv         = Path(age_initial_csv) if age_initial_csv else None
        self.save_age_rasters        = save_age_rasters     # v2.3
        self.transition_size_csv     = Path(transition_size_csv) if transition_size_csv else None  # v2.5
        self.transition_targets_csv  = Path(transition_targets_csv) if transition_targets_csv else None  # v3.1
        self.transition_adjacency_setting_csv = (
            Path(transition_adjacency_setting_csv) if transition_adjacency_setting_csv else None
        )  # v3.1
        self.transition_adjacency_mult_csv = (
            Path(transition_adjacency_mult_csv) if transition_adjacency_mult_csv else None
        )  # v3.1
        self.use_stockflow = use_stockflow   # v3.2

        # Populated by load() — Stock & Flow (v3.2)
        self._stock_types:       list  = []
        self._flow_pathways:     list  = []
        self._flow_order:        dict  = {}
        self._flow_mult_rules:   list  = []
        self._state_attr_rules:  list  = []
        self._initial_stock_links: dict = {}

        # Populated by load()
        self.classes:             dict  = {}
        self.trans_index:         dict  = {}
        self.spatial_mults:       dict  = {}
        self.trans_mult_rules:    list  = []
        self.ecosystem_services:  list  = []
        self.src_tags:            dict  = {}
        self.px_area_ha:          float = 1.0
        self.px_area:             float = 1.0   # v2.2 — in chosen unit
        self._initial_lulc:       np.ndarray | None = None
        self._initial_age:        np.ndarray | None = None   # v2.3
        self._age_rules:          list               = []    # v2.3
        self._size_bins:          dict               = {}    # v2.5 — {group: bins}
        self._target_rules:       list               = []    # v3.1 — raw rules
        self._targets_by_timestep: dict              = {}    # v3.1 — {t: {group: amount}}
        self._adjacency_groups:    set                = set() # v3.1 — groups using CSV-driven adjacency
        self._adjacency_strength_map: dict            = {}    # v3.1 — {group: strength}



        # Populated by run()
        self.iter_dirs: list[Path] = []

    @classmethod
    def from_config(cls) -> "StrategiccEngine":
        """Construct an engine using all paths/settings from strategicc/config.py."""
        return cls(
            lulc_path              = config.LULC_PATH,
            state_classes_csv      = config.STATE_CLASSES_CSV,
            transitions_csv        = config.TRANSITIONS_CSV,
            spatial_mult_csv       = config.SPATIAL_MULT_CSV,
            trans_mult_csv         = config.TRANSITION_MULT_CSV,
            ecosystem_services_csv = config.ECOSYSTEM_SERVICES_CSV,
            mult_dir               = config.MULT_DIR,
            out_dir                = config.OUT_DIR,
            start_year             = config.START_YEAR,
            n_timesteps            = config.N_TIMESTEPS,
            n_iterations           = config.N_ITERATIONS,
            rng_seed               = config.RNG_SEED,
            use_adjacency          = config.USE_ADJACENCY,
            use_spatial_mult       = config.USE_SPATIAL_MULT,
            use_trans_multiplier   = config.USE_TRANS_MULTIPLIER,
            use_seea               = config.USE_SEEA,
            area_unit              = config.AREA_UNIT,
            use_age                = config.USE_AGE,
            age_raster_path        = config.AGE_RASTER_PATH,
            age_initial_csv        = config.AGE_INITIAL_CSV,
            save_age_rasters       = config.SAVE_AGE_RASTERS,
            transition_size_csv    = config.TRANSITION_SIZE_CSV,
            transition_targets_csv = config.TRANSITION_TARGETS_CSV,
            transition_adjacency_setting_csv = config.TRANSITION_ADJACENCY_SETTING_CSV,
            transition_adjacency_mult_csv    = config.TRANSITION_ADJACENCY_MULT_CSV,
            use_stockflow           = config.USE_STOCKFLOW,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all inputs (raster + CSVs). Call before run()."""

        # ── Fetch initial state class from historical LULC zip (v3.4) ────
        if config.FETCH_INITIAL_SC_FROM_ZIP:
            print("\n[0] Fetching initial state class from historical LULC zip...")
            try:
                from strategicc.calibration import extract_initial_state_class
            except ImportError as e:
                raise ImportError(
                    "FETCH_INITIAL_SC_FROM_ZIP=True requires the optional "
                    "rasterio dependency. Install with: pip install rasterio"
                ) from e

            extract_dir = Path("inputs") / "lulc_annual"
            self.lulc_path = extract_initial_state_class(
                zip_path    = config.LULC_ZIP_PATH,
                year        = config.INITIAL_SC_YEAR,
                extract_dir = extract_dir,
            )

        print("\n[1] Reading LULC raster...")
        lulc, self.px_area_ha, self.src_tags = read_lulc(self.lulc_path)
        self._initial_lulc = lulc
        self.px_area = get_pixel_area(self.px_area_ha, self.area_unit)
        unit_label   = UNIT_LABELS[self.area_unit]
        print(f"  Shape: {lulc.shape}  |  "
              f"px_area: {self.px_area_ha:.6f} ha  "
              f"= {self.px_area:.6f} {unit_label}  "
              f"(AREA_UNIT='{self.area_unit}')")

        print("\n[2] Loading state classes...")
        self.classes = load_state_classes(self.state_classes_csv)
        for cid, sc in self.classes.items():
            print(f"  {cid:2d}  {sc.name}")

        print("\n[3] Loading transition rules...")
        rules = load_transitions(self.transitions_csv)
        self._trans_rules  = rules   # v2.3 — raw rules kept for age gate lookup
        self.trans_index   = build_transition_index(rules, self.classes)
        n_rules = sum(len(v) for v in self.trans_index.values())
        print(f"  {n_rules} transition pathways loaded")

        print("\n[4] Loading spatial multipliers...")
        if self.use_spatial_mult:
            entries = load_spatial_mult_index(self.spatial_mult_csv)
            resolved_mult_dir = resolve_mult_dir(self.mult_dir)   # v3.4 — zip support
            self.spatial_mults = load_spatial_multipliers(
                entries, resolved_mult_dir, lulc.shape
            )
        else:
            print("  [Skipped — USE_SPATIAL_MULT=False]")

        print("\n[5] Loading transition multiplier rules...")
        if self.use_trans_multiplier and self.trans_mult_csv.exists():
            self.trans_mult_rules = load_transition_multipliers(self.trans_mult_csv)
            describe_multiplier_rules(self.trans_mult_rules)
        elif self.use_trans_multiplier:
            print(f"  [Warning] {self.trans_mult_csv} not found — "
                  "USE_TRANS_MULTIPLIER set to False")
            self.use_trans_multiplier = False
        else:
            print("  [Skipped — USE_TRANS_MULTIPLIER=False]")

        print("\n[6] Loading ecosystem services (SEEA-EA)...")
        if self.use_seea and self.ecosystem_services_csv.exists():
            self.ecosystem_services = load_ecosystem_services(
                self.ecosystem_services_csv
            )
        elif self.use_seea:
            print(f"  [Warning] {self.ecosystem_services_csv} not found — "
                  "USE_SEEA set to False")
            self.use_seea = False
        else:
            print("  [Skipped — USE_SEEA=False]")

        print("\n[7] Setting up age tracking...")
        if self.use_age:
            rng_age = np.random.default_rng(self.rng_seed)
            if self.age_raster_path and self.age_raster_path.exists():
                print(f"  Loading age raster: {self.age_raster_path}")
                self._initial_age = build_initial_age_from_raster(
                    str(self.age_raster_path)
                )
            elif self.age_initial_csv and self.age_initial_csv.exists():
                print(f"  Building age map from assumptions: {self.age_initial_csv}")
                self._age_rules = load_initial_age_rules(self.age_initial_csv)
                self._initial_age = build_initial_age_from_rules(
                    self._initial_lulc, self.classes, self._age_rules, rng_age
                )
            else:
                print("  [Warning] No age raster or InitialAge.csv found — "
                      "using age=0 for all cells")
                self._initial_age = np.zeros(
                    self._initial_lulc.shape, dtype=np.uint16
                )
            print(f"  Age map ready: mean={self._initial_age.mean():.1f}  "
                  f"max={self._initial_age.max()}")
        else:
            print("  [Skipped — USE_AGE=False]")

        print("\n[8] Loading transition size distribution...")
        if self.transition_size_csv and self.transition_size_csv.exists():
            size_rules    = load_transition_size_rules(self.transition_size_csv)
            self._size_bins = group_size_bins(size_rules)
            if self._size_bins:
                print(f"  Patch-growing enabled for: {list(self._size_bins.keys())}")
        else:
            print("  [Skipped — no TransitionSizeDistribution.csv found; "
                  "all groups use independent-cell firing]")

        print("\n[9] Loading transition targets...")
        if self.transition_targets_csv and self.transition_targets_csv.exists():
            self._target_rules = load_transition_targets(self.transition_targets_csv)
            self._targets_by_timestep = resolve_targets_per_timestep(
                self._target_rules, self.n_timesteps
            )
            active_groups = {
                g for t_targets in self._targets_by_timestep.values()
                for g, amt in t_targets.items() if amt is not None
            }
            if active_groups:
                print(f"  Targets active for: {sorted(active_groups)}")
        else:
            print("  [Skipped — no TransitionTargets.csv found; "
                  "all groups use probability-only firing]")

        print("\n[10] Loading transition adjacency (CSV-driven)...")
        if (self.transition_adjacency_setting_csv
                and self.transition_adjacency_setting_csv.exists()
                and self.transition_adjacency_mult_csv
                and self.transition_adjacency_mult_csv.exists()):
            setting_rules = load_transition_adjacency_setting(
                self.transition_adjacency_setting_csv
            )
            mult_rules = load_transition_adjacency_multipliers(
                self.transition_adjacency_mult_csv
            )
            self._adjacency_groups = {r.group for r in setting_rules}
            self._adjacency_strength_map = build_adjacency_strength_map(mult_rules)

            # Sanity check: warn about groups in Setting but missing a
            # usable (blank-AttributeValue) strength in Multipliers
            missing_strength = self._adjacency_groups - set(self._adjacency_strength_map.keys())
            if missing_strength:
                print(f"  [Warning] {len(missing_strength)} group(s) in "
                      f"TransitionAdjacencySetting.csv have no usable flat "
                      f"strength in TransitionAdjacencyMultipliers.csv "
                      f"(only attribute-based rows found) — these will "
                      f"fall back to the global ADJACENCY_STRENGTH: "
                      f"{sorted(missing_strength)}")

            if self._adjacency_groups:
                print(f"  CSV-driven adjacency active for: "
                      f"{sorted(self._adjacency_groups)}")
        else:
            print("  [Skipped — no TransitionAdjacencySetting.csv / "
                  "TransitionAdjacencyMultipliers.csv pair found; "
                  "all groups use the global ADJACENCY_STRENGTH / "
                  "STRICT_EXPANSION_GROUPS scalar fallback]")

        print("\n[11] Loading Stock & Flow inputs...")
        if self.use_stockflow:
            required = [
                config.STOCK_TYPE_CSV, config.FLOW_TYPE_CSV,
                config.FLOW_ORDER_CSV, config.FLOW_PATHWAYS_CSV,
                config.STATE_ATTRIBUTE_VALUES_CSV,
            ]
            missing_files = [p for p in required if not p.exists()]
            if missing_files:
                print(f"  [Warning] USE_STOCKFLOW=True but missing required "
                      f"file(s): {[str(p) for p in missing_files]} — "
                      f"Stock & Flow disabled for this run.")
                self.use_stockflow = False
            else:
                self._stock_types      = load_stock_types(config.STOCK_TYPE_CSV)
                load_flow_types(config.FLOW_TYPE_CSV)   # validated, not retained
                self._flow_order       = load_flow_order(config.FLOW_ORDER_CSV)
                self._flow_pathways    = load_flow_pathways(config.FLOW_PATHWAYS_CSV)
                self._state_attr_rules = load_state_attribute_values(
                    config.STATE_ATTRIBUTE_VALUES_CSV
                )
                if config.FLOW_MULTIPLIER_CSV.exists():
                    self._flow_mult_rules = load_flow_multipliers(
                        config.FLOW_MULTIPLIER_CSV
                    )
                if config.INITIAL_STOCK_NON_SPATIAL_CSV.exists():
                    self._initial_stock_links = load_initial_stock_links(
                        config.INITIAL_STOCK_NON_SPATIAL_CSV
                    )
                print(f"  Stock & Flow active: {len(self._stock_types)} "
                      f"stock type(s), {len(self._flow_pathways)} pathway(s)")
        else:
            print("  [Skipped — USE_STOCKFLOW=False]")

    def run(self) -> None:
        """
        Run all iterations. Each iteration is saved to its own subfolder.
        Populates self.iter_dirs.
        """
        if self._initial_lulc is None:
            raise RuntimeError("Call load() before run()")

        flags = (
            f"adjacency={'ON' if self.use_adjacency else 'OFF'}  "
            f"spatial_mult={'ON' if self.use_spatial_mult else 'OFF'}  "
            f"trans_mult={'ON' if self.use_trans_multiplier else 'OFF'}  "
            f"age={'ON' if self.use_age else 'OFF'}  "
            f"size_dist={'ON' if self._size_bins else 'OFF'}  "
            f"targets={'ON' if self._target_rules else 'OFF'}  "
            f"stockflow={'ON' if self.use_stockflow else 'OFF'}"
        )
        print(f"\n[12] Running {self.n_iterations} iteration(s)  ({flags})")

        self.iter_dirs = []

        for i in range(self.n_iterations):
            iter_seed = self.rng_seed + i
            iter_dir  = self.out_dir / f"iter_{i + 1:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  ── Iteration {i + 1}/{self.n_iterations}  "
                  f"(seed={iter_seed}) ──")

            maps, transitions, age_maps, stock_maps, flow_records = self._run_single_iteration(iter_seed)

            # Save LULC TIFs (v3.2: gated by RASTER_OUTPUT_SC / *_TIMESTEPS)
            if config.RASTER_OUTPUT_SC:
                stride = max(1, config.RASTER_OUTPUT_SC_TIMESTEPS)
                if stride == 1:
                    save_tifs(maps, self.start_year, self.src_tags, iter_dir)
                else:
                    from strategicc.io.raster import _TAG_TIE_POINT, _TAG_PIXEL_SCALE
                    from PIL import Image
                    iter_dir.mkdir(parents=True, exist_ok=True)
                    keep_tags = {k: self.src_tags[k]
                                 for k in (_TAG_TIE_POINT, _TAG_PIXEL_SCALE, 34735, 34736, 34737)
                                 if k in self.src_tags}
                    save_kwargs = {"compression": "lzw"}
                    if keep_tags:
                        save_kwargs["tiffinfo"] = keep_tags
                    for t, m in enumerate(maps):
                        if t % stride != 0 and t != len(maps) - 1:
                            continue
                        year = self.start_year + t
                        Image.fromarray(m.astype(np.uint8), mode="L").save(
                            str(iter_dir / f"lulc_{year}.tif"), **save_kwargs
                        )

            # Save age TIFs (v2.3, gated by RASTER_OUTPUT_AGE in v3.2)
            if self.use_age and self.save_age_rasters and config.RASTER_OUTPUT_AGE:
                age_dir = iter_dir / "age"
                stride = max(1, config.RASTER_OUTPUT_AGE_TIMESTEPS)
                for t, age_arr in enumerate(age_maps):
                    if t % stride != 0 and t != len(age_maps) - 1:
                        continue
                    save_age_tif(
                        age_arr, self.start_year + t,
                        age_dir, self.src_tags
                    )

            # Save per-iteration tables (v3.2: gated by SUMMARY_OUTPUT_SC/TR)
            if config.SUMMARY_OUTPUT_SC:
                self._save_area_table(maps, iter_dir, iteration=i + 1)
            if config.SUMMARY_OUTPUT_TR:
                self._save_transition_log(transitions, iter_dir, iteration=i + 1)

            # Save transition event rasters (v3.2)
            if config.RASTER_OUTPUT_TRANSITION_EVENTS:
                self._save_transition_event_rasters(
                    transitions, self._initial_lulc.shape, iter_dir
                )

            # Save Stock & Flow outputs (v3.2)
            if self.use_stockflow and stock_maps:
                if config.SAVE_STOCK_RASTERS:
                    self._save_stock_rasters(stock_maps, iter_dir)
                self._save_flow_log(flow_records, iter_dir, iteration=i + 1)
                self._save_stock_table(stock_maps, iter_dir, iteration=i + 1)

            self.iter_dirs.append(iter_dir)

        print(f"\n  All iterations complete. Results in '{self.out_dir}'")

    def diagnostic(self) -> None:
        """Print expected transitions per class/group (no multipliers applied)."""
        lulc  = self._initial_lulc
        shape = lulc.shape
        print("\n[DIAGNOSTIC] Expected transitions per timestep (base probs only):")
        for from_id, outgoing in self.trans_index.items():
            n_cells = int(np.sum(lulc == from_id))
            if n_cells == 0 or not outgoing:
                continue
            sc = self.classes[from_id]
            print(f"\n  Class {from_id} ({sc.name}): {n_cells:,} cells")
            for to_id, base_prob, group in outgoing:
                mult      = get_multiplier(self.spatial_mults, group, shape)
                mean_mult = float(mult[lulc == from_id].mean())
                expected  = n_cells * base_prob * mean_mult
                to_name   = self.classes[to_id].name
                print(
                    f"    → {to_name:20s} [{group:25s}]  "
                    f"base={base_prob:.4f}  mean_sp_mult={mean_mult:.4f}  "
                    f"expected_fires={expected:.0f}"
                )

    # ── Internal: single iteration ────────────────────────────────────────────

    def _run_single_iteration(
        self,
        seed: int,
    ) -> tuple[list[np.ndarray], list[list[TransitionRecord]], list[np.ndarray],
               list[dict[str, np.ndarray]], list[list]]:

        rng        = np.random.default_rng(seed)
        lulc       = self._initial_lulc
        rows, cols = lulc.shape
        n_cls      = max(self.classes.keys())
        shape      = (rows, cols)
        ones       = np.ones(shape, dtype=np.float32)

        maps: list[np.ndarray]                        = [lulc.copy()]
        all_transitions: list[list[TransitionRecord]] = []
        current = lulc.copy()

        # ── Age initialisation (v2.3) ─────────────────────────────────────
        if self.use_age and self._initial_age is not None:
            current_age = self._initial_age.copy()
        else:
            current_age = None
        age_maps: list[np.ndarray] = (
            [current_age.copy()] if current_age is not None else []
        )

        # Build per-transition age reset lookup from transition index
        # trans_age_info[(from_id, to_id, group)] = (age_reset, age_relative)
        trans_age_info: dict[tuple, tuple] = {}
        for rule in self.trans_index.get("_rules", []):
            pass  # populated below from raw rules stored on engine
        # We need the raw rules — store them during build_transition_index
        # Use self._trans_rules if available (set in load())
        raw_rules = getattr(self, "_trans_rules", [])
        for rule in raw_rules:
            key = (rule.from_class, rule.to_class, rule.group)
            trans_age_info[key] = (rule.age_reset, rule.age_relative)

        # ── Stock & Flow initialisation (v3.2) ──────────────────────────────
        if self.use_stockflow:
            current_stocks = init_stocks(
                stock_types       = self._stock_types,
                shape             = shape,
                initial_links     = self._initial_stock_links,
                state_attr_rules  = self._state_attr_rules,
                age_map           = current_age,
            )
            stock_maps: list[dict[str, np.ndarray]] = [
                {k: v.copy() for k, v in current_stocks.items()}
            ]
            all_flow_records: list[list] = []
        else:
            current_stocks = None
            stock_maps = []
            all_flow_records = []

        for t in range(self.n_timesteps):
            year = self.start_year + t

            # ── Sample temporal multipliers ───────────────────────────────
            group_mults: dict[str, float] = {}
            if self.use_trans_multiplier and self.trans_mult_rules:
                group_mults = sample_transition_multipliers(
                    self.trans_mult_rules, rng
                )

            draws = rng.random(shape)

            # ── Adjacency (frozen at start of timestep) ───────────────────
            fracs = (
                compute_neighbor_fractions(current, n_cls)
                if self.use_adjacency else None
            )

            new_map           = current.copy()
            year_transitions: list[TransitionRecord] = []
            cum_prob          = np.zeros(shape, dtype=np.float32)
            transition_fired  = np.zeros(shape, dtype=bool)

            # Age reset accumulators (v2.3)
            fire_age_reset    = np.zeros(shape, dtype=bool)
            fire_age_value    = np.full(shape, -1, dtype=np.int32)

            for from_id, outgoing in self.trans_index.items():
                if not outgoing:
                    continue
                src_mask = (current == from_id)
                if not src_mask.any():
                    continue

                from_name = self.classes[from_id].name

                for to_id, base_prob, group in outgoing:
                    to_name = self.classes[to_id].name

                    # 1. Temporal multiplier
                    t_mult = group_mults.get(group, 1.0)

                    # 2. Age gate (v2.3)
                    if current_age is not None:
                        # Look up age constraints from raw rules
                        age_min_gate = age_max_gate = None
                        for rule in raw_rules:
                            rn_from = rule.from_class.split(":")[0]
                            rn_to   = rule.to_class.split(":")[0]
                            if (rn_from == from_name and
                                rn_to   == to_name   and
                                rule.group == group):
                                age_min_gate = rule.age_min
                                age_max_gate = rule.age_max
                                break
                        age_ok = age_gate_mask(
                            current_age, age_min_gate, age_max_gate
                        )
                    else:
                        age_ok = np.ones(shape, dtype=bool)

                    # 3. Adjacency multiplier
                    if self.use_adjacency:
                        adj_frac = fracs[:, :, to_id]

                        # v3.1: use this group's CSV-driven strength if
                        # available, else fall back to the global scalar.
                        # STRICT_EXPANSION_GROUPS classification (whether
                        # a neighbour of the target class is REQUIRED to
                        # fire at all) is unaffected by the strength
                        # source — it's a separate STRATEGICC-specific
                        # setting, not part of the adjacency strength CSVs.
                        strength = self._adjacency_strength_map.get(
                            group, config.ADJACENCY_STRENGTH
                        )

                        if group in config.STRICT_EXPANSION_GROUPS:
                            reachable = adj_frac > 0.0
                            adj_mult  = adj_frac * strength
                        else:
                            reachable = np.ones(shape, dtype=bool)
                            adj_mult  = 1.0 + adj_frac * strength
                    else:
                        reachable = np.ones(shape, dtype=bool)
                        adj_mult  = ones

                    # 4. Spatial multiplier
                    sp_mult = (
                        get_multiplier(self.spatial_mults, group, shape)
                        if self.use_spatial_mult else ones
                    )

                    # 5. Effective probability
                    p_eff = (base_prob * t_mult * adj_mult * sp_mult).astype(np.float32)

                    # 5b. Transition target override (v3.1)
                    target_amount = self._targets_by_timestep.get(t, {}).get(group)
                    has_target = target_amount is not None

                    # 6. Eligible cells (add age gate)
                    # When a target is active, a group with zero base
                    # probability must still be eligible — the target
                    # itself is the driver, not p_eff. The age/reachable/
                    # source-class gates still apply regardless of target.
                    prob_gate = (p_eff > 0) | has_target
                    eligible = (
                        src_mask & reachable & age_ok
                        & ~transition_fired & prob_gate
                    )
                    if not eligible.any():
                        continue

                    # 7/8. Fire — branch on whether this group uses
                    # patch-growing (v2.5) or independent-cell firing,
                    # further branching on whether a target is active (v3.1)
                    if group in self._size_bins:
                        if has_target:
                            # Target REPLACES the p_eff-derived budget
                            # directly for size-distribution groups,
                            # matching the official target algorithm.
                            budget = target_to_patch_budget(
                                target_amount, self.px_area
                            )
                            fire = grow_patches_for_group(
                                p_eff           = p_eff,
                                eligible        = eligible,
                                size_bins       = self._size_bins[group],
                                px_area_ha      = self.px_area_ha,
                                rng             = rng,
                                budget_override = budget,
                            )
                        else:
                            fire = grow_patches_for_group(
                                p_eff      = p_eff,
                                eligible   = eligible,
                                size_bins  = self._size_bins[group],
                                px_area_ha = self.px_area_ha,
                                rng        = rng,
                            )
                        # Patches consume the full budget mass of the
                        # cells they claim — credit cum_prob accordingly
                        # so other groups competing for the same cells
                        # this timestep see a consistent picture.
                        cum_prob[fire] += p_eff[fire]
                    else:
                        if has_target:
                            # Target SCALES p_eff so the expected fired
                            # area matches the target, preserving
                            # stochastic variance (matches the official
                            # non-size-distribution target algorithm).
                            p_eff = scale_probability_to_target(
                                p_eff, eligible, target_amount, self.px_area
                            )

                        # Cap against remaining budget
                        remaining = np.clip(1.0 - cum_prob, 0.0, 1.0)
                        p_capped  = np.minimum(p_eff, remaining)

                        fire = (
                            eligible
                            & (draws >= cum_prob)
                            & (draws < cum_prob + p_capped)
                        )
                        cum_prob[fire] += p_capped[fire]

                    fired_r, fired_c = np.where(fire)
                    for r, c in zip(fired_r, fired_c):
                        year_transitions.append(
                            TransitionRecord(year, int(r), int(c),
                                             from_id, to_id, group)
                        )

                    new_map[fire]          = to_id
                    transition_fired[fire] = True

                    # 9. Record age reset behaviour (v2.3)
                    if current_age is not None and fire.any():
                        # Find matching rule age reset settings
                        age_reset_val    = True
                        age_relative_val = None
                        for rule in raw_rules:
                            rn_from = rule.from_class.split(":")[0]
                            rn_to   = rule.to_class.split(":")[0]
                            if (rn_from == from_name and
                                rn_to   == to_name   and
                                rule.group == group):
                                age_reset_val    = rule.age_reset
                                age_relative_val = rule.age_relative
                                break

                        if age_reset_val:
                            fire_age_reset[fire] = True
                        if age_relative_val is not None:
                            fire_age_value[fire] = age_relative_val

            current = new_map
            maps.append(current.copy())
            all_transitions.append(year_transitions)

            # ── Update age map (v2.3) ─────────────────────────────────────
            if current_age is not None:
                current_age = update_age(
                    current_age,
                    transition_fired,
                    fire_age_reset,
                    fire_age_value,
                )
                age_maps.append(current_age.copy())

            # ── Run Stock & Flow for this timestep (v3.2) ───────────────────
            if self.use_stockflow and current_stocks is not None:
                flow_mult_sample: dict[str, float] = {}
                if self._flow_mult_rules:
                    flow_mult_sample = sample_flow_multipliers(
                        self._flow_mult_rules, rng
                    )
                current_stocks, year_flow_records = run_flows_for_timestep(
                    stocks            = current_stocks,
                    pathways          = self._flow_pathways,
                    flow_order        = self._flow_order,
                    year_transitions  = year_transitions,
                    age_map           = current_age,
                    state_attr_rules  = self._state_attr_rules,
                    classes           = self.classes,
                    year              = year,
                    flow_mult_sample  = flow_mult_sample,
                    lulc_map          = current,
                )
                stock_maps.append(
                    {k: v.copy() for k, v in current_stocks.items()}
                )
                all_flow_records.append(year_flow_records)

            # Console summary
            m_str = "  ".join(
                f"{g}={v:.3f}" for g, v in group_mults.items()
            ) if group_mults else "—"
            age_str = (f"  age_mean={current_age.mean():.1f}"
                       if current_age is not None else "")
            print(
                f"    t={year + 1}  transitions={len(year_transitions):5,}  "
                f"t_mults=[{m_str}]{age_str}"
            )

        return maps, all_transitions, age_maps, stock_maps, all_flow_records

    # ── Internal: per-iteration disk outputs ──────────────────────────────────

    def _save_area_table(
        self,
        maps: list[np.ndarray],
        out_dir: Path,
        iteration: int,
    ) -> None:
        unit_col = f"area_{self.area_unit}"
        rows = []
        for t, arr in enumerate(maps):
            year = self.start_year + t
            for cid, sc in self.classes.items():
                rows.append({
                    "iteration":  iteration,
                    "year":       year,
                    "class_id":   cid,
                    "class_name": sc.name,
                    unit_col:     float(np.sum(arr == cid)) * self.px_area,
                })
        pd.DataFrame(rows).to_csv(out_dir / "area_table.csv", index=False)

    def _save_transition_log(
        self,
        transitions: list[list[TransitionRecord]],
        out_dir: Path,
        iteration: int,
    ) -> None:
        rows = []
        for year_trans in transitions:
            for rec in year_trans:
                rows.append({
                    "iteration":  iteration,
                    "year":       rec.year,
                    "row":        rec.row,
                    "col":        rec.col,
                    "from_class": self.classes[rec.from_id].name,
                    "to_class":   self.classes[rec.to_id].name,
                    "group":      rec.group,
                })
        pd.DataFrame(rows).to_csv(
            out_dir / "transition_log.csv", index=False
        )

    def _save_transition_event_rasters(
        self,
        transitions: list[list[TransitionRecord]],
        shape:       tuple[int, int],
        out_dir:     Path,
    ) -> None:
        """
        Save one raster per timestep where each fired cell holds the
        destination class ID (0 = no transition that timestep). v3.2 —
        gated by config.RASTER_OUTPUT_TRANSITION_EVENTS /
        RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS.
        """
        from strategicc.io.raster import _TAG_TIE_POINT, _TAG_PIXEL_SCALE
        from PIL import Image

        events_dir = out_dir / "transition_events"
        events_dir.mkdir(parents=True, exist_ok=True)

        keep_tags = {k: self.src_tags[k]
                     for k in (_TAG_TIE_POINT, _TAG_PIXEL_SCALE, 34735, 34736, 34737)
                     if k in self.src_tags}
        save_kwargs = {"compression": "lzw"}
        if keep_tags:
            save_kwargs["tiffinfo"] = keep_tags

        stride = max(1, config.RASTER_OUTPUT_TRANSITION_EVENT_TIMESTEPS)

        for t, year_trans in enumerate(transitions):
            if t % stride != 0 and t != len(transitions) - 1:
                continue
            year = self.start_year + t
            canvas = np.zeros(shape, dtype=np.uint8)
            for rec in year_trans:
                canvas[rec.row, rec.col] = rec.to_id
            Image.fromarray(canvas, mode="L").save(
                str(events_dir / f"events_{year}.tif"), **save_kwargs
            )

    def _save_stock_rasters(
        self,
        stock_maps: list[dict[str, np.ndarray]],
        out_dir:    Path,
    ) -> None:
        """
        Save one raster per stock type per timestep. v3.2 — output
        directory structure: {out_dir}/stocks/{stock_type}/stock_{year}.tif
        """
        from strategicc.io.raster import _TAG_TIE_POINT, _TAG_PIXEL_SCALE
        from PIL import Image

        keep_tags = {k: self.src_tags[k]
                     for k in (_TAG_TIE_POINT, _TAG_PIXEL_SCALE, 34735, 34736, 34737)
                     if k in self.src_tags}
        save_kwargs = {"compression": "lzw"}
        if keep_tags:
            save_kwargs["tiffinfo"] = keep_tags

        if not stock_maps:
            return
        stock_types = list(stock_maps[0].keys())

        for stock_type in stock_types:
            stock_dir = out_dir / "stocks" / stock_type
            stock_dir.mkdir(parents=True, exist_ok=True)
            for t, snapshot in enumerate(stock_maps):
                year = self.start_year + t
                arr  = snapshot[stock_type]
                # Stocks are float — save as float32 GeoTIFF via a simple
                # numpy .save fallback when not using rasterio; here we
                # use Pillow's "F" mode for 32-bit float TIFFs.
                Image.fromarray(arr.astype(np.float32), mode="F").save(
                    str(stock_dir / f"stock_{year}.tif"), **save_kwargs
                )

    def _save_flow_log(
        self,
        flow_records: list[list],
        out_dir:      Path,
        iteration:    int,
    ) -> None:
        """
        Save aggregate flow events (one row per pathway per timestep that
        fired), plus a per-class breakdown (flow_log_by_class.csv, v3.2)
        used by Mode C SEEA-EA valuation.
        """
        rows = []
        class_rows = []
        for year_records in flow_records:
            for rec in year_records:
                rows.append({
                    "iteration":        iteration,
                    "year":             rec.year,
                    "flow_type":        rec.flow_type,
                    "from_stock":       rec.from_stock,
                    "to_stock":         rec.to_stock,
                    "transition_group": rec.transition_group or "",
                    "total_amount":     rec.total_amount,
                })
                if rec.by_class:
                    for class_name, amount in rec.by_class.items():
                        class_rows.append({
                            "iteration":  iteration,
                            "year":       rec.year,
                            "flow_type":  rec.flow_type,
                            "from_stock": rec.from_stock,   # v3.3
                            "to_stock":   rec.to_stock,      # v3.3
                            "class_name": class_name,
                            "amount":     amount,
                        })
        pd.DataFrame(rows).to_csv(out_dir / "flow_log.csv", index=False)
        pd.DataFrame(class_rows).to_csv(
            out_dir / "flow_log_by_class.csv", index=False
        )

    def _save_stock_table(
        self,
        stock_maps: list[dict[str, np.ndarray]],
        out_dir:    Path,
        iteration:  int,
    ) -> None:
        """Save total stock quantity per stock type per timestep (tabular summary)."""
        rows = []
        for t, snapshot in enumerate(stock_maps):
            year = self.start_year + t
            for stock_type, arr in snapshot.items():
                rows.append({
                    "iteration":  iteration,
                    "year":       year,
                    "stock_type": stock_type,
                    "total":      float(arr.sum()),
                    "mean":       float(arr.mean()),
                })
        pd.DataFrame(rows).to_csv(out_dir / "stock_table.csv", index=False)
