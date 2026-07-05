# Changelog

All notable changes to STRATEGICC are documented here.

**A note on how this file was built:** no `CHANGELOG.md` existed before this
entry. Versions 1.1 through 3.5.3 were reconstructed retroactively from
version markers left in module docstrings/inline comments (e.g. `v3.2`
headers) cross-referenced against the release dates published on
[PyPI](https://pypi.org/project/strategicc/#history). Where a version bump
happened without an accompanying docstring change (e.g. 2.2 → 2.2.1), the
entry is left thin rather than guessed at. Going forward, entries should be
written at release time, not reconstructed after the fact.

This project attempts to follow [Semantic Versioning](https://semver.org/).
Note on 3.6.0: strictly, the `EcosystemService` field rename and the
`build_initial_age_from_raster()` return-type change below are breaking
changes at the Python attribute/signature level (no back-compat alias
exists for either) and would warrant a MAJOR bump under strict semver.
It's released as 3.6.0 on the judgment call that the documented public
interface is the CSV schema and `SEEAAccount`/`load_ecosystem_services()`/
`StrategiccEngine`, all of which remain backward compatible. See the
"Versioning note" under the 3.6.0 entry for the full reasoning.

---

## [3.11.0]

> **Note:** no `[3.10.0]` entry exists in this file -- flagging rather than
> backfilling it, since v3.10's actual changes weren't reconstructed from
> source at the time this entry was written. `DISTRIBUTIONS_CSV` (added to
> the manifest template below) was already recognised by `config.py`'s
> `_MANIFEST_SCHEMA` as of that version.

Calibration manifest generation now covers Stock & Flow (Section 7) and
`Distributions.csv`, and gains an in-place update path for manifests the
user has already hand-filled. Plus one bug fix in Stock & Flow output
aggregation. No changes to the simulation engine's transition/spatial-mult
logic or SEEA-EA valuation math.

### Added

- **`fill_manifest_from_calibration()`** (`strategicc/calibration/manifest.py`),
  updates an *existing*, hand-filled `RunManifest.txt` in place -- unlike
  `save_calibration_manifest()`, which always generates a brand-new file
  from scratch. Only the calibration-derived fields (`AgeFileName`,
  `TRANSITIONS_CSV`, `TRANSITION_MULT_CSV`, `TRANSITION_SIZE_CSV`,
  `DISTRIBUTIONS_CSV`) are touched; every other line -- including a
  hand-filled Section 7 -- is preserved untouched. Lines inside fenced
  (` ``` `) blocks are never touched, matching `config.load_manifest()`'s
  own appendix/documentation handling. If `DISTRIBUTIONS_CSV` is requested
  but has no existing line in the file at all, a new line is appended into
  Section 2 (after `TRANSITION_MULT_CSV` if found, else before `SECTION 3`,
  else at end of file) rather than silently skipped.

- **`load_group_map_csv()`** (`strategicc/calibration/transitions.py`),
  builds a `group_map` dict from a user-authored CSV instead of requiring
  it hand-typed as a Python dict every session. Rather than a new file
  format, this reuses the existing `Transitions.csv` schema itself
  (`StateClassIdSource`, `StateClassIdDest`, `TransitionTypeId`,
  `Probability`) -- the user lists which `(from, to, type)` rows to
  calibrate with `Probability` left blank/unused, and this function
  resolves the class labels back to integer ids via a `classes` dict to
  build the same `dict[(from_id, to_id), group_name]` shape that
  `compute_transition_rates()`, `compute_temporal_distribution()`, and
  `compute_size_distribution()` already expect. Rows with an unresolvable
  class label are skipped with a printed warning rather than silently
  dropped.

- **`save_calibration_manifest()`** now accepts a `distributions_path`
  argument and emits a `DISTRIBUTIONS_CSV` row in Section 2 (marked
  `[calibration]` when supplied, `# TODO:` otherwise) -- previously
  omitted entirely, even though `config.py` already recognised the key.

- **`save_calibration_manifest()`** now emits **Section 7 (Stock & Flow)**
  as `# TODO:` placeholders (`USE_STOCKFLOW`, `STOCK_TYPE_CSV`,
  `STOCK_GROUP_CSV`, `STOCK_GROUP_MEMBERSHIP_CSV`, `FLOW_TYPE_CSV`,
  `FLOW_ORDER_CSV`, `FLOW_PATHWAYS_CSV`, `FLOW_MULTIPLIER_CSV`,
  `STATE_ATTRIBUTE_TYPE_CSV`, `STATE_ATTRIBUTE_VALUES_CSV`,
  `INITIAL_STOCK_NON_SPATIAL_CSV`, `SAVE_STOCK_RASTERS`,
  `SEEA_VALUATION_MODE`) -- previously the generated manifest stopped
  after Section 6, silently omitting Stock & Flow from the file entirely.
  None of these are calibration-derivable, so all are `# TODO:` (never
  `[calibration]`).

- `inputs/RunManifest.txt` (master template) and
  `docs/manifest_reference.md` now include a `DISTRIBUTIONS_CSV` row in
  Section 2 -- the same gap that caused it to be missing from a real
  user-authored manifest is now closed at the template level too. A
  filled `Distributions.csv` example was added to the template's Appendix.

### Fixed

- **`aggregate_flow_by_class()`** (`strategicc/stockflow/aggregation.py`)
  crashed with `pd.errors.EmptyDataError` whenever an iteration produced
  no `by_class` flow records: `engine._save_flow_log()` always writes
  `flow_log_by_class.csv` unconditionally (even as a zero-column/empty
  file), and the aggregation function only checked `log_path.exists()`,
  not emptiness, before calling `pd.read_csv()`. Now wrapped in
  `try/except pd.errors.EmptyDataError` so an empty-but-present log is
  skipped rather than raising.

---
## [3.10.0] 

Calibration workflow improvements: predefined output paths, auto-generated
run manifest, and a post-calibration summary report. No changes to the
simulation engine or SEEA-EA accounting.

Added functionality for generating temporal distribution table (normalized annual rate)
---

## [3.9.0] 

Calibration workflow improvements: predefined output paths, auto-generated
run manifest, and a post-calibration summary report. No changes to the
simulation engine or SEEA-EA accounting.

### Added

- **`strategicc.calibration.paths`**, single module defining all
  predefined output paths under `calibration_result/` (relative to cwd).
  All calibration `save_*` functions now default to these paths so a
  typical calibration run requires no path management by the user.
  Explicit `out_path` arguments still accepted for custom layouts
  (backward compatible).

- **`save_calibration_manifest()`** (`strategicc/calibration/manifest.py`)
  generates a `calibration_result/RunManifest_calibrated.txt` pre-filled
  with the four fields that calibration can populate automatically:
  `TRANSITIONS_CSV`, `TRANSITION_MULT_CSV`, `TRANSITION_SIZE_CSV`, and
  `AgeFileName`. All remaining fields are left blank with `# TODO:` hint
  comments explaining what each requires. Feature toggles (`USE_AGE`,
  `USE_TRANS_MULTIPLIER`) are set to match what was actually calibrated
  in the current run.

- **`calibration_summary()`** (`strategicc/calibration/report.py`),
  prints a formatted calibration result table to stdout, saves a
  multi-panel summary PNG (`calibration_result/calibration_summary.png`),
  and returns a dict of key statistics for programmatic access. The plot
  includes: age raster spatial map, age histogram (≤10 bins), calibrated
  transition probability bar chart, temporal multiplier range chart, and
  patch-size cumulative distribution. The printed table lists which of
  the four calibratable inputs were successfully derived and which fields
  still require manual entry.

### Changed

- **`save_transitions_csv()`**, **`save_temporal_distribution_csv()`**,
  **`save_size_distribution_csv()`**, **`save_age_raster()`**, `out_path`
  argument is now optional (defaults to the corresponding predefined path
  under `calibration_result/`). Return type changed from `None` to `Path`
  (the path actually written to), so callers can pass the result directly
  to `save_calibration_manifest()` without tracking paths manually.
  Existing callers passing an explicit `out_path` are unaffected.

- All calibration outputs now call `out_path.parent.mkdir(parents=True,
  exist_ok=True)` before writing, so `calibration_result/` is created
  automatically on first save.

---

## [3.8.0]

Three independent improvements shipped together: calibration gains
patch-size distribution derivation (closes a gap in the calibratable
input set); a pre-existing spatial path bug is fixed across
`outputs.py` and `animate.py`; and implicit scipy/rasterio import
assumptions that broke `import strategicc` on a base install are
corrected.

### Added

- **`compute_size_distribution()`** and **`save_size_distribution_csv()`**
  (`strategicc/calibration/transitions.py`), derive a historical
  patch-size frequency table (`TransitionSizeDistribution.csv`) from the
  same LULC time series stack already loaded for transition-rate
  calibration. Uses 8-connected component labeling (`scipy.ndimage`) to
  identify contiguous patches of transitioning cells per year-pair,
  converts pixel counts to hectares via `px_area_ha`, pools patch sizes
  across all observed year-pairs per group, and bins into an
  equal-frequency (quantile) cumulative histogram. The 8-connected
  default matches `core.patches.grow_patch()`'s own BFS connectivity, so
  calibration and simulation reason about "a patch" consistently.
  Groups with fewer than `min_patches` observed patches are skipped with
  a printed warning; the engine falls back to independent-cell firing for
  any group absent from `TransitionSizeDistribution.csv`. Uses the same
  `group_map` as `compute_transition_rates()` and
  `compute_temporal_distribution()`, keeping one grouping scheme across
  all three calibrated CSVs.

### Fixed

- **Double-nested spatial output path in `aggregate_spatial()` and
  `plot_spatial_summary()`.**
  `aggregate_spatial()` wrote modal/uncertainty rasters to
  `summary_dir / "spatial"` while `plot_spatial_summary()` read from
  `summary_dir` (no subfolder), so the uncertainty overlay panels in
  `plot_spatial_summary()` silently rendered blank and the `lulc_mean_*`
  count in the figure title was always 0. Both functions now write/read
  directly from `summary_dir`. `animate.py`'s `_load_modal_frames()` and
  main entry point had the same hardcoded `/ "spatial"` path and are
  fixed to match. Stale path references in `docs/guides/04_visualization.md`
  and `docs/reference/animate.md` are updated.

- **`strategicc.accounting.outputs.plot_monetary_flows()` and
  `save_all_accounts()` crashed when `SEEAAccount` was constructed
  without `area_df=`.**
  `acct.uncertainty_summary()` returned `None` in that case but was
  called unconditionally, raising `AttributeError: 'NoneType' object has
  no attribute 'set_index'`. Both call sites now guard with
  `if unc_df is not None`, consistent with the existing guard on
  `physical_flow_account()` in the same file.

- **`import strategicc` failed with `ModuleNotFoundError: No module named
  'scipy'` on a base install** (i.e. `pip install strategicc` without
  extras). `strategicc/core/age.py` and `strategicc/outputs.py` both had
  unconditional top-level `from scipy import stats as scipy_stats`
  imports, but scipy was only declared as a `[calibration]` optional
  extra, not a base dependency. Since `core.age`'s truncated-normal age
  sampling and `outputs.py`'s modal-map aggregation are core-path
  features used in every standard run, `scipy` is now a base dependency
  (`pyproject.toml`). The module-level imports are retained as lazy
  inline imports at the point of use (inside the `if rule.age_sd > 0`
  branch and inside `aggregate_spatial()`'s per-timestep loop
  respectively), matching the established rasterio lazy-import pattern
  already used elsewhere in the codebase.

- **`import strategicc.calibration` failed with `ImportError:
  strategicc.calibration requires rasterio` even when rasterio was
  absent but not needed.** `strategicc/calibration/loader.py` wrapped
  `import rasterio` in a module-level `try/except` that re-raised
  immediately, meaning the entire `calibration` package became
  unimportable without rasterio, including `compute_size_distribution()`
  and other functions that never touch rasterio. The import is now
  genuinely deferred into `load_lulc_timeseries()`'s function body via
  a `_require_rasterio()` helper, so `calibration` imports cleanly and
  only `load_lulc_timeseries()` (the one function that actually needs
  rasterio) raises on call.

### Changed

- **`scipy` moved from `[calibration]` optional extra to base
  dependency** (`pyproject.toml`). See Fixed above for rationale.
  `rasterio` remains a `[calibration]`-only extra, all rasterio imports
  throughout the codebase are already correctly deferred to function-call
  time and are unaffected.

---

## [3.7.0]

### Added

- **Transition size distribution support in the simulation engine.**
  `TransitionSizeDistribution.csv` is now loaded and consumed by
  `strategicc/core/patches.py` (patch-growing mechanic introduced in
  2.5.0). Groups present in this file grow as discrete 8-connected
  patches during simulation rather than firing independently
  cell-by-cell. Groups absent from the file keep the existing
  independent-cell mechanic, so partial coverage is valid and safe.

---

## [3.6.0] -- 2026-07-02

Three related fixes, all stemming from the same root issue: area/CRS
assumptions baked into pixel-area and valuation math that were only ever
checked against a single test raster (geographic, `AREA_UNIT="ha"`), and
silently produced wrong numbers outside that case. None of the three
affected any previously published output from this project, see each
entry for why, but all three are live risks for any future run outside
the exact conditions used so far.

### Fixed

- **SEEA-EA valuation was silently wrong whenever `AREA_UNIT != "ha"`.**
  `EcosystemServices.csv` prices are hectare-denominated, but
  `SEEAAccount` multiplied them directly against area figures expressed
  in whatever `AREA_UNIT` the run used (`ha` | `km2` | `px`), with no
  unit conversion. Under the default `AREA_UNIT="ha"` this was a no-op
  (conversion factor of 1.0); switching to `km2` or `px` would have
  produced monetary/physical flow accounts off by the unconverted
  factor, with no error raised.
  `SEEAAccount` now accepts `px_area_ha` (the pixel's real-world size in
  hectares, already computed by the engine) and uses
  `px_area_ha / px_area` to convert area figures back to hectares before
  applying prices. `strategicc/run.py` and all quick-start examples now
  pass it through. Omitting it on a non-`"ha"` run now prints an
  explicit warning instead of failing silently.

- **Pixel-area calculation assumed every raster was in a geographic
  (degrees) CRS.** `_pixel_area_ha()` unconditionally converted pixel
  scale via a degrees→metres approximation. A raster in a *projected*
  CRS (e.g. UTM, pixel scale already in metres, common for LULC
  products derived from Landsat/Sentinel) would be silently
  miscalculated by roughly `111,000²`. This project's actual working
  raster happens to be in EPSG:4326 (geographic), so no published
  result was affected, but the capability gap was real for any
  projected input. `_pixel_area_ha()` now parses `GTModelTypeGeoKey`
  from the GeoTIFF's `GeoKeyDirectoryTag` and branches: projected CRSs
  use pixel scale directly as metres (with a warning if the linear unit
  isn't metres), geographic CRSs keep the original degree-based formula
  unchanged. An undetermined CRS type falls back to the historical
  geographic assumption, now with an explicit warning rather than a
  silent guess.

- **No verification that state class/LULC, age, and spatial multiplier
  rasters shared the same CRS.** A mismatched raster would be resampled
  to the target grid (`load_spatial_multipliers()`'s nearest-neighbour
  resize) as if it were spatially aligned, silently corrupting
  transition targeting and any area-derived accounting, with nothing
  in the run indicating a problem. `engine.load()` now compares every
  age and spatial-multiplier raster's CRS against the LULC raster's and
  **raises immediately, blocking the run**, on a confirmed mismatch. If
  a raster's CRS can't be determined at all (missing GeoTIFF tags), the
  run proceeds with a warning rather than blocking, since that's not
  evidence of an actual mismatch.

### Changed

- **`EcosystemServices.csv` columns renamed** `ValuePerHa` →
  `ValuePerUnitArea`, `PhysicalValuePerHa` → `PhysicalValuePerUnitArea`,
  to make explicit that these are area-based prices, and to avoid
  confusion with Mode C's physical-unit pricing, which is *not*
  area-denominated. (Briefly named `ValuePerUnit`/`PhysicalValuePerUnit`
  mid-development before this final naming.) `EcosystemService`
  dataclass fields renamed to match: `value_per_ha` →
  `value_per_unit_area`, `physical_per_ha` → `physical_per_unit_area`.
  `load_ecosystem_services()` accepts all historical column names with
  a one-time warning, existing CSV files do not need to be edited.

- **Breaking (Python-level):** `strategicc.core.age.build_initial_age_from_raster()`
  now returns `(arr, CRSInfo)` instead of just `arr`, to support the CRS
  consistency check above. `strategicc.core.spatial.load_spatial_multipliers()`
  gained a `reference_crs: CRSInfo | None = None` keyword argument
  (backward compatible, `None` skips the check entirely, matching
  pre-3.6 behaviour).

### Added

- `strategicc.io.raster.CRSInfo`, a lightweight CRS descriptor (model
  type + EPSG code, no GDAL/pyproj dependency) with `.compare()`,
  `.describe()`, and constructors from raw GeoTIFF tags or a rasterio
  CRS object.
- `get_crs_info()` / `assert_crs_consistent()` public helpers.
- Tests: unit-conversion correctness (`km2`-unit run matches `ha`-unit
  run for the same physical area), legacy-column-name loading, projected
  vs geographic pixel-area calculation, CRS mismatch blocking for both
  age and spatial-multiplier rasters, and CRS-match/unknown pass-through
  behaviour.

### Versioning note

Strictly, the `EcosystemService` field rename (no back-compat alias at
the Python attribute level) and the `build_initial_age_from_raster()`
return-type change are breaking changes and would warrant a MAJOR bump
under strict semver. Released as 3.6.0 (MINOR) on the judgment call that
the documented public interface, the CSV schema and the
`SEEAAccount`/`load_ecosystem_services()`/`StrategiccEngine` entry
points, remains backward compatible; only lower-level internals that
aren't part of the documented interface changed shape.

---

## [3.5.3] -- 2026-07-01
## [3.5.2] -- 2026-07-01
### Added
- `strategicc/animate.py` (v3.5): standalone two-panel GIF/MP4 animation,
  modal LULC map per timestep alongside a synced statistics line chart.

*(No 3.5.0/3.5.1 entries, not published to PyPI, or superseded before a
public release; 3.5.2 is the first 3.5.x version on PyPI.)*

---

## [3.4.0] -- 2026-06-30
### Added
- `strategicc/calibration` module (v3.4): derive STRATEGICC inputs
  (transition rates, temporal multiplier distributions, age structure)
  directly from a historical LULC time series supplied as a zip of
  yearly GeoTIFFs.
- Zip-file support for LULC time series input in the calibration loader
  and engine (`strategicc/calibration/loader.py`, `engine.py`).

---

## [3.3.0] -- 2026-06-30
### Changed
- `EcosystemServices.csv`: `ValuePerHa`/`PhysicalValuePerHa` renamed to
  `ValuePerUnit`/`PhysicalValuePerUnit` (superseded in 3.5.4, see above).
- Mode C valuation (stock/flow-sourced physical quantities) documented
  and finalized in `strategicc/accounting/seea.py` and
  `strategicc/accounting/csv_loader.py`.

---

## [3.2.0] -- 2026-06-30
### Added
- `strategicc/stockflow` module: per-cell, per-timestep Stock & Flow
  accounting, material quantities moving between pools via Flow
  Pathways, either age-driven or transition-triggered.
- Mode C SEEA-EA valuation: ecosystem services can pull their physical
  quantity directly from Stock & Flow engine output
  (`StockFlowSource` column, `"flow:<Type>"` / `"stock:<Type>"`).
- Raster/summary output gating options (`RASTER_OUTPUT_SC`,
  `SUMMARY_OUTPUT_SC/TR`, transition event rasters, age raster output).

---

## [3.1.0] -- 2026-06-30
### Added
- Transition Targets (`strategicc/core/targets.py`): area-based overrides
  that replace or scale a transition group's probability-derived budget
  for a given timestep.
- CSV-driven transition adjacency groups and per-group adjacency
  strength configuration.

---

## [3.0.0] -- 2026-06-30
### Added
- `RunManifest.txt` manifest loader (`strategicc/config.py`), load an
  entire run configuration from a single text file instead of setting
  `cfg.*` attributes individually in Python.

---

## [2.5.0] -- 2026-06-30
### Added
- Multi-iteration support in `StrategiccEngine` (`strategicc/engine.py`).
- Patch-growing mechanic (`strategicc/core/patches.py`) for transition
  groups with a historical size distribution
  (`TransitionSizeDistribution.csv`).

---

## [2.4] -- 2026-06-30
### Added
- Calibration submodules: `age.py`, `loader.py`, `temporal.py`,
  `transitions.py`, deriving transition rates and temporal multiplier
  distributions from historical LULC data (later consolidated into the
  `strategicc.calibration` package in 3.4).

---

## [2.2.1] -- 2026-06-29
No docstring-level markers found for this patch; likely a packaging or
minor bug-fix release between 2.2.0 and 2.4.

---

## [2.2.0] -- 2026-06-29
### Added
- Configurable `AREA_UNIT` (`ha` | `km2` | `px`) and `px_area`/`px_area_ha`
  pixel-area tracking in `strategicc/engine.py`.
- Modal-area-table SEEA-EA input path (`modal_to_area_table`) as the
  primary spatially-consistent area source for accounting.
- Age tracking (`strategicc/core/age.py`) with `AgeMin`/`AgeMax` gates.

*(This is also the version whose `AREA_UNIT` design the 3.5.4 fix above
correctly implements for the first time, the configurability existed
since 2.2.0, but SEEA-EA valuation didn't respect it until now.)*

---

## [1.1] -- undated
### Added
- Stochastic transition multipliers (`strategicc/core/multipliers.py`),
  one scalar multiplier per group per timestep, sampled from
  `TransitionMultipliers.csv`.
