# Changelog

All notable changes to STRATEGICC are documented here.

**A note on how this file was built:** no `CHANGELOG.md` existed before this
entry. Versions 1.1 through 3.5.3 were reconstructed retroactively from
version markers left in module docstrings/inline comments (e.g. `—  v3.2`
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

## [3.6.0] — Unreleased

Three related fixes, all stemming from the same root issue: area/CRS
assumptions baked into pixel-area and valuation math that were only ever
checked against a single test raster (geographic, `AREA_UNIT="ha"`), and
silently produced wrong numbers outside that case. None of the three
affected any previously published output from this project — see each
entry for why — but all three are live risks for any future run outside
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
  hectares — already computed by the engine) and uses
  `px_area_ha / px_area` to convert area figures back to hectares before
  applying prices. `strategicc/run.py` and all quick-start examples now
  pass it through. Omitting it on a non-`"ha"` run now prints an
  explicit warning instead of failing silently.

- **Pixel-area calculation assumed every raster was in a geographic
  (degrees) CRS.** `_pixel_area_ha()` unconditionally converted pixel
  scale via a degrees→metres approximation. A raster in a *projected*
  CRS (e.g. UTM, pixel scale already in metres — common for LULC
  products derived from Landsat/Sentinel) would be silently
  miscalculated by roughly `111,000²`. This project's actual working
  raster happens to be in EPSG:4326 (geographic), so no published
  result was affected — but the capability gap was real for any
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
  transition targeting and any area-derived accounting — with nothing
  in the run indicating a problem. `engine.load()` now compares every
  age and spatial-multiplier raster's CRS against the LULC raster's and
  **raises immediately, blocking the run**, on a confirmed mismatch. If
  a raster's CRS can't be determined at all (missing GeoTIFF tags), the
  run proceeds with a warning rather than blocking, since that's not
  evidence of an actual mismatch.

### Changed

- **`EcosystemServices.csv` columns renamed** `ValuePerHa` →
  `ValuePerUnitArea`, `PhysicalValuePerHa` → `PhysicalValuePerUnitArea`,
  to make explicit that these are area-based prices — and to avoid
  confusion with Mode C's physical-unit pricing, which is *not*
  area-denominated. (Briefly named `ValuePerUnit`/`PhysicalValuePerUnit`
  mid-development before this final naming.) `EcosystemService`
  dataclass fields renamed to match: `value_per_ha` →
  `value_per_unit_area`, `physical_per_ha` → `physical_per_unit_area`.
  `load_ecosystem_services()` accepts all historical column names with
  a one-time warning — existing CSV files do not need to be edited.

- **Breaking (Python-level):** `strategicc.core.age.build_initial_age_from_raster()`
  now returns `(arr, CRSInfo)` instead of just `arr`, to support the CRS
  consistency check above. `strategicc.core.spatial.load_spatial_multipliers()`
  gained a `reference_crs: CRSInfo | None = None` keyword argument
  (backward compatible — `None` skips the check entirely, matching
  pre-3.6 behaviour).

### Added

- `strategicc.io.raster.CRSInfo` — a lightweight CRS descriptor (model
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
the documented public interface — the CSV schema and the
`SEEAAccount`/`load_ecosystem_services()`/`StrategiccEngine` entry
points — remains backward compatible; only lower-level internals that
aren't part of the documented interface changed shape.

---

## [3.5.3] — 2026-07-01
## [3.5.2] — 2026-07-01
### Added
- `strategicc/animate.py` (v3.5): standalone two-panel GIF/MP4 animation —
  modal LULC map per timestep alongside a synced statistics line chart.

*(No 3.5.0/3.5.1 entries — not published to PyPI, or superseded before a
public release; 3.5.2 is the first 3.5.x version on PyPI.)*

---

## [3.4.0] — 2026-06-30
### Added
- `strategicc/calibration` module (v3.4): derive STRATEGICC inputs
  (transition rates, temporal multiplier distributions, age structure)
  directly from a historical LULC time series supplied as a zip of
  yearly GeoTIFFs.
- Zip-file support for LULC time series input in the calibration loader
  and engine (`strategicc/calibration/loader.py`, `engine.py`).

---

## [3.3.0] — 2026-06-30
### Changed
- `EcosystemServices.csv`: `ValuePerHa`/`PhysicalValuePerHa` renamed to
  `ValuePerUnit`/`PhysicalValuePerUnit` (superseded in 3.5.4 — see above).
- Mode C valuation (stock/flow-sourced physical quantities) documented
  and finalized in `strategicc/accounting/seea.py` and
  `strategicc/accounting/csv_loader.py`.

---

## [3.2.0] — 2026-06-30
### Added
- `strategicc/stockflow` module: per-cell, per-timestep Stock & Flow
  accounting — material quantities moving between pools via Flow
  Pathways, either age-driven or transition-triggered.
- Mode C SEEA-EA valuation: ecosystem services can pull their physical
  quantity directly from Stock & Flow engine output
  (`StockFlowSource` column, `"flow:<Type>"` / `"stock:<Type>"`).
- Raster/summary output gating options (`RASTER_OUTPUT_SC`,
  `SUMMARY_OUTPUT_SC/TR`, transition event rasters, age raster output).

---

## [3.1.0] — 2026-06-30
### Added
- Transition Targets (`strategicc/core/targets.py`): area-based overrides
  that replace or scale a transition group's probability-derived budget
  for a given timestep.
- CSV-driven transition adjacency groups and per-group adjacency
  strength configuration.

---

## [3.0.0] — 2026-06-30
### Added
- `RunManifest.txt` manifest loader (`strategicc/config.py`) — load an
  entire run configuration from a single text file instead of setting
  `cfg.*` attributes individually in Python.

---

## [2.5.0] — 2026-06-30
### Added
- Multi-iteration support in `StrategiccEngine` (`strategicc/engine.py`).
- Patch-growing mechanic (`strategicc/core/patches.py`) for transition
  groups with a historical size distribution
  (`TransitionSizeDistribution.csv`).

---

## [2.4] — 2026-06-30
### Added
- Calibration submodules: `age.py`, `loader.py`, `temporal.py`,
  `transitions.py` — deriving transition rates and temporal multiplier
  distributions from historical LULC data (later consolidated into the
  `strategicc.calibration` package in 3.4).

---

## [2.2.1] — 2026-06-29
No docstring-level markers found for this patch; likely a packaging or
minor bug-fix release between 2.2.0 and 2.4.

---

## [2.2.0] — 2026-06-29
### Added
- Configurable `AREA_UNIT` (`ha` | `km2` | `px`) and `px_area`/`px_area_ha`
  pixel-area tracking in `strategicc/engine.py`.
- Modal-area-table SEEA-EA input path (`modal_to_area_table`) as the
  primary spatially-consistent area source for accounting.
- Age tracking (`strategicc/core/age.py`) with `AgeMin`/`AgeMax` gates.

*(This is also the version whose `AREA_UNIT` design the 3.5.4 fix above
correctly implements for the first time — the configurability existed
since 2.2.0, but SEEA-EA valuation didn't respect it until now.)*

---

## [1.1] — undated
### Added
- Stochastic transition multipliers (`strategicc/core/multipliers.py`) —
  one scalar multiplier per group per timestep, sampled from
  `TransitionMultipliers.csv`.
