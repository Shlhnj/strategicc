"""
strategicc/calibration/loader.py  —  v2.4
--------------------------------------------
Extract a zip of yearly LULC GeoTIFFs and build an annual raster stack.

Supports two period-detection modes:

1. AUTO  — filenames are parsed for a 4-digit year (e.g. "2010.tif",
           "lulc_2010.tif", "2010_classified.tif"). If consecutive years
           are missing, the most recent prior year is forward-filled with
           a printed warning (assumes no-data gap, not a real value).

2. MANUAL — caller supplies an explicit `periods` list of
           (start_year, end_year, filename_year) tuples, exactly matching
           the original script's PERIODS pattern. Useful when multi-year
           composites (e.g. Landsat 5-year mosaics) need to be expanded
           across several annual slots.

Requires the optional `rasterio` dependency.
"""

from __future__ import annotations
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError as e:
    raise ImportError(
        "strategicc.calibration requires rasterio. "
        "Install with: pip install rasterio"
    ) from e


_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


@dataclass
class LULCTimeSeries:
    """
    An annual LULC raster stack with shared georeferencing.

    Attributes
    ----------
    stack   : uint8 ndarray, shape (n_years, rows, cols)
    years   : list[int] — one entry per layer in `stack`, ascending
    profile : rasterio profile dict (CRS, transform, etc.) from the
              first loaded raster — used to write outputs with matching
              georeferencing.
    """
    stack:   np.ndarray
    years:   list[int]
    profile: dict

    @property
    def shape(self) -> tuple[int, int]:
        return self.stack.shape[1:]

    def year_index(self, year: int) -> int:
        """Return the stack index for a given year, or raise ValueError."""
        try:
            return self.years.index(year)
        except ValueError:
            raise ValueError(
                f"Year {year} not in timeseries (range "
                f"{self.years[0]}–{self.years[-1]})"
            )


def _extract_zip(zip_path: str | Path, extract_dir: str | Path) -> Path:
    """Extract a zip file, returning the directory containing the TIFs."""
    zip_path    = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # If the zip contained a single subfolder, descend into it
    entries = [p for p in extract_dir.iterdir() if not p.name.startswith("__MACOSX")]
    tif_dirs = [extract_dir] + [p for p in entries if p.is_dir()]
    for d in tif_dirs:
        if list(d.glob("*.tif")) or list(d.glob("*.tiff")):
            return d

    return extract_dir


def _auto_detect_years(tif_dir: Path) -> dict[int, Path]:
    """
    Scan a directory for TIFs with a 4-digit year in the filename.
    Returns dict year → filepath. Raises if no years found.
    """
    year_to_path: dict[int, Path] = {}
    for f in sorted(tif_dir.glob("*.tif")) + sorted(tif_dir.glob("*.tiff")):
        match = _YEAR_PATTERN.search(f.stem)
        if match:
            year = int(match.group())
            year_to_path[year] = f

    if not year_to_path:
        raise ValueError(
            f"No TIFs with a detectable 4-digit year found in {tif_dir}. "
            "Use periods= for manual specification instead."
        )

    return year_to_path


def load_lulc_timeseries(
    zip_path:    str | Path,
    extract_dir: str | Path = "/tmp/strategicc_calibration",
    periods:     list[tuple[int, int, int]] | None = None,
    fill_gaps:   bool = True,
) -> LULCTimeSeries:
    """
    Extract a zip of yearly LULC rasters and build an annual stack.

    Parameters
    ----------
    zip_path    : path to a .zip containing yearly LULC GeoTIFFs
    extract_dir : where to extract the zip contents
    periods     : optional manual override — list of
                  (start_year, end_year, filename_year) tuples.
                  If provided, AUTO-detection is skipped entirely and the
                  filename_year raster is repeated for every year in
                  [start_year, end_year], matching the original script's
                  multi-year-composite expansion behaviour.
    fill_gaps   : if True (AUTO mode only), missing years between the
                  earliest and latest detected year are forward-filled
                  from the most recent prior year, with a printed warning.
                  If False, missing years raise an error.

    Returns
    -------
    LULCTimeSeries
    """
    tif_dir = _extract_zip(zip_path, extract_dir)

    profile = None
    annual_maps: list[np.ndarray] = []
    annual_years: list[int] = []

    if periods is not None:
        # ── MANUAL mode ─────────────────────────────────────────────────
        print(f"  [Manual mode] {len(periods)} period(s) specified")
        for start, end, fname_year in periods:
            # Find a file matching this year (flexible: "2010.tif", "lulc_2010.tif")
            candidates = list(tif_dir.glob(f"*{fname_year}*.tif"))
            if not candidates:
                raise FileNotFoundError(
                    f"No TIF found for filename_year={fname_year} in {tif_dir}"
                )
            path = candidates[0]
            with rasterio.open(path) as src:
                data = src.read(1)
                if profile is None:
                    profile = src.profile.copy()

            for year in range(start, end + 1):
                annual_maps.append(data)
                annual_years.append(year)

    else:
        # ── AUTO mode ───────────────────────────────────────────────────
        year_to_path = _auto_detect_years(tif_dir)
        detected_years = sorted(year_to_path.keys())
        print(f"  [Auto mode] Detected {len(detected_years)} year(s): "
              f"{detected_years[0]}–{detected_years[-1]}")

        full_range = list(range(detected_years[0], detected_years[-1] + 1))
        missing    = [y for y in full_range if y not in year_to_path]

        if missing and not fill_gaps:
            raise ValueError(
                f"Missing years in timeseries: {missing}. "
                "Set fill_gaps=True to forward-fill, or supply periods= manually."
            )
        if missing:
            print(f"  [Warning] {len(missing)} missing year(s) forward-filled: "
                  f"{missing}")

        last_data = None
        for year in full_range:
            if year in year_to_path:
                path = year_to_path[year]
                with rasterio.open(path) as src:
                    data = src.read(1)
                    if profile is None:
                        profile = src.profile.copy()
                last_data = data
            else:
                if last_data is None:
                    raise ValueError(
                        f"Cannot forward-fill year {year} — no prior data available"
                    )
                data = last_data

            annual_maps.append(data)
            annual_years.append(year)

    stack = np.array(annual_maps, dtype=np.uint8)
    print(f"  Stack built: {annual_years[0]}–{annual_years[-1]}  "
          f"({len(annual_years)} annual layers, shape={stack.shape[1:]})")

    return LULCTimeSeries(stack=stack, years=annual_years, profile=profile)


def extract_lulc_zip_to_folder(
    zip_path:     str | Path,
    extract_dir:  str | Path,
    force:        bool = False,
) -> dict[int, Path]:
    """
    Extract an ENTIRE historical LULC zip into a persistent folder (v3.4),
    rather than a temporary scratch directory. All years become available
    on disk for reuse — by the calibration module, by repeated initial-state
    selection with different years, or for manual inspection — without
    re-extracting the zip each time.

    Parameters
    ----------
    zip_path    : path to the historical LULC zip (AUTO-detected filenames)
    extract_dir : persistent destination folder (e.g. inputs/lulc_annual/).
                  Created if it doesn't exist.
    force       : if True, re-extract even if the folder already appears
                  populated with detectable yearly TIFs

    Returns
    -------
    dict[year, Path] — same mapping _auto_detect_years() would produce,
    now pointing at files inside the persistent extract_dir
    """
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        try:
            existing = _auto_detect_years(extract_dir)
        except ValueError:
            existing = {}   # empty/unpopulated folder — not yet extracted
        if existing:
            print(f"  [Cache hit] '{extract_dir}' already contains "
                  f"{len(existing)} detectable year(s) — skipping "
                  f"re-extraction. Pass force=True to re-extract.")
            return existing

    tif_dir = _extract_zip(zip_path, extract_dir)
    year_to_path = _auto_detect_years(tif_dir)

    print(f"  Extracted {len(year_to_path)} year(s) from '{zip_path}' "
          f"-> persistent folder '{extract_dir}'")
    return year_to_path


def extract_initial_state_class(
    zip_path:     str | Path,
    year:         int,
    extract_dir:  str | Path,
    force:        bool = False,
) -> Path:
    """
    Select one year's LULC raster as the simulation's initial state class
    raster, extracting the FULL historical zip to a persistent folder if
    not already extracted (v3.4 — changed from single-file caching to
    full-folder extraction, so all years remain available afterward).

    Parameters
    ----------
    zip_path    : path to the historical LULC zip
    year        : the year to use as the initial state class raster
    extract_dir : persistent folder for the full annual extraction
                  (e.g. inputs/lulc_annual/)
    force       : if True, re-extract the zip even if extract_dir already
                  appears populated

    Returns
    -------
    Path to that year's raster within extract_dir

    Raises
    ------
    ValueError : if `year` is not found in the zip's detected year range
    """
    year_to_path = extract_lulc_zip_to_folder(zip_path, extract_dir, force=force)

    if year not in year_to_path:
        available = sorted(year_to_path.keys())
        raise ValueError(
            f"Year {year} not found in zip '{zip_path}'. "
            f"Available years: {available[0]}–{available[-1]} "
            f"({len(available)} total)."
        )

    selected = year_to_path[year]
    print(f"  Initial state class for year {year}: '{selected}'")
    return selected
