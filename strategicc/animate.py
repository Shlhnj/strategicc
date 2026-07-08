"""
strategicc/animate.py -- v3.12.4
---------------------------------
Standalone animation function. Produces a two-panel GIF/MP4:
  - LEFT panel  : modal LULC map per timestep
  - RIGHT panel : a selectable statistics line chart, synced frame-by-frame
                  with the map (or omitted entirely if panel=None)

Not part of the standard run.py pipeline -- call manually after a
simulation completes:

    from strategicc import animate
    animate(out_dir="strategicc_output/", panel="value_per_class")

Encoding uses matplotlib's built-in writers only:
  - GIF -> PillowWriter
  - MP4 -> FFMpegWriter (requires the ffmpeg binary on the system)

Historical years (from a calibrated LULCTimeSeries) can be prepended to
the simulated timeline so the animation shows real past + simulated
future as one continuous sequence.

Historical LEFT panel vs. RIGHT panel coverage (v3.12.4)
---------------------------------------------------------
Prior to v3.12.4, `historical_ts` only extended the LEFT panel (the map):
historical years appeared as real classified rasters, but the RIGHT panel
line chart only ever plotted simulated years -- historical years on the
map had no corresponding point on the line.

As of v3.12.4, `panel="area_per_class"` -- the one panel type with a real
historical equivalent (observed classified-pixel area, via
`strategicc.validation.compute_observed_extent()`) -- draws its
historical side from `historical_ts` when supplied, concatenated with the
simulated `area_modal.csv` series. This requires `px_area_ha` to also be
supplied -- see `animate()`'s docstring (state classes are already loaded
internally from `cfg.STATE_CLASSES_CSV`).

Every other panel type (`value_per_class`, `value_total`,
`transitions_out`, `transitions_in`) has no historical equivalent to draw
from (ecosystem value and transition counts are simulation-only
quantities), so if `historical_ts` is supplied alongside one of those, an
explicit warning is printed rather than silently omitting the historical
line segment. The LEFT panel map still shows historical frames in that
case -- only the RIGHT panel is simulated-years-only.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from PIL import Image

from strategicc.io.csv_loader import StateClass
from strategicc.outputs import _build_cmap, _class_rgb


_VALID_PANELS = {
    None, "value_per_class", "value_total", "area_per_class",
    "transitions_out", "transitions_in",
}


def _load_modal_frames(
    summary_dir: Path,
    years:       list[int],
) -> dict[int, np.ndarray]:
    """Load lulc_mean_YYYY.tif for each requested year."""
    spatial_dir = summary_dir
    frames: dict[int, np.ndarray] = {}
    for year in years:
        tif = spatial_dir / f"lulc_mean_{year}.tif"
        if tif.exists():
            frames[year] = np.array(Image.open(str(tif)), dtype=np.uint8)
    return frames


def _build_panel_series(
    panel:            str | None,
    summary_dir:      Path,
    historical_ts     = None,
    historical_years: list[int] | None = None,
    classes:          dict | None = None,
    px_area_ha:       float | None = None,
) -> pd.DataFrame | None:
    """
    Build the right-panel data series, in a unified long-format schema:
    columns = year, class_name, value.

    For simulated years this is always drawn from the run's summary
    outputs. For historical years, only `panel="area_per_class"` has a
    real historical equivalent (observed classified-pixel area) -- that
    case draws its historical rows from `historical_ts` via
    `compute_observed_extent()` and concatenates them with the simulated
    rows. `historical_ts`, `historical_years`, `classes`, and
    `px_area_ha` are only used by that branch.

    Every other panel type has no historical equivalent, so if
    `historical_ts` is supplied alongside one of them, this prints an
    explicit warning rather than silently only showing simulated years.
    """
    if panel is None:
        return None

    if (
        historical_ts is not None
        and historical_years
        and panel != "area_per_class"
    ):
        print(
            f"  [Warning] historical_ts was supplied but panel='{panel}' has "
            f"no historical equivalent to draw from -- the right panel will "
            f"only show simulated years. The left-panel map will still "
            f"include historical frames. Only panel='area_per_class' "
            f"supports a historical overlay on the right panel."
        )

    if panel == "value_per_class":
        path = summary_dir.parent / "seea" / "seea_total_value_by_class.csv"
        if not path.exists():
            print(f"  [Warning] {path} not found — right panel will be empty.")
            return pd.DataFrame(columns=["year", "class_name", "value"])
        wide = pd.read_csv(path, index_col=0)
        idx_name = wide.index.name or "Year"
        long = wide.reset_index().melt(
            id_vars=idx_name, var_name="class_name", value_name="value",
        ).rename(columns={idx_name: "year"})
        return long

    if panel == "value_total":
        path = summary_dir.parent / "seea" / "seea_total_value_by_class.csv"
        if not path.exists():
            print(f"  [Warning] {path} not found — right panel will be empty.")
            return pd.DataFrame(columns=["year", "class_name", "value"])
        wide = pd.read_csv(path, index_col=0)
        total = wide.sum(axis=1).reset_index()
        total.columns = ["year", "value"]
        total["class_name"] = "Total"
        return total[["year", "class_name", "value"]]

    if panel == "area_per_class":
        path = summary_dir / "area_modal.csv"
        sim_unit: str | None = None
        if not path.exists():
            print(f"  [Warning] {path} not found — right panel will be empty.")
            sim_df = pd.DataFrame(columns=["year", "class_name", "value"])
        else:
            df = pd.read_csv(path)
            acol = next((c for c in df.columns if c.startswith("area_")), None)
            if acol is None:
                sim_df = pd.DataFrame(columns=["year", "class_name", "value"])
            else:
                sim_unit = acol[len("area_"):]  # "ha" / "km2" / "px"
                sim_df = df.rename(columns={acol: "value"})[
                    ["year", "class_name", "value"]
                ]

        if historical_ts is None or not historical_years:
            return sim_df

        if classes is None or px_area_ha is None:
            raise ValueError(
                "animate(panel='area_per_class', historical_ts=...) requires "
                "'px_area_ha' so the historical side of the right panel can "
                "be computed via compute_observed_extent() (state classes "
                "are already loaded internally from cfg.STATE_CLASSES_CSV). "
                "Pass px_area_ha=engine.px_area_ha (or the equivalent used "
                "for the original run)."
            )

        from strategicc.validation.extent import compute_observed_extent

        obs_df = compute_observed_extent(historical_ts, classes, px_area_ha)
        obs_df = obs_df[obs_df["year"].isin(historical_years)]
        obs_df = obs_df.rename(columns={"area_ha": "value"})[
            ["year", "class_name", "value"]
        ]

        if obs_df.empty:
            return sim_df

        # compute_observed_extent() always returns hectares. Convert to
        # whatever unit area_modal.csv actually used (read from its own
        # "area_*" column name — the ground truth for the run, rather
        # than trusting cfg.AREA_UNIT to still match what the run used)
        # so the two series share one axis instead of silently mixing
        # units. Same px_area_ha is assumed for both rasters (reasonable:
        # they're the same study area at the same resolution).
        if sim_unit is not None and sim_unit != "ha":
            if sim_unit == "px":
                obs_df = obs_df.assign(value=obs_df["value"] / px_area_ha)
            else:
                from strategicc.io.raster import get_pixel_area
                factor = get_pixel_area(1.0, sim_unit)
                obs_df = obs_df.assign(value=obs_df["value"] * factor)
        return pd.concat([obs_df, sim_df], ignore_index=True)

    if panel in ("transitions_out", "transitions_in"):
        path = summary_dir / "transitions_all_iterations.csv"
        if not path.exists():
            print(f"  [Warning] {path} not found — right panel will be empty.")
            return pd.DataFrame(columns=["year", "class_name", "value"])
        df = pd.read_csv(path)
        group_col = "from_class" if panel == "transitions_out" else "to_class"
        counts = (
            df.groupby(["iteration", "year", group_col])
            .size()
            .reset_index(name="count")
        )
        median = (
            counts.groupby(["year", group_col])["count"]
            .median()
            .reset_index()
            .rename(columns={group_col: "class_name", "count": "value"})
        )
        return median

    raise ValueError(
        f"Unknown panel option '{panel}'. Valid options: "
        f"{sorted(p for p in _VALID_PANELS if p is not None)} or None"
    )


def _panel_label(panel: str | None) -> str:
    return {
        None:               "",
        "value_per_class":  "Ecosystem Value by Class",
        "value_total":      "Total Ecosystem Value",
        "area_per_class":   "Area by Class",
        "transitions_out":  "Transitions Out (median, per class)",
        "transitions_in":   "Transitions In (median, per class)",
    }[panel]


def animate(
    out_dir:        str | Path,
    panel:          str | None = "value_per_class",
    start_year:     int | None = None,
    end_year:       int | None = None,
    frame_rate:     int = 2,
    output_format:  str = "gif",
    output_path:    str | Path | None = None,
    historical_ts   = None,
    px_area_ha:     float | None = None,
    figsize:        tuple[float, float] = (14, 6),
) -> Path:
    """
    Render a two-panel animation: LULC map (left) + statistics (right).

    Parameters
    ----------
    out_dir       : the simulation's output directory
    panel         : right-panel content, or None for map-only animation.
                    One of: "value_per_class", "value_total",
                    "area_per_class", "transitions_out", "transitions_in", None
    start_year    : first year to animate (defaults to earliest available,
                    including historical_ts years if supplied)
    end_year      : last year to animate (defaults to latest simulated year)
    frame_rate    : frames per second
    output_format : "gif" or "mp4"
    output_path   : defaults to {out_dir}/animation.{format}
    historical_ts : optional LULCTimeSeries (from
                    strategicc.calibration.load_lulc_timeseries()) whose
                    years are PREPENDED to the simulated timeline for the
                    LEFT panel map. As of v3.12.4, if panel="area_per_class"
                    these years are ALSO drawn onto the RIGHT panel line
                    chart (via compute_observed_extent()) -- see
                    'px_area_ha' below, which is required for that case.
                    For every other panel type, historical years appear
                    on the map only; a warning is printed to say so.
    px_area_ha    : pixel area in hectares (e.g. engine.px_area_ha).
                    Required only when both historical_ts is supplied AND
                    panel="area_per_class", so the historical side of the
                    right panel can be computed. Ignored otherwise.
                    compute_observed_extent() always computes in hectares
                    internally; the result is automatically converted to
                    match whichever unit area_modal.csv actually used
                    (read from its own "area_*" column name), so the two
                    series share one axis regardless of AREA_UNIT.
    figsize       : matplotlib figure size in inches

    Returns
    -------
    Path to the saved animation file.
    """
    if panel not in _VALID_PANELS:
        raise ValueError(
            f"Unknown panel option '{panel}'. Valid options: "
            f"{sorted(p for p in _VALID_PANELS if p is not None)} or None"
        )
    if output_format not in ("gif", "mp4"):
        raise ValueError(f"output_format must be 'gif' or 'mp4', got '{output_format}'")

    out_dir     = Path(out_dir)
    summary_dir = out_dir / "summary"
    spatial_dir = summary_dir

    from strategicc.io.csv_loader import load_state_classes
    import strategicc.config as cfg
    try:
        classes = load_state_classes(cfg.STATE_CLASSES_CSV)
    except Exception as e:
        raise RuntimeError(
            "Could not load state classes for the animation legend/colours. "
            "Ensure strategicc.config.STATE_CLASSES_CSV still points at a "
            "valid StateClasses.csv (same one used for the simulation run)."
        ) from e

    sim_years = sorted(
        int(p.stem.replace("lulc_mean_", ""))
        for p in spatial_dir.glob("lulc_mean_*.tif")
    )
    if not sim_years:
        raise ValueError(
            f"No lulc_mean_*.tif files found in '{spatial_dir}'. "
            f"Run outputs.aggregate_spatial() first."
        )

    historical_years: list[int] = []
    historical_frames: dict[int, np.ndarray] = {}
    if historical_ts is not None:
        historical_years = [y for y in historical_ts.years if y < sim_years[0]]
        for y in historical_years:
            historical_frames[y] = historical_ts.stack[historical_ts.year_index(y)]

    all_years = sorted(set(historical_years) | set(sim_years))

    lo = start_year if start_year is not None else all_years[0]
    hi = end_year   if end_year   is not None else all_years[-1]
    frame_years = [y for y in all_years if lo <= y <= hi]

    if not frame_years:
        raise ValueError(
            f"No years available in range [{lo}, {hi}]. "
            f"Available years: {all_years[0]}–{all_years[-1]}"
        )

    n_hist_in_range = len([y for y in frame_years if y in historical_years])
    print(f"  Animating {len(frame_years)} frame(s): "
          f"{frame_years[0]}–{frame_years[-1]} "
          f"({n_hist_in_range} historical + "
          f"{len(frame_years) - n_hist_in_range} simulated)")

    sim_frames = _load_modal_frames(summary_dir, [y for y in frame_years if y in sim_years])
    map_frames: dict[int, np.ndarray] = {**historical_frames, **sim_frames}
    map_frames = {y: map_frames[y] for y in frame_years if y in map_frames}

    if not map_frames:
        raise ValueError("No map frames could be loaded for the requested year range.")

    panel_df = _build_panel_series(
        panel, summary_dir,
        historical_ts=historical_ts,
        historical_years=historical_years,
        classes=classes,
        px_area_ha=px_area_ha,
    )

    cmap   = _build_cmap(classes)
    max_id = max(classes.keys())
    has_panel = panel is not None and panel_df is not None and not panel_df.empty

    if has_panel:
        fig, (ax_map, ax_panel) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.2]}
        )
    else:
        fig, ax_map = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]))
        ax_panel = None

    cursor_artists = []
    if has_panel:
        class_names = sorted(panel_df["class_name"].unique())
        color_lookup = {sc.name: _class_rgb(sc) for sc in classes.values()}
        color_lookup["Total"] = (0.2, 0.2, 0.2)

        for cname in class_names:
            sub = panel_df[panel_df["class_name"] == cname].sort_values("year")
            color = color_lookup.get(cname, (0.5, 0.5, 0.5))
            ax_panel.plot(sub["year"], sub["value"], color=color,
                         linewidth=1.5, alpha=0.85, label=cname, zorder=2)

        ax_panel.set_xlabel("Year", fontsize=10)
        ax_panel.set_ylabel(_panel_label(panel), fontsize=10)
        ax_panel.set_title(_panel_label(panel), fontsize=11)
        ax_panel.legend(loc="upper left", fontsize=7, framealpha=0.85, ncol=2)
        ax_panel.grid(True, alpha=0.25)

        cursor_line = ax_panel.axvline(frame_years[0], color="red",
                                       linewidth=1.5, alpha=0.7, zorder=3)
        cursor_artists.append(cursor_line)

    im_artist = ax_map.imshow(
        map_frames[frame_years[0]], cmap=cmap, vmin=0, vmax=max_id,
        interpolation="nearest"
    )
    ax_map.axis("off")
    title_artist = ax_map.set_title(f"Year {frame_years[0]}", fontsize=12, fontweight="bold")

    plt.tight_layout()

    def _update(frame_idx: int):
        year = frame_years[frame_idx]
        artists = []

        if year in map_frames:
            im_artist.set_data(map_frames[year])
        is_historical = year in historical_years and year not in sim_years
        title_artist.set_text(
            f"Year {year}" + ("  (historical)" if is_historical else "")
        )
        artists.append(im_artist)
        artists.append(title_artist)

        if has_panel and cursor_artists:
            cursor_artists[0].set_xdata([year, year])
            artists.append(cursor_artists[0])

        return artists

    anim = FuncAnimation(
        fig, _update, frames=len(frame_years),
        interval=1000 / max(frame_rate, 1), blit=False,
    )

    if output_path is None:
        output_path = out_dir / f"animation.{output_format}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "gif":
        writer = PillowWriter(fps=frame_rate)
    else:
        try:
            writer = FFMpegWriter(fps=frame_rate)
        except FileNotFoundError as e:
            raise RuntimeError(
                "MP4 output requires the ffmpeg binary to be installed and "
                "available on PATH. Install it (e.g. `apt-get install ffmpeg` "
                "or `conda install ffmpeg`) or use output_format='gif' instead."
            ) from e

    anim.save(str(output_path), writer=writer)
    plt.close(fig)

    print(f"  Animation saved to: {output_path}")
    return output_path
