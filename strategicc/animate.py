"""
strategicc/animate.py  —  v3.5
---------------------------------
Standalone animation function. Produces a two-panel GIF/MP4:
  - LEFT panel  : modal LULC map per timestep
  - RIGHT panel : a selectable statistics line chart, synced frame-by-frame
                  with the map (or omitted entirely if panel=None)

Not part of the standard run.py pipeline — call manually after a
simulation completes:

    from strategicc import animate
    animate(out_dir="strategicc_output/", panel="value_per_class")

Encoding uses matplotlib's built-in writers only:
  - GIF -> PillowWriter
  - MP4 -> FFMpegWriter (requires the ffmpeg binary on the system)

Historical years (from a calibrated LULCTimeSeries) can be prepended to
the simulated timeline so the animation shows real past + simulated
future as one continuous sequence.
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
    spatial_dir = summary_dir / "spatial"
    frames: dict[int, np.ndarray] = {}
    for year in years:
        tif = spatial_dir / f"lulc_mean_{year}.tif"
        if tif.exists():
            frames[year] = np.array(Image.open(str(tif)), dtype=np.uint8)
    return frames


def _build_panel_series(
    panel:       str | None,
    summary_dir: Path,
) -> pd.DataFrame | None:
    """
    Build the right-panel data series for simulated years, in a unified
    long-format schema: columns = year, class_name, value.
    """
    if panel is None:
        return None

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
        if not path.exists():
            print(f"  [Warning] {path} not found — right panel will be empty.")
            return pd.DataFrame(columns=["year", "class_name", "value"])
        df = pd.read_csv(path)
        acol = next((c for c in df.columns if c.startswith("area_")), None)
        if acol is None:
            return pd.DataFrame(columns=["year", "class_name", "value"])
        return df.rename(columns={acol: "value"})[["year", "class_name", "value"]]

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
                    years are PREPENDED to the simulated timeline
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
    spatial_dir = summary_dir / "spatial"

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

    panel_df = _build_panel_series(panel, summary_dir)

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
