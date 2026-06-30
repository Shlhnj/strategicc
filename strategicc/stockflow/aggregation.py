"""
strategicc/stockflow/aggregation.py  —  v3.2
-------------------------------------------------
Aggregates per-iteration Stock & Flow outputs across iterations and by
state class, producing per-class-per-year stock and flow totals suitable
for Mode C SEEA-EA valuation.

For each timestep: load that stock type's raster from every iteration,
compute the per-cell MEDIAN across iterations, mask by the modal LULC
class for that timestep, sum within each class's cells.

Flow totals are aggregated from flow_log.csv (median total per flow_type
per year across iterations) — flows are scalar aggregates already, not
per-cell rasters, so no class masking is needed (eligibility was already
class-gated by the engine at simulation time via FromStateClassId).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from strategicc.io.csv_loader import StateClass


def aggregate_stock_by_class(
    iter_dirs:   list[Path],
    stock_types: list[str],
    classes:     dict[int, StateClass],
    modal_maps:  dict[int, np.ndarray],
    start_year:  int,
    n_timesteps: int,
) -> pd.DataFrame:
    """
    Aggregate per-iteration stock rasters into per-class-per-year totals.

    Returns
    -------
    DataFrame with columns: year, class_id, class_name, stock_type, total
    """
    rows = []
    total_steps = n_timesteps + 1

    for stock_type in stock_types:
        for t in range(total_steps):
            year = start_year + t
            if year not in modal_maps:
                continue
            modal = modal_maps[year]

            stack = []
            for d in iter_dirs:
                tif = d / "stocks" / stock_type / f"stock_{year}.tif"
                if tif.exists():
                    arr = np.array(Image.open(str(tif)), dtype=np.float32)
                    stack.append(arr)

            if not stack:
                continue

            cube   = np.stack(stack, axis=0)
            median = np.median(cube, axis=0)

            for cid, sc in classes.items():
                class_mask = (modal == cid)
                total = float(median[class_mask].sum())
                rows.append({
                    "year":       year,
                    "class_id":   cid,
                    "class_name": sc.name,
                    "stock_type": stock_type,
                    "total":      total,
                })

    return pd.DataFrame(rows)


def aggregate_flow_by_class(
    iter_dirs: list[Path],
) -> pd.DataFrame:
    """
    Aggregate per-iteration per-class flow logs into median
    total-per-class-per-flow-type-per-year.

    Reads flow_log_by_class.csv (v3.2) from each iteration directory.

    Returns
    -------
    DataFrame with columns: year, class_name, flow_type, total
    (median across iterations)
    """
    frames = []
    for d in iter_dirs:
        log_path = d / "flow_log_by_class.csv"
        if log_path.exists():
            df = pd.read_csv(log_path)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["year", "class_name", "flow_type", "total"])

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["iteration", "year", "class_name", "flow_type"])["amount"]
        .sum()
        .reset_index()
    )
    median = (
        grouped.groupby(["year", "class_name", "flow_type"])["amount"]
        .median()
        .reset_index()
        .rename(columns={"amount": "total"})
    )
    return median


def aggregate_flow_by_type(
    iter_dirs: list[Path],
) -> pd.DataFrame:
    """
    Aggregate per-iteration flow logs into median total-per-flow-type-per-year
    (landscape-wide, no class breakdown). Kept for diagnostic/summary use;
    Mode C SEEA valuation uses aggregate_flow_by_class() instead.

    Returns
    -------
    DataFrame with columns: year, flow_type, total (median across iterations)
    """
    frames = []
    for d in iter_dirs:
        log_path = d / "flow_log.csv"
        if log_path.exists():
            frames.append(pd.read_csv(log_path))

    if not frames:
        return pd.DataFrame(columns=["year", "flow_type", "total"])

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["iteration", "year", "flow_type"])["total_amount"]
        .sum()
        .reset_index()
    )
    median = (
        grouped.groupby(["year", "flow_type"])["total_amount"]
        .median()
        .reset_index()
        .rename(columns={"total_amount": "total"})
    )
    return median
