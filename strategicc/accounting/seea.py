"""
strategicc/accounting/seea.py  —  SEEA-EA accounting engine  v2.2
------------------------------------------------------------------
Produces all ecosystem accounts from simulation outputs.

v2.2 changes
------------
* Accepts area_modal_df — derived from modal LULC maps — as the primary
  area input for all accounts. This ensures full consistency between the
  spatial (modal raster) and tabular (SEEA) representations.
* area_df (raw per-iteration) is retained only for the uncertainty summary.
* Area unit is detected automatically from column name (area_ha/km2/px).

Accounts produced
-----------------
1. Extent account        — area per class per year (modal)
2. Transition matrix     — area converted between classes + value change
3. Physical flow account — total physical units supplied per service per year
4. Monetary flow account — total monetary value per service per year
5. Change-in-value       — year-on-year change in total ecosystem value
6. Uncertainty summary   — min/max range across iterations (raw area_df)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc.io.csv_loader import StateClass
from strategicc.accounting.csv_loader import EcosystemService


def _area_col(df: pd.DataFrame) -> str:
    """Detect area column name (area_ha / area_km2 / area_px)."""
    for col in df.columns:
        if col.startswith("area_"):
            return col
    raise ValueError(
        f"No area column found. Expected area_ha, area_km2, or area_px. "
        f"Got: {list(df.columns)}"
    )


def _unit_label(col: str) -> str:
    """'area_ha' → 'ha', 'area_km2' → 'km²', 'area_px' → 'pixels'"""
    mapping = {"ha": "ha", "km2": "km²", "px": "pixels"}
    key = col.replace("area_", "")
    return mapping.get(key, key)


class SEEAAccount:
    """
    SEEA-EA ecosystem accounting engine  v2.2

    Parameters
    ----------
    area_modal_df : area table derived from modal LULC maps — used for all
                    accounts. Schema: year, class_id, class_name, area_{unit}
                    Produced by outputs.modal_to_area_table().

    area_df       : raw per-iteration area table — used ONLY for the
                    uncertainty summary. Schema adds an 'iteration' column.
                    Pass None to skip uncertainty summary.

    trans_df      : concatenated transition_log.csv across all iterations.
                    Used for transition matrix (median counts across iters).

    services      : list of EcosystemService from EcosystemServices.csv

    classes       : dict[int, StateClass]

    px_area       : pixel area in the chosen unit (engine.px_area).
                    Used for transition matrix area calculation.
    """

    def __init__(
        self,
        area_modal_df: pd.DataFrame,
        trans_df:      pd.DataFrame,
        services:      list[EcosystemService],
        classes:       dict[int, StateClass],
        px_area:       float,
        area_df:       pd.DataFrame | None = None,   # raw — for uncertainty only
    ) -> None:
        self.area_modal_df = area_modal_df
        self.trans_df      = trans_df
        self.services      = services
        self.classes       = classes
        self.px_area       = px_area
        self.area_df       = area_df   # may be None

        # Detect area column and unit label from modal df
        self._acol       = _area_col(area_modal_df)
        self._unit_label = _unit_label(self._acol)

        # Build service lookup: class_name → list of services
        self._svc_by_class: dict[str, list[EcosystemService]] = {}
        for svc in services:
            self._svc_by_class.setdefault(svc.state_class, []).append(svc)

        self._years = sorted(area_modal_df["year"].unique())

    # ── Public API ────────────────────────────────────────────────────────────

    def extent_account(self) -> pd.DataFrame:
        """
        Ecosystem extent account.
        Rows: year. Columns: one per class (area in chosen unit).
        Derived from modal LULC maps — spatially consistent.
        """
        pivot = self.area_modal_df.pivot_table(
            index="year", columns="class_name",
            values=self._acol, aggfunc="sum"
        ).fillna(0)
        pivot.columns.name = None
        pivot.index.name   = f"Year"
        pivot.attrs["unit"] = self._unit_label
        return pivot

    def transition_matrix(self) -> pd.DataFrame:
        """
        Ecosystem extent change matrix.
        Shows median area converted from each class (rows) to each class (cols)
        aggregated across all timesteps.
        """
        if self.trans_df.empty:
            return pd.DataFrame()

        class_names = [sc.name for sc in self.classes.values()]

        counts = (
            self.trans_df.groupby(["iteration", "from_class", "to_class"])
            .size()
            .reset_index(name="n_cells")
        )
        median_counts = (
            counts.groupby(["from_class", "to_class"])["n_cells"]
            .median()
            .reset_index()
        )
        median_counts["area"] = median_counts["n_cells"] * self.px_area

        matrix = median_counts.pivot_table(
            index="from_class", columns="to_class",
            values="area", aggfunc="sum"
        ).reindex(index=class_names, columns=class_names).fillna(0)
        matrix.index.name   = f"From \\ To ({self._unit_label})"
        matrix.columns.name = None
        return matrix

    def value_change_matrix(self) -> pd.DataFrame:
        """
        Monetary value change from transitions.
        Each cell = median area converted × (value_per_unit_dest − value_per_unit_src).
        """
        tm = self.transition_matrix()
        if tm.empty:
            return pd.DataFrame()

        total_val: dict[str, float] = {}
        for sc in self.classes.values():
            svcs = self._svc_by_class.get(sc.name, [])
            total_val[sc.name] = sum(s.value_per_ha for s in svcs)

        val_matrix = pd.DataFrame(0.0, index=tm.index, columns=tm.columns)
        for from_cls in tm.index:
            for to_cls in tm.columns:
                area = tm.loc[from_cls, to_cls]
                if area > 0:
                    delta = total_val.get(to_cls, 0) - total_val.get(from_cls, 0)
                    val_matrix.loc[from_cls, to_cls] = area * delta

        val_matrix.index.name   = "From \\ To (currency)"
        val_matrix.columns.name = None
        return val_matrix

    def physical_flow_account(self) -> pd.DataFrame | None:
        """
        Physical ecosystem service flow account (Mode B only).
        Rows: year. Columns: (service_type, service_name, unit). Values: total quantity.
        """
        mode_b = [s for s in self.services if s.has_physical]
        if not mode_b:
            return None

        records = []
        for _, row in self.area_modal_df.iterrows():
            svcs = [s for s in self._svc_by_class.get(row["class_name"], [])
                    if s.has_physical]
            for svc in svcs:
                records.append({
                    "year":         row["year"],
                    "class":        row["class_name"],
                    "service_type": svc.service_type,
                    "service_name": svc.service_name,
                    "unit":         svc.physical_unit,
                    "flow":         svc.physical_per_ha * row[self._acol],
                })

        df = pd.DataFrame(records)
        pivot = df.pivot_table(
            index="year",
            columns=["service_type", "service_name", "unit"],
            values="flow",
            aggfunc="sum",
        ).fillna(0)
        pivot.index.name = "Year"
        return pivot

    def monetary_flow_account(self) -> pd.DataFrame:
        """
        Monetary ecosystem service flow account.
        Rows: year. Columns: (service_type, service_name). Values: total value.
        Note: ValuePerHa in EcosystemServices.csv is treated as value per
        area unit — if AREA_UNIT='km2', ensure values are per km².
        """
        records = []
        for _, row in self.area_modal_df.iterrows():
            for svc in self._svc_by_class.get(row["class_name"], []):
                records.append({
                    "year":         row["year"],
                    "class":        row["class_name"],
                    "service_type": svc.service_type,
                    "service_name": svc.service_name,
                    "currency":     svc.currency,
                    "value":        svc.value_per_ha * row[self._acol],
                })

        df = pd.DataFrame(records)
        pivot = df.pivot_table(
            index="year",
            columns=["service_type", "service_name"],
            values="value",
            aggfunc="sum",
        ).fillna(0)
        pivot.index.name = "Year"
        return pivot

    def total_value_by_class(self) -> pd.DataFrame:
        """Total monetary value per class per year. Used for stacked area plots."""
        records = []
        for _, row in self.area_modal_df.iterrows():
            svcs  = self._svc_by_class.get(row["class_name"], [])
            total = sum(s.value_per_ha for s in svcs) * row[self._acol]
            records.append({
                "year":  row["year"],
                "class": row["class_name"],
                "value": total,
            })
        df = pd.DataFrame(records)
        pivot = df.pivot_table(
            index="year", columns="class", values="value", aggfunc="sum"
        ).fillna(0)
        pivot.index.name   = "Year"
        pivot.columns.name = None
        return pivot

    def change_in_value(self) -> pd.DataFrame:
        """Year-on-year change in total ecosystem service value."""
        tv    = self.total_value_by_class()
        delta = tv.diff()
        delta["Total"] = delta.sum(axis=1)
        delta.index.name = "Year"
        return delta

    def uncertainty_summary(self) -> pd.DataFrame | None:
        """
        Min/max range of total ecosystem value across iterations.
        Returns None if area_df (raw) was not provided.
        """
        if self.area_df is None or self.area_df.empty:
            return None

        raw_acol = _area_col(self.area_df)
        records  = []
        for (iteration, year, class_name), grp in self.area_df.groupby(
            ["iteration", "year", "class_name"]
        ):
            area = grp[raw_acol].sum()
            svcs = self._svc_by_class.get(class_name, [])
            val  = sum(s.value_per_ha for s in svcs) * area
            records.append({"iteration": iteration, "year": year, "value": val})

        df    = pd.DataFrame(records)
        stats = (
            df.groupby("year")["value"]
            .agg(median="median", min="min", max="max")
            .reset_index()
        )
        stats["range_pct"] = (
            (stats["max"] - stats["min"]) / stats["median"].replace(0, np.nan) * 100
        ).round(1)
        stats.columns = ["Year", "Median value", "Min value", "Max value", "Range (%)"]
        return stats
