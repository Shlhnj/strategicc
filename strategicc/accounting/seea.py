"""
strategicc/accounting/seea.py  —  SEEA-EA accounting engine  v2.0
------------------------------------------------------------------
Produces all ecosystem accounts from simulation outputs.

Accounts produced
-----------------
1. Extent account        — area (ha) per class per year
2. Transition matrix     — area converted between classes + value change
3. Physical flow account — total physical units supplied per service per year
                           (Mode B only, where PhysicalUnit is specified)
4. Monetary flow account — total monetary value per service per year
5. Change-in-value       — year-on-year change in total ecosystem value
6. Uncertainty summary   — min/max range across iterations (reported once)

All accounts use the median area across iterations as the central estimate,
with min/max noted in the uncertainty summary.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from strategicc.io.csv_loader import StateClass
from strategicc.accounting.csv_loader import EcosystemService


class SEEAAccount:
    """
    SEEA-EA ecosystem accounting engine.

    Parameters
    ----------
    area_df      : concatenated area_table.csv across all iterations
                   (columns: iteration, year, class_id, class_name, area_ha)
    trans_df     : concatenated transition_log.csv across all iterations
                   (columns: iteration, year, row, col, from_class, to_class, group)
    services     : list of EcosystemService from EcosystemServices.csv
    classes      : dict[int, StateClass] from load_state_classes()
    px_area_ha   : pixel area in hectares (for transition matrix area calc)
    """

    def __init__(
        self,
        area_df:    pd.DataFrame,
        trans_df:   pd.DataFrame,
        services:   list[EcosystemService],
        classes:    dict[int, StateClass],
        px_area_ha: float,
    ) -> None:
        self.area_df    = area_df
        self.trans_df   = trans_df
        self.services   = services
        self.classes    = classes
        self.px_area_ha = px_area_ha

        # Build service lookup: class_name → list of services
        self._svc_by_class: dict[str, list[EcosystemService]] = {}
        for svc in services:
            self._svc_by_class.setdefault(svc.state_class, []).append(svc)

        # Compute median area across iterations once
        self._median_area = self._compute_median_area()
        self._years       = sorted(self._median_area["year"].unique())

    # ── Public API ────────────────────────────────────────────────────────────

    def extent_account(self) -> pd.DataFrame:
        """
        Ecosystem extent account.
        Rows: year. Columns: one per class (area in ha). Matches SEEA-EA Tab #2.
        """
        pivot = self._median_area.pivot_table(
            index="year", columns="class_name", values="area_ha", aggfunc="sum"
        ).fillna(0)
        pivot.columns.name = None
        pivot.index.name   = "Year"
        return pivot

    def transition_matrix(self) -> pd.DataFrame:
        """
        Ecosystem extent change matrix.
        Shows median area (ha) converted from each class (rows) to each class (cols)
        aggregated across all timesteps. Matches SEEA-EA Tab #3.
        Also includes a value_change_matrix() for monetary impact.
        """
        if self.trans_df.empty:
            return pd.DataFrame()

        class_names = [sc.name for sc in self.classes.values()]

        # Median transition counts per from/to pair across iterations
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
        median_counts["area_ha"] = median_counts["n_cells"] * self.px_area_ha

        matrix = median_counts.pivot_table(
            index="from_class", columns="to_class",
            values="area_ha", aggfunc="sum"
        ).reindex(index=class_names, columns=class_names).fillna(0)
        matrix.index.name   = "From \\ To"
        matrix.columns.name = None
        return matrix

    def value_change_matrix(self) -> pd.DataFrame:
        """
        Monetary value change from transitions (from_class → to_class).
        Each cell = median area converted × (value_per_ha_destination − value_per_ha_source).
        Negative = value loss, positive = value gain.
        """
        tm = self.transition_matrix()
        if tm.empty:
            return pd.DataFrame()

        # Total value per ha per class (sum across all services for that class)
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

        val_matrix.index.name   = "From \\ To"
        val_matrix.columns.name = None
        return val_matrix

    def physical_flow_account(self) -> pd.DataFrame | None:
        """
        Physical ecosystem service flow account (Mode B only).
        Rows: year. Columns: (service_type, service_name, unit). Values: total quantity.
        Returns None if no services have physical units defined.
        """
        mode_b = [s for s in self.services if s.has_physical]
        if not mode_b:
            return None

        records = []
        for _, row in self._median_area.iterrows():
            svcs = [s for s in self._svc_by_class.get(row["class_name"], [])
                    if s.has_physical]
            for svc in svcs:
                records.append({
                    "year":         row["year"],
                    "class":        row["class_name"],
                    "service_type": svc.service_type,
                    "service_name": svc.service_name,
                    "unit":         svc.physical_unit,
                    "flow":         svc.physical_per_ha * row["area_ha"],
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
        Rows: year. Columns: (service_type, service_name). Values: total value (currency).
        """
        records = []
        for _, row in self._median_area.iterrows():
            for svc in self._svc_by_class.get(row["class_name"], []):
                records.append({
                    "year":         row["year"],
                    "class":        row["class_name"],
                    "service_type": svc.service_type,
                    "service_name": svc.service_name,
                    "currency":     svc.currency,
                    "value":        svc.value_per_ha * row["area_ha"],
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
        """
        Total monetary value per class per year (area × sum of all ValuePerHa).
        Useful for stacked area plots.
        """
        records = []
        for _, row in self._median_area.iterrows():
            svcs = self._svc_by_class.get(row["class_name"], [])
            total = sum(s.value_per_ha for s in svcs) * row["area_ha"]
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
        """
        Year-on-year change in total ecosystem service value.
        Rows: year. Columns: class + Total. Values: delta value from previous year.
        """
        tv = self.total_value_by_class()
        delta = tv.diff()
        delta["Total"] = delta.sum(axis=1)
        delta.index.name = "Year"
        return delta

    def uncertainty_summary(self) -> pd.DataFrame:
        """
        Min/max range of total ecosystem value across iterations, per year.
        Reported once as a context note (not repeated per account).
        """
        # Total value per iteration per year
        records = []
        for (iteration, year, class_name), grp in self.area_df.groupby(
            ["iteration", "year", "class_name"]
        ):
            area = grp["area_ha"].sum()
            svcs = self._svc_by_class.get(class_name, [])
            val  = sum(s.value_per_ha for s in svcs) * area
            records.append({"iteration": iteration, "year": year, "value": val})

        df = pd.DataFrame(records)
        stats = (
            df.groupby("year")["value"]
            .agg(median="median", min="min", max="max")
            .reset_index()
        )
        stats["range_pct"] = ((stats["max"] - stats["min"]) / stats["median"] * 100).round(1)
        stats.columns      = ["Year", "Median value", "Min value", "Max value", "Range (%)"]
        return stats

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_median_area(self) -> pd.DataFrame:
        """Median area per class per year across iterations."""
        median = (
            self.area_df.groupby(["year", "class_id", "class_name"])["area_ha"]
            .median()
            .reset_index()
        )
        return median
