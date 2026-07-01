"""
strategicc/accounting/seea.py  —  SEEA-EA accounting engine  v3.3
------------------------------------------------------------------
Produces all ecosystem accounts from simulation outputs.

v3.3 changes
------------
* EcosystemServices.csv columns renamed ValuePerHa/PhysicalValuePerHa ->
  ValuePerUnitArea/PhysicalValuePerUnitArea (old names still accepted). These
  prices are always hectare-denominated. Fixed a unit-consistency bug:
  when AREA_UNIT != "ha", area_modal_df/area_df are expressed in km2 or
  raw pixel counts, but valuation was multiplying hectare-based prices
  by those figures directly with no conversion. SEEAAccount now accepts
  px_area_ha and converts area back to hectares before pricing.

v3.2 changes
------------
* Optional stock_df / flow_df parameters (from
  strategicc.stockflow.aggregation) enable Mode C valuation: services
  whose EcosystemServices.csv row sets StockFlowSource pull their
  physical quantity directly from the Stock & Flow engine's per-class
  totals instead of a static PhysicalValuePerUnitArea, with ValuePerUnitArea then
  acting as a price PER PHYSICAL UNIT rather than per area.

Accounts produced
-----------------
1. Extent account        — area per class per year (modal)
2. Transition matrix     — area converted between classes + value change
3. Physical flow account — total physical units supplied per service per year
4. Monetary flow account — total monetary value per service per year
5. Change-in-value       — year-on-year change in total ecosystem value
6. Uncertainty summary   — min/max range across iterations (raw area_df)
7. Stock account         — total stock per class per year (Mode C, v3.2)
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
    SEEA-EA ecosystem accounting engine  v3.3

    Parameters
    ----------
    area_modal_df : area table derived from modal LULC maps — used for all
                    area-based accounts. Schema: year, class_id, class_name,
                    area_{unit}. Produced by outputs.modal_to_area_table().
                    area_{unit} is expressed in whatever AREA_UNIT the run
                    used (ha | km2 | px) — see px_area_ha below.

    area_df       : raw per-iteration area table — used ONLY for the
                    uncertainty summary. Schema adds an 'iteration' column.
                    Pass None to skip uncertainty summary.

    trans_df      : concatenated transition_log.csv across all iterations.
                    Used for transition matrix (median counts across iters).

    services      : list of EcosystemService from EcosystemServices.csv.
                    ValuePerUnitArea / PhysicalValuePerUnitArea are always
                    hectare-denominated (see accounting/csv_loader.py).

    classes       : dict[int, StateClass]

    px_area       : pixel area in the run's chosen AREA_UNIT (engine.px_area).
                    Used for transition matrix area calculation and for
                    unit detection on area_modal_df.

    px_area_ha    : (v3.3) pixel area in hectares (engine.px_area_ha).
                    Required to correctly value area-based ecosystem
                    services (Mode A/B) when AREA_UNIT != "ha" — used to
                    convert area figures in area_modal_df/area_df (which
                    are in the run's AREA_UNIT) back to hectares before
                    applying ValuePerUnitArea/PhysicalValuePerUnitArea. If omitted,
                    a factor of 1.0 is assumed (i.e. area is treated as
                    already being in hectares) — a warning is printed if
                    the detected area unit isn't "ha" in that case, since
                    valuation would then be silently wrong.

    stock_df      : (v3.2) DataFrame from
                    stockflow.aggregation.aggregate_stock_by_class().
                    Schema: year, class_id, class_name, stock_type, total.
                    Required for Mode C services with stockflow_kind="stock".

    flow_df       : (v3.2) DataFrame from
                    stockflow.aggregation.aggregate_flow_by_class().
                    Schema: year, class_name, flow_type, total.
                    Required for Mode C services with stockflow_kind="flow".
    """

    def __init__(
        self,
        area_modal_df: pd.DataFrame,
        trans_df:      pd.DataFrame,
        services:      list[EcosystemService],
        classes:       dict[int, StateClass],
        px_area:       float,
        px_area_ha:    float | None = None,   # v3.3
        area_df:       pd.DataFrame | None = None,
        stock_df:      pd.DataFrame | None = None,   # v3.2
        flow_df:       pd.DataFrame | None = None,   # v3.2
    ) -> None:
        self.area_modal_df = area_modal_df
        self.trans_df      = trans_df
        self.services      = services
        self.classes       = classes
        self.px_area       = px_area
        self.area_df       = area_df
        self.stock_df      = stock_df
        self.flow_df       = flow_df

        # Detect area column and unit label from modal df
        self._acol       = _area_col(area_modal_df)
        self._unit_label = _unit_label(self._acol)

        # v3.3 — conversion factor from area_modal_df's unit back to hectares.
        # area_ha = area_in_chosen_unit * self._ha_per_unit
        if px_area_ha is None:
            self._ha_per_unit = 1.0
            has_valuable_services = any(
                s.value_per_unit_area or s.physical_per_unit_area for s in services
            )
            if has_valuable_services and self._acol != "area_ha":
                print(
                    f"  [Warning] SEEAAccount received no px_area_ha and "
                    f"area_modal_df is in '{self._unit_label}', not hectares. "
                    f"ValuePerUnitArea/PhysicalValuePerUnitArea are hectare-denominated "
                    f"(see csv_loader.py) — without px_area_ha, valuation will "
                    f"silently treat {self._unit_label} figures as if they were "
                    f"hectares. Pass px_area_ha=engine.px_area_ha to fix this."
                )
        else:
            self._ha_per_unit = (px_area_ha / px_area) if px_area else 1.0

        # Build service lookup: class_name → list of services
        self._svc_by_class: dict[str, list[EcosystemService]] = {}
        for svc in services:
            self._svc_by_class.setdefault(svc.state_class, []).append(svc)

        self._years = sorted(area_modal_df["year"].unique())

    # ── Internal: Mode C lookup helpers ─────────────────────────────────────

    def _lookup_stockflow_quantity(
        self,
        svc:        EcosystemService,
        class_name: str,
        year:       int,
    ) -> float:
        """
        Look up the physical quantity for a Mode C (stock_flow-linked)
        service, for one class and year. Returns 0.0 if not found or if
        the required aggregation DataFrame was not supplied.
        """
        kind = svc.stockflow_kind
        type_name = svc.stockflow_type_name

        if kind == "stock":
            if self.stock_df is None or self.stock_df.empty:
                return 0.0
            match = self.stock_df[
                (self.stock_df["class_name"] == class_name)
                & (self.stock_df["year"] == year)
                & (self.stock_df["stock_type"] == type_name)
            ]
            return float(match["total"].sum()) if not match.empty else 0.0

        if kind == "flow":
            if self.flow_df is None or self.flow_df.empty:
                return 0.0
            match = self.flow_df[
                (self.flow_df["class_name"] == class_name)
                & (self.flow_df["year"] == year)
                & (self.flow_df["flow_type"] == type_name)
            ]
            return float(match["total"].sum()) if not match.empty else 0.0

        return 0.0

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
            total_val[sc.name] = sum(s.value_per_unit_area for s in svcs)

        val_matrix = pd.DataFrame(0.0, index=tm.index, columns=tm.columns)
        for from_cls in tm.index:
            for to_cls in tm.columns:
                area = tm.loc[from_cls, to_cls]
                if area > 0:
                    area_ha = area * self._ha_per_unit
                    delta = total_val.get(to_cls, 0) - total_val.get(from_cls, 0)
                    val_matrix.loc[from_cls, to_cls] = area_ha * delta

        val_matrix.index.name   = "From \\ To (currency)"
        val_matrix.columns.name = None
        return val_matrix

    def _service_physical_qty(
        self, svc: EcosystemService, class_name: str, year: int, area: float,
    ) -> float | None:
        """
        Unified physical-quantity resolver across all three modes.
        Returns None if no physical quantity applies (Mode A, no
        PhysicalUnit defined).

        `area` is in the run's AREA_UNIT — converted to hectares before
        applying physical_per_unit_area, which is always hectare-denominated.
        """
        if svc.has_stockflow_source:
            return self._lookup_stockflow_quantity(svc, class_name, year)
        if svc.has_physical:
            area_ha = area * self._ha_per_unit
            return svc.physical_per_unit_area * area_ha
        return None

    def _service_monetary_value(
        self, svc: EcosystemService, class_name: str, year: int, area: float,
    ) -> float:
        """
        Unified monetary-value resolver across all three modes.

        Mode A/B: value = ValuePerUnitArea * area_ha  (ValuePerUnitArea is a
                  hectare-denominated price; `area`, given in the run's
                  AREA_UNIT, is converted to hectares first)
        Mode C:   value = stockflow_quantity * ValuePerUnitArea
                  (ValuePerUnitArea is reinterpreted as price PER PHYSICAL UNIT,
                  not area-denominated, so no ha conversion applies)
        """
        if svc.has_stockflow_source:
            qty = self._lookup_stockflow_quantity(svc, class_name, year)
            return qty * svc.value_per_unit_area
        area_ha = area * self._ha_per_unit
        return svc.value_per_unit_area * area_ha

    def physical_flow_account(self) -> pd.DataFrame | None:
        """
        Physical ecosystem service flow account (Mode B and Mode C).
        Rows: year. Columns: (service_type, service_name, unit). Values: total quantity.
        """
        has_any_physical = any(
            s.has_physical or s.has_stockflow_source for s in self.services
        )
        if not has_any_physical:
            return None

        records = []
        for _, row in self.area_modal_df.iterrows():
            svcs = [
                s for s in self._svc_by_class.get(row["class_name"], [])
                if s.has_physical or s.has_stockflow_source
            ]
            for svc in svcs:
                qty = self._service_physical_qty(
                    svc, row["class_name"], row["year"], row[self._acol]
                )
                if qty is None:
                    continue
                unit = svc.physical_unit or (
                    f"{svc.stockflow_type_name} ({svc.stockflow_kind})"
                    if svc.has_stockflow_source else ""
                )
                records.append({
                    "year":         row["year"],
                    "class":        row["class_name"],
                    "service_type": svc.service_type,
                    "service_name": svc.service_name,
                    "unit":         unit,
                    "flow":         qty,
                })

        if not records:
            return None

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

        Mode A/B services: ValuePerUnitArea is hectare-denominated; area is
        converted from the run's AREA_UNIT back to hectares first.
        Mode C services (v3.2): ValuePerUnitArea is treated as price PER PHYSICAL
        UNIT, applied to the stock/flow-sourced quantity.
        """
        records = []
        for _, row in self.area_modal_df.iterrows():
            for svc in self._svc_by_class.get(row["class_name"], []):
                value = self._service_monetary_value(
                    svc, row["class_name"], row["year"], row[self._acol]
                )
                records.append({
                    "year":         row["year"],
                    "class":        row["class_name"],
                    "service_type": svc.service_type,
                    "service_name": svc.service_name,
                    "currency":     svc.currency,
                    "value":        value,
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
            total = sum(
                self._service_monetary_value(
                    s, row["class_name"], row["year"], row[self._acol]
                )
                for s in svcs
            )
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

        Assumes area_df uses the same AREA_UNIT as area_modal_df (true for
        any single engine run) — reuses the same ha-conversion factor.
        """
        if self.area_df is None or self.area_df.empty:
            return None

        raw_acol = _area_col(self.area_df)
        records  = []
        for (iteration, year, class_name), grp in self.area_df.groupby(
            ["iteration", "year", "class_name"]
        ):
            area    = grp[raw_acol].sum()
            area_ha = area * self._ha_per_unit
            svcs = self._svc_by_class.get(class_name, [])
            val  = sum(s.value_per_unit_area for s in svcs) * area_ha
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
