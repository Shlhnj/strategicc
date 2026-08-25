"""
strategicc/accounting/seea.py  —  SEEA-EA accounting engine  v3.5
------------------------------------------------------------------
Produces all ecosystem accounts from simulation outputs.

v3.5 changes (strategicc 3.17)
-------------------------------
* New physical_flow_account_seea() / monetary_flow_account_seea()
  produce actual supply/use table pairs matching SEEA EA Tables
  7.1a/7.1b (physical) and 9.1a/9.1b (monetary): a supply table (year x
  class x service) and a use table (year x user_type x service),
  rather than the single collapsed-across-class total that
  physical_flow_account()/monetary_flow_account() report (those two
  are unchanged and remain the simpler summary form). The use table is
  built from EcosystemServices.csv's new optional UserType/UserShare
  columns (csv_loader.py v3.5); services without a UserType are folded
  into a single "Unspecified" user at share 1.0, so this works
  unchanged on existing EcosystemServices.csv files, just with one
  undifferentiated user column. For every (year, service), supply
  total and use total are constructed to be identical by design (the
  SUT identity SEEA EA para. 7.7 requires) — see
  test_flow_account_seea_supply_use_identity in tests/test_accounting.py.
* New monetary_asset_account_seea() produces the SEEA EA Table 10.1
  layout: Opening value, Ecosystem enhancement, Ecosystem degradation,
  Ecosystem conversions (Additions/Reductions), Other changes in
  volume (Catastrophic losses), Revaluations, Net change in value,
  Closing value — all reconciling exactly by construction (Net change
  = Closing - Opening for every class and period). Requires a new
  AssetValuationParams.csv (csv_loader.py v3.5, load_asset_valuation_
  params()), passed to SEEAAccount as asset_valuation_params=. Opening/
  Closing are NPV of that period's total service value per class (see
  _npv() below); conversions are valued using extent_account_seea()'s
  physical Additions/Reductions at the class's per-hectare value;
  catastrophic_groups= (like extent_account_seea()'s managed_groups=)
  optionally splits Reductions into ordinary vs. catastrophic losses;
  Revaluations isolates the pure price-growth contribution to the
  closing NPV via PriceGrowthRate; Reappraisals is not modelled and is
  always reported as 0.0, since STRATEGICC has no mechanism to
  generate a genuine methodology-change entry. Enhancement/Degradation
  is the *residual* needed to make Net change reconcile exactly — this
  is a documented approximation, not SEEA EA's condition-attributed
  split (para. 10.12), since STRATEGICC has no compiled condition
  account. If ConditionProxy is supplied in AssetValuationParams.csv,
  its direction of change is compared to the residual's sign only as a
  sanity check (a printed warning on disagreement), not used to
  compute the split.

v3.4 changes
------------
* extent_account() gained a Total column (sum across classes per year).
  Under an area-conserved run this stays constant year to year, which
  is the built-in check SEEA EA's own Total row/column is meant to
  give the compiler. This was missing entirely before.
* New extent_account_seea() produces the ecosystem extent account in
  the actual layout of SEEA EA Table 4.1: rows are accounting entries
  (Opening extent, Additions, Reductions, Net change in extent,
  Closing extent) per accounting period, columns are ecosystem types
  plus Total. extent_account() itself is unchanged in shape (still a
  flat year-by-class time series) and remains the right input for
  plotting/summary use; extent_account_seea() is the one that actually
  matches the standard's own table structure and is what should be
  cited/reported as "the SEEA EA extent account" in an accounting
  report. Additions/Reductions are derived from trans_df (the
  transition log), so trans_df must be supplied to SEEAAccount for
  this method to work. Managed/unmanaged splitting is optional and
  opt-in via managed_groups=, since SEEA EA leaves that classification
  to the compiler and STRATEGICC cannot infer it from the data alone.

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
from strategicc.accounting.csv_loader import EcosystemService, AssetValuationParams


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

    asset_valuation_params : (v3.5) dict[str, AssetValuationParams] from
                    csv_loader.load_asset_valuation_params(), keyed by
                    StateClassId ("ALL" is the fallback default).
                    Required for monetary_asset_account_seea().
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
        asset_valuation_params: dict[str, AssetValuationParams] | None = None,  # v3.5
    ) -> None:
        self.area_modal_df = area_modal_df
        self.trans_df      = trans_df
        self.services      = services
        self.classes       = classes
        self.px_area       = px_area
        self.area_df       = area_df
        self.stock_df      = stock_df
        self.flow_df       = flow_df
        self.asset_valuation_params = asset_valuation_params or {}

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

        # Build service lookups: class_name → list of services.
        # _svc_by_class keeps every row — needed to build the v3.5 use
        # table, and correct for the pre-existing additive pattern of
        # multiple rows sharing one service_name (e.g. Carbon Storage =
        # stock:AGB + stock:Soil, each contributing its own amount).
        # _svc_canonical_by_class collapses to ONE row per (class,
        # service_name) ONLY for genuine v3.5 user-split groups — every
        # row in the group has an explicit UserType, meaning they all
        # repeat the SAME total (see csv_loader.py's reconciliation
        # warning), so summing them all would double/n-count it.
        # Groups without an explicit UserType on every row are additive
        # (the original, pre-3.5 behaviour) and keep every row. This is
        # what physical_flow_account(), monetary_flow_account() and
        # total_value_by_class() use.
        self._svc_by_class: dict[str, list[EcosystemService]] = {}
        _by_key: dict[tuple[str, str], list[EcosystemService]] = {}
        for svc in services:
            self._svc_by_class.setdefault(svc.state_class, []).append(svc)
            _by_key.setdefault((svc.state_class, svc.service_name), []).append(svc)

        self._svc_canonical_by_class: dict[str, list[EcosystemService]] = {}
        for (cls, _name), group in _by_key.items():
            is_split_group = len(group) > 1 and all(s.has_explicit_user for s in group)
            reps = [group[0]] if is_split_group else group
            self._svc_canonical_by_class.setdefault(cls, []).extend(reps)

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
        Ecosystem extent account (flat time series).
        Rows: year. Columns: one per class (area in chosen unit) + Total.
        Derived from modal LULC maps — spatially consistent.

        v3.4: added the Total column (sum across classes for that year).
        Under an area-conserved run this should stay constant year to
        year — it is the area-conservation check SEEA EA's own Total
        row/column is meant to provide. This is a flat per-year table,
        not the period-structured (Opening/Additions/Reductions/Net
        change/Closing) layout of SEEA EA Table 4.1 — use
        extent_account_seea() for that.
        """
        pivot = self.area_modal_df.pivot_table(
            index="year", columns="class_name",
            values=self._acol, aggfunc="sum"
        ).fillna(0)
        pivot.columns.name = None
        pivot.index.name   = f"Year"
        pivot["Total"] = pivot.sum(axis=1)
        pivot.attrs["unit"] = self._unit_label
        return pivot

    def extent_account_seea(
        self,
        managed_groups: set[str] | None = None,
    ) -> pd.DataFrame:
        """
        Ecosystem extent account in SEEA EA Table 4.1 layout (v3.4, new
        in strategicc 3.17).

        One block per accounting period (year t -> year t+1): Opening
        extent, Additions, Reductions, Net change in extent, Closing
        extent. Columns are ecosystem classes plus a Total column, so
        area conservation can be checked directly per period.

        Opening/Closing extent for each period come from
        extent_account() (the class-by-class extent already derived
        from area_modal_df). Additions/Reductions per class are derived
        from trans_df (the per-timestep transition log): for a period
        year_t -> year_t+1, Additions to a class are the areas that
        flowed IN from every other class during that step; Reductions
        are the areas that flowed OUT to every other class. Both
        Net change (Additions - Reductions) and the reconciled
        Closing - Opening are useful cross-checks of each other, but
        only Net change (Additions - Reductions) is reported as a row,
        matching SEEA EA's own layout; if it disagrees with
        Closing - Opening derived from extent_account(), that signals
        the modal-map closing extent and the logged transitions have
        drifted apart (e.g. from cells changing class more than once
        within a period), which is worth checking in the run output.

        managed_groups : optional set of transition `group` names (as
            used in transition_log.csv / TransitionGroups.csv) to treat
            as "managed". If given, Additions and Reductions are each
            further split into managed/unmanaged rows, per SEEA EA's
            optional finer breakdown. SEEA EA leaves the managed vs.
            unmanaged classification to the compiler; STRATEGICC cannot
            infer it automatically, hence this is opt-in. If omitted,
            Additions/Reductions are each reported as a single
            (undivided) row.

        Returns
        -------
        DataFrame with a MultiIndex (Period, Entry) on the rows and
        columns [class_name, ..., Total]. Written to CSV this reads as
        one block of rows per accounting period, in Table 4.1's own
        row order.
        """
        if self.trans_df is None or self.trans_df.empty:
            raise ValueError(
                "extent_account_seea() requires trans_df (the "
                "transition log) to derive Additions/Reductions per "
                "period — pass trans_df= when constructing SEEAAccount."
            )

        extent = self.extent_account()
        class_names = [c for c in extent.columns if c != "Total"]
        years = list(extent.index)
        if len(years) < 2:
            raise ValueError(
                "extent_account_seea() needs at least two years of "
                "extent data to form one accounting period."
            )

        group_key = [] if managed_groups is None else ["group"]
        counts = (
            self.trans_df
            .groupby(["iteration", "year", "from_class", "to_class"] + group_key)
            .size()
            .reset_index(name="n_cells")
        )
        median_counts = (
            counts.groupby(["year", "from_class", "to_class"] + group_key)["n_cells"]
            .median()
            .reset_index()
        )
        median_counts["area"] = median_counts["n_cells"] * self.px_area

        def _flow(period_start_year: int, direction: str, tag: str | None) -> pd.Series:
            # direction: "in" (additions, group by to_class) or
            # "out" (reductions, group by from_class). `year` on a
            # TransitionRecord is the OPENING year of that step, i.e.
            # the transition from state[year] to state[year+1] — so a
            # period year_t -> year_t+1 is looked up by year==year_t.
            sub = median_counts[median_counts["year"] == period_start_year]
            if tag is not None:
                is_managed = sub["group"].isin(managed_groups)
                sub = sub[is_managed] if tag == "Managed" else sub[~is_managed]
            col = "to_class" if direction == "in" else "from_class"
            out = sub.groupby(col)["area"].sum()
            return out.reindex(class_names).fillna(0.0)

        rows: list[tuple[tuple[str, str], dict]] = []
        for i in range(len(years) - 1):
            y0, y1 = years[i], years[i + 1]
            period = f"{y0}\u2013{y1}"
            opening = extent.loc[y0, class_names]
            closing = extent.loc[y1, class_names]

            if managed_groups is None:
                additions  = _flow(y0, "in", None)
                reductions = _flow(y0, "out", None)
                entries = [
                    ("Opening extent",       opening),
                    ("Additions",            additions),
                    ("Reductions",           reductions),
                    ("Net change in extent", additions - reductions),
                    ("Closing extent",       closing),
                ]
            else:
                add_m = _flow(y0, "in", "Managed")
                add_u = _flow(y0, "in", "Unmanaged")
                red_m = _flow(y0, "out", "Managed")
                red_u = _flow(y0, "out", "Unmanaged")
                net = (add_m + add_u) - (red_m + red_u)
                entries = [
                    ("Opening extent",                  opening),
                    ("Additions — managed expansions",  add_m),
                    ("Additions — unmanaged expansions", add_u),
                    ("Reductions — managed reductions",  red_m),
                    ("Reductions — unmanaged reductions", red_u),
                    ("Net change in extent",             net),
                    ("Closing extent",                   closing),
                ]

            for entry_name, series in entries:
                series = pd.Series(series, index=class_names).astype(float)
                row = series.to_dict()
                row["Total"] = float(series.sum())
                rows.append(((period, entry_name), row))

        index = pd.MultiIndex.from_tuples(
            [r[0] for r in rows], names=["Period", "Entry"]
        )
        df = pd.DataFrame([r[1] for r in rows], index=index,
                           columns=class_names + ["Total"])
        df.attrs["unit"] = self._unit_label
        return df

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
            svcs = self._svc_canonical_by_class.get(sc.name, [])
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
                s for s in self._svc_canonical_by_class.get(row["class_name"], [])
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
            for svc in self._svc_canonical_by_class.get(row["class_name"], []):
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

    def physical_flow_account_seea(self) -> dict[str, pd.DataFrame] | None:
        """
        Physical ecosystem services flow account as a supply/use table
        pair, matching SEEA EA Tables 7.1a (supply) / 7.1b (use).

        Returns {"supply": DataFrame, "use": DataFrame}, or None if no
        service has a physical unit or stock/flow source (same
        precondition as physical_flow_account()).

        supply : rows (year, class), columns (service_type,
                 service_name, unit) — the physical quantity each
                 ecosystem type supplied that year, exactly matching
                 physical_flow_account()'s totals once summed over
                 class.
        use    : rows (year, user_type), columns (service_type,
                 service_name, unit) — the same total split across
                 users via each service's UserType/UserShare
                 (csv_loader.py v3.5). Services without an explicit
                 UserType appear under a single "Unspecified" user at
                 the full total, so for every (year, service) column,
                 supply.sum() == use.sum() by construction (the SUT
                 identity, SEEA EA para. 7.7).
        """
        has_any_physical = any(
            s.has_physical or s.has_stockflow_source for s in self.services
        )
        if not has_any_physical:
            return None

        supply_records, use_records = [], []
        for _, row in self.area_modal_df.iterrows():
            class_name, year, area = row["class_name"], row["year"], row[self._acol]

            for svc in self._svc_canonical_by_class.get(class_name, []):
                if not (svc.has_physical or svc.has_stockflow_source):
                    continue
                qty = self._service_physical_qty(svc, class_name, year, area)
                if qty is None:
                    continue
                unit = svc.physical_unit or (
                    f"{svc.stockflow_type_name} ({svc.stockflow_kind})"
                    if svc.has_stockflow_source else ""
                )
                supply_records.append({
                    "year": year, "class": class_name,
                    "service_type": svc.service_type, "service_name": svc.service_name,
                    "unit": unit, "flow": qty,
                })

            for svc in self._svc_by_class.get(class_name, []):
                if not (svc.has_physical or svc.has_stockflow_source):
                    continue
                qty = self._service_physical_qty(svc, class_name, year, area)
                if qty is None:
                    continue
                unit = svc.physical_unit or (
                    f"{svc.stockflow_type_name} ({svc.stockflow_kind})"
                    if svc.has_stockflow_source else ""
                )
                use_records.append({
                    "year": year, "user_type": svc.user_type,
                    "service_type": svc.service_type, "service_name": svc.service_name,
                    "unit": unit, "flow": qty * svc.user_share,
                })

        if not supply_records:
            return None

        supply = pd.DataFrame(supply_records).pivot_table(
            index=["year", "class"],
            columns=["service_type", "service_name", "unit"],
            values="flow", aggfunc="sum",
        ).fillna(0)
        use = pd.DataFrame(use_records).pivot_table(
            index=["year", "user_type"],
            columns=["service_type", "service_name", "unit"],
            values="flow", aggfunc="sum",
        ).fillna(0)
        supply.index.names = ["Year", "Ecosystem type"]
        use.index.names    = ["Year", "User type"]
        return {"supply": supply, "use": use}

    def monetary_flow_account_seea(self) -> dict[str, pd.DataFrame]:
        """
        Monetary ecosystem services flow account as a supply/use table
        pair, matching SEEA EA Tables 9.1a (supply) / 9.1b (use).

        Returns {"supply": DataFrame, "use": DataFrame} — same shape
        and construction as physical_flow_account_seea(), in monetary
        terms. For every (year, service) column, supply.sum() ==
        use.sum() by construction.
        """
        supply_records, use_records = [], []
        for _, row in self.area_modal_df.iterrows():
            class_name, year, area = row["class_name"], row["year"], row[self._acol]

            for svc in self._svc_canonical_by_class.get(class_name, []):
                value = self._service_monetary_value(svc, class_name, year, area)
                supply_records.append({
                    "year": year, "class": class_name,
                    "service_type": svc.service_type, "service_name": svc.service_name,
                    "value": value,
                })

            for svc in self._svc_by_class.get(class_name, []):
                value = self._service_monetary_value(svc, class_name, year, area)
                use_records.append({
                    "year": year, "user_type": svc.user_type,
                    "service_type": svc.service_type, "service_name": svc.service_name,
                    "value": value * svc.user_share,
                })

        supply = pd.DataFrame(supply_records).pivot_table(
            index=["year", "class"],
            columns=["service_type", "service_name"],
            values="value", aggfunc="sum",
        ).fillna(0)
        use = pd.DataFrame(use_records).pivot_table(
            index=["year", "user_type"],
            columns=["service_type", "service_name"],
            values="value", aggfunc="sum",
        ).fillna(0)
        supply.index.names = ["Year", "Ecosystem type"]
        use.index.names    = ["Year", "User type"]
        return {"supply": supply, "use": use}

    def total_value_by_class(self) -> pd.DataFrame:
        """Total monetary value per class per year. Used for stacked area plots."""
        records = []
        for _, row in self.area_modal_df.iterrows():
            svcs  = self._svc_canonical_by_class.get(row["class_name"], [])
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

    # ── Internal: v3.5 monetary asset account helpers ──────────────────────

    @staticmethod
    def _npv(
        annual_value: float, discount_rate: float,
        asset_life_years: int, price_growth_rate: float = 0.0,
    ) -> float:
        """
        NPV of a stream of `asset_life_years` annual cash flows starting
        at `annual_value` and growing at `price_growth_rate` per year,
        discounted at `discount_rate`, income earned at the end of each
        year (SEEA EA's own assumption — see chap. 10 worked example).
        Summed explicitly term by term (asset_life_years is at most a
        few hundred) rather than a closed-form annuity formula, so
        there's no edge case when price_growth_rate == discount_rate.
        """
        if asset_life_years <= 0:
            return 0.0
        r, g = discount_rate, price_growth_rate
        return sum(
            annual_value * ((1 + g) ** t) / ((1 + r) ** (t + 1))
            for t in range(asset_life_years)
        )

    def _class_transition_area(
        self, period_start_year: int, direction: str,
        class_names: list[str], group_filter: set[str] | None = None,
    ) -> pd.Series:
        """
        Median area transitioning in ('in', grouped by to_class) or out
        ('out', grouped by from_class) of each class for the single
        period starting at period_start_year, optionally restricted to
        transitions whose `group` is in group_filter. Mirrors
        extent_account_seea()'s internal flow logic; kept separate here
        since this needs an independent group_filter dimension
        (catastrophic vs. not) rather than extent_account_seea()'s
        managed/unmanaged one.
        """
        empty = pd.Series(0.0, index=class_names)
        if self.trans_df is None or self.trans_df.empty:
            return empty
        counts = (
            self.trans_df
            .groupby(["iteration", "year", "from_class", "to_class", "group"])
            .size().reset_index(name="n_cells")
        )
        median_counts = (
            counts.groupby(["year", "from_class", "to_class", "group"])["n_cells"]
            .median().reset_index()
        )
        median_counts["area"] = median_counts["n_cells"] * self.px_area
        sub = median_counts[median_counts["year"] == period_start_year]
        if group_filter is not None:
            sub = sub[sub["group"].isin(group_filter)]
        col = "to_class" if direction == "in" else "from_class"
        out = sub.groupby(col)["area"].sum()
        return out.reindex(class_names).fillna(0.0)

    def monetary_asset_account_seea(
        self,
        catastrophic_groups: set[str] | None = None,
    ) -> pd.DataFrame:
        """
        Monetary ecosystem asset account in SEEA EA Table 10.1 layout
        (v3.5, new in strategicc 3.17).

        One block per accounting period: Opening value, Ecosystem
        enhancement, Ecosystem degradation, Ecosystem conversions
        (Additions/Reductions valued), Other changes in volume
        (Catastrophic losses / Reappraisals), Revaluations, Net change
        in value, Closing value. Columns are ecosystem classes plus
        Total. Requires asset_valuation_params (from
        csv_loader.load_asset_valuation_params()) to have been passed
        to SEEAAccount, and requires trans_df.

        Method (documented here since Table 10.1's entries require
        choices SEEA EA leaves to the compiler):

        * Opening/Closing value: NPV of that period's actual total
          service value for the class (total_value_by_class()), using
          DiscountRate/AssetLifeYears/PriceGrowthRate from
          AssetValuationParams.csv, per SEEA EA's own worked assumption
          of a constant future flow at the current rate (chap. 10,
          appendix A10.1) with income earned at year end.
        * Ecosystem conversions: Additions/Reductions come from
          extent_account_seea()'s physical area entries (no
          managed_groups split here — that's a separate, orthogonal
          dimension). Additions are valued at the class's per-area
          value in the period's closing year; Reductions at its
          per-area value in the opening year, matching SEEA EA para.
          10.12's requirement that these align with the physical extent
          account.
        * Other changes in volume — Catastrophic losses: if
          catastrophic_groups is given, that portion of Reductions
          whose transition `group` is in catastrophic_groups is
          reported here instead of under ordinary Reductions (reusing
          extent_account_seea()'s managed_groups= pattern — SEEA EA
          leaves this classification to the compiler too). Reappraisals
          is always reported as 0.0 — STRATEGICC has no mechanism to
          generate a genuine methodology-change entry, so this is
          honestly absent rather than silently missing.
        * Revaluations: isolates the pure price-growth contribution to
          the closing valuation — NPV at PriceGrowthRate minus NPV of
          the same annual value at 0 growth. Zero whenever
          PriceGrowthRate is 0 (the honest default).
        * Ecosystem enhancement / degradation: the RESIDUAL needed so
          that Net change in value reconciles exactly with
          Closing - Opening (Enhancement = max(residual, 0),
          Degradation = min(residual, 0)). This is a documented
          approximation, not SEEA EA's condition-attributed split (SEEA
          EA para. 10.12 ties this to the compiled condition account,
          which STRATEGICC does not have). If a class's
          AssetValuationParams.csv row supplies ConditionProxy, its
          direction of change is compared to the residual's sign purely
          as a sanity check — a printed warning if they disagree, never
          used to compute the split itself.

        Raises ValueError if trans_df or asset_valuation_params is
        missing, or if any class lacks both its own row and an "ALL"
        fallback in asset_valuation_params.
        """
        if self.trans_df is None or self.trans_df.empty:
            raise ValueError(
                "monetary_asset_account_seea() requires trans_df — pass "
                "trans_df= when constructing SEEAAccount."
            )
        if not self.asset_valuation_params:
            raise ValueError(
                "monetary_asset_account_seea() requires asset_valuation_params "
                "— pass asset_valuation_params=load_asset_valuation_params(...) "
                "when constructing SEEAAccount."
            )

        extent   = self.extent_account()
        ext_seea = self.extent_account_seea()
        tv       = self.total_value_by_class()
        class_names = [c for c in extent.columns if c != "Total"]
        years    = list(extent.index)

        missing = [
            c for c in class_names
            if c not in self.asset_valuation_params and "ALL" not in self.asset_valuation_params
        ]
        if missing:
            raise ValueError(
                f"monetary_asset_account_seea(): no AssetValuationParams row "
                f"for class(es) {missing} and no 'ALL' fallback row."
            )

        def _params(class_name: str) -> AssetValuationParams:
            return self.asset_valuation_params.get(
                class_name, self.asset_valuation_params.get("ALL")
            )

        rows: list[tuple[tuple[str, str], dict]] = []
        for i in range(len(years) - 1):
            y0, y1 = years[i], years[i + 1]
            period = f"{y0}\u2013{y1}"

            catastrophic_area = (
                self._class_transition_area(y0, "out", class_names, catastrophic_groups)
                if catastrophic_groups else pd.Series(0.0, index=class_names)
            )

            entries = {
                "Opening value": {}, "Ecosystem enhancement": {},
                "Ecosystem degradation": {}, "Ecosystem conversions — additions": {},
                "Ecosystem conversions — reductions": {},
                "Other changes in volume — catastrophic losses": {},
                "Other changes in volume — reappraisals": {},
                "Revaluations": {}, "Net change in value": {}, "Closing value": {},
            }

            for cls in class_names:
                p = _params(cls)
                r, L, g = p.discount_rate, p.asset_life_years, p.price_growth_rate

                opening_value = self._npv(tv.loc[y0, cls], r, L, g)
                closing_value = self._npv(tv.loc[y1, cls], r, L, g)

                opening_area = extent.loc[y0, cls]
                closing_area = extent.loc[y1, cls]
                val_per_area_y0 = (tv.loc[y0, cls] / opening_area) if opening_area > 0 else 0.0
                val_per_area_y1 = (tv.loc[y1, cls] / closing_area) if closing_area > 0 else 0.0

                additions_area  = ext_seea.loc[(period, "Additions"), cls]
                reductions_area = ext_seea.loc[(period, "Reductions"), cls]
                cat_area        = min(catastrophic_area.get(cls, 0.0), reductions_area)
                ordinary_red_area = reductions_area - cat_area

                additions_value = additions_area * val_per_area_y1
                reductions_value = ordinary_red_area * val_per_area_y0
                catastrophic_value = cat_area * val_per_area_y0

                revaluation = (
                    self._npv(tv.loc[y1, cls], r, L, g)
                    - self._npv(tv.loc[y1, cls], r, L, 0.0)
                )
                reappraisal = 0.0  # not modelled

                net_change = closing_value - opening_value
                explained = (
                    additions_value - reductions_value - catastrophic_value
                    + revaluation + reappraisal
                )
                residual = net_change - explained
                enhancement = max(residual, 0.0)
                degradation = min(residual, 0.0)

                if p.condition_proxy is not None and self.stock_df is not None \
                        and not self.stock_df.empty and residual != 0:
                    cp0 = self.stock_df[
                        (self.stock_df["class_name"] == cls) & (self.stock_df["year"] == y0)
                        & (self.stock_df["stock_type"] == p.condition_proxy)
                    ]["total"].sum()
                    cp1 = self.stock_df[
                        (self.stock_df["class_name"] == cls) & (self.stock_df["year"] == y1)
                        & (self.stock_df["stock_type"] == p.condition_proxy)
                    ]["total"].sum()
                    cond_delta = cp1 - cp0
                    if cond_delta != 0 and (cond_delta > 0) != (residual > 0):
                        print(f"  [Warning] monetary_asset_account_seea(): '{cls}' "
                              f"{period} — Enhancement/Degradation residual "
                              f"({residual:+.2f}) disagrees in sign with its "
                              f"ConditionProxy '{p.condition_proxy}' change "
                              f"({cond_delta:+.2f}); residual split kept as-is "
                              f"(see method docstring — this is a sanity check, "
                              f"not an input to the split).")

                entries["Opening value"][cls]                              = opening_value
                entries["Ecosystem enhancement"][cls]                      = enhancement
                entries["Ecosystem degradation"][cls]                      = degradation
                entries["Ecosystem conversions — additions"][cls]          = additions_value
                entries["Ecosystem conversions — reductions"][cls]         = reductions_value
                entries["Other changes in volume — catastrophic losses"][cls] = catastrophic_value
                entries["Other changes in volume — reappraisals"][cls]     = reappraisal
                entries["Revaluations"][cls]                               = revaluation
                entries["Net change in value"][cls]                        = net_change
                entries["Closing value"][cls]                              = closing_value

            for entry_name, per_class in entries.items():
                row = dict(per_class)
                row["Total"] = float(sum(per_class.values()))
                rows.append(((period, entry_name), row))

        index = pd.MultiIndex.from_tuples([r[0] for r in rows], names=["Period", "Entry"])
        df = pd.DataFrame([r[1] for r in rows], index=index,
                           columns=class_names + ["Total"])
        return df

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
            svcs = self._svc_canonical_by_class.get(class_name, [])
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
