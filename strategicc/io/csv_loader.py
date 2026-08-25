"""
strategicc/accounting/csv_loader.py  —  v3.5
-------------------------------------
Parse EcosystemServices.csv into EcosystemService dataclasses, and
AssetValuationParams.csv into AssetValuationParams dataclasses.

v3.5 changes (strategicc 3.17)
-------------------------------
EcosystemServices.csv gained two optional columns, UserType and
UserShare, so the monetary/physical flow accounts can be split into a
supply table (by ecosystem type) and a use table (by economic-unit
type) matching SEEA EA Tables 7.1a/7.1b and 9.1a/9.1b, rather than
only ever reporting a single collapsed total per service.

A service with more than one user gets one row per user, all sharing
the same StateClassId/ServiceName/ValuePerUnitArea (the full, undivided
total for that service), each row's UserShare giving that user's
fraction of it (0-1). Shares recorded for the same (StateClassId,
ServiceName) pair should sum to 1.0; SEEAAccount warns if they don't.
If UserType/UserShare are omitted for a service entirely, it is
treated as a single user "Unspecified" at UserShare 1.0 — existing
EcosystemServices.csv files without these columns are unaffected.

Example (Cropland's crop provisioning split 60/40 between households
and the agriculture sector):
    StateClassId, ServiceName, ServiceType, ValuePerUnitArea, Currency, UserType, UserShare
    Cropland, Crop Provisioning, Provisioning, 8000, IDR, Households, 0.6
    Cropland, Crop Provisioning, Provisioning, 8000, IDR, Agriculture, 0.4

Note: this covers only the *final*-use side of Tables 7.1b/9.1b
(economic units consuming a service). It does not cover intermediate
service flows between ecosystem assets (e.g. pollination from
grassland used by cropland) — that needs a supplier-class-to-consumer
-class link, which is out of scope for this column addition.

UNIT CONVENTION (v3.3)
-----------------------
ValuePerUnitArea and PhysicalValuePerUnitArea are always denominated PER HECTARE,
regardless of the engine's configured AREA_UNIT (ha | km2 | px). This is
a fixed, physically meaningful reference unit — hectares don't change
size when you switch a run's display unit to km2 or px pixel counts.

SEEAAccount converts the area figures it receives (which ARE expressed
in whatever AREA_UNIT the run used) back to hectares internally before
applying these prices, using px_area_ha (the known real-world size of
one raster pixel). Callers don't need to do any conversion themselves —
just author the CSV in per-hectare terms and pass px_area_ha through to
SEEAAccount.

The columns were renamed from ValuePerHa/PhysicalValuePerHa (pre-v3.3)
to ValuePerUnitArea/PhysicalValuePerUnitArea to stop implying that the *engine's*
area unit had to be hectares — it doesn't; only the price basis does.
Old column names are still accepted for backward compatibility (with a
one-time warning) and are interpreted identically.

CSV format (three modes supported):

Mode A — monetary value per ha only (no physical unit):
    StateClassId, ServiceName, ServiceType, ValuePerUnitArea, Currency
    Mangrove, Ecotourism, Cultural, 12500000, IDR

Mode B — physical unit + monetary value per ha (static, area-based):
    StateClassId, ServiceName, ServiceType, ValuePerUnitArea, Currency, PhysicalUnit, PhysicalValuePerUnitArea
    Mangrove, Carbon Sequestration, Regulating, 97500000, IDR, MgC/ha, 1300

Mode C — physical quantity sourced from the Stock & Flow engine (v3.2):
    StateClassId, ServiceName, ServiceType, ValuePerUnitArea, Currency, PhysicalUnit, PhysicalValuePerUnitArea, StockFlowSource
    Mangrove, Carbon Sequestration, Regulating, 75000, IDR, MgC, , flow:NPP
    Mangrove, Carbon Storage, Regulating, 75000, IDR, MgC, , stock:Biomass

    StockFlowSource format: "flow:<FlowTypeId>" or "stock:<StockTypeId>".
    When set, PhysicalValuePerUnitArea is ignored — the physical quantity is
    instead read directly from the Stock & Flow engine's per-class total
    (stock_table.csv total, or flow_log.csv total_amount summed across
    matching flow_type rows for that year), and ValuePerUnitArea is then
    treated as a price PER PHYSICAL UNIT (not per area) — i.e. monetary
    value = stock_flow_quantity * ValuePerUnitArea. (Mode C values are not
    area-denominated at all, so the hectare convention above doesn't
    apply to them.)

ServiceType must be one of: Provisioning, Regulating, Cultural

AssetValuationParams.csv (new in v3.5 / strategicc 3.17)
----------------------------------------------------------
Supplies the NPV/valuation parameters needed for
SEEAAccount.monetary_asset_account_seea() (SEEA EA Table 10.1):

    StateClassId, DiscountRate, AssetLifeYears, PriceGrowthRate, ConditionProxy, ConditionReferenceLevel
    Mangrove, 0.02, 100, 0.00, Biomass, 180
    Cropland, 0.02, 50, 0.01, , 
    ALL, 0.02, 100, 0.00, ,

StateClassId "ALL" supplies default parameters used for any class
without its own row. DiscountRate and AssetLifeYears are required
(real discount rate, and assumed years of future service flow used in
the NPV sum). PriceGrowthRate is optional (default 0.0) and drives the
Revaluations entry — leaving it 0 is a legitimate choice, it reports
that no price growth is being modelled, rather than silently omitting
the concept. ConditionProxy/ConditionReferenceLevel are optional: if
given, ConditionProxy must name an existing stock_type from stock_df
(e.g. "Biomass"), used as an approximate condition index (proxy value
/ ConditionReferenceLevel) to sanity-check the direction of the
Enhancement/Degradation split. This is a proxy, not a compiled SEEA EA
condition account (chap. 5), and is documented as such.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path


VALID_SERVICE_TYPES = {"Provisioning", "Regulating", "Cultural"}


@dataclass
class EcosystemService:
    """One ecosystem service entry from EcosystemServices.csv."""
    state_class:        str           # matches StateClass name e.g. "Mangrove"
    service_name:       str           # e.g. "Carbon Sequestration"
    service_type:       str           # "Provisioning" | "Regulating" | "Cultural"
    value_per_unit_area:      float         # monetary value PER HECTARE per year (Mode A/B)
                                       # OR price per physical unit (Mode C)
                                       # — always hectare-denominated for A/B,
                                       # see module docstring UNIT CONVENTION.
    currency:           str           # e.g. "IDR"
    physical_unit:      str | None    # e.g. "MgC/ha" — None if Mode A
    physical_per_unit_area:    float | None  # physical quantity PER HECTARE — None if Mode A/C
    stockflow_source:   str | None = None   # v3.2 — "flow:<Type>" | "stock:<Type>" | None
    user_type:           str = "Unspecified"  # v3.5 — economic-unit user of this service
    user_share:          float = 1.0          # v3.5 — this user's fraction (0-1) of the service
    has_explicit_user:    bool = False          # v3.5 — True only if UserType was set in the CSV

    @property
    def has_physical(self) -> bool:
        return self.physical_unit is not None and self.physical_per_unit_area is not None

    @property
    def has_stockflow_source(self) -> bool:
        return self.stockflow_source is not None

    @property
    def stockflow_kind(self) -> str | None:
        """Returns 'flow' or 'stock', or None if not Mode C."""
        if self.stockflow_source is None:
            return None
        return self.stockflow_source.split(":", 1)[0]

    @property
    def stockflow_type_name(self) -> str | None:
        """Returns the FlowTypeId or StockTypeId name, or None if not Mode C."""
        if self.stockflow_source is None:
            return None
        parts = self.stockflow_source.split(":", 1)
        return parts[1] if len(parts) == 2 else None


def load_ecosystem_services(path: str | Path) -> list[EcosystemService]:
    """
    Parse EcosystemServices.csv.

    Required columns:
        StateClassId, ServiceName, ServiceType, ValuePerUnitArea, Currency
        (legacy name ValuePerHa is still accepted — see module docstring)

    Optional columns (Mode B):
        PhysicalUnit, PhysicalValuePerUnitArea
        (legacy name PhysicalValuePerHa is still accepted)

    All ValuePerUnitArea / PhysicalValuePerUnitArea figures are interpreted as
    PER HECTARE regardless of the run's AREA_UNIT — see module docstring.

    Returns
    -------
    list of EcosystemService
    """
    path = Path(path)
    services: list[EcosystemService] = []
    warned_legacy_value_col = False
    warned_legacy_phys_col  = False

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        # Column name resolution (new name preferred, legacy names accepted:
        # "ValuePerUnit"/"PhysicalValuePerUnit" existed briefly pre-3.5.4;
        # "ValuePerHa"/"PhysicalValuePerHa" are the original pre-3.3 names)
        value_col = next(
            (c for c in ("ValuePerUnitArea", "ValuePerUnit", "ValuePerHa") if c in fieldnames),
            "ValuePerUnitArea",
        )
        phys_col = next(
            (c for c in ("PhysicalValuePerUnitArea", "PhysicalValuePerUnit", "PhysicalValuePerHa") if c in fieldnames),
            "PhysicalValuePerUnitArea",
        )

        for i, row in enumerate(reader, start=2):
            state_class  = row.get("StateClassId", "").strip()
            service_name = row.get("ServiceName", "").strip()
            service_type = row.get("ServiceType", "").strip()
            currency     = row.get("Currency", "").strip()

            # Parse monetary value
            if value_col != "ValuePerUnitArea" and not warned_legacy_value_col:
                print(f"  [Warning] '{path.name}' uses legacy column '{value_col}' — "
                      f"still interpreted as per-hectare, but consider renaming to "
                      f"'ValuePerUnitArea' (see accounting/csv_loader.py docstring).")
                warned_legacy_value_col = True
            try:
                value_per_unit_area = float(row.get(value_col, "").strip())
            except (ValueError, AttributeError):
                print(f"  [Warning] Row {i}: invalid {value_col} — skipped")
                continue

            # Validate service type
            if service_type not in VALID_SERVICE_TYPES:
                print(f"  [Warning] Row {i}: unknown ServiceType '{service_type}' "
                      f"— must be one of {VALID_SERVICE_TYPES}")
                continue

            if not state_class or not service_name:
                print(f"  [Warning] Row {i}: missing StateClassId or ServiceName — skipped")
                continue

            # Optional physical columns (Mode B)
            phys_unit = row.get("PhysicalUnit", "").strip() or None
            phys_raw  = row.get(phys_col, "").strip()
            if phys_col != "PhysicalValuePerUnitArea" and phys_raw and not warned_legacy_phys_col:
                print(f"  [Warning] '{path.name}' uses legacy column '{phys_col}' — "
                      f"still interpreted as per-hectare, but consider renaming to "
                      f"'PhysicalValuePerUnitArea'.")
                warned_legacy_phys_col = True
            try:
                phys_per_unit = float(phys_raw) if phys_raw else None
            except ValueError:
                phys_per_unit = None

            # Mode C (v3.2): StockFlowSource overrides physical sourcing
            sf_source_raw = row.get("StockFlowSource", "").strip() or None
            if sf_source_raw is not None:
                if ":" not in sf_source_raw or sf_source_raw.split(":", 1)[0] not in ("flow", "stock"):
                    print(f"  [Warning] Row {i} ({state_class} / {service_name}): "
                          f"invalid StockFlowSource '{sf_source_raw}' — expected "
                          f"'flow:<Type>' or 'stock:<Type>'. Falling back to Mode A/B.")
                    sf_source_raw = None
                else:
                    # Mode C: PhysicalValuePerUnitArea is not used (quantity comes
                    # from the Stock & Flow engine instead), so any value
                    # there is intentionally ignored, not validated as an error.
                    phys_per_unit = None

            # Both must be present for Mode B, or both absent for Mode A/C
            if sf_source_raw is None and (phys_unit is None) != (phys_per_unit is None):
                print(f"  [Warning] Row {i} ({state_class} / {service_name}): "
                      "PhysicalUnit and PhysicalValuePerUnitArea must both be present "
                      "or both absent — treating as Mode A")
                phys_unit = phys_per_unit = None

            # Optional user split (v3.5): UserType / UserShare
            user_type_field = row.get("UserType", "").strip()
            has_explicit_user = bool(user_type_field)
            user_type_raw  = user_type_field or "Unspecified"
            user_share_raw = row.get("UserShare", "").strip()
            try:
                user_share = float(user_share_raw) if user_share_raw else 1.0
            except ValueError:
                print(f"  [Warning] Row {i} ({state_class} / {service_name}): "
                      f"invalid UserShare '{user_share_raw}' — treated as 1.0")
                user_share = 1.0

            services.append(EcosystemService(
                state_class      = state_class,
                service_name     = service_name,
                service_type     = service_type,
                value_per_unit_area    = value_per_unit_area,
                currency         = currency,
                physical_unit    = phys_unit,
                physical_per_unit_area  = phys_per_unit,
                stockflow_source = sf_source_raw,
                user_type        = user_type_raw,
                user_share       = user_share,
                has_explicit_user = has_explicit_user,
            ))

    # v3.5 — warn if a service's UserShare entries don't sum to ~1.0.
    # Only applies to GENUINE user-split groups (every row in the group
    # has an explicit UserType). Multiple rows sharing a service_name
    # with NO UserType set is the older, legitimate pattern of several
    # StockFlowSource components (e.g. stock:AGB + stock:Soil) summing
    # into one named service — that's additive, not a split, and must
    # not be flagged or share-divided.
    _by_key: dict[tuple[str, str], list[EcosystemService]] = {}
    for s in services:
        _by_key.setdefault((s.state_class, s.service_name), []).append(s)
    for (sc, sn), group in _by_key.items():
        if len(group) < 2 or not all(s.has_explicit_user for s in group):
            continue
        total_share = sum(s.user_share for s in group)
        if abs(total_share - 1.0) > 1e-6:
            print(f"  [Warning] '{sc}' / '{sn}': UserShare entries sum to "
                  f"{total_share:.4f}, not 1.0 ({len(group)} user row(s)) — "
                  f"use-table totals for this service won't equal its supply.")
        vals = {s.value_per_unit_area for s in group}
        if len(vals) > 1:
            print(f"  [Warning] '{sc}' / '{sn}': ValuePerUnitArea differs across "
                  f"its {len(group)} UserType rows ({sorted(vals)}) — each row "
                  f"should repeat the SAME total, with UserShare dividing it up, "
                  f"not a different total per user.")

    n_stockflow = sum(s.has_stockflow_source for s in services)
    n_split_use = sum(1 for s in services if s.user_type != "Unspecified")
    print(f"  {len(services)} ecosystem service entries loaded "
          f"({sum(s.has_physical for s in services)} with physical units, "
          f"{n_stockflow} stock/flow-sourced, {n_split_use} with an explicit "
          f"UserType)")
    return services


@dataclass
class AssetValuationParams:
    """One row from AssetValuationParams.csv — NPV parameters for
    SEEAAccount.monetary_asset_account_seea() (SEEA EA Table 10.1)."""
    state_class:              str            # matches StateClass name, or "ALL" for the default
    discount_rate:             float          # real discount rate, e.g. 0.02
    asset_life_years:          int            # years of future service flow summed for NPV
    price_growth_rate:          float = 0.0    # annual price growth; 0.0 = no revaluation modelled
    condition_proxy:            str | None = None   # stock_type name in stock_df, or None
    condition_reference_level:   float | None = None  # ConditionProxy value treated as index 1.0


def load_asset_valuation_params(path: str | Path) -> dict[str, AssetValuationParams]:
    """
    Parse AssetValuationParams.csv into {state_class: AssetValuationParams},
    keyed by StateClassId ("ALL" is kept as its own key and used by
    SEEAAccount as the fallback for any class without its own row).

    Required columns: StateClassId, DiscountRate, AssetLifeYears
    Optional columns: PriceGrowthRate (default 0.0), ConditionProxy,
    ConditionReferenceLevel (both must be present together or both absent).
    """
    path = Path(path)
    params: dict[str, AssetValuationParams] = {}

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=2):
            state_class = row.get("StateClassId", "").strip()
            if not state_class:
                print(f"  [Warning] Row {i}: missing StateClassId — skipped")
                continue
            try:
                discount_rate = float(row.get("DiscountRate", "").strip())
                asset_life    = int(float(row.get("AssetLifeYears", "").strip()))
            except (ValueError, AttributeError):
                print(f"  [Warning] Row {i} ({state_class}): invalid DiscountRate "
                      f"or AssetLifeYears — skipped")
                continue
            growth_raw = row.get("PriceGrowthRate", "").strip()
            try:
                price_growth = float(growth_raw) if growth_raw else 0.0
            except ValueError:
                price_growth = 0.0

            cproxy = row.get("ConditionProxy", "").strip() or None
            cref_raw = row.get("ConditionReferenceLevel", "").strip()
            try:
                cref = float(cref_raw) if cref_raw else None
            except ValueError:
                cref = None
            if (cproxy is None) != (cref is None):
                print(f"  [Warning] Row {i} ({state_class}): ConditionProxy and "
                      f"ConditionReferenceLevel must both be present or both "
                      f"absent — ignoring both.")
                cproxy = cref = None

            if state_class in params:
                print(f"  [Warning] Row {i}: duplicate StateClassId "
                      f"'{state_class}' in AssetValuationParams.csv — "
                      f"overwriting the earlier row.")

            params[state_class] = AssetValuationParams(
                state_class               = state_class,
                discount_rate              = discount_rate,
                asset_life_years           = asset_life,
                price_growth_rate           = price_growth,
                condition_proxy             = cproxy,
                condition_reference_level    = cref,
            )

    n_condition = sum(1 for p in params.values() if p.condition_proxy is not None)
    print(f"  {len(params)} asset valuation parameter row(s) loaded "
          f"({'has' if 'ALL' in params else 'no'} ALL default, "
          f"{n_condition} with a ConditionProxy)")
    return params
