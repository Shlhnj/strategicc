"""
strategicc/accounting/csv_loader.py  —  v3.3
-------------------------------------
Parse EcosystemServices.csv into EcosystemService dataclasses.

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

            services.append(EcosystemService(
                state_class      = state_class,
                service_name     = service_name,
                service_type     = service_type,
                value_per_unit_area    = value_per_unit_area,
                currency         = currency,
                physical_unit    = phys_unit,
                physical_per_unit_area  = phys_per_unit,
                stockflow_source = sf_source_raw,
            ))

    n_stockflow = sum(s.has_stockflow_source for s in services)
    print(f"  {len(services)} ecosystem service entries loaded "
          f"({sum(s.has_physical for s in services)} with physical units, "
          f"{n_stockflow} stock/flow-sourced)")
    return services
