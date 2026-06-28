"""
strategicc/accounting/csv_loader.py
-------------------------------------
Parse EcosystemServices.csv into EcosystemService dataclasses.

CSV format (two modes supported):

Mode A — monetary value per ha only (no physical unit):
    StateClassId, ServiceName, ServiceType, ValuePerHa, Currency
    Mangrove, Ecotourism, Cultural, 12500000, IDR

Mode B — physical unit + monetary value per ha:
    StateClassId, ServiceName, ServiceType, ValuePerHa, Currency, PhysicalUnit, PhysicalValuePerHa
    Mangrove, Carbon Sequestration, Regulating, 97500000, IDR, MgC/ha, 1300

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
    value_per_ha:       float         # monetary value per hectare per year
    currency:           str           # e.g. "IDR"
    physical_unit:      str | None    # e.g. "MgC/ha" — None if Mode A
    physical_per_ha:    float | None  # physical quantity per ha — None if Mode A

    @property
    def has_physical(self) -> bool:
        return self.physical_unit is not None and self.physical_per_ha is not None


def load_ecosystem_services(path: str | Path) -> list[EcosystemService]:
    """
    Parse EcosystemServices.csv.

    Required columns:
        StateClassId, ServiceName, ServiceType, ValuePerHa, Currency

    Optional columns (Mode B):
        PhysicalUnit, PhysicalValuePerHa

    Returns
    -------
    list of EcosystemService
    """
    path = Path(path)
    services: list[EcosystemService] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=2):
            state_class  = row.get("StateClassId", "").strip()
            service_name = row.get("ServiceName", "").strip()
            service_type = row.get("ServiceType", "").strip()
            currency     = row.get("Currency", "").strip()

            # Parse monetary value
            try:
                value_per_ha = float(row.get("ValuePerHa", "").strip())
            except (ValueError, AttributeError):
                print(f"  [Warning] Row {i}: invalid ValuePerHa — skipped")
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
            phys_raw  = row.get("PhysicalValuePerHa", "").strip()
            try:
                phys_per_ha = float(phys_raw) if phys_raw else None
            except ValueError:
                phys_per_ha = None

            # Both must be present for Mode B, or both absent for Mode A
            if (phys_unit is None) != (phys_per_ha is None):
                print(f"  [Warning] Row {i} ({state_class} / {service_name}): "
                      "PhysicalUnit and PhysicalValuePerHa must both be present "
                      "or both absent — treating as Mode A")
                phys_unit = phys_per_ha = None

            services.append(EcosystemService(
                state_class     = state_class,
                service_name    = service_name,
                service_type    = service_type,
                value_per_ha    = value_per_ha,
                currency        = currency,
                physical_unit   = phys_unit,
                physical_per_ha = phys_per_ha,
            ))

    print(f"  {len(services)} ecosystem service entries loaded "
          f"({sum(s.has_physical for s in services)} with physical units)")
    return services
