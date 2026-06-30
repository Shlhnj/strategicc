"""
strategicc/stockflow/csv_loader.py  —  v3.2
-----------------------------------------------
Parsers for all Stock & Flow CSV inputs.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path

from strategicc.io.csv_loader import (
    _strip_type_suffix, _parse_int_or_none, _parse_float_or_none,
)


# ─────────────────────────────────────────────────────────────────────────────
# Simple name-list tables
# ─────────────────────────────────────────────────────────────────────────────

def load_stock_types(path: str | Path) -> list[str]:
    """Parse Stock Type.csv -> list of stock type names."""
    path = Path(path)
    names = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Name", "").strip()
            if name:
                names.append(name)
    print(f"  {len(names)} stock type(s) loaded: {names}")
    return names


def load_flow_types(path: str | Path) -> list[str]:
    """Parse Flow Type.csv -> list of flow type names."""
    path = Path(path)
    names = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Name", "").strip()
            if name:
                names.append(name)
    print(f"  {len(names)} flow type(s) loaded: {names}")
    return names


def load_flow_order(path: str | Path) -> dict[str, int]:
    """Parse Flow Order.csv -> {flow_type_name: order_index}."""
    path = Path(path)
    order: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            ft = _strip_type_suffix(row.get("FlowTypeId", "").strip())
            ordinal = _parse_int_or_none(row.get("Order", ""))
            if ft and ordinal is not None:
                order[ft] = ordinal
    print(f"  Flow order resolved for {len(order)} type(s): "
          f"{sorted(order.items(), key=lambda kv: kv[1])}")
    return order


def load_stock_groups(path: str | Path) -> list[str]:
    """Parse Stock Group.csv -> list of stock group names."""
    path = Path(path)
    names = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = _strip_type_suffix(row.get("Name", "").strip())
            if name:
                names.append(name)
    print(f"  {len(names)} stock group(s) loaded: {names}")
    return names


def load_stock_group_membership(path: str | Path) -> dict[str, list[str]]:
    """Parse Stock Type-Group Membership.csv -> {group_name: [stock_type_names]}."""
    path = Path(path)
    membership: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            stock_type = row.get("StockTypeId", "").strip()
            group      = _strip_type_suffix(row.get("StockGroupId", "").strip())
            if stock_type and group:
                membership.setdefault(group, []).append(stock_type)
    print(f"  Stock group membership: {membership}")
    return membership


# ─────────────────────────────────────────────────────────────────────────────
# State Attribute Type / Values
# ─────────────────────────────────────────────────────────────────────────────

def load_state_attribute_types(path: str | Path) -> list[str]:
    """Parse State Attribute Type.csv -> list of attribute names."""
    path = Path(path)
    names = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Name", "").strip()
            if name:
                names.append(name)
    print(f"  {len(names)} state attribute type(s) loaded: {names}")
    return names


@dataclass
class StateAttributeValueRule:
    """
    One row from State Attribute Values.csv — defines an age-bracketed
    lookup table for a State Attribute (e.g. NPP per age bracket).
    """
    attribute_type: str
    state_class:    str | None
    age_min:        int | None
    age_max:        int | None
    value:          float


def load_state_attribute_values(path: str | Path) -> list[StateAttributeValueRule]:
    """Parse State Attribute Values.csv."""
    path  = Path(path)
    rules: list[StateAttributeValueRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            attr = row.get("StateAttributeTypeId", "").strip()
            val  = _parse_float_or_none(row.get("Value", ""))
            if not attr or val is None:
                continue

            rules.append(StateAttributeValueRule(
                attribute_type = attr,
                state_class    = row.get("StateClassId", "").strip() or None,
                age_min        = _parse_int_or_none(row.get("AgeMin", "")),
                age_max        = _parse_int_or_none(row.get("AgeMax", "")),
                value          = val,
            ))

    n_attrs = len(set(r.attribute_type for r in rules))
    print(f"  {len(rules)} state attribute value rule(s) loaded "
          f"across {n_attrs} attribute type(s)")
    return rules


def lookup_state_attribute(
    rules:          list[StateAttributeValueRule],
    attribute_type: str,
    age:            int,
    state_class:    str | None = None,
) -> float | None:
    """
    Look up the value of a State Attribute for a given age (and optionally
    state class). If multiple rules match, the most specific (state_class
    matches exactly) wins over a wildcard (state_class is None), then the
    narrowest/highest age_min wins.
    """
    candidates = [
        r for r in rules
        if r.attribute_type == attribute_type
        and (r.age_min is None or age >= r.age_min)
        and (r.age_max is None or age <= r.age_max)
        and (r.state_class is None or r.state_class == state_class)
    ]
    if not candidates:
        return None

    exact = [r for r in candidates if r.state_class == state_class]
    pool = exact if exact else candidates

    best = max(pool, key=lambda r: (r.age_min if r.age_min is not None else -1))
    return best.value


# ─────────────────────────────────────────────────────────────────────────────
# Flow Pathways
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowPathwayRule:
    """
    One row from Flow Pathways.csv.

    Semantics (per official ST-Sim Flow Pathways reference):
    - TransitionGroupId blank  -> automatic flow, invoked every timestep
    - TransitionGroupId set    -> flow only invoked when that transition
                                   group fires on a cell this timestep
    - StateAttributeTypeId set -> flow QUANTITY is sourced from the state
                                   attribute lookup (age-indexed)
    - StateAttributeTypeId blank -> flow quantity is a proportion of the
                                   current From-Stock total
    - TargetType:
        "Flow"      -> flow_amount = source_quantity * Multiplier
        "ToStock"   -> Multiplier is the target PROPORTION of pre-flow
                        To-Stock; flow_amount = target_to_stock - pre_flow_to_stock
        "FromStock" -> Multiplier is the target PROPORTION of pre-flow
                        From-Stock; flow_amount = pre_flow_from_stock - target_from_stock
    """
    from_state_class: str | None
    from_age_min:      int | None
    from_stock_type:   str
    to_state_class:    str | None
    to_age_min:        int | None
    to_stock_type:     str
    transition_group:  str | None
    state_attribute:   str | None
    flow_type:         str
    target_type:       str
    multiplier:        float


_TARGET_TYPE_MAP = {
    "": "Flow",
    "flow": "Flow",
    "to stock": "ToStock",
    "tostock": "ToStock",
    "from stock": "FromStock",
    "fromstock": "FromStock",
}


def load_flow_pathways(path: str | Path) -> list[FlowPathwayRule]:
    """Parse Flow Pathways.csv."""
    path  = Path(path)
    rules: list[FlowPathwayRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            from_stock = row.get("FromStockTypeId", "").strip()
            to_stock   = row.get("ToStockTypeId", "").strip()
            flow_type  = _strip_type_suffix(row.get("FlowTypeId", "").strip())
            multiplier = _parse_float_or_none(row.get("Multiplier", ""))

            if not from_stock or not to_stock or not flow_type or multiplier is None:
                continue

            target_raw  = row.get("TargetType", "").strip().lower()
            target_type = _TARGET_TYPE_MAP.get(target_raw, "Flow")

            rules.append(FlowPathwayRule(
                from_state_class = row.get("FromStateClassId", "").strip() or None,
                from_age_min     = _parse_int_or_none(row.get("FromAgeMin", "")),
                from_stock_type  = from_stock,
                to_state_class   = row.get("ToStateClassId", "").strip() or None,
                to_age_min       = _parse_int_or_none(row.get("ToAgeMin", "")),
                to_stock_type    = to_stock,
                transition_group = _strip_type_suffix(
                    row.get("TransitionGroupId", "").strip()
                ) or None,
                state_attribute  = row.get("StateAttributeTypeId", "").strip() or None,
                flow_type        = flow_type,
                target_type      = target_type,
                multiplier       = multiplier,
            ))

    n_automatic = sum(1 for r in rules if r.transition_group is None)
    n_triggered = len(rules) - n_automatic
    print(f"  {len(rules)} flow pathway(s) loaded "
          f"({n_automatic} automatic, {n_triggered} transition-triggered)")
    return rules


# ─────────────────────────────────────────────────────────────────────────────
# Flow Multiplier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowMultiplierRule:
    """One row from Flow Multiplier.csv."""
    flow_type:    str
    distribution: str
    dist_min:     float
    dist_max:     float


def load_flow_multipliers(path: str | Path) -> list[FlowMultiplierRule]:
    """Parse Flow Multiplier.csv."""
    path  = Path(path)
    rules: list[FlowMultiplierRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            flow_type = _strip_type_suffix(row.get("FlowGroupId", "").strip())
            if not flow_type:
                flow_type = _strip_type_suffix(
                    row.get("FlowMultiplierTypeId", "").strip()
                )
            dist = row.get("DistributionType", "").strip()
            dmin = _parse_float_or_none(row.get("DistributionMin", ""))
            dmax = _parse_float_or_none(row.get("DistributionMax", ""))

            if not flow_type or not dist or dmin is None or dmax is None:
                continue

            rules.append(FlowMultiplierRule(
                flow_type    = flow_type,
                distribution = dist,
                dist_min     = dmin,
                dist_max     = dmax,
            ))

    print(f"  {len(rules)} flow multiplier rule(s) loaded")
    return rules


# ─────────────────────────────────────────────────────────────────────────────
# Initial Stock - Non Spatial
# ─────────────────────────────────────────────────────────────────────────────

def load_initial_stock_links(path: str | Path) -> dict[str, str]:
    """Parse Initial Stock - Non Spatial.csv -> {stock_type: state_attribute_type}."""
    path = Path(path)
    links: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            stock = row.get("StockTypeId", "").strip()
            attr  = row.get("StateAttributeTypeId", "").strip()
            if stock and attr:
                links[stock] = attr
    print(f"  {len(links)} initial stock link(s) loaded: {links}")
    return links
