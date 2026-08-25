"""
strategicc/stockflow/engine.py  —  v3.3
-------------------------------------------
Per-cell, per-timestep stock and flow computation.

v3.3 changes (strategicc 3.18) — BUG FIX
-------------------------------------------
init_stocks()/build_age_attribute_cache() now accept an optional
class_map (+ classes dict) and use it. Previously, initial stock
seeding always called lookup_state_attribute() with state_class=None,
which only matches WILDCARD rules (a State Attribute Values.csv row
with a blank StateClassId). Any rule scoped to a specific class (e.g.
"Mangrove:All" — the normal, correct way to write this table, since
initial biomass/soil carbon legitimately differs by ecosystem type)
could never match, so initial stocks came out all-zero regardless of
what the CSV actually specified, for any project using class-scoped
State Attribute rules (which is the expected/standard usage, not an
edge case). Nothing after t=0 creates stock from nothing, so this
silently zeroed every downstream stock-linked value for the entire
run. Fixed by threading the initial LULC array through as class_map,
resolving each cell's StateClass.full_name (matching the CSV's
StateClassId column format) before the lookup. Backward compatible:
omitting class_map/classes keeps the old (only-wildcards-match)
behavior rather than erroring. See tests/test_stock_initialization.py.

Design
------
Stocks are tracked as a dict of float32 arrays, one per Stock Type, shape
(rows, cols), persisting across timesteps. Each timestep, flows are
processed in Flow Order sequence. For each pathway, eligible cells are
determined (automatic = all cells matching age/class gates; transition-
triggered = only cells where that transition group fired this timestep),
then flow_amount is computed according to TargetType (Flow / ToStock /
FromStock), then subtracted from From-Stock and added to To-Stock.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from strategicc.stockflow.csv_loader import (
    FlowPathwayRule, FlowMultiplierRule, StateAttributeValueRule,
    lookup_state_attribute,
)
from strategicc.core.transitions import TransitionRecord


@dataclass
class FlowRecord:
    """One aggregate flow event: pathway fired, total quantity moved."""
    year:             int
    flow_type:        str
    from_stock:       str
    to_stock:         str
    transition_group: str | None
    total_amount:     float
    by_class:         dict[str, float] | None = None   # v3.2 — {class_name: amount}


def sample_flow_multipliers(
    rules: list[FlowMultiplierRule],
    rng:   np.random.Generator,
) -> dict[str, float]:
    """Sample one stochastic multiplier per flow type for this timestep."""
    result: dict[str, float] = {}
    for rule in rules:
        if rule.distribution.lower() == "uniform":
            result[rule.flow_type] = float(rng.uniform(rule.dist_min, rule.dist_max))
    return result


def build_age_attribute_cache(
    age_map:          np.ndarray,
    state_attr_rules: list[StateAttributeValueRule],
    attribute_type:   str,
    class_map:        np.ndarray | None = None,
    classes:          dict | None = None,
) -> np.ndarray:
    """
    Vectorise the age-bracket lookup for a State Attribute across an
    entire age raster (per-cell Python lookups would be far too slow).

    v3.18: added class_map/classes. Without them (the old call
    signature), every lookup passes state_class=None — which
    lookup_state_attribute() only matches against WILDCARD rules
    (State Attribute Values.csv rows with a blank StateClassId). Any
    rule scoped to a specific class (e.g. "Mangrove:All", the normal,
    expected way to write this table, since e.g. biomass legitimately
    differs by ecosystem type) could never match, silently returning
    0.0 for every cell regardless of what the CSV actually specified —
    this was the root cause of initial stocks coming out all-zero
    whenever State Attribute Values.csv used class-scoped rows.

    When class_map (per-cell class ID array, e.g. the initial LULC
    raster) and classes (id -> StateClass, for full_name resolution —
    State Attribute Values.csv's StateClassId column matches
    StateClass.full_name, e.g. "Mangrove:All", not the short .name)
    are supplied, the lookup is done per (age, class) pair instead of
    per age alone, so class-scoped rules are actually used.

    Returns
    -------
    float32 array, shape matching age_map. Cells with no matching rule
    get value 0.0.
    """
    out = np.zeros(age_map.shape, dtype=np.float32)

    if class_map is None or classes is None:
        # Backward-compatible path: no class info available, only
        # wildcard (state_class=None) rules can ever match.
        unique_ages = np.unique(age_map)
        age_to_value: dict[int, float] = {}
        for age in unique_ages:
            val = lookup_state_attribute(
                state_attr_rules, attribute_type, int(age), state_class=None
            )
            age_to_value[int(age)] = val if val is not None else 0.0
        for age, val in age_to_value.items():
            out[age_map == age] = val
        return out

    full_name_by_id = {cid: sc.full_name for cid, sc in classes.items()}
    unique_class_ids = np.unique(class_map)
    for class_id in unique_class_ids:
        class_name = full_name_by_id.get(int(class_id))
        if class_name is None:
            continue
        cls_mask = (class_map == class_id)
        unique_ages = np.unique(age_map[cls_mask])
        for age in unique_ages:
            val = lookup_state_attribute(
                state_attr_rules, attribute_type, int(age), state_class=class_name
            )
            if val is None:
                continue
            out[cls_mask & (age_map == age)] = val
    return out


def init_stocks(
    stock_types:      list[str],
    shape:            tuple[int, int],
    initial_links:    dict[str, str],
    state_attr_rules: list[StateAttributeValueRule],
    age_map:          np.ndarray | None,
    class_map:        np.ndarray | None = None,
    classes:          dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Build the initial (t=0) stock arrays. For each stock type linked via
    Initial Stock - Non Spatial.csv, the starting quantity per cell is the
    linked State Attribute's value looked up at that cell's initial age
    (age=0 if no age tracking), AND at that cell's initial state class if
    class_map/classes are supplied (v3.18 — see build_age_attribute_cache()
    docstring for why this matters: State Attribute Values.csv rows are
    normally scoped to a specific class, e.g. "Mangrove:All", and without
    class information those rows can never match, silently zeroing every
    stock regardless of what the CSV specifies).
    """
    stocks: dict[str, np.ndarray] = {}
    full_name_by_id = (
        {cid: sc.full_name for cid, sc in classes.items()} if classes else {}
    )
    for stock_type in stock_types:
        arr = np.zeros(shape, dtype=np.float32)
        attr = initial_links.get(stock_type)
        if attr is not None:
            if age_map is not None:
                arr = build_age_attribute_cache(
                    age_map, state_attr_rules, attr,
                    class_map=class_map, classes=classes,
                )
            elif class_map is not None and classes is not None:
                # No age tracking, but we do know each cell's class —
                # look up at age=0 per class rather than one global value.
                for class_id in np.unique(class_map):
                    class_name = full_name_by_id.get(int(class_id))
                    if class_name is None:
                        continue
                    val = lookup_state_attribute(
                        state_attr_rules, attr, age=0, state_class=class_name
                    )
                    if val is not None:
                        arr[class_map == class_id] = val
            else:
                val = lookup_state_attribute(state_attr_rules, attr, age=0)
                if val is not None:
                    arr[:] = val
        stocks[stock_type] = arr
    return stocks


def run_flows_for_timestep(
    stocks:             dict[str, np.ndarray],
    pathways:            list[FlowPathwayRule],
    flow_order:           dict[str, int],
    year_transitions:      list[TransitionRecord],
    age_map:                np.ndarray | None,
    state_attr_rules:        list[StateAttributeValueRule],
    classes:                  dict,
    year:                      int,
    flow_mult_sample:           dict[str, float],
    lulc_map:                    np.ndarray | None = None,
    default_flow_order:          int = 999,
) -> tuple[dict[str, np.ndarray], list[FlowRecord]]:
    """
    Run all flow pathways for one timestep, mutating and returning updated
    stock arrays plus a log of aggregate FlowRecords.

    Parameters
    ----------
    lulc_map : current LULC class-id array, shape matching stocks. Required
               for pathways that specify a FromStateClassId gate — without
               it, such pathways apply to ALL cells regardless of class
               (a warning is printed once if this occurs).
    """
    shape = next(iter(stocks.values())).shape

    fired_by_group: dict[str, np.ndarray] = {}
    for rec in year_transitions:
        mask = fired_by_group.setdefault(
            rec.group, np.zeros(shape, dtype=bool)
        )
        mask[rec.row, rec.col] = True

    name_to_id: dict[str, int] = {}
    for cid, sc in classes.items():
        name_to_id[sc.name]      = cid   # short name, e.g. "Mangrove"
        name_to_id[sc.full_name] = cid   # full label, e.g. "Mangrove:All"

    ordered_pathways = sorted(
        pathways,
        key=lambda r: flow_order.get(r.flow_type, default_flow_order)
    )

    flow_records: list[FlowRecord] = []

    for rule in ordered_pathways:
        if rule.from_stock_type not in stocks or rule.to_stock_type not in stocks:
            continue

        # ── Eligibility mask ────────────────────────────────────────────
        if rule.transition_group is None:
            eligible = np.ones(shape, dtype=bool)
        else:
            eligible = fired_by_group.get(
                rule.transition_group, np.zeros(shape, dtype=bool)
            )
            if not eligible.any():
                continue

        # From State Class gate — restricts the flow to cells currently
        # in the specified class. Without this, automatic flows (e.g.
        # NPP) would incorrectly apply to every cell in the landscape
        # regardless of land cover.
        if rule.from_state_class is not None:
            if lulc_map is not None:
                from_id = name_to_id.get(rule.from_state_class)
                if from_id is not None:
                    eligible = eligible & (lulc_map == from_id)
                else:
                    eligible = np.zeros(shape, dtype=bool)
            # If lulc_map is None, the gate cannot be applied — eligible
            # remains unrestricted by class (documented limitation).

        if rule.from_age_min is not None and age_map is not None:
            eligible = eligible & (age_map >= rule.from_age_min)

        if not eligible.any():
            continue

        # ── Effective multiplier (with stochastic flow multiplier) ──────
        t_mult = flow_mult_sample.get(rule.flow_type, 1.0)
        eff_multiplier = rule.multiplier * t_mult

        from_stock = stocks[rule.from_stock_type]
        to_stock   = stocks[rule.to_stock_type]

        # ── Source quantity ──────────────────────────────────────────────
        # v3.18: pass class_map/classes through here too — this call site
        # has the SAME bug build_age_attribute_cache()/init_stocks() had:
        # without class info, only wildcard (blank StateClassId) State
        # Attribute rules can ever match. lulc_map/classes are already
        # available as parameters to this function, just weren't being
        # forwarded. A state_attribute-sourced flow whose rule is
        # class-scoped (unlike e.g. a pure age-scaled NPP rule, which
        # happens to be a wildcard and was unaffected) would otherwise
        # silently source zero regardless of its current LULC class.
        if rule.state_attribute is not None:
            if age_map is not None:
                source_qty = build_age_attribute_cache(
                    age_map, state_attr_rules, rule.state_attribute,
                    class_map=lulc_map, classes=classes,
                )
            else:
                source_qty = np.zeros(shape, dtype=np.float32)
        else:
            source_qty = from_stock

        # ── Flow amount, branching on TargetType ─────────────────────────
        if rule.target_type == "ToStock":
            target_to = to_stock * eff_multiplier
            flow_amount = target_to - to_stock
        elif rule.target_type == "FromStock":
            target_from = from_stock * eff_multiplier
            flow_amount = from_stock - target_from
        else:  # "Flow" (default)
            flow_amount = source_qty * eff_multiplier

        flow_amount = np.where(eligible, flow_amount, 0.0).astype(np.float32)

        # Only constrain flow_amount by the From-Stock's own quantity for
        # the simple "Flow" target type when sourced from the stock itself
        # (state_attribute is None) — this is the case where the From-Stock
        # is genuinely being depleted by its own current contents (e.g.
        # Biomass -> Atmosphere emission proportional to current biomass).
        # The "ToStock"/"FromStock" target types and any state-attribute-
        # sourced flow represent an externally-driven target rate, not a
        # depletion of From-Stock's own pool, so they are NOT clipped here.
        if rule.target_type == "Flow" and rule.state_attribute is None:
            flow_amount = np.clip(flow_amount, -from_stock, from_stock)

        if rule.target_type == "Flow":
            flow_amount = np.maximum(flow_amount, 0.0)

        stocks[rule.from_stock_type] = np.clip(
            from_stock - flow_amount, 0.0, None
        ).astype(np.float32)
        stocks[rule.to_stock_type] = np.clip(
            to_stock + flow_amount, 0.0, None
        ).astype(np.float32)

        total = float(flow_amount.sum())
        if total != 0.0:
            by_class: dict[str, float] | None = None
            if lulc_map is not None:
                id_to_name = {cid: sc.name for cid, sc in classes.items()}
                by_class = {}
                for cid, name in id_to_name.items():
                    class_total = float(flow_amount[lulc_map == cid].sum())
                    if class_total != 0.0:
                        by_class[name] = class_total

            flow_records.append(FlowRecord(
                year             = year,
                flow_type        = rule.flow_type,
                from_stock       = rule.from_stock_type,
                to_stock         = rule.to_stock_type,
                transition_group = rule.transition_group,
                total_amount     = total,
                by_class         = by_class,
            ))

    return stocks, flow_records
