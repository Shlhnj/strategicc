"""
strategicc/stockflow  —  v3.2
--------------------------------
Stock and Flow accounting: per-cell, per-timestep tracking of material
quantities moving between pools (Stock Types) via Flow Pathways, either
automatically (age-driven) or triggered by transitions.
"""

from .csv_loader import (
    load_stock_types,
    load_flow_types,
    load_flow_order,
    load_stock_groups,
    load_stock_group_membership,
    load_state_attribute_types,
    load_state_attribute_values,
    load_flow_pathways,
    load_flow_multipliers,
    load_initial_stock_links,
    lookup_state_attribute,
    StateAttributeValueRule,
    FlowPathwayRule,
    FlowMultiplierRule,
)
from .engine import (
    init_stocks,
    run_flows_for_timestep,
    sample_flow_multipliers,
    build_age_attribute_cache,
    FlowRecord,
)
from .aggregation import (
    aggregate_stock_by_class,
    aggregate_flow_by_class,
    aggregate_flow_by_type,
    build_asset_account,
)

__all__ = [
    "load_stock_types", "load_flow_types", "load_flow_order",
    "load_stock_groups", "load_stock_group_membership",
    "load_state_attribute_types", "load_state_attribute_values",
    "load_flow_pathways", "load_flow_multipliers",
    "load_initial_stock_links", "lookup_state_attribute",
    "StateAttributeValueRule", "FlowPathwayRule", "FlowMultiplierRule",
    "init_stocks", "run_flows_for_timestep", "sample_flow_multipliers",
    "build_age_attribute_cache", "FlowRecord",
    "aggregate_stock_by_class", "aggregate_flow_by_class",
    "aggregate_flow_by_type", "build_asset_account",
]
