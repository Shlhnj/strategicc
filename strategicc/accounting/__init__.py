"""
strategicc/accounting  —  SEEA-EA accounting module
----------------------------------------------------------
Produces ecosystem extent, physical flow, monetary flow,
transition matrix, and change-in-value accounts from
simulation outputs.
"""

from .seea import SEEAAccount
from .csv_loader import load_ecosystem_services, EcosystemService
from .outputs import (
    save_all_accounts,
    plot_monetary_flows,
    plot_value_by_service,
    plot_transition_heatmap,
)

__all__ = [
    "SEEAAccount", "load_ecosystem_services", "EcosystemService",
    "save_all_accounts", "plot_monetary_flows",
    "plot_value_by_service", "plot_transition_heatmap",
]
