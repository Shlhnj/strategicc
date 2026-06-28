"""
strategicc/core/transitions.py  —  1a. Transition definitions
--------------------------------------------------------
Converts loaded CSV data into the internal lookup structures used by the
engine, and defines the TransitionRecord dataclass for event logging.
"""

from __future__ import annotations
from dataclasses import dataclass
from strategicc.io.csv_loader import StateClass, TransitionRule


# ─────────────────────────────────────────────────────────────────────────────
# TransitionRecord — one fired transition event
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionRecord:
    year:       int
    row:        int
    col:        int
    from_id:    int
    to_id:      int
    group:      str


# ─────────────────────────────────────────────────────────────────────────────
# Internal index structure
# ─────────────────────────────────────────────────────────────────────────────

# TransitionIndex maps:
#   from_class_id (int)  →  list of (to_class_id, base_probability, group_name)
TransitionIndex = dict[int, list[tuple[int, float, str]]]


def build_transition_index(
    rules:   list[TransitionRule],
    classes: dict[int, StateClass],
) -> TransitionIndex:
    """
    Convert a flat list of TransitionRules into a per-source-class lookup dict.

    The engine iterates over source classes and needs O(1) access to all
    outgoing transitions, so we pre-index here rather than searching at runtime.

    Parameters
    ----------
    rules   : output of load_transitions()
    classes : output of load_state_classes()

    Returns
    -------
    TransitionIndex  —  { from_id: [(to_id, probability, group), ...], ... }

    Notes
    -----
    * State class names are matched by their short label (StateLabelXId),
      e.g. "Mangrove" matches "Mangrove:All".
    * Rules with unrecognised source or destination classes are skipped with
      a warning so a bad CSV row never silently kills the whole run.
    """
    # Build reverse lookup: short_name / full_name → id
    name_to_id: dict[str, int] = {}
    for cid, sc in classes.items():
        name_to_id[sc.name]      = cid   # "Mangrove"
        name_to_id[sc.full_name] = cid   # "Mangrove:All"

    index: TransitionIndex = {cid: [] for cid in classes}

    for rule in rules:
        from_id = name_to_id.get(rule.from_class)
        to_id   = name_to_id.get(rule.to_class)

        if from_id is None:
            print(f"  [Warning] Unknown source class '{rule.from_class}' — skipped")
            continue
        if to_id is None:
            print(f"  [Warning] Unknown dest class '{rule.to_class}' — skipped")
            continue

        index[from_id].append((to_id, rule.probability, rule.group))

    return index
