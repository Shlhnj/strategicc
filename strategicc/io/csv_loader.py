"""
strategicc/io/csv_loader.py
---------------------
Parse ST-Sim–format CSV files into plain Python dicts / lists.

Supported files
---------------
StateClasses.csv                 → load_state_classes()
Transitions.csv                  → load_transitions()
TransitionSpatialMultipliers.csv → load_spatial_mult_index()
TransitionMultipliers.csv        → load_transition_multipliers()   v1.1
"""

from __future__ import annotations
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StateClass:
    """One row from StateClasses.csv."""
    id:        int
    name:      str          # e.g. "Mangrove"
    full_name: str          # e.g. "Mangrove:All"
    color:     tuple[int, int, int, int]   # (A, R, G, B) as in ST-Sim


@dataclass
class TransitionRule:
    """One probabilistic transition row from Transitions.csv."""
    from_class:    str
    to_class:      str
    group:         str
    probability:   float
    iteration:     int | None = None
    timestep:      int | None = None
    age_min:       int | None = None
    age_max:       int | None = None
    age_reset:     bool       = True    # v2.3 — reset age to 0 on transition
    age_relative:  int | None = None    # v2.3 — relative age after transition


@dataclass
class SpatialMultEntry:
    """One row from TransitionSpatialMultipliers.csv."""
    group:    str           # TransitionGroupId  e.g. "Inundation"
    filename: str           # MultiplierFileName
    iteration: int | None = None
    timestep:  int | None = None


@dataclass
class TransitionMultiplierRule:
    """
    One row from TransitionMultipliers.csv  (v1.1).

    A scalar multiplier is sampled from DistributionType(DistributionMin,
    DistributionMax) once per timestep and applied to all base probabilities
    for that group in that timestep.

    Supported DistributionType values: 'Uniform'  (others reserved for v1.2+)
    """
    group:            str           # TransitionGroupId (suffix stripped)
    distribution:     str           # e.g. "Uniform"
    dist_min:         float
    dist_max:         float
    amount:           float | None = None   # optional fixed amount (unused if dist set)
    iteration:        int   | None = None
    timestep:         int   | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_color(raw: str) -> tuple[int, int, int, int]:
    """
    Parse ST-Sim ARGB color string  "255,0,128,255"  →  (255, 0, 128, 255).
    Falls back to opaque black on any parse error.
    """
    try:
        parts = [int(x.strip()) for x in raw.strip('"').split(",")]
        if len(parts) == 4:
            return tuple(parts)           # type: ignore[return-value]
        if len(parts) == 3:
            return (255, *parts)          # assume fully opaque
    except (ValueError, AttributeError):
        pass
    return (255, 0, 0, 0)


def _strip_type_suffix(group: str) -> str:
    """'Agriculture_expansion [Type]'  →  'Agriculture_expansion'"""
    return re.sub(r"\s*\[.*?\]", "", group).strip()


def _parse_int_or_none(val: str) -> int | None:
    val = val.strip()
    return int(val) if val else None


def _parse_float_or_none(val: str) -> float | None:
    val = val.strip()
    return float(val) if val else None


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_state_classes(path: str | Path) -> dict[int, StateClass]:
    """
    Parse StateClasses.csv.

    Expected columns (ST-Sim format):
        Name, StateLabelXId, StateLabelYId, Id, Color, Legend, Description, IsAutoName

    Returns
    -------
    dict mapping class_id (int) → StateClass
    """
    path = Path(path)
    classes: dict[int, StateClass] = {}

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid   = int(row["Id"].strip())
            name  = row["StateLabelXId"].strip()
            full  = row["Name"].strip()
            color = _parse_color(row.get("Color", "255,0,0,0"))
            classes[cid] = StateClass(id=cid, name=name, full_name=full, color=color)

    return classes


def load_transitions(path: str | Path) -> list[TransitionRule]:
    """
    Parse Transitions.csv (ST-Sim probabilistic transitions block only).

    Rows whose Probability column is empty or non-numeric are skipped
    (handles the size-distribution rows appended at the bottom of some exports).

    Expected columns:
        Iteration, Timestep, StratumIdSource, StateClassIdSource,
        StratumIdDest, StateClassIdDest, SecondaryStratumId, TertiaryStratumId,
        TransitionTypeId, Probability, Proportion, AgeMin, AgeMax,
        AgeRelative, AgeReset, TSTMin, TSTMax, TSTRelative

    Returns
    -------
    list of TransitionRule (one per valid row)
    """
    path  = Path(path)
    rules: list[TransitionRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            prob = _parse_float_or_none(row.get("Probability", ""))
            if prob is None:
                continue   # skip size-distribution rows / blank rows

            from_cls = row["StateClassIdSource"].strip()
            to_cls   = row["StateClassIdDest"].strip()
            group    = _strip_type_suffix(row["TransitionTypeId"].strip())

            if not from_cls or not to_cls or not group:
                continue

            rules.append(TransitionRule(
                from_class    = from_cls,
                to_class      = to_cls,
                group         = group,
                probability   = prob,
                iteration     = _parse_int_or_none(row.get("Iteration", "")),
                timestep      = _parse_int_or_none(row.get("Timestep", "")),
                age_min       = _parse_int_or_none(row.get("AgeMin", "")),
                age_max       = _parse_int_or_none(row.get("AgeMax", "")),
                age_reset     = row.get("AgeReset", "").strip().lower() != "no",
                age_relative  = _parse_int_or_none(row.get("AgeRelative", "")),
            ))

    return rules


def load_spatial_mult_index(path: str | Path) -> list[SpatialMultEntry]:
    """
    Parse TransitionSpatialMultipliers.csv.

    Expected columns:
        Iteration, Timestep, TransitionGroupId, TransitionMultiplierTypeId,
        MultiplierFileName

    Returns
    -------
    list of SpatialMultEntry
    """
    path    = Path(path)
    entries: list[SpatialMultEntry] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = row.get("MultiplierFileName", "").strip()
            group = _strip_type_suffix(row.get("TransitionGroupId", "").strip())
            if not fname or not group:
                continue
            entries.append(SpatialMultEntry(
                group     = group,
                filename  = fname,
                iteration = _parse_int_or_none(row.get("Iteration", "")),
                timestep  = _parse_int_or_none(row.get("Timestep", "")),
            ))

    return entries


def load_transition_multipliers(path: str | Path) -> list[TransitionMultiplierRule]:
    """
    Parse TransitionMultipliers.csv  (v1.1).

    Expected columns (ST-Sim format):
        Iteration, Timestep, StratumId, SecondaryStratumId, TertiaryStratumId,
        StateClassId, AgeMin, AgeMax, TSTGroupId, TSTMin, TSTMax,
        TransitionGroupId, TransitionMultiplierTypeId,
        Amount, DistributionType, DistributionFrequencyId,
        DistributionSD, DistributionMin, DistributionMax

    Only TransitionGroupId, DistributionType, DistributionMin, DistributionMax
    are used in v1.1.  Rows missing any of these are skipped with a warning.

    Returns
    -------
    list of TransitionMultiplierRule
    """
    path  = Path(path)
    rules: list[TransitionMultiplierRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            group = _strip_type_suffix(row.get("TransitionGroupId", "").strip())
            dist  = row.get("DistributionType", "").strip()
            dmin  = _parse_float_or_none(row.get("DistributionMin", ""))
            dmax  = _parse_float_or_none(row.get("DistributionMax", ""))

            if not group or not dist or dmin is None or dmax is None:
                continue

            if dmin > dmax:
                print(f"  [Warning] TransitionMultipliers: "
                      f"DistributionMin > DistributionMax for '{group}' — swapping")
                dmin, dmax = dmax, dmin

            rules.append(TransitionMultiplierRule(
                group        = group,
                distribution = dist,
                dist_min     = dmin,
                dist_max     = dmax,
                amount       = _parse_float_or_none(row.get("Amount", "")),
                iteration    = _parse_int_or_none(row.get("Iteration", "")),
                timestep     = _parse_int_or_none(row.get("Timestep", "")),
            ))

    print(f"  {len(rules)} transition multiplier rule(s) loaded")
    return rules


# ─────────────────────────────────────────────────────────────────────────────
# Initial age assumptions  (v2.3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InitialAgeRule:
    """
    One row from InitialAge.csv — assumed starting age per class when no
    age raster is available.

    Expected columns:
        StateClassId, AgeMean, AgeSD, AgeMin, AgeMax

    AgeMean / AgeSD define a truncated normal distribution sampled per cell.
    AgeMin / AgeMax are hard clamps (defaults: 0 / 999).
    """
    state_class: str
    age_mean:    float
    age_sd:      float  = 0.0
    age_min:     int    = 0
    age_max:     int    = 999


def load_initial_age_rules(path: str | Path) -> list[InitialAgeRule]:
    """
    Parse InitialAge.csv.

    Returns
    -------
    list of InitialAgeRule, one per state class row
    """
    path  = Path(path)
    rules: list[InitialAgeRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cls_name = row.get("StateClassId", "").strip()
            mean_raw = _parse_float_or_none(row.get("AgeMean", ""))
            if not cls_name or mean_raw is None:
                continue
            rules.append(InitialAgeRule(
                state_class = cls_name,
                age_mean    = mean_raw,
                age_sd      = float(row.get("AgeSD",  "0").strip() or 0),
                age_min     = int(  row.get("AgeMin", "0").strip() or 0),
                age_max     = int(  row.get("AgeMax", "999").strip() or 999),
            ))

    print(f"  {len(rules)} initial age rule(s) loaded")
    return rules
