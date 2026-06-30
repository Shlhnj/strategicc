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


# ─────────────────────────────────────────────────────────────────────────────
# Transition size distribution  (v2.5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionSizeRule:
    """
    One row from TransitionSizeDistribution.csv — defines the historical
    patch-size frequency table for a transition group.

    Expected columns (ST-Sim format):
        Transition Type/Group, Maximum Area (Hectares), Relative Amount

    Each row is one BIN of the histogram: "Relative Amount" % of patches
    fall in (0, Maximum Area] hectares, where bins are cumulative — i.e.
    the previous row's Maximum Area is this bin's implicit lower bound.

    Example:
        Fire [Type], 1,    40   →  40% of patches are 0–1 ha
        Fire [Type], 10,   30   →  30% of patches are 1–10 ha
        Fire [Type], 100,  20   →  20% of patches are 10–100 ha
        Fire [Type], 1000, 10   →  10% of patches are 100–1000 ha
    """
    group:          str
    max_area_ha:    float
    relative_amount: float


def load_transition_size_rules(path: str | Path) -> list[TransitionSizeRule]:
    """
    Parse TransitionSizeDistribution.csv.

    Groups present in this file are eligible for patch-growing in the
    engine (v2.5); groups absent keep the independent-cell mechanic.

    Returns
    -------
    list of TransitionSizeRule, ordered as they appear in the file
    (order matters — bins are interpreted as cumulative ranges in
    ascending Maximum Area order per group).
    """
    path  = Path(path)
    rules: list[TransitionSizeRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            group = _strip_type_suffix(
                row.get("Transition Type/Group", "").strip()
            )
            max_area = _parse_float_or_none(row.get("Maximum Area (Hectares)", ""))
            rel_amt  = _parse_float_or_none(row.get("Relative Amount", ""))

            if not group or max_area is None or rel_amt is None:
                continue

            rules.append(TransitionSizeRule(
                group           = group,
                max_area_ha     = max_area,
                relative_amount = rel_amt,
            ))

    n_groups = len(set(r.group for r in rules))
    print(f"  {len(rules)} size distribution bin(s) loaded "
          f"across {n_groups} group(s)")
    return rules


def group_size_bins(
    rules: list[TransitionSizeRule],
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Convert a flat list of TransitionSizeRule into per-group cumulative bins.

    Returns
    -------
    dict[group_name, list[(min_area_ha, max_area_ha, probability)]]
    where probability is the normalised relative_amount (sums to 1.0 per group)
    and min_area_ha is the previous row's max_area_ha (0.0 for the first bin).
    """
    by_group: dict[str, list[TransitionSizeRule]] = {}
    for r in rules:
        by_group.setdefault(r.group, []).append(r)

    result: dict[str, list[tuple[float, float, float]]] = {}
    for group, group_rules in by_group.items():
        total = sum(r.relative_amount for r in group_rules)
        if total <= 0:
            continue
        bins = []
        prev_max = 0.0
        for r in group_rules:
            prob = r.relative_amount / total
            bins.append((prev_max, r.max_area_ha, prob))
            prev_max = r.max_area_ha
        result[group] = bins

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Transition targets  (v3.1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionTargetRule:
    """
    One row from TransitionTargets.csv — an area-based override that
    replaces (or scales) a transition group's probability-derived budget
    for a given iteration/timestep.

    Expected columns:
        Iteration, Timestep, StratumId, SecondaryStratumId, TertiaryStratumId,
        TransitionGroupId, Amount, DistributionType, DistributionFrequencyId,
        DistributionSD, DistributionMin, DistributionMax

    Persistence semantics (matches the source ST-Sim behaviour):
        A target specified at Timestep T applies from T onward, for every
        subsequent timestep, until a NEW record for the same TransitionGroupId
        is encountered at a later timestep. A record with `amount=None` and
        no distribution explicitly turns the target OFF from that timestep
        onward (group reverts to its normal probability-derived budget).

    Only `group`, `timestep`, and `amount` are used in v3.1 — Iteration and
    Stratum scoping are parsed but not yet applied (single-stratum landscapes
    only for now).
    """
    group:         str
    timestep:      int | None   # None = applies from t=0
    amount:        float | None # target area in the engine's configured AREA_UNIT; None = target OFF
    iteration:     int | None = None
    distribution_type: str | None = None
    distribution_min:  float | None = None
    distribution_max:  float | None = None


def load_transition_targets(path: str | Path) -> list[TransitionTargetRule]:
    """
    Parse TransitionTargets.csv.

    Returns
    -------
    list of TransitionTargetRule, ordered as they appear in the file.
    Rows are NOT pre-resolved into a per-timestep lookup here — see
    strategicc.core.targets.resolve_targets_per_timestep() for that.
    """
    path  = Path(path)
    rules: list[TransitionTargetRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            group = _strip_type_suffix(row.get("TransitionGroupId", "").strip())
            if not group:
                continue

            amount_raw = row.get("Amount", "").strip()
            dist_raw   = row.get("DistributionType", "").strip()

            # A row with both Amount and DistributionType blank explicitly
            # turns the target OFF for this group from this timestep onward.
            amount = _parse_float_or_none(amount_raw) if amount_raw else None

            rules.append(TransitionTargetRule(
                group              = group,
                timestep           = _parse_int_or_none(row.get("Timestep", "")),
                amount             = amount,
                iteration          = _parse_int_or_none(row.get("Iteration", "")),
                distribution_type  = dist_raw or None,
                distribution_min   = _parse_float_or_none(row.get("DistributionMin", "")),
                distribution_max   = _parse_float_or_none(row.get("DistributionMax", "")),
            ))

    print(f"  {len(rules)} transition target rule(s) loaded "
          f"across {len(set(r.group for r in rules))} group(s)")
    return rules


# ─────────────────────────────────────────────────────────────────────────────
# Transition adjacency settings and multipliers  (v3.1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionAdjacencySettingRule:
    """
    One row from TransitionAdjacencySetting.csv.

    Expected columns:
        TransitionGroupId, StateClassId, StateAttributeTypeId,
        NeighborhoodRadius, UpdateFrequency

    A row's presence for a group is the signal that the group uses
    CSV-driven adjacency multipliers instead of the global scalar
    ADJACENCY_STRENGTH fallback. In v3.1 only the simple scalar case is
    supported: StateAttributeTypeId, NeighborhoodRadius, and
    UpdateFrequency are parsed but not yet used — they are reserved for
    the attribute-based interpolated-lookup mechanic (a separate, larger
    feature requiring State Attribute Type/Values support).
    """
    group:                  str
    state_class:            str | None = None
    state_attribute_type:   str | None = None
    neighborhood_radius:    float | None = None
    update_frequency:       int | None = None


@dataclass
class TransitionAdjacencyMultiplierRule:
    """
    One row from TransitionAdjacencyMultipliers.csv.

    Expected columns:
        Iteration, Timestep, StratumId, SecondaryStratumId, TertiaryStratumId,
        TransitionGroupId, AttributeValue, Amount, DistributionType,
        DistributionFrequencyId, DistributionSD, DistributionMin, DistributionMax

    In v3.1, only rows with AttributeValue blank are used — these apply
    as a flat per-group strength (Amount), equivalent to the existing
    ADJACENCY_STRENGTH mechanic but specified per-group via CSV instead
    of a single global constant. Rows with AttributeValue populated are
    parsed (for forward compatibility / round-tripping) but are not yet
    used by the engine — that requires the attribute-based neighbourhood
    lookup mechanic, not implemented in v3.1.
    """
    group:          str
    amount:         float
    attribute_value: float | None = None
    iteration:      int | None = None
    timestep:       int | None = None


def load_transition_adjacency_setting(
    path: str | Path,
) -> list[TransitionAdjacencySettingRule]:
    """
    Parse TransitionAdjacencySetting.csv.

    Returns
    -------
    list of TransitionAdjacencySettingRule. The set of group names present
    in this file (via .group) determines which groups use CSV-driven
    adjacency strength instead of the global ADJACENCY_STRENGTH /
    STRICT_EXPANSION_GROUPS fallback.
    """
    path  = Path(path)
    rules: list[TransitionAdjacencySettingRule] = []

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            group = _strip_type_suffix(row.get("TransitionGroupId", "").strip())
            if not group:
                continue

            rules.append(TransitionAdjacencySettingRule(
                group                 = group,
                state_class           = row.get("StateClassId", "").strip() or None,
                state_attribute_type  = row.get("StateAttributeTypeId", "").strip() or None,
                neighborhood_radius   = _parse_float_or_none(row.get("NeighborhoodRadius", "")),
                update_frequency      = _parse_int_or_none(row.get("UpdateFrequency", "")),
            ))

    print(f"  {len(rules)} adjacency setting rule(s) loaded "
          f"for {len(set(r.group for r in rules))} group(s)")
    return rules


def load_transition_adjacency_multipliers(
    path: str | Path,
) -> list[TransitionAdjacencyMultiplierRule]:
    """
    Parse TransitionAdjacencyMultipliers.csv.

    Rows with a blank Amount are skipped (malformed — Amount is required
    per the source format). Rows with AttributeValue populated are parsed
    but flagged with a one-time warning summary, since the interpolated
    attribute-lookup mechanic is not yet implemented (v3.1 supports the
    simple scalar case only).

    Returns
    -------
    list of TransitionAdjacencyMultiplierRule
    """
    path  = Path(path)
    rules: list[TransitionAdjacencyMultiplierRule] = []
    n_attribute_rows = 0

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            group  = _strip_type_suffix(row.get("TransitionGroupId", "").strip())
            amount = _parse_float_or_none(row.get("Amount", ""))
            if not group or amount is None:
                continue

            attr_val = _parse_float_or_none(row.get("AttributeValue", ""))
            if attr_val is not None:
                n_attribute_rows += 1

            rules.append(TransitionAdjacencyMultiplierRule(
                group           = group,
                amount          = amount,
                attribute_value = attr_val,
                iteration       = _parse_int_or_none(row.get("Iteration", "")),
                timestep        = _parse_int_or_none(row.get("Timestep", "")),
            ))

    if n_attribute_rows:
        print(f"  [Warning] {n_attribute_rows} row(s) have a populated "
              f"AttributeValue — attribute-based interpolated adjacency "
              f"lookup is not yet implemented; these rows' Amount values "
              f"will be ignored unless a blank-AttributeValue row also "
              f"exists for the same group.")

    print(f"  {len(rules)} adjacency multiplier rule(s) loaded "
          f"for {len(set(r.group for r in rules))} group(s)")
    return rules


def build_adjacency_strength_map(
    multiplier_rules: list[TransitionAdjacencyMultiplierRule],
) -> dict[str, float]:
    """
    Collapse TransitionAdjacencyMultiplierRule rows into a simple
    {group: strength} map, using only the blank-AttributeValue (flat
    scalar) rows. If multiple blank-AttributeValue rows exist for the
    same group (e.g. across different timesteps), the LAST one in file
    order wins — matches the persistence-style semantics used elsewhere
    in the format (most recent record applies).

    Returns
    -------
    dict[group_name, strength] — ready to use as a per-group replacement
    for the global ADJACENCY_STRENGTH constant.
    """
    strength_map: dict[str, float] = {}
    for rule in multiplier_rules:
        if rule.attribute_value is None:
            strength_map[rule.group] = rule.amount
    return strength_map
