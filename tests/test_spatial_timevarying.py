"""
tests/test_spatial_timevarying.py  —  v3.15
Unit tests for time-varying (per-Timestep) spatial multipliers:
load_spatial_multipliers() loading every entry instead of collapsing by
group, resolve_spatial_multipliers_per_timestep() forward-filling them,
and get_multiplier() resolving the right array (or ones) for a given t.

Tests of pure forward-fill/lookup logic (resolve_spatial_multipliers_per_
timestep, get_multiplier) construct the "loaded" dict directly rather than
going through actual raster files, since load_spatial_multipliers() also
applies power-sharpening (strategicc.config.SHARPENING_POWER) which is an
orthogonal concern with its own project-specific per-group tuning — mixing
the two would make these tests fragile to unrelated config changes.
Tests that exercise load_spatial_multipliers() itself (sorting order,
missing-file handling) use real synthetic TIFFs but only assert on
timestep bookkeeping / shape / "is it ones", never on exact post-power
values.
"""

import numpy as np
from PIL import Image

from strategicc.core.spatial import (
    load_spatial_multipliers,
    resolve_spatial_multipliers_per_timestep,
    get_multiplier,
)
from strategicc.io.csv_loader import SpatialMultEntry

ONES = np.ones((4, 4), dtype=np.float32)


def _write_tif(path, value):
    arr = np.full((4, 4), value, dtype=np.float32)
    Image.fromarray(arr).save(str(path), tiffinfo={33550: (30.0, 30.0, 0.0)})


def _const(value):
    return np.full((4, 4), value, dtype=np.float32)


# ── resolve_spatial_multipliers_per_timestep() / get_multiplier() ──────────
# (pure logic, arrays constructed directly — no raster I/O, no power curve)

def test_single_static_entry_applies_at_every_timestep():
    """Pre-v3.15 convention: one entry, timestep=None -> applies at every t,
    and the SAME array object is reused (no accidental copying per t)."""
    loaded = {"Mangrove_recruitment": [(None, _const(0.5))]}
    resolved = resolve_spatial_multipliers_per_timestep(loaded, n_timesteps=5)

    for t in range(5):
        mult = get_multiplier(resolved, "Mangrove_recruitment", (4, 4), t=t)
        assert np.allclose(mult, 0.5)
    assert resolved[0]["Mangrove_recruitment"] is resolved[4]["Mangrove_recruitment"]


def test_group_absent_returns_ones_at_every_timestep():
    resolved = resolve_spatial_multipliers_per_timestep({}, n_timesteps=3)
    for t in range(3):
        mult = get_multiplier(resolved, "Colonization", (4, 4), t=t)
        assert np.array_equal(mult, ONES)


def test_multiple_timesteps_forward_fill_and_switch():
    """Two entries (timestep=0 and timestep=3): before t=3 uses the first
    raster, from t=3 onward switches to the second — never early."""
    loaded = {
        "Colonization": [(0, _const(0.2)), (3, _const(0.9))],
    }
    resolved = resolve_spatial_multipliers_per_timestep(loaded, n_timesteps=6)

    for t in [0, 1, 2]:
        mult = get_multiplier(resolved, "Colonization", (4, 4), t=t)
        assert np.allclose(mult, 0.2), f"t={t} should still use the t=0 raster"

    for t in [3, 4, 5]:
        mult = get_multiplier(resolved, "Colonization", (4, 4), t=t)
        assert np.allclose(mult, 0.9), f"t={t} should have switched to the t=3 raster"


def test_timestep_before_earliest_entry_is_ones_not_future_raster():
    """No look-ahead: a group whose only raster starts at timestep=5 must
    be ones (no effect) at t=0..4, never peek forward at the t=5 raster."""
    loaded = {"Inundation": [(5, _const(0.7))]}
    resolved = resolve_spatial_multipliers_per_timestep(loaded, n_timesteps=8)

    for t in range(5):
        mult = get_multiplier(resolved, "Inundation", (4, 4), t=t)
        assert np.array_equal(mult, ONES), \
            f"t={t} is before the earliest entry (t=5); must be ones, not the future raster"

    for t in [5, 6, 7]:
        mult = get_multiplier(resolved, "Inundation", (4, 4), t=t)
        assert np.allclose(mult, 0.7)


def test_none_timestep_sorts_before_explicit_timesteps_in_resolution():
    """A None-timestep entry mixed with explicit ones still applies from
    t=0 (sorts first), then explicit timesteps take over in order."""
    loaded = {
        "Sedimentation": [(2, _const(0.6)), (None, _const(0.3))],  # unsorted input
    }
    resolved = resolve_spatial_multipliers_per_timestep(loaded, n_timesteps=4)

    assert np.allclose(get_multiplier(resolved, "Sedimentation", (4, 4), t=0), 0.3)
    assert np.allclose(get_multiplier(resolved, "Sedimentation", (4, 4), t=1), 0.3)
    assert np.allclose(get_multiplier(resolved, "Sedimentation", (4, 4), t=2), 0.6)
    assert np.allclose(get_multiplier(resolved, "Sedimentation", (4, 4), t=3), 0.6)


# ── load_spatial_multipliers() — loading/sorting/missing-file bookkeeping ──

def test_load_returns_sorted_series_none_first(tmp_path):
    """Entries loaded out of order in the CSV are sorted None-first, then
    ascending — matching resolve_targets_per_timestep()'s convention."""
    _write_tif(tmp_path / "a.tif", 0.9)
    _write_tif(tmp_path / "b.tif", 0.9)
    _write_tif(tmp_path / "c.tif", 0.9)
    entries = [
        SpatialMultEntry(group="Agriculture_expansion", filename="c.tif", timestep=10),
        SpatialMultEntry(group="Agriculture_expansion", filename="a.tif", timestep=None),
        SpatialMultEntry(group="Agriculture_expansion", filename="b.tif", timestep=2),
    ]

    loaded = load_spatial_multipliers(entries, tmp_path, (4, 4))
    timesteps = [t for t, _ in loaded["Agriculture_expansion"]]
    assert timesteps == [None, 2, 10]


def test_missing_file_for_one_timestep_falls_back_to_ones_for_that_slot(tmp_path):
    """A missing raster for one Timestep row degrades to ones for that
    slot only, not for the whole group's series."""
    _write_tif(tmp_path / "present.tif", 0.9)
    entries = [
        SpatialMultEntry(group="Urbanization", filename="present.tif", timestep=0),
        SpatialMultEntry(group="Urbanization", filename="missing.tif", timestep=4),
    ]

    loaded = load_spatial_multipliers(entries, tmp_path, (4, 4))
    series = dict(loaded["Urbanization"])   # {timestep: array}

    assert not np.array_equal(series[0], ONES), "present.tif should have actually loaded"
    assert np.array_equal(series[4], ONES), "missing.tif should degrade to ones for its slot"

    resolved = resolve_spatial_multipliers_per_timestep(loaded, n_timesteps=6)
    mult_late = get_multiplier(resolved, "Urbanization", (4, 4), t=5)
    assert np.array_equal(mult_late, ONES)
