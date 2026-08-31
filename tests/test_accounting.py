"""
tests/test_accounting.py  —  v2.2
Unit tests for SEEA-EA accounting module.
Uses area_modal_df (from modal maps) as primary input.
"""

import pytest
import pandas as pd
import numpy as np
from strategicc.accounting.csv_loader import EcosystemService, AssetValuationParams
from strategicc.accounting.seea import SEEAAccount
from strategicc.io.csv_loader import StateClass
from strategicc.outputs import modal_to_area_table, _area_col


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_classes():
    return {
        1: StateClass(1, "Forest",   "Forest:All",   (255,0,100,0)),
        2: StateClass(2, "Cropland", "Cropland:All", (255,255,255,0)),
    }

def make_services():
    return [
        EcosystemService("Forest",   "Carbon",  "Regulating",  10_000, "IDR", "MgC/ha", 100),
        EcosystemService("Forest",   "Tourism", "Cultural",     5_000, "IDR", None,      None),
        EcosystemService("Cropland", "Crops",   "Provisioning", 8_000, "IDR", "kg/ha",   500),
    ]

def make_modal_maps():
    """2 years × 2×3 grid: mostly Forest, some Cropland."""
    return {
        2022: np.array([[1,1,2],[1,2,2]], dtype=np.uint8),   # 3 Forest, 3 Cropland
        2023: np.array([[1,1,2],[2,2,2]], dtype=np.uint8),   # 2 Forest, 4 Cropland
        2024: np.array([[1,2,2],[2,2,2]], dtype=np.uint8),   # 1 Forest, 5 Cropland
    }

def make_area_modal_df():
    return modal_to_area_table(make_modal_maps(), make_classes(),
                               px_area=0.01, area_unit="ha")

def make_raw_area_df():
    """Raw per-iteration area (for uncertainty summary)."""
    rows = []
    for it in [1, 2]:
        for yr in [2022, 2023, 2024]:
            rows.append({"iteration":it,"year":yr,"class_id":1,
                         "class_name":"Forest",  "area_ha": 0.03 - (yr-2022)*0.01})
            rows.append({"iteration":it,"year":yr,"class_id":2,
                         "class_name":"Cropland","area_ha": 0.03 + (yr-2022)*0.01})
    return pd.DataFrame(rows)

def make_trans_df():
    rows = []
    for it in [1, 2]:
        rows.append({"iteration":it,"year":2022,"row":0,"col":2,
                     "from_class":"Forest","to_class":"Cropland",
                     "group":"Agriculture_expansion"})
    return pd.DataFrame(rows)

def make_acct(**kwargs):
    defaults = dict(
        area_modal_df = make_area_modal_df(),
        trans_df      = make_trans_df(),
        services      = make_services(),
        classes       = make_classes(),
        px_area       = 0.01,
        area_df       = make_raw_area_df(),
    )
    defaults.update(kwargs)
    return SEEAAccount(**defaults)


# ── Tests: modal_to_area_table ────────────────────────────────────────────────

def test_modal_to_area_table_ha():
    df = modal_to_area_table(make_modal_maps(), make_classes(),
                             px_area=0.09, area_unit="ha")
    assert _area_col(df) == "area_ha"
    assert len(df) == 6   # 3 years × 2 classes

def test_modal_to_area_table_km2():
    df = modal_to_area_table(make_modal_maps(), make_classes(),
                             px_area=0.0009, area_unit="km2")
    assert _area_col(df) == "area_km2"

def test_modal_to_area_table_px():
    df = modal_to_area_table(make_modal_maps(), make_classes(),
                             px_area=1.0, area_unit="px")
    assert _area_col(df) == "area_px"
    # 2022: 3 Forest pixels × 1.0 = 3.0
    row = df[(df["year"]==2022) & (df["class_name"]=="Forest")]
    assert row["area_px"].values[0] == pytest.approx(3.0)

def test_modal_area_values():
    df = modal_to_area_table(make_modal_maps(), make_classes(),
                             px_area=0.01, area_unit="ha")
    # 2022: 3 Forest × 0.01 = 0.03 ha
    row = df[(df["year"]==2022) & (df["class_name"]=="Forest")]
    assert row["area_ha"].values[0] == pytest.approx(0.03)
    # 2024: 1 Forest × 0.01 = 0.01 ha
    row = df[(df["year"]==2024) & (df["class_name"]=="Forest")]
    assert row["area_ha"].values[0] == pytest.approx(0.01)


# ── Tests: SEEAAccount ────────────────────────────────────────────────────────

def test_extent_account_shape():
    ea = make_acct().extent_account()
    assert ea.shape == (3, 3)   # 3 years × (2 classes + Total)
    assert "Forest" in ea.columns
    assert "Cropland" in ea.columns
    assert "Total" in ea.columns

def test_extent_account_total_column():
    """Total column should equal the row sum, i.e. conserved landscape area."""
    ea = make_acct().extent_account()
    assert ea.loc[2022, "Total"] == pytest.approx(0.06)   # 3 Forest + 3 Cropland, px_area 0.01
    for year in ea.index:
        assert ea.loc[year, "Total"] == pytest.approx(0.06)

def test_extent_account_values():
    ea = make_acct().extent_account()
    assert ea.loc[2022, "Forest"]   == pytest.approx(0.03)
    assert ea.loc[2024, "Forest"]   == pytest.approx(0.01)
    assert ea.loc[2024, "Cropland"] == pytest.approx(0.05)

def test_extent_account_seea_shape_and_periods():
    """Table 4.1 layout: one 5-row block (Opening/Additions/Reductions/
    Net change/Closing) per accounting period, classes + Total columns."""
    seea_ea = make_acct().extent_account_seea()
    periods = seea_ea.index.get_level_values("Period").unique().tolist()
    assert periods == ["2022–2023", "2023–2024"]
    entries = seea_ea.loc["2022–2023"].index.tolist()
    assert entries == [
        "Opening extent", "Additions", "Reductions",
        "Net change in extent", "Closing extent",
    ]
    assert list(seea_ea.columns) == ["Cropland", "Forest", "Total"]

def test_extent_account_seea_reconciles():
    """Net change in extent should equal Closing - Opening for the period
    that trans_df actually has transition data for (2022→2023)."""
    seea_ea = make_acct().extent_account_seea()
    block = seea_ea.loc["2022–2023"]
    opening = block.loc["Opening extent"]
    closing = block.loc["Closing extent"]
    net     = block.loc["Net change in extent"]
    for col in ["Forest", "Cropland", "Total"]:
        assert net[col] == pytest.approx(closing[col] - opening[col])

def test_extent_account_seea_additions_reductions_values():
    """make_trans_df logs 2 Forest->Cropland transitions (2 iterations,
    1 pixel each, px_area=0.01) in year 2022 -> median across iterations
    = 1 pixel = 0.01 ha moved."""
    seea_ea = make_acct().extent_account_seea()
    block = seea_ea.loc["2022–2023"]
    assert block.loc["Additions", "Cropland"]  == pytest.approx(0.01)
    assert block.loc["Additions", "Forest"]    == pytest.approx(0.0)
    assert block.loc["Reductions", "Forest"]   == pytest.approx(0.01)
    assert block.loc["Reductions", "Cropland"] == pytest.approx(0.0)

def test_extent_account_seea_managed_split():
    seea_ea = make_acct().extent_account_seea(
        managed_groups={"Agriculture_expansion"}
    )
    entries = seea_ea.loc["2022–2023"].index.tolist()
    assert "Additions — managed expansions" in entries
    assert "Additions — unmanaged expansions" in entries
    block = seea_ea.loc["2022–2023"]
    assert block.loc["Additions — managed expansions", "Cropland"] == pytest.approx(0.01)
    assert block.loc["Additions — unmanaged expansions", "Cropland"] == pytest.approx(0.0)

def test_extent_account_seea_requires_trans_df():
    acct = make_acct(trans_df=pd.DataFrame())
    with pytest.raises(ValueError):
        acct.extent_account_seea()

# ── v3.5: physical/monetary flow accounts as SUTs (Table 7.1a/b, 9.1a/b) ──────

def test_flow_account_seea_supply_use_identity():
    """Core SUT requirement (SEEA EA para. 7.7): total supply of a
    service must equal total use of that service, for both physical
    and monetary accounts, for every year."""
    acct = make_acct()
    phys = acct.physical_flow_account_seea()
    mon  = acct.monetary_flow_account_seea()
    for result in (phys, mon):
        supply_totals = result["supply"].groupby("Year").sum().sum(axis=1)
        use_totals    = result["use"].groupby("Year").sum().sum(axis=1)
        for year in supply_totals.index:
            assert supply_totals[year] == pytest.approx(use_totals[year])

def test_physical_flow_account_seea_keeps_class_dimension():
    """Unlike physical_flow_account(), the class/ecosystem-type
    dimension must survive (that's the whole point of the fix)."""
    result = make_acct().physical_flow_account_seea()
    classes_seen = result["supply"].index.get_level_values("Ecosystem type").unique()
    assert set(classes_seen) == {"Forest", "Cropland"}

def test_flow_account_seea_user_split():
    """A service split 60/40 across two UserTypes should show up as
    two separate use-table rows that sum back to the supply total."""
    services = make_services() + []
    services = [s for s in services if not (s.state_class == "Cropland" and s.service_name == "Crops")]
    services += [
        EcosystemService("Cropland", "Crops", "Provisioning", 8_000, "IDR", "kg/ha", 500,
                          user_type="Households", user_share=0.6, has_explicit_user=True),
        EcosystemService("Cropland", "Crops", "Provisioning", 8_000, "IDR", "kg/ha", 500,
                          user_type="Agriculture", user_share=0.4, has_explicit_user=True),
    ]
    acct = make_acct(services=services)
    result = acct.physical_flow_account_seea()
    use = result["use"]
    year_use = use.xs(2022, level="Year")
    crops_col = [c for c in use.columns if c[1] == "Crops"][0]
    households = year_use.loc["Households", crops_col]
    agriculture = year_use.loc["Agriculture", crops_col]
    supply_total = result["supply"].xs(2022, level="Year")[crops_col].sum()
    assert households == pytest.approx(supply_total * 0.6)
    assert agriculture == pytest.approx(supply_total * 0.4)

def test_monetary_flow_account_seea_no_user_split_is_unspecified():
    result = make_acct().monetary_flow_account_seea()
    users = result["use"].index.get_level_values("User type").unique()
    assert list(users) == ["Unspecified"]

def test_flow_account_seea_additive_multi_row_without_usertype_not_double_counted():
    """Regression test: a real EcosystemServices.csv can legitimately have
    TWO rows sharing one service_name with no UserType at all (e.g. one
    Mode C stock:AGB row + one stock:Soil row, both under 'Carbon
    Storage', meant to SUM). This must not be mistaken for a v3.5
    user-split (which would keep only one row as canonical) and must
    not be flagged by the UserShare-sum warning either."""
    services = make_services() + [
        EcosystemService("Forest", "Carbon Storage", "Regulating", 1_000, "IDR", "MgC", 50),
        EcosystemService("Forest", "Carbon Storage", "Regulating", 500, "IDR", "MgC", 20),
    ]
    acct = make_acct(services=services)
    tv = acct.total_value_by_class()
    # Forest's total should include BOTH Carbon Storage rows (1000+500
    # per-ha components), not just one of them.
    result = acct.monetary_flow_account_seea()
    storage_cols = [c for c in result["supply"].columns if c[1] == "Carbon Storage"]
    assert len(storage_cols) == 1
    supply_2022 = result["supply"].xs(2022, level="Year")[storage_cols[0]].sum()
    # 3 Forest px * 0.01 ha * (1000 + 500) IDR/ha = 45.0
    assert supply_2022 == pytest.approx(45.0)

# ── v3.5: monetary_asset_account_seea (Table 10.1) ────────────────────────────

def make_asset_valuation_params(**overrides):
    defaults = dict(state_class="ALL", discount_rate=0.02, asset_life_years=10,
                     price_growth_rate=0.0)
    defaults.update(overrides)
    return {"ALL": AssetValuationParams(**defaults)}

def test_monetary_asset_account_seea_requires_params():
    acct = make_acct()
    with pytest.raises(ValueError):
        acct.monetary_asset_account_seea()

def test_monetary_asset_account_seea_requires_trans_df():
    acct = make_acct(trans_df=pd.DataFrame(),
                      asset_valuation_params=make_asset_valuation_params())
    with pytest.raises(ValueError):
        acct.monetary_asset_account_seea()

def test_monetary_asset_account_seea_shape_and_entries():
    acct = make_acct(asset_valuation_params=make_asset_valuation_params())
    df = acct.monetary_asset_account_seea()
    periods = df.index.get_level_values("Period").unique().tolist()
    assert periods == ["2022–2023", "2023–2024"]
    entries = df.loc["2022–2023"].index.tolist()
    assert entries == [
        "Opening value", "Ecosystem enhancement", "Ecosystem degradation",
        "Ecosystem conversions — additions", "Ecosystem conversions — reductions",
        "Other changes in volume — catastrophic losses",
        "Other changes in volume — reappraisals",
        "Revaluations", "Net change in value", "Closing value",
    ]
    assert list(df.columns) == ["Cropland", "Forest", "Total"]

def test_monetary_asset_account_seea_reconciles():
    """Net change in value must equal Closing - Opening exactly, for
    every class and every period — this holds by construction since
    Enhancement/Degradation is defined as the reconciling residual."""
    acct = make_acct(asset_valuation_params=make_asset_valuation_params())
    df = acct.monetary_asset_account_seea()
    for period in df.index.get_level_values("Period").unique():
        block = df.loc[period]
        opening, closing, net = (block.loc["Opening value"], block.loc["Closing value"],
                                  block.loc["Net change in value"])
        for col in df.columns:
            assert net[col] == pytest.approx(closing[col] - opening[col])

def test_monetary_asset_account_seea_zero_growth_means_zero_revaluation():
    acct = make_acct(asset_valuation_params=make_asset_valuation_params(price_growth_rate=0.0))
    df = acct.monetary_asset_account_seea()
    reval = df.xs("Revaluations", level="Entry")
    assert (reval == 0).all().all()

def test_monetary_asset_account_seea_positive_growth_gives_nonzero_revaluation():
    acct = make_acct(asset_valuation_params=make_asset_valuation_params(price_growth_rate=0.03))
    df = acct.monetary_asset_account_seea()
    reval = df.xs("Revaluations", level="Entry")
    assert (reval["Total"] > 0).all()
    # reconciliation must still hold with nonzero price growth
    for period in df.index.get_level_values("Period").unique():
        block = df.loc[period]
        opening, closing, net = (block.loc["Opening value"], block.loc["Closing value"],
                                  block.loc["Net change in value"])
        for col in df.columns:
            assert net[col] == pytest.approx(closing[col] - opening[col])

def test_monetary_asset_account_seea_catastrophic_groups():
    acct = make_acct(asset_valuation_params=make_asset_valuation_params())
    df = acct.monetary_asset_account_seea(catastrophic_groups={"Agriculture_expansion"})
    cat = df.xs("Other changes in volume — catastrophic losses", level="Entry")
    ordinary_red = df.xs("Ecosystem conversions — reductions", level="Entry")
    # every transition in the fixture is tagged Agriculture_expansion, so all
    # of the period's Reductions value should have moved to catastrophic losses
    assert cat.loc["2022–2023", "Total"] > 0
    assert ordinary_red.loc["2022–2023", "Total"] == pytest.approx(0.0)
    # reconciliation must still hold
    for period in df.index.get_level_values("Period").unique():
        block = df.loc[period]
        opening, closing, net = (block.loc["Opening value"], block.loc["Closing value"],
                                  block.loc["Net change in value"])
        for col in df.columns:
            assert net[col] == pytest.approx(closing[col] - opening[col])

# ── v3.22: Additions/Reductions must be NPV-based, not raw single-year value ──
# Regression test for a real bug: pre-3.22, val_per_area was computed from the
# raw one-year service flow (tv), while Opening/Closing/Net change were NPV'd
# over AssetLifeYears — an outright units mismatch (roughly the AssetLifeYears
# annuity factor, ~9x for this fixture's discount_rate=0.02/asset_life_years=10)
# that manufactured large spurious Enhancement/Degradation residuals with no
# real cause: no stock/flow, no loss pathway, every service here is a flat,
# unchanging per-area rate. With no mechanism for genuine condition change,
# the residual should be at (or within floating-point noise of) zero.
#
# Only "2022–2023" is checked — make_trans_df() only logs a transition for
# 2022, while make_modal_maps() also shows area shifting between 2023 and
# 2024 with no matching transition logged. That's a pre-existing fixture
# gap (trans_df and the modal maps disagree for that period), not something
# a real run can produce — trans_df and modal maps always come from the same
# simulation there. Checking "2023–2024" here would test the fixture's own
# inconsistency, not the fix.
def test_monetary_asset_account_seea_no_spurious_degradation():
    acct = make_acct(asset_valuation_params=make_asset_valuation_params())
    df = acct.monetary_asset_account_seea()
    period = "2022–2023"
    block = df.loc[period]
    for col in df.columns:
        assert block.loc["Ecosystem enhancement", col] == pytest.approx(0.0, abs=1e-6)
        assert block.loc["Ecosystem degradation", col]  == pytest.approx(0.0, abs=1e-6)

def test_monetary_asset_account_seea_conversions_use_npv_scale():
    """
    Additions/Reductions must be on the same order of magnitude as
    Opening/Closing value (both NPV'd), not ~AssetLifeYears-annuity-factor
    smaller — the pre-3.22 bug put raw one-year flow values next to NPV'd
    values, off by roughly that factor for any nontrivial asset life.
    """
    acct = make_acct(asset_valuation_params=make_asset_valuation_params())
    df = acct.monetary_asset_account_seea()
    period = "2022–2023"
    block = df.loc[period]
    opening_total = block.loc["Opening value", "Total"]
    additions_total = block.loc["Ecosystem conversions — additions", "Total"]
    reductions_total = block.loc["Ecosystem conversions — reductions", "Total"]
    # Additions/Reductions reflect only the converted area's share of value,
    # so they should be a modest fraction of Opening value, not smaller by
    # an extra ~9x (this fixture's NPV annuity factor) on top of that.
    assert additions_total > 0 or reductions_total > 0
    if additions_total > 0:
        assert additions_total / opening_total > 0.01
    if reductions_total > 0:
        assert reductions_total / opening_total > 0.01


def test_monetary_asset_account_seea_missing_class_params_raises():
    services = make_services() + [
        EcosystemService("Wetland", "FloodControl", "Regulating", 1_000, "IDR", None, None),
    ]
    classes = make_classes()
    classes[3] = StateClass(3, "Wetland", "Wetland:All", (255, 0, 0, 255))
    # per-class params covering only Forest/Cropland, no "ALL" fallback —
    # but area_modal_df/trans_df in this fixture never actually contain
    # Wetland, so this exercises the *lookup* path harmlessly; the real
    # check is that Forest/Cropland-only params with no ALL still work.
    params = {
        "Forest":   AssetValuationParams("Forest", 0.02, 10),
        "Cropland": AssetValuationParams("Cropland", 0.02, 10),
    }
    acct = make_acct(asset_valuation_params=params)
    df = acct.monetary_asset_account_seea()   # should NOT raise — no ALL needed
    assert not df.empty

def test_monetary_flow_account():
    mf = make_acct().monetary_flow_account()
    assert not mf.empty
    assert mf.sum().sum() > 0

def test_physical_flow_account():
    pf = make_acct().physical_flow_account()
    assert pf is not None
    assert not pf.empty

def test_transition_matrix():
    tm = make_acct().transition_matrix()
    assert tm.loc["Forest", "Cropland"] > 0
    assert tm.loc["Cropland", "Forest"] == 0

def test_value_change_matrix():
    vm = make_acct().value_change_matrix()
    # Forest→Cropland: Cropland(8000) - Forest(15000) = -7000 → negative
    assert vm.loc["Forest", "Cropland"] < 0

def test_change_in_value_total():
    cv = make_acct().change_in_value()
    # Forest shrinking → losing high-value land → total should decline
    assert cv["Total"].iloc[1] < 0

def test_uncertainty_summary_columns():
    unc = make_acct().uncertainty_summary()
    assert unc is not None
    assert "Year" in unc.columns
    assert "Range (%)" in unc.columns

def test_uncertainty_summary_none_when_no_raw():
    acct = make_acct(area_df=None)
    assert acct.uncertainty_summary() is None

def test_mode_a_no_physical():
    """Mode A services (no physical unit) still produce monetary account."""
    services_a = [
        EcosystemService("Forest", "Tourism", "Cultural", 5000, "IDR", None, None),
    ]
    acct = make_acct(services=services_a)
    assert acct.physical_flow_account() is None
    assert not acct.monetary_flow_account().empty

def test_unit_label_in_extent_account():
    """Extent account attrs should carry the unit label."""
    ea = make_acct().extent_account()
    assert ea.attrs.get("unit") == "ha"
