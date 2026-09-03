# Glossary

Terms used across the `strategicc` docs, split into SEEA EA accounting concepts (general, from the UN SEEA Ecosystem Accounting standard) and STRATEGICC-specific terms (package-specific, from the STSM/`strategicc` source).

## SEEA EA accounting terms

**Ecosystem extent**, The area of each ecosystem type (state class) at a point in time, and how it changes. Reported in the extent account (SEEA EA Table 4.1).

**Ecosystem condition**, The overall quality of an ecosystem relative to a reference state. `strategicc` doesn't currently compute a condition account.

**Ecosystem service**, A contribution an ecosystem makes to human benefit (e.g. carbon sequestration, coastal protection, fishery provisioning). Defined per state class in `EcosystemServices.csv`.

**Physical flow account**, Ecosystem services measured in physical units (Mg carbon, kg fish, m³ water) rather than money. Requires Mode B or C services, see `physical_flow_account()` / `physical_flow_account_seea()` in [reference/accounting.md](reference/accounting.md).

**Monetary flow account**, Ecosystem services valued in currency. SEEA EA Tables 9.1a (supply, by year and class) and 9.1b (use, by year and beneficiary).

**Monetary asset account**, Values the ecosystem itself as a stock of wealth, not just its annual service flow, SEEA EA Table 10.1. Opening value, changes over the period (enhancement, degradation, conversion between ecosystem types, other volume changes, revaluation), and closing value. Built by `monetary_asset_account_seea()`.

**Supply and use table**, SEEA EA's standard split for any flow account: *supply* (which ecosystem type produced the service, by year) and *use* (which beneficiary group consumed it, by year). `supply.sum() == use.sum()` per year and service by construction.

**Transition matrix / change matrix**, Area (or value) moved from each ecosystem type to each other ecosystem type over the accounting period. The off-diagonal of an extent account.

**Enhancement / degradation**, In an asset account, the portion of value change attributed to condition improving or worsening within the same ecosystem type (as opposed to conversion to a different type). In `strategicc`, this is a residual computed to make Net change reconcile exactly, an approximation, not SEEA EA's full condition-attributed split.

**Revaluation**, A change in asset value caused purely by a change in price/valuation parameters, not by any physical change to the ecosystem.

**Reappraisal**, A change in reported value caused by a change in valuation *methodology* itself (e.g. switching to a better model). `strategicc` always reports this as `0.0`, it has no mechanism to distinguish a methodology change from ordinary revaluation.

## STRATEGICC-specific terms

**STSM (State-and-Transition Simulation Model)**, The underlying simulation framework (Daniel et al. 2016): a landscape is represented as cells, each in a discrete *state* (state class), which can *transition* to other states each timestep according to configured probabilities.

**State class**, A discrete land-cover / ecosystem category (e.g. Mangrove, Aquaculture). Defined in `StateClasses.csv`, referenced everywhere by integer class ID.

**Transition**, A probabilistic rule that a cell in one state class may become another over one timestep. Defined in `Transitions.csv`, organized into named **transition groups** (e.g. `Aquaculture_expansion`).

**Iteration**, One full stochastic run of the simulation from start to end year. `N_ITERATIONS` independent iterations are run to build a distribution of possible outcomes, not just a single point estimate.

**Timestep**, One simulated year (or other period) within a single iteration.

**Stratum**, An optional spatial zoning layer (primary/secondary/tertiary) that can restrict which transitions apply where. As of the current version, parsed into config but not yet consumed anywhere in the engine, reserved for future use.

**Adjacency**, A rule set that biases transition probability based on what's already nearby (e.g. cells next to existing aquaculture are more likely to convert to aquaculture too). Controlled by `USE_ADJACENCY` and `ADJACENCY_STRENGTH`.

**Spatial multiplier**, A 0-1 normalized raster that biases *where* a transition group is more or less likely to fire, independent of adjacency (e.g. distance to a road or market). Configured via `TransitionSpatialMultipliers.csv` and `MULT_DIR`.

**Transition multiplier**, A year-to-year scaling factor sampled from a distribution (`TransitionMultipliers.csv` + `Distributions.csv`) that adds historical variability on top of a transition's base probability, so simulated runs reproduce the same average rate as the static input while preserving realistic year-to-year swings.

**Age**, How long (in years) a cell has continuously been in its current state class. Enables age-indexed values (e.g. young vs. mature mangrove sequestering carbon at different rates) via `StateAttributeValues.csv`. Optional (`USE_AGE`).

**Modal map**, For a given year, the state class that the majority of iterations agree a given cell is in. `outputs.aggregate_spatial()` builds this across all iterations; it's the spatially consistent map that all downstream area/SEEA-EA tables are computed from (not the raw per-iteration data).

**RunManifest.txt**, A single text file that can set every `strategicc.config` value at once (see [manifest_reference.md](manifest_reference.md)), as an alternative to setting `cfg.X = ...` attributes directly in code. Only lines outside fenced ` ``` ` code blocks are parsed as live configuration.

**Stock & Flow**, An optional layer (`USE_STOCKFLOW`) tracking a continuous quantity (e.g. carbon) moving between named pools (**stock types**, e.g. Biomass, Atmosphere) via named transfers (**flow types**, e.g. NPP, Emission), governed by **flow pathways** that can be automatic (fire every timestep) or triggered by a specific transition group firing.

**Valuation Mode A / B / C**, Which of `EcosystemServices.csv`'s optional columns are filled for a given service row, determining how its value is computed: **A**, static per-hectare value only; **B**, adds a static physical quantity per hectare; **C**, physical quantity comes from the actual simulated Stock & Flow output (`StockFlowSource = "flow:<Type>"` or `"stock:<Type>"`) rather than a flat rate.

**UserType / UserShare**, Optional `EcosystemServices.csv` columns attributing a service's use to a named beneficiary group, feeding the *use* side of the SEEA EA supply/use tables. Services without a `UserType` default to a single `"Unspecified"` beneficiary at 100% share.

**Diagnostic**, `engine.diagnostic()`: prints expected transitions per class/group with no multipliers applied, before running the full simulation, as a sanity check on the configured probabilities.

**Calibration**, Deriving `Transitions.csv`, `TransitionMultipliers.csv`, and/or `age.tif` automatically from a historical time series of classified rasters (`strategicc.calibration`), rather than specifying transition rates by hand.

**Hindcast correction**, Comparing a calibrated simulation's output against the historical record it was calibrated from, then adjusting (`correct_multipliers()`) the transition multipliers so simulated rates better match observed rates (`strategicc.validation`).
