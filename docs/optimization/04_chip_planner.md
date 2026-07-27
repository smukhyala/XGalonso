# Chip Planner

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Chip Planner |
| Version | 1.0 |
| Status | Deferred (Post-MVP) |
| Owner | Optimization |
| Dependencies | [Wildcard Planner](03_wildcard_planner.md), [Transfer Planner](02_transfer_planner.md), [Prediction Models](../ml/07_prediction_models.md) |
| Last updated | 2026-07-27 |

---

## 1. Deferral notice

**No chip logic ships in the MVP.** Binding decision D5 is explicit: chip **state** is modelled, chip
**logic** is not built. There is no chip endpoint in the [Public API](../api/01_public_api.md), no
chip term in the [Transfer Planner](02_transfer_planner.md) objective, and no chip recommendation in
`xg recommend`.

What does exist in the MVP is state. `packages/domain` represents, from day one:

- which chips an entry has already played, and in which gameweek;
- which chips remain available in the current window;
- whether a chip is active in the gameweek being planned.

This is read from `chips` in `entry/{entry_id}/history/` and `active_chip` in
`entry/{entry_id}/event/{gw}/picks/`, and it is stored so that chip logic is never blocked by a
missing field or a schema migration later. Modelling state is a few columns. Modelling logic is a
subsystem. Only the first is in scope.

**The set of chips is read from the API payload, not hardcoded.** FPL has added and retired chips
between seasons, so a planner that enumerates a fixed chip list will be wrong the season the list
changes. The named chips below are the design target, not an assertion about any particular season's
roster.

---

## 2. Objective

Jointly optimize the season's chips and produce a chip roadmap with expected value.

| Chip | Effect | Value driver |
|---|---|---|
| Wildcard | Unlimited transfers for one gameweek, squad rebuilt from scratch | Fixture swings, injury clusters, accumulated squad decay. See [Wildcard Planner](03_wildcard_planner.md) |
| Free Hit | Unlimited transfers for one gameweek; the squad reverts afterwards | Blank and double gameweeks, where a temporary squad is worth far more than a permanent one |
| Bench Boost | Bench points count for one gameweek | Requires a genuinely strong bench and a gameweek where all 15 play |
| Triple Captain | Captain scores triple rather than double | Concentrated in a single high-ceiling fixture, ideally a double gameweek |

### 2.1 Why joint rather than sequential

Chips compete for the same scarce gameweeks. The double gameweeks that make Bench Boost valuable are
the same ones that make Triple Captain and Free Hit valuable, and playing a wildcard beforehand is
often what makes a Bench Boost possible at all. Optimizing each chip independently double-counts the
same opportunity and produces a roadmap that cannot actually be executed.

The correct formulation is a scheduling problem: assign at most one chip per gameweek, respecting
window constraints and one-use-per-chip constraints, maximizing total expected points over the
season subject to the joint feasibility of the underlying squad path.

---

## 3. Chip windows

| Window | Gameweeks | Note |
|---|---|---|
| First half | GW2–GW19 | One wildcard available |
| Second half | GW20–GW38 | A second wildcard available |

**Wildcards are unavailable in GW1** — the first window opens at GW2. An unused first-half wildcard
does not carry into the second half.

**Window boundaries, chip identifiers and all scoring constants load from a pinned snapshot of the
FPL payload with a recorded fetch timestamp and a drift check. They are never Python literals.**

---

## 4. Output: the season chip roadmap

| Field | Description |
|---|---|
| Chip | Which chip |
| Target gameweek | Recommended gameweek to play it |
| Expected value | Expected points gain over not playing it, on the HOLD convention |
| Confidence | Calibrated probability the plan beats the no-chip path |
| Alternatives | Next-best gameweeks with their expected values, so a user can see how tight the call is |
| Explanation | Deterministic reason codes: fixture density, blank or double gameweek, bench strength, captaincy ceiling |
| Revision trigger | Conditions that should force the roadmap to be recomputed, such as a fixture reschedule or a key injury |

A roadmap is a plan under uncertainty over a 38-gameweek horizon, so it is a **living** artefact.
Expected values 20 gameweeks out are weak, and the roadmap must present them as weak rather than as
commitments. Confidence decays with horizon distance, and the revision trigger exists because
blank and double gameweeks are frequently not known until well into the season.

---

## 5. Prerequisites before this is built

| Prerequisite | Why |
|---|---|
| Wildcard planner shipped | Wildcard is the highest-value chip and the one others depend on |
| Blank and double gameweek detection | Free Hit, Bench Boost and Triple Captain value is dominated by these |
| Reliable long-horizon projections | A roadmap is a 38-gameweek object |
| Season-path simulation | Chips interact through the squad path, not just through individual gameweeks |
| Chip state in `packages/domain` | Already in scope for the MVP |

---

## Related documents

- [Wildcard Planner](03_wildcard_planner.md) — the first chip to be built, also deferred
- [Transfer Planner](02_transfer_planner.md) — the shipping optimizer, chip-free by design
- [Prediction Models](../ml/07_prediction_models.md) — long-horizon projections this needs
- [Public API](../api/01_public_api.md) — no chip routes in the MVP surface
- [Documentation Index](../README.md)
