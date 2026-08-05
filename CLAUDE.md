# XG Alonso — Engineering Guide

You are contributing to XG Alonso.

This repository prioritizes engineering quality over implementation speed.

Never write code that only satisfies today's requirements.

Every component should be designed assuming the platform will continue growing.

---

## Binding Decisions

These were settled with the project owner. Where a document or an existing implementation
contradicts one of them, the decision wins. Do not relitigate them while implementing.

| # | Decision |
|---|---|
| D1 | Local-only first — no Docker, cloud, or hosting in the first slice |
| D2 | DuckDB + Parquet only; no PostgreSQL. Storage sits behind a repository interface |
| D3 | Public FPL team ID only; no authentication. Prices reconstructed from public transfer history |
| D4 | CLI first, then FastAPI, then Next.js. No frontend in the MVP [†](#d4-superseded-2026-08-04) |
| D5 | No chips in the MVP. Chip *state* is modelled; chip *logic* is not built |
| D6 | Official FPL API only, zero budget. No scraping, no paid providers |
| D7 | Historical backfill from 2022/23 — the earliest season with xG in the API |
| D8 | Component-based points modelling, converted through versioned scoring rules |
| D9 | Useful by GW1 of 2026/27, refined in-season |
| D10 | Product first; research platform deepened afterwards |
| D11 | Price model deferred — no current-season price data exists at GW1 |
| D12 | 300-700 quality candidate features, **not** thousands |

### Amendments

The decision text above is a log and is never rewritten. Where reality has moved, it is recorded
here with a date, so the original decision and the departure from it are both legible.

<a id="d4-superseded-2026-08-04"></a>
**† D4 superseded 2026-08-04.** The Next.js surface shipped ahead of the original sequencing. The
CLI-first *ordering* was still followed — CLI, then FastAPI, then Next.js — so what lapsed is the
"no frontend in the MVP" clause, not the surface order. `apps/web` serves four routes over
`apps/api` and holds no modelling logic, which is the constraint D4 existed to protect. The web
front end is therefore **in scope and shipped**; do not treat a request touching it as
relitigating D4.

D12's range is a **ceiling, not a target.** The build is currently under it — 180 catalogue specs,
231 distinct feature columns — and that is compliance, not a shortfall to be closed by generating
filler.

---

## Verified Constants

**FPL scoring values and squad constraints are never written as Python literals.** They load from a
pinned snapshot of the `bootstrap-static` payload, with a recorded fetch timestamp and a drift check
that runs on every ingest. `game_config.scoring` and `game_config.rules` exist in that payload, so
these constants are machine-readable and there is no reason to transcribe them.

The worked example of why: **a goalkeeper goal is worth 10 points, not the widely assumed 6.**
Nearly every community model has that wrong because someone typed it from memory. Transcription is
how that class of error enters a codebase silently and survives review.

Constants that must come from the pinned snapshot rather than from a developer's memory:

- `goals_scored` per position — `{GKP: 10, DEF: 6, MID: 5, FWD: 4}`
- `assists` — `3`
- `clean_sheets` per position — `{GKP: 4, DEF: 4, MID: 1, FWD: 0}`
- `saves` — `1`, `bonus` — `1`
- `yellow_cards` — `-1`, `red_cards` — `-3`, `own_goals` — `-2`
- `defensive_contribution` per position — `{GKP: 0, DEF: 2, MID: 2, FWD: 2}`
- Squad size `15`, starting XI `11`, max `3` per club
- Budget `1000` (tenths of a million), `transfers_sell_on_fee` `0.5`
- `max_extra_free_transfers` `4`, so free transfers cap at `5`; `transfers_cap` `20`
- Positional quotas — GKP `2` (play 1-1), DEF `5` (play 3-5), MID `5` (play 2-5), FWD `3` (play 1-3)

The same rule applies to any constant the API publishes: read it from the snapshot, version the
snapshot, and fail loudly when a drift check detects a change.

Known preseason hazards to guard against, because they silently poison features:

- `strength_attack_*` and `strength_defence_*` are `0` for all 20 teams preseason
- `strength` is `null` preseason
- `strength_overall_*` uses a 1-5 scale preseason versus roughly 1000-1400 in-season
- `entry/{id}/event/{gw}/picks/` returns `404` before the deadline
- `elements[].code` is the stable cross-season player identifier — use it, not `id`

---

## Philosophy

The project consists of independent ML systems connected through clean interfaces.

Avoid coupling.

Favor modularity.

Favor reproducibility.

Favor explainability.

---

## Engineering Standards

Never duplicate logic.

Prefer reusable abstractions.

Document assumptions.

Write deterministic pipelines.

Every prediction must be reproducible.

---

## Prediction Philosophy

Predictions are not products.

Predictions support recommendations.

Whenever possible ask:

"How does this improve downstream decisions?"

---

## Data Philosophy

Raw data is immutable.

Never overwrite raw datasets.

Always version transformations.

Always timestamp snapshots.

Everything should be reproducible.

---

## Feature Philosophy

Features are products.

Features should:

- be versioned
- be documented
- contain metadata
- track lineage
- record importance

Every feature should know:

- where it came from
- how it was generated
- when it was introduced
- whether it improved performance

---

## Machine Learning Philosophy

Do not optimize for leaderboard metrics.

Optimize for recommendation quality.

A slightly worse regression model that produces better transfer decisions is preferred.

---

## Optimization Philosophy

Predictions estimate reality.

Optimization chooses actions.

Do not confuse the two.

---

## Documentation Philosophy

Before implementing a subsystem:

- read its documentation
- understand dependencies
- verify interfaces

Only then write code.

---

## Testing Philosophy

Every subsystem must be testable independently.

- Feature generation
- Prediction
- Optimization
- Evaluation
- Deployment

should all be individually testable.

---

## Long-Term Philosophy

This repository should eventually resemble an internal ML platform.

The goal is not building an app.

The goal is building an intelligent decision system.
