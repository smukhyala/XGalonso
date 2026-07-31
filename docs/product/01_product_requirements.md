# Product Requirements

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Product Requirements |
| Version | 1.0 |
| Status | Draft |
| Owner | Product |
| Dependencies | [Vision](../vision/00_vision.md), [Repository Structure](../architecture/01_repository_structure.md) |
| Last updated | 2026-07-27 |

---

## 1. Executive Summary

XG Alonso is a machine-learning powered sports intelligence platform whose first application is Fantasy Premier League (FPL).

Rather than ranking players, XG Alonso recommends optimal decisions by combining:

- Rich football data
- Automated feature engineering
- Automated interaction discovery
- Learned player/team/manager representations
- Predictive models
- Optimization under FPL constraints

The system answers:

- Who should I buy?
- Who should I sell?
- Should I roll?
- Should I take a hit?
- Should I wildcard?
- Which transfer package is best?
- Why?

Sections 1-7 describe the target product. Section 8 states what ships in the MVP and what is
deliberately deferred.

---

## 2. Product Vision

Users should feel like they have an AI quantitative analyst managing their squad.

Every recommendation should be:

- Personalized
- Explainable
- Data-driven
- Continuously improving

The product should optimize long-term FPL performance rather than individual predictions.

---

## 3. Target Users

### Casual Managers

Need simple explanations and one-click recommendations.

### Competitive Managers

Care about expected value, fixture swings, ownership, and price movements.

### Elite Managers

Want advanced metrics, feature importance, uncertainty, optimization settings, and model transparency.

---

## 4. Core Product Pillars

### A. Squad Intelligence

Import an FPL team by public team ID (D3 — no authentication) and evaluate:

- Current squad quality
- Squad structure
- Bench quality
- Budget allocation
- Future flexibility
- Injury risk
- Rotation risk

Outputs:

- Squad Health Score
- Weakest positions
- Strength by position
- Projected points (1, 3, 6 GW)

Squad structure and budget checks depend on FPL constants — squad size 15, starting XI 11, max 3
per club, budget 1000 tenths of a million, positional quotas GKP 2 / DEF 5 / MID 5 / FWD 3. These
load from a pinned snapshot of the FPL payload with a recorded fetch timestamp and a drift check.
They are never Python literals.

### B. Recommendation Engine

Recommend:

- Single transfers
- Multi-player transfer packages
- Formation changes
- Bench order
- Captain
- Vice captain
- Wildcard timing — **deferred** (D5); the wildcard is also unavailable in GW1, since the windows are GW2-19 and GW20-38
- Future transfer plans

Each recommendation includes:

- Expected point gain
- Expected value gain
- Risk
- Confidence
- Explanation
- Supporting features

### C. Market Intelligence

**Deferred (D11).** This pillar depends on the price model, and no current-season price data exists
at GW1. The design is retained; nothing here ships in the MVP.

Predict:

- Price rises
- Price drops
- Undervalued players
- Overvalued players

Compare a modelled market expectation against XG Alonso intrinsic valuation.

FPL publishes no official price predictor. Any market-expectation signal is something this project
would have to model itself, reconstructed from public transfer history and ownership movement
(D3, D6). It is not an available input to be read off the API.

### D. Feature Scientist

**Deferred (D10)** — product first, research platform deepened afterwards. The Feature Factory
ships in the MVP; the automated Feature Scientist that sits on top of it does not.

The platform automatically discovers predictive football features.

It should:

- Generate 300-700 quality candidate features (D12 — a bounded target, not thousands)
- Discover interactions
- Rank importance
- Remove redundant features
- Track feature versions

Users can inspect feature importance and understand why models changed.

This is a primary differentiator.

### E. Representation Learning

**Deferred (D10).** Added after the data and feature pipeline is reliable.

Learn embeddings for:

- Players
- Teams
- Managers
- Fixtures

Applications:

- Similar player search
- Tactical similarity
- Transfer discovery
- Cold-start handling

---

## 5. Functional Requirements

### FR-1 Data Ingestion

Support:

- Official FPL
- Historical FPL — backfilled from 2022/23, the earliest season with xG in the API (D7)
- Underlying football statistics — sourced from the FPL API's own `expected_goals`, `expected_assists`, `expected_goal_involvements`, and `expected_goals_conceded` fields, published per player per gameweek from 2022/23 onward
- Odds — **out of scope** under D6: no paid providers, no scraping
- Injury information — from FPL availability fields only
- Press conference metadata — **out of scope** under D6; no external source is permitted
- Fixture data

### FR-2 Prediction Models

Predict:

- Minutes
- Points — component-based, converted through versioned scoring rules (D8)
- Price changes — **deferred** (D11)
- Fair value — **deferred** (D11), depends on the price model
- Injury risk

### FR-3 Optimization

Optimize:

- Current GW
- 3 GW
- 6 GW

Subject to:

- Budget
- Position limits
- Club limits
- Selling prices
- Free transfers
- Hit costs

Constraint values — squad size 15, starting XI 11, max 3 per club, budget 1000 tenths of a
million, `transfers_sell_on_fee` 0.5, `max_extra_free_transfers` 4 (free transfers therefore cap at
5), `transfers_cap` 20, and the positional quotas GKP 2 (play 1-1), DEF 5 (play 3-5), MID 5 (play
2-5), FWD 3 (play 1-3) — load from a pinned snapshot of `game_config.rules` in the FPL payload,
with a recorded fetch timestamp and a drift check. They are never Python literals.

The same rule governs scoring values used to convert predicted components into points. A
goalkeeper goal is worth 10 points, not the widely assumed 6.

### FR-4 Wildcard Planner

**Deferred** (D5). The wildcard is unavailable in GW1 regardless, since the windows are GW2-19 and
GW20-38.

Recommend:

- Whether to wildcard
- Best wildcard squad
- Best wildcard week
- Why

### FR-5 Explainability

Every recommendation should answer:

- Why?
- What changed?
- What are the risks?
- What assumptions matter?

Explanations are reason-coded and derived deterministically from structured evidence. Natural
language may rephrase a reason; it may never invent one.

---

## 6. Non-Functional Requirements

- Full reproducibility
- Timestamped datasets
- Versioned models
- Versioned features
- Recommendation latency budget — a full recommendation run for one squad completes in under 60 seconds end-to-end on a local machine (D1), and a cached prediction lookup returns in under 500 ms. These are budgets to be enforced by tests, not measured results.
- Per-gameweek retraining — models retrain once per gameweek, after that gameweek's results are final. Between deadlines the system performs inference-only refreshes on new intraday data (availability, prices, fixture changes); it does not retrain daily.
- Weekly champion/challenger evaluation

---

## 7. Success Metrics

Primary:

- Incremental expected FPL points over hold baseline.

Secondary:

- Price prediction accuracy — measurable only once the price model exists (D11)
- Recommendation acceptance
- Transfer regret
- Calibration
- Feature discovery quality

---

## 8. MVP Scope

The MVP is a local, CLI-driven vertical slice that produces a defensible transfer recommendation
for a real squad by GW1 of 2026/27 (D9).

Deliver:

1. FPL ingestion from the official API (D6), backfilled from 2022/23 (D7)
2. Canonical tables in DuckDB + Parquet behind a repository interface (D2)
3. Point-in-time Feature Factory v1
4. Feature registry — definitions, versions, lineage, metadata
5. Expected-minutes baseline model
6. Component-based points baseline, converted through versioned scoring rules (D8)
7. Squad import by public FPL team ID (D3)
8. Single and double transfer optimizer, evaluated against a hold baseline
9. Reason-coded explanations for every recommendation
10. CLI as the only interface (D4)

Explicitly deferred, so the cut is visible rather than silent:

| Deferred item | Reason |
|---|---|
| Feature Scientist v1 | D10 — product first, research platform afterwards |
| Interaction discovery | D10 |
| Embeddings and representation learning | D10 |
| Price model, fair value, Market Intelligence | D11 — no current-season price data at GW1 |
| Wildcard recommender | D5, and the wildcard is unavailable in GW1 |
| Chip logic (Free Hit, Bench Boost, Triple Captain) | D5 — chip *state* is modelled, chip *logic* is not built |
| Recommendation dashboard and any web frontend | D4 — CLI first, then FastAPI, then Next.js |
| Odds and press conference ingestion | D6 — official FPL API only, no paid providers |
| Docker, cloud, hosting | D1 — local-only first |

Future versions add richer embeddings, continual learning, and automated experimentation.

---

## Related documents

- [Documentation index](../README.md)
- [Vision](../vision/00_vision.md)
- [Repository Structure](../architecture/01_repository_structure.md)
- [Data Sources](../data/01_data_sources.md)
- [Database Schema](../data/04_database_schema.md)
- [Feature Factory](../ml/02_feature_factory.md)
- [Feature Scientist](../ml/03_feature_scientist.md)
- [Embeddings](../ml/06_embeddings.md)
- [Prediction Models](../ml/07_prediction_models.md)
- [Transfer Planner](../optimization/02_transfer_planner.md)
- [Wildcard Planner](../optimization/03_wildcard_planner.md)
- [Chip Planner](../optimization/04_chip_planner.md)
- [Public API](../api/01_public_api.md)
- [Dashboard](../frontend/02_dashboard.md)
- [Build Plan](../implementation/01_build_plan.md)
