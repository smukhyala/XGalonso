# Product Requirements

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Product Requirements |
| Version | 1.1 |
| Status | Draft — §4 and §8 reconciled against the code 2026-08-04; §1-3, §5-7 are unverified target product |
| Owner | Product |
| Dependencies | [Vision](../vision/00_vision.md), [Repository Structure](../architecture/01_repository_structure.md) |
| Last updated | 2026-08-04 |

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

**Shipped, after the product, as D10 sequenced it** — in `packages/discovery`, and not through the
interface [Feature Scientist](../ml/03_feature_scientist.md) specified. Surfaced by `xg discover`
and the `/discovery` view.

The platform automatically discovers predictive football features, conditioned on the manager's
objective rather than on a single global accuracy metric.

| Requirement | Status |
|---|---|
| Generate 300-700 quality candidate features (D12 — a bounded target, not thousands) | Partly. 231 distinct columns today; D12 is a ceiling and the build is under it |
| Discover interactions | Expressible via `Arith(MUL)` in the DSL; the set search exists but is not yet wired. See [Interaction Discovery](../ml/04_interaction_discovery.md) |
| Rank importance | Shipped — `xg importance`, `GET /features/importance`, measured out of sample |
| Remove redundant features | Partly. Acceptance rejects a candidate that does not add utility over the existing set; there is no standing redundancy sweep over accepted features |
| Track feature versions | Shipped — `CATALOGUE_VERSION`, the discovery registry, and per-experiment manifests |

Users can inspect feature importance and understand why models changed. Rejected candidates are
shown alongside accepted ones; a discovery surface that shows only its successes is a marketing
page.

### E. Representation Learning

**Partially shipped**, in `discovery/embeddings.py` and `discovery/clusters.py`.

| Embedding | Status |
|---|---|
| Players | Built — seeded, deterministic, versioned |
| Teams | Not built |
| Managers | Not built, and not planned; FPL publishes nothing that would identify a head coach's rotation tendency without inference from lineups |
| Fixtures | Not built. Matchup context comes from the opponent-strength features instead |

Applications:

- Similar player search — exposed as **cluster membership**, not a nearest-neighbour index
- Tactical similarity — not built; needs team embeddings
- Transfer discovery — partly, through cluster-conditioned candidate generation
- Cold-start handling — partly, through cluster priors

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

Delivered:

| # | Item | Status |
|---|---|---|
| 1 | FPL ingestion from the official API (D6), backfilled from 2022/23 (D7) | Built |
| 2 | Canonical tables behind a repository interface (D2) | Built — but on **Parquet only**. `DuckDBTableStore` exists behind the same protocol and is constructed nowhere outside tests; there is no `.duckdb` file. D2 named both, and the boundary it required is what made using only one a non-event |
| 3 | Point-in-time Feature Factory v1 | Built, with a mechanical leakage harness and a negative control |
| 4 | Feature registry — definitions, versions, lineage, metadata | **Not built as specified.** There is no `feature_registry` table. Definitions are frozen `FeatureSpec` values in code, versioned by `CATALOGUE_VERSION`; lineage is carried on the artifact manifest. The discovery registry (nine `discovery_*` tables) covers discovered features only |
| 5 | Expected-minutes baseline model | Built |
| 6 | Component-based points baseline via versioned scoring rules (D8) | Built |
| 7 | Squad import by public FPL entry ID (D3) | Built, with `--squad-file` for the pre-deadline case |
| 8 | Transfer optimizer against a hold baseline | Built. Single transfer; the double-transfer package is not |
| 9 | Reason-coded explanations for every recommendation | Built |
| 10 | CLI as the only interface (D4) | Superseded — the API and web app also ship. The CLI-first ordering held |

Still deferred, so the cut stays visible rather than silent:

| Deferred item | Reason |
|---|---|
| Price model, fair value, Market Intelligence | D11 — no current-season price data at GW1 |
| Wildcard recommender | D5, and the wildcard is unavailable in GW1 |
| Chip logic (Free Hit, Bench Boost, Triple Captain) | D5 — chip *state* is modelled, chip *logic* is not built |
| Multi-transfer packages | Not deferred by decision; simply not built. Slice 1 is single-transfer |
| Odds and press conference ingestion | D6 — official FPL API only, no paid providers |
| Docker, cloud, hosting | D1 — local-only first |

Three items this section listed as deferred have since shipped: the Feature Scientist, interaction
*expression*, and representation learning — all in `packages/discovery`, all after the product, as
D10 sequenced. See §4D and §4E for exactly which parts.

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
