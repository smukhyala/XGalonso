# XG Alonso Documentation

Navigation index for the XG Alonso engineering documentation suite.

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Documentation Index |
| Version | 1.0 |
| Status | Active |
| Owner | Platform |
| Dependencies | None |
| Last updated | 2026-07-27 |

---

## How to read this suite

Documents carry a `Status` field from a closed vocabulary. **Read it before treating a document
as a commitment.**

| Status | Meaning |
|---|---|
| `Active` | Describes what is being built now |
| `Build Specification` | Approved design, implementation in progress or imminent |
| `Draft` | Written but not yet reconciled against binding decisions |
| `Deferred (Post-MVP)` | Design retained deliberately; **not** in the MVP. Nothing here ships yet |

A `Deferred (Post-MVP)` document is not dead — it holds real design substance and its contracts
constrain what the MVP must not make impossible. It is simply not a promise about this release.

---

## Start here

1. [Vision](vision/00_vision.md) — why this exists and what it optimizes for
2. [Product Requirements](product/01_product_requirements.md) — the workflows the system must support
3. [Repository Structure](architecture/01_repository_structure.md) — package boundaries and dependency rules
4. [Build Plan](implementation/01_build_plan.md) — phases and sequencing
5. `../CLAUDE.md` — engineering standards that override default behaviour

---

## Binding decisions

These were settled with the project owner and are **not** open for relitigation during
implementation. Individual documents may predate them; where a document conflicts, the decision
wins and the document is scheduled for correction.

| # | Decision |
|---|---|
| D1 | Local-only first — no Docker, cloud, or hosting in the first slice |
| D2 | DuckDB + Parquet only; no PostgreSQL. Storage behind a repository interface |
| D3 | Public FPL team ID only; no authentication. Prices reconstructed from public transfer history |
| D4 | CLI first, then FastAPI, then Next.js. No frontend in the MVP |
| D5 | No chips in the MVP. Chip *state* is modelled; chip *logic* is not built |
| D6 | Zero budget, no paid providers. Official FPL API first; **relaxed 2026-07-29** to permit fetching free public data the API does not publish, and only from origins whose `robots.txt` permits it. See `match_event_data.md` |
| D7 | Historical backfill from 2022/23 — the earliest season with xG in the API |
| D8 | Component-based points modelling, converted through versioned scoring rules |
| D9 | Useful by GW1 of 2026/27, refined in-season |
| D10 | Product first; research platform deepened afterwards |
| D11 | Price model deferred — no current-season price data exists at GW1 |
| D12 | 300–700 quality candidate features, **not** thousands |

---

## Architecture

| Document | Status | Purpose |
|---|---|---|
| [01_repository_structure.md](architecture/01_repository_structure.md) | Build Specification | Monorepo layout, package ownership, dependency direction |
| `02_system_architecture.md` | Planned | End-to-end data and control flow |
| `03_service_boundaries.md` | Planned | What may call what, and why |

**Dependency direction.** Arrows point toward dependencies; nothing may point back.

```mermaid
flowchart TD
    apps --> explanations
    apps --> optimization
    apps --> prediction
    explanations --> feature_scientist
    optimization --> feature_factory
    prediction --> feature_factory
    prediction --> embeddings
    feature_scientist --> domain
    feature_factory --> domain
    embeddings --> domain
    feature_factory --> data_contracts
    domain --> data_contracts
    evaluation --> data_contracts
    observability --> data_contracts
    pipelines -.orchestrates.-> apps
```

`pipelines` may orchestrate every package. `domain` holds pure football and FPL rules with no
database or API dependency.

---

## Data

| Document | Status | Purpose |
|---|---|---|
| [01_data_sources.md](data/01_data_sources.md) | Draft | Source register, endpoints, cadence, timestamps |
| `02_data_contracts.md` | Planned | Shared schemas and the four-timestamp rule |
| `03_identity_resolution.md` | Planned | Cross-season player identity, transfers, promoted clubs |
| [04_database_schema.md](data/04_database_schema.md) | Draft | Canonical table definitions |
| `05_data_quality.md` | Planned | Validation gates, quarantine policy, leakage register |

**Medallion layers.** Bronze is immutable and append-only; nothing overwrites a raw snapshot.

```mermaid
flowchart LR
    A[Official FPL API] --> B[Bronze: raw snapshots]
    B --> C[Silver: canonical normalized]
    C --> D[Gold: point-in-time model-ready]
    D --> E[Features]
```

---

## Machine learning

| Document | Status | Purpose |
|---|---|---|
| `01_ml_architecture.md` | Planned | How the ML subsystems compose |
| [02_feature_factory.md](ml/02_feature_factory.md) | Build Specification | Deterministic candidate-feature generation. **The deepest spec in the repo** |
| [03_feature_scientist.md](ml/03_feature_scientist.md) | Deferred (Post-MVP) | Automated feature evaluation and promotion |
| [04_interaction_discovery.md](ml/04_interaction_discovery.md) | Deferred (Post-MVP) | Controlled interaction search and gating |
| [05_player_clustering.md](ml/05_player_clustering.md) | Deferred (Post-MVP) | Archetype clustering for priors and similarity |
| [06_embeddings.md](ml/06_embeddings.md) | Deferred (Post-MVP) | Representation learning |
| [07_prediction_models.md](ml/07_prediction_models.md) | Draft | Minutes, components, points, price, fair value |
| `08_continual_learning.md` | Planned | Per-gameweek retraining and champion/challenger |
| `09_evaluation.md` | Planned | Walk-forward protocol and metrics |

---

## Optimization

The optimizer is the product; predictions are inputs to it.

| Document | Status | Purpose |
|---|---|---|
| `01_squad_optimizer.md` | Planned | Constraints, starting XI, formation, bench |
| [02_transfer_planner.md](optimization/02_transfer_planner.md) | Draft | Single, double and package transfers |
| [03_wildcard_planner.md](optimization/03_wildcard_planner.md) | Deferred (Post-MVP) | Wildcard squad and timing |
| [04_chip_planner.md](optimization/04_chip_planner.md) | Deferred (Post-MVP) | Free Hit, Bench Boost, Triple Captain |
| `05_captain_and_bench.md` | Planned | Captaincy and bench ordering |

---

## Interfaces

| Document | Status | Purpose |
|---|---|---|
| [01_public_api.md](api/01_public_api.md) | Draft | CLI surface first, HTTP API second |
| `02_internal_contracts.md` | Planned | Package-to-package contracts |
| `01_information_architecture.md` | Planned | Web app screens and navigation |
| [02_dashboard.md](frontend/02_dashboard.md) | Deferred (Post-MVP) | Dashboard views |
| `03_design_system.md` | Planned | Visual language |

---

## Implementation

| Document | Status | Purpose |
|---|---|---|
| [01_build_plan.md](implementation/01_build_plan.md) | Draft | Phase sequencing |
| `02_mvp_milestones.md` | Planned | Milestone acceptance criteria |
| `03_testing_strategy.md` | Planned | Unit, property, golden, integration, leakage, e2e |
| `04_release_checklist.md` | Planned | Release gates and definition of done |

---

## Research

| Document | Status | Purpose |
|---|---|---|
| [01_knowledge_lab.md](research/01_knowledge_lab.md) | Deferred (Post-MVP) | Accumulated hypotheses, experiments, outcomes |

---

## Conventions

- **Scoring and constraint constants are never transcribed.** They load from a pinned snapshot of
  the FPL payload with a recorded fetch timestamp, verified by a drift check on every ingest.
  A goalkeeper goal is worth 10 points, not the widely assumed 6 — transcription is how that kind
  of error enters a codebase silently.
- **Point-in-time correctness is non-negotiable.** Every feature reflects only information whose
  `available_time` precedes the prediction timestamp.
- **Walk-forward validation only.** Random splits on temporal data are forbidden.
- **Raw data is immutable.** Transformations are versioned; snapshots are timestamped.
- **Notebooks are never the only implementation** of a source adapter, feature, model, backtest,
  or optimizer.
