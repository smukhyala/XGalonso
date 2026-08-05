<!-- claims
package: packages/discovery
symbols: xg_alonso.discovery.acceptance, xg_alonso.discovery.registry:DiscoveryRegistry, xg_alonso.discovery.utility:feature_utility, xg_alonso.discovery.hypotheses:generate_from_residuals
commands: xg discover, xg build-discovery-frame, xg importance
routes: GET /features/discovered, GET /hypotheses
-->

# Feature Scientist Design

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Feature Scientist |
| Version | 1.1 |
| Status | Superseded — the capability shipped, this interface did not |
| Owner | ML Platform |
| Dependencies | [Feature Factory](02_feature_factory.md), [Prediction Models](07_prediction_models.md), [Knowledge Lab](../research/01_knowledge_lab.md) |
| Last updated | 2026-08-04 |

> **Status correction, 2026-08-04.** This document carried `Deferred (Post-MVP)` while the capability
> it describes was shipping. That is now wrong in the other direction, and the honest statement is
> narrower than "implemented":
>
> **The capability shipped, in `packages/discovery`.** Feature evaluation, redundancy handling,
> selection, acceptance and retirement all run — `xg discover` end to end, surfaced at
> `GET /features/discovered` and `GET /hypotheses`. See
> [Objective-Conditioned Feature Discovery](../objective_conditioned_feature_discovery.md), which
> describes what was actually built.
>
> **The interface below was not built.** There is no `packages/feature_scientist`, no Feature Card,
> no Feature Report, no twelve-stage pipeline, and no SHAP stage. What shipped is a different design
> reached by a different route: hypotheses aimed at *measured residual weakness*, compiled to a safe
> expression tree, gated by a leakage proof, backtested walk-forward against noise and shuffled
> controls, and scored by **utility under a stated objective** rather than by a global accuracy
> metric. That last difference is the substantive one — this document assumes one right answer for
> every manager, and the shipped system does not.
>
> The design below is retained because its stage vocabulary still names real concerns, several of
> which the shipped loop handles differently and one of which (redundancy analysis against the
> existing accepted set) it handles less thoroughly.

---

## 1. Purpose

The Feature Scientist is an AutoML-inspired research layer that continuously evaluates, selects,
versions, and retires engineered features created by the Feature Factory.

Unlike the Feature Factory, which is deterministic, the Feature Scientist is adaptive. Its
objective is to improve downstream recommendation quality rather than simply improve prediction
metrics.

The candidate corpus it operates over is deliberately bounded: D12 caps it at **300-700 quality
candidate features, not thousands**. The Feature Scientist's job is to keep that corpus sharp, not
to grow it without limit. The current corpus sits below the floor of that range rather than at it —
224 distinct columns — which is a ceiling being respected, not a target being hit.

---

## 2. Responsibilities

- Evaluate every candidate feature.
- Discover nonlinear feature interactions.
- Identify redundant features.
- Build target-specific feature sets.
- Produce Feature Cards and Feature Reports.
- Recommend feature promotion or retirement.
- Maintain a versioned Feature Registry.

---

## 3. Pipeline

The evaluation pipeline runs twelve stages in a fixed order. Every stage is deterministic given
its inputs and its recorded configuration.

```mermaid
flowchart TD
    A[Candidate Features] --> B[Data Quality Gate]
    B --> C[Leakage Detection]
    C --> D[Coverage Check]
    D --> E[Mutual Information]
    E --> F[Correlation Clustering]
    F --> G[Tree-based Importance]
    G --> H[SHAP Analysis]
    H --> I[Interaction Discovery]
    I --> J[Walk-forward Validation]
    J --> K[Stability Analysis]
    K --> L[Promotion / Retirement]
```

---

## 4. Feature Lifecycle

```mermaid
flowchart LR
    A[Generated] --> B[Candidate]
    B --> C[Evaluating]
    C --> D[Production]
    D --> E[Deprecated]
    E --> F[Archived]
```

Each feature stores:

- Version
- Source columns
- Generator
- Parameters
- Lineage
- Coverage
- Missing rate
- Importance
- Stability score
- SHAP rank
- Models using the feature
- Date introduced
- Date retired
- Retirement reason

---

## 5. Feature Registry

The registry is the source of truth.

It answers:

- Which models use this feature?
- Which generator created it?
- Which experiments promoted it?
- What replaced it?
- How has its importance changed over time?

The registry is stored relationally in DuckDB behind the repository interface (D2). There is no
separate metadata service and no graph database.

---

## 6. Automatic Interaction Discovery

Search interactions only between semantically compatible feature families.

Examples:

- Expected Minutes × Fixture Difficulty
- Home × Rolling xG
- Manager Rotation × Fixture Congestion
- Team xG × Opponent xGA
- Embedding Similarity × Opponent Cluster

Reject interactions with:

- Target leakage
- Sparse support
- High instability
- Duplicate information

---

## 7. Feature Reports

Every retraining produces:

- New candidate features
- Promoted features
- Retired features
- Largest gain
- Largest loss
- Top interactions
- Drift warnings

---

## 8. Acceptance Criteria

- Reproducible feature selection
- Walk-forward validated
- Feature metadata complete
- Automatic reports generated
- Feature sets versioned

---

## Related documents

- [Feature Factory](02_feature_factory.md) — deterministic generation of the candidate features this layer evaluates
- [Prediction Models](07_prediction_models.md) — the downstream consumers whose recommendation quality is the objective
- [Player Clustering](05_player_clustering.md) — supplies cluster-derived candidate features
- [Embeddings](06_embeddings.md) — supplies embedding-derived candidate features
- [Knowledge Lab](../research/01_knowledge_lab.md) — stores Feature Cards, experiments, and promotion decisions
- [Repository Structure](../architecture/01_repository_structure.md) — package boundaries and dependency rules
