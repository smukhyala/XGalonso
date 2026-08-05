<!-- claims
package: packages/discovery
symbols: xg_alonso.discovery.embeddings, xg_alonso.discovery.clusters
-->

# Representation Learning

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Embeddings |
| Version | 1.1 |
| Status | Partially shipped — player embeddings only |
| Owner | ML Platform |
| Dependencies | [Feature Factory](02_feature_factory.md), [Player Clustering](05_player_clustering.md), [Prediction Models](07_prediction_models.md) |
| Last updated | 2026-08-04 |

> **Status correction, 2026-08-04.** This document carried `Deferred (Post-MVP)` after part of it had
> shipped. Being precise about which part:
>
> | Section | Reality |
> |---|---|
> | 2.1 Player embeddings | **Built**, in `discovery/embeddings.py`. Seeded, deterministic, versioned, and consumed by clustering |
> | 2.2 Team embeddings | **Not built.** The preseason `strength_*` guard below is still the right warning for whoever builds them |
> | 2.3 Manager embeddings | **Not built**, and not planned — FPL publishes nothing that would identify a head coach's rotation tendency without inference from lineups |
> | 2.4 Fixture embeddings | **Not built.** Matchup context is carried by the opponent-strength features in `features/opponent.py` instead |
>
> It shipped inside `packages/discovery`, not as `packages/embeddings`; that package does not exist.
> Similarity search (section 3) is exposed as cluster membership rather than a nearest-neighbour
> index over the vectors.
>
> See [Player Embeddings and Clusters](../player_embeddings_and_clusters.md) for what was built.

---

## 1. Purpose

Embeddings encode players, teams, managers, and fixtures into dense vectors that capture football
context beyond manually engineered statistics.

---

## 2. Embedding Types

### 2.1 Player

Represents:

- Finishing
- Creativity
- Defensive contribution
- Role
- Consistency
- Volatility
- Minutes profile

### 2.2 Team

Represents:

- Tactical identity
- Attack
- Defense
- Pressing
- Possession

Team embeddings must not be trained on the preseason `strength_*` fields without a guard:
`strength_attack_*` and `strength_defence_*` are `0` for all 20 teams before the season starts,
`strength` is `null`, and `strength_overall_*` uses a 1-5 scale preseason versus roughly 1000-1400
in-season. Training on those values silently encodes a constant.

### 2.3 Manager

Represents:

- Rotation tendency
- Formation preference
- Substitution patterns
- Youth usage

### 2.4 Fixture

Represents matchup context.

---

## 3. Consumers

- Prediction models
- Similarity search
- Transfer optimizer
- Feature Factory
- Cold-start initialization

---

## 4. Storage

Each embedding stores:

- `entity_id`
- `embedding_version`
- `vector`
- `generated_at`
- `training_dataset`

For players, `entity_id` is `elements[].code`, the stable cross-season identifier, so embeddings
remain joinable across seasons. Vectors are persisted as Parquet and read through the repository
interface (D2).

---

## 5. Similarity API

Expose:

- `nearest_neighbors(entity)`
- `similarity(entity_a, entity_b)`
- `cluster(entity)`

---

## 6. Versioning

Embeddings are immutable after publication. Models reference `embedding_version` explicitly.

---

## 7. Acceptance Criteria

- Deterministic generation pipeline
- Version registry
- Offline and online retrieval
- Integrated into prediction pipeline

Because representation learning is post-MVP, embedding-derived features must be null-safe in the
same way cluster-derived features are: a missing `embedding_version` degrades a consumer to its
defined fallback rather than failing.

---

## Related documents

- [Player Clustering](05_player_clustering.md) — clusters are computed over these representations
- [Feature Factory](02_feature_factory.md) — turns embeddings and similarities into candidate features
- [Feature Scientist](03_feature_scientist.md) — evaluates embedding-derived features and their interactions
- [Prediction Models](07_prediction_models.md) — the primary consumer, and the source of cold-start initialization needs
- [Transfer Planner](../optimization/02_transfer_planner.md) — uses similarity to propose alternatives
