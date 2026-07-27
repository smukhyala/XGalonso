# Interaction Discovery Workflow

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Interaction Discovery |
| Version | 1.0 |
| Status | Deferred (Post-MVP) |
| Owner | ML Platform |
| Dependencies | [Feature Factory](02_feature_factory.md), [Feature Scientist](03_feature_scientist.md) |
| Last updated | 2026-07-27 |

---

## 1. Purpose and Scope

This document specifies the workflow that turns raw interaction candidates into promoted production features.

Ownership is split across two subsystems:

- The [Feature Factory](02_feature_factory.md) **generates** interaction candidates. The generator contract, supported interaction forms, candidate sources, worked examples, and the blocked-combination rules live in §7.9 of that document.
- The [Feature Scientist](03_feature_scientist.md) **evaluates** those candidates and decides which are promoted, using the five stages below.

This document covers the evaluation and promotion path only. It does not restate the generator contract.

## 2. Deferral

This workflow is deferred post-MVP.

- Automated interaction candidates are a Phase 2 item, and SHAP-driven interaction discovery is a Phase 3 item.
- Automated interaction discovery must not begin before point-in-time correctness, feature metadata, and deterministic materialization are complete in the Feature Factory.
- Until then, any interaction feature in a production feature set is hand-specified, reviewed, and carries the same Feature Card and lineage requirements as any other feature.

The stages below are specified now so the contract is stable when the work starts, not because the work is scheduled.

## 3. Stage 1: Eligibility

Features must pass:

- sufficient coverage;
- sufficient variance;
- acceptable leakage score;
- stable generation;
- semantic compatibility.

A feature that fails any eligibility criterion is excluded from candidate generation entirely, rather than generated and then screened out later.

## 4. Stage 2: Candidate Generation

Generate interactions among:

- top univariate features;
- complementary feature families;
- model residual drivers;
- domain-allowed pairs;
- targeted triples when pairwise evidence exists.

Triples are generated only where pairwise evidence already exists. The search is deliberately narrow: interaction discovery is controlled, not unrestricted.

## 5. Stage 3: Cheap Screening

Use:

- mutual information;
- univariate gain;
- permutation lift;
- residual correlation;
- small tree models;
- sampled cross-validation.

Screening exists to reduce the candidate set cheaply before any full walk-forward evaluation is spent on it.

## 6. Stage 4: Full Evaluation

Evaluate surviving interactions using walk-forward folds.

Track:

- mean validation lift;
- worst-fold lift;
- stability;
- added latency;
- missingness;
- correlation with existing features;
- target leakage risk.

## 7. Stage 5: Promotion

An interaction can enter a production feature set only when:

- average lift exceeds the minimum threshold;
- no major fold degrades beyond tolerance;
- importance is stable across retrains;
- lineage is complete;
- inference cost is acceptable.

Promotion is the only path into a versioned feature set. A promoted interaction carries a full Feature Card and lineage record exactly like a base feature, and models load it through the registry rather than by direct reference.

---

## Related documents

- [Feature Factory](02_feature_factory.md) — generates the candidates; §7.9 defines the interaction generator contract and blocked combinations
- [Feature Scientist](03_feature_scientist.md) — owns evaluation, selection, promotion, and retirement
- [Prediction Models](07_prediction_models.md) — consumes the promoted feature sets
- [Embeddings](06_embeddings.md) — source of the embedding-similarity interaction inputs
- [Player Clustering](05_player_clustering.md) — source of archetype and cluster terms used in interactions
