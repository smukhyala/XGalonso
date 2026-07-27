# Knowledge Lab

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Knowledge Lab |
| Version | 1.0 |
| Status | Deferred (Post-MVP) |
| Owner | ML Platform |
| Dependencies | [Feature Scientist](../ml/03_feature_scientist.md), [Embeddings](../ml/06_embeddings.md), [Prediction Models](../ml/07_prediction_models.md), [Database Schema](../data/04_database_schema.md) |
| Last updated | 2026-07-27 |

---

## 1. Vision

Knowledge Lab is the long-term memory of XG Alonso.

Instead of only training models, the platform accumulates football knowledge.

Every experiment, hypothesis, feature, embedding, and recommendation becomes searchable.

---

## 2. Objectives

- Preserve discoveries across seasons.
- Explain why the platform believes something.
- Drive future feature generation.
- Provide human-readable research notes.

---

## 3. Knowledge Objects

### 3.1 Hypothesis

Example:

> 'Penalty takers outperform baseline after fixture swings.'

Fields:

- Status
- Confidence
- Supporting experiments
- First observed
- Last validated

### 3.2 Insight

Example:

> 'Opponent-adjusted rolling xG became more predictive than raw xG after GW11.'

### 3.3 Experiment

Stores:

- Dataset
- Feature set
- Model
- Metrics
- Outcome
- Decision

### 3.4 Feature Card

Links to [Feature Scientist](../ml/03_feature_scientist.md).

### 3.5 Embedding Report

Explains player clusters and nearest neighbours.

---

## 4. Weekly Research Cycle

1. Retrain models
2. Evaluate features
3. Generate hypotheses
4. Validate hypotheses
5. Publish research report
6. Update the knowledge record

Step 6 writes relational records in DuckDB plus Markdown reports. It does not update a graph
database — see § 7.

---

## 5. Report Format — Illustrative Example Only

> **This is a fabricated example showing the shape of a weekly report. It is not a result.**
> No models have been trained, no features have been generated, and no research cycle has run.
> The repository currently contains no implementation. Every number below is invented purely to
> demonstrate layout, and none of it should be cited, quoted, or treated as a measurement.

Illustrative report body:

- 312 new candidate features
- 8 promoted
- 4 retired
- Best new interaction: Expected Minutes × Opponent xGA
- Biggest drift: Transfer momentum importance increased 18%
- New hypothesis: High-press teams amplify winger upside against low-possession opponents.

A note on the first line, since the placeholder is misleading about scale: the candidate corpus is
targeted at **300-700 quality features in total** (D12), not thousands, and not 312 *new* ones per
week. A realistic weekly figure is a small number of additions against a stable corpus. The
illustrative value above was chosen to fill a slot, not to describe intended behaviour.

---

## 6. Query Examples

- Why is Palmer recommended?
- Which features matter most for defenders?
- How has fixture congestion importance changed?
- Which player archetypes emerged this season?
- Which hypotheses were rejected?

---

## 7. Future Work

An earlier version of this document proposed representing all knowledge as a graph connecting:

```text
Players ↔ Teams ↔ Managers ↔ Features ↔ Experiments ↔ Models ↔ Recommendations
```

with the stated aim of enabling semantic search and research across seasons.

**A graph database is explicitly out of scope.** The MVP uses relational records in DuckDB plus
Markdown reports (D2), and there is no second storage engine. The relationships above are real and
worth modelling, but they are modelled as join tables in DuckDB, not as nodes and edges in a
dedicated graph store.

Semantic search across seasons remains a legitimate future goal. If it is pursued, it is pursued
on top of the relational store and the existing embedding infrastructure, and any proposal to add
a graph database has to argue its case from scratch against D2.

---

## Related documents

- [Feature Scientist](../ml/03_feature_scientist.md) — produces Feature Cards, promotion decisions, and drift warnings
- [Embeddings](../ml/06_embeddings.md) — source of the embedding reports and archetype narratives
- [Player Clustering](../ml/05_player_clustering.md) — supplies the archetypes referenced in reports
- [Prediction Models](../ml/07_prediction_models.md) — supplies experiment metrics and outcomes
- [Database Schema](../data/04_database_schema.md) — where hypotheses, experiments, and decisions are persisted
- [Vision](../vision/00_vision.md) — why accumulated knowledge is a product goal rather than a research luxury
