# Vision

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Vision |
| Version | 1.0 |
| Status | Active |
| Owner | Product |
| Dependencies | None |
| Last updated | 2026-07-27 |

---

## Mission

Build a Fantasy Premier League decision engine that tells a manager what to do, not merely who is
likely to score.

The long-term vision is not simply predicting player performance.

The vision is constructing a continually learning sports intelligence platform capable of
transforming large quantities of football data into strategic decisions under the game's real
constraints.

Three things distinguish the approach:

- **Automated feature engineering** — features are generated, versioned, and evaluated by the
  system rather than hand-picked once and frozen.
- **Decision optimization over prediction** — the optimizer is the product; predictions are inputs
  to it.
- **Continual learning** — every completed gameweek becomes training data, so the platform
  improves across the season instead of being tuned once.

Fantasy Premier League is the first application.

The underlying architecture should generalize to other fantasy sports, betting markets, player
valuation, and sports analytics.

---

## Why This Exists

Most Fantasy Premier League tools rely on:

- manually selected features
- static prediction models
- subjective recommendations
- simple rankings

These systems answer:

> "Who is likely to score points?"

That is not the user's actual problem.

The user's problem is:

> "What should I do next?"

This requires solving a decision problem rather than a prediction problem.

---

## Product Principles

### Principle 1

Everything begins with data.

Predictions cannot exceed the quality of the information available.

The system should aggressively collect, normalize, validate, and enrich football data.

### Principle 2

Representation matters.

Raw statistics are rarely sufficient.

The system should learn latent representations of:

- players
- teams
- managers
- fixtures
- tactical styles

These representations should improve downstream prediction.

### Principle 3

Feature engineering is a first-class citizen.

Feature engineering is not preprocessing.

Feature engineering is one of the primary learning systems.

The Feature Factory should become one of the largest components of the project.

### Principle 4

Predictions are intermediate outputs.

Predicted points do not help users.

Recommendations help users.

Every prediction ultimately feeds an optimization engine.

### Principle 5

Recommendations must be explainable.

Users should understand:

- why a recommendation exists
- what changed
- what assumptions matter
- how confident the system is

### Principle 6

The system continuously improves.

Every completed gameweek becomes new training data.

The platform should become stronger throughout the season.

---

## Long-Term Goal

The system should eventually function like an analyst.

Not:

```text
Player A
6.8 expected points
```

Instead:

```text
Player A is undervalued relative to his expected production.

Although his ownership remains low, our model expects
a 73% probability of a price rise before the deadline.

His projected six-game output exceeds similarly priced
alternatives by 9.2 points while carrying below-average
rotation risk.
```

The system should produce conclusions rather than numbers.

The illustration above is about output shape, not about a shipped capability: the price-rise
probability depends on the price model, which is deferred (D11) because no current-season price
data exists at GW1. The numbers shown are illustrative, not measured.

---

## Success Criteria

The project succeeds if users trust recommendations rather than predictions.

The recommendation engine is the product.

Prediction models exist to support it.

---

## Competitive Advantage

This project differentiates itself through five technical innovations.

These describe the mature platform. Sequencing follows D10 — product first, research platform
deepened afterwards — so interaction discovery, learned representations, and the automated Feature
Scientist land after the first end-to-end recommendation works.

### Automated Feature Engineering

300-700 quality candidate features generated automatically (D12).

The target is deliberately bounded. Point-in-time correctness, documented lineage, and measured
contribution matter more than raw candidate count.

### Automated Interaction Discovery

Discovery of nonlinear football relationships without manual engineering.

### Learned Representations

- Player embeddings
- Team embeddings
- Manager embeddings
- Fixture embeddings

### Decision Optimization

Optimization over complete squads rather than individual players.

Scoring values and squad constraints used by the optimizer load from a pinned snapshot of the FPL
payload with a recorded fetch timestamp and a drift check. They are never Python literals — a
goalkeeper goal is worth 10 points, not the widely assumed 6, and transcription is how that error
spreads.

### Continual Learning

The platform improves every gameweek.

---

## Non Goals

The project is NOT intended to:

- predict every football statistic perfectly
- replace human football knowledge
- optimize solely for historical accuracy

Instead, optimize decision quality.

The objective is maximizing Fantasy Premier League performance.

---

## Related documents

- [Documentation index](../README.md)
- [Product Requirements](../product/01_product_requirements.md)
- [Repository Structure](../architecture/01_repository_structure.md)
- [Feature Factory](../ml/02_feature_factory.md)
- [Feature Scientist](../ml/03_feature_scientist.md)
- [Embeddings](../ml/06_embeddings.md)
- [Prediction Models](../ml/07_prediction_models.md)
- [Transfer Planner](../optimization/02_transfer_planner.md)
- [Build Plan](../implementation/01_build_plan.md)
