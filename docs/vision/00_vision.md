# Vision

Version: 1.0

---

# Mission

Build the world's most technically sophisticated Fantasy Premier League decision engine.

The long-term vision is not simply predicting player performance.

The vision is constructing a continually learning sports intelligence platform capable of transforming massive quantities of football data into optimal strategic decisions.

Fantasy Premier League is the first application.

The underlying architecture should generalize to other fantasy sports, betting markets, player valuation, and sports analytics.

---

# Why This Exists

Most Fantasy Premier League tools rely on:

- manually selected features
- static prediction models
- subjective recommendations
- simple rankings

These systems answer:

"Who is likely to score points?"

That is not the user's actual problem.

The user's problem is:

"What should I do next?"

This requires solving a decision problem rather than a prediction problem.

---

# Product Principles

## Principle 1

Everything begins with data.

Predictions cannot exceed the quality of the information available.

The system should aggressively collect, normalize, validate, and enrich football data.

---

## Principle 2

Representation matters.

Raw statistics are rarely sufficient.

The system should learn latent representations of:

- players
- teams
- managers
- fixtures
- tactical styles

These representations should improve downstream prediction.

---

## Principle 3

Feature engineering is a first-class citizen.

Feature engineering is not preprocessing.

Feature engineering is one of the primary learning systems.

The Feature Factory should become one of the largest components of the project.

---

## Principle 4

Predictions are intermediate outputs.

Predicted points do not help users.

Recommendations help users.

Every prediction ultimately feeds an optimization engine.

---

## Principle 5

Recommendations must be explainable.

Users should understand:

why a recommendation exists

what changed

what assumptions matter

how confident the system is

---

## Principle 6

The system continuously improves.

Every completed gameweek becomes new training data.

The platform should become stronger throughout the season.

---

# Long-Term Goal

The system should eventually function like an analyst.

Not:

```

Player A
6.8 expected points

```

Instead:

```

Player A is undervalued relative to his expected production.

Although his ownership remains low, our model expects
a 73% probability of a price rise before the deadline.

His projected six-game output exceeds similarly priced
alternatives by 9.2 points while carrying below-average
rotation risk.

```

The system should produce conclusions rather than numbers.

---

# Success Criteria

The project succeeds if users trust recommendations rather than predictions.

The recommendation engine is the product.

Prediction models exist to support it.

---

# Competitive Advantage

This project differentiates itself through five technical innovations.

## Automated Feature Engineering

Thousands of candidate features generated automatically.

---

## Automated Interaction Discovery

Discovery of nonlinear football relationships without manual engineering.

---

## Learned Representations

Player embeddings.

Team embeddings.

Manager embeddings.

Fixture embeddings.

---

## Decision Optimization

Optimization over complete squads rather than individual players.

---

## Continual Learning

The platform improves every gameweek.

---

# Non Goals

The project is NOT intended to:

- predict every football statistic perfectly
- replace human football knowledge
- optimize solely for historical accuracy

Instead, optimize decision quality.

The objective is maximizing Fantasy Premier League performance.
