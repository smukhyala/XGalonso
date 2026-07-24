# CLAUDE.md

You are contributing to XG Alonso.

This repository prioritizes engineering quality over implementation speed.

Never write code that only satisfies today's requirements.

Every component should be designed assuming the platform will continue growing.

---

# Philosophy

The project consists of independent ML systems connected through clean interfaces.

Avoid coupling.

Favor modularity.

Favor reproducibility.

Favor explainability.

---

# Engineering Standards

Never duplicate logic.

Prefer reusable abstractions.

Document assumptions.

Write deterministic pipelines.

Every prediction must be reproducible.

---

# Prediction Philosophy

Predictions are not products.

Predictions support recommendations.

Whenever possible ask:

"How does this improve downstream decisions?"

---

# Data Philosophy

Raw data is immutable.

Never overwrite raw datasets.

Always version transformations.

Always timestamp snapshots.

Everything should be reproducible.

---

# Feature Philosophy

Features are products.

Features should:

be versioned

be documented

contain metadata

track lineage

record importance

Every feature should know:

where it came from

how it was generated

when it was introduced

whether it improved performance

---

# Machine Learning Philosophy

Do not optimize for leaderboard metrics.

Optimize for recommendation quality.

A slightly worse regression model that produces better transfer decisions is preferred.

---

# Optimization Philosophy

Predictions estimate reality.

Optimization chooses actions.

Do not confuse the two.

---

# Documentation Philosophy

Before implementing a subsystem:

read its documentation

understand dependencies

verify interfaces

Only then write code.

---

# Testing Philosophy

Every subsystem must be testable independently.

Feature generation

Prediction

Optimization

Evaluation

Deployment

should all be individually testable.

---

# Long-Term Philosophy

This repository should eventually resemble an internal ML platform.

The goal is not building an app.

The goal is building an intelligent decision system.
