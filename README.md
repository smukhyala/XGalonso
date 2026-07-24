# XG Alonso

> A continually learning sports intelligence platform that transforms raw football data into actionable Fantasy Premier League decisions through automated feature engineering, representation learning, machine learning, and optimization.

---

## Overview

XG Alonso is an ML-first decision engine for Fantasy Premier League (FPL).

Unlike traditional FPL tools that rank players using a fixed set of statistics or manually engineered models, XG Alonso continuously learns from football data by automatically generating features, discovering predictive interactions, learning player representations, and optimizing complete squad decisions.

The system predicts:

- Expected FPL points
- Price movements
- Expected minutes
- Player value
- Squad value
- Transfer opportunities

These predictions are then combined inside an optimization engine which recommends:

- Transfers
- Multi-player transfer packages
- Captain choices
- Bench order
- Starting XI
- Wildcard timing
- Long-term squad planning

The objective is **not** to predict football.

The objective is to maximize Fantasy Premier League performance.

---

## Philosophy

Football prediction is only one component of Fantasy Premier League.

Winning FPL requires balancing:

- Player quality
- Fixtures
- Rotation
- Injuries
- Market behavior
- Budget
- Future flexibility
- Price appreciation
- Squad structure

This project treats FPL as a constrained optimization problem rather than a ranking problem.

---

## Core Idea

Instead of asking

> "Who scores the most points?"

we ask

> "Given my current squad, budget, future fixtures, transfer availability, market dynamics, and uncertainty, what sequence of decisions maximizes my expected long-term score?"

---

## Technical Pillars

XG Alonso consists of six primary systems.

1. Data Platform

Continuously ingests football, fixture, player, market, and FPL data.

2. Feature Factory

Automatically engineers thousands of candidate features.

3. Feature Scientist

Discovers useful features, interactions, and representations.

4. Prediction Layer

Predicts football outcomes and Fantasy outcomes.

5. Optimization Engine

Finds optimal squad decisions.

6. Continual Learning

Retrains and improves after every gameweek.

---

## Technical Differentiators

Most FPL models:

```

Raw Data
↓

100 Features

↓

XGBoost

↓

Predictions

```

XG Alonso:

```

Raw Data

↓

Feature Factory

↓

3000+ Candidate Features

↓

Feature Scientist

↓

Interaction Discovery

↓

Embeddings

↓

Prediction Models

↓

Optimization Engine

↓

Recommendations

```

The project is centered around **automatic representation learning**, not simply prediction.

---

## Repository Structure

(To be completed as development progresses.)

```
xg-alonso/

docs/

backend/

frontend/

models/

pipelines/

feature_factory/

feature_scientist/

optimization/

database/

experiments/

evaluation/

deployment/

```

---

## Current Status

Planning Phase

The repository currently contains engineering documentation that defines every subsystem before implementation begins.
