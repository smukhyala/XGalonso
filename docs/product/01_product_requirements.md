# Product Requirements Document (PRD)

**Project:** XG Alonso  
**Version:** 1.0  
**Status:** Draft

---

# 1. Executive Summary

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

---

# 2. Product Vision

Users should feel like they have an AI quantitative analyst managing their squad.

Every recommendation should be:

- Personalized
- Explainable
- Data-driven
- Continuously improving

The product should optimize long-term FPL performance rather than individual predictions.

---

# 3. Target Users

## Casual Managers
Need simple explanations and one-click recommendations.

## Competitive Managers
Care about expected value, fixture swings, ownership, and price movements.

## Elite Managers
Want advanced metrics, feature importance, uncertainty, optimization settings, and model transparency.

---

# 4. Core Product Pillars

## A. Squad Intelligence

Import an FPL team and evaluate:

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

---

## B. Recommendation Engine

Recommend:

- Single transfers
- Multi-player transfer packages
- Formation changes
- Bench order
- Captain
- Vice captain
- Wildcard timing
- Future transfer plans

Each recommendation includes:

- Expected point gain
- Expected value gain
- Risk
- Confidence
- Explanation
- Supporting features

---

## C. Market Intelligence

Predict:

- Price rises
- Price drops
- Undervalued players
- Overvalued players

Compare:

Official FPL market expectations

vs.

XG Alonso intrinsic valuation.

---

## D. Feature Scientist

The platform automatically discovers predictive football features.

It should:

- Generate thousands of candidate features
- Discover interactions
- Rank importance
- Remove redundant features
- Track feature versions

Users can inspect feature importance and understand why models changed.

This is a primary differentiator.

---

## E. Representation Learning

Learn embeddings for:

- Players
- Teams
- Managers
- Fixtures

Applications:

- Similar player search
- Tactical similarity
- Transfer discovery
- Cold-start handling

---

# 5. Functional Requirements

## FR-1 Data Ingestion

Support:

- Official FPL
- Historical FPL
- Underlying football statistics
- Odds (optional MVP+)
- Injury information
- Press conference metadata
- Fixture data

---

## FR-2 Prediction Models

Predict:

- Minutes
- Points
- Price changes
- Fair value
- Injury risk

---

## FR-3 Optimization

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

---

## FR-4 Wildcard Planner

Recommend:

- Whether to wildcard
- Best wildcard squad
- Best wildcard week
- Why

---

## FR-5 Explainability

Every recommendation should answer:

Why?

What changed?

What are the risks?

What assumptions matter?

---

# 6. Non-Functional Requirements

- Full reproducibility
- Timestamped datasets
- Versioned models
- Versioned features
- Fast prediction API
- Daily retraining
- Weekly champion/challenger evaluation

---

# 7. Success Metrics

Primary:

- Incremental expected FPL points over hold baseline.

Secondary:

- Price prediction accuracy
- Recommendation acceptance
- Transfer regret
- Calibration
- Feature discovery quality

---

# 8. MVP Scope

Deliver:

1. Data ingestion
2. Feature Factory v1
3. Feature Scientist v1
4. Points model
5. Price model
6. Transfer optimizer
7. Wildcard recommender
8. Recommendation dashboard

Future versions add richer embeddings, continual learning, and automated experimentation.
