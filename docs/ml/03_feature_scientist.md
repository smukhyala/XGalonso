Feature Scientist Design

Purpose

The Feature Scientist is an AutoML-inspired research layer that continuously evaluates,selects, versions, and retires engineered features created by the Feature Factory.

Unlike the Feature Factory, which is deterministic, the Feature Scientist is adaptive.Its objective is to improve downstream recommendation quality rather than simply improveprediction metrics.

Responsibilities

Evaluate every candidate feature.

Discover nonlinear feature interactions.

Identify redundant features.

Build target-specific feature sets.

Produce Feature Cards and Feature Reports.

Recommend feature promotion or retirement.

Maintain a versioned Feature Registry.

Pipeline

Candidate Features→ Data Quality Gate→ Leakage Detection→ Coverage Check→ Mutual Information→ Correlation Clustering→ Tree-based Importance→ SHAP Analysis→ Interaction Discovery→ Walk-forward Validation→ Stability Analysis→ Promotion / Retirement

Feature Lifecycle

Generated→ Candidate→ Evaluating→ Production→ Deprecated→ Archived

Each feature stores:

Version

Source columns

Generator

Parameters

Lineage

Coverage

Missing rate

Importance

Stability score

SHAP rank

Models using the feature

Date introduced

Date retired

Retirement reason

Feature Registry

The registry is the source of truth.

It answers:

Which models use this feature?

Which generator created it?

Which experiments promoted it?

What replaced it?

How has its importance changed over time?

Automatic Interaction Discovery

Search interactions only between semantically compatible feature families.

Examples:

Expected Minutes × Fixture Difficulty

Home × Rolling xG

Manager Rotation × Fixture Congestion

Team xG × Opponent xGA

Embedding Similarity × Opponent Cluster

Reject interactions with:

Target leakage

Sparse support

High instability

Duplicate information

Feature Reports

Every retraining produces:

New candidate features

Promoted features

Retired features

Largest gain

Largest loss

Top interactions

Drift warnings

Acceptance Criteria

Reproducible feature selection

Walk-forward validated

Feature metadata complete

Automatic reports generated

Feature sets versioned
