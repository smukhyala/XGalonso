Player Clustering Design

Purpose

Player clustering groups footballers by learned statistical profile rather than position labels.Clusters provide richer priors for prediction, cold-start handling, recommendation diversity,and similarity search.

Inputs

Per-90 underlying stats

Expected minutes

Touch profile

Shot profile

Chance creation

Defensive actions

Set-piece role

Team style embedding

Age

Position

Pipeline

Canonical Features→ Scaling→ Dimensionality Reduction (optional)→ Clustering→ Stability Evaluation→ Cluster Labels→ Cluster Embeddings→ Production Registry

Requirements

Recomputed after every completed gameweek

Versioned

Stable across retraining

Human-readable summaries

Outputs

For every player:

cluster_id

cluster_version

nearest_neighbors

distance_to_centroid

confidence

archetype_summary

Product Uses

Similar-player search

Cold-start priors

Recommendation alternatives

Feature generation

Transfer suggestions

Acceptance Criteria

Stable clusters

Versioned outputs

Integrated into feature store
