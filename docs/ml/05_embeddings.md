Representation Learning

Purpose

Embeddings encode players, teams, managers, and fixtures into dense vectors that capturefootball context beyond manually engineered statistics.

Embedding Types

Player

Represents:

Finishing

Creativity

Defensive contribution

Role

Consistency

Volatility

Minutes profile

Team

Represents:

Tactical identity

Attack

Defense

Pressing

Possession

Manager

Represents:

Rotation tendency

Formation preference

Substitution patterns

Youth usage

Fixture

Represents matchup context.

Consumers

Prediction models

Similarity search

Transfer optimizer

Feature Factory

Cold-start initialization

Storage

Each embedding stores:

entity_id

embedding_version

vector

generated_at

training_dataset

Similarity API

Expose:

nearest_neighbors(entity)

similarity(entity_a, entity_b)

cluster(entity)

Versioning

Embeddings are immutable after publication.Models reference embedding_version explicitly.

Acceptance Criteria

Deterministic generation pipeline

Version registry

Offline and online retrieval

Integrated into prediction pipeline
