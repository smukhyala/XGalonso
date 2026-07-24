Feature Factory Design Specification

Project: XG AlonsoDocument: Feature FactoryVersion: 1.0Status: Build SpecificationPrimary Owner: ML PlatformDepends On: Data contracts, canonical player identity, raw and normalized data layersConsumed By: Feature Scientist, prediction models, embedding pipelines, recommendation engine

1. Purpose

The Feature Factory is the central machine-learning infrastructure component in XG Alonso.

Its responsibility is to transform timestamped football, Fantasy Premier League, market, fixture, availability, tactical, and contextual data into a large, reproducible, versioned catalog of candidate features.

The Feature Factory does not decide which features are useful. It generates valid candidates with explicit lineage and hands them to the Feature Scientist for evaluation, interaction discovery, selection, promotion, and retirement.

The system must support:

deterministic feature generation;

point-in-time correctness;

reusable feature generators;

feature metadata and lineage;

batch and incremental execution;

multiple prediction targets;

thousands of candidate features;

safe interaction generation;

feature versioning;

feature quality validation;

offline training and online inference parity.

The Feature Factory is designed around one rule:

Every feature must represent only information available at the prediction timestamp.

2. Product Role

The Feature Factory supplies inputs for the following products:

Expected-minutes prediction

Next-gameweek points prediction

Multi-gameweek points prediction

Goal, assist, clean-sheet, and bonus component models

Price-rise and price-fall prediction

Player fair-value estimation

Player, team, manager, and fixture embeddings

Transfer optimization

Wildcard timing

Recommendation explanation

The system must be target-aware without becoming target-specific. A feature can be globally reusable, model-specific, horizon-specific, or prohibited for particular targets.

Example:

net_transfers_last_6h is highly relevant to price-change prediction.

The same feature may be excluded from intrinsic football-performance models to prevent market behavior from dominating performance estimates.

official_ep_next must be banned from points-model training if it contains post-deadline or derived forward information unavailable at prediction time.

3. Core Concepts

3.1 Entity Grain

The canonical modeling grain is:

player_id × prediction_timestamp × target_fixture_or_gameweek

For most gameweek models, one row represents one player immediately before a gameweek deadline.

For intraday price models, one row represents one player at a timestamped market snapshot.

For fixture-level models, one row represents one player-fixture pair.

3.2 Feature Definition

A feature definition is the declarative specification of how a feature is constructed.

Example:

name: player_xg_ewma_5
entity: player_fixture
source_column: xg
generator: exponentially_weighted_mean
lookback:
  type: appearances
  value: 5
parameters:
  alpha: 0.45
availability_delay: 0h
null_policy: position_median
tags:
  - player_performance
  - expected_goals
  - recency

3.3 Feature Value

A feature value is the output for a specific entity row and feature version.

player_id=123
prediction_timestamp=2026-09-11T17:30:00Z
feature_name=player_xg_ewma_5
feature_version=3
value=0.41

3.4 Feature Set

A feature set is a versioned collection of feature definitions approved for a model family.

Examples:

minutes_model_features_v4

points_model_features_v11

price_model_features_v7

embedding_input_features_v3

3.5 Point-in-Time Join

A point-in-time join selects only source records whose event time and publication time make them available before the prediction cutoff.

For every source record, the platform should distinguish:

event_time: when the underlying event occurred;

observed_time: when XG Alonso collected it;

available_time: when the information became publicly usable;

processed_time: when the pipeline transformed it.

Feature joins must filter on available_time <= prediction_timestamp.

4. High-Level Architecture

Raw Immutable Data
        |
        v
Normalization and Identity Resolution
        |
        v
Canonical Event Tables
        |
        v
Feature Generator Registry
        |
        +-------------------+
        |                   |
        v                   v
Base Feature Generators   Context Builders
        |                   |
        +---------+---------+
                  |
                  v
Derived Feature Graph
                  |
                  v
Validation and Leakage Gates
                  |
                  v
Offline Feature Store
                  |
                  +----------------------+
                  |                      |
                  v                      v
          Feature Scientist       Online Materialization
                  |                      |
                  v                      v
          Selected Feature Sets   Prediction Services

5. Subsystems

5.1 Source Adapters

Source adapters expose normalized tables to the Feature Factory.

Required adapters:

Official FPL player snapshots

Official FPL fixture data

Official FPL live-gameweek data

Official or extracted price-predictor data

Historical FPL gameweek data

Underlying event and expected-statistics data

Team-level match data

Injury and availability events

Manager and coaching-tenure data

Optional bookmaker probabilities

Optional weather and travel context

User squad and transfer-state data

Every adapter must publish:

schema;

primary key;

event time;

available time;

freshness expectations;

missingness expectations;

known leakage risks;

license or usage constraints.

5.2 Feature Generator Registry

All generators are registered through a common interface.

class FeatureGenerator(Protocol):
    name: str
    version: int
    supported_entities: set[str]
    required_columns: set[str]

    def validate_definition(self, definition: FeatureDefinition) -> None:
        ...

    def generate(
        self,
        frame: DataFrame,
        context: GenerationContext,
        definition: FeatureDefinition,
    ) -> DataFrame:
        ...

    def metadata(
        self,
        definition: FeatureDefinition,
    ) -> FeatureMetadata:
        ...

Generators must be:

deterministic;

stateless during execution;

independently testable;

idempotent;

schema-validated;

safe against future leakage.

5.3 Feature Definition Compiler

The compiler converts declarative feature definitions into an executable dependency graph.

It must:

validate source columns;

validate entity grain;

resolve upstream dependencies;

detect cyclic dependencies;

calculate feature hashes;

plan execution order;

reuse shared intermediate computations;

emit a reproducible execution manifest.

5.4 Context Builders

Context builders create reusable contextual tables.

Examples:

opponent strength as of the prediction timestamp;

current manager tenure;

current set-piece hierarchy;

player role;

team rest schedule;

fixture congestion;

home and away split;

current FPL ownership state;

current user selling price;

upcoming fixture horizon.

5.5 Validation Gates

No feature is materialized until it passes:

schema validation;

point-in-time validation;

null-rate checks;

range checks;

cardinality checks;

duplicate-key checks;

invariant checks;

deterministic rerun checks;

known-leakage rules;

distribution drift checks.

5.6 Feature Store

The MVP may use Parquet plus PostgreSQL metadata. A production version may use a dedicated feature store, but the abstraction should not depend on a vendor.

The system requires:

offline historical storage;

latest-value materialization;

point-in-time retrieval;

feature-set versioning;

training-serving consistency;

feature lineage lookup.

6. Feature Taxonomy

The Feature Factory should generate features across the following families.

6.1 Player Performance

Raw and transformed measures:

minutes;

starts;

goals;

assists;

expected goals;

non-penalty expected goals;

expected assists;

shots;

shots on target;

big chances;

touches in the box;

key passes;

progressive carries;

progressive passes;

bonus;

defensive contributions;

clean sheets;

cards;

own goals;

saves;

goals prevented;

set-piece attempts.

Transformations:

rolling mean;

rolling median;

rolling sum;

rolling maximum;

rolling minimum;

rolling standard deviation;

coefficient of variation;

exponentially weighted mean;

exponentially weighted variance;

trend slope;

change from previous window;

percentile within position;

z-score within position;

per-90 normalization;

per-start normalization;

share of team total;

opponent-adjusted values;

residual versus model expectation.

Windows:

last 1 appearance;

last 3 appearances;

last 5 appearances;

last 8 appearances;

last 10 appearances;

season to date;

previous season;

same-manager tenure;

last 30, 60, and 90 days.

6.2 Minutes and Role

start rate;

substitute rate;

expected minutes;

minutes when starting;

median substitution minute;

probability of 60+ minutes;

probability of 75+ minutes;

consecutive starts;

matches since last start;

team lineup stability;

competition for position;

teammate injury effects;

formation-specific role;

set-piece rank;

penalty-taking probability;

corner-taking probability;

direct free-kick probability;

tactical-position stability;

manager-specific usage;

European-match rotation tendency.

6.3 Team Attack and Defense

team xG;

team non-penalty xG;

team xGA;

goals for;

goals against;

shots for;

shots allowed;

big chances for;

big chances allowed;

possession;

field tilt;

pressing intensity;

set-piece xG;

counterattack rate;

clean-sheet rate;

scoring rate;

home attack strength;

away attack strength;

home defensive strength;

away defensive strength;

strength versus opponent tiers;

strength under current manager;

form adjusted for opponent quality.

6.4 Opponent Matchup

opponent xGA;

opponent shots conceded;

opponent chances conceded by zone;

opponent goals conceded by position;

opponent set-piece weakness;

opponent aerial weakness;

opponent transition defense;

opponent pressing vulnerability;

opponent clean-sheet probability;

historical performance of comparable player archetypes;

player historical output versus opponent;

player underlying output versus opponent;

team historical output versus opponent;

opponent-adjusted player residual.

Historical opponent features must include sample-size metadata and shrinkage.

For a player with only three appearances against an opponent, use:

shrunk_rate =
weight × player_vs_opponent_rate
+ (1 - weight) × player_baseline_rate

where weight increases with relevant sample size.

6.5 Temporal and Calendar Context

day of week;

kickoff hour;

local time;

early kickoff indicator;

late kickoff indicator;

days since prior appearance;

days until next match;

number of matches in last 7, 14, and 21 days;

number of matches in next 7 and 14 days;

international-break return;

holiday-period indicator;

month;

season phase;

gameweek number;

daylight and weather context where available.

Calendar features should be treated as candidate features, not trusted signals. The Feature Scientist must penalize low-sample and unstable effects.

Example:

player_points_on_saturday_early_kickoff

player_xg_weekday_residual

team_performance_monday_matches

These must include:

sample count;

empirical-Bayes shrinkage;

confidence interval;

out-of-sample stability.

6.6 Home and Away Context

player home xG;

player away xG;

player home points;

player away points;

team home attack;

team away attack;

opponent home defense;

opponent away defense;

venue-specific performance;

travel distance;

rest-adjusted home advantage;

manager-specific home advantage.

6.7 Manager Context

manager tenure;

matches under manager;

manager attack strength;

manager defense strength;

average substitution minute;

rotation rate;

lineup consistency;

youth usage;

set-piece production;

favored formations;

player start rate under manager;

player output under manager;

team style embedding;

manager-change indicator;

new-manager bounce candidates;

pre/post-manager-change residuals.

6.8 Fixture Congestion and Fatigue

rest days;

travel distance;

European away travel;

extra-time played;

cup match minutes;

international minutes;

number of starts in prior 14 days;

age-adjusted congestion;

injury-history-adjusted congestion;

manager rotation tendency × congestion;

position × congestion;

travel × kickoff-time interaction.

6.9 FPL Market

current price;

starting price;

price change season to date;

ownership;

effective ownership estimate;

transfers in;

transfers out;

net transfers;

net transfers per hour;

transfer velocity;

transfer acceleration;

ownership-normalized transfer rate;

official predictor progress;

distance to threshold;

progress velocity;

progress acceleration;

hours until price update;

prior rise or fall recency;

market reaction after points haul;

market reaction after injury;

market-versus-model divergence.

6.10 Intrinsic Value

expected points per million;

expected minutes per million;

value over positional replacement;

fair-price residual;

projected six-gameweek points minus price-adjusted baseline;

scarcity-adjusted value;

captaincy-adjusted value;

transfer flexibility contribution;

expected resale value;

ownership-adjusted upside.

6.11 User-Specific Squad Context

purchase price;

current selling price;

locked-in value;

money in bank;

free transfers;

hit cost;

player role in current starting XI;

bench dependency;

club-slot pressure;

captaincy coverage;

replacement affordability;

path-to-target-player cost;

squad structural imbalance;

future transfer flexibility;

wildcard state;

chip state.

6.12 Availability and Risk

injury status;

chance of playing;

suspension status;

yellow-card accumulation;

return-from-injury period;

prior injury burden;

source agreement;

availability confidence;

press-conference recency;

rotation risk;

minutes uncertainty;

model disagreement;

missing-data risk.

6.13 Similarity and Embedding Features

nearest-player similarity;

player-cluster ID;

team-cluster ID;

manager-cluster ID;

fixture-cluster ID;

cosine similarity to opponent-vulnerable archetypes;

similarity to high-performing historical fixtures;

embedding distance to replacement candidates;

cluster-relative form;

cluster-relative price.

7. Generator Types

7.1 Rolling Generator

Supported operators:

mean;

sum;

median;

min;

max;

standard deviation;

quantile;

count;

start rate.

Supported windows:

matches;

starts;

days;

gameweeks;

manager tenure.

7.2 Exponential Generator

Produces recency-weighted features.

Parameters:

alpha;

half-life;

minimum observations;

missing-value behavior.

7.3 Trend Generator

Produces:

linear slope;

robust Theil-Sen slope;

acceleration;

change-point indicator;

recent-minus-long-term average.

7.4 Ratio Generator

Produces ratios with guarded denominators.

Examples:

goals / xG;

points / price;

shots on target / shots;

player xG / team xG;

net transfers / ownership.

Every ratio must define:

denominator floor;

clipping bounds;

null policy.

7.5 Residual Generator

Fits or consumes a baseline expectation and stores the residual.

Examples:

goals minus expected goals;

actual minutes minus expected minutes;

FPL points minus underlying-points expectation;

price progress minus market-model expectation.

7.6 Group-Normalization Generator

Produces:

z-score;

percentile;

rank;

deviation from group median.

Groups may include:

position;

team;

price band;

player cluster;

fixture tier;

gameweek.

7.7 Split Generator

Builds conditional histories:

home only;

away only;

under current manager;

versus top quartile defenses;

versus bottom quartile defenses;

early kickoff;

weekend;

weekday;

European-congestion matches;

starts only.

Every split feature must carry sample count and shrinkage.

7.8 Shrinkage Generator

Used for sparse contextual histories such as:

versus a specific opponent;

performance on a weekday;

output in a particular kickoff window;

performance under a manager;

venue-specific performance.

Recommended empirical-Bayes estimate:

posterior_mean =
(n × observed_mean + prior_strength × prior_mean)
/
(n + prior_strength)

The generator should also emit:

raw mean;

shrunk mean;

sample count;

posterior uncertainty.

7.9 Interaction Generator

Interaction discovery is controlled, not unrestricted.

Supported interactions:

numeric × numeric;

numeric × binary;

category-specific numeric residual;

monotonic transformations;

ratios;

thresholds;

embedding similarity interactions.

Candidate interactions are generated from:

domain-compatible feature families;

high-performing univariate features;

residual error analysis;

tree-model interaction scores;

SHAP interaction values;

repeated cross-validation stability.

Examples:

expected minutes × opponent xGA;

home × rolling xG;

rest days × age;

manager rotation rate × fixture congestion;

price momentum × expected points;

penalty-taker probability × opponent foul rate;

player archetype similarity × opponent weakness.

The generator must block:

direct target-derived interactions;

duplicate algebraic equivalents;

unstable sparse combinations;

high-cardinality entity IDs;

interactions lacking enough historical support.

7.10 Sequence Generator

Creates ordered pattern features:

consecutive starts;

consecutive blanks;

consecutive returns;

trend after return from injury;

streak length;

time since last haul;

time since last goal;

state transitions.

7.11 Embedding Join Generator

Joins learned vectors or derived similarity measures into tabular features.

Embeddings are versioned and must be generated only from information available before the prediction timestamp.

8. Interaction Discovery Workflow

The Feature Factory generates interaction candidates, while the Feature Scientist evaluates them.

Stage 1: Eligibility

Features must pass:

sufficient coverage;

sufficient variance;

acceptable leakage score;

stable generation;

semantic compatibility.

Stage 2: Candidate Generation

Generate interactions among:

top univariate features;

complementary feature families;

model residual drivers;

domain-allowed pairs;

targeted triples when pairwise evidence exists.

Stage 3: Cheap Screening

Use:

mutual information;

univariate gain;

permutation lift;

residual correlation;

small tree models;

sampled cross-validation.

Stage 4: Full Evaluation

Evaluate surviving interactions using walk-forward folds.

Track:

mean validation lift;

worst-fold lift;

stability;

added latency;

missingness;

correlation with existing features;

target leakage risk.

Stage 5: Promotion

An interaction can enter a production feature set only when:

average lift exceeds the minimum threshold;

no major fold degrades beyond tolerance;

importance is stable across retrains;

lineage is complete;

inference cost is acceptable.

9. Feature Metadata

Every feature must have a Feature Card.

Required fields:

feature_id: uuid
name: player_xg_ewma_5
version: 3
display_name: Player xG EWMA, last 5 appearances
description: Recency-weighted expected goals across the player's previous five appearances.
entity: player_fixture
dtype: float
owner: ml-platform
generator: exponentially_weighted_mean
generator_version: 2
source_tables:
  - player_match_stats
source_columns:
  - xg
dependencies: []
lookback:
  type: appearances
  value: 5
parameters:
  alpha: 0.45
available_at: fixture_finalization
null_policy: position_median
valid_range:
  min: 0
  max: 3
tags:
  - performance
  - xg
  - recency
prohibited_targets: []
created_at: 2026-08-01
status: candidate

Evaluation metadata:

coverage: 0.992
missing_rate: 0.008
training_importance_mean: 0.041
shap_rank_mean: 14
permutation_lift: 0.0038
stability_score: 0.87
redundancy_cluster: 22
selected_feature_sets:
  - points_model_features_v11
last_evaluated_at: 2026-11-02

10. Feature Lineage

The platform must answer:

Which raw fields produced this feature?

Which code version generated it?

Which upstream features does it depend on?

Which models use it?

Which recommendations were influenced by it?

When did it enter production?

When was it retired?

Lineage graph:

raw_understat_shots
        |
        v
normalized_player_match_stats.shots
        |
        v
player_shots_rolling_mean_5
        |
        +--------------------+
        |                    |
        v                    v
shots_x_home          shots_x_opponent_xga
        |                    |
        +---------+----------+
                  |
                  v
points_model_features_v11
                  |
                  v
points_model_v18
                  |
                  v
recommendation_run_2026_11_02

11. Point-in-Time Correctness

The most serious implementation risk is leakage.

The system must provide a reusable point-in-time join utility.

def point_in_time_join(
    entities: DataFrame,
    source: DataFrame,
    entity_keys: list[str],
    prediction_time_col: str,
    source_available_time_col: str,
) -> DataFrame:
    ...

Prohibited examples:

using final bonus points to predict the same gameweek;

using a price snapshot collected after midnight to predict that price change;

using an injury update published after the deadline;

using full-season aggregates when reconstructing an earlier gameweek;

using current club identity for a player before a transfer;

using revised historical expected statistics not available at the original time without marking them as retrospective-only.

Every backtest must persist the exact data_cutoff_timestamp.

12. Data Quality Rules

Feature generation should fail loudly when critical assumptions break.

Required checks:

one row per entity key;

monotonic timestamps;

valid player-team membership interval;

no fixture dated before prior fixture ordering;

no negative minutes;

plausible price range;

ownership between 0 and 100;

predictor progress within accepted range or explicitly unbounded;

no impossible future records;

correct promoted-team mapping;

consistent manager tenure interval.

Non-critical issues may quarantine affected rows rather than fail the entire pipeline.

13. Missing Data

Missingness may itself carry information.

For each feature, define one of:

leave null;

zero-fill;

group median;

historical prior;

explicit missing indicator;

model-native missing handling;

prohibit row.

Do not use global mean imputation by default.

Examples:

No xG history for a promoted player: use league- or source-adjusted prior plus a missing-history indicator.

No manager history: use manager-archetype prior or league prior.

No specific-opponent history: use shrunk player baseline.

No price-predictor snapshot: mark unavailable rather than imputing fake progress.

14. Cold-Start Strategy

Cold-start entities include:

promoted players;

transferred players;

new managers;

newly promoted teams;

players returning from long absences;

players with role changes.

Fallback hierarchy:

current-entity history;

history from another league adjusted by competition strength;

player-cluster prior;

position and price-band prior;

league-wide prior.

Embeddings and player clustering should provide priors for sparse entities.

15. Execution Modes

15.1 Historical Backfill

Produces point-in-time features for prior seasons and gameweeks.

Requirements:

deterministic;

resumable;

partitioned by season and gameweek;

capable of rebuilding one feature family;

emits audit manifest.

15.2 Gameweek Materialization

Runs before each gameweek deadline.

Outputs:

model-ready gameweek feature table;

feature-quality report;

latest entity features;

prediction input snapshot.

15.3 Intraday Market Refresh

Runs during active price-change monitoring.

Only recomputes time-sensitive features such as:

ownership;

transfers;

price-predictor progress;

hours until price update;

price velocity;

availability updates.

15.4 Post-Gameweek Finalization

After official results lock:

updates labels;

materializes completed-game features;

recalculates rolling histories;

publishes training partitions;

triggers Feature Scientist evaluation.

16. Performance Requirements

MVP targets:

full current-season gameweek feature build: under 10 minutes;

intraday price refresh: under 60 seconds;

latest feature lookup per player: under 100 ms from cache;

one historical season backfill: under 30 minutes on a developer machine;

deterministic rerun equality: exact for integer and categorical outputs, tolerance-based for floating point.

Prefer Polars or DuckDB for local analytical execution. Avoid premature distributed infrastructure.

17. Suggested Package Structure

feature_factory/
├── __init__.py
├── contracts/
│   ├── definitions.py
│   ├── metadata.py
│   └── execution.py
├── registry/
│   ├── generator_registry.py
│   ├── feature_registry.py
│   └── feature_set_registry.py
├── compiler/
│   ├── parser.py
│   ├── validator.py
│   ├── dependency_graph.py
│   └── planner.py
├── generators/
│   ├── rolling.py
│   ├── exponential.py
│   ├── trend.py
│   ├── ratio.py
│   ├── residual.py
│   ├── normalization.py
│   ├── split.py
│   ├── shrinkage.py
│   ├── interaction.py
│   ├── sequence.py
│   └── embedding_join.py
├── contexts/
│   ├── player_context.py
│   ├── team_context.py
│   ├── opponent_context.py
│   ├── fixture_context.py
│   ├── manager_context.py
│   ├── market_context.py
│   └── user_squad_context.py
├── quality/
│   ├── leakage.py
│   ├── validation.py
│   ├── drift.py
│   └── reports.py
├── materialization/
│   ├── offline.py
│   ├── online.py
│   └── manifests.py
└── cli.py

18. Configuration

Feature definitions should live in version-controlled YAML.

configs/features/
├── base/
├── player_performance/
├── minutes/
├── team/
├── matchup/
├── temporal/
├── manager/
├── congestion/
├── market/
├── valuation/
├── user_context/
├── embeddings/
└── interactions/

Example:

features:
  - name: player_points_rolling_mean_5
    generator: rolling
    entity: player_gameweek
    source_column: total_points
    window:
      unit: appearances
      size: 5
    aggregation: mean
    min_periods: 2

  - name: player_points_saturday_shrunk
    generator: shrinkage_split
    entity: player_gameweek
    source_column: total_points
    split:
      column: day_of_week
      value: saturday
    prior_feature: player_points_rolling_mean_10
    prior_strength: 8

19. Testing Strategy

Unit Tests

Test each generator with small fixtures covering:

normal behavior;

missing values;

insufficient history;

duplicate records;

boundary timestamps;

categorical splits;

numerical stability.

Golden Tests

Persist small canonical input-output datasets. Any feature logic change must explicitly update the golden artifact.

Leakage Tests

Construct future records and verify that they never affect earlier predictions.

Property Tests

Examples:

rolling count never exceeds window;

ownership percentile stays in [0, 1];

shrinking with zero observations equals prior;

increasing sample size moves posterior toward observed mean;

historical rebuild is invariant to records added after the cutoff.

Integration Tests

Test:

raw adapter to feature table;

one complete gameweek build;

training-serving parity;

registry and lineage creation;

feature-set loading.

20. Observability

Every run should emit:

run ID;

feature definitions hash;

source dataset versions;

row count;

execution duration;

feature count;

failed features;

null-rate changes;

distribution-shift warnings;

leakage checks;

materialized partitions.

A feature-quality dashboard should show:

coverage;

drift;

importance;

redundancy;

stability;

production usage.

21. Security and Compliance

Do not store user FPL credentials when team ID access is sufficient.

Respect source terms and rate limits.

Keep source adapters replaceable.

Store provenance and license notes.

Avoid exposing licensed raw data through public APIs.

Separate user-specific features from globally shared training data.

22. MVP Scope

Feature Factory v1 must include:

Player, team, fixture, and market source adapters

Point-in-time joins

Rolling, exponential, ratio, trend, normalization, split, and shrinkage generators

Feature registry and Feature Cards

Offline Parquet materialization

Current-gameweek materialization

Leakage checks

Feature-quality report

YAML feature definitions

Integration with the first points and price models

The MVP should generate approximately 300–700 candidate features, not thousands for the sake of scale.

Quality and correctness are more important than raw feature count.

23. Phase 2

Add:

automated interaction candidates;

residual-guided generation;

embedding joins;

player and team clusters;

manager context;

intraday price refresh;

user-specific squad features;

feature-set promotion workflows.

24. Phase 3

Add:

automated feature proposal from residual analysis;

semantic duplicate detection;

interaction discovery using SHAP;

cross-target feature transfer;

feature retirement;

feature-cost-aware selection;

richer cold-start priors;

multi-league support.

25. Acceptance Criteria

The Feature Factory is ready for production integration when:

a historical gameweek can be rebuilt without future leakage;

the same definitions produce identical features across reruns;

every feature has complete metadata and lineage;

model training loads feature sets only through the registry;

price and points models can use separate approved feature sets;

current-gameweek materialization finishes within the target runtime;

failed or drifting features are visible in run reports;

adding a feature requires configuration and a reusable generator rather than bespoke pipeline code.

26. Claude Code Implementation Order

Claude should implement in this order:

Data and feature contracts

Generator protocol

Registry

Rolling generator

Point-in-time join utility

YAML definition parser

Compiler and dependency graph

Validation framework

Offline materialization

Exponential, ratio, trend, normalization generators

Split and shrinkage generators

Feature Cards and lineage

CLI

Integration tests

Current-gameweek execution

Interaction generator only after the base system is stable

Do not begin automated interaction discovery before point-in-time correctness, metadata, and deterministic materialization are complete.
