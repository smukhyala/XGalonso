Repository Structure and Ownership

Project: XG AlonsoVersion: 1.0Status: Build Specification

1. Recommended Repository Layout

xg-alonso/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── package.json
├── pnpm-workspace.yaml
├── docker-compose.yml
├── Makefile
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
│
├── apps/
│   ├── web/
│   │   ├── app/
│   │   ├── components/
│   │   ├── features/
│   │   ├── lib/
│   │   ├── public/
│   │   ├── tests/
│   │   └── package.json
│   │
│   └── api/
│       ├── src/
│       │   ├── api/
│       │   ├── auth/
│       │   ├── dependencies/
│       │   ├── middleware/
│       │   ├── services/
│       │   └── main.py
│       ├── tests/
│       └── Dockerfile
│
├── packages/
│   ├── data_contracts/
│   ├── domain/
│   ├── feature_factory/
│   ├── feature_scientist/
│   ├── embeddings/
│   ├── prediction/
│   ├── optimization/
│   ├── explanations/
│   ├── evaluation/
│   └── observability/
│
├── pipelines/
│   ├── ingestion/
│   │   ├── fpl/
│   │   ├── price_predictor/
│   │   ├── historical/
│   │   ├── underlying_stats/
│   │   ├── availability/
│   │   └── odds/
│   ├── normalization/
│   ├── identity_resolution/
│   ├── feature_materialization/
│   ├── training/
│   ├── backtesting/
│   └── recommendations/
│
├── configs/
│   ├── environments/
│   ├── sources/
│   ├── features/
│   ├── feature_sets/
│   ├── models/
│   ├── optimization/
│   └── experiments/
│
├── data/
│   ├── README.md
│   ├── samples/
│   ├── schemas/
│   └── fixtures/
│
├── models/
│   ├── registry/
│   ├── minutes/
│   ├── points/
│   ├── price/
│   ├── valuation/
│   └── embeddings/
│
├── infra/
│   ├── docker/
│   ├── database/
│   ├── migrations/
│   ├── github_actions/
│   ├── monitoring/
│   └── deployment/
│
├── scripts/
│   ├── bootstrap_dev.sh
│   ├── ingest_current_data.py
│   ├── backfill_season.py
│   ├── build_features.py
│   ├── train_models.py
│   ├── run_backtest.py
│   └── generate_recommendations.py
│
├── notebooks/
│   ├── exploration/
│   ├── feature_research/
│   ├── model_research/
│   └── archived/
│
├── tests/
│   ├── integration/
│   ├── end_to_end/
│   ├── golden/
│   └── performance/
│
├── docs/
│   ├── vision/
│   │   ├── 00_vision.md
│   │   ├── 01_principles.md
│   │   └── 02_success_metrics.md
│   ├── product/
│   │   ├── 01_product_requirements.md
│   │   ├── 02_user_flows.md
│   │   ├── 03_feature_specifications.md
│   │   └── 04_acceptance_criteria.md
│   ├── architecture/
│   │   ├── 01_repository_structure.md
│   │   ├── 02_system_architecture.md
│   │   ├── 03_service_boundaries.md
│   │   └── 04_deployment_architecture.md
│   ├── data/
│   │   ├── 01_data_sources.md
│   │   ├── 02_data_contracts.md
│   │   ├── 03_identity_resolution.md
│   │   ├── 04_database_schema.md
│   │   └── 05_data_quality.md
│   ├── ml/
│   │   ├── 01_ml_architecture.md
│   │   ├── 02_feature_factory.md
│   │   ├── 03_feature_scientist.md
│   │   ├── 04_interaction_discovery.md
│   │   ├── 05_player_clustering.md
│   │   ├── 06_embeddings.md
│   │   ├── 07_prediction_models.md
│   │   ├── 08_continual_learning.md
│   │   └── 09_evaluation.md
│   ├── optimization/
│   │   ├── 01_squad_optimizer.md
│   │   ├── 02_transfer_planner.md
│   │   ├── 03_wildcard_planner.md
│   │   ├── 04_chip_planner.md
│   │   └── 05_captain_and_bench.md
│   ├── api/
│   │   ├── 01_public_api.md
│   │   ├── 02_internal_contracts.md
│   │   └── 03_authentication.md
│   ├── frontend/
│   │   ├── 01_information_architecture.md
│   │   ├── 02_dashboard.md
│   │   ├── 03_design_system.md
│   │   └── 04_visualizations.md
│   ├── operations/
│   │   ├── 01_observability.md
│   │   ├── 02_model_registry.md
│   │   ├── 03_incident_response.md
│   │   └── 04_security.md
│   └── implementation/
│       ├── 01_build_plan.md
│       ├── 02_mvp_milestones.md
│       ├── 03_testing_strategy.md
│       └── 04_release_checklist.md
│
└── .github/
    ├── workflows/
    ├── ISSUE_TEMPLATE/
    ├── pull_request_template.md
    └── CODEOWNERS

2. Architectural Rule

Use a modular monorepo, not microservices.

The system is complex in domain logic but does not initially require independently deployed services. Keep the API, data pipelines, feature platform, models, and optimizer in one repository with strict package boundaries.

Split deployment units only when there is an operational reason, such as:

intraday ingestion requires an independent schedule;

model inference needs separate scaling;

the web frontend has a separate hosting platform;

long-running training jobs need a worker environment.

3. Top-Level Ownership

apps/web

The user-facing Next.js application.

Owns:

squad dashboard;

player explorer;

recommendation cards;

wildcard planner UI;

feature-importance views;

player similarity views;

model and data freshness indicators.

Must not contain model logic or optimization logic.

apps/api

FastAPI application that exposes product workflows.

Owns:

request validation;

authentication;

orchestration;

API response models;

caching;

user-specific access.

Must call domain packages rather than duplicate their logic.

packages/data_contracts

Shared schemas for:

players;

teams;

fixtures;

gameweeks;

snapshots;

feature rows;

predictions;

recommendations;

model metadata.

This package should have minimal dependencies.

packages/domain

Pure FPL and football domain rules.

Owns:

squad constraints;

price and selling-price rules;

position rules;

club limits;

transfer accounting;

chip states;

gameweek horizons.

No database or API dependencies.

packages/feature_factory

Deterministic candidate-feature generation.

Owns:

generators;

definitions;

registry;

lineage;

point-in-time joins;

materialization;

feature-quality validation.

packages/feature_scientist

Automated feature evaluation and promotion.

Owns:

univariate screening;

redundancy analysis;

stability scoring;

interaction evaluation;

feature-set selection;

feature retirement;

Feature Scientist reports.

packages/embeddings

Representation learning.

Owns:

player embeddings;

team embeddings;

manager embeddings;

fixture embeddings;

clustering;

similarity search;

embedding versioning.

packages/prediction

Model training and inference.

Owns:

expected minutes;

expected points;

price movement;

fair value;

model calibration;

prediction contracts.

packages/optimization

Decision layer.

Owns:

starting XI optimization;

transfer packages;

multi-gameweek planning;

wildcard timing;

chip timing;

captain and bench decisions.

packages/explanations

Converts structured evidence into user-facing reasoning.

Owns:

deterministic reason codes;

feature-contribution summaries;

recommendation comparisons;

natural-language rendering.

LLMs may rewrite explanations, but they must not invent underlying reasons.

packages/evaluation

Research and production evaluation.

Owns:

walk-forward validation;

model metrics;

calibration;

recommendation backtests;

policy regret;

baseline comparisons;

experiment reports.

packages/observability

Shared logging and monitoring utilities.

Owns:

run IDs;

structured logs;

metrics;

lineage IDs;

tracing;

freshness checks.

4. Data Directory Rule

Do not commit full raw datasets.

The data/ directory should contain only:

tiny samples;

schemas;

test fixtures;

documentation;

local development placeholders.

Use object storage or an ignored local directory for actual data.

Recommended ignored paths:

.data/raw/
.data/silver/
.data/gold/
.artifacts/models/
.artifacts/features/

5. Configuration Rule

Behavior that changes between feature sets, models, experiments, or environments should live in configuration.

Use typed YAML or TOML for:

source settings;

feature definitions;

feature sets;

model hyperparameters;

optimization weights;

backtest windows;

environment endpoints.

Do not hide important product behavior in magic constants.

6. Notebook Rule

Notebooks are for exploration, not production.

Any logic that survives experimentation must move into a package with tests.

A notebook may:

inspect data;

visualize features;

test hypotheses;

compare prototypes.

A notebook may not be the only implementation of:

a source adapter;

a feature;

a model;

a backtest;

an optimizer.

7. Dependency Direction

Allowed dependency direction:

apps
  ↓
explanations / optimization / prediction
  ↓
feature_scientist / embeddings / feature_factory
  ↓
domain / data_contracts

Pipelines may orchestrate all packages.

Forbidden:

domain importing API or database code;

feature_factory importing frontend code;

prediction containing FPL transfer rules;

optimization rebuilding model features;

frontend computing recommendation logic.

8. Suggested Python Namespaces

Use one installable Python workspace.

xg_alonso.contracts
xg_alonso.domain
xg_alonso.features
xg_alonso.feature_scientist
xg_alonso.embeddings
xg_alonso.prediction
xg_alonso.optimization
xg_alonso.explanations
xg_alonso.evaluation
xg_alonso.observability

The physical monorepo can still preserve package-level ownership.

9. Suggested Frontend Feature Structure

apps/web/features/
├── squad/
├── players/
├── transfers/
├── wildcard/
├── captain/
├── market/
├── feature_lab/
├── similarity/
└── settings/

Each feature folder should contain:

components;

hooks;

types;

API bindings;

tests.

10. Recommended Initial Build Order

Do not create every folder on day one.

Start with:

xg-alonso/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── apps/
│   └── api/
├── packages/
│   ├── data_contracts/
│   ├── domain/
│   ├── feature_factory/
│   ├── prediction/
│   ├── optimization/
│   └── evaluation/
├── pipelines/
│   ├── ingestion/
│   ├── normalization/
│   └── feature_materialization/
├── configs/
├── tests/
└── docs/

Add the web app after the first end-to-end recommendation can be generated from a CLI or API.

Add embeddings and the automated Feature Scientist after the base data and feature pipeline is reliable.

11. MVP Milestone Mapping

Milestone 1: Data Foundation

Create:

contracts;

source adapters;

raw snapshots;

normalized tables;

identity resolution.

Milestone 2: Feature Platform

Create:

generator registry;

point-in-time joins;

rolling features;

Feature Cards;

offline feature store.

Milestone 3: Baseline Models

Create:

minutes model;

points model;

price model;

walk-forward evaluation.

Milestone 4: Decision Engine

Create:

squad import;

one-transfer optimizer;

multi-transfer packages;

no-action baseline.

Milestone 5: Wildcard Planner

Create:

best wildcard squad;

wildcard-now versus wildcard-later comparison;

chip opportunity-cost logic.

Milestone 6: Product UI

Create:

squad dashboard;

recommendations;

player comparison;

market scanner;

explanations.

Milestone 7: Feature Scientist

Create:

candidate evaluation;

redundancy removal;

stable feature-set selection;

interaction discovery;

feature reports.

Milestone 8: Embeddings and Clustering

Create:

player vectors;

player clusters;

similarity search;

cold-start priors.

12. Branching and Pull Requests

Recommended branch naming:

feat/feature-factory-registry
feat/fpl-bootstrap-adapter
fix/point-in-time-leakage
docs/feature-scientist
experiment/player-clustering-v1

Pull requests should include:

problem;

design;

affected contracts;

tests;

data or model impact;

migration notes;

screenshots where relevant.

Any change affecting features or models must mention reproducibility impact.

13. Definition of Done

A subsystem is not complete until it has:

typed interfaces;

unit tests;

integration coverage;

documentation;

observability;

configuration;

failure behavior;

sample usage;

ownership.

For ML components, also require:

evaluation baseline;

feature or model version;

data cutoff;

saved metrics;

reproducible command.

14. Claude Code Operating Rule

Claude should treat documentation as the source of truth.

Before implementing a task:

read CLAUDE.md;

read the relevant design document;

inspect existing contracts;

identify affected tests;

propose a narrow implementation plan;

implement the smallest complete vertical slice;

run formatting, type checks, and tests;

update documentation when contracts change.

Claude should not scaffold unused services or speculative abstractions merely because they appear in the long-term repository tree.
