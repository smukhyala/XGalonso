# Objective-Conditioned Automated Feature Discovery — Implementation Plan

**Status:** design, grounded in the repository at `87e9c60`.
**Author:** this change.
**Scope:** extend XG Alonso so that *which* representation of the player pool is built
depends on *what the manager is trying to achieve*, rather than on a single global
accuracy metric.

Everything below names files and interfaces that exist today. Where a capability does
not exist, it is marked **NEW**. Where a design document specifies something that was
never built, that is stated explicitly rather than assumed.

---

## 1. Current architecture

### 1.1 Shape

A `uv` workspace of eight libraries, two pipelines and two apps, all sharing the PEP 420
namespace package `xg_alonso`. Python 3.12, `mypy --strict`, `ruff`, and — most
importantly for this change — **`import-linter` enforces the dependency direction**
(`.importlinter`):

```
xg_alonso.api
xg_alonso.cli : xg_alonso.pipelines
xg_alonso.evaluation
xg_alonso.optimization : xg_alonso.explanations
xg_alonso.prediction
xg_alonso.features
xg_alonso.domain : xg_alonso.storage
xg_alonso.contracts
```

Plus three `forbidden` contracts that matter here:

- `duckdb` is importable only from `xg_alonso.storage`
- `xg_alonso.domain` is pure — no `duckdb`, `polars`, `httpx`, `pyarrow`, `sqlite3`
- `xg_alonso.contracts` depends on no other internal package (`polars` **is** allowed)
- `httpx` is importable only from ingestion

Any new code must land inside this lattice, not beside it.

### 1.2 What each layer actually does

| Package | Real responsibility | Key modules |
|---|---|---|
| `contracts` | Frozen Pydantic vocabulary. Four contracts are explicitly frozen: prediction, reason codes, folds, storage protocols. | `prediction.py`, `folds.py`, `storage.py`, `evidence.py`, `provenance.py`, `recommendation.py`, `squad.py`, `identifiers.py`, `reason_codes.py`, `form.py` |
| `domain` | Pure FPL rules, loaded from a **pinned bootstrap snapshot**, never typed as literals. | `scoring.py`, `rules.py`, `constraints.py`, `pricing.py`, `purchase_prices.py`, `drift.py` |
| `storage` | The only package permitted to import `duckdb`. | `duckdb_store.py`, `bronze.py` |
| `features` | Point-in-time-safe candidate generation. | `catalogue.py`, `generators.py`, `point_in_time.py`, `opponent.py`, `career.py`, `recency.py`, `archetypes.py`, `assemble.py`, `leakage.py`, `slice1.py` |
| `prediction` | Component models + inference. | `trained.py`, `inference.py`, `dataset.py`, `baseline.py`, `evidence.py`, `calibration.py`, `form.py`, `refresh.py` |
| `optimization` | The decision layer. | `transfer.py`, `squad_builder.py`, `lineup.py`, `horizon.py` |
| `explanations` | Structured evidence → reason codes → prose. | `reasons.py`, `player.py`, `derivation.py`, `history.py`, `lineup_diff.py`, `render.py` |
| `evaluation` | Walk-forward *policy* backtesting and importance. | `backtest.py`, `policies.py`, `accuracy.py`, `importance.py`, `report.py` |
| `cli` | Composition root. `xg` binary, 12 commands. | `main.py`, `pipeline.py` |
| `api` | FastAPI over the same service. 7 routes. | `main.py`, `service.py` |

### 1.3 Prediction target

**Components, not points** (decision D8). `prediction/dataset.py::COMPONENT_LABELS` is:

```
minutes, starts, goals_scored, assists, clean_sheets,
goals_conceded, saves, bonus, yellow_cards
```

Nine `HistGradientBoosting` models (`prediction/trained.py`) predict these; the domain
layer prices them through versioned scoring rules (`domain/scoring.py::assemble_points`).
Counts use Poisson loss, `label_clean_sheets` / `label_starts` are classifiers.

This is load-bearing for the new work: **objective-conditioned evaluation must score the
assembled `expected_points` and the resulting decision, not a single component.**

### 1.4 Data actually available

`.data/silver/player_gameweek_stats.parquet` — **113,270 rows, 37 columns, four full
seasons** (2022-23 … 2025-26, GW1-38 each, 778-865 players per season).

The complete raw-field vocabulary a feature DSL can bind to:

```
player_code, gameweek_id, season, fixture_id, opponent_team_id, was_home,
minutes, starts, goals_scored, assists, clean_sheets, goals_conceded, saves,
yellow_cards, red_cards, own_goals, penalties_saved, penalties_missed,
bonus, bps, total_points,
expected_goals, expected_assists, expected_goal_involvements, expected_goals_conceded,
defensive_contribution, value, selected, transfers_in, transfers_out, transfers_balance,
influence, creativity, threat, ict_index,
kickoff_time, available_time
```

`players_history.parquet` — 3,268 rows: `player_code, season, web_name, position,
team_name, opening_price`.

**The only null column is `defensive_contribution`** — 73.7% null, present for 2025-26
only, exactly as `CLAUDE.md` warns.

#### 1.4.1 Fields the brief asks for that DO NOT EXIST

Phase 7 of the brief lists an embedding vocabulary. Against decision D6 (official FPL API
only, zero budget) the following are **not obtainable** and must not be invented:

| Requested | Status |
|---|---|
| expected goals, expected assists, expected threat | ✅ `expected_goals`, `expected_assists`, `threat` |
| minutes, starts, price, ownership, transfers in/out, FPL points | ✅ `minutes`, `starts`, `value`, `selected`, `transfers_in/out`, `total_points` |
| team attacking / defensive strength, opponent strength | ✅ derived by `features/opponent.py::build_opponent_strength` (table inversion) |
| home/away, rest | ✅ `was_home`; rest derived from `kickoff_time` by `features/recency.py` |
| shots, shots in the box, big chances, touches in the box | ❌ **not in the FPL API** |
| key passes, passes into the penalty area | ❌ **not in the FPL API** |
| set-piece share, penalty share | ❌ not published; only `penalties_missed` / `penalties_saved` outcomes exist |
| opponent pressing intensity | ❌ **not in the FPL API** |
| expected minutes, uncertainty measures | ✅ but as *model outputs* (`MinutesPrediction`), not raw fields |

Consequence: the brief's worked example — `touch_dependency_rolling_5 ×
opponent_pressing_intensity` — **is not computable in this repository.** The hypothesis
library will use the nearest *computable* mechanisms (`threat`, `creativity`, `bps`,
`ict_index` as Opta-derived composites; opponent conceded-xG as defensive weakness), and
the plan records the substitution rather than shipping a feature that silently reads
zeros.

### 1.5 Backtesting methodology (already correct)

Two independent, complementary harnesses already exist and both are strictly time-ordered:

1. **`contracts/folds.py::walk_forward_folds`** — the *only* sanctioned fold constructor.
   It is structurally incapable of shuffling: it validates ascending, unique gameweeks
   and raises on an unsorted sequence. Supports expanding/sliding windows and an
   **embargo**. Used by `prediction/trained.py::train_component_models`.
2. **`evaluation/backtest.py::walk_forward`** — a *policy* backtest. Two squads start
   identical, one acts on recommendations, one holds; both are scored on the same actual
   outcomes. `GameweekOutcome.decision_delta` deliberately isolates a single week's
   decision from squad drift.

`evaluation/policies.py` supplies five controls — `model`, `highest_form`,
`most_expensive`, `random`, `hold` — all choosing from an *identical* legal candidate
set, so any difference is selection skill alone.

**This is the decision-quality machinery the objective layer must reuse, not replace.**

### 1.6 Leakage doctrine (already correct, and reusable)

`features/leakage.py` is the highest-value asset for this change:

- `find_leakage(builder, entities, source, future_records)` rebuilds features with
  future-stamped records appended and returns any column whose value moved.
- `assert_no_leakage` raises `LeakageDetected` naming every offending column.
- **`assert_detects_leakage`** is a negative control: it runs a deliberately leaky builder
  and fails if the harness does *not* catch it. A harness that never fails is
  indistinguishable from a broken one.
- `make_future_records` perturbs numeric columns while holding entity keys fixed.

Every discovered feature program will be pushed through this harness before it is allowed
into a registry. That is the single most important integration point in this plan.

Underneath it, `features/generators.py::stage_window` recomputes each entity row's window
**from that row's own vantage point**, ordered by `available_time` (not gameweek), and
`features/point_in_time.py` enforces tz-aware timestamps and deterministic tie-breaking.

`shrunk_rate_as_of` additionally **refuses** to compute a pooled empirical prior when
entities span more than one cutoff, because that is a real cross-row leak invisible to a
single-cutoff harness. New group-level primitives must respect the same rule.

### 1.7 Existing clustering / embedding infrastructure

`features/archetypes.py` (`ARCHETYPE_VERSION = "archetype_v1"`) already implements:

- hand-declared `STYLE_AXES` per position (`GKP`/`DEF`/`MID`/`FWD`), each axis naming
  source columns with fallbacks
- z-standardisation, then a **hand-written Lloyd's k-means with k-means++ seeding**
  (`_kmeans`, seed `20260728`) — written out deliberately so `feature_factory` keeps one
  numeric dependency chain and labels stay stable between runs
- `k = min(4, n // 6)`, minimum population 24, minimum relative spread 0.05 measured
  **before** scaling
- automatic label generation from the axes a centroid is extreme on, with collision
  widening
- nearest-neighbour comparables in the same space

**What it does not do**, and this change must add:

| Missing | Needed for |
|---|---|
| a gameweek dimension — one static clustering, no time axis | Phase 8 dynamic clusters |
| soft membership probabilities — assignment is hard `argmin` | Phase 10 gated features |
| any objective conditioning | Phase 9, the central innovation |
| persistence — the model is rebuilt in memory each run | Phase 13 registries |
| fit/apply separation — no way to fit on a training fold and apply to validation | Phase 11 leak-free clustering |
| cluster-transition history | Phase 8 |

Note also: `packages/feature_factory/pyproject.toml` deliberately does **not** depend on
scikit-learn. `sklearn` appears at exactly one import site in the whole repo
(`prediction/trained.py:28`). New clustering that wants GMM/PCA must therefore either
stay hand-written or live in a package that declares the dependency.

### 1.8 Persistence

DuckDB behind `contracts/storage.py::TableStore` (`write_table`, `append_table`,
`read_table`, `query`, `table_exists`, `execute`), implemented once in
`storage/duckdb_store.py`. Bronze snapshots are content-hashed and append-only via
`BronzeSnapshotStore`.

Artifacts on disk today: `.data/bronze/`, `.data/silver/*.parquet`, `.data/gold/*.parquet`,
`.data/models/*.pkl`, `.data/pinned/rules_2026-27.json`, `.data/reports/`, `.data/signals/`.

Feature importance already persists to Parquet with a schema version
(`evaluation/importance.py::write_importance`, `IMPORTANCE_SCHEMA_VERSION`).

**Decision: registries go through the `TableStore` protocol.** Introducing SQLite would
contradict D2 and add a second persistence story for no benefit.

**Changed during implementation:** the concrete store is Parquet, not DuckDB.
`.importlinter` forbids `xg_alonso.cli` and `xg_alonso.api` from reaching `duckdb`
transitively, so the composition root cannot construct a `DuckDBTableStore` — a
constraint the plan missed and the boundary check caught. D2 names DuckDB *and*
Parquet; `storage/parquet_store.py` is the Parquet half, driver-free and driven
through the same protocol. DuckDB remains available to anything wanting SQL.

### 1.9 Recommendation and optimization logic

`optimization/transfer.py` exhaustively scores every legal single transfer
(`rank_single_transfers`), enforcing position match, affordability against selling price
plus bank, the three-per-club limit and no re-buying. It scores by **re-picking the
starting XI after the swap** (`starting_xi_points`) rather than differencing two players,
with `_RISK_WEIGHT = 0.25` on prediction uncertainty and `_MIN_NET_GAIN = 0.15`.

`horizon_valued` re-prices predictions over a multi-week horizon by *substitution* — one
choke point every scoring path already reads — and inflates `expected_points_sd` with
horizon length.

`TransferBoard` / `PlayerBestMove` already carry the alternatives and a grounded reason
for every squad member with no move.

**The objective layer plugs in here**, at the scoring function and the candidate filter —
not by forking the optimizer.

### 1.10 Frontend

Next.js 16 (app router), React 19, **Tailwind v4 with `@theme` tokens**, TypeScript 5.7.
No component library — no shadcn, no Radix. Design system is `apps/web/app/globals.css`:
a deliberate "floodlit night" palette where colour is a *positional system*
(`--color-gkp/def/mid/fwd`), plus `--color-pitch/surface/raised/line/chalk/muted/dim`,
`--color-gain/loss`, three font families and `.tnum`, `.eyebrow`, `.hairline`, `.rise`
utilities.

Pages: `app/page.tsx`, `app/features/page.tsx`. Components: `Alternatives`, `Depth`,
`History`, `LineupDiff`, `Pitch`, `PlayerLedger`, `SquadBuild`, `TheCall`.
`lib/api.ts` is a hand-written typed client; `next.config.mjs` proxies `/api/*` to
`XG_API_ORIGIN` (default `http://127.0.0.1:8000`).

**New UI must extend this system, not introduce a component library.**

### 1.11 Code hygiene

Searched `packages apps pipelines tests` for `TODO|FIXME|XXX|HACK|NotImplemented`:
**zero matches.** The 29 hits for "placeholder"/"mock" are all legitimate — template
vocabulary in `reason_codes.py` (including a real `template_placeholders()` function) and
`respx` HTTP mocking in `tests/ingestion/test_ingest.py`.

Every RNG is explicitly seeded with `20260727` or `20260728`.

This codebase is **finished, not scaffolded**. There is no half-built feature-discovery
subsystem to complete — which means this change is genuinely additive, and the main risk
is disturbing what works rather than filling a gap.

### 1.12 What is documented but NOT implemented

`docs/ml/` contains design documents for subsystems that do not exist in code:

- `03_feature_scientist.md` — the accept/reject loop. **Spec only.** `features/__init__.py`
  says so outright: *"The Feature Factory generates candidates; it does not decide which
  are useful. That judgement belongs to the Feature Scientist, which is deferred per D10."*
- `04_interaction_discovery.md` — **spec only.** `catalogue.py` states: *"Nothing is
  crossed with anything: interactions are a separate, gated concern."*
- `05_player_clustering.md`, `06_embeddings.md` — **partially** realised by
  `archetypes.py` (static, hard-assignment, position-scoped); no embeddings module, no
  temporal or objective conditioning.

This change implements the Feature Scientist and interaction discovery, conditioned on an
objective — which is what makes it new rather than a catch-up.

### 1.13 Scheduled jobs

None. `.github/workflows/ci.yml` only. No cron, no queue, no worker. Phase 16's
long-running experiment states must therefore be **synchronous with a clean interface**,
not a job-queue integration.

---

## 2. Proposed architecture

### 2.1 One new package, plus targeted extensions

The temptation is a parallel subsystem. That would duplicate folds, leakage checking and
backtesting — exactly the "four incompatible designs" failure the contracts layer exists
to prevent. Instead:

**NEW package `packages/discovery` → `xg-alonso-discovery`, module `xg_alonso.discovery`,
inserted between `evaluation` and `cli`:**

```
xg_alonso.api
xg_alonso.cli : xg_alonso.pipelines
xg_alonso.discovery              <-- NEW
xg_alonso.evaluation
xg_alonso.optimization : xg_alonso.explanations
xg_alonso.prediction
xg_alonso.features
xg_alonso.domain : xg_alonso.storage
xg_alonso.contracts
```

**Everything else extends a package that already owns the concern:**

| Concern | Lands in | Why there |
|---|---|---|
| Objective / constraint / belief / hypothesis / evaluation schemas | `contracts` (NEW modules) | contracts *is* the shared vocabulary; every other package must speak it |
| Objective compiler, constraint validation | `domain` (NEW module) | pure text→struct + rule checking, no I/O — exactly domain's remit |
| Belief-adjusted predictions | `prediction` (NEW module) | it owns prediction shaping; cf. existing `prediction/form.py`, which already adjusts predictions from outside signals |
| Player embeddings, dynamic + objective-conditioned clusters | `discovery` (NEW modules) | **changed during implementation.** Planned for `features`, but objective-conditioned clustering needs PCA and silhouette from scikit-learn, and `feature_factory` deliberately keeps a numpy-only dependency chain — its hand-written k-means exists to preserve that. Pushing sklearn down a layer to avoid moving two modules would have made it load-bearing for all feature generation. |
| Objective-aware squad scoring | `optimization` (NEW module) | plugs into `transfer.py`'s scoring seam |
| DSL, utility scoring, acceptance, registries, search, experiment runner | `discovery` (NEW package) | needs every layer below; nothing below may depend on it |

### 2.2 Keeping the engine reusable outside FPL

The brief requires the generic engine be reusable outside FPL. Enforced mechanically, the
way this repo enforces everything else — a new `import-linter` `forbidden` contract:

```ini
[importlinter:contract:discovery-core-is-generic]
name = the discovery core knows nothing about football
type = forbidden
source_modules =
    xg_alonso.discovery.dsl
    xg_alonso.discovery.utility
    xg_alonso.discovery.acceptance
    xg_alonso.discovery.search
    xg_alonso.discovery.registry
forbidden_modules =
    xg_alonso.domain
    xg_alonso.features
    xg_alonso.prediction
    xg_alonso.optimization
    xg_alonso.evaluation
```

The core may import `xg_alonso.contracts` (for `TableStore` and folds) plus `polars` /
`numpy`. The FPL bindings live in `xg_alonso.discovery.fpl.*` and
`xg_alonso.discovery.experiment`, which may import anything.

This is a claim the build checks, not a comment.

### 2.3 Module inventory

```
packages/data_contracts/src/xg_alonso/contracts/
  objective.py     NEW  ManagerObjective, ManagerConstraints, UserBelief,
                        RiskPreference, OwnershipPreference, PrimaryMetric,
                        ObjectiveBundle, CompiledIntent
  discovery.py     NEW  FeatureHypothesis, DiscoveredFeatureSpec, FeatureEvaluation,
                        FoldMetrics, ComplementarityClass, AcceptanceStatus,
                        HypothesisStatus, ExperimentManifest, ExperimentStatus,
                        PlayerClusterAssignment, ClusterSummary

packages/domain/src/xg_alonso/domain/
  objectives.py    NEW  OBJECTIVE_PRESETS (6), preset lookup, weight resolution
  intent.py        NEW  deterministic natural-language compiler + confidence scoring
  belief.py        NEW  belief decay + multiplier maths (pure)

packages/prediction/src/xg_alonso/prediction/
  beliefs.py       NEW  apply_beliefs(): returns BOTH raw and adjusted predictions


packages/optimization/src/xg_alonso/optimization/
  objective.py     NEW  ObjectiveScorer: turns ManagerObjective + predictions into the
                        scalar the transfer search maximises; constraint filtering

packages/discovery/src/xg_alonso/discovery/       NEW PACKAGE
  embeddings.py    fit/apply-split player representations (moved here from features)
  clusters.py      ClusterModel; soft membership; rolling; objective-conditioned
  harness.py       walk-forward incremental evaluation, controls, subgroup metrics
  dsl.py           feature program AST, primitives, serialization, static validation
  compile.py       AST -> polars expression plan, executed against staged windows
  utility.py       FeatureUtility, metric protocols, objective-specific metrics
  acceptance.py    configurable acceptance policy, classification
  registry.py      DuckDB-backed registries + immutable evaluation history
  search.py        greedy forward / bounded beam complementary search, bundles
  memory.py        hypothesis memory, lesson extraction, exploration policy
  residuals.py     FPL: residual analysis by segment -> evidence for generation
  hypotheses.py    FPL: seeded hypothesis library (15) + evidence-driven generator
  experiment.py    FPL: the orchestrator; emits ExperimentManifest
  llm.py           optional LLM adapter protocol — NO client dependency
```

### 2.4 The feature program DSL

**No `eval`, no `exec`, no LLM-generated Python.** A feature program is a frozen Pydantic
tree, serialisable to JSON, compiled to a Polars expression plan:

```
Program := Source(column)
         | Rolling(child, window, agg, min_periods)     # mean|median|std|min|max|sum|percentile
         | ShrunkRate(num, den, window, prior_strength)
         | Lag(child, periods)
         | EwmMean(child, halflife)
         | Trend(child, window)
         | TimeSince(event_column)
         | Arith(op, left, right)                        # add|sub|mul|safe_div|min|max
         | Unary(op, child)                              # log1p|clip|zscore|percentile_rank
         | GroupRel(child, by, op)                       # rank|share|dev_from_mean
         | ClusterRel(child, cluster_ref, op)            # rank|dev_from_centroid
         | Interact(left, right)                         # sugar over Arith(mul)
         | Const(value)
```

Static validation, **before** any computation, rejects:

- unknown source columns (checked against the actual silver schema)
- any column on a configurable target denylist (`total_points` as a *direct* target-period
  read, label columns)
- windows ≤ 0 or above a configured maximum
- expression depth above a configured maximum
- raw division (only `safe_div` with an explicit epsilon exists — division is
  *unrepresentable*, not merely discouraged)
- programs whose canonical hash collides with an existing registered feature
  (near-duplicate semantics)
- estimated node count above a compute budget

Then, **before acceptance**, dynamic validation via the existing
`features/leakage.py::assert_no_leakage`, paired with `assert_detects_leakage` on a
knowingly-leaky variant so the harness is proven live for every program family.

Every program carries a **content hash** of its canonical JSON. That hash is the version:
identical semantics cannot register twice, and a changed program is a new version rather
than a silent mutation.

### 2.5 Objective-conditioned scoring

```
FeatureUtility =
    w_prediction     * predictive_gain
  + w_decision       * decision_quality_gain
  + w_objective      * objective_specific_gain
  + w_stability      * temporal_stability
  + w_complementarity* conditional_incremental_gain
  - w_complexity     * feature_complexity
  - w_missingness    * missing_data_penalty
  - w_turnover       * feature_turnover
  - w_leakage        * leakage_risk
```

Metrics are chosen from what the repo can actually compute:

*Predictive* — MAE, RMSE, bias, Spearman, top-k precision and lift
(`evaluation/accuracy.py` already implements all of these, including the crucial
degeneracy guard where `skill` is forced to zero for a constant predictor).

*Decision* — realised points of the recommended squad, realised captain points,
per-decision win rate, mean decision delta, mean regret and calibration error
(`evaluation/backtest.py::BacktestResult` already implements all of these).

*Objective-specific* — supplied by the objective itself through a protocol, so a new
objective never edits the search engine:

```python
class ObjectiveMetric(Protocol):
    name: str

    def score(self, ctx: EvaluationContext) -> float: ...
```

Registered per `primary_metric`: `expected_points`, `expected_rank_gain`,
`downside_protection`, `team_value_growth`, `captaincy_upside`, `differential_yield`,
`transfer_flexibility`.

### 2.6 Objective-conditioned clusters

Both required approaches, and they are compared against three controls (static position
clusters, ordinary unsupervised clusters, no clusters):

**A — objective-weighted distance.** Reweight embedding dimensions by objective
relevance before clustering. Deterministic, cheap, interpretable, works at this data
volume.

**B — objective-specific supervised projection.** Fit a low-rank linear projection
(ridge-regularised, closed-form) mapping the base embedding onto the objective's own
target, then cluster in that space. **Fitted only on training-fold rows.** Guarded by a
minimum-rows check that refuses rather than degrades, because at 113k player-gameweeks
split across walk-forward folds, a projection fitted on a thin fold is noise with a matrix
behind it.

Cluster count is chosen from a *combination* — silhouette, membership stability across
adjacent gameweeks, minimum cluster size, and downstream predictive utility — never one
unsupervised metric alone.

### 2.7 Belief handling

`UserBelief` never overwrites a prediction. `prediction/beliefs.py::apply_beliefs` returns
a `BeliefAdjustment` carrying **both** the raw and the adjusted `PlayerPrediction`, plus
the multiplier and its decay. The recommendation layer runs the optimizer twice — with
and without adjustment — and the UI shows both. Confidence sensitivity is reported by
sweeping the confidence and recording where the recommendation flips.

This mirrors the existing `prediction/form.py::apply_form_signals`, which already applies
clamped outside signals evaluated against the deadline rather than the wall clock.

---

## 3. Reusable existing components

| Existing | Reused for | Notes |
|---|---|---|
| `contracts/folds.py::walk_forward_folds` | every temporal split | the only sanctioned constructor; cannot shuffle |
| `features/leakage.py` (all four functions) | validating every discovered program | including the negative control |
| `features/generators.py::stage_window` | the DSL's temporal primitives | already recomputes windows per entity vantage point, ordered by `available_time` |
| `features/point_in_time.py` | tz-aware, deterministic joins | |
| `features/archetypes.py::_kmeans` | clustering baseline | seeded Lloyd's + k-means++, already stable across runs |
| `prediction/trained.py::train_component_models` | the model family under test | plus `usable_features` for all-null column safety |
| `evaluation/accuracy.py::score_predictions` | predictive metrics | degeneracy guards already correct |
| `evaluation/backtest.py::walk_forward` + `policies.py` | decision metrics | five controls already share one legal candidate set |
| `evaluation/importance.py::permutation_importance` | importance + residual evidence | already out-of-sample, already reports stability and family totals |
| `contracts/storage.py::TableStore` | all registries | keeps D2 reversible |
| `optimization/transfer.py::rank_single_transfers` | constrained recommendation | objective plugs into the scoring seam |
| `contracts/provenance.py::PredictionProvenance`, `RunManifest` | experiment manifests | `git_dirty` / `promotable` already modelled |
| `apps/web/app/globals.css` | all new UI | floodlit-night tokens |

---

## 4. Schema and storage changes

**No migration of existing tables. Every change is additive.** New DuckDB tables created
through `TableStore.execute`, all versioned:

| Table | Grain | Purpose |
|---|---|---|
| `discovery_objectives` | objective id + version | ManagerObjective definitions |
| `discovery_hypotheses` | hypothesis id | FeatureHypothesis, incl. `parent_hypothesis_id`, `generation_source` |
| `discovery_features` | feature id + version (content hash) | DiscoveredFeatureSpec, serialized program |
| `discovery_feature_dependencies` | feature id → raw column | lineage |
| `discovery_evaluations` | (feature id, version, objective id, evaluated_at) | **append-only**; immutable history |
| `discovery_fold_metrics` | evaluation id + fold index | per-fold detail retained |
| `discovery_cluster_models` | cluster model version | config, chosen k, selection evidence |
| `discovery_cluster_assignments` | (player, gameweek, model version) | soft membership, distance, objective id or null |
| `discovery_experiments` | experiment id | manifest, status, seeds, data cutoff, code version |
| `discovery_lessons` | lesson id | compact hypothesis-family memory |

`discovery_evaluations` is written with `append_table` only — never `write_table` — so an
evaluation history cannot be rewritten.

Contracts additions are new modules; **no existing contract is modified**, which keeps
`tests/contracts/test_frozen_contracts.py` green by construction.

---

## 5. Migration risks

| Risk | Severity | Mitigation |
|---|---|---|
| Breaking the frozen contracts | high | add new modules only; never edit `prediction.py`, `folds.py`, `storage.py`, `reason_codes.py`. The frozen-contract test is the check. |
| Import-linter layer violation | high | new package declared in `.importlinter` *before* code is written; `make boundaries` in CI |
| A discovered feature leaks | **critical** | every program passes `assert_no_leakage`; a leaky twin passes `assert_detects_leakage`; registry refuses an unvalidated program |
| Clusters/scalers fitted on validation rows | **critical** | fit/apply split is structural — `ClusterModel.fit()` returns an object whose `.apply()` is the only way to score new rows; a leakage test asserts fold isolation |
| Optimising on the final test period | high | the last season is held out and never used for acceptance; acceptance runs on earlier folds only |
| Duplicated backtest machinery | medium | reuse `walk_forward_folds` and `evaluation/*`; the generic engine takes folds as input rather than constructing its own |
| `defensive_contribution` 73.7% null poisoning discovered features | medium | missingness is a first-class penalty in `FeatureUtility` and an acceptance gate; `usable_features` already drops all-null columns per fold |
| Runtime blow-up from combinatorial search | medium | bounded beam width + per-experiment compute budget + node-count cap in static validation; no unconstrained search |
| Claiming significance that was not tested | medium | report fold counts and spreads; language in reports says "improved in k of n folds", never "significant" unless a test is actually run |
| Adding sklearn to `feature_factory` | low | clustering that needs GMM/PCA lives in `discovery` (which declares sklearn) or stays hand-written; `feature_factory`'s dependency set is unchanged |
| Web app has no component library | low | extend `globals.css` tokens; no new npm dependency |

---

## 6. Staged implementation sequence

Vertical slices, each independently testable and each leaving the repo green.

**Slice 1 — deterministic end-to-end spine**
Contracts schemas → objective presets + structured compiler → DSL + static validation +
compiler → leakage validation → walk-forward evaluator → DuckDB registry → acceptance
report. One seeded hypothesis proves the path.

**Slice 2 — required and complementary features**
`required_features` support; conditional incremental evaluation `Model(R)` vs
`Model(R ∪ {f})`; matched-complexity random-feature and shuffled controls; bounded beam
search; feature bundles; complementarity classification.

**Slice 3 — embeddings and static clusters**
Player-gameweek and player-static embeddings from *available* columns; PCA; clustering
with multi-criterion k selection; machine-generated summaries from dominant dimensions;
cluster-conditioned evaluation.

**Slice 4 — dynamic and objective-conditioned clusters**
Rolling per-gameweek fitting; transition history; soft membership; Approach A
(objective-weighted distance) and Approach B (supervised projection); comparison against
three controls.

**Slice 5 — evidence-driven hypothesis generation**
Residual analysis by segment; the 15-hypothesis seeded library; generator driven by
measured weakness; hypothesis memory with duplicate suppression and an
exploit/correct/explore/cover allocation. **The LLM path is an adapter interface only** —
there is no client library and no key in this repository, so the shipped generator is
deterministic and the adapter is documented as unavailable.

**Slice 6 — recommendation integration**
Constraint filtering; belief-adjusted vs raw recommendations; objective-aware scoring;
opportunity cost for locked players; sensitivity.

**Slice 7 — surfaces**
CLI commands following existing `typer` patterns; FastAPI routes following existing
patterns; UI extending the floodlit-night system.

---

## 7. Testing strategy

Mirrors the existing layout (`tests/<subsystem>/test_*.py`) and marker vocabulary
(`leakage`, `golden`, `e2e`, `network`).

**Unit** — objective parsing and confidence; constraint validation; belief decay; DSL
parse/serialise round-trip; program hashing; static rejection of every forbidden
construct; rolling-window correctness against hand-computed values; feature dependency
resolution; versioning; cluster assignment determinism; objective weighting; utility
scoring.

**Leakage (`@pytest.mark.leakage`, every one with a negative control)** —
future targets cannot enter a computed program; future ownership/price cannot be read;
rolling values stop at the correct gameweek; **cluster models fitted only on training
folds** (assert that appending validation rows does not move a training-fold centroid);
scalers and projections fitted only on training folds; the DSL's group primitives do not
pool across cutoffs.

**Integration** — compile objective → generate hypotheses → validate → backtest →
register; locked player → legal constrained recommendation; required feature →
complementary search; objective switch → different feature ranking; objective switch →
different clusters; belief adjustment → both predictions retained.

**Regression** — the entire existing suite must stay green. `make check`
(`lint types boundaries banned-strings test`) is the gate.

---

## 8. Unresolved assumptions

1. **Rank-gain objectives use a proxy.** True overall rank requires the global FPL
   distribution, which is not ingested. `expected_rank_gain` is modelled as a
   variance-and-ownership-adjusted points differential against the field, and is labelled
   a **proxy** everywhere it appears. It is not a rank.

2. **Ownership is `selected` (absolute count), not effective ownership.** EO — which
   accounts for captaincy — is not published. Differential logic uses `selected` share of
   the top of the market, and says so.

3. **Team-value growth is not price-change prediction.** D11 defers the price model, and
   no current-season price data exists at GW1. `team_value_growth` scores *transfer
   momentum* (`transfers_balance` relative to ownership), which is a documented leading
   indicator, not a price forecast.

4. **Chip planning stays state-only.** D5 excludes chip logic; `chip_plan` is carried
   through constraints and validated, but no chip is optimised.

5. **No LLM.** Confirmed by search: no `anthropic`/`openai` package in `uv.lock`, no
   `.env`, no key in the environment. Slice 5 ships a deterministic generator plus an
   adapter protocol. Any LLM rationale, if a client is ever supplied, is labelled a
   **hypothesis**, never a causal finding.

6. **Positions come from `players_history.position`.** `player_gameweek_stats` has no
   position column, so position-scoped work joins through `players_history` per season.

7. **The 2025-26 season is the held-out test period** and is not used for acceptance
   decisions. Acceptance runs on 2022-23 … 2024-25 folds.

8. **`defensive_contribution` is unusable as a historical feature.** Present for one
   season only. Programs may reference it, but the missingness penalty will normally
   reject them, and that rejection is the correct answer rather than a bug.

---

## 9. What is automated, what is deterministic, what uses an LLM

Stated plainly because the brief requires it, and because the distinction is what makes
the claim honest:

- **Deterministic:** everything shipped. The DSL, compilation, leakage validation,
  backtesting, utility scoring, acceptance, clustering, embeddings, the objective
  compiler's structured path, and the seeded hypothesis library. Every RNG is seeded.
- **Automated but not intelligent:** hypothesis generation from measured residual
  weakness. The generator searches a declared space of mechanisms and instantiates
  programs; it does not reason about football.
- **LLM:** nothing, in this repository. An adapter protocol exists so a client can be
  supplied later; there is none today, and no code path depends on one.
- **Predictive association, not causation:** every reported relationship. No causal claim
  is made anywhere in the output vocabulary.
