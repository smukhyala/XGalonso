# Database Schema

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Database Schema |
| Version | 1.1 |
| Status | Draft — schema design, **not** as-built. See section 0 |
| Owner | Data Platform |
| Dependencies | [Data Sources](01_data_sources.md), [Feature Factory](../ml/02_feature_factory.md), [Repository Structure](../architecture/01_repository_structure.md) |
| Last updated | 2026-08-04 |

---

## 0. As built — read this before the DDL

**Nothing in this document's DDL has been executed.** No table in sections 2, 3 or 4 exists as a
table. This is a schema *design*; what runs is flatter, and the gap is large enough that a reader
who skipped this note would be misled about every section below.

What actually persists, under `.data/`:

| Layer | Reality |
|---|---|
| Bronze | JSON snapshots on disk, one directory per endpoint family, through `FileSystemBronzeStore`. Immutable and append-only, as designed |
| Silver | Two parquet files: `player_gameweek_stats.parquet` and `players_history.parquet` |
| Gold | Three parquet files: `training_frame.parquet`, `features_gw1.parquet`, `feature_importance.parquet` |
| Pinned constants | `pinned/rules_2026-27.json`, not a `game_config_snapshots` table |
| Discovery | The nine `discovery_*` tables in `registry.py` — the **only** tables in the system with a declared schema and a version constant (`REGISTRY_SCHEMA_VERSION`), and they too are written through `ParquetTableStore` |

**The canonical key is `player_code`, not `player_id`.** This document keys most tables on
`player_id` with `season` alongside. The code does not: `player_code` (`elements[].code`) is stable
across seasons and is what every silver and gold frame is keyed on, precisely so a cross-season
join does not silently pair two different footballers. Where a season-scoped `player_id` appears it
is a payload field being carried, never a key. The convention in §1.1 states this correctly; the DDL
below contradicts it.

**Sections 2, 3 and 4 are retained as design.** The column-level thinking in them — four
timestamps, integer money, provenance on every row, the reconciliation rule — is followed by the
parquet frames even though the DDL is not. Read them for that, not as a description of storage.

---

## 1. Storage decision

**DuckDB + Parquet only. There is no PostgreSQL** (D2).

In practice, **Parquet only**. `DuckDBTableStore` exists, is tested, and is constructed nowhere
outside the test suite; there is no `.duckdb` file on disk. `apps/cli` and `apps/api` are both
forbidden from importing `duckdb` by the `duckdb-isolation` contract in `.importlinter`, so the
composition root uses `ParquetTableStore`. That is not a failure of D2 — it is the boundary D2
required doing its job. The store was swapped and no caller changed.

Nothing in this project runs a database server, and nothing is containerised (D1).

All persistence sits **behind a repository interface**. Application, optimizer and feature code
depend on the interface, never on `duckdb` directly:

```python
from typing import Protocol, Sequence
from datetime import datetime


class PlayerRepository(Protocol):
    """Read access to canonical player rows.

    Implementations must be point-in-time honest: no method returns a row whose
    available_time is later than as_of.
    """

    def get_players(self, season: str, as_of: datetime) -> Sequence["PlayerRow"]: ...

    def get_player_gameweek_stats(
        self,
        season: str,
        player_ids: Sequence[int],
        up_to_event: int,
    ) -> Sequence["PlayerGameweekStatsRow"]: ...
```

The DuckDB implementation lives in the storage adapter package alongside the Parquet one. Swapping
to a server-backed store later is a new implementation of the same protocol and touches no caller.
That is the entire reason the interface exists — and it has already paid for itself once, when the
running system settled on Parquet without a downstream change.

The protocol above is illustrative. The shipped protocols are `TableStore` and `BronzeStore` in
`packages/data_contracts/src/xg_alonso/contracts/storage.py`, typed in terms of polars frames rather
than row sequences.

### 1.1 Conventions

| Convention | Rule |
|---|---|
| Season key | `season VARCHAR` in `'2026-27'` form. Every season-scoped table carries it |
| Cross-season identity | `player_code` and `team_code` (`elements[].code`, `teams[].code`) |
| Season identity | `player_id`, `team_id` — re-issued each season, never a standalone key |
| Money | Stored in **tenths of a million**, exactly as the API reports. `INTEGER`, never `DOUBLE`. `55` means £5.5m |
| Time | `TIMESTAMPTZ`, always UTC |
| Provenance | Every silver table carries `snapshot_id`, `observed_time`, `available_time`, `processed_time` |
| Naming | `snake_case`, singular column names, plural table names |

**Money is never a float.** FPL prices are exact tenths; representing them as `DOUBLE` introduces
comparison errors in budget constraints where the optimizer needs exact feasibility.

---

## 2. Slice-1 tables

**Designed, never created.** These five were the intended canonical surface. What runs is two silver
parquet frames; `player_gameweek_stats` is the one whose grain and column set the shipped frame
actually follows, and it is keyed on `player_code`, not `player_id`.

### 2.1 `teams`

```sql
CREATE TABLE teams (
    season                  VARCHAR   NOT NULL,  -- '2026-27'; teams.id is re-issued per season
    team_id                 INTEGER   NOT NULL,  -- teams[].id, 1-20 alphabetical, season-scoped
    team_code               INTEGER   NOT NULL,  -- teams[].code, stable across seasons
    name                    VARCHAR   NOT NULL,  -- full club name, e.g. 'Arsenal'
    short_name              VARCHAR   NOT NULL,  -- three-letter code, e.g. 'ARS'
    strength                INTEGER,             -- NULL preseason; verified null before GW1
    strength_overall_home   INTEGER,             -- 1-5 preseason, ~1000-1400 in-season: scale differs
    strength_overall_away   INTEGER,             -- same dual-scale hazard as above
    strength_attack_home    INTEGER,             -- 0 for all 20 teams preseason
    strength_attack_away    INTEGER,             -- 0 for all 20 teams preseason
    strength_defence_home   INTEGER,             -- 0 for all 20 teams preseason
    strength_defence_away   INTEGER,             -- 0 for all 20 teams preseason
    strength_scale          VARCHAR   NOT NULL,  -- 'preseason_1_5' | 'in_season_1000_1400' | 'unknown'
    pulse_id                INTEGER,             -- Premier League feed id, nullable
    snapshot_id             VARCHAR   NOT NULL,  -- bronze snapshot this row was derived from
    observed_time           TIMESTAMPTZ NOT NULL,-- when XG Alonso received the payload
    available_time          TIMESTAMPTZ NOT NULL,-- when the information became publicly usable
    processed_time          TIMESTAMPTZ NOT NULL,-- when this row was written
    PRIMARY KEY (season, team_id)
);
```

`strength_scale` is written by the ingest adapter, not inferred by consumers. It exists because the
same column carries a 1–5 value preseason and a ~1000–1400 value in-season; a rolling feature that
spans the transition without normalising is silently wrong.

### 2.2 `players`

Current-state dimension, one row per player per season. Time-varying price and ownership series are
**not** stored here — per-gameweek price lives on `player_gameweek_stats.value`, and intraday
series belong to the deferred `player_snapshots` table.

```sql
CREATE TABLE players (
    season                       VARCHAR   NOT NULL,  -- '2026-27'
    player_id                    INTEGER   NOT NULL,  -- elements[].id, season-scoped
    player_code                  INTEGER   NOT NULL,  -- elements[].code, stable cross-season key
    team_id                      INTEGER   NOT NULL,  -- FK to teams(season, team_id)
    team_code                    INTEGER   NOT NULL,  -- denormalised for cross-season joins
    element_type                 INTEGER   NOT NULL,  -- 1=GKP 2=DEF 3=MID 4=FWD, from element_types
    first_name                   VARCHAR   NOT NULL,  -- given name as published
    second_name                  VARCHAR   NOT NULL,  -- family name as published
    web_name                     VARCHAR   NOT NULL,  -- short display name used in the UI
    status                       VARCHAR   NOT NULL,  -- a=available d=doubtful i=injured s=suspended u=unavailable n=not in squad
    chance_of_playing_this_round INTEGER,             -- 0-100, NULL when FPL states nothing
    chance_of_playing_next_round INTEGER,             -- 0-100, NULL when FPL states nothing
    news                         VARCHAR,             -- free-text availability note, may be empty
    news_added                   TIMESTAMPTZ,         -- when the note was published; drives availability recency
    now_cost                     INTEGER   NOT NULL,  -- current price in tenths of a million
    cost_change_start            INTEGER   NOT NULL,  -- tenths changed since season start
    cost_change_event            INTEGER   NOT NULL,  -- tenths changed since last gameweek
    selected_by_percent          DOUBLE    NOT NULL,  -- global ownership percent as reported
    total_points                 INTEGER   NOT NULL,  -- season-to-date points, snapshot value
    minutes                      INTEGER   NOT NULL,  -- season-to-date minutes, snapshot value
    form                         DOUBLE,              -- FPL's own form metric; NULL preseason
    points_per_game              DOUBLE,              -- FPL's own PPG; NULL preseason
    ep_this                      DOUBLE,              -- FPL expected points this GW; BANNED as a model feature
    ep_next                      DOUBLE,              -- FPL expected points next GW; BANNED as a model feature
    squad_number                 INTEGER,             -- shirt number, frequently NULL
    photo                        VARCHAR,             -- asset filename, presentation only
    snapshot_id                  VARCHAR   NOT NULL,  -- bronze snapshot this state came from
    observed_time                TIMESTAMPTZ NOT NULL,
    available_time               TIMESTAMPTZ NOT NULL,
    processed_time               TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, player_id)
);
```

`ep_this` and `ep_next` are stored for comparison and are **banned as training features** — they are
FPL's own forward-looking estimate and their derivation is not point-in-time auditable. Storing them
lets XG Alonso benchmark against them; using them would make the model a copy of them.

### 2.3 `gameweeks`

```sql
CREATE TABLE gameweeks (
    season                  VARCHAR   NOT NULL,  -- '2026-27'
    event_id                INTEGER   NOT NULL,  -- events[].id, 1-38
    name                    VARCHAR   NOT NULL,  -- 'Gameweek 1'
    deadline_time           TIMESTAMPTZ NOT NULL,-- squad lock; the canonical prediction_timestamp
    deadline_time_epoch     BIGINT    NOT NULL,  -- as published, retained for exact round-tripping
    finished                BOOLEAN   NOT NULL,  -- all fixtures played
    data_checked            BOOLEAN   NOT NULL,  -- bonus and BPS final; gates available_time for bonus
    is_previous             BOOLEAN   NOT NULL,  -- snapshot-relative flag, not a stable fact
    is_current              BOOLEAN   NOT NULL,  -- snapshot-relative flag, not a stable fact
    is_next                 BOOLEAN   NOT NULL,  -- snapshot-relative flag, not a stable fact
    average_entry_score     INTEGER,             -- NULL until the gameweek finishes
    highest_score           INTEGER,             -- NULL until the gameweek finishes
    most_selected           INTEGER,             -- player_id, NULL before the deadline
    most_transferred_in     INTEGER,             -- player_id, NULL before the deadline
    most_captained          INTEGER,             -- player_id, NULL before the deadline
    most_vice_captained     INTEGER,             -- player_id, NULL before the deadline
    top_element             INTEGER,             -- highest scoring player_id, NULL until finished
    transfers_made          INTEGER,             -- global transfer count for the gameweek
    ranked_count            INTEGER,             -- entries ranked in this gameweek
    chip_plays              JSON,                -- [{chip_name, num_played}]; state only, no chip logic (D5)
    snapshot_id             VARCHAR   NOT NULL,
    observed_time           TIMESTAMPTZ NOT NULL,
    available_time          TIMESTAMPTZ NOT NULL,
    processed_time          TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, event_id)
);
```

`is_previous`, `is_current` and `is_next` describe the snapshot, not the season. Never derive
"current gameweek" from them in a historical query — derive it from `deadline_time` against the
prediction timestamp.

### 2.4 `fixtures`

```sql
CREATE TABLE fixtures (
    season                  VARCHAR   NOT NULL,  -- '2026-27'
    fixture_id              INTEGER   NOT NULL,  -- fixtures[].id, season-scoped
    fixture_code            BIGINT    NOT NULL,  -- fixtures[].code, stable across seasons
    event_id                INTEGER,             -- gameweek; NULL when not yet scheduled
    kickoff_time            TIMESTAMPTZ,         -- NULL for unscheduled fixtures
    provisional_start_time  BOOLEAN   NOT NULL,  -- true when kickoff_time may still move
    team_h                  INTEGER   NOT NULL,  -- home team_id, season-scoped
    team_a                  INTEGER   NOT NULL,  -- away team_id, season-scoped
    team_h_score            INTEGER,             -- NULL until played
    team_a_score            INTEGER,             -- NULL until played
    team_h_difficulty       INTEGER   NOT NULL,  -- FDR 1-5 for the home side
    team_a_difficulty       INTEGER   NOT NULL,  -- FDR 1-5 for the away side
    started                 BOOLEAN,             -- NULL before the fixture is scheduled
    finished                BOOLEAN   NOT NULL,  -- match complete
    finished_provisional    BOOLEAN   NOT NULL,  -- gates available_time for match-derived stats
    minutes                 INTEGER   NOT NULL,  -- elapsed match minutes, 0 before kickoff
    pulse_id                BIGINT,              -- Premier League feed id, nullable
    stats                   JSON,                -- per-match stat breakdown, populated after kickoff
    snapshot_id             VARCHAR   NOT NULL,
    observed_time           TIMESTAMPTZ NOT NULL,
    available_time          TIMESTAMPTZ NOT NULL,
    processed_time          TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, fixture_id)
);
```

Fixtures with `event_id IS NULL` are real and must not be dropped. Fixture-count and congestion
features have to represent "scheduled but undated" explicitly rather than treating it as absent.

### 2.5 `player_gameweek_stats`

The core fact table. One row per player per fixture — note that double gameweeks produce **two rows
for the same `(season, player_id, event_id)`**, which is why `fixture_id` is part of the key.

```sql
CREATE TABLE player_gameweek_stats (
    season                         VARCHAR   NOT NULL,  -- '2026-27'
    player_id                      INTEGER   NOT NULL,  -- season-scoped elements[].id
    player_code                    INTEGER   NOT NULL,  -- stable cross-season key
    event_id                       INTEGER   NOT NULL,  -- gameweek
    fixture_id                     INTEGER   NOT NULL,  -- in key: double gameweeks give 2 rows per event
    opponent_team                  INTEGER   NOT NULL,  -- opponent team_id, season-scoped
    was_home                       BOOLEAN   NOT NULL,  -- venue
    kickoff_time                   TIMESTAMPTZ,         -- event_time for this row
    team_h_score                   INTEGER,             -- NULL until played
    team_a_score                   INTEGER,             -- NULL until played
    minutes                        INTEGER   NOT NULL,  -- minutes played
    starts                         INTEGER   NOT NULL,  -- 1 if started, else 0
    goals_scored                   INTEGER   NOT NULL,  -- goals; points value is position-dependent
    assists                        INTEGER   NOT NULL,  -- assists
    clean_sheets                   INTEGER   NOT NULL,  -- 1 if a clean sheet was recorded
    goals_conceded                 INTEGER   NOT NULL,  -- goals conceded while on the pitch
    own_goals                      INTEGER   NOT NULL,  -- own goals
    penalties_saved                INTEGER   NOT NULL,  -- penalties saved (goalkeepers)
    penalties_missed               INTEGER   NOT NULL,  -- penalties missed
    yellow_cards                   INTEGER   NOT NULL,  -- yellow cards
    red_cards                      INTEGER   NOT NULL,  -- red cards
    saves                          INTEGER   NOT NULL,  -- saves made
    bonus                          INTEGER   NOT NULL,  -- PROVISIONAL until gameweeks.data_checked
    bps                            INTEGER   NOT NULL,  -- bonus points system score, also provisional
    influence                      DOUBLE    NOT NULL,  -- ICT component
    creativity                     DOUBLE    NOT NULL,  -- ICT component
    threat                         DOUBLE    NOT NULL,  -- ICT component
    ict_index                      DOUBLE    NOT NULL,  -- composite ICT index
    expected_goals                 DOUBLE,              -- xG; available from 2022/23 onward (D7)
    expected_assists               DOUBLE,              -- xA; available from 2022/23 onward
    expected_goal_involvements     DOUBLE,              -- xG + xA
    expected_goals_conceded        DOUBLE,              -- team xGC while the player was on the pitch
    defensive_contribution         INTEGER,             -- DC stat; scores 2 pts for DEF/MID/FWD, 0 for GKP
    clearances_blocks_interceptions INTEGER,            -- CBI component of DC
    recoveries                     INTEGER,             -- ball recoveries
    tackles                        INTEGER,             -- tackles
    total_points                   INTEGER   NOT NULL,  -- ground truth; must reconcile to component sum
    value                          INTEGER   NOT NULL,  -- price at this gameweek, tenths of a million
    selected                       INTEGER   NOT NULL,  -- entries selecting the player
    transfers_in                   INTEGER   NOT NULL,  -- transfers in during this gameweek
    transfers_out                  INTEGER   NOT NULL,  -- transfers out during this gameweek
    transfers_balance              INTEGER   NOT NULL,  -- transfers_in - transfers_out
    source                         VARCHAR   NOT NULL,  -- 'live_api' | 'community_archive'
    snapshot_id                    VARCHAR   NOT NULL,
    observed_time                  TIMESTAMPTZ NOT NULL,
    available_time                 TIMESTAMPTZ NOT NULL,-- gated on finished_provisional; bonus on data_checked
    processed_time                 TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, player_id, event_id, fixture_id)
);
```

`total_points` is stored as ground truth and is also the **reconciliation target**: applying the
pinned scoring rules to the component columns must reproduce it. A mismatch is a hard ingest
failure, because it means either the scoring snapshot has drifted or a component column is
misinterpreted. This is what makes component-based points modelling (D8) auditable.

`source` distinguishes live-API rows from community-archive rows so any evaluation can be re-run
against live rows only. See [Data Sources §1.1](01_data_sources.md).

---

## 3. Supporting tables

**Designed, never created.** Both rules below are implemented; neither is implemented as a table.
The bronze index is directory structure plus per-snapshot metadata written by
`FileSystemBronzeStore`, and the pinned constants are a JSON file at `.data/pinned/rules_<season>.json`
loaded through `SquadRules.from_payload`. The drift check runs on every ingest as designed.

### 3.1 `raw_snapshots`

The bronze index. Enforces the immutability rule by making every silver row traceable.

```sql
CREATE TABLE raw_snapshots (
    snapshot_id     VARCHAR   NOT NULL,  -- ULID; inherited by every derived row
    endpoint        VARCHAR   NOT NULL,  -- 'bootstrap-static/', 'event/7/live/', ...
    request_url     VARCHAR   NOT NULL,  -- fully resolved URL actually called
    http_status     INTEGER   NOT NULL,  -- 200, 404 (a valid preseason picks response), ...
    payload_path    VARCHAR   NOT NULL,  -- Parquet/JSON path on local disk; append-only
    payload_sha256  VARCHAR   NOT NULL,  -- checksum; detects any post-hoc mutation
    observed_time   TIMESTAMPTZ NOT NULL,-- HTTP response time
    run_id          VARCHAR   NOT NULL,  -- ingest run that produced this snapshot
    PRIMARY KEY (snapshot_id)
);
```

### 3.2 `game_config_snapshots`

The pinned constants. **Scoring and constraint constants load from this table, never from Python
literals.**

```sql
CREATE TABLE game_config_snapshots (
    config_id       VARCHAR   NOT NULL,  -- pinned config version referenced by model/optimizer runs
    season          VARCHAR   NOT NULL,  -- '2026-27'
    scoring         JSON      NOT NULL,  -- verbatim game_config.scoring from bootstrap-static
    rules           JSON      NOT NULL,  -- verbatim game_config.rules: squad size, budget, limits
    element_types   JSON      NOT NULL,  -- positional quotas and formation bounds
    fetched_at      TIMESTAMPTZ NOT NULL,-- recorded fetch timestamp, required by the pinning rule
    payload_sha256  VARCHAR   NOT NULL,  -- drift check compares live payload against this
    is_pinned       BOOLEAN   NOT NULL,  -- exactly one pinned row per season
    PRIMARY KEY (config_id)
);
```

The drift check runs on every ingest: if the live `game_config` hash differs from the pinned row,
the run fails and a human re-pins deliberately. Values that must come from here rather than being
transcribed include goals-scored points by position (goalkeeper goals are worth **10**, not 6),
clean-sheet values, `defensive_contribution` values, squad size, budget, per-club limit, and the
free-transfer cap. See [Transfer Planner §3](../optimization/02_transfer_planner.md).

---

## 4. Deferred tables

Named and reserved so their eventual arrival is not a migration surprise. **None are built**, but
two have been overtaken by something in a different shape and should be read with that in mind:

- `embeddings` and `clusters` — the capability shipped. Cluster models and assignments persist to
  `discovery_cluster_models` and `discovery_cluster_assignments`; the embedding vectors themselves
  are recomputed per fold rather than stored, because they are conditioned on the objective and a
  stored vector without its objective would be a trap.
- `knowledge_objects` — partly overtaken by `discovery_hypotheses`, `discovery_evaluations` and
  `discovery_lessons`, which cover within-experiment learning. Cross-season accumulation is not
  built.

The nine `discovery_*` tables are the only declared, versioned schema in the system. They are
listed in `packages/discovery/src/xg_alonso/discovery/registry.py` and carry
`REGISTRY_SCHEMA_VERSION`.

| Table | Purpose | Blocked on |
|---|---|---|
| `player_snapshots` | Intraday price and ownership time series at snapshot grain; the substrate for price-change modelling | Price model deferred (D11); no current-season price history exists at GW1 |
| `feature_registry` | One row per feature definition: name, generator, parameters, version, lineage, tags, introduction date, retirement date | [Feature Factory](../ml/02_feature_factory.md) materialization |
| `feature_values` | Materialized feature values at `(player_id, prediction_timestamp, feature_name, feature_version)` grain | Feature registry |
| `embeddings` | Player, team, manager and fixture vectors with embedding version and training data cutoff | [Embeddings](../ml/06_embeddings.md), post-MVP |
| `clusters` | Player archetype assignments, cluster centroids, cluster version | [Player Clustering](../ml/05_player_clustering.md), post-MVP |
| `model_registry` | Trained model artefacts: model version, feature-set version, training window, walk-forward metrics, champion/challenger state | Baseline models shipping first |
| `recommendations` | Persisted recommendation cards with full provenance, for backtesting realised decision quality | Optimizer output stabilising |
| `transfer_packages` | Multi-transfer candidate packages with cost, hit accounting and expected gain | Multi-transfer planning; slice 1 is single-transfer only |
| `wildcard_runs` | Wildcard squad and timing evaluations | Chips excluded from MVP (D5) — see [Wildcard Planner](../optimization/03_wildcard_planner.md) |
| `knowledge_objects` | Hypotheses, experiments and outcomes accumulated by the research layer | [Knowledge Lab](../research/01_knowledge_lab.md), post-MVP (D10) |

`player_match_stats` from the original table list is superseded by `player_gameweek_stats`, which is
the same grain under a name that matches the API's own vocabulary. No content is lost by the rename.

---

## 5. Acceptance criteria

Met, against the parquet frames rather than against tables:

- Loading a pinned bronze snapshot twice produces identical contents.
- Component-to-`total_points` reconciliation passes for every row in `player_gameweek_stats`.
- No monetary column is a floating-point type.
- All reads in application code go through the `TableStore` / `BronzeStore` protocols. `duckdb`
  outside the storage adapter is not merely absent — it is unimportable, enforced by
  `.importlinter` on every CI run rather than by a grep.

Not met, and superseded rather than outstanding:

- *"Every slice-1 table is creatable from a single idempotent DDL script"* — there is no DDL script
  and no table. Frames are written whole by the pipeline that produces them.
- *"No table keyed on `player_id` or `team_id` omits `season`"* — nothing is keyed on `player_id`.
  The stronger rule the code follows is that the key is `player_code`, which needs no season to be
  unambiguous.
- *"Every silver row resolves to an existing `raw_snapshots.snapshot_id`"* — there is no
  `raw_snapshots` table, so this is unenforced. Provenance is carried on the artifact manifest and
  in the bronze directory layout. **This is a real gap**: a silver frame cannot today be traced back
  to the exact snapshot that produced it by a mechanical check.

---

## Related documents

- [Data Sources](01_data_sources.md) — endpoints, cadence and the four timestamps
- [Feature Factory](../ml/02_feature_factory.md) — the primary consumer of these tables
- [Prediction Models](../ml/07_prediction_models.md) — component targets sourced here
- [Transfer Planner](../optimization/02_transfer_planner.md) — constraint constants from `game_config_snapshots`
- [Public API](../api/01_public_api.md) — surfaces reading through the repository interface
- [Repository Structure](../architecture/01_repository_structure.md) — where the storage adapter lives
- [Build Plan](../implementation/01_build_plan.md) — phase 4 delivers these tables
- [Documentation Index](../README.md)
