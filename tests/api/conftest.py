"""A data root the HTTP surface can actually be served from, built in a temp dir.

`apps/api` is three thousand lines behind eighteen routes, and until this
directory existed none of it was executed by a test. The obstacle was never the
wiring — `main._service` is already a FastAPI dependency, so an override is one
line — it was that `DecisionService` reads a populated `.data` tree and a fresh
clone has almost none of one.

**So the fixture builds the tree rather than the objects.** Everything here
writes real files in the real layout: a gzipped bronze snapshot through
`FileSystemBronzeStore`, a silver parquet conforming to
`PLAYER_GAMEWEEK_STATS_SCHEMA`, a gold importance table through
`write_importance`, a discovery registry through `DiscoveryRegistry`. Nothing is
monkeypatched and no loader is stubbed, so these tests exercise the same reading
path production does — a mocked repository would have proved only that the mock
returns what it was told to.

**Scoring rules and squad constraints are never synthesised.** They come
verbatim from `data/fixtures/fpl/bootstrap_static_2026_27.json`, which is the
pinned snapshot CLAUDE.md requires. Only the *roster* around them is
synthetic, because that fixture truncates its player list and cannot fill a
15-man squad — the same trade `tests/test_end_to_end.py` documents and makes.

**`api_data_root` is deliberately the only seam.** A parallel effort is
committing shareable fixtures under `data/fixtures/` for an offline `xg demo`.
When those land, this file changes in one place: `api_data_root` copies or
points at the committed tree instead of calling `_build_data_root`, and every
test above it is untouched. That is why the builder is a private function behind
a single fixture rather than a set of fixtures each test composes for itself.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from tests.conftest import BOOTSTRAP_FIXTURE
from xg_alonso.api.service import DecisionService, ServiceConfig
from xg_alonso.contracts.provenance import SourceTimestamps, TimeSource
from xg_alonso.pipelines.ingestion import SOURCE_BOOTSTRAP, SOURCE_FIXTURES
from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA, conform
from xg_alonso.storage import FileSystemBronzeStore

if TYPE_CHECKING:
    from fastapi.testclient import TestClient as TestClientType

# Importing `fastapi.testclient` emits a StarletteDeprecationWarning about
# httpx2 at module-import time. It is swallowed here, once, rather than added to
# the project-wide `filterwarnings` list: a global ignore would also hide the
# warning if it ever started firing somewhere that matters, and the test modules
# take their client from the `client` fixture so this stays the only import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

#: When the snapshot was taken. Before the GW1 deadline in the fixture, so
#: `/health` reports `stale=False` rather than being permanently out of date.
SNAPSHOT_TIME = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)

#: The deadline the fixture's GW1 carries. Asserted against, so it is named.
GW1_DEADLINE = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)

#: The entry id every squad-shaped test uses.
ENTRY_ID = 1

_POSITION_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
_ELEMENT_TYPE = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

#: Clubs in the fixture. The synthetic roster spreads across all of them so the
#: three-per-club cap is satisfiable rather than something the builder dodges.
_CLUB_COUNT = 6


def _synthetic_roster(real_payload: dict[str, Any]) -> dict[str, Any]:
    """Real rules, plus enough players to field a legal squad.

    Prices rise with the slot index and so does prior-season output, which is
    what lets a test assert that the optimizer picked *sensibly* rather than
    merely legally. `selected_by_percent` varies too, because the differential
    objective divides by the most-owned player and a constant column would make
    `_ownership_share` return a uniform map that cannot express a lean.
    """
    payload: dict[str, Any] = json.loads(json.dumps(real_payload))

    elements: list[dict[str, Any]] = []
    element_id = 1
    for position, quota in _POSITION_QUOTA.items():
        for slot in range(quota * 3):
            elements.append(
                {
                    "id": element_id,
                    "code": 900000 + element_id,
                    "web_name": f"{position}{slot}",
                    "first_name": position,
                    "second_name": str(slot),
                    "element_type": _ELEMENT_TYPE[position],
                    "team": (slot % _CLUB_COUNT) + 1,
                    "team_code": (slot % _CLUB_COUNT) + 1,
                    "now_cost": 40 + slot * 5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "selected_by_percent": str(round(1.0 + slot * 1.5, 1)),
                }
            )
            element_id += 1
    payload["elements"] = elements
    payload["events"] = [
        {
            "id": 1,
            "deadline_time": GW1_DEADLINE.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": False,
            "is_current": False,
            "is_next": True,
        }
    ]
    return payload


def _synthetic_history(payload: dict[str, Any]) -> pl.DataFrame:
    """Prior-season rows, with output rising with price.

    One season-aggregate row per player, `available_time` well before the
    deadline so every row is legitimately visible at the cutoff. That is the
    point-in-time contract the feature build enforces; a row dated after the
    deadline would simply be dropped and the features would come out empty.
    """
    rows: list[dict[str, Any]] = []
    season_end = datetime(2026, 6, 30, tzinfo=UTC)
    for element in payload["elements"]:
        quality = (element["now_cost"] - 40) / 40.0
        rows.append(
            {
                "player_code": element["code"],
                "gameweek_id": None,
                "season": "2025/26",
                "fixture_id": None,
                "opponent_team_id": None,
                "was_home": None,
                "minutes": int(600 + quality * 2400),
                "starts": int(6 + quality * 26),
                "goals_scored": int(quality * 18),
                "assists": int(quality * 8),
                "clean_sheets": 5,
                "goals_conceded": 30,
                "saves": 0,
                "yellow_cards": 3,
                "red_cards": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "bonus": int(quality * 20),
                "bps": int(quality * 500),
                "total_points": int(30 + quality * 180),
                "expected_goals": quality * 15.0,
                "expected_assists": quality * 7.0,
                "expected_goal_involvements": quality * 22.0,
                "expected_goals_conceded": 30.0,
                "defensive_contribution": None,
                "value": element["now_cost"],
                "kickoff_time": season_end,
                "available_time": season_end,
            }
        )
    return conform(pl.DataFrame(rows, infer_schema_length=None), PLAYER_GAMEWEEK_STATS_SCHEMA)


def legal_squad_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The cheapest legal fifteen, with money left over.

    Slots 1-11 are a legal 1-4-4-2 and 12-15 the bench. Laying the squad out in
    flat position order would put both goalkeepers in the XI, which the real
    formation rules correctly reject — and the squad is validated on load, so
    that would fail as a 422 rather than as a bad fixture.
    """
    by_position: dict[int, list[dict[str, Any]]] = {}
    for element in payload["elements"]:
        by_position.setdefault(element["element_type"], []).append(element)
    for group in by_position.values():
        group.sort(key=lambda e: int(e["now_cost"]))

    club_count: dict[int, int] = {}
    chosen: dict[str, list[dict[str, Any]]] = {}
    for position, quota in _POSITION_QUOTA.items():
        picked: list[dict[str, Any]] = []
        for element in by_position[_ELEMENT_TYPE[position]]:
            if len(picked) == quota:
                break
            if club_count.get(element["team"], 0) >= 3:
                continue
            club_count[element["team"]] = club_count.get(element["team"], 0) + 1
            picked.append(element)
        assert len(picked) == quota, f"the synthetic roster could not fill {position}"
        chosen[position] = picked

    starting = chosen["GKP"][:1] + chosen["DEF"][:4] + chosen["MID"][:4] + chosen["FWD"][:2]
    bench = chosen["GKP"][1:] + chosen["DEF"][4:] + chosen["MID"][4:] + chosen["FWD"][2:]

    picks: list[dict[str, Any]] = []
    spend = 0
    for slot, element in enumerate(starting + bench, start=1):
        picks.append(
            {
                "element": element["id"],
                "position": slot,
                "purchase_price": element["now_cost"],
                "is_captain": slot == 1,
                "is_vice_captain": slot == 2,
            }
        )
        spend += int(element["now_cost"])

    return {"picks": picks, "bank": 1000 - spend, "free_transfers": 1}


def _write_bronze(root: Path, payload: dict[str, Any]) -> None:
    store = FileSystemBronzeStore(root / "bronze")
    timestamps = SourceTimestamps(
        event_time=SNAPSHOT_TIME,
        observed_time=SNAPSHOT_TIME,
        available_time=SNAPSHOT_TIME,
        processed_time=SNAPSHOT_TIME,
        time_source=TimeSource.DEADLINE,
    )
    store.write(
        source=SOURCE_BOOTSTRAP,
        payload=json.dumps(payload).encode("utf-8"),
        timestamps=timestamps,
        run_id="api-fixture",
    )
    # An empty fixture list is honest: the fixture snapshot is preseason and the
    # schedule is genuinely unpublished. `/players` therefore reports a null
    # opponent per horizon week, which is a state the shape must survive.
    store.write(
        source=SOURCE_FIXTURES,
        payload=b"[]",
        timestamps=timestamps,
        run_id="api-fixture",
    )


def _write_events(root: Path) -> None:
    """What `xg refresh` leaves behind, so `/health` freshness is exercised.

    Two material rows and one immaterial one, because `unseen_events` counts
    only `critical` and `material` — a test whose every row counted could not
    tell filtering from summing.
    """
    directory = root / "events"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "last_checked.json").write_text(
        json.dumps({"checked_at": SNAPSHOT_TIME.isoformat()}), encoding="utf-8"
    )
    (directory / "player_events.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"player_code": 900001, "materiality": "critical"},
                {"player_code": 900002, "materiality": "material"},
                {"player_code": 900003, "materiality": "minor"},
            )
        )
        + "\n",
        encoding="utf-8",
    )


#: How many events in the fixture log count as unseen. Named so the assertion
#: reads as a claim about filtering rather than as a magic number.
MATERIAL_EVENTS = 2


def _write_importance(root: Path) -> None:
    """A small gold importance table, with both slices and both fold indices.

    Two folds, because `rank_stability` returns nothing below two and a table
    with one fold would leave that branch permanently unmeasured. A degenerate
    label is included, because it must be reported in `degenerate_labels` and
    excluded from the ranking, and a table without one cannot show that.
    """
    from xg_alonso.evaluation.importance import (
        ALL_POSITIONS,
        FeatureImportance,
        ImportanceTable,
        write_importance,
    )

    rows: list[FeatureImportance] = []
    features = (
        ("minutes_mean_5", "minutes"),
        ("xg_per90_shrunk", "attacking"),
        ("points_per90_shrunk", "returns"),
    )
    for fold in (0, 1):
        for position in (ALL_POSITIONS, "DEF"):
            for index, (name, family) in enumerate(features):
                rows.append(
                    FeatureImportance(
                        feature_name=name,
                        family=family,
                        label="label_total_points",
                        fold_index=fold,
                        baseline_mae=1.0,
                        permuted_mae=1.0 + 0.1 * (len(features) - index) + 0.01 * fold,
                        mae_delta_std=0.01,
                        repeats=3,
                        degenerate_label=False,
                        position=position,
                        label_weight=1.0,
                        rows_measured=500 if position == ALL_POSITIONS else 90,
                    )
                )
            rows.append(
                FeatureImportance(
                    feature_name="minutes_mean_5",
                    family="minutes",
                    label="label_red_cards",
                    fold_index=fold,
                    baseline_mae=0.0,
                    permuted_mae=0.0,
                    mae_delta_std=0.0,
                    repeats=3,
                    degenerate_label=True,
                    position=position,
                    label_weight=0.0,
                    rows_measured=500,
                )
            )

    write_importance(
        ImportanceTable(
            rows=tuple(rows),
            catalogue_version="test_catalogue_v1",
            model_fingerprint="fingerprint-not-loaded",
            computed_at=SNAPSHOT_TIME,
            label_weights={"label_total_points": 1.0, "label_red_cards": 0.0},
        ),
        root / "gold" / "feature_importance.parquet",
    )


#: The objective the discovery fixture files its verdicts under. Matches a
#: shipped preset so `/features/discovered?objective_id=` is a realistic call.
DISCOVERY_OBJECTIVE = "expected_points"

#: The cluster model the fixture registers, and the player it places.
CLUSTER_MODEL_VERSION = "clusters-test-v1"
CLUSTERED_PLAYER = 900001

#: The experiment the registry fixture records, fetched by id in a test.
EXPERIMENT_ID = "exp-api-fixture-1"


def _write_discovery(root: Path) -> None:
    """One of each registry record, so the six read routes return content.

    Populated rather than left absent because every one of these routes has two
    behaviours — a row shape and a graceful empty — and only a populated
    registry exercises the first. The empty half is covered by `sparse_client`,
    whose data root has no `discovery/` directory at all.
    """
    from xg_alonso.contracts.discovery import (
        AcceptanceStatus,
        ClusterSummary,
        ComplementarityClass,
        DiscoveredFeatureSpec,
        ExperimentManifest,
        ExperimentStage,
        FeatureEvaluation,
        FeatureHypothesis,
        FoldMetrics,
        HypothesisStatus,
        PlayerClusterAssignment,
        ValidationStatus,
    )
    from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
    from xg_alonso.discovery.registry import DiscoveryRegistry
    from xg_alonso.storage import ParquetTableStore

    registry = DiscoveryRegistry(ParquetTableStore(root / "discovery"))

    hypothesis = FeatureHypothesis(
        id="hyp-minutes-trend",
        title="Rising minutes precede returns",
        football_rationale=(
            "A player being trusted with more minutes each week is a manager's "
            "revealed opinion, published before any of the attacking output it "
            "leads to."
        ),
        falsification_condition=(
            "No improvement in out-of-sample MAE on any fold once minutes level "
            "is already in the model."
        ),
        expected_relationship="positive, strongest for attacking midfielders",
        transformation_plan="slope of minutes over the last five appearances",
        required_raw_fields=("minutes",),
        status=HypothesisStatus.ACCEPTED,
    )
    registry.register_hypothesis(hypothesis)

    spec = DiscoveredFeatureSpec(
        id="feat-minutes-trend",
        name="minutes_trend_5",
        version="a" * 16,
        program=json.dumps({"op": "slope", "column": "minutes", "window": 5}),
        input_columns=("minutes",),
        hypothesis_id=hypothesis.id,
        objective_tags=(DISCOVERY_OBJECTIVE,),
        # The registry refuses anything that has not cleared the leakage
        # harness, which is the gate that makes a registered feature mean
        # something. The fixture satisfies it rather than routing around it.
        validation_status=ValidationStatus.LEAKAGE_PASSED,
    )
    registry.register_feature(spec)

    registry.record_evaluation(
        FeatureEvaluation(
            feature_id=spec.id,
            feature_version=spec.version,
            objective_id=DISCOVERY_OBJECTIVE,
            backtest_start=1,
            backtest_end=10,
            folds=(
                FoldMetrics(
                    fold_index=0,
                    train_rows=400,
                    validate_rows=100,
                    baseline_metric=1.20,
                    candidate_metric=1.15,
                ),
                FoldMetrics(
                    fold_index=1,
                    train_rows=500,
                    validate_rows=100,
                    baseline_metric=1.18,
                    candidate_metric=1.14,
                ),
            ),
            incremental_value=0.035,
            stability=0.9,
            missingness=0.02,
            leakage_checks=("shuffled_future", "cutoff_shift"),
            leakage_passed=True,
            accepted=AcceptanceStatus.ACCEPTED,
            complementarity=ComplementarityClass.OBJECTIVE_SPECIFIC,
        )
    )

    registry.register_cluster_model(
        [
            ClusterSummary(
                cluster_model_version=CLUSTER_MODEL_VERSION,
                cluster_id=0,
                objective_id=DISCOVERY_OBJECTIVE,
                size=12,
                label="nailed high-volume starter",
                dominant_features=(("minutes_mean_5", 1.8), ("xg_per90_shrunk", 1.1)),
            )
        ]
    )
    registry.register_assignments(
        [
            PlayerClusterAssignment(
                player_code=PlayerCode(CLUSTERED_PLAYER),
                gameweek=GameweekId(1),
                season="2025-26",
                cluster_model_version=CLUSTER_MODEL_VERSION,
                cluster_id=0,
                membership_probability=0.82,
                distance_to_centroid=0.4,
                objective_id=DISCOVERY_OBJECTIVE,
            )
        ]
    )

    registry.register_experiment(
        ExperimentManifest(
            experiment_id=EXPERIMENT_ID,
            stage=ExperimentStage.COMPLETED,
            objective_id=DISCOVERY_OBJECTIVE,
            objective_version="1",
            constraints_hash="c" * 16,
            data_cutoff=GW1_DEADLINE,
            seasons=("2025-26",),
            hypotheses_proposed=1,
            features_compiled=1,
            features_accepted=1,
            features_rejected=0,
            metrics=(("mae", 1.145),),
            code_version="d" * 40,
            git_dirty=False,
            started_at=SNAPSHOT_TIME,
            completed_at=SNAPSHOT_TIME,
        )
    )


def _build_data_root(root: Path, *, rich: bool) -> Path:
    """Write a `.data`-shaped tree under `root`.

    ``rich`` adds everything an optional loader looks for — events, the gold
    importance table, the discovery registry. Without it only bronze and silver
    exist, which is what a clone that has run `xg ingest` and nothing else looks
    like, and is therefore the state the degradation paths must survive.
    """
    payload = _synthetic_roster(json.loads(BOOTSTRAP_FIXTURE.read_text()))
    _write_bronze(root, payload)

    (root / "silver").mkdir(parents=True, exist_ok=True)
    _synthetic_history(payload).write_parquet(root / "silver" / "player_gameweek_stats.parquet")

    (root / "squad.json").write_text(json.dumps(legal_squad_payload(payload)), encoding="utf-8")
    # A payload with no `picks` key, for the documented 422 path.
    (root / "not_a_squad.json").write_text(json.dumps({"bank": 0}), encoding="utf-8")

    if rich:
        _write_events(root)
        _write_importance(root)
        _write_discovery(root)
    return root


@pytest.fixture(scope="session", autouse=True)
def _sealed_environment() -> Iterator[None]:
    """No network, and no model inherited from whoever is running the suite.

    `ServiceConfig.model_path` defaults from `XG_MODEL_PATH`, so a developer
    with that exported would silently test a different prediction path from CI.
    `XG_ALONSO_OFFLINE` is set rather than assumed: CI exports it, a laptop does
    not, and a squad fetch that reached the real FPL API would make
    `/squad/{entry_id}`'s 404 test pass or fail on connectivity.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XG_ALONSO_OFFLINE", "1")
        patch.delenv("XG_MODEL_PATH", raising=False)
        yield


@pytest.fixture(scope="session")
def api_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fully populated data root — the seam the whole suite hangs from.

    When the committed offline fixtures land under `data/fixtures/`, this body
    is what changes; nothing above it is aware of where the tree came from.
    """
    return _build_data_root(tmp_path_factory.mktemp("api-data"), rich=True)


@pytest.fixture(scope="session")
def sparse_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Bronze and silver only. What `xg ingest` alone leaves behind."""
    return _build_data_root(tmp_path_factory.mktemp("api-data-sparse"), rich=False)


@pytest.fixture(scope="session")
def squad_file(api_data_root: Path) -> Path:
    """A legal squad on disk. `entry/{id}/.../picks/` 404s before the deadline."""
    return api_data_root / "squad.json"


@pytest.fixture(scope="session")
def bootstrap_roster(api_data_root: Path) -> dict[str, Any]:
    """The exact payload the service loaded, read back out of bronze.

    Read back rather than rebuilt so a test comparing a response against the
    roster is comparing against what was actually served.
    """
    store = FileSystemBronzeStore(api_data_root / "bronze")
    ref = store.latest(SOURCE_BOOTSTRAP)
    assert ref is not None, "the fixture data root has no bootstrap snapshot"
    payload: dict[str, Any] = json.loads(store.read(ref))
    return payload


@pytest.fixture(scope="session")
def api_service(api_data_root: Path) -> DecisionService:
    """The service over the populated root, built once for the whole session.

    Session-scoped because it is the expensive object: `DecisionService` caches
    predictions, archetypes, situational history and the horizon per gameweek,
    so one instance builds the catalogue once for the whole file. A fresh
    service per test would rebuild it per test and turn a five-second file into
    a several-minute one.
    """
    return DecisionService(ServiceConfig(data_root=api_data_root))


@pytest.fixture(scope="session")
def sparse_service(sparse_data_root: Path) -> DecisionService:
    """The service over the bronze-and-silver-only root."""
    return DecisionService(ServiceConfig(data_root=sparse_data_root))


def _client_for(service: DecisionService) -> Iterator[TestClientType]:
    """A client whose routes resolve to `service`.

    `main._service` is `lru_cache`d, but `dependency_overrides` replaces the
    dependency *before* FastAPI ever resolves it, so the cache is never
    consulted and never populated. That is relied on deliberately rather than
    worked around: clearing the cache would leave the real `_service` reachable
    and a route that forgot its `ServiceDep` would silently read `.data` from
    the developer's working tree.

    **Function-scoped, and it has to be.** `app` is a module singleton and
    `dependency_overrides` is one mutable dict on it, so two session-scoped
    clients would not be two clients — the second to be built would overwrite
    the first's override and every test in the file would silently talk to
    whichever service happened to be set up last. The *service* is what is
    session-scoped; the override around it lives exactly as long as one test.
    """
    from xg_alonso.api.main import _service, app

    app.dependency_overrides[_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(_service, None)


@pytest.fixture
def client(api_service: DecisionService) -> Iterator[TestClientType]:
    """The client for the populated root."""
    yield from _client_for(api_service)


@pytest.fixture
def sparse_client(sparse_service: DecisionService) -> Iterator[TestClientType]:
    """The client for the bronze-and-silver-only root."""
    yield from _client_for(sparse_service)
