"""The committed demo fixtures, and the guarantees a stranger's clone relies on.

These are the only data files in the repository that anything other than a test
reads, so they need the protections a dataset gets rather than the ones a test
constant gets: a size ceiling, a provenance record that matches the bytes, a
schema that still agrees with the pipeline, and a demonstration that a legal
squad can actually be assembled from them.

All marked ``golden``. The marker was declared in ``pyproject.toml`` and used by
nothing; a committed fixture comparison is exactly what it was declared for.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import build_demo_fixture
import polars as pl
import pytest
from typer.testing import CliRunner

from xg_alonso.cli.main import _demo_lock, _pool_wiring_gaps, app
from xg_alonso.contracts.identifiers import PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.domain.constraints import check_squad
from xg_alonso.domain.rules import SquadRules
from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "data" / "fixtures"

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def provenance() -> dict[str, Any]:
    sidecar = FIXTURE_ROOT / build_demo_fixture.PROVENANCE_NAME
    if not sidecar.exists():
        pytest.fail(f"no {sidecar}; run `make fixtures`")
    return dict(json.loads(sidecar.read_text()))


@pytest.fixture(scope="module")
def fixture_stats() -> pl.DataFrame:
    return pl.read_parquet(FIXTURE_ROOT / "silver" / "player_gameweek_stats.parquet")


@pytest.fixture(scope="module")
def fixture_history() -> pl.DataFrame:
    return pl.read_parquet(FIXTURE_ROOT / "silver" / "players_history.parquet")


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def test_the_size_ceiling_matches_the_pre_commit_hook() -> None:
    """The constant in the generator is the hook's number, not a guess at it.

    If someone relaxes ``--maxkb`` and the generator does not follow, fixtures
    grow past a limit this suite still believes in; if someone tightens it, the
    committed fixtures start failing on a contributor's machine and nowhere
    else. Reading both is the only way the two cannot drift apart.
    """
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    match = re.search(r"--maxkb=(\d+)", config)
    assert match is not None, "check-added-large-files no longer sets --maxkb"
    assert int(match.group(1)) * 1024 == build_demo_fixture.MAX_FIXTURE_BYTES


def test_no_committed_fixture_exceeds_the_ceiling() -> None:
    oversized = {
        path.relative_to(FIXTURE_ROOT).as_posix(): path.stat().st_size
        for path in sorted(FIXTURE_ROOT.rglob("*"))
        if path.is_file() and path.stat().st_size > build_demo_fixture.MAX_FIXTURE_BYTES
    }
    assert oversized == {}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_fixtures_match_their_recorded_hashes_and_sizes() -> None:
    """Raw data is immutable: an edited fixture must be detectable, not silent."""
    assert build_demo_fixture.check(FIXTURE_ROOT) == []


def test_provenance_records_what_the_data_philosophy_requires(
    provenance: dict[str, Any],
) -> None:
    assert provenance["generator"] == "tools/build_demo_fixture.py"
    assert provenance["generator_version"] == build_demo_fixture.GENERATOR_VERSION
    assert provenance["seasons"] == list(build_demo_fixture.FIXTURE_SEASONS)
    assert provenance["seed"] > 0
    # A timestamped snapshot, per the Data Philosophy section of CLAUDE.md.
    assert provenance["extracted_at"].endswith("+00:00")
    for record in provenance["files"]:
        assert record["derived_from"], f"{record['path']} records no source"


def test_the_sample_is_labelled_as_a_sample(provenance: dict[str, Any]) -> None:
    """The warning is load-bearing: it is what the demo banner is built from."""
    assert "not evidence about football" in provenance["warning"].lower()


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_gameweek_stats_fixture_still_matches_the_silver_schema(
    fixture_stats: pl.DataFrame,
) -> None:
    """Detects a fixture left behind by a schema change.

    Compared as an ordered mapping of name to dtype rather than as a set of
    names: a column silently re-typed from ``Int64`` to ``Float64`` would keep
    every name test green while changing what the pipeline computes.
    """
    expected = {name: str(dtype) for name, dtype in PLAYER_GAMEWEEK_STATS_SCHEMA.items()}
    actual = {name: str(dtype) for name, dtype in fixture_stats.schema.items()}
    assert actual == expected


def test_the_sidecar_detects_drift_without_re_deriving_the_fixture(
    provenance: dict[str, Any],
) -> None:
    """A fixture goes stale when the code around it moves, not when it moves.

    Both digests are recorded at extraction. A mismatch means the silver
    contract or the feature catalogue changed after these bytes were cut, and
    that the fixture — and this whole suite's idea of what the demo proves —
    needs a look. Neither hash requires reading the parquet, so this stays the
    cheapest check in the file and the first one to go red.
    """
    from xg_alonso.features.schema import catalogue_hash

    assert provenance["silver_schema_sha256"] == build_demo_fixture.silver_schema_digest()
    assert provenance["catalogue_hash"] == catalogue_hash()


def test_the_fixture_supports_the_whole_current_feature_catalogue(
    fixture_stats: pl.DataFrame,
) -> None:
    """The functional half of the staleness check.

    A digest says the catalogue moved; this says whether the move broke
    anything. Run over one short window, because the question is whether every
    declared feature can be *computed* from these columns, not how good it is.
    """
    from xg_alonso.features.schema import model_feature_names
    from xg_alonso.prediction import build_training_frame

    data = build_training_frame(fixture_stats, seasons=["2024-25"], min_gameweek=30)
    missing = [name for name in model_feature_names() if name not in data.frame.columns]
    assert missing == []


def test_players_history_fixture_carries_every_column_the_cli_reads(
    fixture_history: pl.DataFrame,
) -> None:
    """The contract `build-discovery-frame` and `discover` depend on.

    Not a copy of the producing function's output shape — a contract stated
    from the consumer's side, so widening the producer stays free and dropping
    something a command needs does not.
    """
    required = {"player_code", "season", "web_name", "position", "team_name"}
    assert required <= set(fixture_history.columns)


def test_pinned_rules_fixture_is_a_snapshot_not_a_transcription() -> None:
    """The demo must load scoring from a pinned payload, as every command does.

    CLAUDE.md is emphatic that scoring values are never Python literals, so the
    check is structural: the fixture has to be a real ``game_config`` payload
    the loader can parse, rather than a hand-written stand-in that happens to
    contain the right numbers.
    """
    from xg_alonso.cli.main import _load_pinned_rules
    from xg_alonso.contracts.identifiers import parse_season

    snapshot = json.loads((FIXTURE_ROOT / "pinned" / "rules_2026-27.json").read_text())
    # The three things a pinned snapshot is: the payload, when it was fetched,
    # and the digest the drift check compares against.
    assert set(snapshot) == {"payload", "fetched_at", "source_sha256"}
    assert snapshot["fetched_at"].endswith("+00:00")
    assert len(snapshot["source_sha256"]) == 64
    published = snapshot["payload"]["game_config"]["scoring"]

    scoring, squad = _load_pinned_rules(FIXTURE_ROOT, parse_season("2026-27"))
    assert scoring is not None
    assert squad is not None
    # Compared against the payload on both sides. The point is that the loader
    # and the file agree — not what any particular number is. A literal here
    # would be the exact transcription CLAUDE.md forbids, and the goalkeeper
    # goal is the constant it names as the worked example.
    for label, position in (
        ("GKP", Position.GKP),
        ("DEF", Position.DEF),
        ("MID", Position.MID),
        ("FWD", Position.FWD),
    ):
        assert scoring.goals_scored[position] == published["goals_scored"][label]


# ---------------------------------------------------------------------------
# Usability — can these fixtures actually do the job?
# ---------------------------------------------------------------------------


def _cheapest_legal_15(pool: pl.DataFrame, *, rules: SquadRules) -> list[SquadPick]:
    """Greedily fill the positional quotas from the cheapest players.

    Quotas and the per-club cap are read off ``rules`` — which came from the
    pinned snapshot — rather than written out here, so this stays a test of the
    fixture rather than a second copy of the rulebook.
    """
    # The silver table labels goalkeepers ``GK``; the contract's enum says
    # ``GKP``. One mapping, at the boundary where the two meet.
    labels = {"GK": Position.GKP, "DEF": Position.DEF, "MID": Position.MID, "FWD": Position.FWD}
    quota = {rule.position: rule.squad_select for rule in rules.positions}
    picks: list[SquadPick] = []
    per_club: dict[str, int] = {}
    slot = 0

    for label, position in labels.items():
        needed = quota[position]
        taken = 0
        for row in pool.filter(pl.col("position") == label).sort("price").iter_rows(named=True):
            if per_club.get(row["team_name"], 0) >= rules.max_per_club:
                continue
            slot += 1
            taken += 1
            per_club[row["team_name"]] = per_club.get(row["team_name"], 0) + 1
            price = TenthsOfMillion(int(row["price"]))
            picks.append(
                SquadPick(
                    player_code=PlayerCode(int(row["player_code"])),
                    position=position,
                    team_id=TeamId(int(row["team_index"])),
                    purchase_price=price,
                    current_price=price,
                    selling_price=price,
                    squad_slot=slot,
                )
            )
            if taken == needed:
                break
        assert taken == needed, f"only {taken} of {needed} {label} available in the fixture"
    return picks


@pytest.mark.parametrize("season", build_demo_fixture.FIXTURE_SEASONS)
def test_a_legal_15_can_be_filled_from_the_sample(
    season: str, fixture_stats: pl.DataFrame, fixture_history: pl.DataFrame
) -> None:
    """The reason the sample is stratified, asserted rather than assumed.

    A uniform sample of the same size fits under the ceiling just as well and
    cannot do this — it lands short in whichever position happens to be scarce,
    or piles into three clubs. Checked with the real squad rules from the real
    pinned snapshot, not with a local reimplementation of them.
    """
    from xg_alonso.cli.main import _load_pinned_rules
    from xg_alonso.contracts.identifiers import parse_season

    _, squad_rules = _load_pinned_rules(FIXTURE_ROOT, parse_season("2026-27"))
    assert squad_rules is not None

    prices = (
        fixture_stats.filter(pl.col("season") == season)
        .sort("gameweek_id")
        .group_by(["player_code", "season"])
        .agg(pl.col("value").last().alias("price"))
    )
    pool = fixture_history.join(prices, on=["player_code", "season"], how="inner").with_columns(
        # SquadPick wants an integer club id; the silver table names clubs, so
        # index them stably. Only distinctness matters for the max-per-club rule.
        pl.col("team_name").rank("dense").cast(pl.Int64).alias("team_index")
    )

    picks = _cheapest_legal_15(pool, rules=squad_rules)
    assert len(picks) == 15

    violations = check_squad(picks, rules=squad_rules, budget=squad_rules.total_budget)
    assert violations == [], [f"{v.rule}: {v.detail}" for v in violations]


def test_the_sample_spans_enough_clubs_for_the_three_per_club_rule(
    fixture_stats: pl.DataFrame, fixture_history: pl.DataFrame
) -> None:
    """Fifteen players at three per club needs five clubs; a demo needs many more."""
    sampled = fixture_history.join(
        fixture_stats.select("player_code", "season").unique(),
        on=["player_code", "season"],
        how="semi",
    )
    for season in build_demo_fixture.FIXTURE_SEASONS:
        clubs = sampled.filter(pl.col("season") == season)["team_name"].n_unique()
        assert clubs >= 18, f"{season} covers only {clubs} clubs"


# ---------------------------------------------------------------------------
# The friendly guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["build-discovery-frame"], "xg backfill"),
        (["discover", "anything"], "xg build-discovery-frame"),
        (["train"], "xg backfill"),
        (["backtest"], "xg backfill"),
        (["score"], "xg backfill"),
        (["importance"], "xg backfill"),
    ],
)
def test_a_missing_table_names_the_command_that_writes_it(
    command: list[str], expected: str, tmp_path: Path
) -> None:
    """No bare FileNotFoundError anywhere a first run can reach.

    ``build-discovery-frame`` read two parquet files with no check at all, so a
    clean clone met a traceback whose only content was a path.
    """
    result = CliRunner().invoke(app, [*command, "--data-root", str(tmp_path / "empty")])
    assert result.exit_code == 1, result.output
    assert expected in result.output
    assert not isinstance(result.exception, FileNotFoundError)


# ---------------------------------------------------------------------------
# The two-manager demo
# ---------------------------------------------------------------------------


def test_pool_wiring_is_detected_rather_than_assumed() -> None:
    """`discover-demo` must notice when the feasible-pool phase lands.

    Hard-coding "not yet" would keep the command refusing after the dependency
    arrived, and nobody would find out until they read the source.
    """
    without = _pool_wiring_gaps(pl.DataFrame({"player_code": [1]}))
    assert any("price_tenths" in gap for gap in without)
    assert any("team_id" in gap for gap in without)

    with_columns = _pool_wiring_gaps(
        pl.DataFrame({"player_code": [1], "price_tenths": [50], "team_id": [3]})
    )
    assert not any("price_tenths" in gap for gap in with_columns)


def test_discover_demo_refuses_loudly_when_the_pool_is_not_wired(tmp_path: Path) -> None:
    """A refusal, not a fabricated difference.

    Two managers measured over the same global population *can* reach different
    verdicts through objective compilation alone. Reporting that as "different
    constraints, different features" would be the wrong cause attached to a
    real effect, which is the failure mode this whole command exists to avoid.

    Passes either way once the dependency lands: a wired build runs the two
    managers and exits 0 only if their verdicts genuinely differ.
    """
    demo_root = tmp_path / "demo"
    (demo_root / "gold").mkdir(parents=True)
    # A stub frame so the probe is reached without paying for a real build.
    pl.DataFrame({"player_code": [1]}).write_parquet(
        demo_root / "gold" / "discovery_training.parquet"
    )

    result = CliRunner().invoke(app, ["discover-demo", "--demo-root", str(demo_root)])

    if result.exit_code == 2:
        assert "NOT YET WIRED" in result.output
        assert "price_tenths" in result.output
        return
    assert result.exit_code == 0, result.output
    assert "manager A only" in result.output


# ---------------------------------------------------------------------------
# The demo itself
# ---------------------------------------------------------------------------

#: Measured at ~75s end to end on an idle 10-core laptop: 0.1s features, 17s
#: train, 49s discover (with controls), 1s recommend, the rest interpreter
#: start. The ceiling is four times that, to catch the change that turns a
#: stranger's first run into a coffee break — a fold count creeping back up, a
#: control set that stops being sampled.
DEMO_SECONDS_CEILING = 300.0

#: Above this load per core, the wall-clock number measures the machine rather
#: than the demo and the assertion is skipped.
#:
#: Not a hedge — a correction. This assertion first ran on a box carrying five
#: concurrent test suites at load 65 across 10 cores, and the demo took 485s
#: for reasons that had nothing to do with this repository. A timing test that
#: fails on contention teaches people to ignore it, which costs more than the
#: regression it was meant to catch. Load is sampled *before* the run, so a
#: measurement is either taken on a quiet machine or not claimed at all.
MAX_LOAD_PER_CORE = 1.5


def test_a_second_demo_is_refused_rather_than_allowed_to_collide(tmp_path: Path) -> None:
    """The default scratch root is shared, so two runs must not both have it.

    Found the hard way: two demos ran at once, one called ``--fresh`` and
    deleted the tree the other was reading, and the second died on a
    ``FileNotFoundError`` for a discovery table that had existed a moment
    before. Nothing about that traceback would have told the reader what
    happened.
    """
    demo_root = tmp_path / "demo"
    demo_root.mkdir()

    with _demo_lock(demo_root):
        result = CliRunner().invoke(app, ["demo", "--demo-root", str(demo_root)])

    assert result.exit_code == 1, result.output
    assert "Another demo is already using" in result.output

    # ...and the claim is released afterwards, or the first run would poison
    # every later one on the same machine.
    with _demo_lock(demo_root):
        pass


@pytest.mark.e2e
def test_demo_runs_offline_from_committed_fixtures_alone(tmp_path: Path) -> None:
    """The whole promise: clone, install, run, see a recommendation.

    Run as a subprocess with ``XG_ALONSO_OFFLINE=1`` and a scratch root under
    ``tmp_path``, so the test proves the offline path over the committed
    fixtures rather than quietly succeeding because this machine happens to
    have a populated ``.data``.
    """
    demo_root = tmp_path / "demo"
    load_before = os.getloadavg()[0] / (os.cpu_count() or 1)

    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "xg_alonso.cli.main",
            "demo",
            "--demo-root",
            str(demo_root),
            "--fixtures",
            str(FIXTURE_ROOT),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "XG_ALONSO_OFFLINE": "1"},
        # Generous relative to the ceiling: a contended machine is allowed to be
        # slow, it is just not allowed to have its slowness recorded as a
        # regression. The functional assertions below still have to pass.
        timeout=DEMO_SECONDS_CEILING * 8,
        check=False,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    output = completed.stdout

    # The banner is the first thing printed. A demo that shows a confident
    # recommendation before saying what it is standing on is the failure mode.
    assert output.splitlines()[0].startswith("=== FIXTURE DATA")
    assert "not evidence about football" in output.lower()

    # Every stage ran.
    for number, stage in enumerate(("build-features", "train", "discover", "recommend"), start=1):
        assert f"{number}/4  {stage}" in output

    # ...and the ones that matter produced real content, not just a heading.
    assert "Verdicts" in output, "the discovery loop produced no verdicts"
    assert "TRANSFER — GW" in output, "no recommendation was rendered"

    # Everything was written into the scratch root, and nothing into `.data`.
    assert (demo_root / "gold" / "discovery_training.parquet").exists()
    assert (demo_root / "models" / "component_models.pkl").exists()
    assert ".data/" not in output

    # The timing claim, made only when it can be made honestly.
    if load_before > MAX_LOAD_PER_CORE:
        pytest.skip(
            f"load was {load_before:.1f} per core before the run "
            f"(ceiling {MAX_LOAD_PER_CORE}); the demo took {elapsed:.0f}s, which "
            "measures the machine rather than the pipeline"
        )
    assert elapsed < DEMO_SECONDS_CEILING, (
        f"the demo took {elapsed:.0f}s against a {DEMO_SECONDS_CEILING:.0f}s ceiling, "
        f"on a machine at {load_before:.1f} load per core"
    )
