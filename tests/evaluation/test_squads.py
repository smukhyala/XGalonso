"""Starting squads, and where each one sits.

The published headline walked one season from one starting squad and never
said whether that was a favourable place to begin. These generators exist so
the question can be answered with a number rather than a shrug, and the tests
here pin the properties that make the number trustworthy: legality, identity
that does not depend on construction order, seeds that are actually used, and
an ownership template that cannot see the week it is choosing for.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from xg_alonso.contracts.evaluation import SquadSource
from xg_alonso.contracts.identifiers import EntryId, GameweekId, PlayerCode
from xg_alonso.domain.rules import SquadRules
from xg_alonso.evaluation.squads import (
    is_legal,
    load_squad_artifact,
    most_expensive_legal_squad,
    random_legal_squads,
    squad_identity,
    template_ownership_squad,
    write_squad_artifact,
)

FIXTURE = Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"
T0 = datetime(2026, 8, 21, tzinfo=UTC)


@pytest.fixture(scope="module")
def rules() -> SquadRules:
    payload: dict[str, Any] = json.loads(FIXTURE.read_text())
    return SquadRules.from_bootstrap(payload, version="2026-27", source_sha256="b" * 64)


@pytest.fixture(scope="module")
def players(rules: SquadRules) -> pl.DataFrame:
    """A pool wide enough that club and budget limits actually bind."""
    rows: list[dict[str, Any]] = []
    code = 1
    for position, count in (("GKP", 12), ("DEF", 30), ("MID", 30), ("FWD", 18)):
        for i in range(count):
            rows.append(
                {
                    "player_code": code,
                    "position": position,
                    "team_id": 1 + (i % 20),
                    "current_price": 40 + (i % 8) * 5,
                    "web_name": f"p{code}",
                }
            )
            code += 1
    return pl.DataFrame(rows)


def _stats(*, gameweek: int, favour: int) -> pl.DataFrame:
    """Ownership over the whole pool, ranked differently per `favour`.

    Every player carries a value. An earlier version gave ownership to eight
    players only, which meant the same eight topped their position groups in
    both scenarios and the ranking never actually drove selection — the
    negative control caught it, which is what negative controls are for.
    """
    codes = list(range(1, 91))
    return pl.DataFrame(
        {
            "player_code": codes,
            "season": ["2026-27"] * len(codes),
            "gameweek_id": [gameweek] * len(codes),
            # A rotation, so `favour` genuinely reorders every position group.
            "selected": [((c * favour) % 97) * 1000 for c in codes],
            "kickoff_time": [T0 + timedelta(days=7 * gameweek)] * len(codes),
        }
    )


class TestMostExpensiveLegal:
    def test_it_produces_a_legal_squad(self, players: pl.DataFrame, rules: SquadRules) -> None:
        squad = most_expensive_legal_squad(
            players, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        assert is_legal(squad, rules=rules)
        assert len(squad.squad.picks) == rules.squad_size

    def test_it_spends_rather_than_banks(self, players: pl.DataFrame, rules: SquadRules) -> None:
        """An earlier version left 36m unspent and made everything beat hold."""
        squad = most_expensive_legal_squad(
            players, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        assert int(squad.squad.bank) < int(rules.total_budget) * 0.05

    def test_it_sets_no_captain(self, players: pl.DataFrame, rules: SquadRules) -> None:
        """The armband is chosen from predictions at decision time, not here."""
        squad = most_expensive_legal_squad(
            players, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        assert not any(p.is_captain for p in squad.squad.picks)

    def test_it_is_deterministic(self, players: pl.DataFrame, rules: SquadRules) -> None:
        a = most_expensive_legal_squad(
            players, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        b = most_expensive_legal_squad(
            players, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        assert a.squad_id == b.squad_id


class TestSquadIdentity:
    def test_it_does_not_depend_on_pick_order(
        self, players: pl.DataFrame, rules: SquadRules
    ) -> None:
        """Otherwise one squad occupies two run directories and pairing breaks."""
        squad = most_expensive_legal_squad(
            players, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(6)
        ).squad
        reordered = squad.model_copy(
            update={
                "picks": tuple(
                    p.model_copy(update={"squad_slot": 16 - p.squad_slot})
                    for p in reversed(squad.picks)
                )
            }
        )
        assert squad_identity(squad) == squad_identity(reordered)

    def test_a_different_player_changes_it(self, players: pl.DataFrame, rules: SquadRules) -> None:
        squad = most_expensive_legal_squad(
            players, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(6)
        ).squad
        swapped = squad.model_copy(
            update={
                "picks": (
                    squad.picks[0].model_copy(update={"player_code": PlayerCode(9999)}),
                    *squad.picks[1:],
                )
            }
        )
        assert squad_identity(squad) != squad_identity(swapped)


class TestRandomLegalSquads:
    def test_every_draw_is_legal(self, players: pl.DataFrame, rules: SquadRules) -> None:
        for squad in random_legal_squads(
            players, rules=rules, seed=7, count=4, entry_id=EntryId(1), gameweek=GameweekId(6)
        ):
            assert is_legal(squad, rules=rules)

    def test_the_same_seed_reproduces_the_squad(
        self, players: pl.DataFrame, rules: SquadRules
    ) -> None:
        a = random_legal_squads(
            players, rules=rules, seed=7, count=1, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        b = random_legal_squads(
            players, rules=rules, seed=7, count=1, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        assert a[0].squad_id == b[0].squad_id

    def test_different_seeds_produce_different_squads(
        self, players: pl.DataFrame, rules: SquadRules
    ) -> None:
        """A generator that ignores its seed makes a cohort of identical squads."""
        a = random_legal_squads(
            players, rules=rules, seed=1, count=1, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        b = random_legal_squads(
            players, rules=rules, seed=2, count=1, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        assert a[0].squad_id != b[0].squad_id

    def test_each_draw_within_a_batch_differs(
        self, players: pl.DataFrame, rules: SquadRules
    ) -> None:
        squads = random_legal_squads(
            players, rules=rules, seed=3, count=5, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        assert len({s.squad_id for s in squads}) == 5


@pytest.mark.leakage
class TestTemplateOwnershipCannotSeeItsOwnWeek:
    def test_it_reads_only_the_previous_gameweek(
        self, players: pl.DataFrame, rules: SquadRules
    ) -> None:
        """Gameweek g's ownership is recorded with g's result, after its deadline.

        Appending it must not change the squad chosen for g — if it does, the
        template knows which players the crowd piled into *because* of what
        happened.
        """
        before = template_ownership_squad(
            players,
            _stats(gameweek=5, favour=3),
            rules=rules,
            season="2026-27",
            gameweek=GameweekId(6),
            entry_id=EntryId(1),
        )
        with_future = template_ownership_squad(
            players,
            pl.concat([_stats(gameweek=5, favour=3), _stats(gameweek=6, favour=11)]),
            rules=rules,
            season="2026-27",
            gameweek=GameweekId(6),
            entry_id=EntryId(1),
        )
        assert before.squad_id == with_future.squad_id

    def test_the_control_the_previous_week_does_move_it(
        self, players: pl.DataFrame, rules: SquadRules
    ) -> None:
        """The negative control: if g-1 ownership did not matter either, the
        test above would pass for the wrong reason."""
        one = template_ownership_squad(
            players,
            _stats(gameweek=5, favour=3),
            rules=rules,
            season="2026-27",
            gameweek=GameweekId(6),
            entry_id=EntryId(1),
        )
        other = template_ownership_squad(
            players,
            _stats(gameweek=5, favour=11),
            rules=rules,
            season="2026-27",
            gameweek=GameweekId(6),
            entry_id=EntryId(1),
        )
        assert one.squad_id != other.squad_id

    def test_it_names_itself_a_substitute(self, players: pl.DataFrame, rules: SquadRules) -> None:
        """Real entry picks are unobtainable; the report must not imply otherwise."""
        squad = template_ownership_squad(
            players,
            _stats(gameweek=5, favour=3),
            rules=rules,
            season="2026-27",
            gameweek=GameweekId(6),
            entry_id=EntryId(1),
        )
        assert "unobtainable" in squad.provenance["caveat"]
        assert squad.source is SquadSource.TEMPLATE_OWNERSHIP


class TestArtifactRoundTrip:
    def test_a_squad_survives_a_write_and_a_read(
        self, players: pl.DataFrame, rules: SquadRules, tmp_path: Path
    ) -> None:
        """Solver-built squads are reproducible from an artifact, not a seed."""
        original = most_expensive_legal_squad(
            players, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(6)
        )
        path = tmp_path / "squad.json"
        write_squad_artifact(original, path)
        restored = load_squad_artifact(path)

        assert restored.squad_id == original.squad_id
        assert restored.squad == original.squad
        assert restored.source is original.source
