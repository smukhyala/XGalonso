"""The decomposed squad score.

``score_squad`` returned a single number, which is enough to rank two policies
and not enough to understand either. These tests pin the parts and, more
importantly, pin that the parts still sum to the whole.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.simulation import CaptainSource
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.evaluation.backtest import score_squad
from xg_alonso.evaluation.simulator import simulate_squad

FIXTURE = Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def rules(payload: dict[str, Any]) -> SquadRules:
    return SquadRules.from_bootstrap(payload, version="2026-27", source_sha256="b" * 64)


def _squad(rules: SquadRules) -> SquadState:
    """Slots 1-11 are a legal 1-4-4-2; 12-15 are MID, DEF, FWD, GKP."""
    layout = (
        [Position.GKP]
        + [Position.DEF] * 4
        + [Position.MID] * 4
        + [Position.FWD] * 2
        + [Position.MID, Position.DEF, Position.FWD, Position.GKP]
    )
    picks = [
        SquadPick(
            player_code=PlayerCode(i + 1),
            position=position,
            team_id=TeamId(1 + i // rules.max_per_club),
            purchase_price=TenthsOfMillion(50),
            current_price=TenthsOfMillion(50),
            selling_price=TenthsOfMillion(50),
            squad_slot=i + 1,
            is_captain=(i == 4),
            is_vice_captain=(i == 9),
        )
        for i, position in enumerate(layout)
    ]
    return SquadState(
        entry_id=EntryId(1),
        gameweek=GameweekId(5),
        picks=tuple(picks),
        bank=TenthsOfMillion(0),
        free_transfers=1,
    )


class TestTheDecompositionSums:
    def test_parts_equal_the_total(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        points = {p.player_code: 3 for p in squad.picks}
        minutes = {p.player_code: 90 for p in squad.picks}

        score = simulate_squad(squad, points, rules=rules, minutes=minutes)

        assert score.starters_points + score.autosub_points + score.captaincy_points == score.total

    def test_bench_points_are_never_in_the_total(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        points = {p.player_code: 0 for p in squad.picks}
        for pick in squad.bench:
            points[pick.player_code] = 20
        minutes = {p.player_code: 90 for p in squad.picks}

        score = simulate_squad(squad, points, rules=rules, minutes=minutes)

        assert score.bench_points == 80
        assert score.total == 0


class TestAutosubsAreOffWithoutMinutes:
    def test_omitting_minutes_reproduces_the_outcome_blind_score(self, rules: SquadRules) -> None:
        """Every caller that has not been given minutes must be unaffected."""
        squad = _squad(rules)
        points = {p.player_code: 4 for p in squad.picks}
        points[squad.picks[0].player_code] = 0  # the keeper blanked

        blind = simulate_squad(squad, points, rules=rules)
        assert blind.substitutions == ()
        assert blind.captaincy.source is CaptainSource.CAPTAIN

    def test_score_squad_still_returns_the_total(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        points = {p.player_code: 2 for p in squad.picks}

        assert score_squad(squad, points) == simulate_squad(squad, points).total


class TestAutosubsChangeTheTotal:
    def test_a_substitute_adds_his_points(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        starter = squad.starters[5]  # a midfielder
        substitute = squad.bench[0]  # slot 12, also a midfielder

        points = {p.player_code: 1 for p in squad.picks}
        points[starter.player_code] = 0
        points[substitute.player_code] = 9
        minutes = {p.player_code: 90 for p in squad.picks}
        minutes[starter.player_code] = 0

        score = simulate_squad(squad, points, rules=rules, minutes=minutes)

        assert score.autosub_points == 9
        assert substitute.player_code in score.final_xi
        assert starter.player_code not in score.final_xi

    def test_the_vice_is_doubled_when_the_captain_blanks(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        captain = next(p for p in squad.picks if p.is_captain)
        vice = next(p for p in squad.picks if p.is_vice_captain)

        points = {p.player_code: 1 for p in squad.picks}
        points[captain.player_code] = 0
        points[vice.player_code] = 7
        minutes = {p.player_code: 90 for p in squad.picks}
        minutes[captain.player_code] = 0

        score = simulate_squad(squad, points, rules=rules, minutes=minutes)

        assert score.captaincy.source is CaptainSource.VICE_CAPTAIN
        assert score.captaincy_points == 7
