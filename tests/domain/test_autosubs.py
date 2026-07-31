"""Automatic substitutions and captaincy.

Every constant is read from the pinned bootstrap snapshot — squad sizes,
positional minima, the captain multiplier. A test that hardcodes ``3`` for the
defender minimum can only ever agree with an implementation that hardcodes the
same ``3``, so it would confirm a shared misconception rather than catch one.
That is the rule `CLAUDE.md` states for the source; it applies at least as much
to the tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xg_alonso.contracts.identifiers import PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.simulation import CaptainSource, SkipReason
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.domain.autosubs import apply_autosubs, formation_of, played, resolve_captaincy
from xg_alonso.domain.constraints import check_starting_xi
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringThresholds

FIXTURE = Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def rules(payload: dict[str, Any]) -> SquadRules:
    return SquadRules.from_bootstrap(payload, version="2026-27", source_sha256="b" * 64)


@pytest.fixture(scope="module")
def multiplier() -> int:
    return ScoringThresholds().captain_multiplier


def _pick(code: int, position: Position, slot: int, *, team: int = 1) -> SquadPick:
    return SquadPick(
        player_code=PlayerCode(code),
        position=position,
        team_id=TeamId(team),
        purchase_price=TenthsOfMillion(50),
        current_price=TenthsOfMillion(50),
        selling_price=TenthsOfMillion(50),
        squad_slot=slot,
    )


def _xi(shape: tuple[int, int, int, int], *, start: int = 1) -> list[SquadPick]:
    """An eleven in the given ``(GKP, DEF, MID, FWD)`` shape, slots 1..11.

    Clubs are spread so the three-per-club cap never interferes with a test
    that is about formation legality.
    """
    order = (
        [Position.GKP] * shape[0]
        + [Position.DEF] * shape[1]
        + [Position.MID] * shape[2]
        + [Position.FWD] * shape[3]
    )
    return [_pick(start + i, position, i + 1, team=1 + (i % 5)) for i, position in enumerate(order)]


def _bench(positions: list[Position], *, start: int = 100) -> list[SquadPick]:
    return [
        _pick(start + i, position, 12 + i, team=1 + (i % 5)) for i, position in enumerate(positions)
    ]


def _all_played(picks: list[SquadPick]) -> dict[PlayerCode, int]:
    return {pick.player_code: 90 for pick in picks}


class TestPlayed:
    def test_a_missing_row_means_no_minutes(self) -> None:
        """Absence is zero, not unknown — otherwise blanks disable autosubs."""
        assert not played({}, PlayerCode(1))

    def test_zero_minutes_is_not_playing(self) -> None:
        assert not played({PlayerCode(1): 0}, PlayerCode(1))

    def test_one_minute_is_playing(self) -> None:
        assert played({PlayerCode(1): 1}, PlayerCode(1))


class TestScenario1CaptainOutViceIn:
    def test_the_vice_inherits_the_armband(self, rules: SquadRules, multiplier: int) -> None:
        xi = _xi((1, 4, 4, 2))
        captain, vice = xi[4], xi[9]
        minutes = _all_played(xi)
        minutes[captain.player_code] = 0

        result = apply_autosubs(starters=xi, bench=[], minutes=minutes, rules=rules)
        captaincy = resolve_captaincy(
            captain=captain.player_code,
            vice_captain=vice.player_code,
            final_xi=result.final_xi,
            minutes=minutes,
            rules=rules,
            multiplier=multiplier,
        )

        assert captaincy.source is CaptainSource.VICE_CAPTAIN
        assert captaincy.holder == vice.player_code
        assert captaincy.multiplier == multiplier


class TestScenario2CaptainAndViceBothOut:
    def test_nobody_is_doubled(self, rules: SquadRules, multiplier: int) -> None:
        xi = _xi((1, 4, 4, 2))
        captain, vice = xi[4], xi[9]
        minutes = _all_played(xi)
        minutes[captain.player_code] = 0
        minutes[vice.player_code] = 0

        captaincy = resolve_captaincy(
            captain=captain.player_code,
            vice_captain=vice.player_code,
            final_xi=tuple(p.player_code for p in xi),
            minutes=minutes,
            rules=rules,
            multiplier=multiplier,
        )

        assert captaincy.source is CaptainSource.NONE
        assert captaincy.holder is None
        assert captaincy.multiplier == 1

    def test_the_replacement_is_not_doubled(self, rules: SquadRules, multiplier: int) -> None:
        """FPL doubles nobody. It does not promote the substitute."""
        xi = _xi((1, 4, 4, 2))
        captain, vice = xi[4], xi[9]
        bench = _bench([Position.MID])
        minutes = _all_played(xi) | _all_played(bench)
        minutes[captain.player_code] = 0
        minutes[vice.player_code] = 0

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)
        captaincy = resolve_captaincy(
            captain=captain.player_code,
            vice_captain=vice.player_code,
            final_xi=result.final_xi,
            minutes=minutes,
            rules=rules,
            multiplier=multiplier,
        )

        assert bench[0].player_code in result.final_xi
        assert captaincy.holder is None


class TestScenario3BenchOrderDecidesLegality:
    """Same minutes, same points — only the bench order differs."""

    @staticmethod
    def _setup(rules: SquadRules) -> tuple[list[SquadPick], dict[PlayerCode, int]]:
        minimum = rules.rule_for(Position.DEF).min_play
        xi = _xi((1, minimum, 5, 2))
        minutes = _all_played(xi)
        defenders = [p for p in xi if p.position is Position.DEF]
        minutes[defenders[0].player_code] = 0
        minutes[defenders[1].player_code] = 0
        return xi, minutes

    def test_a_mismatched_bench_can_cover_only_one(self, rules: SquadRules) -> None:
        xi, minutes = self._setup(rules)
        bench = _bench([Position.MID, Position.DEF, Position.FWD, Position.GKP])
        minutes |= _all_played(bench)

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)

        assert len(result.substitutions) == 1
        assert result.substitutions[0].player_on == bench[1].player_code
        assert {s.reason for s in result.skipped} == {SkipReason.FORMATION_WOULD_BE_ILLEGAL}
        assert len(result.final_xi) == rules.starting_size

    def test_a_matched_bench_covers_both(self, rules: SquadRules) -> None:
        xi, minutes = self._setup(rules)
        bench = _bench([Position.DEF, Position.DEF, Position.MID, Position.GKP])
        minutes |= _all_played(bench)

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)

        assert len(result.substitutions) == 2
        by_code = {p.player_code: p for p in xi + bench}
        final = [by_code[c] for c in result.final_xi]
        assert formation_of(final) == (1, rules.rule_for(Position.DEF).min_play, 5, 2)
        assert check_starting_xi(final, rules=rules) == []


class TestScenario4GoalkeeperAutosub:
    def test_only_the_reserve_keeper_can_cover_the_keeper(self, rules: SquadRules) -> None:
        xi = _xi((1, 4, 4, 2))
        keeper = xi[0]
        bench = _bench([Position.MID, Position.DEF, Position.FWD, Position.GKP])
        minutes = _all_played(xi) | _all_played(bench)
        minutes[keeper.player_code] = 0

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)

        assert len(result.substitutions) == 1
        assert result.substitutions[0].player_on == bench[3].player_code
        assert result.substitutions[0].player_off == keeper.player_code

    def test_the_keeper_rule_uses_no_special_case(self, rules: SquadRules) -> None:
        """The outfielders are refused for the *same* reason a defender is.

        If the implementation branched on position, the keeper case would need
        its own skip reason. It does not, and this asserts that.
        """
        xi = _xi((1, 4, 4, 2))
        bench = _bench([Position.MID, Position.DEF, Position.FWD, Position.GKP])
        minutes = _all_played(xi) | _all_played(bench)
        minutes[xi[0].player_code] = 0

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)
        refused = {s.player_on: s.reason for s in result.skipped}

        assert refused[bench[0].player_code] is SkipReason.FORMATION_WOULD_BE_ILLEGAL
        assert refused[bench[1].player_code] is SkipReason.FORMATION_WOULD_BE_ILLEGAL
        assert refused[bench[2].player_code] is SkipReason.FORMATION_WOULD_BE_ILLEGAL

    def test_a_reserve_keeper_who_also_missed_changes_nothing(self, rules: SquadRules) -> None:
        xi = _xi((1, 4, 4, 2))
        bench = _bench([Position.MID, Position.DEF, Position.FWD, Position.GKP])
        minutes = _all_played(xi) | _all_played(bench)
        minutes[xi[0].player_code] = 0
        minutes[bench[3].player_code] = 0

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)
        refused = {s.player_on: s.reason for s in result.skipped}

        assert result.substitutions == ()
        assert xi[0].player_code in result.final_xi
        assert refused[bench[3].player_code] is SkipReason.BENCH_PLAYER_DID_NOT_PLAY


class TestScenario11BenchPlayerDidNotPlay:
    def test_the_first_available_substitute_comes_on(self, rules: SquadRules) -> None:
        xi = _xi((1, 3, 5, 2))
        vacancy = next(p for p in xi if p.position is Position.MID)
        bench = _bench([Position.MID, Position.MID])
        minutes = _all_played(xi) | _all_played(bench)
        minutes[vacancy.player_code] = 0
        minutes[bench[0].player_code] = 0

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)

        assert result.skipped[0].reason is SkipReason.BENCH_PLAYER_DID_NOT_PLAY
        assert result.substitutions[0].player_on == bench[1].player_code


class TestScenario12MultipleAutosubPaths:
    """A substitute may cross positions when the resulting shape is legal."""

    def test_a_defender_covers_a_midfielder(self, rules: SquadRules) -> None:
        xi = _xi((1, 3, 5, 2))
        vacancy = next(p for p in xi if p.position is Position.MID)
        bench = _bench([Position.DEF, Position.MID, Position.FWD])
        minutes = _all_played(xi) | _all_played(bench)
        minutes[vacancy.player_code] = 0

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)
        by_code = {p.player_code: p for p in xi + bench}
        final = [by_code[c] for c in result.final_xi]

        assert result.substitutions[0].player_on == bench[0].player_code
        assert formation_of(final) == (1, 4, 4, 2)
        assert check_starting_xi(final, rules=rules) == []

    def test_the_result_does_not_depend_on_starter_order(self, rules: SquadRules) -> None:
        xi = _xi((1, 3, 5, 2))
        vacancy = next(p for p in xi if p.position is Position.MID)
        bench = _bench([Position.DEF, Position.MID, Position.FWD])
        minutes = _all_played(xi) | _all_played(bench)
        minutes[vacancy.player_code] = 0

        forward = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)
        reversed_ = apply_autosubs(
            starters=list(reversed(xi)), bench=bench, minutes=minutes, rules=rules
        )

        assert set(forward.final_xi) == set(reversed_.final_xi)
        assert forward.substitutions == reversed_.substitutions


class TestEveryStarterPlayed:
    def test_no_substitution_and_every_bench_player_says_why(self, rules: SquadRules) -> None:
        xi = _xi((1, 4, 4, 2))
        bench = _bench([Position.MID, Position.DEF, Position.FWD, Position.GKP])
        minutes = _all_played(xi) | _all_played(bench)

        result = apply_autosubs(starters=xi, bench=bench, minutes=minutes, rules=rules)

        assert result.substitutions == ()
        assert {s.reason for s in result.skipped} == {SkipReason.NO_VACANCY}
        assert set(result.final_xi) == {p.player_code for p in xi}


class TestViceCaptaincyCanBeDisabled:
    def test_the_rule_is_read_not_assumed(self, rules: SquadRules, multiplier: int) -> None:
        """`sys_vice_captain_enabled` is published; the code used to assume it."""
        xi = _xi((1, 4, 4, 2))
        captain, vice = xi[4], xi[9]
        minutes = _all_played(xi)
        minutes[captain.player_code] = 0

        disabled = rules.model_copy(update={"vice_captain_enabled": False})
        captaincy = resolve_captaincy(
            captain=captain.player_code,
            vice_captain=vice.player_code,
            final_xi=tuple(p.player_code for p in xi),
            minutes=minutes,
            rules=disabled,
            multiplier=multiplier,
        )

        assert captaincy.source is CaptainSource.NONE
