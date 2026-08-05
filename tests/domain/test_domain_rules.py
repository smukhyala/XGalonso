"""Domain rule tests, run against a real (reduced) FPL payload.

These deliberately assert against the committed fixture rather than against
constants written here. A test that hardcodes its expectation reproduces
whatever misconception the implementation has — which is precisely the failure
mode that makes a goalkeeper goal worth 6 points in most FPL models.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from tests.conftest import BOOTSTRAP_FIXTURE, RULES_FETCHED_AT

from xg_alonso.contracts import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerCode,
    Position,
    SquadPick,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.domain import (
    ScoringRules,
    SquadRules,
    assemble_points,
    check_squad,
    check_starting_xi,
    selling_price,
)

NOW = RULES_FETCHED_AT


@pytest.fixture(scope="module")
def payload(bootstrap_payload: dict[str, Any]) -> dict[str, Any]:
    """The pinned snapshot. Shared and read-only; deep-copy before mutating."""
    return bootstrap_payload


@pytest.fixture(scope="module")
def scoring(scoring_rules: ScoringRules) -> ScoringRules:
    """Named `scoring` here because that is what this module's tests ask for."""
    return scoring_rules


class TestScoringRulesFromPayload:
    def test_goalkeeper_goal_is_ten_not_six(self, scoring: ScoringRules) -> None:
        """The value nearly every community model gets wrong.

        Guarding it explicitly because the wrong answer (6) is the intuitive one
        and would never be questioned in review.
        """
        assert scoring.goals_scored[Position.GKP] == 10

    def test_position_scaled_values_match_the_payload(
        self, scoring: ScoringRules, payload: dict[str, Any]
    ) -> None:
        published = payload["game_config"]["scoring"]
        for position in Position:
            assert scoring.goals_scored[position] == published["goals_scored"][position.value]
            assert scoring.clean_sheets[position] == published["clean_sheets"][position.value]

    def test_flat_values_match_the_payload(
        self, scoring: ScoringRules, payload: dict[str, Any]
    ) -> None:
        published = payload["game_config"]["scoring"]
        assert scoring.assists == published["assists"]
        assert scoring.yellow_cards == published["yellow_cards"]
        assert scoring.red_cards == published["red_cards"]
        assert scoring.short_play == published["short_play"]
        assert scoring.long_play == published["long_play"]

    def test_missing_scoring_block_fails_loudly(self) -> None:
        """Falling back to transcribed defaults is the failure being prevented."""
        with pytest.raises(KeyError, match="never be assumed"):
            ScoringRules.from_bootstrap(
                {"game_config": {}}, version="x", source_sha256="c" * 64, fetched_at=NOW
            )

    def test_thresholds_are_marked_for_verification(self, scoring: ScoringRules) -> None:
        """FPL does not publish these, so they are asserted by us and must be flagged."""
        for name, field in type(scoring.thresholds).model_fields.items():
            assert field.description is not None
            assert "VERIFY" in field.description, f"{name} must be marked for verification"

    def test_defensive_contribution_threshold_differs_by_position(
        self, scoring: ScoringRules
    ) -> None:
        assert scoring.defensive_contribution_threshold(
            Position.DEF
        ) < scoring.defensive_contribution_threshold(Position.MID)


class TestSquadRulesFromPayload:
    def test_core_constraints_match_the_payload(
        self, squad_rules: SquadRules, payload: dict[str, Any]
    ) -> None:
        published = payload["game_config"]["rules"]
        assert squad_rules.squad_size == published["squad_squadsize"]
        assert squad_rules.starting_size == published["squad_squadplay"]
        assert squad_rules.max_per_club == published["squad_team_limit"]
        assert squad_rules.total_budget == published["squad_total_spend"]

    def test_free_transfers_cap_derives_from_the_payload(
        self, squad_rules: SquadRules, payload: dict[str, Any]
    ) -> None:
        """FPL changed carry-over rules recently; a hardcoded cap would be stale."""
        expected = 1 + payload["game_config"]["rules"]["max_extra_free_transfers"]
        assert squad_rules.max_free_transfers == expected

    def test_position_quotas_sum_to_the_squad_size(self, squad_rules: SquadRules) -> None:
        assert sum(p.squad_select for p in squad_rules.positions) == squad_rules.squad_size

    def test_formation_bounds_can_fill_the_xi(self, squad_rules: SquadRules) -> None:
        assert sum(p.min_play for p in squad_rules.positions) <= squad_rules.starting_size
        assert sum(p.max_play for p in squad_rules.positions) >= squad_rules.starting_size


class TestSellingPrice:
    def test_profit_is_halved_and_rounded_down(self, squad_rules: SquadRules) -> None:
        """Bought at 5.0, now 5.5: half of the 0.5 rise is 0.25, floored to 0.2."""
        assert (
            selling_price(
                purchase_price=TenthsOfMillion(50),
                current_price=TenthsOfMillion(55),
                rules=squad_rules,
            )
            == 52
        )

    def test_even_profit_splits_exactly(self, squad_rules: SquadRules) -> None:
        assert (
            selling_price(
                purchase_price=TenthsOfMillion(50),
                current_price=TenthsOfMillion(54),
                rules=squad_rules,
            )
            == 52
        )

    def test_losses_are_absorbed_in_full(self, squad_rules: SquadRules) -> None:
        """A fall is not shared: bought at 7.0, now 6.5, sells at 6.5."""
        assert (
            selling_price(
                purchase_price=TenthsOfMillion(70),
                current_price=TenthsOfMillion(65),
                rules=squad_rules,
            )
            == 65
        )

    def test_unchanged_price_sells_at_cost(self, squad_rules: SquadRules) -> None:
        assert (
            selling_price(
                purchase_price=TenthsOfMillion(60),
                current_price=TenthsOfMillion(60),
                rules=squad_rules,
            )
            == 60
        )

    def test_single_tenth_rise_is_not_realised(self, squad_rules: SquadRules) -> None:
        """Half of one tenth floors to zero — the classic off-by-one-tenth case."""
        assert (
            selling_price(
                purchase_price=TenthsOfMillion(50),
                current_price=TenthsOfMillion(51),
                rules=squad_rules,
            )
            == 50
        )

    def test_selling_price_never_exceeds_current(self, squad_rules: SquadRules) -> None:
        for purchase in range(38, 140, 3):
            for rise in range(0, 30):
                current = purchase + rise
                got = selling_price(
                    purchase_price=TenthsOfMillion(purchase),
                    current_price=TenthsOfMillion(current),
                    rules=squad_rules,
                )
                assert purchase <= got <= current

    def test_negative_price_rejected(self, squad_rules: SquadRules) -> None:
        with pytest.raises(ValueError, match="negative"):
            selling_price(
                purchase_price=TenthsOfMillion(-1),
                current_price=TenthsOfMillion(50),
                rules=squad_rules,
            )


def _pick(code: int, position: Position, team: int, slot: int, price: int = 50) -> SquadPick:
    return SquadPick(
        player_code=PlayerCode(code),
        position=position,
        team_id=TeamId(team),
        purchase_price=TenthsOfMillion(price),
        current_price=TenthsOfMillion(price),
        selling_price=TenthsOfMillion(price),
        squad_slot=slot,
    )


def _legal_squad(squad_rules: SquadRules) -> list[SquadPick]:
    """A legal 15 in a 1-4-4-2 starting shape, spread across enough clubs."""
    layout = [
        (Position.GKP, 1),
        (Position.DEF, 4),
        (Position.MID, 4),
        (Position.FWD, 2),  # 11 starters
        (Position.GKP, 1),
        (Position.DEF, 1),
        (Position.MID, 1),
        (Position.FWD, 1),  # 4 bench
    ]
    picks: list[SquadPick] = []
    code, slot, team = 1000, 1, 1
    for position, count in layout:
        for _ in range(count):
            picks.append(_pick(code, position, team, slot))
            code += 1
            slot += 1
            team = team % 10 + 1
    return picks


class TestConstraints:
    def test_a_legal_squad_has_no_violations(self, squad_rules: SquadRules) -> None:
        picks = _legal_squad(squad_rules)
        assert check_squad(picks, rules=squad_rules) == []
        assert check_starting_xi(picks[:11], rules=squad_rules) == []

    def test_fourth_player_from_one_club_is_rejected(self, squad_rules: SquadRules) -> None:
        picks = _legal_squad(squad_rules)
        picks = [
            p.model_copy(update={"team_id": TeamId(1)}) if i < 4 else p for i, p in enumerate(picks)
        ]
        violations = check_squad(picks, rules=squad_rules)
        assert any(v.rule == "max_per_club" for v in violations)

    def test_over_budget_is_rejected(self, squad_rules: SquadRules) -> None:
        expensive = [
            p.model_copy(
                update={
                    "purchase_price": TenthsOfMillion(100),
                    "current_price": TenthsOfMillion(100),
                    "selling_price": TenthsOfMillion(100),
                }
            )
            for p in _legal_squad(squad_rules)
        ]
        violations = check_squad(expensive, rules=squad_rules)
        assert any(v.rule == "budget" for v in violations)

    def test_illegal_formation_is_rejected(self, squad_rules: SquadRules) -> None:
        """Two goalkeepers in the XI breaks the GKP 1-1 bound."""
        picks = _legal_squad(squad_rules)
        xi = [picks[0], picks[11].model_copy(update={"squad_slot": 2}), *picks[1:10]]
        violations = check_starting_xi(xi, rules=squad_rules)
        assert any(v.rule == "formation" for v in violations)

    def test_wrong_position_count_is_rejected(self, squad_rules: SquadRules) -> None:
        picks = _legal_squad(squad_rules)
        picks[-1] = picks[-1].model_copy(update={"position": Position.MID})
        violations = check_squad(picks, rules=squad_rules)
        assert any(v.rule == "position_quota" for v in violations)


class TestPointsAssembly:
    def _components(self, **kw: Any) -> ComponentExpectations:
        base: dict[str, Any] = {
            "minutes": MinutesPrediction(
                p_appearance=1.0,
                p_start=1.0,
                expected_minutes=90.0,
                p_60_plus=1.0,
                minutes_sd=0.0,
            ),
            "goals": 0.0,
            "assists": 0.0,
            "clean_sheet_probability": 0.0,
            "goals_conceded": 0.0,
            "saves": 0.0,
            "yellow_cards": 0.0,
            "red_cards": 0.0,
            "own_goals": 0.0,
            "penalties_saved": 0.0,
            "penalties_missed": 0.0,
            "defensive_contribution_probability": 0.0,
            "bonus": 0.0,
        }
        base.update(kw)
        return ComponentExpectations(**base)

    def test_a_full_ninety_earns_the_long_play_points(self, scoring: ScoringRules) -> None:
        result = assemble_points(self._components(), Position.MID, scoring)
        assert result.appearance == pytest.approx(scoring.long_play)
        assert result.total == pytest.approx(scoring.long_play)

    def test_a_certain_goal_scores_the_position_value(self, scoring: ScoringRules) -> None:
        gk = assemble_points(self._components(goals=1.0), Position.GKP, scoring)
        fwd = assemble_points(self._components(goals=1.0), Position.FWD, scoring)
        assert gk.goals == 10.0
        assert fwd.goals == 4.0

    def test_the_same_components_score_differently_by_position(self, scoring: ScoringRules) -> None:
        """The whole point of D8: re-scoring is a rules change, not a retrain."""
        components = self._components(goals=0.5, clean_sheet_probability=0.4)
        by_position = {p: assemble_points(components, p, scoring).total for p in Position}
        assert len(set(by_position.values())) == len(Position)

    def test_short_appearance_pays_less_than_a_long_one(self, scoring: ScoringRules) -> None:
        short = self._components(
            minutes=MinutesPrediction(
                p_appearance=1.0,
                p_start=0.0,
                expected_minutes=20.0,
                p_60_plus=0.0,
                minutes_sd=5.0,
            )
        )
        result = assemble_points(short, Position.MID, scoring)
        assert result.appearance == pytest.approx(scoring.short_play)

    def test_goals_conceded_uses_the_documented_divisor(self, scoring: ScoringRules) -> None:
        components = self._components(goals_conceded=2.0)
        result = assemble_points(components, Position.DEF, scoring)
        expected = (2.0 / scoring.thresholds.goals_conceded_per_deduction) * scoring.goals_conceded[
            Position.DEF
        ]
        assert result.goals_conceded == pytest.approx(expected)
        assert result.goals_conceded < 0

    def test_saves_use_the_documented_divisor(self, scoring: ScoringRules) -> None:
        result = assemble_points(self._components(saves=3.0), Position.GKP, scoring)
        assert result.saves == pytest.approx(scoring.saves)

    def test_cards_reduce_the_total(self, scoring: ScoringRules) -> None:
        result = assemble_points(self._components(yellow_cards=1.0), Position.DEF, scoring)
        assert result.cards == pytest.approx(scoring.yellow_cards)
        assert result.total < scoring.long_play

    def test_breakdown_always_sums_to_its_own_total(self, scoring: ScoringRules) -> None:
        components = self._components(
            goals=0.4,
            assists=0.3,
            clean_sheet_probability=0.35,
            goals_conceded=1.1,
            saves=2.0,
            yellow_cards=0.2,
            defensive_contribution_probability=0.5,
            bonus=0.7,
        )
        for position in Position:
            breakdown = assemble_points(components, position, scoring)
            parts = [
                breakdown.appearance,
                breakdown.goals,
                breakdown.assists,
                breakdown.clean_sheets,
                breakdown.goals_conceded,
                breakdown.saves,
                breakdown.cards,
                breakdown.own_goals,
                breakdown.penalties,
                breakdown.defensive_contribution,
                breakdown.bonus,
            ]
            assert breakdown.total == pytest.approx(sum(parts))


class TestFixtureIntegrity:
    def test_fixture_stays_within_the_commit_budget(self) -> None:
        """Golden fixtures live in git, so they must stay small."""
        size = BOOTSTRAP_FIXTURE.stat().st_size
        assert size < 256 * 1024, f"fixture is {size / 1024:.0f}KB; budget is 256KB"

    def test_fixture_records_its_provenance(self, payload: dict[str, Any]) -> None:
        provenance = payload["_fixture_provenance"]
        assert provenance["source"].startswith("https://fantasy.premierleague.com/api/")
        assert len(provenance["full_payload_sha256"]) == 64
        assert hashlib.sha256(b"").hexdigest() != provenance["full_payload_sha256"]
