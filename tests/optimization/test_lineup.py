"""Starting XI, captaincy and formation tests.

Captaincy doubles a player's return, so it is the largest single term in FPL
scoring. Every backtest result before this module existed was produced with the
goalkeeper as captain, because the captain was whatever sat in squad slot 1.
These tests exist so that cannot recur silently.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.domain.rules import PositionRule, SquadRules
from xg_alonso.optimization.lineup import (
    CAPTAIN_MULTIPLIER,
    best_starting_xi,
    legal_formations,
    starting_xi_points,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _rules() -> SquadRules:
    return SquadRules(
        version="test",
        source_sha256="a" * 64,
        squad_size=15,
        starting_size=11,
        max_per_club=3,
        total_budget=TenthsOfMillion(1000),
        sell_on_fee=0.5,
        sell_at_purchase_price=False,
        max_extra_free_transfers=4,
        transfers_cap=20,
        positions=(
            PositionRule(position=Position.GKP, squad_select=2, min_play=1, max_play=1),
            PositionRule(position=Position.DEF, squad_select=5, min_play=3, max_play=5),
            PositionRule(position=Position.MID, squad_select=5, min_play=2, max_play=5),
            PositionRule(position=Position.FWD, squad_select=3, min_play=1, max_play=3),
        ),
    )


def _prediction(code: int, points: float) -> PlayerPrediction:
    minutes = MinutesPrediction(
        p_appearance=1.0, p_start=0.9, expected_minutes=85.0, p_60_plus=0.85, minutes_sd=8.0
    )
    breakdown = PointsBreakdown(
        appearance=points,
        goals=0.0,
        assists=0.0,
        clean_sheets=0.0,
        goals_conceded=0.0,
        saves=0.0,
        cards=0.0,
        own_goals=0.0,
        penalties=0.0,
        defensive_contribution=0.0,
        bonus=0.0,
    )
    return PlayerPrediction(
        player_code=PlayerCode(code),
        position=Position.MID,
        from_gameweek=GameweekId(1),
        horizon_gameweeks=1,
        components=ComponentExpectations(
            minutes=minutes,
            goals=0.0,
            assists=0.0,
            clean_sheet_probability=0.0,
            goals_conceded=0.0,
            saves=0.0,
            yellow_cards=0.0,
            red_cards=0.0,
            own_goals=0.0,
            penalties_saved=0.0,
            penalties_missed=0.0,
            defensive_contribution_probability=0.0,
            bonus=0.0,
        ),
        breakdown=breakdown,
        expected_points=points,
        expected_points_sd=0.5,
        scoring_rules_version="test",
        provenance=PredictionProvenance(
            model_name="t",
            model_version="1",
            model_artifact_sha256="b" * 64,
            feature_set_name="f",
            feature_set_version="1",
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="r",
            code_version="c",
        ),
    )


def _pick(code: int, position: Position, slot: int, team: int = 1) -> SquadPick:
    price = TenthsOfMillion(50)
    return SquadPick(
        player_code=PlayerCode(code),
        position=position,
        team_id=TeamId(team),
        purchase_price=price,
        current_price=price,
        selling_price=price,
        squad_slot=slot,
    )


def _squad(scores: dict[int, float]) -> tuple[list[SquadPick], dict[PlayerCode, PlayerPrediction]]:
    """A legal 15 (2 GKP, 5 DEF, 5 MID, 3 FWD) with the given expected points."""
    layout = (
        [(Position.GKP, c) for c in (1, 2)]
        + [(Position.DEF, c) for c in (3, 4, 5, 6, 7)]
        + [(Position.MID, c) for c in (8, 9, 10, 11, 12)]
        + [(Position.FWD, c) for c in (13, 14, 15)]
    )
    picks = [_pick(code, pos, i + 1, team=1 + code % 8) for i, (pos, code) in enumerate(layout)]
    predictions = {PlayerCode(code): _prediction(code, scores.get(code, 1.0)) for _, code in layout}
    return picks, predictions


class TestFormations:
    def test_the_eight_real_fpl_formations_are_derived(self) -> None:
        shapes = {f"{d}-{m}-{f}" for _, d, m, f in legal_formations(_rules())}
        assert shapes == {
            "3-4-3",
            "3-5-2",
            "4-3-3",
            "4-4-2",
            "4-5-1",
            "5-2-3",
            "5-3-2",
            "5-4-1",
        }

    def test_every_shape_fields_eleven_with_one_keeper(self) -> None:
        for gkp, d, m, f in legal_formations(_rules()):
            assert gkp == 1
            assert gkp + d + m + f == 11


class TestXiSelection:
    def test_the_formation_follows_where_the_points_are(self) -> None:
        """A squad with three strong forwards should play three forwards."""
        picks, predictions = _squad({13: 9.0, 14: 9.0, 15: 9.0})
        selection = best_starting_xi(picks, predictions, _rules())
        assert selection.formation[3] == 3, f"expected 3 forwards, got {selection.formation}"

    def test_positional_minima_are_respected_even_when_costly(self) -> None:
        """A goalkeeper must start however badly he scores."""
        picks, predictions = _squad({1: 0.0, 2: 0.0})
        selection = best_starting_xi(picks, predictions, _rules())
        keepers = [p for p in selection.starters if p.position is Position.GKP]
        assert len(keepers) == 1
        defenders = [p for p in selection.starters if p.position is Position.DEF]
        assert len(defenders) >= 3

    def test_the_xi_is_not_simply_the_top_eleven(self) -> None:
        """Positional minima force weaker players in — that is the whole problem."""
        picks, predictions = _squad({8: 9, 9: 9, 10: 9, 11: 9, 12: 9, 13: 8, 14: 8, 15: 8})
        selection = best_starting_xi(picks, predictions, _rules())
        chosen = {int(p.player_code) for p in selection.starters}
        top_eleven = {
            int(c) for c in sorted(predictions, key=lambda k: -predictions[k].expected_points)[:11]
        }
        assert chosen != top_eleven

    def test_exactly_eleven_start(self) -> None:
        picks, predictions = _squad({})
        selection = best_starting_xi(picks, predictions, _rules())
        assert len(selection.starters) == 11
        assert len(selection.bench) == 4

    def test_starters_and_bench_partition_the_squad(self) -> None:
        picks, predictions = _squad({5: 7.0, 13: 8.0})
        selection = best_starting_xi(picks, predictions, _rules())
        combined = {int(p.player_code) for p in selection.starters + selection.bench}
        assert combined == {int(p.player_code) for p in picks}

    def test_an_unfillable_squad_loses_rather_than_raising(self) -> None:
        """Thousands of candidates are ranked; an illegal one should just lose."""
        picks, predictions = _squad({})
        selection = best_starting_xi(picks[:4], predictions, _rules())
        assert selection.expected_points == 0.0
        assert selection.starters == ()


class TestCaptaincy:
    def test_the_captain_is_the_best_expected_scorer_in_the_xi(self) -> None:
        picks, predictions = _squad({10: 12.0})
        selection = best_starting_xi(picks, predictions, _rules())
        assert selection.captain == PlayerCode(10)

    def test_the_captain_is_never_the_goalkeeper_when_anyone_outscores_him(self) -> None:
        """The exact defect: every earlier backtest captained the keeper."""
        picks, predictions = _squad({1: 4.0, 10: 9.0})
        selection = best_starting_xi(picks, predictions, _rules())
        captain_pick = next(p for p in picks if p.player_code == selection.captain)
        assert captain_pick.position is not Position.GKP

    def test_captaincy_is_actually_worth_double(self) -> None:
        picks, predictions = _squad({10: 10.0})
        selection = best_starting_xi(picks, predictions, _rules())
        plain = sum(predictions[p.player_code].expected_points for p in selection.starters)
        assert selection.expected_points == pytest.approx(plain + 10.0)

    def test_vice_captain_is_the_second_best(self) -> None:
        picks, predictions = _squad({10: 12.0, 11: 11.0})
        selection = best_starting_xi(picks, predictions, _rules())
        assert selection.captain == PlayerCode(10)
        assert selection.vice_captain == PlayerCode(11)

    def test_captain_and_vice_are_different_players(self) -> None:
        picks, predictions = _squad({})
        selection = best_starting_xi(picks, predictions, _rules())
        assert selection.captain != selection.vice_captain


class TestBenchAwareScoring:
    def test_upgrading_a_benched_player_below_the_xi_gains_nothing(self) -> None:
        """The defect that inflated transfer counts.

        A bench-out move used to score its full expected difference even though
        nothing that actually scores had changed. The squad here has a clearly
        dominant XI, so the improved player remains unable to displace anyone.
        """
        rules = _rules()
        strong = {
            1: 5.0,  # first-choice keeper
            3: 8.0,
            4: 8.0,
            5: 8.0,  # three strong defenders
            8: 9.0,
            9: 9.0,
            10: 9.0,
            11: 9.0,
            12: 9.0,  # five strong midfielders
            13: 8.0,
            14: 8.0,  # two strong forwards
        }
        picks, predictions = _squad(strong)
        before = starting_xi_points(picks, predictions, rules)

        benched = best_starting_xi(picks, predictions, rules).bench
        target = next(p for p in benched if p.position is Position.DEF)
        assert predictions[target.player_code].expected_points == 1.0

        # A small improvement, still far below the weakest starter.
        predictions[target.player_code] = _prediction(int(target.player_code), 1.5)
        after = starting_xi_points(picks, predictions, rules)

        assert after == pytest.approx(before), (
            "improving a player who still does not start must not change the score"
        )

    def test_upgrading_a_benched_player_above_the_xi_does_gain(self) -> None:
        rules = _rules()
        picks, predictions = _squad({})
        before = starting_xi_points(picks, predictions, rules)
        predictions[PlayerCode(15)] = _prediction(15, 20.0)
        after = starting_xi_points(picks, predictions, rules)
        assert after > before, "a player good enough to start must change the score"

    def test_the_score_includes_the_captain_doubling(self) -> None:
        picks, predictions = _squad({10: 10.0})
        rules = _rules()
        total = starting_xi_points(picks, predictions, rules)
        selection = best_starting_xi(picks, predictions, rules)
        plain = sum(predictions[p.player_code].expected_points for p in selection.starters)
        assert total == pytest.approx(plain * 1.0 + 10.0 * (CAPTAIN_MULTIPLIER - 1))

    def test_selection_is_deterministic(self) -> None:
        picks, predictions = _squad({5: 6.0, 10: 6.0, 13: 6.0})
        rules = _rules()
        first = best_starting_xi(picks, predictions, rules)
        second = best_starting_xi(list(reversed(picks)), predictions, rules)
        assert first.captain == second.captain
        assert {int(p.player_code) for p in first.starters} == {
            int(p.player_code) for p in second.starters
        }

    def test_a_player_without_a_prediction_scores_zero_not_an_error(self) -> None:
        picks, predictions = _squad({})
        del predictions[PlayerCode(10)]
        selection = best_starting_xi(picks, predictions, _rules())
        assert len(selection.starters) == 11
