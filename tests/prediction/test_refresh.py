"""Refresh-planner tests.

The planner exists because the naive design — one lookup per player per
gameweek — is roughly 600 lookups a week, most about players nobody will own.
So the tests are about what it *declines* to spend budget on, and about the
honesty of what it reports having covered.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from xg_alonso.contracts.form import (
    FormDirection,
    FormSignal,
    FormStrength,
    SignalSet,
)
from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.prediction.refresh import plan_refresh

NOW = datetime(2026, 8, 1, tzinfo=UTC)


def _prediction(code: int, points: float) -> PlayerPrediction:
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
            minutes=MinutesPrediction(
                p_appearance=1.0, p_start=0.9, expected_minutes=85.0, p_60_plus=0.8, minutes_sd=8.0
            ),
            goals=0.2,
            assists=0.1,
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
        expected_points_sd=1.0,
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


def _league(clubs: int = 20, per_club: int = 30):  # type: ignore[no-untyped-def]
    """A realistic league: 600 players across 20 clubs, output spread widely."""
    predictions, teams, prices = [], {}, {}
    code = 1
    for club in range(clubs):
        for slot in range(per_club):
            points = 6.0 - slot * 0.2
            predictions.append(_prediction(code, max(points, 0.1)))
            teams[PlayerCode(code)] = TeamId(club + 1)
            prices[PlayerCode(code)] = TenthsOfMillion(40 + slot * 3)
            code += 1
    return predictions, teams, prices


def _signal(code: int, expires: datetime) -> FormSignal:
    return FormSignal(
        player_code=PlayerCode(code),
        direction=FormDirection.NEGATIVE,
        strength=FormStrength.CLEAR,
        summary="Reported to be struggling for form and confidence.",
        sources=("https://example.com/report",),
        observed_at=NOW - timedelta(days=1),
        expires_at=expires,
    )


class TestCost:
    def test_the_plan_never_exceeds_its_budget(self) -> None:
        """The whole point: work must be bounded, not proportional to the league."""
        predictions, teams, prices = _league()
        plan = plan_refresh(
            predictions, teams=teams, prices=prices, signals=SignalSet(), at=NOW, budget=5
        )
        assert plan.lookups <= 5

    def test_lookups_are_per_club_not_per_player(self) -> None:
        """Team news is published per club; 20 lookups replace 600."""
        predictions, teams, prices = _league()
        plan = plan_refresh(
            predictions, teams=teams, prices=prices, signals=SignalSet(), at=NOW, budget=20
        )
        assert plan.lookups <= 20
        assert plan.covered_players > plan.lookups * 2

    def test_each_club_appears_at_most_once(self) -> None:
        predictions, teams, prices = _league()
        plan = plan_refresh(
            predictions, teams=teams, prices=prices, signals=SignalSet(), at=NOW, budget=20
        )
        seen = [request.team_id for request in plan.requests]
        assert len(seen) == len(set(seen))


class TestRelevance:
    def test_players_who_cannot_change_a_decision_are_skipped(self) -> None:
        """A story about a player nobody would start alters nothing."""
        predictions, teams, prices = _league()
        plan = plan_refresh(
            predictions,
            teams=teams,
            prices=prices,
            signals=SignalSet(),
            at=NOW,
            budget=20,
            relevance_floor=5.0,
        )
        assert plan.total_players < len(predictions)

    def test_unaffordable_players_are_skipped(self) -> None:
        predictions, teams, prices = _league()
        rich = plan_refresh(
            predictions, teams=teams, prices=prices, signals=SignalSet(), at=NOW, budget=20
        )
        poor = plan_refresh(
            predictions,
            teams=teams,
            prices=prices,
            signals=SignalSet(),
            at=NOW,
            budget=20,
            affordable_at=TenthsOfMillion(45),
        )
        assert poor.total_players < rich.total_players

    def test_an_owned_player_survives_both_filters(self) -> None:
        """A slump in someone you own changes a decision with no transfer needed."""
        predictions, teams, prices = _league()
        cheap_and_bad = PlayerCode(30)  # last slot of the first club
        plan = plan_refresh(
            predictions,
            teams=teams,
            prices=prices,
            signals=SignalSet(),
            at=NOW,
            budget=20,
            relevance_floor=99.0,
            affordable_at=TenthsOfMillion(0),
            owned=frozenset({cheap_and_bad}),
        )
        covered = {code for request in plan.requests for code in request.players}
        assert cheap_and_bad in covered


class TestStaleness:
    def test_a_fresh_signal_is_not_refetched(self) -> None:
        predictions, teams, prices = _league(clubs=1, per_club=2)
        signals = SignalSet(
            signals=tuple(_signal(code, NOW + timedelta(days=30)) for code in (1, 2))
        )
        plan = plan_refresh(
            predictions, teams=teams, prices=prices, signals=signals, at=NOW, budget=20
        )
        assert plan.lookups == 0

    def test_a_signal_near_expiry_is_refetched(self) -> None:
        """Waiting for it to lapse leaves a window where the projection reverts."""
        predictions, teams, prices = _league(clubs=1, per_club=2)
        signals = SignalSet(
            signals=tuple(_signal(code, NOW + timedelta(hours=12)) for code in (1, 2))
        )
        plan = plan_refresh(
            predictions, teams=teams, prices=prices, signals=signals, at=NOW, budget=20
        )
        assert plan.lookups == 1

    def test_fresh_players_still_count_toward_coverage(self) -> None:
        """Coverage describes the squad, not just this run's fetches."""
        predictions, teams, prices = _league(clubs=1, per_club=2)
        signals = SignalSet(
            signals=tuple(_signal(code, NOW + timedelta(days=30)) for code in (1, 2))
        )
        plan = plan_refresh(
            predictions, teams=teams, prices=prices, signals=signals, at=NOW, budget=20
        )
        assert plan.total_players == 2


class TestOrdering:
    def test_the_most_valuable_clubs_come_first(self) -> None:
        """If the budget runs out, what is dropped must be what mattered least."""
        predictions, teams, prices = _league()
        plan = plan_refresh(
            predictions, teams=teams, prices=prices, signals=SignalSet(), at=NOW, budget=20
        )
        priorities = [request.priority for request in plan.requests]
        assert priorities == sorted(priorities, reverse=True)

    def test_the_plan_reports_what_it_left_out(self) -> None:
        """'20 lookups' reads as thorough; the fraction reads as what it is."""
        predictions, teams, prices = _league()
        plan = plan_refresh(
            predictions, teams=teams, prices=prices, signals=SignalSet(), at=NOW, budget=3
        )
        assert plan.skipped_teams > 0
        assert str(plan.total_players) in plan.describe()
