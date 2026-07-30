"""Shared factories for optimization tests.

Lifted out of ``test_squad_builder`` rather than copied when the requirements
tests needed the same pool. Two definitions of "a candidate" drifting apart is
how a test suite starts asserting against a shape production never produces.
"""

from __future__ import annotations

from datetime import UTC, datetime

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.domain.rules import PositionRule, SquadRules
from xg_alonso.optimization.squad_builder import SquadCandidate

__all__ = ["CUTOFF", "make_candidate", "make_prediction", "make_rules"]

CUTOFF = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)


def make_prediction(code: int, position: Position, points: float) -> PlayerPrediction:
    """A prediction carrying `points`, with the breakdown made to agree.

    All the points are booked to `appearance` so the breakdown sums to the
    total — `PlayerPrediction` rejects a breakdown that disagrees with its own
    total, which is the contract that stops an explanation drifting from its
    number.
    """
    return PlayerPrediction(
        player_code=PlayerCode(code),
        position=position,
        from_gameweek=GameweekId(1),
        horizon_gameweeks=1,
        components=ComponentExpectations(
            minutes=MinutesPrediction(
                p_appearance=1.0,
                p_start=0.9,
                expected_minutes=85.0,
                p_60_plus=0.85,
                minutes_sd=8.0,
            ),
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
        breakdown=PointsBreakdown(
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
        ),
        expected_points=points,
        expected_points_sd=1.0,
        scoring_rules_version="test",
        provenance=PredictionProvenance(
            model_name="test",
            model_version="1",
            model_artifact_sha256="b" * 64,
            feature_set_name="f",
            feature_set_version="test_v1",
            data_cutoff=CUTOFF,
            predicted_at=CUTOFF,
            run_id="test",
            code_version="test",
        ),
    )


def make_candidate(
    *,
    code: int,
    position: Position,
    price: int,
    expected_points: float,
    team_id: int,
) -> SquadCandidate:
    """One selectable player. Keyword-only, because five positional numbers in a
    row is how a price ends up in the points column."""
    return SquadCandidate(
        player_code=PlayerCode(code),
        position=position,
        team_id=TeamId(team_id),
        price=TenthsOfMillion(price),
        prediction=make_prediction(code, position, expected_points),
    )


def make_rules() -> SquadRules:
    """Real FPL shape, loaded the way production loads it."""
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
