"""Combining a player's fixtures within one gameweek.

**Why the components stay per-fixture.** ``MinutesPrediction`` bounds
``expected_minutes`` at 120 and every probability at 1.0, and
``contracts/prediction.py`` is frozen by policy. Two fixtures at 85 minutes is
170, so a double gameweek is *literally not expressible* in the component
model. Points and the breakdown are therefore summed while ``components`` keep
describing a single fixture — the same compromise
``optimization.horizon.horizon_valued`` already ships, and for the same reason.

That is a real limitation, not a tidy one: any reason built from
``components.minutes`` for a double-gameweek player understates him. It is
stated here, at the point of aggregation, because the alternative is a reader
discovering it from a number that looks wrong.

**Uncertainty adds in quadrature, unlike the horizon.** Two fixtures in one
gameweek are separate matches, so their errors are independent and the standard
deviations combine as ``sqrt(a² + b²)``. ``horizon_valued`` inflates
multiplicatively instead, because it is modelling forecast decay over weeks
rather than the sum of independent events. Two different uncertainties, two
different rules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)

__all__ = ["blank_prediction", "collapse_by_player", "combine_gameweek_fixtures"]

#: A player whose club has no fixture. Every probability is zero rather than a
#: prior, because "no match" is certainty about absence, not ignorance.
_BLANK_MINUTES = MinutesPrediction(
    p_appearance=0.0, p_start=0.0, expected_minutes=0.0, p_60_plus=0.0, minutes_sd=0.0
)


def blank_prediction(
    *,
    player_code: PlayerCode,
    position: Position,
    template: PlayerPrediction,
) -> PlayerPrediction:
    """A zeroed prediction for a player with no fixture.

    Zero rather than a small number: a blanking player cannot score, so he must
    be unselectable as captain rather than merely unlucky. Provenance and the
    gameweek are carried from ``template`` so the row is still traceable.
    """
    return PlayerPrediction(
        player_code=player_code,
        position=position,
        from_gameweek=template.from_gameweek,
        horizon_gameweeks=template.horizon_gameweeks,
        components=ComponentExpectations(
            minutes=_BLANK_MINUTES,
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
            appearance=0.0,
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
        expected_points=0.0,
        expected_points_sd=0.0,
        scoring_rules_version=template.scoring_rules_version,
        provenance=template.provenance,
    )


def combine_gameweek_fixtures(predictions: Sequence[PlayerPrediction]) -> PlayerPrediction:
    """Sum one player's fixtures within a gameweek into a single prediction.

    Args:
        predictions: One player's rows for one gameweek, in fixture order. Must
            be non-empty and must all name the same player.

    Returns:
        A prediction whose points and breakdown are the totals. ``components``
        describe the *first* fixture only — see the module docstring.

    Raises:
        ValueError: on an empty sequence, or rows for more than one player.
            Both are programming errors, and a silent answer to either would be
            a number nobody could interpret.
    """
    if not predictions:
        raise ValueError("no predictions to combine; a blank needs blank_prediction()")

    codes = {p.player_code for p in predictions}
    if len(codes) != 1:
        raise ValueError(f"combine_gameweek_fixtures got rows for {len(codes)} players: {codes}")

    if len(predictions) == 1:
        return predictions[0]

    first = predictions[0]
    breakdown = PointsBreakdown(
        appearance=sum(p.breakdown.appearance for p in predictions),
        goals=sum(p.breakdown.goals for p in predictions),
        assists=sum(p.breakdown.assists for p in predictions),
        clean_sheets=sum(p.breakdown.clean_sheets for p in predictions),
        goals_conceded=sum(p.breakdown.goals_conceded for p in predictions),
        saves=sum(p.breakdown.saves for p in predictions),
        cards=sum(p.breakdown.cards for p in predictions),
        own_goals=sum(p.breakdown.own_goals for p in predictions),
        penalties=sum(p.breakdown.penalties for p in predictions),
        defensive_contribution=sum(p.breakdown.defensive_contribution for p in predictions),
        bonus=sum(p.breakdown.bonus for p in predictions),
    )

    return first.model_copy(
        update={
            "breakdown": breakdown,
            # Rounded so the frozen `breakdown.total == expected_points`
            # validator is not defeated by float summation order.
            "expected_points": round(breakdown.total, 9),
            "expected_points_sd": round(
                math.sqrt(sum(p.expected_points_sd**2 for p in predictions)), 9
            ),
        }
    )


def collapse_by_player(
    predictions: Sequence[PlayerPrediction],
) -> dict[PlayerCode, PlayerPrediction]:
    """Group per-fixture predictions by player and combine each group.

    Replaces ``{p.player_code: p for p in predictions}``, which keeps the *last*
    row per player. That comprehension was lossless only because fixtures were
    already deduplicated to one per club upstream; the moment a player can have
    two rows it silently discards one, and in the direction that understates a
    double gameweek.
    """
    grouped: dict[PlayerCode, list[PlayerPrediction]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.player_code, []).append(prediction)
    return {code: combine_gameweek_fixtures(rows) for code, rows in grouped.items()}
