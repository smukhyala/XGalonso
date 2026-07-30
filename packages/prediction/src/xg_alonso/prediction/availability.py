"""Apply FPL's published chance of playing to a projection.

**The gap this closes.** ``status`` is a hard gate — an injured player is not
selectable and the squad builder drops him. But ``chance_of_playing_next_round``
is a *soft* number FPL publishes alongside it, and nothing read it. A player on
"75% chance of playing" was scored as fully fit, which is the difference between
a sensible captain and a bad one.

**This is not a form signal, and the difference is the point.**
:mod:`.form` handles outside information — somebody's reading of a match report
— and clamps it to ±15% precisely because it is judgement. What FPL publishes
here is not judgement, it is the game's own statement about a player's
availability, and clamping it would be substituting our caution for their fact.
A 25% chance means a 75% reduction and is applied as one.

**Minutes are what actually move.** A doubtful player is not a worse footballer;
he is the same footballer less likely to be on the pitch. So the probability
scales the minutes distribution, and every component scales with it — a player
who plays a quarter as often scores a quarter as often, and that falls out of
the minutes rather than being applied to points directly.

**Uncertainty widens, never narrows.** A 50% chance of playing is a genuinely
bimodal outcome — he plays or he does not — and reporting the midpoint with the
original confidence would understate exactly the risk a manager is deciding
about.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.prediction import (
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
)

__all__ = [
    "CERTAIN",
    "apply_availability",
    "availability_factor",
]

#: The probability at or above which no adjustment is made.
#:
#: FPL clears the field entirely for a fully fit player, so ``None`` and ``100``
#: mean the same thing and both leave a projection alone.
CERTAIN: Final[int] = 100

#: Below this, a player is doubtful enough that the spread of outcomes matters
#: more than their mean. Used only to widen uncertainty, never to change points.
_BIMODAL_BELOW: Final[int] = 90


def availability_factor(chance: int | None) -> float:
    """What a published chance of playing scales a projection by.

    ``None`` is fully fit rather than unknown: FPL clears the field when a
    player has no doubt against him, so treating it as missing would discount
    every healthy player in the game.

    Values outside 0-100 are clamped rather than trusted. The field is
    documented as a percentage and has never been anything else, but a projection
    silently multiplied by 1.4 because an upstream field changed shape is not a
    failure anybody would notice.
    """
    if chance is None:
        return 1.0
    return max(0.0, min(float(chance), 100.0)) / 100.0


def _scale_minutes(minutes: MinutesPrediction, factor: float) -> MinutesPrediction:
    """Scale a minutes distribution by a probability of featuring at all.

    Every probability scales and the standard deviation grows, because the
    outcome is more bimodal rather than more certain. The contract's invariants
    — start implies appearance, sixty-plus implies appearance — survive because
    scaling all three by the same factor preserves their ordering.
    """
    appearance = minutes.p_appearance * factor
    start = min(minutes.p_start * factor, appearance)
    sixty = min(minutes.p_60_plus * factor, appearance)

    # A coin-flip outcome is the widest, so the added spread peaks at 0.5 and
    # vanishes at both ends: a certain starter and a certain absentee are both
    # perfectly predictable.
    spread = 1.0 + 2.0 * factor * (1.0 - factor)

    return MinutesPrediction(
        p_appearance=round(max(0.0, min(appearance, 1.0)), 6),
        p_start=round(max(0.0, min(start, 1.0)), 6),
        expected_minutes=round(max(0.0, minutes.expected_minutes * factor), 6),
        p_60_plus=round(max(0.0, min(sixty, 1.0)), 6),
        minutes_sd=round(minutes.minutes_sd * spread, 6),
    )


def _scale_breakdown(breakdown: PointsBreakdown, factor: float) -> PointsBreakdown:
    """Scale every scoring term by the same factor.

    All of them, including the negative ones. A player who is half as likely to
    appear is half as likely to be booked, and leaving the card term untouched
    would make a doubtful player look *worse* than his availability implies
    rather than merely less valuable.
    """
    return breakdown.model_copy(
        update={
            field: round(getattr(breakdown, field) * factor, 6)
            for field in (
                "appearance",
                "goals",
                "assists",
                "clean_sheets",
                "goals_conceded",
                "saves",
                "cards",
                "own_goals",
                "penalties",
                "defensive_contribution",
                "bonus",
            )
            if hasattr(breakdown, field)
        }
    )


def apply_availability(
    predictions: Sequence[PlayerPrediction],
    chances: Mapping[PlayerCode, int | None],
    *,
    floor: float = 0.0,
) -> list[PlayerPrediction]:
    """Scale projections by FPL's published chance of playing.

    Args:
        predictions: Model output, already assembled into points.
        chances: ``chance_of_playing_next_round`` per player. A player absent
            from the mapping is left alone — absent means "not told", and
            inventing a discount for him would penalise every player the caller
            happened not to look up.
        floor: Lowest factor to apply. Zero by default, so a 0% player projects
            zero, which is correct and lets the optimizer drop him on the
            numbers rather than only on the hard status gate.

    Returns:
        Predictions, scaled where a chance is published and untouched elsewhere.
    """
    adjusted: list[PlayerPrediction] = []

    for prediction in predictions:
        if prediction.player_code not in chances:
            adjusted.append(prediction)
            continue

        factor = max(floor, availability_factor(chances[prediction.player_code]))
        if factor >= 1.0:
            adjusted.append(prediction)
            continue

        breakdown = _scale_breakdown(prediction.breakdown, factor)
        components = prediction.components.model_copy(
            update={"minutes": _scale_minutes(prediction.components.minutes, factor)}
        )
        chance = chances[prediction.player_code]
        widen = 1.0 + (1.0 - factor) if (chance or 0) < _BIMODAL_BELOW else 1.0

        adjusted.append(
            prediction.model_copy(
                update={
                    "components": components,
                    "breakdown": breakdown,
                    "expected_points": breakdown.total,
                    "expected_points_sd": round(prediction.expected_points_sd * widen, 6),
                }
            )
        )

    return adjusted
