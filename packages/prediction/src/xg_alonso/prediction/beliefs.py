"""Applying a manager's intuition without letting it become evidence.

**The raw prediction is never overwritten.** :func:`apply_beliefs` returns a
:class:`BeliefAdjustment` carrying *both* the original and the adjusted
prediction, plus the multiplier and the reason for it. Every downstream surface
can therefore show what the model said, what the manager believes, and what the
two produce together — as three separate things.

That is the whole design constraint. A belief that silently replaced a
prediction would be indistinguishable from data by the time it reached a
recommendation screen, and the user would be reading their own hunch back to
themselves with a model's authority attached.

**Adjustments are bounded and multiplicative.** The clamp is what keeps a hunch
from becoming an assertion: at maximum confidence a belief moves a projection by
:data:`BELIEF_CLAMP`, and no combination of beliefs can exceed it. An unbounded
adjustment would let ``confidence=1.0`` mean "ignore the model", which is not
what a manager means when they say they are sure.

This mirrors :mod:`xg_alonso.prediction.form`, which already applies clamped
outside signals — a belief is the same shape of thing with a different source,
and giving it a different mechanism would mean two ways to nudge a projection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
from xg_alonso.contracts.objective import (
    BeliefEntity,
    BeliefProposition,
    UserBelief,
)
from xg_alonso.contracts.prediction import PlayerPrediction
from xg_alonso.domain.scoring import ScoringRules, assemble_points

__all__ = [
    "BELIEF_CLAMP",
    "BeliefAdjustment",
    "apply_beliefs",
    "belief_sensitivity",
]

#: Maximum fractional movement a belief may produce, at confidence 1.0.
#:
#: 25%, matching the spirit of ``FORM_SIGNAL_CLAMP``. Large enough that a strong
#: belief visibly changes a ranking and can flip a marginal transfer; small
#: enough that it cannot manufacture a recommendation the evidence does not
#: support at all. A manager who is certain still does not get to overrule four
#: seasons of data with a sentence.
BELIEF_CLAMP: Final[float] = 0.25


@dataclass(frozen=True)
class BeliefAdjustment:
    """One player's raw and belief-adjusted prediction, side by side."""

    player_code: PlayerCode
    raw: PlayerPrediction
    adjusted: PlayerPrediction
    multiplier: float
    beliefs_applied: tuple[UserBelief, ...] = ()

    @property
    def delta(self) -> float:
        """Expected points the belief added. Negative when it argues against."""
        return self.adjusted.expected_points - self.raw.expected_points

    @property
    def moved(self) -> bool:
        return abs(self.multiplier - 1.0) > 1e-9

    def explain(self) -> str:
        if not self.moved:
            return "no belief applies to this player"
        direction = "raised" if self.delta > 0 else "lowered"
        reasons = "; ".join(b.rationale or b.proposition.value for b in self.beliefs_applied)
        return (
            f"belief {direction} the projection from {self.raw.expected_points:.2f} to "
            f"{self.adjusted.expected_points:.2f} ({self.delta:+.2f}) — {reasons}"
        )


#: How strongly each proposition moves a projection at full confidence, before
#: the clamp. Values below 1.0 mean the proposition speaks to a component rather
#: than to the whole projection, so it should not move the total as hard.
_STRENGTH: Final[dict[BeliefProposition, float]] = {
    BeliefProposition.OUTPERFORM_MODEL: 1.0,
    BeliefProposition.UNDERPERFORM_MODEL: 1.0,
    BeliefProposition.WILL_RETURN: 0.9,
    BeliefProposition.WILL_START: 0.7,
    BeliefProposition.WILL_NOT_START: 1.0,
    BeliefProposition.CLEAN_SHEET: 0.6,
}


def _multiplier(beliefs: Sequence[UserBelief], gameweek: GameweekId) -> float:
    """Combined, clamped multiplier for one player in one gameweek.

    Beliefs are summed in *signed strength* before clamping, not multiplied
    together. Two beliefs pointing the same way should reinforce each other but
    must not compound past the clamp — and two pointing opposite ways should
    partly cancel, which is the honest reading of a manager who is unsure.
    """
    total = 0.0
    for belief in beliefs:
        weight = belief.weight_at(gameweek)
        if weight <= 0.0:
            continue
        total += belief.proposition.direction * _STRENGTH.get(belief.proposition, 1.0) * weight
    return 1.0 + BELIEF_CLAMP * max(-1.0, min(1.0, total))


def apply_beliefs(
    predictions: Sequence[PlayerPrediction],
    beliefs: Sequence[UserBelief],
    *,
    gameweek: GameweekId,
    rules: ScoringRules,
    team_of: dict[PlayerCode, int] | None = None,
) -> list[BeliefAdjustment]:
    """Adjust projections by stated beliefs, keeping both versions.

    Args:
        predictions: The model's own output. Returned unchanged inside each
            adjustment.
        beliefs: What the manager claims.
        gameweek: Used to scope and decay each belief.
        rules: Scoring rules, so the adjusted points are *reassembled* from
            adjusted components rather than scaled at the total. That keeps the
            breakdown summing to the total, which the prediction contract
            enforces — a scaled total with an unscaled breakdown would be
            rejected outright, and rightly so.
        team_of: Player to club, needed to route team-level beliefs.

    Returns:
        One adjustment per prediction, in the input order. Players no belief
        touches still appear, with a multiplier of exactly 1.0 — so a caller can
        always read both versions without checking whether one exists.
    """
    by_player: dict[int, list[UserBelief]] = {}
    by_team: dict[int, list[UserBelief]] = {}
    for belief in beliefs:
        target = by_player if belief.entity_type is BeliefEntity.PLAYER else by_team
        target.setdefault(int(belief.entity_id), []).append(belief)

    out: list[BeliefAdjustment] = []
    for prediction in predictions:
        code = int(prediction.player_code)
        applicable = list(by_player.get(code, ()))
        if team_of is not None and by_team:
            applicable.extend(by_team.get(int(team_of.get(prediction.player_code, -1)), ()))

        applicable = [b for b in applicable if b.applies_to(gameweek)]
        if not applicable:
            out.append(
                BeliefAdjustment(
                    player_code=prediction.player_code,
                    raw=prediction,
                    adjusted=prediction,
                    multiplier=1.0,
                )
            )
            continue

        factor = _multiplier(applicable, gameweek)
        out.append(
            BeliefAdjustment(
                player_code=prediction.player_code,
                raw=prediction,
                adjusted=_rescale(prediction, factor, rules),
                multiplier=round(factor, 6),
                beliefs_applied=tuple(applicable),
            )
        )
    return out


def _rescale(prediction: PlayerPrediction, factor: float, rules: ScoringRules) -> PlayerPrediction:
    """Rebuild a prediction with its components scaled, then re-price it.

    Components are scaled and the points **reassembled through the domain**,
    rather than the total being multiplied. Two reasons, and both are load
    bearing:

    - the contract requires the breakdown to sum to the total, so a scaled total
      with an untouched breakdown is invalid by construction;
    - only the domain knows the scoring rules, and a belief must not become a
      second place where points are computed.

    Probabilities are clamped to [0, 1] and minutes to [0, 90], so a strong
    belief cannot push a player past what the game allows. ``expected_points_sd``
    is deliberately **not** reduced: a belief is not evidence, and acting on one
    should not make the projection look more certain than the data made it.
    """
    components = prediction.components
    minutes = components.minutes

    scaled_minutes = minutes.model_copy(
        update={
            "expected_minutes": max(0.0, min(90.0, minutes.expected_minutes * factor)),
            "p_start": max(0.0, min(1.0, minutes.p_start * factor)),
            "p_appearance": max(0.0, min(1.0, minutes.p_appearance * factor)),
            "p_60_plus": max(0.0, min(1.0, minutes.p_60_plus * factor)),
        }
    )
    # The contract's own invariants: starting implies appearing, and reaching 60
    # minutes implies appearing. Scaling each independently can break both.
    scaled_minutes = scaled_minutes.model_copy(
        update={
            "p_start": min(scaled_minutes.p_start, scaled_minutes.p_appearance),
            "p_60_plus": min(scaled_minutes.p_60_plus, scaled_minutes.p_appearance),
        }
    )

    scaled = components.model_copy(
        update={
            "minutes": scaled_minutes,
            "goals": max(0.0, components.goals * factor),
            "assists": max(0.0, components.assists * factor),
            "bonus": max(0.0, components.bonus * factor),
            "clean_sheet_probability": max(
                0.0, min(1.0, components.clean_sheet_probability * factor)
            ),
            "saves": max(0.0, components.saves * factor),
        }
    )

    breakdown = assemble_points(scaled, prediction.position, rules)
    return prediction.model_copy(
        update={
            "components": scaled,
            "breakdown": breakdown,
            "expected_points": breakdown.total,
        }
    )


def belief_sensitivity(
    prediction: PlayerPrediction,
    belief: UserBelief,
    *,
    gameweek: GameweekId,
    rules: ScoringRules,
    steps: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[tuple[float, float], ...]:
    """How the projection moves as the stated confidence is swept.

    Returned as ``(confidence, expected_points)`` pairs. This is what turns "the
    recommendation changed because of your belief" into something a user can
    interrogate: if the transfer flips at confidence 0.4 and the manager only
    feels 0.3 sure, they can see that and say so.
    """
    out: list[tuple[float, float]] = []
    for level in steps:
        probe = belief.model_copy(update={"confidence": level})
        factor = _multiplier([probe], gameweek)
        out.append((level, round(_rescale(prediction, factor, rules).expected_points, 4)))
    return tuple(out)
