"""Scoring a squad under a manager's objective, and honouring their constraints.

The existing optimizer maximises one thing: expected points, minus a hit, minus a
fixed multiple of uncertainty. That is the right objective for a manager
maximising expected points and the *wrong* one for everybody else — most
obviously for a manager forty points behind, for whom a certain average is the
one outcome that guarantees staying behind.

This module supplies the two seams the objective needs, and deliberately does not
fork :mod:`xg_alonso.optimization.transfer`:

:func:`objective_value`
    The scalar a candidate is ranked by. Expected points remain the base term;
    the objective changes how variance, ownership, captaincy and price are
    priced on top of it.

:func:`constraint_filter`
    Removes candidates a hard constraint forbids, **before** anything is scored.
    A locked player is not a large penalty — he is not on the market, and
    implementing "keep Haaland" as a penalty means that with a good enough
    alternative the optimizer sells him anyway, which is precisely what the user
    said not to do.

**Opportunity cost is measured, not hidden.** :func:`opportunity_cost` reports
what a lock gave up, so a constraint's effect stays separable from a prediction's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from xg_alonso.contracts.identifiers import PlayerCode, TenthsOfMillion
from xg_alonso.contracts.objective import (
    ManagerConstraints,
    ManagerObjective,
    OwnershipPreference,
    PrimaryMetric,
)
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.squad import SquadPick, SquadState

__all__ = [
    "ConstraintReport",
    "ObjectiveContext",
    "constraint_filter",
    "objective_value",
    "objective_valued",
    "opportunity_cost",
    "validate_constraints",
]

#: Ownership share above which a player counts as template. Applied to the
#: *share of the pool* rather than to a raw `selected` count, so the threshold
#: does not drift as the game's total player base changes.
_TEMPLATE_SHARE: Final[float] = 0.25


@dataclass(frozen=True)
class ObjectiveContext:
    """Per-player context the objective needs beyond the prediction itself."""

    ownership: Mapping[PlayerCode, float] = field(default_factory=dict)
    """Share of managers owning each player, in [0, 1].

    A player absent from this mapping has *unknown* ownership and contributes
    nothing to the ownership term. Defaulting him to the league average would
    invent a fact: an unowned player and an average-owned one are exactly the
    distinction a differential objective exists to make.
    """

    prices: Mapping[PlayerCode, TenthsOfMillion] = field(default_factory=dict)
    transfer_momentum: Mapping[PlayerCode, float] = field(default_factory=dict)


def objective_value(
    prediction: PlayerPrediction,
    *,
    objective: ManagerObjective,
    context: ObjectiveContext | None = None,
    is_captain: bool = False,
) -> float:
    """What one player is worth *to this manager*.

    Expected points is the base. Every other term is the objective's own view of
    what else matters:

    **Variance carries a sign.** ``signed_uncertainty_penalty`` is positive for a
    conservative objective and **negative** for an aggressive one, so a chaser is
    rewarded for volatility rather than merely penalised less for it. This is the
    single most important line in the module: without it, "aggressive" degrades
    to "slightly less cautious" and every objective recommends the same squad.

    **Ownership carries a sign too.** A rank-protector gains from owning what the
    field owns; a chaser loses by it. The same number, opposite signs.

    **Captaincy is scored on the ceiling, gated by playing.** A captain's value is
    the upside, and upside a player never reaches because he is on the bench is
    not upside. ``p_60_plus`` gates it.
    """
    ctx = context or ObjectiveContext()
    value = prediction.expected_points

    # Variance, signed by risk appetite.
    value -= objective.signed_uncertainty_penalty * prediction.expected_points_sd

    # Ownership, signed by preference.
    share = ctx.ownership.get(prediction.player_code)
    if share is not None and objective.ownership_preference is not OwnershipPreference.NEUTRAL:
        sign = objective.ownership_preference.ownership_sign
        # Scaled by the player's own points: owning a template *bench* player
        # neither protects nor costs a rank, so the adjustment has to be
        # proportional to what he actually scores.
        value += sign * share * prediction.expected_points * 0.5

    if is_captain:
        ceiling = prediction.expected_points + 1.5 * prediction.expected_points_sd
        secure = prediction.components.minutes.p_60_plus
        value += objective.captaincy_weight * (ceiling * secure - prediction.expected_points)

    if objective.primary_metric is PrimaryMetric.TEAM_VALUE_GROWTH or objective.team_value_weight:
        momentum = ctx.transfer_momentum.get(prediction.player_code, 0.0)
        value += objective.team_value_weight * momentum

    if objective.primary_metric is PrimaryMetric.DOWNSIDE_PROTECTION:
        # A crude lower quantile: mean minus one standard deviation, floored at
        # zero because a player cannot score negatively often enough for the
        # normal approximation to hold in the left tail.
        floor = max(0.0, prediction.expected_points - prediction.expected_points_sd)
        value = 0.5 * value + 0.5 * floor

    if objective.primary_metric is PrimaryMetric.DIFFERENTIAL_YIELD and share is not None:
        value *= 1.0 + (1.0 - min(1.0, share / _TEMPLATE_SHARE)) * 0.3

    return value


@dataclass(frozen=True)
class ConstraintReport:
    """What the constraints removed, and why. Never silent."""

    excluded_players: tuple[PlayerCode, ...] = ()
    locked_players: tuple[PlayerCode, ...] = ()
    locked_positions: tuple[Position, ...] = ()
    excluded_teams: tuple[int, ...] = ()
    removed_candidates: int = 0
    protected_picks: tuple[PlayerCode, ...] = ()

    @property
    def is_binding(self) -> bool:
        return bool(
            self.excluded_players
            or self.locked_players
            or self.locked_positions
            or self.excluded_teams
        )

    def explain(self) -> tuple[str, ...]:
        """Plain statements of what the manager's constraints did to the search."""
        lines: list[str] = []
        if self.locked_players:
            lines.append(
                f"{len(self.locked_players)} player(s) held by your instruction and never "
                "offered for sale"
            )
        if self.locked_positions:
            names = ", ".join(p.value for p in self.locked_positions)
            lines.append(f"no transfers considered in: {names}")
        if self.excluded_players:
            lines.append(f"{len(self.excluded_players)} player(s) excluded from the buy list")
        if self.excluded_teams:
            lines.append(f"{len(self.excluded_teams)} club(s) excluded")
        if self.removed_candidates:
            lines.append(
                f"{self.removed_candidates} otherwise-legal option(s) removed by your constraints"
            )
        return tuple(lines)


def validate_constraints(constraints: ManagerConstraints, squad: SquadState) -> tuple[str, ...]:
    """Problems that would make a constrained search vacuous. Empty means workable.

    Run **before** discovery or optimization, per the brief. A constraint set with
    no solution should say so immediately rather than surface as an empty
    recommendation twenty seconds later, which reads as "nothing was worth doing"
    and means "nothing was possible".
    """
    problems: list[str] = []
    owned = {p.player_code for p in squad.picks}

    missing = [c for c in constraints.locked_players if c not in owned]
    if missing:
        problems.append(
            f"locked players {sorted(int(c) for c in missing)} are not in the squad, so "
            "they cannot be kept"
        )

    held = set(constraints.locked_players)
    frozen_positions = constraints.locked_position_set()
    movable = [
        p for p in squad.picks if p.player_code not in held and p.position not in frozen_positions
    ]
    if not movable:
        problems.append(
            "every squad member is locked by a player lock or a position lock, so no "
            "transfer is possible at all"
        )

    if constraints.minimum_bank > squad.bank and constraints.max_transfers != 0:
        problems.append(
            f"minimum bank {constraints.minimum_bank} exceeds the current bank {squad.bank}, "
            "so only money-raising transfers can satisfy it"
        )

    return tuple(problems)


def constraint_filter(
    squad: SquadState,
    *,
    constraints: ManagerConstraints,
    candidate_codes: Sequence[PlayerCode],
    candidate_teams: Mapping[PlayerCode, int],
) -> tuple[frozenset[PlayerCode], frozenset[PlayerCode], ConstraintReport]:
    """Which squad members may leave, and which candidates may be bought.

    Returns ``(sellable, buyable, report)``. Applied **before** scoring, so a
    forbidden move is never ranked — a penalty large enough to usually prevent a
    sale is still a penalty a good enough alternative can overcome, and "usually
    honours your constraint" is not honouring it.
    """
    held = set(constraints.locked_players)
    frozen_positions = constraints.locked_position_set()
    protected: list[PlayerCode] = []

    sellable: set[PlayerCode] = set()
    for pick in squad.picks:
        if pick.player_code in held or pick.position in frozen_positions:
            protected.append(pick.player_code)
            continue
        sellable.add(pick.player_code)

    excluded_players = set(constraints.excluded_players)
    excluded_teams = set(constraints.excluded_teams)
    owned = {p.player_code for p in squad.picks}

    buyable: set[PlayerCode] = set()
    removed = 0
    for code in candidate_codes:
        if code in owned:
            continue
        if code in excluded_players or candidate_teams.get(code, -1) in excluded_teams:
            removed += 1
            continue
        buyable.add(code)

    return (
        frozenset(sellable),
        frozenset(buyable),
        ConstraintReport(
            excluded_players=tuple(sorted(excluded_players)),
            locked_players=tuple(sorted(held)),
            locked_positions=tuple(sorted(frozen_positions, key=lambda p: p.value)),
            excluded_teams=tuple(sorted(excluded_teams)),
            removed_candidates=removed,
            protected_picks=tuple(protected),
        ),
    )


def opportunity_cost(
    locked: SquadPick,
    *,
    predictions: Mapping[PlayerCode, PlayerPrediction],
    candidates: Sequence[PlayerCode],
    prices: Mapping[PlayerCode, TenthsOfMillion],
    budget: TenthsOfMillion,
    objective: ManagerObjective,
    context: ObjectiveContext | None = None,
) -> tuple[PlayerCode | None, float]:
    """What holding a locked player gave up: the best affordable alternative.

    Returned rather than acted on. The brief asks that a locked player be kept,
    the squad optimised around him, and the **opportunity cost surfaced** — three
    separate requirements, and the third is what keeps a constraint honest. A
    user who is told "keeping Haaland cost you 1.2 projected points" can decide
    whether they still mean it; one who is told nothing cannot.

    Returns ``(best_alternative, points_forgone)``. Forgone is zero when nothing
    affordable would have scored more, which is a real and common answer.
    """
    current = predictions.get(locked.player_code)
    if current is None:
        return None, 0.0

    ctx = context or ObjectiveContext()
    held_value = objective_value(current, objective=objective, context=ctx)

    best: PlayerCode | None = None
    best_value = held_value
    for code in candidates:
        if code == locked.player_code:
            continue
        prediction = predictions.get(code)
        price = prices.get(code)
        if prediction is None or price is None:
            continue
        if prediction.position is not locked.position:
            continue
        if price > budget:
            continue
        value = objective_value(prediction, objective=objective, context=ctx)
        if value > best_value:
            best, best_value = code, value

    return best, round(max(0.0, best_value - held_value), 6)


def objective_valued(
    predictions: Mapping[PlayerCode, PlayerPrediction],
    *,
    objective: ManagerObjective,
    context: ObjectiveContext | None = None,
) -> dict[PlayerCode, PlayerPrediction]:
    """Re-price every prediction under the objective, for the search to consume.

    **Why substitute the predictions rather than thread a scorer through the
    search.** This is the same argument :func:`~xg_alonso.optimization.transfer.
    horizon_valued` already makes, and it applies with more force here. A
    transfer is chosen by scoring the eleven a squad would field, and that
    scoring runs through ``starting_xi_points``, captaincy and the bench in
    several places. Threading an objective through each of them would leave some
    paths using it and others not — and the ones that did not would be exactly
    the subtle cases, a captain chosen on raw points while the squad around him
    was chosen on rank gain.

    Swapping the number every one of those paths already reads is a single
    substitution point that **cannot be partially applied**.

    The breakdown is scaled with the total so the two still agree, which the
    prediction contract enforces. ``expected_points_sd`` is left alone: the
    objective changes how variance is *priced*, not how much of it there is, and
    shrinking it here would double-count the risk preference.
    """
    ctx = context or ObjectiveContext()
    out: dict[PlayerCode, PlayerPrediction] = {}

    for code, prediction in predictions.items():
        value = objective_value(prediction, objective=objective, context=ctx)
        base = prediction.expected_points
        factor = (value / base) if abs(base) > 1e-9 else 0.0
        breakdown = prediction.breakdown
        out[code] = prediction.model_copy(
            update={
                "breakdown": breakdown.model_copy(
                    update={
                        field_name: getattr(breakdown, field_name) * factor
                        for field_name in (
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
                    }
                ),
                "expected_points": value,
            }
        )
    return out
