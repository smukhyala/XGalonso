"""Squad legality predicates.

Every recommendation passes through these before it is shown to anyone. A
recommendation the game would reject is worse than no recommendation: it costs
the user a transfer to discover.

Violations are returned as a list rather than raised, because a caller
evaluating thousands of candidate transfers wants a cheap yes/no, and a caller
explaining a rejection wants every reason at once.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import product

from xg_alonso.contracts.constraints import SquadViolation
from xg_alonso.contracts.identifiers import TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules

__all__ = [
    "SquadViolation",
    "check_squad",
    "check_starting_xi",
    "is_legal_squad",
    "legal_formations",
]

# `SquadViolation` lives in `contracts` because a simulation *carries* one and
# `contracts` may not import `domain`. Re-exported here, where it is produced,
# so every existing import keeps resolving and there is still exactly one type.


def legal_formations(rules: SquadRules) -> list[tuple[int, int, int, int]]:
    """Every legal ``(GKP, DEF, MID, FWD)`` shape, derived from the rules.

    Read from ``min_play``/``max_play`` rather than hardcoded, so a rule change
    is picked up from the pinned snapshot like every other constant.

    Lives here rather than in :mod:`xg_alonso.optimization.lineup`, where it was
    first written, because it is a statement about squad legality and nothing
    else — it takes only :class:`~xg_alonso.domain.rules.SquadRules` and returns
    shapes, with no reference to points, predictions or search.
    :func:`~xg_alonso.domain.context_features.encode_context` needs it and sits
    below ``optimization`` in the layering, so the choice was to move it down or
    to write it twice. ``optimization`` re-exports it, so every existing import
    still resolves to this one definition.
    """
    bounds = [
        range(rules.rule_for(p).min_play, rules.rule_for(p).max_play + 1)
        for p in (Position.GKP, Position.DEF, Position.MID, Position.FWD)
    ]
    shapes: list[tuple[int, int, int, int]] = []
    for gkp, dfd, mid, fwd in product(*bounds):
        if gkp + dfd + mid + fwd == rules.starting_size:
            shapes.append((gkp, dfd, mid, fwd))
    return shapes


def check_squad(
    picks: Sequence[SquadPick],
    *,
    rules: SquadRules,
    bank: TenthsOfMillion = TenthsOfMillion(0),
    budget: TenthsOfMillion | None = None,
) -> list[SquadViolation]:
    """Every way a 15-player squad breaks the rules. Empty means legal.

    Args:
        budget: The ceiling this squad's value is checked against. Defaults to
            ``rules.total_budget``, which is the *purchase* cap.

    **Why the budget is a parameter.** FPL's spend limit binds when a squad is
    assembled, not forever. A squad whose players appreciate is worth more than
    it cost and is entirely legal — so checking every existing squad against the
    opening budget would flag every successful season as illegal. A simulator
    walking a season passes the squad's own starting value instead. The default
    preserves the original behaviour, so no existing caller changes.
    """
    violations: list[SquadViolation] = []

    if len(picks) != rules.squad_size:
        violations.append(
            SquadViolation(
                rule="squad_size",
                detail=f"squad holds {len(picks)} players, needs {rules.squad_size}",
            )
        )

    codes = [p.player_code for p in picks]
    duplicates = [c for c, n in Counter(codes).items() if n > 1]
    if duplicates:
        violations.append(
            SquadViolation(
                rule="duplicate_player",
                detail=f"player codes appear more than once: {sorted(duplicates)}",
            )
        )

    by_position = Counter(p.position for p in picks)
    for rule in rules.positions:
        have = by_position.get(rule.position, 0)
        if have != rule.squad_select:
            violations.append(
                SquadViolation(
                    rule="position_quota",
                    detail=(
                        f"{rule.position.value}: squad has {have}, needs exactly "
                        f"{rule.squad_select}"
                    ),
                )
            )

    by_club: Counter[TeamId] = Counter(p.team_id for p in picks)
    for team_id, count in sorted(by_club.items()):
        if count > rules.max_per_club:
            violations.append(
                SquadViolation(
                    rule="max_per_club",
                    detail=(
                        f"club {team_id} contributes {count} players, limit is {rules.max_per_club}"
                    ),
                )
            )

    ceiling = rules.total_budget if budget is None else budget
    total = sum(p.selling_price for p in picks) + bank
    if total > ceiling:
        violations.append(
            SquadViolation(
                rule="budget",
                detail=f"squad value plus bank is {total}, budget is {ceiling}",
            )
        )

    return violations


def check_starting_xi(starters: Sequence[SquadPick], *, rules: SquadRules) -> list[SquadViolation]:
    """Every way a starting XI breaks the formation rules. Empty means legal."""
    violations: list[SquadViolation] = []

    if len(starters) != rules.starting_size:
        violations.append(
            SquadViolation(
                rule="starting_size",
                detail=f"XI holds {len(starters)} players, needs {rules.starting_size}",
            )
        )

    by_position: Counter[Position] = Counter(p.position for p in starters)
    for rule in rules.positions:
        have = by_position.get(rule.position, 0)
        if have < rule.min_play or have > rule.max_play:
            violations.append(
                SquadViolation(
                    rule="formation",
                    detail=(
                        f"{rule.position.value}: XI has {have}, allowed "
                        f"{rule.min_play}-{rule.max_play}"
                    ),
                )
            )

    return violations


def is_legal_squad(state: SquadState, *, rules: SquadRules) -> bool:
    """Whether a squad and its starting XI both satisfy every rule."""
    return not check_squad(state.picks, rules=rules, bank=state.bank) and not check_starting_xi(
        state.starters, rules=rules
    )
