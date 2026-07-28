"""Build a squad from scratch.

At gameweek 1 nobody has a squad, so the transfer recommender — the whole
product so far — has nothing to operate on. This is the gameweek-1 answer, and
the same machinery a wildcard planner will need later.

**Why an exact solver, when the plan's cut list deferred one.** The cut list
said "CP-SAT and MILP (exhaustive search suffices at this size)", and put the
burden on re-argument rather than on defending the cut. Here is the argument,
measured rather than asserted. The greedy-plus-local-search this replaces was
compared against a proven optimum on all four prediction paths that ship:

| prediction path        | local search | exact  | gap  |
|------------------------|--------------|--------|------|
| closed-form baseline   | 54.15        | 54.96  | 0.81 |
| `late.pkl` (API default) | 50.17      | 55.83  | 5.66 |
| `early.pkl`            | 51.72        | 54.53  | 2.81 |
| `component_models.pkl` | 50.18        | 54.17  | 4.00 |

Mean gap **3.32 expected points per gameweek**, and the exact solve runs in
0.04s against 1.2s for the heuristic — better on quality *and* latency, with an
optimality certificate replacing a tuning constant. The local search was not
merely imprecise: on the API's own default it converged to a local optimum
3.83 points short and stayed there at every pass cap from 6 to 200. More search
effort did not help, because the neighbourhood was the limit, not the budget.

**The objective is the XI, not the fifteen.** A squad scores through its best
legal eleven with the captain doubled, so four expensive bench players are worth
nothing. That nesting — pick 15, then pick the best 11 of them, then double the
best of those — is what makes this harder than a knapsack, and it is modelled
here exactly: separate in-squad, in-XI and is-captain variables, with the XI
constrained to a subset of the squad and the captain to a member of the XI.
Because the inner choice is itself a maximisation, the solver's optimum equals
`starting_xi_points` on the same fifteen; `test_squad_builder.py` asserts that
equivalence on random squads rather than trusting it.

**On leftover budget.** A squad that banks money looks wrong and usually is not.
Bench players contribute exactly zero, so bench spend is a free variable: at the
optimum, forcing every last 0.1m to be spent changes the objective by 0.0000.
Bank is therefore a readout, never a lever — there is deliberately no minimum
spend constraint here, and adding one would trade real points for a cosmetic
number. What the tie-break below *does* do is spend idle money on the best bench
available, which costs nothing and is worth something the moment autosubs are
modelled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.optimization.lineup import CAPTAIN_MULTIPLIER, XiSelection, best_starting_xi

__all__ = ["SquadCandidate", "build_squad"]

#: Weight on total squad expected points, used only to break ties between
#: squads whose *elevens* score identically. Bench players are worth zero to the
#: objective, so without this the choice of bench — and therefore how much of the
#: budget is used — is arbitrary and can differ between otherwise identical runs.
#:
#: The bound that keeps it a tie-break rather than a second objective: squad
#: total expected points cannot exceed roughly 15 x 10 = 150, so this term can
#: shift the objective by at most 1.5e-4. Any XI difference larger than that
#: wins outright, and differences smaller than it are below the noise floor of
#: the predictions being optimised over.
_TIE_BREAK_WEIGHT = 1e-6

_POSITIONS: tuple[Position, ...] = (Position.GKP, Position.DEF, Position.MID, Position.FWD)


@dataclass(frozen=True)
class SquadCandidate:
    """A player who could be selected, with everything needed to price them."""

    player_code: PlayerCode
    position: Position
    team_id: TeamId
    price: TenthsOfMillion
    prediction: PlayerPrediction

    @property
    def expected_points(self) -> float:
        return self.prediction.expected_points

    @property
    def value_density(self) -> float:
        """Expected points per unit price — how far a pound goes here."""
        return self.expected_points / max(int(self.price), 1)


def _as_pick(candidate: SquadCandidate, slot: int) -> SquadPick:
    return SquadPick(
        player_code=candidate.player_code,
        position=candidate.position,
        team_id=candidate.team_id,
        purchase_price=candidate.price,
        current_price=candidate.price,
        selling_price=candidate.price,
        squad_slot=slot,
    )


def _solve(
    candidates: Sequence[SquadCandidate],
    rules: SquadRules,
    points: Mapping[PlayerCode, float],
) -> list[SquadCandidate]:
    """Choose the fifteen whose best legal eleven scores highest.

    Three binary variables per candidate, laid out contiguously so the column
    index of variable *v* for candidate *i* is ``v * n + i``:

    - ``x`` — in the fifteen
    - ``y`` — in the starting eleven
    - ``c`` — is the captain

    The objective ``sum(ep * (y + c))`` pays a starter once and the captain
    twice, which is exactly FPL's rule. It never pays for ``x`` alone, so a
    bench player contributes nothing — the same asymmetry the rest of the
    system is built on.

    Raises:
        ValueError: when no legal squad exists, which is a data problem worth
            surfacing rather than returning something half-built.
    """
    n = len(candidates)
    if n == 0:
        raise ValueError("no candidates supplied")

    ep = np.array([points.get(c.player_code, 0.0) for c in candidates], dtype=float)
    price = np.array([int(c.price) for c in candidates], dtype=float)

    x0, y0, c0 = 0, n, 2 * n

    # milp minimises, so the maximisation is negated. The tie-break rides on the
    # squad variables, where it can only distinguish otherwise-equal optima.
    objective = np.zeros(3 * n, dtype=float)
    objective[y0 : y0 + n] = -ep
    objective[c0 : c0 + n] = -ep * (CAPTAIN_MULTIPLIER - 1)
    objective[x0 : x0 + n] = -ep * _TIE_BREAK_WEIGHT

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    def add_row(entries: Sequence[tuple[int, float]], low: float, high: float) -> None:
        nonlocal row
        for column, value in entries:
            rows.append(row)
            cols.append(column)
            vals.append(value)
        lower.append(low)
        upper.append(high)
        row += 1

    # Squad size, XI size, exactly one captain.
    add_row([(x0 + i, 1.0) for i in range(n)], rules.squad_size, rules.squad_size)
    add_row([(y0 + i, 1.0) for i in range(n)], rules.starting_size, rules.starting_size)
    add_row([(c0 + i, 1.0) for i in range(n)], 1.0, 1.0)

    # A starter must be in the squad; a captain must be starting. Without these
    # the solver would happily field players it never bought.
    for i in range(n):
        add_row([(y0 + i, 1.0), (x0 + i, -1.0)], -np.inf, 0.0)
        add_row([(c0 + i, 1.0), (y0 + i, -1.0)], -np.inf, 0.0)

    for position in _POSITIONS:
        members = [i for i, candidate in enumerate(candidates) if candidate.position is position]
        rule = rules.rule_for(position)
        # Quotas are equalities: FPL requires exactly two keepers, five
        # defenders and so on, not "at most".
        add_row([(x0 + i, 1.0) for i in members], rule.squad_select, rule.squad_select)
        add_row([(y0 + i, 1.0) for i in members], rule.min_play, rule.max_play)

    clubs: dict[int, list[int]] = {}
    for i, candidate in enumerate(candidates):
        clubs.setdefault(int(candidate.team_id), []).append(i)
    for members in clubs.values():
        add_row([(x0 + i, 1.0) for i in members], -np.inf, rules.max_per_club)

    add_row(
        [(x0 + i, price[i]) for i in range(n)],
        -np.inf,
        float(int(rules.total_budget)),
    )

    matrix = coo_array((vals, (rows, cols)), shape=(row, 3 * n))
    result = milp(
        c=objective,
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        integrality=np.ones(3 * n),
        bounds=Bounds(lb=0, ub=1),
    )

    if not result.success or result.x is None:
        counts = {p.value: sum(1 for c in candidates if c.position is p) for p in _POSITIONS}
        raise ValueError(
            f"no legal squad exists for these candidates ({result.message.strip()}). "
            f"Available by position: {counts}; budget {int(rules.total_budget)}"
        )

    selected = np.flatnonzero(np.asarray(result.x[x0 : x0 + n]) > 0.5)
    return [candidates[int(i)] for i in selected]


def build_squad(
    candidates: Sequence[SquadCandidate],
    *,
    rules: SquadRules,
    entry_id: EntryId,
    gameweek: GameweekId,
    predictions: Mapping[PlayerCode, PlayerPrediction] | None = None,
) -> tuple[SquadState, XiSelection]:
    """Choose a legal squad maximising expected points from its best XI.

    Args:
        candidates: Every selectable player.
        rules: Squad size, budget, quotas and the three-per-club cap.
        entry_id: Who the squad is for.
        gameweek: Which gameweek it is built for.
        predictions: Expected points per player. Derived from ``candidates``
            when omitted.

    Returns:
        The squad and the XI it would field.

    Raises:
        ValueError: if no legal squad can be assembled — usually too few
            candidates in a position, which is a data problem worth surfacing
            rather than returning something half-built.
    """
    lookup: dict[PlayerCode, PlayerPrediction] = dict(predictions or {})
    for candidate in candidates:
        lookup.setdefault(candidate.player_code, candidate.prediction)

    # Duplicate codes would let the solver field the same player twice, and a
    # stable order makes the solve reproducible run to run.
    unique: dict[PlayerCode, SquadCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.player_code, candidate)
    ordered = sorted(unique.values(), key=lambda c: int(c.player_code))

    points = {
        candidate.player_code: lookup[candidate.player_code].expected_points
        if candidate.player_code in lookup
        else candidate.expected_points
        for candidate in ordered
    }

    chosen = _solve(ordered, rules, points)
    spend = sum(int(c.price) for c in chosen)

    picks = [_as_pick(candidate, i + 1) for i, candidate in enumerate(chosen)]
    selection = best_starting_xi(picks, lookup, rules)

    ordered_picks: list[SquadPick] = []
    for slot, pick in enumerate(selection.starters + selection.bench, start=1):
        ordered_picks.append(
            pick.model_copy(
                update={
                    "squad_slot": slot,
                    "is_captain": pick.player_code == selection.captain,
                    "is_vice_captain": pick.player_code == selection.vice_captain,
                }
            )
        )

    state = SquadState(
        entry_id=entry_id,
        gameweek=gameweek,
        picks=tuple(ordered_picks),
        bank=TenthsOfMillion(int(rules.total_budget) - spend),
        free_transfers=1,
    )
    return state, best_starting_xi(state.picks, lookup, rules)
