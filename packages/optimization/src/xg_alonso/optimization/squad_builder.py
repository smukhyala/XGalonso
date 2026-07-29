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
Bench players contribute exactly zero to the objective, so bench spend is a free
variable: at the optimum, forcing every last 0.1m to be spent changes the score
by 0.0000. Bank is therefore a readout, never a lever — there is deliberately no
minimum spend constraint here, and adding one would trade real points for a
cosmetic number.

What the tie-break does with that free variable is buy the *cheapest bench that
can still autosub*, rather than the highest-scoring one. A substitute only ever
scores when a starter is withdrawn or Bench Boost is played, and the chip is not
modelled (D5) — so paying for bench quality spends the eleven's budget on points
that are almost never collected. The eleven is unaffected either way; the
difference is whether the remainder sits in a fourth substitute or in the bank,
and the bank is a transfer a manager can actually make.
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
from xg_alonso.contracts.objective import Requirement, SquadRequirements
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.optimization.lineup import CAPTAIN_MULTIPLIER, XiSelection, best_starting_xi
from xg_alonso.optimization.requirements import (
    ConstraintRow,
    RequirementOutcome,
    as_requirements,
    compile_requirements,
)

__all__ = ["ConstrainedBuild", "SquadCandidate", "build_constrained_squad", "build_squad"]

#: Weight on the bench tie-break, used only to choose between squads whose
#: *elevens* score identically.
#:
#: **This used to maximise total squad points, and that was the wrong tie-break.**
#: A bench player scores nothing unless a starter is withdrawn or the Bench Boost
#: chip is played, and the chip is not modelled at all (D5). Preferring the
#: highest-scoring bench therefore spent real money buying points that, in the
#: overwhelming majority of gameweeks, are never collected — and it spent that
#: money out of the same budget the eleven draws on.
#:
#: The eleven itself does not change: the solver already maximises it subject to
#: the whole-squad budget, so the optimum was correct before and is correct now.
#: What changes is where the leftover goes. It now comes back as bank, which is
#: transfer flexibility a manager can actually use, instead of sitting in a
#: fourth substitute who will not play.
#:
#: The bound that keeps it a tie-break rather than a second objective: a bench
#: of four cannot cost more than about 4 x 150 = 600, so this term can shift the
#: objective by at most 6e-5. Any XI difference larger than that wins outright,
#: and differences smaller than it are far below the noise floor of the
#: predictions being optimised over.
_BENCH_COST_WEIGHT = 1e-7

#: What a bench player who is certain to play is worth, in price units, against
#: one who is certain not to.
#:
#: A bench is not merely dead weight: when a starter is withdrawn before kick-off
#: an autosub takes his place, and a substitute who never plays cannot. So the
#: cheapest possible bench is not the best one — five tenths of a million is
#: worth paying for a substitute who actually appears. Expressed in price units
#: so it trades against cost in the same currency rather than through a second
#: opaque weight.
_BENCH_PLAYS_BONUS = 5.0

#: Weight on spending the budget *on the eleven*, used only to break ties.
#:
#: **Unspent money at gameweek 1 scores nothing.** Before the first deadline
#: transfers are unlimited, so there is no future move for a bank balance to
#: fund — it is simply budget that was not converted into players. That is not
#: true mid-season, where money buys flexibility, which is why this is a
#: squad-build tie-break and not a change to the objective.
#:
#: It only ever decides between elevens the model scores as equal, and the model
#: is *measurably* least able to tell them apart at exactly the prices this
#: pushes toward: on a held-out season its rank correlation falls from 0.67 in
#: the budget tier to 0.31 above £11m, while it under-projects the £8-11m tier
#: by 0.35 points a gameweek. Where it claims indifference between a cheap
#: player and an expensive one, the expensive one is the better bet, because the
#: model's own error is in that direction.
#:
#: Bounded like every other tie-break here: an eleven cannot cost more than
#: about 11 x 150 = 1650, so this shifts the objective by at most 1.6e-4.
_XI_SPEND_WEIGHT = 1e-7

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


@dataclass(frozen=True)
class _Solution:
    """What the MILP chose: the fifteen, the eleven, and the armband."""

    squad: list[SquadCandidate]
    starters: frozenset[PlayerCode]
    captain: PlayerCode | None


def _solve(
    candidates: Sequence[SquadCandidate],
    rules: SquadRules,
    points: Mapping[PlayerCode, float],
    extra: Sequence[ConstraintRow] = (),
    *,
    strict: bool = True,
) -> _Solution | None:
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

    Args:
        extra: Requirement rows over the same variables. Hard bounds, never
            penalties — see :mod:`.requirements`.
        strict: Raise when no legal squad exists. Set ``False`` by the
            relaxation search, which needs "infeasible" as an answer rather
            than an exception so it can drop a requirement and try again.

    Raises:
        ValueError: when no legal squad exists and ``strict``. Without
            requirements that is a data problem worth surfacing rather than
            returning something half-built.
    """
    n = len(candidates)
    if n == 0:
        raise ValueError("no candidates supplied")

    ep = np.array([points.get(c.player_code, 0.0) for c in candidates], dtype=float)
    price = np.array([int(c.price) for c in candidates], dtype=float)
    appearance = np.array(
        [c.prediction.components.minutes.p_appearance for c in candidates], dtype=float
    )

    x0, y0, c0 = 0, n, 2 * n

    # milp minimises, so the maximisation is negated.
    objective = np.zeros(3 * n, dtype=float)
    objective[y0 : y0 + n] = -ep
    objective[c0 : c0 + n] = -ep * (CAPTAIN_MULTIPLIER - 1)

    # Bench tie-break. A player is on the bench exactly when he is in the squad
    # and not in the XI, so putting +cost on the squad variable and -cost on the
    # XI variable charges it to bench members only and cancels for starters —
    # no extra variable needed for a quantity that is already implied.
    bench_cost = (price - _BENCH_PLAYS_BONUS * appearance) * _BENCH_COST_WEIGHT
    objective[x0 : x0 + n] += bench_cost
    objective[y0 : y0 + n] -= bench_cost

    # Reward spending on the eleven, so a tie is broken toward the squad that
    # converts more of the budget into players who actually score. Applied to
    # the XI variable only, so it never argues for an expensive bench.
    objective[y0 : y0 + n] -= price * _XI_SPEND_WEIGHT

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

    # Requirement rows last, so a manager's constraints sit on top of the
    # league's rather than competing with them.
    for extra_row in extra:
        add_row(list(extra_row.entries), extra_row.lower, extra_row.upper)

    matrix = coo_array((vals, (rows, cols)), shape=(row, 3 * n))
    result = milp(
        c=objective,
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        integrality=np.ones(3 * n),
        bounds=Bounds(lb=0, ub=1),
    )

    if not result.success or result.x is None:
        if not strict:
            return None
        counts = {p.value: sum(1 for c in candidates if c.position is p) for p in _POSITIONS}
        raise ValueError(
            f"no legal squad exists for these candidates ({result.message.strip()}). "
            f"Available by position: {counts}; budget {int(rules.total_budget)}"
        )

    values = np.asarray(result.x)
    selected = np.flatnonzero(values[x0 : x0 + n] > 0.5)
    starting = np.flatnonzero(values[y0 : y0 + n] > 0.5)
    wearing = np.flatnonzero(values[c0 : c0 + n] > 0.5)

    # The solver's own eleven is returned, not just its fifteen. Re-deriving the
    # XI afterwards with `best_starting_xi` silently discards every requirement
    # that binds `y` or `c`, which turns "must start" into "must be in the
    # squad" and drops a required captaincy entirely.
    return _Solution(
        squad=[candidates[int(i)] for i in selected],
        starters=frozenset(candidates[int(i)].player_code for i in starting),
        captain=candidates[int(wearing[0])].player_code if wearing.size else None,
    )


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

    solution = _solve(ordered, rules, points)
    assert solution is not None
    return _assemble(
        solution.squad, lookup, rules, entry_id=entry_id, gameweek=gameweek, solution=solution
    )


def _assemble(
    chosen: Sequence[SquadCandidate],
    lookup: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    *,
    entry_id: EntryId,
    gameweek: GameweekId,
    solution: _Solution | None = None,
) -> tuple[SquadState, XiSelection]:
    """Turn a chosen fifteen into a squad with its slots and armbands set.

    ``solution`` pins the eleven and the captain the solver chose. Without it
    the eleven is re-derived, which is right for an unconstrained build — the
    two agree, and `test_solver_objective_equals_repo_scorer` asserts it — but
    wrong the moment a requirement binds the XI.
    """
    spend = sum(int(c.price) for c in chosen)
    picks = [_as_pick(candidate, i + 1) for i, candidate in enumerate(chosen)]
    selection = (
        _selection_from(picks, lookup, rules, solution)
        if solution is not None
        else best_starting_xi(picks, lookup, rules)
    )

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
    final = (
        _selection_from(list(state.picks), lookup, rules, solution)
        if solution is not None
        else best_starting_xi(state.picks, lookup, rules)
    )
    return state, final


def _selection_from(
    picks: Sequence[SquadPick],
    lookup: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    solution: _Solution,
) -> XiSelection:
    """Build the selection the solver chose, bench ordered by expected points.

    The vice-captain is the best starter who is not the captain: FPL's rule, and
    the same one `best_starting_xi` applies. Only the *choice* of eleven and
    captain comes from the solver; everything derived from that choice is still
    computed here so the two paths cannot describe the same squad differently.
    """
    starters = [p for p in picks if p.player_code in solution.starters]
    bench = [p for p in picks if p.player_code not in solution.starters]

    def points_of(pick: SquadPick) -> float:
        prediction = lookup.get(pick.player_code)
        return float(prediction.expected_points) if prediction else 0.0

    bench.sort(key=lambda p: (p.position is Position.GKP, -points_of(p)))

    captain = solution.captain
    ranked = sorted(starters, key=lambda p: -points_of(p))
    vice = next((p.player_code for p in ranked if p.player_code != captain), None)

    shape = tuple(sum(1 for p in starters if p.position is position) for position in _POSITIONS)
    total = sum(points_of(p) for p in starters)
    if captain is not None:
        total += points_of(next(p for p in starters if p.player_code == captain)) * (
            CAPTAIN_MULTIPLIER - 1
        )

    return XiSelection(
        starters=tuple(starters),
        bench=tuple(bench),
        captain=captain,
        vice_captain=vice,
        formation=(shape[0], shape[1], shape[2], shape[3]),
        expected_points=total,
    )


@dataclass(frozen=True)
class ConstrainedBuild:
    """A squad built to a manager's requirements, and what they cost.

    ``relaxed`` is the part worth reading first. It is empty when everything
    the manager asked for held; when it is not, those requirements are the ones
    that had to give for a legal squad to exist at all.
    """

    squad: SquadState
    selection: XiSelection
    outcomes: tuple[RequirementOutcome, ...]
    expected_points: float
    unconstrained_points: float

    @property
    def relaxed(self) -> tuple[Requirement, ...]:
        return tuple(o.requirement for o in self.outcomes if not o.honoured)

    @property
    def feasible_as_asked(self) -> bool:
        return not self.relaxed

    @property
    def total_cost(self) -> float:
        """Points given up against the unconstrained optimum."""
        return round(self.unconstrained_points - self.expected_points, 4)


def _xi_points(
    solution: _Solution,
    lookup: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
) -> float:
    """What a solution scores through the eleven it actually chose.

    Scored from the solver's own XI rather than a re-derived best eleven,
    because under a requirement those differ — and pricing a constrained squad
    against an unconstrained scoring of it would report the cost of a lineup
    nobody is fielding.
    """
    picks = [_as_pick(candidate, i + 1) for i, candidate in enumerate(solution.squad)]
    return float(_selection_from(picks, lookup, rules, solution).expected_points)


def build_constrained_squad(
    candidates: Sequence[SquadCandidate],
    *,
    rules: SquadRules,
    entry_id: EntryId,
    gameweek: GameweekId,
    requirements: SquadRequirements | None = None,
    predictions: Mapping[PlayerCode, PlayerPrediction] | None = None,
    price_requirements: bool = True,
) -> ConstrainedBuild:
    """Build the best legal squad that honours what the manager asked for.

    Requirements are hard bounds on the same variables the solver already uses,
    so a request is either honoured exactly or reported as impossible — never
    approximated. When the full set has no solution, requirements are dropped in
    :meth:`SquadRequirements.relaxation_order` until one does, and every dropped
    requirement is named.

    Args:
        price_requirements: Also measure what each honoured requirement cost, by
            re-solving without it. One extra solve per requirement — cheap at
            0.04s each, and it is the number that answers "is keeping him worth
            it". Turn off for a hot path that only needs the squad.

    Raises:
        ValueError: if no legal squad exists even with every requirement
            dropped. That is a data problem, not a requirements problem.
    """
    bundle = requirements or SquadRequirements()

    lookup: dict[PlayerCode, PlayerPrediction] = dict(predictions or {})
    for candidate in candidates:
        lookup.setdefault(candidate.player_code, candidate.prediction)

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

    codes = [c.player_code for c in ordered]
    positions = [c.position for c in ordered]
    teams = [int(c.team_id) for c in ordered]
    prices = [int(c.price) for c in ordered]

    def rows_for(wanted: Sequence[Requirement]) -> list[ConstraintRow]:
        compiled, _ = compile_requirements(
            wanted,
            codes=codes,
            positions=positions,
            teams=teams,
            prices=prices,
            total_budget=int(rules.total_budget),
        )
        return compiled

    asked = as_requirements(bundle)
    _, unmet = compile_requirements(
        asked,
        codes=codes,
        positions=positions,
        teams=teams,
        prices=prices,
        total_budget=int(rules.total_budget),
    )
    inexpressible = {id(requirement): note for requirement, note in unmet}

    # The free optimum, for pricing the whole requirement set against.
    free = _solve(ordered, rules, points)
    assert free is not None
    unconstrained = _xi_points(free, lookup, rules)

    # Relax in the manager's declared order until a legal squad exists. Exact
    # rather than inferred: every drop is a re-solve, and at 0.04s a handful of
    # them is cheaper than explaining a wrong answer.
    expressible = [r for r in asked if id(r) not in inexpressible]
    order = [r for r in bundle.relaxation_order() if r in expressible]
    order.extend(r for r in expressible if r not in order)

    kept = list(expressible)
    dropped: list[Requirement] = []
    chosen = _solve(ordered, rules, points, rows_for(kept), strict=not kept)

    for candidate_to_drop in order:
        if chosen is not None:
            break
        kept.remove(candidate_to_drop)
        dropped.append(candidate_to_drop)
        chosen = _solve(ordered, rules, points, rows_for(kept), strict=not kept)

    if chosen is None:  # pragma: no cover - the empty set is the base solve
        message = "no legal squad exists even with every requirement dropped"
        raise ValueError(message)

    achieved = _xi_points(chosen, lookup, rules)

    outcomes: list[RequirementOutcome] = []
    for requirement in asked:
        note = inexpressible.get(id(requirement))
        if note is not None:
            outcomes.append(RequirementOutcome(requirement=requirement, honoured=False, note=note))
            continue
        if requirement in dropped:
            outcomes.append(
                RequirementOutcome(
                    requirement=requirement,
                    honoured=False,
                    note="dropped so that a legal squad could exist",
                )
            )
            continue

        cost: float | None = None
        if price_requirements:
            others = [r for r in kept if r is not requirement]
            without = _solve(ordered, rules, points, rows_for(others), strict=False)
            if without is not None:
                cost = round(_xi_points(without, lookup, rules) - achieved, 4)
        outcomes.append(RequirementOutcome(requirement=requirement, honoured=True, cost=cost))

    squad, selection = _assemble(
        chosen.squad, lookup, rules, entry_id=entry_id, gameweek=gameweek, solution=chosen
    )
    return ConstrainedBuild(
        squad=squad,
        selection=selection,
        outcomes=tuple(outcomes),
        expected_points=round(achieved, 4),
        unconstrained_points=round(unconstrained, 4),
    )
