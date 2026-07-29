"""Compile a manager's requirements into constraint rows, and explain them.

**Requirements are constraints, never penalties.** A penalty large enough to
usually prevent something is still one a good enough alternative overcomes, and
"usually starts Haaland" is not starting Haaland. The squad solver already
carries a binary per player for the fifteen (``x``), the eleven (``y``) and the
captaincy (``c``), so every requirement below is one linear row over variables
that already exist — nothing is approximated.

**Infeasibility is a decision, not an error.** "Haaland, Salah, Palmer, three
Arsenal defenders, and 0.5 in the bank" frequently has no solution, and
answering "no legal squad exists" tells a manager nothing they can act on. When
the set cannot hold, requirements are dropped one at a time in relaxation order
until it can, and the ones that had to give are named. The solve is fast enough
(0.04s on the full candidate pool) that doing this exactly, by re-solving, beats
inferring it from duals.

**The price of a requirement is measured the same way.** What Haaland costs is
the difference between the best squad with him and the best squad without,
holding everything else the manager asked for. That is a re-solve too, and it is
the number that actually answers "is keeping him worth it".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, assert_never

from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.objective import (
    ManagerConstraints,
    Requirement,
    RequirementKind,
    SquadRequirements,
)
from xg_alonso.contracts.prediction import Position

__all__ = [
    "ConstraintRow",
    "RequirementOutcome",
    "as_requirements",
    "coherence_problems",
    "compile_requirements",
    "requirements_from_constraints",
    "summarise",
]

#: Solver variable blocks, in the layout `squad_builder` already uses: the
#: column for variable *v* of candidate *i* is ``v * n + i``.
_SQUAD: Final[int] = 0
_XI: Final[int] = 1
_CAPTAIN: Final[int] = 2

_INF: Final[float] = float("inf")


@dataclass(frozen=True)
class ConstraintRow:
    """One linear row, tagged with the requirement that produced it.

    The tag is the point. A row that cannot be traced back to something the
    manager said cannot be named in an infeasibility report, and an unnameable
    constraint is one they cannot decide about.
    """

    requirement: Requirement | None
    entries: tuple[tuple[int, float], ...]
    lower: float
    upper: float

    @property
    def label(self) -> str:
        return self.requirement.label if self.requirement else "league rules"


@dataclass(frozen=True)
class RequirementOutcome:
    """What happened to one requirement, and what it cost."""

    requirement: Requirement
    honoured: bool
    cost: float | None = None
    """Expected points given up by honouring it, holding the others fixed.

    ``None`` when not measured. Zero is a real answer and a useful one — it
    means the requirement asked for something the optimizer wanted anyway.
    """

    note: str = ""

    @property
    def summary(self) -> str:
        if not self.honoured:
            return f"{self.requirement.label} — could not be honoured{f': {self.note}' if self.note else ''}"
        if self.cost is None:
            return f"{self.requirement.label} — honoured"
        if self.cost <= 0.005:
            return f"{self.requirement.label} — honoured, costs nothing"
        return f"{self.requirement.label} — honoured, costs {self.cost:.2f} points"


def requirements_from_constraints(constraints: ManagerConstraints) -> tuple[Requirement, ...]:
    """Read the transfer-path constraints as from-scratch requirements.

    On the transfer path a locked player is one that may not be *sold*; with no
    squad to sell from, the same field means he must be *bought*. Both compile
    to "he is in the fifteen", so this translates rather than duplicating the
    fields — and it means a manager's constraints behave the same whether they
    arrive by CLI, API or intent compiler.

    Note this produces ``MUST_INCLUDE``, never ``MUST_START``. Nothing in the
    transfer vocabulary distinguishes the eleven from the fifteen, and inventing
    the stronger reading here would silently over-constrain a squad.
    """
    produced: list[Requirement] = []

    for code in constraints.locked_players:
        produced.append(
            Requirement(
                kind=RequirementKind.MUST_INCLUDE,
                label=f"player {int(code)} must be in the squad",
                players=(code,),
                priority=2,
            )
        )
    for code in constraints.excluded_players:
        produced.append(
            Requirement(
                kind=RequirementKind.MUST_EXCLUDE,
                label=f"player {int(code)} must not be picked",
                players=(code,),
                priority=2,
            )
        )
    for quota in constraints.minimum_players_by_team:
        produced.append(
            Requirement(
                kind=RequirementKind.CLUB_FLOOR,
                label=f"at least {quota.count} from club {int(quota.team_id)}",
                team_id=quota.team_id,
                count=quota.count,
                priority=1,
            )
        )
    for quota in constraints.maximum_players_by_team:
        produced.append(
            Requirement(
                kind=RequirementKind.CLUB_CEILING,
                label=f"at most {quota.count} from club {int(quota.team_id)}",
                team_id=quota.team_id,
                count=quota.count,
                priority=1,
            )
        )
    if int(constraints.minimum_bank) > 0:
        produced.append(
            Requirement(
                kind=RequirementKind.BANK_FLOOR,
                label=f"leave {int(constraints.minimum_bank) / 10:.1f}m in the bank",
                amount=constraints.minimum_bank,
                priority=0,
            )
        )
    return tuple(produced)


def _column(block: int, index: int, count: int) -> int:
    return block * count + index


def compile_requirements(
    requirements: Sequence[Requirement],
    *,
    codes: Sequence[PlayerCode],
    positions: Sequence[Position],
    teams: Sequence[int],
    prices: Sequence[int],
    total_budget: int,
) -> tuple[list[ConstraintRow], list[tuple[Requirement, str]]]:
    """Turn requirements into solver rows.

    Returns the rows and any requirement that could not be expressed at all —
    a player who is not in the candidate pool, for instance. Those are returned
    rather than dropped, because a requirement silently ignored produces a squad
    that looks like it honoured a request it never saw.

    Args:
        codes: Candidate player codes, in solver column order.
        positions: Each candidate's position, parallel to ``codes``.
        teams: Each candidate's club id, parallel to ``codes``.
        prices: Each candidate's price in tenths, parallel to ``codes``.
        total_budget: The squad budget in tenths.
    """
    count = len(codes)
    index_of = {code: i for i, code in enumerate(codes)}
    rows: list[ConstraintRow] = []
    unmet: list[tuple[Requirement, str]] = []

    for requirement in requirements:
        kind = requirement.kind

        if kind in {
            RequirementKind.MUST_START,
            RequirementKind.MUST_INCLUDE,
            RequirementKind.MUST_EXCLUDE,
            RequirementKind.MUST_CAPTAIN,
        }:
            missing = [c for c in requirement.players if c not in index_of]
            if missing:
                unmet.append(
                    (
                        requirement,
                        f"not among the candidates: {[int(c) for c in missing]}",
                    )
                )
                continue

            block = {
                RequirementKind.MUST_START: _XI,
                RequirementKind.MUST_INCLUDE: _SQUAD,
                RequirementKind.MUST_EXCLUDE: _SQUAD,
                RequirementKind.MUST_CAPTAIN: _CAPTAIN,
            }[kind]
            bound = 0.0 if kind is RequirementKind.MUST_EXCLUDE else 1.0

            # One row per player rather than a summed row. A sum of three
            # equalling three is the same constraint, but a violation of it
            # names three players when only one is the problem.
            for code in requirement.players:
                column = _column(block, index_of[code], count)
                rows.append(
                    ConstraintRow(
                        requirement=requirement,
                        entries=((column, 1.0),),
                        lower=bound,
                        upper=bound,
                    )
                )
            continue

        if kind in {RequirementKind.CLUB_FLOOR, RequirementKind.CLUB_CEILING}:
            members = [i for i, team in enumerate(teams) if team == int(requirement.team_id or -1)]
            if not members:
                unmet.append((requirement, "no candidates from that club"))
                continue
            wanted = float(requirement.count or 0)
            floor = kind is RequirementKind.CLUB_FLOOR
            rows.append(
                ConstraintRow(
                    requirement=requirement,
                    entries=tuple((_column(_SQUAD, i, count), 1.0) for i in members),
                    lower=wanted if floor else -_INF,
                    upper=_INF if floor else wanted,
                )
            )
            continue

        if kind is RequirementKind.FORMATION:
            try:
                defenders, midfielders, forwards = requirement.formation_counts()
            except ValueError as exc:
                unmet.append((requirement, str(exc)))
                continue
            wanted_by_position = {
                Position.DEF: defenders,
                Position.MID: midfielders,
                Position.FWD: forwards,
            }
            for position, wanted in wanted_by_position.items():
                members = [i for i, p in enumerate(positions) if p is position]
                rows.append(
                    ConstraintRow(
                        requirement=requirement,
                        entries=tuple((_column(_XI, i, count), 1.0) for i in members),
                        lower=float(wanted),
                        upper=float(wanted),
                    )
                )
            continue

        if kind is RequirementKind.BANK_FLOOR:
            bank = int(requirement.amount or 0)
            spendable = float(total_budget - bank)
            if spendable <= 0:
                unmet.append((requirement, "the bank floor consumes the whole budget"))
                continue
            rows.append(
                ConstraintRow(
                    requirement=requirement,
                    entries=tuple(
                        (_column(_SQUAD, i, count), float(prices[i])) for i in range(count)
                    ),
                    lower=-_INF,
                    upper=spendable,
                )
            )
            continue

        # Exhaustive by construction. If a kind is added to the enum without a
        # branch here, this fails type-checking rather than silently compiling
        # the requirement to nothing.
        assert_never(kind)

    return rows, unmet


def coherence_problems(
    requirements: Sequence[Requirement],
    *,
    teams_of: Mapping[PlayerCode, int],
    max_per_club: int,
) -> list[str]:
    """Contradictions detectable without solving at all.

    Cheap and worth doing first: a manager who asks for a player twice under
    opposite requirements should be told immediately, not after a relaxation
    pass that blames something else.
    """
    problems: list[str] = []

    starting = {
        code for r in requirements if r.kind is RequirementKind.MUST_START for code in r.players
    }
    included = {
        code for r in requirements if r.kind is RequirementKind.MUST_INCLUDE for code in r.players
    }
    excluded = {
        code for r in requirements if r.kind is RequirementKind.MUST_EXCLUDE for code in r.players
    }

    both = (starting | included) & excluded
    if both:
        problems.append(f"players {sorted(int(c) for c in both)} are both required and excluded")

    captains = [
        code for r in requirements if r.kind is RequirementKind.MUST_CAPTAIN for code in r.players
    ]
    if len(set(captains)) > 1:
        problems.append(
            f"two different captains requested: {sorted(int(c) for c in set(captains))}"
        )

    # The three-per-club cap is a league rule, so a request for four is not a
    # tight squeeze — it is impossible, and saying so beats a relaxation pass.
    per_club: dict[int, int] = {}
    for code in starting | included:
        club = teams_of.get(code)
        if club is not None:
            per_club[club] = per_club.get(club, 0) + 1
    for club, held in per_club.items():
        if held > max_per_club:
            problems.append(
                f"{held} players required from club {club}, but the league allows {max_per_club}"
            )

    return problems


def summarise(outcomes: Sequence[RequirementOutcome]) -> str:
    """A short report a person can read, honoured and broken alike."""
    if not outcomes:
        return "No requirements were given."
    return "\n".join(f"  {outcome.summary}" for outcome in outcomes)


def as_requirements(bundle: SquadRequirements) -> tuple[Requirement, ...]:
    """Every requirement in a bundle, explicit ones plus translated constraints.

    Explicit requirements come first so that a duplicate arising from both
    sources relaxes in the order the manager stated, not the order the
    translation happened to produce.
    """
    return (*bundle.requirements, *requirements_from_constraints(bundle.constraints))
