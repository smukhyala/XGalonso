"""Tests for squad requirements: hard constraints, relaxation and pricing.

The assertions that matter are the ones about *hardness*. A requirement that is
honoured most of the time is the failure this whole module exists to prevent, so
several of these deliberately make the required player a terrible one — if the
constraint were a penalty, a good enough alternative would overcome it and the
test would catch that.
"""

from __future__ import annotations

import pytest

from tests.optimization.helpers import make_candidate, make_rules
from xg_alonso.contracts.identifiers import EntryId, GameweekId, PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.objective import (
    ManagerConstraints,
    Requirement,
    RequirementKind,
    SquadRequirements,
    TeamQuota,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.domain.rules import SquadRules
from xg_alonso.optimization.requirements import (
    coherence_problems,
    compile_requirements,
    requirements_from_constraints,
)
from xg_alonso.optimization.squad_builder import (
    ConstrainedBuild,
    SquadCandidate,
    build_constrained_squad,
)


def _pool() -> list[SquadCandidate]:
    """A pool deep enough to build several legal squads from.

    Points fall with player code, so the optimum is predictable and a
    requirement forcing a high code in is always a real sacrifice.
    """
    pool: list[SquadCandidate] = []
    code = 1
    per_position = {Position.GKP: 6, Position.DEF: 20, Position.MID: 20, Position.FWD: 12}
    for position, count in per_position.items():
        for index in range(count):
            pool.append(
                make_candidate(
                    code=code,
                    position=position,
                    team_id=code % 12,
                    price=40 + (index % 5) * 5,
                    expected_points=10.0 - index * 0.25,
                )
            )
            code += 1
    return pool


def _build(
    requirements: SquadRequirements | None = None,
    *,
    pool: list[SquadCandidate] | None = None,
    rules: SquadRules | None = None,
) -> ConstrainedBuild:
    return build_constrained_squad(
        pool or _pool(),
        rules=rules or make_rules(),
        entry_id=EntryId(1),
        gameweek=GameweekId(1),
        requirements=requirements,
    )


def _require(kind: RequirementKind, label: str, **payload: object) -> Requirement:
    return Requirement(kind=kind, label=label, **payload)  # type: ignore[arg-type]


class TestHardness:
    def test_no_requirements_reproduces_the_free_optimum(self) -> None:
        built = _build()
        assert built.feasible_as_asked
        assert built.expected_points == built.unconstrained_points
        assert built.total_cost == 0.0

    def test_a_required_player_starts_however_bad_he_is(self) -> None:
        """The whole point. A penalty would let a better alternative win."""
        pool = _pool()
        worst = min(pool, key=lambda c: c.expected_points)
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.MUST_START,
                        "worst player must start",
                        players=(worst.player_code,),
                    ),
                )
            ),
            pool=pool,
        )
        starters = {p.player_code for p in built.selection.starters}
        assert worst.player_code in starters
        assert built.feasible_as_asked

    def test_must_include_permits_the_bench(self) -> None:
        """ "I want him" and "I want him starting" are different requests."""
        pool = _pool()
        worst = min(pool, key=lambda c: c.expected_points)
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.MUST_INCLUDE,
                        "worst player in the squad",
                        players=(worst.player_code,),
                    ),
                )
            ),
            pool=pool,
        )
        squad = {p.player_code for p in built.squad.picks}
        starters = {p.player_code for p in built.selection.starters}
        assert worst.player_code in squad
        assert worst.player_code not in starters, "a bad player should be benched, not started"

    def test_an_excluded_player_is_never_picked(self) -> None:
        pool = _pool()
        best = max(pool, key=lambda c: c.expected_points)
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.MUST_EXCLUDE,
                        "never pick the best player",
                        players=(best.player_code,),
                    ),
                )
            ),
            pool=pool,
        )
        assert best.player_code not in {p.player_code for p in built.squad.picks}

    def test_a_required_captain_wears_the_armband(self) -> None:
        pool = _pool()
        target = sorted(pool, key=lambda c: -c.expected_points)[8]
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.MUST_CAPTAIN,
                        "captain him",
                        players=(target.player_code,),
                    ),
                )
            ),
            pool=pool,
        )
        assert built.selection.captain == target.player_code


class TestStructure:
    def test_a_formation_fixes_the_shape(self) -> None:
        built = _build(
            SquadRequirements(
                requirements=(_require(RequirementKind.FORMATION, "3-5-2", formation="3-5-2"),)
            )
        )
        shape = dict.fromkeys(Position, 0)
        for pick in built.selection.starters:
            shape[pick.position] += 1
        assert (shape[Position.DEF], shape[Position.MID], shape[Position.FWD]) == (3, 5, 2)

    def test_a_club_floor_is_met(self) -> None:
        pool = _pool()
        club = 3
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.CLUB_FLOOR,
                        "three from club 3",
                        team_id=TeamId(club),
                        count=3,
                    ),
                )
            ),
            pool=pool,
        )
        by_code = {c.player_code: c for c in pool}
        held = sum(1 for p in built.squad.picks if int(by_code[p.player_code].team_id) == club)
        assert held >= 3

    def test_a_bank_floor_leaves_money_unspent(self) -> None:
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.BANK_FLOOR,
                        "leave 5.0m",
                        amount=TenthsOfMillion(50),
                    ),
                )
            )
        )
        assert int(built.squad.bank) >= 50

    @pytest.mark.parametrize("shape", ["4-4-2", "3-4-3", "5-3-2", "3-5-2"])
    def test_every_legal_shape_is_reachable(self, shape: str) -> None:
        built = _build(
            SquadRequirements(
                requirements=(_require(RequirementKind.FORMATION, shape, formation=shape),)
            )
        )
        assert built.feasible_as_asked, f"{shape} should be buildable"


class TestInfeasibility:
    def test_an_impossible_set_names_what_gave(self) -> None:
        """Four forwards cannot all start — only three may play."""
        pool = _pool()
        forwards = [c for c in pool if c.position is Position.FWD][:4]
        built = _build(
            SquadRequirements(
                requirements=tuple(
                    _require(
                        RequirementKind.MUST_START,
                        f"forward {index} must start",
                        players=(candidate.player_code,),
                        priority=index,
                    )
                    for index, candidate in enumerate(forwards)
                )
            ),
            pool=pool,
        )
        assert not built.feasible_as_asked
        assert len(built.relaxed) >= 1
        # It still returns a squad rather than an error.
        assert len(built.squad.picks) == 15

    def test_relaxation_follows_priority(self) -> None:
        """The lowest priority gives first, so a manager's ranking is honoured."""
        pool = _pool()
        forwards = [c for c in pool if c.position is Position.FWD][:4]
        built = _build(
            SquadRequirements(
                requirements=tuple(
                    _require(
                        RequirementKind.MUST_START,
                        f"forward {index}",
                        players=(candidate.player_code,),
                        priority=index,
                    )
                    for index, candidate in enumerate(forwards)
                )
            ),
            pool=pool,
        )
        relaxed_labels = {r.label for r in built.relaxed}
        assert "forward 0" in relaxed_labels
        assert "forward 3" not in relaxed_labels, "the highest priority must survive"

    def test_a_player_outside_the_pool_is_reported_not_ignored(self) -> None:
        """Silently dropping this would return a squad that looks compliant."""
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.MUST_START,
                        "a player who does not exist",
                        players=(PlayerCode(999_999),),
                    ),
                )
            )
        )
        assert not built.feasible_as_asked
        assert "not among the candidates" in built.outcomes[0].note

    def test_contradictions_are_caught_without_solving(self) -> None:
        code = PlayerCode(1)
        problems = coherence_problems(
            [
                _require(RequirementKind.MUST_START, "start him", players=(code,)),
                _require(RequirementKind.MUST_EXCLUDE, "never him", players=(code,)),
            ],
            teams_of={code: 1},
            max_per_club=3,
        )
        assert any("required and excluded" in p for p in problems)

    def test_four_from_one_club_is_flagged_as_impossible(self) -> None:
        """A league rule, not a tight squeeze — worth saying before a solve."""
        codes = [PlayerCode(i) for i in range(1, 5)]
        problems = coherence_problems(
            [
                _require(RequirementKind.MUST_INCLUDE, f"player {i}", players=(code,))
                for i, code in enumerate(codes)
            ],
            teams_of=dict.fromkeys(codes, 7),
            max_per_club=3,
        )
        assert any("league allows 3" in p for p in problems)


class TestPricing:
    def test_a_costly_requirement_is_priced(self) -> None:
        pool = _pool()
        worst = min(pool, key=lambda c: c.expected_points)
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.MUST_START,
                        "start the worst player",
                        players=(worst.player_code,),
                    ),
                )
            ),
            pool=pool,
        )
        outcome = built.outcomes[0]
        assert outcome.honoured
        assert outcome.cost is not None
        assert outcome.cost > 0, "forcing the worst player in must cost something"
        assert built.total_cost > 0

    def test_a_free_requirement_costs_nothing(self) -> None:
        """Asking for what the optimizer wanted anyway is not a sacrifice."""
        pool = _pool()
        best = max(pool, key=lambda c: c.expected_points)
        built = _build(
            SquadRequirements(
                requirements=(
                    _require(
                        RequirementKind.MUST_START,
                        "start the best player",
                        players=(best.player_code,),
                    ),
                )
            ),
            pool=pool,
        )
        assert built.outcomes[0].cost == pytest.approx(0.0, abs=1e-6)
        assert "costs nothing" in built.outcomes[0].summary

    def test_pricing_can_be_switched_off(self) -> None:
        pool = _pool()
        worst = min(pool, key=lambda c: c.expected_points)
        built = build_constrained_squad(
            pool,
            rules=make_rules(),
            entry_id=EntryId(1),
            gameweek=GameweekId(1),
            requirements=SquadRequirements(
                requirements=(
                    _require(RequirementKind.MUST_START, "start him", players=(worst.player_code,)),
                )
            ),
            price_requirements=False,
        )
        assert built.outcomes[0].cost is None


class TestTranslation:
    def test_locked_players_become_squad_membership(self) -> None:
        """On a from-scratch build, "locked" means bought, not unsold."""
        produced = requirements_from_constraints(
            ManagerConstraints(locked_players=(PlayerCode(7),))
        )
        assert produced[0].kind is RequirementKind.MUST_INCLUDE
        assert produced[0].players == (PlayerCode(7),)

    def test_never_promotes_a_lock_to_a_start(self) -> None:
        """Nothing in the transfer vocabulary distinguishes the eleven."""
        produced = requirements_from_constraints(
            ManagerConstraints(locked_players=(PlayerCode(7),))
        )
        assert all(r.kind is not RequirementKind.MUST_START for r in produced)

    def test_team_quotas_translate_both_ways(self) -> None:
        produced = requirements_from_constraints(
            ManagerConstraints(
                minimum_players_by_team=(TeamQuota(team_id=TeamId(3), count=2),),
                maximum_players_by_team=(TeamQuota(team_id=TeamId(4), count=1),),
            )
        )
        kinds = {r.kind for r in produced}
        assert RequirementKind.CLUB_FLOOR in kinds
        assert RequirementKind.CLUB_CEILING in kinds


class TestContract:
    def test_a_requirement_without_its_payload_is_refused(self) -> None:
        """A row that binds nothing is worse than an error — it looks honoured."""
        with pytest.raises(ValueError, match="requires at least one player"):
            _require(RequirementKind.MUST_START, "start nobody")

    def test_a_club_rule_needs_a_club(self) -> None:
        with pytest.raises(ValueError, match="team_id and a count"):
            _require(RequirementKind.CLUB_FLOOR, "some club")

    def test_two_captains_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one captain"):
            _require(
                RequirementKind.MUST_CAPTAIN,
                "two captains",
                players=(PlayerCode(1), PlayerCode(2)),
            )

    @pytest.mark.parametrize("shape", ["3-5-3", "2-5-2", "3-5", "x-y-z"])
    def test_an_illegal_formation_is_refused(self, shape: str) -> None:
        requirement = _require(RequirementKind.FORMATION, shape, formation=shape)
        with pytest.raises(ValueError, match=r"shape|outfielders"):
            requirement.formation_counts()

    def test_relaxation_order_is_lowest_priority_first(self) -> None:
        low = _require(RequirementKind.FORMATION, "low", formation="4-4-2", priority=0)
        high = _require(RequirementKind.FORMATION, "high", formation="3-5-2", priority=9)
        bundle = SquadRequirements(requirements=(low, high))
        assert bundle.relaxation_order()[0] is low

    def test_ties_relax_most_recent_first(self) -> None:
        first = _require(RequirementKind.FORMATION, "first", formation="4-4-2")
        second = _require(RequirementKind.FORMATION, "second", formation="3-5-2")
        bundle = SquadRequirements(requirements=(first, second))
        assert bundle.relaxation_order()[0] is second


class TestCompilation:
    def test_a_start_row_targets_the_xi_block(self) -> None:
        """`must_start` must bind `y`, not `x` — the bug that would silently bench."""
        codes = [PlayerCode(1), PlayerCode(2)]
        rows, unmet = compile_requirements(
            [_require(RequirementKind.MUST_START, "start 2", players=(PlayerCode(2),))],
            codes=codes,
            positions=[Position.FWD, Position.FWD],
            teams=[1, 2],
            prices=[50, 50],
            total_budget=1000,
        )
        assert not unmet
        # Block 1 (XI) starts at column len(codes); candidate 2 is index 1.
        assert rows[0].entries == ((len(codes) + 1, 1.0),)
        assert (rows[0].lower, rows[0].upper) == (1.0, 1.0)

    def test_an_include_row_targets_the_squad_block(self) -> None:
        codes = [PlayerCode(1), PlayerCode(2)]
        rows, _ = compile_requirements(
            [_require(RequirementKind.MUST_INCLUDE, "include 1", players=(PlayerCode(1),))],
            codes=codes,
            positions=[Position.FWD, Position.FWD],
            teams=[1, 2],
            prices=[50, 50],
            total_budget=1000,
        )
        assert rows[0].entries == ((0, 1.0),)

    def test_a_bank_floor_that_eats_the_budget_is_reported(self) -> None:
        _, unmet = compile_requirements(
            [
                _require(
                    RequirementKind.BANK_FLOOR,
                    "leave everything",
                    amount=TenthsOfMillion(1000),
                )
            ],
            codes=[PlayerCode(1)],
            positions=[Position.FWD],
            teams=[1],
            prices=[50],
            total_budget=1000,
        )
        assert unmet
        assert "whole budget" in unmet[0][1]
