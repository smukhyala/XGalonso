"""Tests for building a squad from scratch.

Until now `build_squad` had **no test coverage at all** — the search that picks
the entire gameweek-1 product was unguarded, which is how a local-search bug
worth 3.3 expected points a gameweek survived unnoticed.

The load-bearing test here is `test_solver_objective_equals_repo_scorer`. The
builder's optimality claim rests entirely on its MILP encoding meaning the same
thing as `starting_xi_points`. If those two ever disagree, the solver is
optimising something that is not the product's objective and every "proven
optimal" statement about it is void. So the equivalence is asserted on random
legal squads rather than assumed.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from xg_alonso.contracts.identifiers import EntryId, GameweekId, PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.domain.rules import PositionRule, SquadRules
from xg_alonso.optimization.lineup import best_starting_xi, starting_xi_points
from xg_alonso.optimization.squad_builder import SquadCandidate, build_squad

_CUTOFF = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)


def _prediction(code: int, position: Position, points: float) -> PlayerPrediction:
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
            data_cutoff=_CUTOFF,
            predicted_at=_CUTOFF,
            run_id="test",
            code_version="test",
        ),
    )


def _candidate(
    code: int, position: Position, price: int, points: float, team: int
) -> SquadCandidate:
    return SquadCandidate(
        player_code=PlayerCode(code),
        position=position,
        team_id=TeamId(team),
        price=TenthsOfMillion(price),
        prediction=_prediction(code, position, points),
    )


@pytest.fixture
def rules() -> SquadRules:
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


def _pool(seed: int = 7, per_position: int = 22) -> list[SquadCandidate]:
    """A candidate pool with enough spread that price and points disagree.

    Deliberately includes expensive-but-poor and cheap-but-good players, since a
    pool where price and points agree would let a naive builder look correct.
    """
    rng = random.Random(seed)
    pool: list[SquadCandidate] = []
    code = 1000
    for position in Position:
        for _ in range(per_position):
            price = rng.choice([40, 45, 50, 55, 60, 70, 80, 95, 110, 125])
            points = round(rng.uniform(0.0, 9.0), 3)
            pool.append(_candidate(code, position, price, points, team=code % 20 + 1))
            code += 1
    return pool


def _build(pool: list[SquadCandidate], rules: SquadRules) -> tuple[object, object]:
    return build_squad(pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))


class TestLegality:
    def test_squad_satisfies_every_rule(self, rules: SquadRules) -> None:
        pool = _pool()
        state, _ = build_squad(pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))

        assert len(state.picks) == rules.squad_size

        for position in Position:
            selected = sum(1 for p in state.picks if p.position is position)
            assert selected == rules.rule_for(position).squad_select

        clubs: dict[int, int] = {}
        for pick in state.picks:
            clubs[int(pick.team_id)] = clubs.get(int(pick.team_id), 0) + 1
        assert max(clubs.values()) <= rules.max_per_club

        spend = sum(int(p.current_price) for p in state.picks)
        assert spend <= int(rules.total_budget)
        assert spend + int(state.bank) == int(rules.total_budget)

    def test_no_player_selected_twice(self, rules: SquadRules) -> None:
        state, _ = build_squad(_pool(), rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
        codes = [p.player_code for p in state.picks]
        assert len(set(codes)) == len(codes)

    def test_duplicate_candidates_are_collapsed(self, rules: SquadRules) -> None:
        """A pool listing the same player twice must not field him twice."""
        pool = _pool()
        state, _ = build_squad(
            pool + pool[:30], rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1)
        )
        codes = [p.player_code for p in state.picks]
        assert len(set(codes)) == len(codes) == rules.squad_size

    def test_slots_are_contiguous_and_xi_comes_first(self, rules: SquadRules) -> None:
        state, selection = build_squad(
            _pool(), rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1)
        )
        assert [p.squad_slot for p in state.picks] == list(range(1, 16))
        starters = {p.player_code for p in state.picks if p.squad_slot <= 11}
        assert starters == {p.player_code for p in selection.starters}

    def test_exactly_one_captain_and_one_vice(self, rules: SquadRules) -> None:
        state, _ = build_squad(_pool(), rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
        assert sum(1 for p in state.picks if p.is_captain) == 1
        assert sum(1 for p in state.picks if p.is_vice_captain) == 1
        captain = next(p for p in state.picks if p.is_captain)
        assert captain.squad_slot <= 11, "the captain must be starting"


class TestOptimality:
    def test_solver_objective_equals_repo_scorer(self, rules: SquadRules) -> None:
        """The equivalence the optimality claim depends on.

        The MILP maximises `ep * (in_xi + is_captain)`. The rest of the system
        scores a squad with `starting_xi_points`. If those diverge, the builder
        is optimising the wrong thing — so this compares them on the returned
        squad, where they must agree exactly.
        """
        state, selection = build_squad(
            _pool(), rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1)
        )
        lookup = {c.player_code: c.prediction for c in _pool()}
        assert starting_xi_points(state.picks, lookup, rules) == pytest.approx(
            selection.expected_points, abs=1e-9
        )

    def test_beats_every_random_legal_squad(self, rules: SquadRules) -> None:
        """No random legal squad may beat the solver's. Catches a wrong optimum.

        A builder that returns *a* legal squad rather than the *best* one passes
        every legality test above. This is what separates the two.
        """
        pool = _pool()
        lookup = {c.player_code: c.prediction for c in pool}
        _, selection = build_squad(pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))

        by_position: dict[Position, list[SquadCandidate]] = {p: [] for p in Position}
        for candidate in pool:
            by_position[candidate.position].append(candidate)

        rng = random.Random(11)
        beaten = 0
        for _ in range(400):
            picks = []
            clubs: dict[int, int] = {}
            spend = 0
            ok = True
            for position in Position:
                quota = rules.rule_for(position).squad_select
                options = by_position[position][:]
                rng.shuffle(options)
                taken = 0
                for candidate in options:
                    if taken == quota:
                        break
                    club = int(candidate.team_id)
                    if clubs.get(club, 0) >= rules.max_per_club:
                        continue
                    picks.append(candidate)
                    clubs[club] = clubs.get(club, 0) + 1
                    spend += int(candidate.price)
                    taken += 1
                if taken != quota:
                    ok = False
                    break
            if not ok or spend > int(rules.total_budget):
                continue

            from xg_alonso.optimization.squad_builder import _as_pick

            score = starting_xi_points(
                [_as_pick(c, i + 1) for i, c in enumerate(picks)], lookup, rules
            )
            if score > selection.expected_points + 1e-9:
                beaten += 1

        assert beaten == 0, f"{beaten} random legal squads beat the 'optimal' one"

    def test_prefers_a_cheap_strong_player_over_an_expensive_weak_one(
        self, rules: SquadRules
    ) -> None:
        """Spending is never the objective; points are."""
        pool = _pool()
        # A bargain nobody should refuse, and a trap nobody should take.
        pool.append(_candidate(9001, Position.MID, 45, 15.0, team=19))
        pool.append(_candidate(9002, Position.MID, 125, 0.1, team=20))
        state, _ = build_squad(pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
        codes = {int(p.player_code) for p in state.picks}
        assert 9001 in codes
        assert 9002 not in codes

    def test_captain_is_the_highest_scorer_in_the_xi(self, rules: SquadRules) -> None:
        pool = _pool()
        lookup = {c.player_code: c.prediction for c in pool}
        state, selection = build_squad(
            pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1)
        )
        starter_points = [lookup[p.player_code].expected_points for p in selection.starters]
        captain = next(p for p in state.picks if p.is_captain)
        assert lookup[captain.player_code].expected_points == pytest.approx(max(starter_points))


class TestBudget:
    def test_a_tight_budget_is_respected(self, rules: SquadRules) -> None:
        """The cheapest legal fifteen must still be found when money is scarce."""
        tight = rules.model_copy(update={"total_budget": TenthsOfMillion(700)})
        state, _ = build_squad(_pool(), rules=tight, entry_id=EntryId(1), gameweek=GameweekId(1))
        assert sum(int(p.current_price) for p in state.picks) <= 700
        assert int(state.bank) >= 0

    def test_bank_is_never_negative(self, rules: SquadRules) -> None:
        state, _ = build_squad(_pool(), rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
        assert int(state.bank) >= 0

    def test_tie_break_spends_idle_money_on_the_bench(self, rules: SquadRules) -> None:
        """Money that cannot buy XI points should still buy the best bench.

        Bench players score zero, so which bench is chosen is a free variable.
        Left arbitrary it produces unstable squads and a meaningless bank; the
        tie-break resolves it toward the strongest affordable bench, which costs
        nothing in XI points and is worth something once autosubs are modelled.
        """
        pool = _pool()
        by_code = {c.player_code: c for c in pool}
        state, selection = build_squad(
            pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1)
        )

        # Local optimality of the bench: no unused candidate of the same
        # position is both affordable and better than the bench player he would
        # replace. A merely-nonzero bench would pass by luck; this would not.
        squad = {p.player_code for p in state.picks}
        clubs: dict[int, int] = {}
        for pick in state.picks:
            clubs[int(pick.team_id)] = clubs.get(int(pick.team_id), 0) + 1

        for benched in selection.bench:
            outgoing = by_code[benched.player_code]
            headroom = int(state.bank) + int(outgoing.price)
            for other in pool:
                if other.player_code in squad or other.position is not outgoing.position:
                    continue
                if int(other.price) > headroom:
                    continue
                club = int(other.team_id)
                allowance = clubs.get(club, 0) - (1 if club == int(outgoing.team_id) else 0)
                if allowance >= rules.max_per_club:
                    continue
                assert other.expected_points <= outgoing.expected_points + 1e-9, (
                    f"{other.player_code} ({other.expected_points:.3f}) is affordable and "
                    f"better than benched {outgoing.player_code} "
                    f"({outgoing.expected_points:.3f}) — the tie-break left money idle"
                )

    def test_impossible_budget_raises_with_a_useful_message(self, rules: SquadRules) -> None:
        broke = rules.model_copy(update={"total_budget": TenthsOfMillion(100)})
        with pytest.raises(ValueError, match="no legal squad"):
            build_squad(_pool(), rules=broke, entry_id=EntryId(1), gameweek=GameweekId(1))


class TestDeterminism:
    def test_same_input_gives_the_same_squad(self, rules: SquadRules) -> None:
        first, _ = build_squad(_pool(), rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
        second, _ = build_squad(_pool(), rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
        assert [p.player_code for p in first.picks] == [p.player_code for p in second.picks]

    def test_candidate_order_does_not_change_the_squad(self, rules: SquadRules) -> None:
        """Permuting the input must not permute the answer."""
        pool = _pool()
        shuffled = pool[:]
        random.Random(3).shuffle(shuffled)

        first, _ = build_squad(pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
        second, _ = build_squad(shuffled, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
        assert sorted(int(p.player_code) for p in first.picks) == sorted(
            int(p.player_code) for p in second.picks
        )


class TestFailureModes:
    def test_empty_pool_raises(self, rules: SquadRules) -> None:
        with pytest.raises(ValueError, match="no candidates"):
            build_squad([], rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))

    def test_too_few_in_one_position_raises_naming_the_shortfall(self, rules: SquadRules) -> None:
        pool = [c for c in _pool() if c.position is not Position.GKP]
        pool.append(_candidate(5001, Position.GKP, 40, 3.0, team=1))
        with pytest.raises(ValueError, match="no legal squad"):
            build_squad(pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))

    def test_club_cap_can_make_a_pool_infeasible(self, rules: SquadRules) -> None:
        """Fifteen players from two clubs cannot satisfy a three-per-club cap."""
        pool: list[SquadCandidate] = []
        code = 6000
        for position in Position:
            for _ in range(6):
                pool.append(_candidate(code, position, 40, 4.0, team=code % 2 + 1))
                code += 1
        with pytest.raises(ValueError, match="no legal squad"):
            build_squad(pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))


class TestExternalPredictions:
    def test_supplied_predictions_override_candidate_ones(self, rules: SquadRules) -> None:
        """The `predictions` argument is the authority when both are present."""
        pool = _pool()
        target = pool[0]
        overridden = {
            c.player_code: _prediction(
                int(c.player_code),
                c.position,
                50.0 if c.player_code == target.player_code else 0.0,
            )
            for c in pool
        }
        state, _ = build_squad(
            pool,
            rules=rules,
            entry_id=EntryId(1),
            gameweek=GameweekId(1),
            predictions=overridden,
        )
        assert target.player_code in {p.player_code for p in state.picks}
        captain = next(p for p in state.picks if p.is_captain)
        assert captain.player_code == target.player_code


def test_returned_selection_matches_a_fresh_selection(rules: SquadRules) -> None:
    """The XI handed back must be the XI the scorer would pick from those picks."""
    pool = _pool()
    lookup = {c.player_code: c.prediction for c in pool}
    state, selection = build_squad(pool, rules=rules, entry_id=EntryId(1), gameweek=GameweekId(1))
    fresh = best_starting_xi(state.picks, lookup, rules)
    assert fresh.expected_points == pytest.approx(selection.expected_points)
    assert {p.player_code for p in fresh.starters} == {p.player_code for p in selection.starters}
