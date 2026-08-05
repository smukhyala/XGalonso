"""The reachable-player pool: what it removes, what it keeps, what it admits.

Two tests here carry more weight than the rest.

:meth:`TestItChangesWhatIsMeasured.test_a_signal_outside_the_pool_becomes_invisible`
is the one that proves the mask does anything at all. It plants a signal that
exists only among expensive players, masks to a broke manager, and asserts the
signal is no longer visible. Without it, every other test in this file could
pass on a mask that was quietly all-`True`.

:meth:`TestRefusalIsReported.test_a_frame_without_prices_degrades_and_says_so`
is the one that keeps the system honest. A pool that cannot be derived must
fall back to the global population *and label the result*, because a gain
measured on cheap players is not comparable to one measured on everybody and a
number carrying the wrong label is worse than a missing one.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl
import pytest

from xg_alonso.contracts.context import DecisionContext
from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.objective import (
    ManagerConstraints,
    ManagerObjective,
    ObjectiveBundle,
    PrimaryMetric,
    TeamQuota,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.discovery.feasible import REACHABLE_COLUMN, PoolSignature, feasible_pool
from xg_alonso.domain.rules import SquadRules

_LAYOUT = ((Position.GKP, 2), (Position.DEF, 5), (Position.MID, 5), (Position.FWD, 3))
_LEGAL_XI_ORDER = (0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 1, 6, 11, 14)


@pytest.fixture(scope="module")
def rules() -> SquadRules:
    fixture = (
        Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"
    )
    return SquadRules.from_bootstrap(
        json.loads(fixture.read_text()), version="2026-27", source_sha256="b" * 64
    )


def _squad(*, bank: int = 30, price: int = 50) -> SquadState:
    picks: list[SquadPick] = []
    slot = 1
    code = 100
    for position, count in _LAYOUT:
        for _ in range(count):
            picks.append(
                SquadPick(
                    player_code=PlayerCode(code),
                    position=position,
                    team_id=TeamId(1 + code % 12),
                    purchase_price=TenthsOfMillion(price),
                    current_price=TenthsOfMillion(price),
                    selling_price=TenthsOfMillion(price),
                    squad_slot=slot,
                )
            )
            slot += 1
            code += 1
    ordered = [
        picks[original].model_copy(update={"squad_slot": index + 1})
        for index, original in enumerate(_LEGAL_XI_ORDER)
    ]
    return SquadState(
        entry_id=EntryId(1),
        gameweek=GameweekId(8),
        picks=tuple(ordered),
        bank=TenthsOfMillion(bank),
        free_transfers=1,
    )


def _frame(*, n: int = 240, with_price: bool = True, with_team: bool = True) -> pl.DataFrame:
    """A market: prices 40-160, four positions, twelve clubs."""
    positions = [Position.GKP, Position.DEF, Position.MID, Position.FWD]
    rows = {
        "player_code": [100 + i for i in range(n)],
        "position": [positions[i % 4].value for i in range(n)],
        "label_season": ["2025-26"] * n,
        "label_gameweek": [8] * n,
    }
    if with_price:
        rows["price_tenths"] = [40 + (i % 25) * 5 for i in range(n)]
    if with_team:
        rows["team_id"] = [1 + i % 12 for i in range(n)]
    return pl.DataFrame(rows)


def _spread(values: pl.Series) -> float:
    """Standard deviation as a plain float.

    Polars types its aggregates as a wide union (a `std` may be a
    `timedelta` for temporal columns), so the narrowing happens once here
    rather than as a cast at every call site.
    """
    result = values.std()
    return float(result) if isinstance(result, (int, float)) else 0.0


def _context(
    *, constraints: ManagerConstraints | None = None, squad: SquadState | None = None
) -> DecisionContext:
    return DecisionContext(
        bundle=ObjectiveBundle(
            objective=ManagerObjective(
                id="expected_points_balanced_h1_neutral",
                name="points",
                primary_metric=PrimaryMetric.EXPECTED_POINTS,
            ),
            constraints=constraints or ManagerConstraints(),
        ),
        squad=squad,
        as_of_gameweek=GameweekId(8),
    )


class TestRefusalIsReported:
    def test_no_constraints_means_no_pool(self, rules: SquadRules) -> None:
        pool = feasible_pool(_frame(), context=_context(squad=_squad()), rules=rules)
        assert not pool.diagnostics.applied
        assert pool.binding == ()
        assert bool(pool.mask.all())
        assert "did not narrow" in pool.diagnostics.reason

    def test_a_frame_without_prices_degrades_and_says_so(self, rules: SquadRules) -> None:
        """The budget constraint cannot be evaluated, so it must not be claimed."""
        squad = _squad(bank=0)
        constraints = ManagerConstraints(minimum_bank=TenthsOfMillion(500))
        pool = feasible_pool(
            _frame(with_price=False),
            context=_context(constraints=constraints, squad=squad),
            rules=rules,
        )
        assert not pool.diagnostics.applied
        assert "budget_ceiling" not in pool.binding
        assert pool.signature.key() == "global"

    def test_an_empty_frame_is_handled(self, rules: SquadRules) -> None:
        pool = feasible_pool(_frame(n=0), context=_context(squad=_squad()), rules=rules)
        assert not pool.diagnostics.applied
        assert pool.diagnostics.share == 0.0

    def test_a_frame_without_player_codes_degrades(self, rules: SquadRules) -> None:
        frame = _frame().drop("player_code")
        pool = feasible_pool(frame, context=_context(squad=_squad()), rules=rules)
        assert not pool.diagnostics.applied

    def test_diagnostics_describe_themselves(self, rules: SquadRules) -> None:
        squad = _squad()
        excluded = ManagerConstraints(excluded_players=(PlayerCode(101),))
        pool = feasible_pool(
            _frame(), context=_context(constraints=excluded, squad=squad), rules=rules
        )
        assert "reachable" in pool.diagnostics.describe()
        assert "%" in pool.diagnostics.describe()


class TestWhatIsRemoved:
    def test_excluded_players_are_dropped(self, rules: SquadRules) -> None:
        frame = _frame()
        constraints = ManagerConstraints(excluded_players=(PlayerCode(101), PlayerCode(102)))
        pool = feasible_pool(frame, context=_context(constraints=constraints), rules=rules)
        reachable = frame.filter(pool.mask).get_column("player_code").to_list()
        assert 101 not in reachable
        assert 102 not in reachable
        assert "excluded_players" in pool.binding

    def test_excluded_teams_are_dropped(self, rules: SquadRules) -> None:
        frame = _frame()
        constraints = ManagerConstraints(excluded_teams=(TeamId(3),))
        pool = feasible_pool(frame, context=_context(constraints=constraints), rules=rules)
        assert 3 not in frame.filter(pool.mask).get_column("team_id").to_list()
        assert "excluded_teams" in pool.binding

    def test_a_full_position_is_closed(self, rules: SquadRules) -> None:
        """Locking all three forwards means no unlocked forward can be bought."""
        squad = _squad()
        forwards = tuple(p.player_code for p in squad.picks if p.position is Position.FWD)
        frame = _frame()
        pool = feasible_pool(
            frame,
            context=_context(constraints=ManagerConstraints(locked_players=forwards), squad=squad),
            rules=rules,
        )
        assert Position.FWD in pool.closed_positions
        remaining = frame.filter(pool.mask)
        unlocked_forwards = remaining.filter(
            (pl.col("position") == Position.FWD.value)
            & (~pl.col("player_code").is_in([int(c) for c in forwards]))
        )
        assert unlocked_forwards.height == 0

    def test_a_broke_manager_loses_the_expensive_end(self, rules: SquadRules) -> None:
        squad = _squad(bank=0, price=40)
        locked = tuple(p.player_code for p in squad.picks[:13])
        frame = _frame()
        pool = feasible_pool(
            frame,
            context=_context(constraints=ManagerConstraints(locked_players=locked), squad=squad),
            rules=rules,
        )
        assert "budget_ceiling" in pool.binding
        assert pool.per_slot_budget is not None

        # Scoped to *unlocked* players: a locked player is deliberately exempt
        # from the ceiling, because you must still rank your own captain even
        # when he is unaffordable to buy. See
        # `TestLockedPlayersStayInThePool.test_a_locked_player_survives_the_budget_ceiling`.
        buyable = frame.filter(pool.mask).filter(
            ~pl.col("player_code").is_in([int(c) for c in locked])
        )
        assert buyable.height > 0
        dearest = buyable.select(pl.col("price_tenths").max()).item()
        assert int(dearest) <= int(pool.per_slot_budget)

    def test_a_club_at_its_ceiling_is_closed(self, rules: SquadRules) -> None:
        squad = _squad()
        on_club_one = tuple(p.player_code for p in squad.picks if int(p.team_id) == 1)
        assert len(on_club_one) >= 1
        constraints = ManagerConstraints(
            locked_players=on_club_one,
            maximum_players_by_team=(TeamQuota(team_id=TeamId(1), count=len(on_club_one)),),
        )
        frame = _frame()
        pool = feasible_pool(
            frame, context=_context(constraints=constraints, squad=squad), rules=rules
        )
        assert "club_ceiling" in pool.binding
        assert pool.signature.club_restricted


class TestLockedPlayersStayInThePool:
    """They consume slots and budget, but you still have to rank them."""

    def test_a_locked_player_survives_a_closed_position(self, rules: SquadRules) -> None:
        squad = _squad()
        forwards = tuple(p.player_code for p in squad.picks if p.position is Position.FWD)
        frame = _frame()
        pool = feasible_pool(
            frame,
            context=_context(constraints=ManagerConstraints(locked_players=forwards), squad=squad),
            rules=rules,
        )
        reachable = set(frame.filter(pool.mask).get_column("player_code").to_list())
        assert {int(c) for c in forwards} <= reachable

    def test_a_locked_player_survives_the_budget_ceiling(self, rules: SquadRules) -> None:
        """You must still rank your captain even if he is unaffordable to buy."""
        squad = _squad(bank=0, price=40)
        locked = tuple(p.player_code for p in squad.picks[:13])
        # Put a locked player at the very top of the market.
        frame = _frame().with_columns(
            pl.when(pl.col("player_code") == int(locked[0]))
            .then(pl.lit(9999))
            .otherwise(pl.col("price_tenths"))
            .alias("price_tenths")
        )
        pool = feasible_pool(
            frame,
            context=_context(constraints=ManagerConstraints(locked_players=locked), squad=squad),
            rules=rules,
        )
        reachable = set(frame.filter(pool.mask).get_column("player_code").to_list())
        assert int(locked[0]) in reachable


class TestOnlyHardConstraintsAreRead:
    """The objective must never narrow the pool — that would be a filter."""

    @pytest.mark.parametrize("metric", list(PrimaryMetric), ids=lambda m: m.value)
    def test_the_objective_does_not_change_the_mask(
        self, metric: PrimaryMetric, rules: SquadRules
    ) -> None:
        squad = _squad()
        constraints = ManagerConstraints(excluded_players=(PlayerCode(101),))
        frame = _frame()

        def build(chosen: PrimaryMetric) -> list[bool]:
            context = DecisionContext(
                bundle=ObjectiveBundle(
                    objective=ManagerObjective(
                        id=chosen.value, name=chosen.value, primary_metric=chosen
                    ),
                    constraints=constraints,
                ),
                squad=squad,
            )
            return feasible_pool(frame, context=context, rules=rules).mask.to_list()

        assert build(metric) == build(PrimaryMetric.EXPECTED_POINTS)

    def test_hits_and_transfer_caps_do_not_change_the_mask(self, rules: SquadRules) -> None:
        """They change how many moves you may make, not which players exist."""
        squad = _squad()
        frame = _frame()
        base = ManagerConstraints(excluded_players=(PlayerCode(101),))
        loose = ManagerConstraints(
            excluded_players=(PlayerCode(101),), max_points_hit=12, max_transfers=5
        )
        first = feasible_pool(frame, context=_context(constraints=base, squad=squad), rules=rules)
        second = feasible_pool(frame, context=_context(constraints=loose, squad=squad), rules=rules)
        assert first.mask.to_list() == second.mask.to_list()


class TestItChangesWhatIsMeasured:
    """The test that proves the mask is not quietly all-True."""

    def test_a_signal_outside_the_pool_becomes_invisible(self, rules: SquadRules) -> None:
        """Plant a signal only among expensive players, then mask to a broke manager.

        Without this, every other assertion in this file would pass on a mask
        that removed nothing. The correlation between the planted column and the
        target must survive globally and vanish under the pool.
        """
        frame = _frame(n=400)
        # A column that is informative only above 100 (10.0m).
        frame = frame.with_columns(
            pl.when(pl.col("price_tenths") > 100)
            .then(pl.col("price_tenths") * 1.0)
            .otherwise(pl.lit(0.0))
            .alias("premium_signal")
        )

        squad = _squad(bank=0, price=40)
        locked = tuple(p.player_code for p in squad.picks[:13])
        pool = feasible_pool(
            frame,
            context=_context(constraints=ManagerConstraints(locked_players=locked), squad=squad),
            rules=rules,
        )
        assert pool.diagnostics.applied

        globally = frame.get_column("premium_signal")
        # Unlocked only: locked players are exempt from the ceiling by design,
        # and a manager cannot buy them anyway — they are in the pool to be
        # ranked, not to be acquired.
        reachable = (
            frame.filter(pool.mask)
            .filter(~pl.col("player_code").is_in([int(c) for c in locked]))
            .get_column("premium_signal")
        )

        assert _spread(globally) > 0.0, "the signal must vary globally"
        assert reachable.len() > 0
        assert _spread(reachable) == 0.0, (
            "under a broke manager's pool the premium signal is constant, so a "
            "feature built on it can carry no information — which is the whole "
            "point of measuring on the reachable population"
        )

    def test_the_mask_attaches_as_a_column(self, rules: SquadRules) -> None:
        """A column travels with its rows; a Series would misalign after a filter."""
        frame = _frame()
        constraints = ManagerConstraints(excluded_players=(PlayerCode(101),))
        pool = feasible_pool(frame, context=_context(constraints=constraints), rules=rules)
        attached = pool.attach(frame)
        assert REACHABLE_COLUMN in attached.columns

        # Surviving a filter is the property a Series would fail.
        shuffled = attached.filter(pl.col("position") != Position.GKP.value)
        excluded_row = shuffled.filter(pl.col("player_code") == 101)
        if excluded_row.height:
            assert not excluded_row.get_column(REACHABLE_COLUMN).item()


class TestSignatureCardinality:
    """Registry evidence is keyed on this, so it must stay bounded."""

    def test_random_constraint_sets_collapse_to_few_signatures(self, rules: SquadRules) -> None:
        rng = random.Random(20260804)
        frame = _frame(n=300)
        codes = [p.player_code for p in _squad().picks]

        keys: set[str] = set()
        digests: set[str] = set()
        for _ in range(300):
            squad = _squad(bank=rng.randint(0, 90), price=rng.choice([40, 50, 60]))
            constraints = ManagerConstraints(
                locked_players=tuple(rng.sample(codes, rng.randint(0, 15))),
                minimum_bank=TenthsOfMillion(rng.randint(0, 20)),
            )
            context = _context(constraints=constraints, squad=squad)
            digests.add(context.feasibility_digest())
            keys.add(feasible_pool(frame, context=context, rules=rules).signature.key())

        # 2^4 positions x 5 budget bands x 6 share bands x 2 club states, plus
        # "global". Bounded by construction, asserted against the sampler rather
        # than against the arithmetic — the arithmetic is what an edit invalidates.
        assert len(keys) <= 16 * 5 * 6 * 2 + 1

        # And the claim that actually matters: distinct situations, far fewer
        # registry buckets. Without this the bound above could be satisfied by a
        # signature that simply never collapses anything.
        assert len(digests) > 250, "sampler produced too few distinct situations"
        assert len(keys) * 3 < len(digests), (
            f"{len(keys)} signatures against {len(digests)} distinct pools is not compression"
        )

    def test_an_unapplied_pool_is_global(self) -> None:
        assert PoolSignature(applied=False).key() == "global"
