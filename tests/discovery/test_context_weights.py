"""Context-weighted similarity, and the reduction that makes it comparable.

The load-bearing test here is
:meth:`TestExactReduction.test_an_unconstrained_context_is_bitwise_identical`.
Everything the context-conditioned work claims rests on the shipped
objective-only path being a *strict special case* of it: if the two agree
exactly when there are no constraints, then any measured difference between them
is attributable to the constraints and to nothing else. A generalisation that
perturbed the neutral case even slightly would make every later comparison —
including the acceptance gate for the learned representation — uninterpretable.

So it is asserted bitwise, with :func:`numpy.array_equal`, not with a tolerance.
"""

from __future__ import annotations

import numpy as np
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
    OwnershipPreference,
    PrimaryMetric,
    RiskPreference,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.discovery.clusters import context_weights, objective_weights
from xg_alonso.discovery.embeddings import DEFAULT_EMBEDDING_COLUMNS

_LAYOUT = ((Position.GKP, 2), (Position.DEF, 5), (Position.MID, 5), (Position.FWD, 3))
_LEGAL_XI_ORDER = (0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 1, 6, 11, 14)

_ALL_METRICS = tuple(PrimaryMetric)


def _squad(*, bank: int = 30, free_transfers: int = 1) -> SquadState:
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
                    purchase_price=TenthsOfMillion(50),
                    current_price=TenthsOfMillion(50),
                    selling_price=TenthsOfMillion(50),
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
        free_transfers=free_transfers,
    )


def _objective(
    metric: PrimaryMetric = PrimaryMetric.EXPECTED_POINTS,
    *,
    risk: RiskPreference = RiskPreference.BALANCED,
    ownership: OwnershipPreference = OwnershipPreference.NEUTRAL,
    horizon: int = 1,
) -> ManagerObjective:
    return ManagerObjective(
        id=f"{metric.value}_{risk.value}_h{horizon}_{ownership.value}",
        name=metric.value,
        primary_metric=metric,
        risk_preference=risk,
        ownership_preference=ownership,
        planning_horizon=horizon,
    )


def _context(
    *,
    objective: ManagerObjective | None = None,
    constraints: ManagerConstraints | None = None,
    squad: SquadState | None = None,
) -> DecisionContext:
    return DecisionContext(
        bundle=ObjectiveBundle(
            objective=objective or _objective(),
            constraints=constraints or ManagerConstraints(),
        ),
        squad=squad,
        as_of_gameweek=GameweekId(8),
    )


class TestExactReduction:
    """The shipped path must be a strict special case, bit for bit."""

    @pytest.mark.parametrize("metric", _ALL_METRICS, ids=lambda m: m.value)
    def test_an_unconstrained_context_is_bitwise_identical(self, metric: PrimaryMetric) -> None:
        """Across every objective, not just the default one."""
        objective = _objective(metric)
        context = _context(objective=objective, squad=_squad())
        assert np.array_equal(
            context_weights(context, DEFAULT_EMBEDDING_COLUMNS),
            objective_weights(objective, DEFAULT_EMBEDDING_COLUMNS),
        )

    def test_identical_without_a_squad(self) -> None:
        """A from-scratch build has no constraint pressure to read."""
        objective = _objective(PrimaryMetric.DIFFERENTIAL_YIELD)
        assert np.array_equal(
            context_weights(_context(objective=objective), DEFAULT_EMBEDDING_COLUMNS),
            objective_weights(objective, DEFAULT_EMBEDDING_COLUMNS),
        )

    def test_identical_for_constraints_that_do_not_bind(self) -> None:
        """Hits and transfer caps change what you may do, not what is similar."""
        objective = _objective()
        constraints = ManagerConstraints(max_points_hit=12, max_transfers=3)
        assert np.array_equal(
            context_weights(
                _context(objective=objective, constraints=constraints, squad=_squad()),
                DEFAULT_EMBEDDING_COLUMNS,
            ),
            objective_weights(objective, DEFAULT_EMBEDDING_COLUMNS),
        )

    def test_identical_when_only_lightly_locked(self) -> None:
        """One locked player is not a different problem."""
        objective = _objective()
        squad = _squad()
        constraints = ManagerConstraints(locked_players=(squad.picks[0].player_code,))
        assert np.array_equal(
            context_weights(
                _context(objective=objective, constraints=constraints, squad=squad),
                DEFAULT_EMBEDDING_COLUMNS,
            ),
            objective_weights(objective, DEFAULT_EMBEDDING_COLUMNS),
        )


class TestBudgetPressure:
    def test_a_broke_manager_weights_price_higher(self) -> None:
        """With no money, two equal players at different prices are not alternatives."""
        objective = _objective()
        rich = context_weights(
            _context(objective=objective, squad=_squad(bank=80)), DEFAULT_EMBEDDING_COLUMNS
        )
        broke = context_weights(
            _context(objective=objective, squad=_squad(bank=2)), DEFAULT_EMBEDDING_COLUMNS
        )
        price = DEFAULT_EMBEDDING_COLUMNS.index("value_mean_1")
        assert broke[price] > rich[price]

    def test_the_effect_is_graded(self) -> None:
        objective = _objective()
        price = DEFAULT_EMBEDDING_COLUMNS.index("value_mean_1")
        readings = [
            context_weights(
                _context(objective=objective, squad=_squad(bank=bank)), DEFAULT_EMBEDDING_COLUMNS
            )[price]
            for bank in (2, 10, 80)
        ]
        assert readings[0] > readings[1] > readings[2]

    def test_only_the_price_axis_moves(self) -> None:
        """A budget constraint must not quietly reshape unrelated axes."""
        objective = _objective()
        rich = context_weights(
            _context(objective=objective, squad=_squad(bank=80)), DEFAULT_EMBEDDING_COLUMNS
        )
        broke = context_weights(
            _context(objective=objective, squad=_squad(bank=2)), DEFAULT_EMBEDDING_COLUMNS
        )
        moved = {
            name
            for name, before, after in zip(DEFAULT_EMBEDDING_COLUMNS, rich, broke, strict=True)
            if before != after
        }
        assert moved == {"value_mean_1"}


class TestLockPressure:
    def test_a_frozen_squad_weights_durability_higher(self) -> None:
        """With two moves left you live with a rotation risk far longer."""
        objective = _objective()
        squad = _squad()
        codes = tuple(p.player_code for p in squad.picks)

        free = context_weights(
            _context(objective=objective, squad=squad), DEFAULT_EMBEDDING_COLUMNS
        )
        frozen = context_weights(
            _context(
                objective=objective,
                constraints=ManagerConstraints(locked_players=codes[:14]),
                squad=squad,
            ),
            DEFAULT_EMBEDDING_COLUMNS,
        )
        minutes = DEFAULT_EMBEDDING_COLUMNS.index("minutes_mean_20")
        assert frozen[minutes] > free[minutes]

    def test_the_effect_is_graded(self) -> None:
        objective = _objective()
        squad = _squad()
        codes = tuple(p.player_code for p in squad.picks)
        minutes = DEFAULT_EMBEDDING_COLUMNS.index("minutes_mean_20")
        readings = [
            context_weights(
                _context(
                    objective=objective,
                    constraints=ManagerConstraints(locked_players=codes[:n]),
                    squad=squad,
                ),
                DEFAULT_EMBEDDING_COLUMNS,
            )[minutes]
            for n in (14, 10, 0)
        ]
        assert readings[0] > readings[1] > readings[2]


class TestCompositionWithTheObjective:
    def test_a_constraint_never_lowers_an_objective_emphasis(self) -> None:
        """Composition is `max`, so conditioning only ever sharpens an axis.

        A constraint that could *reduce* an objective's emphasis would let a
        hard bound quietly overrule a stated preference — the inversion the
        objective/constraint split exists to prevent.
        """
        squad = _squad(bank=2)
        codes = tuple(p.player_code for p in squad.picks)
        constraints = ManagerConstraints(locked_players=codes[:14])

        for metric in _ALL_METRICS:
            objective = _objective(metric)
            base = objective_weights(objective, DEFAULT_EMBEDDING_COLUMNS)
            conditioned = context_weights(
                _context(objective=objective, constraints=constraints, squad=squad),
                DEFAULT_EMBEDDING_COLUMNS,
            )
            assert np.all(conditioned >= base), metric.value

    def test_an_objective_that_already_dominates_an_axis_is_not_double_counted(self) -> None:
        """`TRANSFER_FLEXIBILITY` already sets price to 2.0; broke sets 2.4."""
        objective = _objective(PrimaryMetric.TRANSFER_FLEXIBILITY)
        price = DEFAULT_EMBEDDING_COLUMNS.index("value_mean_1")
        conditioned = context_weights(
            _context(objective=objective, squad=_squad(bank=2)), DEFAULT_EMBEDDING_COLUMNS
        )
        assert conditioned[price] == pytest.approx(2.4)

    def test_weights_stay_positive_and_finite(self) -> None:
        squad = _squad(bank=0)
        codes = tuple(p.player_code for p in squad.picks)
        for metric in _ALL_METRICS:
            weights = context_weights(
                _context(
                    objective=_objective(metric),
                    constraints=ManagerConstraints(locked_players=codes),
                    squad=squad,
                ),
                DEFAULT_EMBEDDING_COLUMNS,
            )
            assert np.all(np.isfinite(weights))
            assert np.all(weights > 0.0)


class TestTwoManagersSameSquad:
    """The post's central claim, at the similarity layer.

    Same squad, same objective, different constraints — therefore a different
    notion of which players are alternatives to each other. This is the smallest
    end-to-end demonstration that constraints reach the *representation* rather
    than being applied after it.
    """

    def test_same_objective_and_squad_different_constraints_differ(self) -> None:
        objective = _objective(PrimaryMetric.EXPECTED_POINTS)
        squad = _squad(bank=30)
        codes = tuple(p.player_code for p in squad.picks)

        protective = _context(
            objective=objective,
            constraints=ManagerConstraints(locked_players=codes[:14]),
            squad=squad,
        )
        open_handed = _context(objective=objective, squad=squad)

        assert not np.array_equal(
            context_weights(protective, DEFAULT_EMBEDDING_COLUMNS),
            context_weights(open_handed, DEFAULT_EMBEDDING_COLUMNS),
        )
