"""Decision metrics: the seven that were never registered, and what they cost.

The single most important test here is
:meth:`TestDecisionsNotRegressions.test_a_worse_regression_with_better_picks_scores_higher`.
CLAUDE.md states the principle — *"a slightly worse regression model that
produces better transfer decisions is preferred"* — and until this module
existed nothing in the codebase could tell the two apart: ``objective_gain``
silently returned ``predictive_gain``, so every objective ranked every candidate
identically. That test is the principle made executable.

:meth:`TestEveryObjectiveResolves.test_every_primary_metric_has_a_metric` is the
regression guard for the original defect. A `PrimaryMetric` with no registered
metric falls back to the predictive gain again, and the failure is invisible in
every report.
"""

from __future__ import annotations

import numpy as np
import pytest

from xg_alonso.contracts.objective import PrimaryMetric
from xg_alonso.discovery.decision_metrics import (
    DECISION_METRICS,
    TOP_K,
    captaincy_upside,
    decision_gain,
    differential_yield,
    downside_protection,
    expected_points,
    expected_rank_gain,
    objective_gain,
    register_decision_metrics,
    team_value_growth,
    transfer_flexibility,
    turnover_penalty,
)
from xg_alonso.discovery.utility import MetricContext, MetricRegistry


def _context(
    predicted: list[float],
    actual: list[float],
    *,
    ownership: list[float] | None = None,
    momentum: list[float] | None = None,
    gameweek: list[int] | None = None,
    entity: list[int] | None = None,
) -> MetricContext:
    extras: dict[str, np.ndarray] = {}
    if ownership is not None:
        extras["ownership"] = np.array(ownership, dtype=np.float64)
    if momentum is not None:
        extras["transfer_momentum"] = np.array(momentum, dtype=np.float64)
    groups: dict[str, np.ndarray] = {}
    if gameweek is not None:
        groups["gameweek"] = np.array(gameweek, dtype=np.int64)
    if entity is not None:
        groups["entity"] = np.array(entity, dtype=np.int64)
    return MetricContext(
        predicted=np.array(predicted, dtype=np.float64),
        actual=np.array(actual, dtype=np.float64),
        extras=extras,
        groups=groups,
    )


class TestEveryObjectiveResolves:
    def test_every_primary_metric_has_a_metric(self) -> None:
        """The regression guard for the original defect.

        An unresolved metric silently falls back to the predictive gain, and
        every objective then scores every candidate identically while the
        reports look completely normal.
        """
        missing = [m.value for m in PrimaryMetric if DECISION_METRICS.get(m.value) is None]
        assert not missing, f"objectives with no decision metric: {missing}"

    def test_registration_refuses_to_replace_silently(self) -> None:
        """Swapping a metric changes what every past evaluation of it meant."""
        registry = MetricRegistry()
        register_decision_metrics(registry)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(PrimaryMetric.EXPECTED_POINTS.value, expected_points)

    def test_registering_twice_is_a_no_op_not_an_error(self) -> None:
        registry = MetricRegistry()
        register_decision_metrics(registry)
        register_decision_metrics(registry)
        assert registry.get(PrimaryMetric.EXPECTED_POINTS.value) is not None


class TestDecisionsNotRegressions:
    """The point of the whole module."""

    def test_a_worse_regression_with_better_picks_scores_higher(self) -> None:
        """CLAUDE.md's stated preference, as an assertion.

        Two models over the same twenty-five players. ``sharp`` is *worse* on
        mean absolute error across the field but ranks the genuinely best
        players first. ``blunt`` is closer on average and ranks them badly. A
        decision metric must prefer ``sharp``; a regression metric prefers
        ``blunt``, and preferring ``blunt`` is how a model that is pleasant on
        paper loses points every week.
        """
        players = 25
        actual = [float(i) for i in range(players)]  # 0..24, best last

        # Ranks correctly (higher actual -> higher prediction) but is badly
        # calibrated in level.
        sharp = [float(i) * 3.0 for i in range(players)]
        # Nearly perfect in level, but ranks the field backwards.
        blunt = [float(players - 1 - i) + 0.1 for i in range(players)]

        sharp_mae = float(np.mean(np.abs(np.array(sharp) - np.array(actual))))
        blunt_mae = float(np.mean(np.abs(np.array(blunt) - np.array(actual))))
        assert sharp_mae > blunt_mae, "the premise requires sharp to be the worse regression"

        sharp_picks = expected_points(_context(sharp, actual))
        blunt_picks = expected_points(_context(blunt, actual))
        assert sharp_picks > blunt_picks

    def test_accuracy_outside_the_top_k_does_not_matter(self) -> None:
        """A decision metric ignores what a manager will never consider.

        Wrecking the predictions of players ranked far down the list must not
        move the score, because none of them was going to be bought.
        """
        players = 60
        actual = [float(i % 7) for i in range(players)]
        good = [float(players - i) for i in range(players)]

        wrecked = list(good)
        for i in range(TOP_K + 5, players):
            wrecked[i] = -999.0

        assert expected_points(_context(good, actual)) == pytest.approx(
            expected_points(_context(wrecked, actual))
        )


class TestTheSevenMetrics:
    def test_expected_points_reads_the_picks(self) -> None:
        # Top-2 predicted are the last two, which score 10 and 20.
        score = expected_points(_context([1, 2, 3, 4], [0, 0, 10, 20]))
        assert score == pytest.approx(7.5)  # mean of all four, since k > n

    def test_rank_gain_discounts_template_players(self) -> None:
        """Identical returns, different ownership, different value."""
        differential = expected_rank_gain(
            _context([2.0, 1.0], [10.0, 10.0], ownership=[0.02, 0.02])
        )
        template = expected_rank_gain(_context([2.0, 1.0], [10.0, 10.0], ownership=[0.9, 0.9]))
        assert differential > template

    def test_rank_gain_without_ownership_falls_back_rather_than_zeroing(self) -> None:
        """A silent zero would assert every pick was template."""
        assert expected_rank_gain(_context([2.0, 1.0], [10.0, 6.0])) == pytest.approx(8.0)

    def test_downside_protection_punishes_a_boom_or_bust_set(self) -> None:
        steady = downside_protection(_context([5, 4, 3, 2], [6, 6, 6, 6]))
        volatile = downside_protection(_context([5, 4, 3, 2], [24, 0, 0, 0]))
        assert steady > volatile

    def test_captaincy_reads_one_player_per_gameweek(self) -> None:
        """Two gameweeks; the top pick returns 12 then 2, so the mean is 7."""
        score = captaincy_upside(
            _context([5, 1, 5, 1], [12, 0, 2, 0], gameweek=[1, 1, 2, 2], entity=[1, 2, 1, 2])
        )
        assert score == pytest.approx(7.0)

    def test_differential_yield_ignores_template_entirely(self) -> None:
        """Stricter than rank gain: a template player contributes nothing."""
        score = differential_yield(
            _context([3.0, 2.0, 1.0], [10.0, 100.0, 10.0], ownership=[0.05, 0.85, 0.05])
        )
        assert score == pytest.approx(10.0)

    def test_differential_yield_is_zero_when_every_pick_is_template(self) -> None:
        score = differential_yield(_context([2.0, 1.0], [50.0, 50.0], ownership=[0.9, 0.95]))
        assert score == 0.0

    def test_team_value_growth_needs_momentum(self) -> None:
        """Without the indicator there is nothing to agree with."""
        assert team_value_growth(_context([1.0, 2.0], [1.0, 2.0])) == 0.0
        agreeing = team_value_growth(_context([1.0, 2.0, 3.0], [0, 0, 0], momentum=[1.0, 2.0, 3.0]))
        assert agreeing > 0.9

    def test_transfer_flexibility_prefers_a_stable_ranking(self) -> None:
        stable = transfer_flexibility(
            _context(
                [3, 2, 1, 3, 2, 1], [0] * 6, gameweek=[1, 1, 1, 2, 2, 2], entity=[1, 2, 3, 1, 2, 3]
            )
        )
        churning = transfer_flexibility(
            _context(
                [3, 2, 1, 1, 2, 3], [0] * 6, gameweek=[1, 1, 1, 2, 2, 2], entity=[1, 2, 3, 1, 2, 3]
            )
        )
        assert stable > churning

    def test_every_metric_survives_an_empty_context(self) -> None:
        empty = _context([], [])
        for name in (m.value for m in PrimaryMetric):
            metric = DECISION_METRICS.get(name)
            assert metric is not None
            assert np.isfinite(metric(empty))


class TestTurnoverAlignsOnEntity:
    """The latent defect this module's first caller exposed.

    ``utility.turnover_score`` correlates two arrays positionally, which is only
    correct when both hold the same entities in the same order. Consecutive FPL
    gameweeks do not — players are injured, rested and rotated. Feeding it raw
    per-gameweek slices would measure *roster* churn and report it as *ranking*
    churn.
    """

    def test_a_stable_ranking_reports_no_churn(self) -> None:
        score = turnover_penalty(
            _context(
                [3, 2, 1, 3, 2, 1], [0] * 6, gameweek=[1, 1, 1, 2, 2, 2], entity=[1, 2, 3, 1, 2, 3]
            )
        )
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_a_reversed_ranking_reports_full_churn(self) -> None:
        score = turnover_penalty(
            _context(
                [3, 2, 1, 1, 2, 3], [0] * 6, gameweek=[1, 1, 1, 2, 2, 2], entity=[1, 2, 3, 1, 2, 3]
            )
        )
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_a_changed_roster_is_not_mistaken_for_churn(self) -> None:
        """The defect, directly.

        Gameweek 2 holds a different set of players in a different order, but
        every player shared between the two weeks keeps its relative position.
        Positional comparison would report heavy churn; entity-aligned
        comparison correctly reports none.
        """
        score = turnover_penalty(
            _context(
                predicted=[3, 2, 1, 9, 5, 4],
                actual=[0] * 6,
                # Player 4 replaces player 1 in week 2. Among the players
                # present in *both* weeks, 2 still outranks 3 — so the ranking
                # did not churn, only the roster did.
                gameweek=[1, 1, 1, 2, 2, 2],
                entity=[1, 2, 3, 4, 2, 3],
            )
        )
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_without_entities_it_reports_nothing_rather_than_guessing(self) -> None:
        assert turnover_penalty(_context([1, 2], [0, 0], gameweek=[1, 2])) == 0.0

    def test_without_gameweeks_it_is_unmeasured(self) -> None:
        assert turnover_penalty(_context([1, 2], [0, 0])) is None


class TestUnmeasuredIsDistinctFromZero:
    """A term that could not be computed must never read as a measured zero."""

    def test_objective_gain_is_none_without_contexts(self) -> None:
        assert objective_gain(PrimaryMetric.EXPECTED_POINTS.value, None, None) is None

    def test_objective_gain_is_none_for_an_unknown_metric(self) -> None:
        ctx = _context([1.0, 2.0], [1.0, 2.0])
        assert objective_gain("no_such_metric", ctx, ctx) is None

    def test_decision_gain_is_none_without_contexts(self) -> None:
        assert decision_gain(None, None) is None

    def test_a_genuine_improvement_is_positive(self) -> None:
        actual = [float(i) for i in range(30)]
        worse = _context([float(30 - i) for i in range(30)], actual)
        better = _context([float(i) for i in range(30)], actual)
        gain = decision_gain(worse, better)
        assert gain is not None
        assert gain > 0.0

    def test_objectives_disagree_about_the_same_candidate(self) -> None:
        """Two objectives, one field, opposite verdicts.

        The property that was false while the registry was empty — every
        objective then scored every candidate identically, because they were all
        reading the predictive gain.

        Forty players in two halves. The *volatile* half averages more but
        blanks half the time; the *steady* half averages less and never blanks.
        A model that ranks the volatile half first wins on expected points and
        loses on downside protection. Both groups must exceed ``TOP_K`` or each
        model picks the entire field and the ranking cannot matter — which is
        exactly how the first version of this test fooled itself.
        """
        half = TOP_K + 5
        volatile = [30.0 if i % 2 == 0 else 0.0 for i in range(half)]  # mean 15, p10 = 0
        steady = [10.0] * half  # mean 10, p10 = 10
        actual = volatile + steady

        volatile_first = _context([float(2 * half - i) for i in range(2 * half)], actual)
        steady_first = _context([float(i) for i in range(2 * half)], actual)

        points = DECISION_METRICS.get(PrimaryMetric.EXPECTED_POINTS.value)
        floor = DECISION_METRICS.get(PrimaryMetric.DOWNSIDE_PROTECTION.value)
        assert points is not None
        assert floor is not None

        assert points(volatile_first) > points(steady_first)
        assert floor(steady_first) > floor(volatile_first)
