"""Distributions and paired comparisons.

The properties under test are the ones that decide whether a reported interval
means anything: that replicates are averaged before pairing rather than
broadcast, that correlated conditions are resampled in blocks, and that a
sample too small to block on says so instead of quietly reporting a narrow
interval.
"""

from __future__ import annotations

import pytest

from xg_alonso.contracts.evaluation import BootstrapUnit
from xg_alonso.evaluation.statistics import (
    aggregate,
    bootstrap_mean,
    paired_comparison,
)


class TestAggregateIsAlwaysADistribution:
    def test_it_carries_percentiles_not_just_a_mean(self) -> None:
        result = aggregate([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.n == 5
        assert result.mean == pytest.approx(3.0)
        assert result.median == pytest.approx(3.0)
        assert result.p5 < result.p25 < result.p75 < result.p95

    def test_the_worst_and_best_are_traceable(self) -> None:
        """A number nobody can trace to a run is a number nobody can investigate."""
        result = aggregate([5.0, 1.0, 3.0], labels=["a", "b", "c"])
        assert result.worst_label == "b"
        assert result.best_label == "a"

    def test_an_empty_sample_is_refused(self) -> None:
        """Reporting n=0 is honest; reporting a mean of zero is not."""
        with pytest.raises(ValueError, match="empty sample"):
            aggregate([])

    def test_a_single_value_has_no_spread(self) -> None:
        result = aggregate([4.0])
        assert result.sd == 0.0
        assert result.worst == result.best == 4.0


class TestBootstrapBlocksOnSeasons:
    def test_too_few_blocks_degrades_and_says_so(self) -> None:
        """Silently narrowing the interval would be the worst available answer."""
        values = [1.0, 2.0, 3.0, 4.0]
        blocks = ["2024-25", "2024-25", "2025-26", "2025-26"]
        interval = bootstrap_mean(
            values, blocks=blocks, resamples=200, unit=BootstrapUnit.SEASON_BLOCK
        )

        assert interval.unit is BootstrapUnit.CONDITION
        assert "at least three" in interval.caveat

    def test_three_blocks_are_resampled_as_blocks(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        blocks = ["a", "a", "b", "b", "c", "c"]
        interval = bootstrap_mean(
            values, blocks=blocks, resamples=200, unit=BootstrapUnit.SEASON_BLOCK
        )

        assert interval.unit is BootstrapUnit.SEASON_BLOCK
        assert interval.caveat == ""
        assert interval.n_units == 3

    def test_block_resampling_is_wider_than_condition_resampling(self) -> None:
        """Which is the point: correlated conditions carry less information."""
        values = [1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 5.0, 5.0, 5.0]
        blocks = ["a"] * 3 + ["b"] * 3 + ["c"] * 3

        blocked = bootstrap_mean(
            values, blocks=blocks, resamples=2000, unit=BootstrapUnit.SEASON_BLOCK
        )
        flat = bootstrap_mean(values, resamples=2000, unit=BootstrapUnit.CONDITION)

        assert (blocked.high - blocked.low) > (flat.high - flat.low)

    def test_it_is_reproducible_from_its_seed(self) -> None:
        values = [1.0, 4.0, 2.0, 8.0]
        a = bootstrap_mean(values, resamples=500, seed=7, unit=BootstrapUnit.CONDITION)
        b = bootstrap_mean(values, resamples=500, seed=7, unit=BootstrapUnit.CONDITION)
        assert (a.low, a.high) == (b.low, b.high)

    def test_the_interval_brackets_the_point(self) -> None:
        interval = bootstrap_mean([1.0, 2.0, 3.0, 4.0], resamples=500, unit=BootstrapUnit.CONDITION)
        assert interval.low <= interval.point <= interval.high


class TestPairingAveragesReplicatesFirst:
    def test_replicates_do_not_inflate_the_sample_size(self) -> None:
        """A deterministic policy has one value; broadcasting it would triple n."""
        policy = {"c1": [10.0, 12.0, 14.0], "c2": [4.0, 6.0, 8.0]}
        baseline = {"c1": [10.0], "c2": [5.0]}

        result = paired_comparison(
            policy, baseline, policy="random", baseline="hold", metric="vs_hold", resamples=200
        )

        assert result.n_conditions == 2
        assert result.differences.n == 2
        assert result.differences.mean == pytest.approx(1.5)

    def test_replicate_spread_is_reported_separately(self) -> None:
        policy = {"c1": [10.0, 12.0, 14.0]}
        baseline = {"c1": [10.0]}
        result = paired_comparison(
            policy, baseline, policy="random", baseline="hold", metric="m", resamples=200
        )
        assert result.seed_sensitivity_sd is not None
        assert result.seed_sensitivity_sd > 0

    def test_a_deterministic_policy_has_no_seed_sensitivity(self) -> None:
        result = paired_comparison(
            {"c1": [10.0]},
            {"c1": [8.0]},
            policy="model",
            baseline="hold",
            metric="m",
            resamples=200,
        )
        assert result.seed_sensitivity_sd is None

    def test_won_lost_and_tied_sum_to_the_conditions(self) -> None:
        policy = {"a": [3.0], "b": [1.0], "c": [2.0]}
        baseline = {"a": [1.0], "b": [5.0], "c": [2.0]}
        result = paired_comparison(
            policy, baseline, policy="p", baseline="b", metric="m", resamples=200
        )

        assert (result.conditions_won, result.conditions_lost, result.conditions_tied) == (1, 1, 1)
        assert result.conditions_won + result.conditions_lost + result.conditions_tied == 3
        assert result.probability_of_outperforming == pytest.approx(1 / 3)

    def test_conditions_neither_ran_are_excluded(self) -> None:
        result = paired_comparison(
            {"a": [1.0], "b": [2.0]},
            {"a": [0.0], "c": [9.0]},
            policy="p",
            baseline="b",
            metric="m",
            resamples=200,
        )
        assert result.n_conditions == 1

    def test_sharing_no_condition_is_refused(self) -> None:
        """An unpaired comparison across different squads measures the squads."""
        with pytest.raises(ValueError, match="share no starting conditions"):
            paired_comparison(
                {"a": [1.0]},
                {"b": [1.0]},
                policy="p",
                baseline="b",
                metric="m",
                resamples=200,
            )

    def test_an_identical_policy_has_no_effect_size(self) -> None:
        result = paired_comparison(
            {"a": [1.0], "b": [2.0]},
            {"a": [1.0], "b": [2.0]},
            policy="p",
            baseline="b",
            metric="m",
            resamples=200,
        )
        assert result.effect_size_dz is None
        assert result.probability_of_outperforming == 0.0

    def test_a_diagnostic_comparison_is_labelled_as_one(self) -> None:
        """The oracle is an upper bound and must never read as a baseline."""
        result = paired_comparison(
            {"a": [5.0]},
            {"a": [1.0]},
            policy="oracle",
            baseline="hold",
            metric="m",
            resamples=200,
            diagnostic=True,
        )
        assert result.diagnostic
        assert "upper bound" in result.row()
