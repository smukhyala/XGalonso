"""Horizon-valuation tests.

A transfer is permanent and its cost is paid once, so judging it on one gameweek
undervalues buying a better player and overvalues chasing a fixture. These check
that the horizon actually changes that comparison, and that the discount stays
an honest statement about uncertainty rather than a knob that quietly rescales
every number on the screen.
"""

from __future__ import annotations

import pytest

from xg_alonso.optimization.horizon import (
    DEFAULT_DISCOUNT,
    DEFAULT_HORIZON,
    discount_weights,
    transfer_is_worth_a_hit,
    value_over_horizon,
)


class TestWeights:
    def test_the_first_gameweek_is_undiscounted(self) -> None:
        """Units stay 'points in a gameweek'; a sum-normalised weighting would
        silently rescale every figure the product displays."""
        assert discount_weights()[0] == 1.0

    def test_weights_decay(self) -> None:
        weights = discount_weights()
        assert weights == tuple(sorted(weights, reverse=True))

    def test_the_default_horizon_halves_by_the_end(self) -> None:
        weights = discount_weights(DEFAULT_HORIZON, DEFAULT_DISCOUNT)
        assert 0.4 < weights[-1] < 0.6

    def test_a_discount_of_one_is_a_plain_sum(self) -> None:
        assert discount_weights(4, 1.0) == (1.0, 1.0, 1.0, 1.0)

    def test_an_empty_or_invalid_horizon_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one gameweek"):
            discount_weights(0)
        with pytest.raises(ValueError, match="discount must be"):
            discount_weights(3, 0.0)


class TestValuation:
    def test_a_single_gameweek_reduces_to_itself(self) -> None:
        assert value_over_horizon([4.2]).total == pytest.approx(4.2)

    def test_later_gameweeks_count_for_less(self) -> None:
        front = value_over_horizon([6.0, 2.0, 2.0]).total
        back = value_over_horizon([2.0, 2.0, 6.0]).total
        assert front > back

    def test_the_immediate_value_is_kept_for_comparison(self) -> None:
        """Without it there is no way to say whether the horizon changed anything."""
        value = value_over_horizon([3.0, 5.0, 5.0, 5.0, 5.0])
        assert value.immediate == 3.0
        assert value.lift_over_immediate > 0

    def test_a_one_week_spike_is_worth_less_than_it_looks(self) -> None:
        """The mistake a single-gameweek objective makes, stated as a test."""
        spike = value_over_horizon([9.0, 1.0, 1.0, 1.0, 1.0])
        steady = value_over_horizon([4.0, 4.0, 4.0, 4.0, 4.0])
        assert spike.immediate > steady.immediate
        assert steady.total > spike.total

    def test_an_empty_horizon_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one gameweek"):
            value_over_horizon([])

    def test_the_explanation_shows_every_week(self) -> None:
        rendered = value_over_horizon([3.0, 4.0, 5.0]).explain()
        for week in ("3.00", "4.00", "5.00"):
            assert week in rendered


class TestHits:
    def test_a_hit_can_pay_over_a_horizon_when_it_never_would_in_one_week(self) -> None:
        """Why managers take hits and the old objective never would have."""
        gain = value_over_horizon([1.5] * 5)
        assert gain.immediate - 4 < 0
        assert transfer_is_worth_a_hit(gain, hit_cost=4)

    def test_a_marginal_gain_still_does_not_justify_a_hit(self) -> None:
        gain = value_over_horizon([0.4] * 5)
        assert not transfer_is_worth_a_hit(gain, hit_cost=4)

    def test_a_free_transfer_needs_only_to_beat_the_threshold(self) -> None:
        gain = value_over_horizon([0.2] * 5)
        assert transfer_is_worth_a_hit(gain, hit_cost=0, threshold=0.15)
