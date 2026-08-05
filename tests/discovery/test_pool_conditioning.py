"""The wire: constraints actually change what the harness measures.

`tests/discovery/test_feasible.py` proves the mask is derived correctly.
This file proves it is *consulted* — that setting
:attr:`~xg_alonso.discovery.harness.HarnessConfig.evaluate_on` changes the
number that comes back, and changes it in the direction a manager would expect.

Without this the two modules could both be individually correct and never meet,
which is precisely the failure the audit found in the first place: a
`MetricRegistry` with no registered implementations, a `beam_search` called from
nowhere, a conditioning story that was true of the design and false of the
running code.

The central assertion is
:meth:`TestScoringIsRestricted.test_a_feature_that_only_helps_the_unreachable_is_worthless`.
It plants a feature that predicts the target *only* for expensive players, then
measures it twice: once globally, once as a manager who cannot afford any of
them. The same feature must be valuable in the first measurement and worthless
in the second, because that is the entire claim.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from xg_alonso.discovery.feasible import REACHABLE_COLUMN
from xg_alonso.discovery.harness import HarnessConfig, evaluate_feature_set
from xg_alonso.discovery.search import ScoreResult

_SEASON = "2024-25"


def _frame(*, players: int = 60, gameweeks: int = 24, seed: int = 20260804) -> pl.DataFrame:
    """A frame where the target is driven by different signals in each price tier.

    ``cheap_signal`` predicts the target for players priced under 70;
    ``premium_signal`` predicts it for players at or above 70. Neither carries
    information about the other tier, so a measurement restricted to one tier
    must see only that tier's signal.
    """
    rng = np.random.default_rng(seed)
    rows = players * gameweeks

    player_code = np.repeat(np.arange(100, 100 + players), gameweeks)
    gameweek = np.tile(np.arange(1, gameweeks + 1), players)
    # Half the field is cheap, half premium — a stable split, not a random one.
    price = np.repeat(np.where(np.arange(players) % 2 == 0, 45, 110), gameweeks).astype(np.int64)

    cheap_signal = rng.normal(size=rows)
    premium_signal = rng.normal(size=rows)
    noise = rng.normal(scale=0.35, size=rows)

    is_premium = price >= 70
    target = np.where(is_premium, 3.0 * premium_signal, 3.0 * cheap_signal) + noise

    return pl.DataFrame(
        {
            "player_code": player_code,
            "label_season": [_SEASON] * rows,
            "label_gameweek": gameweek,
            "price_tenths": price,
            "cheap_signal": cheap_signal,
            "premium_signal": premium_signal,
            "baseline_noise": rng.normal(size=rows),
            "label_total_points": target,
        }
    )


def _config(evaluate_on: str | None = None) -> HarnessConfig:
    return HarnessConfig(
        target="label_total_points",
        min_train_gameweeks=8,
        validate_gameweeks=4,
        embargo_gameweeks=1,
        max_folds=3,
        evaluate_on=evaluate_on,
    )


def _measured(result: ScoreResult) -> bool:
    """Whether the harness actually produced a number.

    `evaluate_feature_set` reports 'could not measure' as an infinite
    metric rather than raising, so 'no measurement' stays distinguishable
    from 'measured zero'.
    """
    return math.isfinite(result.metric)


def _gain(frame: pl.DataFrame, candidate: str, *, evaluate_on: str | None = None) -> float:
    """Improvement in MAE from adding ``candidate`` to the baseline.

    Positive means the candidate helped. Returns ``0.0`` when either side could
    not be measured, which is the harness's own convention for "no measurement"
    rather than "no effect".
    """
    config = _config(evaluate_on)
    base = evaluate_feature_set(frame, baseline_columns=("baseline_noise",), config=config)
    with_candidate = evaluate_feature_set(
        frame, baseline_columns=("baseline_noise",), candidate_columns=(candidate,), config=config
    )
    if not _measured(base) or not _measured(with_candidate):
        return 0.0
    return base.metric - with_candidate.metric


def _reachable(frame: pl.DataFrame, *, max_price: int) -> pl.DataFrame:
    """Attach the mask a manager with a hard price ceiling would produce."""
    return frame.with_columns((pl.col("price_tenths") <= max_price).alias(REACHABLE_COLUMN))


class TestScoringIsRestricted:
    def test_a_feature_that_only_helps_the_unreachable_is_worthless(self) -> None:
        """The whole claim, in one assertion.

        ``premium_signal`` predicts the target for expensive players and nothing
        else. Measured globally it is clearly valuable. Measured as a manager
        who cannot afford a single expensive player, it must be worth
        approximately nothing — not "somewhat less", but no better than the
        noise it is competing against.
        """
        frame = _reachable(_frame(), max_price=70)

        globally = _gain(frame, "premium_signal")
        as_a_broke_manager = _gain(frame, "premium_signal", evaluate_on=REACHABLE_COLUMN)

        assert globally > 0.15, f"the planted signal is not detectable at all ({globally:.3f})"
        assert as_a_broke_manager < globally / 4, (
            f"restricting to reachable players barely changed the measurement "
            f"({globally:.3f} global vs {as_a_broke_manager:.3f} restricted); the "
            "mask is not reaching the scorer"
        )

    def test_the_mirror_case_holds(self) -> None:
        """And a feature that helps only the *reachable* survives restriction.

        Without this, the test above would pass on a mask that simply destroyed
        every measurement.
        """
        frame = _reachable(_frame(), max_price=70)

        globally = _gain(frame, "cheap_signal")
        as_a_broke_manager = _gain(frame, "cheap_signal", evaluate_on=REACHABLE_COLUMN)

        assert as_a_broke_manager > 0.15, (
            f"the cheap signal vanished under a cheap-only pool ({as_a_broke_manager:.3f}); "
            "the mask is destroying measurements rather than focusing them"
        )
        assert as_a_broke_manager > globally * 0.8

    def test_two_managers_rank_the_same_features_differently(self) -> None:
        """Same frame, same features, different constraints, different answer.

        This is the post's claim reduced to its smallest testable form. Manager
        A can only buy cheap players; manager B only expensive ones. The feature
        each should care about is the opposite one.
        """
        cheap_only = _reachable(_frame(), max_price=70)
        premium_only = _frame().with_columns((pl.col("price_tenths") > 70).alias(REACHABLE_COLUMN))

        a_prefers_cheap = _gain(cheap_only, "cheap_signal", evaluate_on=REACHABLE_COLUMN) > _gain(
            cheap_only, "premium_signal", evaluate_on=REACHABLE_COLUMN
        )
        b_prefers_premium = _gain(
            premium_only, "premium_signal", evaluate_on=REACHABLE_COLUMN
        ) > _gain(premium_only, "cheap_signal", evaluate_on=REACHABLE_COLUMN)

        assert a_prefers_cheap
        assert b_prefers_premium


class TestDegradation:
    def test_an_absent_column_measures_globally(self) -> None:
        """A missing mask must not raise, and must not silently half-apply.

        `feasible_pool` reports separately whether a pool was applied, so the
        honest behaviour here is to measure everything — an older frame stays
        runnable and the "was it applied" question is answered in one place.
        """
        frame = _frame()
        with_flag = _gain(frame, "premium_signal", evaluate_on=REACHABLE_COLUMN)
        without = _gain(frame, "premium_signal")
        assert with_flag == pytest.approx(without)

    def test_an_all_true_mask_matches_the_unrestricted_measurement(self) -> None:
        frame = _frame().with_columns(pl.lit(value=True).alias(REACHABLE_COLUMN))
        assert _gain(frame, "cheap_signal", evaluate_on=REACHABLE_COLUMN) == pytest.approx(
            _gain(frame, "cheap_signal")
        )

    def test_a_mask_that_leaves_too_few_rows_reports_unusable(self) -> None:
        """Below the harness's validation floor, the answer is "cannot measure".

        Returning a confident number from nine validation rows is the failure
        this floor exists to prevent, and the mask must not be able to sneak
        under it.
        """
        frame = _frame().with_columns((pl.col("player_code") == 100).alias(REACHABLE_COLUMN))
        result = evaluate_feature_set(
            frame,
            baseline_columns=("baseline_noise",),
            candidate_columns=("cheap_signal",),
            config=_config(REACHABLE_COLUMN),
        )
        assert not _measured(result)

    def test_null_mask_entries_are_treated_as_unreachable(self) -> None:
        """A null is not a licence to score. It means "unknown", so exclude it."""
        frame = _frame().with_columns(
            pl.when(pl.col("price_tenths") <= 70)
            .then(pl.lit(value=True))
            .otherwise(None)
            .alias(REACHABLE_COLUMN)
        )
        strict = _reachable(_frame(), max_price=70)
        assert _gain(frame, "cheap_signal", evaluate_on=REACHABLE_COLUMN) == pytest.approx(
            _gain(strict, "cheap_signal", evaluate_on=REACHABLE_COLUMN)
        )


class TestFittingIsNotRestricted:
    """Fit on everything, evaluate on the subset — the decomposition matters.

    Masking the training rows too would confound two effects: a feature would
    look worse partly because it is worse for this manager, and partly because
    the model saw a fraction of the data. Only the second is an artefact, and
    only separating them makes the first interpretable.
    """

    def test_the_baseline_is_fitted_on_the_full_frame(self) -> None:
        """A tiny reachable set still gets a well-fitted model behind it.

        If training were masked too, restricting to a quarter of the field would
        degrade the *baseline* as well, and the measured incremental gain would
        move for reasons having nothing to do with the manager's constraints.
        """
        frame = _reachable(_frame(), max_price=70)
        restricted = evaluate_feature_set(
            frame,
            baseline_columns=("baseline_noise", "cheap_signal"),
            config=_config(REACHABLE_COLUMN),
        )
        assert _measured(restricted)
        # The cheap tier is genuinely predictable, so a model fitted on the whole
        # frame and scored on that tier must beat pure noise comfortably.
        noise_only = evaluate_feature_set(
            frame, baseline_columns=("baseline_noise",), config=_config(REACHABLE_COLUMN)
        )
        assert restricted.metric < noise_only.metric
