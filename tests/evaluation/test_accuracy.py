"""Tests for scoring assembled expected points against outcomes.

The most important test here is `test_constant_predictor_is_flagged_degenerate`.
It is a negative control: it reconstructs the exact failure that shipped once
already — a model predicting zero for everybody, scoring well on MAE — and
asserts this module refuses to call it skilful. If that test ever passes because
the guard was removed, the guard was doing nothing.
"""

from __future__ import annotations

import polars as pl
import pytest

from xg_alonso.evaluation import score_predictions, spearman
from xg_alonso.evaluation.accuracy import PRICE_BANDS


def _frame(
    predicted: list[float],
    actual: list[float],
    *,
    position: list[str] | None = None,
    price: list[int] | None = None,
    minutes: list[int] | None = None,
) -> pl.DataFrame:
    data: dict[str, list[object]] = {
        "player_code": list(range(1, len(predicted) + 1)),
        "predicted": list(predicted),
        "actual": list(actual),
    }
    if position is not None:
        data["position"] = list(position)
    if price is not None:
        data["price"] = list(price)
    if minutes is not None:
        data["minutes"] = list(minutes)
    return pl.DataFrame(data)


class TestSpearman:
    def test_perfect_agreement_is_one(self) -> None:
        assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)

    def test_perfect_inversion_is_minus_one(self) -> None:
        assert spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)

    def test_monotone_but_nonlinear_still_scores_one(self) -> None:
        """Rank correlation must not care about scale — that is why it is used."""
        assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 1000.0]) == pytest.approx(1.0)

    def test_constant_prediction_has_no_rank_information(self) -> None:
        assert spearman([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]) == 0.0

    def test_too_few_points_returns_zero(self) -> None:
        assert spearman([1.0], [1.0]) == 0.0

    def test_ties_share_average_rank(self) -> None:
        """Two tied predictions must not be ordered arbitrarily."""
        assert spearman([1.0, 1.0, 2.0], [5.0, 5.0, 9.0]) == pytest.approx(1.0)


class TestDegeneracyGuard:
    def test_constant_predictor_is_flagged_degenerate(self) -> None:
        """The negative control.

        Labels here are 90% zeros, so predicting zero everywhere yields a very
        low MAE and beats the mean-predicting baseline outright. That is exactly
        the shape that scored +48% skill before the guard existed.
        """
        actual = [0.0] * 18 + [8.0, 12.0]
        report = score_predictions(_frame([0.0] * 20, actual))

        assert report.overall.mae < report.overall.baseline_mae, (
            "precondition: a constant zero must genuinely beat the mean on this "
            "label, otherwise the control proves nothing"
        )
        assert report.overall.degenerate
        assert report.overall.skill == 0.0
        assert any("constant" in w for w in report.warnings)

    def test_working_predictor_is_not_flagged(self) -> None:
        actual = [0.0, 2.0, 6.0, 9.0, 1.0, 4.0]
        report = score_predictions(_frame([0.5, 2.5, 5.0, 8.0, 1.5, 3.5], actual))

        assert not report.overall.degenerate
        assert report.overall.skill > 0.0
        assert report.overall.spearman > 0.9

    def test_degenerate_slice_reports_zero_skill_per_position(self) -> None:
        """The guard must apply inside slices, not only to the pooled number."""
        report = score_predictions(
            _frame(
                [0.0, 0.0, 0.0, 1.0, 5.0, 9.0],
                [0.0, 0.0, 7.0, 1.0, 5.0, 9.0],
                position=["GKP", "GKP", "GKP", "MID", "MID", "MID"],
            )
        )
        slices = {s.name: s for s in report.by_position}
        assert slices["GKP"].degenerate
        assert slices["GKP"].skill == 0.0
        assert not slices["MID"].degenerate


class TestErrorMetrics:
    def test_mae_rmse_and_bias(self) -> None:
        report = score_predictions(_frame([3.0, 5.0], [1.0, 9.0]))
        # Errors are +2 and -4.
        assert report.overall.mae == pytest.approx(3.0)
        assert report.overall.rmse == pytest.approx(pytest.approx(10.0**0.5))
        assert report.overall.bias == pytest.approx(-1.0)

    def test_bias_sign_means_over_prediction(self) -> None:
        """Positive bias must mean the model predicts too high, not too low."""
        report = score_predictions(_frame([5.0, 5.0], [1.0, 1.0]))
        assert report.overall.bias > 0

    def test_rmse_punishes_large_errors_more_than_mae(self) -> None:
        """RMSE exceeds MAE only when error magnitudes differ.

        Equal-magnitude errors give RMSE == MAE exactly, so the uneven case is
        the one that actually demonstrates the penalty.
        """
        uneven = score_predictions(_frame([5.0, 5.0], [5.0, 15.0])).overall
        assert uneven.rmse > uneven.mae

        equal_magnitude = score_predictions(_frame([0.0, 10.0], [5.0, 5.0])).overall
        assert equal_magnitude.rmse == pytest.approx(equal_magnitude.mae)

        perfect = score_predictions(_frame([5.0, 5.0], [5.0, 5.0])).overall
        assert perfect.rmse == pytest.approx(0.0)


class TestTopK:
    def test_precision_rewards_picking_the_real_best(self) -> None:
        # Prediction ranks players exactly right.
        predicted = [float(i) for i in range(20)]
        actual = [float(i) for i in range(20)]
        report = score_predictions(_frame(predicted, actual))
        top10 = next(t for t in report.top_k if t.k == 10)
        assert top10.precision == 1.0
        assert top10.lift > 1.0

    def test_inverted_ranking_scores_zero_precision(self) -> None:
        predicted = [float(i) for i in range(20)]
        actual = [float(20 - i) for i in range(20)]
        report = score_predictions(_frame(predicted, actual))
        top10 = next(t for t in report.top_k if t.k == 10)
        assert top10.precision == 0.0
        assert top10.lift < 1.0

    def test_ranking_uses_predictions_not_outcomes(self) -> None:
        """Guards against measuring hindsight.

        If top-k ever sorted by `actual`, precision would be 1.0 no matter how
        bad the model was. This asserts a deliberately terrible model scores
        badly, which only holds if the ranking really comes from predictions.
        """
        predicted = [1.0] * 10 + [0.0] * 10
        actual = [0.0] * 10 + [10.0] * 10
        report = score_predictions(_frame(predicted, actual))
        top10 = next(t for t in report.top_k if t.k == 10)
        assert top10.precision == 0.0
        assert top10.mean_actual < top10.mean_overall

    def test_ranking_is_measured_within_a_gameweek(self) -> None:
        """The decision unit is one deadline, not a whole season.

        This model ranks players perfectly *inside* each gameweek but has no
        idea which gameweek is high-scoring. A manager choosing at each deadline
        would be served perfectly, so per-gameweek precision must be 1.0 — while
        the pooled ranking, which asks a question nobody faces, is misled by the
        cross-week scale difference.
        """
        # Twelve players a week. Within each week the prediction order matches
        # the outcome order exactly; week 2 simply scores an order of magnitude
        # higher than week 1, which no per-deadline decision depends on.
        per_week = 12
        frame = pl.DataFrame(
            {
                "player_code": list(range(1, 2 * per_week + 1)),
                "gameweek_id": [1] * per_week + [2] * per_week,
                "predicted": [float(i) for i in range(per_week, 0, -1)] * 2,
                "actual": [float(i) for i in range(per_week, 0, -1)]
                + [float(100 + i) for i in range(per_week, 0, -1)],
            }
        )
        grouped = next(t for t in score_predictions(frame).top_k if t.k == 10)
        assert grouped.precision == 1.0, "perfect within-week ranking must score 1.0"

        pooled = next(t for t in score_predictions(frame.drop("gameweek_id")).top_k if t.k == 10)
        assert pooled.precision < 1.0, (
            "pooling weeks asks a question no manager faces, and must not be "
            "what the grouped metric reports"
        )

    def test_grouped_lift_averages_across_gameweeks(self) -> None:
        per_week = 20
        # In each week one player returns everything and the rest blank, so the
        # predicted top ten captures that whole return.
        frame = pl.DataFrame(
            {
                "player_code": list(range(1, 2 * per_week + 1)),
                "gameweek_id": [1] * per_week + [2] * per_week,
                "predicted": [float(i) for i in range(per_week, 0, -1)] * 2,
                "actual": [20.0] + [0.0] * (per_week - 1) + [10.0] + [0.0] * (per_week - 1),
            }
        )
        top10 = next(t for t in score_predictions(frame).top_k if t.k == 10)
        # Week 1: top ten mean 20/10 = 2.0, field mean 20/20 = 1.0.
        # Week 2: top ten mean 10/10 = 1.0, field mean 10/20 = 0.5.
        assert top10.mean_actual == pytest.approx(1.5)
        assert top10.mean_overall == pytest.approx(0.75)
        assert top10.lift == pytest.approx(2.0)

    def test_top_k_is_permutation_invariant(self) -> None:
        """Row order must not change the result, including through ties."""
        predicted = [5.0, 5.0, 5.0, 1.0, 9.0, 9.0]
        actual = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        base = _frame(predicted, actual)
        shuffled = base.sort("player_code", descending=True)

        first = score_predictions(base).top_k
        second = score_predictions(shuffled).top_k
        assert [t.precision for t in first] == [t.precision for t in second]


class TestSlices:
    def test_price_bands_partition_without_overlap(self) -> None:
        """Every price maps to exactly one band, or a player is double-counted."""
        for price in (0, 39, 40, 54, 55, 79, 80, 109, 110, 145, 300):
            matches = [name for name, low, high in PRICE_BANDS if low <= price < high]
            assert len(matches) == 1, f"price {price} matched {matches}"

    def test_price_band_slices_are_reported(self) -> None:
        report = score_predictions(
            _frame(
                [0.5, 1.0, 4.0, 7.0],
                [0.0, 2.0, 5.0, 6.0],
                price=[40, 65, 95, 130],
            )
        )
        assert [s.name for s in report.by_price_band] == [name for name, _, _ in PRICE_BANDS]
        assert all(s.n == 1 for s in report.by_price_band)

    def test_played_only_excludes_non_appearances(self) -> None:
        report = score_predictions(
            _frame(
                [0.1, 0.1, 5.0, 6.0],
                [0.0, 0.0, 6.0, 4.0],
                minutes=[0, 0, 90, 90],
            )
        )
        assert report.appeared_only is not None
        assert report.appeared_only.n == 2
        assert report.overall.n == 4

    def test_played_only_is_harder_than_pooled(self) -> None:
        """Non-appearances are trivially predictable and inflate the headline."""
        predicted = [0.0] * 8 + [5.0, 6.0]
        actual = [0.0] * 8 + [9.0, 1.0]
        report = score_predictions(_frame(predicted, actual, minutes=[0] * 8 + [90, 90]))
        assert report.appeared_only is not None
        assert report.appeared_only.mae > report.overall.mae


class TestContract:
    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="needs columns"):
            score_predictions(pl.DataFrame({"player_code": [1], "predicted": [1.0]}))

    def test_nulls_are_dropped_and_reported(self) -> None:
        frame = pl.DataFrame(
            {
                "player_code": [1, 2, 3],
                "predicted": [1.0, None, 3.0],
                "actual": [1.0, 2.0, 3.0],
            }
        )
        report = score_predictions(frame)
        assert report.overall.n == 2
        assert any("null" in w for w in report.warnings)

    def test_empty_input_does_not_crash(self) -> None:
        report = score_predictions(
            pl.DataFrame(
                {"player_code": [], "predicted": [], "actual": []},
                schema={"player_code": pl.Int64, "predicted": pl.Float64, "actual": pl.Float64},
            )
        )
        assert report.overall.n == 0
        assert "no scorable rows" in report.warnings

    def test_optional_columns_are_genuinely_optional(self) -> None:
        report = score_predictions(_frame([1.0, 2.0], [1.0, 3.0]))
        assert report.by_position == ()
        assert report.by_price_band == ()
        assert report.appeared_only is None
        assert report.overall.n == 2

    def test_summary_renders_every_section(self) -> None:
        report = score_predictions(
            _frame(
                [0.5, 1.0, 4.0, 7.0, 2.0, 3.0],
                [0.0, 2.0, 5.0, 6.0, 1.0, 4.0],
                position=["GKP", "DEF", "MID", "FWD", "MID", "DEF"],
                price=[40, 65, 95, 130, 60, 50],
                minutes=[0, 90, 90, 90, 45, 90],
            )
        )
        text = report.summary()
        assert "By position" in text
        assert "By price band" in text
        assert "Ranking" in text
        assert "played only" in text
