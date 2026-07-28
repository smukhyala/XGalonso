"""Training-pipeline tests.

The load-bearing property is that a training row is built from the manager's
vantage point at that gameweek's deadline. If that fails, everything downstream
validates beautifully and loses money — so it is tested directly rather than
assumed from the fact that the generators are point-in-time safe.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest
from conftest import FAST, T0
from conftest import synthetic_stats as _stats

from xg_alonso.prediction.dataset import (
    COMPONENT_LABELS,
    TrainingData,
    build_training_frame,
)
from xg_alonso.prediction.trained import ComponentModels, train_component_models


class TestTrainingFrame:
    def test_it_produces_rows_with_features_and_labels(self) -> None:
        data = build_training_frame(_stats(), min_gameweek=4)
        assert data.rows > 0
        assert all(c in data.frame.columns for c in data.feature_columns)
        assert all(c in data.frame.columns for c in data.label_columns)

    def test_labels_are_components_not_points(self) -> None:
        """D8: a model trained on point totals learns a scoring system that changes."""
        data = build_training_frame(_stats(), min_gameweek=4)
        assert "label_total_points" not in data.label_columns
        assert "label_goals_scored" in data.label_columns
        assert "label_minutes" in data.label_columns

    def test_early_gameweeks_are_skipped(self) -> None:
        data = build_training_frame(_stats(), min_gameweek=6)
        assert min(data.gameweeks) >= 6

    def test_empty_selection_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="no rows"):
            build_training_frame(_stats(), seasons=["1999-00"])

    def test_features_cannot_see_the_labelled_gameweek(self) -> None:
        """The property everything else depends on.

        A row for gameweek N must be identical whether or not gameweek N's own
        results exist in the source. If it is not, the model is being trained on
        the answer.
        """
        full = _stats(gameweeks=16)
        target_gw = 12

        # Drop the target gameweek's results and rebuild the same row.
        without_target = full.filter(pl.col("gameweek_id") != target_gw)
        # Re-add only the label-bearing rows with impossible values, so any leak
        # into the features would be glaring.
        poisoned = full.with_columns(
            pl.when(pl.col("gameweek_id") == target_gw)
            .then(pl.lit(999.0))
            .otherwise(pl.col("goals_scored"))
            .alias("goals_scored")
        )

        clean = build_training_frame(full, min_gameweek=4).frame.filter(
            pl.col("label_gameweek") == target_gw
        )
        tainted = build_training_frame(poisoned, min_gameweek=4).frame.filter(
            pl.col("label_gameweek") == target_gw
        )

        # Compare per player, not by position. Row order is not the property
        # under test here, and comparing positionally would report a reordering
        # as contamination.
        feature = "goals_scored_mean_5"
        joined = clean.select("player_code", feature).join(
            tainted.select("player_code", pl.col(feature).alias("__tainted")),
            on="player_code",
            how="inner",
        )
        assert joined.height == clean.height
        changed = joined.filter(pl.col(feature) != pl.col("__tainted"))
        assert changed.is_empty(), (
            f"{changed.height} players' features changed when the labelled "
            "gameweek's own results changed — the training set is contaminated"
        )
        del without_target

    def test_row_order_is_deterministic(self) -> None:
        """An unstable frame can change how a model bins ties across runs."""
        stats = _stats(gameweeks=12)
        first = build_training_frame(stats, min_gameweek=4).frame
        second = build_training_frame(stats, min_gameweek=4).frame
        assert first["player_code"].to_list() == second["player_code"].to_list()
        assert first.equals(second)


class TestTrainedModels:
    def test_it_learns_something_from_signal_bearing_data(self) -> None:
        """On data where output tracks quality, the model must beat a constant."""
        data = build_training_frame(_stats(players=60, gameweeks=26), min_gameweek=4)
        models = train_component_models(
            data.frame,
            feature_columns=data.feature_columns,
            label_columns=data.label_columns,
            min_train_gameweeks=8,
            validate_gameweeks=4,
            model_kwargs=FAST,
        )
        skill = models.skill_by_label()
        assert skill, "no folds produced a report"
        assert skill["label_minutes"] > 0.1, (
            f"minutes should be highly learnable here, got {skill['label_minutes']:.1%}"
        )

    def test_folds_are_walk_forward(self) -> None:
        data = build_training_frame(_stats(gameweeks=26), min_gameweek=4)
        models = train_component_models(
            data.frame,
            feature_columns=data.feature_columns,
            label_columns=data.label_columns,
            min_train_gameweeks=8,
            validate_gameweeks=4,
            embargo_gameweeks=1,
            model_kwargs=FAST,
        )
        assert models.folds
        for fold in models.folds:
            assert fold.validate_start > fold.train_end
            assert fold.validate_start - fold.train_end - 1 == 1, "embargo must be honoured"

    def test_predictions_have_one_value_per_row(self) -> None:
        data = build_training_frame(_stats(), min_gameweek=4)
        models = train_component_models(
            data.frame,
            feature_columns=data.feature_columns,
            label_columns=data.label_columns,
            min_train_gameweeks=8,
            model_kwargs=FAST,
        )
        predictions = models.predict(data.frame.head(25))
        assert predictions
        for label, values in predictions.items():
            assert len(values) == 25, f"{label} returned the wrong number of predictions"

    def test_counts_are_never_negative(self) -> None:
        """A negative expected goal count is not a small error, it is nonsense."""
        data = build_training_frame(_stats(), min_gameweek=4)
        models = train_component_models(
            data.frame,
            feature_columns=data.feature_columns,
            label_columns=data.label_columns,
            min_train_gameweeks=8,
            model_kwargs=FAST,
        )
        for label, values in models.predict(data.frame.head(50)).items():
            assert (values >= 0).all(), f"{label} produced a negative prediction"

    def test_probabilities_stay_in_range(self) -> None:
        data = build_training_frame(_stats(), min_gameweek=4)
        models = train_component_models(
            data.frame,
            feature_columns=data.feature_columns,
            label_columns=data.label_columns,
            min_train_gameweeks=8,
            model_kwargs=FAST,
        )
        predictions = models.predict(data.frame.head(50))
        for label in ("label_clean_sheets", "label_starts"):
            if label in predictions:
                assert ((predictions[label] >= 0) & (predictions[label] <= 1)).all()

    def test_fingerprint_is_stable_and_identifying(self) -> None:
        """Provenance needs an artifact hash that changes when the model does."""
        data = build_training_frame(_stats(), min_gameweek=4)
        kwargs = {
            "feature_columns": data.feature_columns,
            "label_columns": data.label_columns,
            "min_train_gameweeks": 8,
        }
        first = train_component_models(data.frame, model_kwargs=FAST, **kwargs)  # type: ignore[arg-type]
        again = train_component_models(data.frame, model_kwargs=FAST, **kwargs)  # type: ignore[arg-type]
        assert first.fingerprint() == again.fingerprint()
        assert len(first.fingerprint()) == 64

        fewer = train_component_models(
            data.frame,
            feature_columns=data.feature_columns[:20],
            label_columns=data.label_columns,
            min_train_gameweeks=8,
            model_kwargs=FAST,
        )
        assert fewer.fingerprint() != first.fingerprint()

    def test_empty_training_frame_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            train_component_models(
                pl.DataFrame(),
                feature_columns=("a",),
                label_columns=("label_minutes",),
            )

    def test_too_little_history_is_rejected(self) -> None:
        """Training without out-of-sample evidence is fitting, not training."""
        data = build_training_frame(_stats(gameweeks=8), min_gameweek=4)
        with pytest.raises(ValueError, match="need at least"):
            train_component_models(
                data.frame,
                feature_columns=data.feature_columns,
                label_columns=data.label_columns,
                min_train_gameweeks=30,
                model_kwargs=FAST,
            )


class TestAllNullFeatures:
    """An entirely-null column crashes the estimator's binner.

    Not hypothetical: `defensive_contribution` exists for one season only, so
    any training run excluding 2025/26 already carried an all-null column. The
    failure surfaces as a numpy sliding-window error from inside sklearn that
    names no column.
    """

    def test_an_all_null_column_is_dropped_rather_than_crashing(self) -> None:
        from xg_alonso.prediction.trained import usable_features

        frame = pl.DataFrame(
            {
                "good": [1.0, 2.0, 3.0, 4.0],
                "empty": [None, None, None, None],
                "sparse": [1.0, None, None, 2.0],
            }
        )
        usable = usable_features(frame, ("good", "empty", "sparse"))
        assert "empty" not in usable
        assert "good" in usable
        assert "sparse" in usable, "a partly-null column is still informative"

    def test_a_missing_column_is_not_reported_as_usable(self) -> None:
        frame = pl.DataFrame({"good": [1.0, 2.0]})
        from xg_alonso.prediction.trained import usable_features

        assert usable_features(frame, ("good", "absent")) == ("good",)


class TestLabelCoverage:
    def test_every_declared_component_label_is_produced(self) -> None:
        data = build_training_frame(_stats(), min_gameweek=4)
        for component in COMPONENT_LABELS:
            assert f"label_{component}" in data.label_columns


class TestZeroInflatedLabelsDoNotCollapse:
    """Regression tests for the constant-output bug.

    ``loss="absolute_error"`` fits the conditional *median*. FPL component
    labels are 74-97% zeros, so the median is zero everywhere and every
    regression model collapsed to a constant zero — while still scoring ~+50%
    on an MAE-based skill metric, because predicting zero genuinely does
    minimise MAE for a rare event.

    The result was a model whose expected points contained no attacking signal
    at all, and a metric that applauded it. Both halves are tested here.
    """

    def _sparse_stats(self, *, players: int = 60, gameweeks: int = 26) -> pl.DataFrame:
        """History where goals are rare but genuinely predictable from quality."""
        rows = []
        for player in range(1, players + 1):
            quality = player / players
            for week in range(1, gameweeks + 1):
                kickoff = T0 + timedelta(days=7 * week)
                # Good players score roughly every third game, poor ones never.
                scored = 1.0 if (quality > 0.7 and week % 3 == 0) else 0.0
                rows.append(
                    {
                        "player_code": player,
                        "season": "2025-26",
                        "gameweek_id": week,
                        "opponent_team_id": ((week + player) % 6) + 1,
                        "was_home": (week + player) % 2 == 0,
                        "kickoff_time": kickoff,
                        "available_time": kickoff + timedelta(hours=3),
                        "minutes": 20 + quality * 70,
                        "starts": 1.0 if quality > 0.4 else 0.0,
                        "goals_scored": scored,
                        "assists": scored * 0.5,
                        "clean_sheets": 1.0 if (week + player) % 3 == 0 else 0.0,
                        "goals_conceded": 2.0 - quality,
                        "saves": quality * 2.0,
                        "yellow_cards": (week % 5 == 0) * 1.0,
                        "bonus": scored * 3.0,
                        "bps": quality * 30,
                        "total_points": 2 + quality * 8,
                        "expected_goals": quality * 0.9,
                        "expected_assists": quality * 0.5,
                        "expected_goal_involvements": quality * 1.4,
                        "expected_goals_conceded": 1.8 - quality,
                        "defensive_contribution": float(week % 5) + quality,
                        "selected": 100000 * quality + week,
                        "transfers_in": 5000 * quality + week * 3,
                        "transfers_out": 3000 * (1 - quality) + week * 2,
                        "transfers_balance": 2000 * quality - week,
                        "influence": quality * 40 + week * 0.3,
                        "creativity": quality * 30 + week * 0.2,
                        "threat": quality * 50 + week * 0.4,
                        "ict_index": quality * 12 + week * 0.1,
                    }
                )
        frame = pl.DataFrame(rows, infer_schema_length=None)
        return frame.with_columns(
            pl.col("kickoff_time").dt.replace_time_zone("UTC"),
            pl.col("available_time").dt.replace_time_zone("UTC"),
        )

    def _fit(self) -> tuple[ComponentModels, TrainingData]:
        data = build_training_frame(self._sparse_stats(), min_gameweek=4)
        models = train_component_models(
            data.frame,
            feature_columns=data.feature_columns,
            label_columns=data.label_columns,
            min_train_gameweeks=8,
            model_kwargs=FAST,
        )
        return models, data

    def test_the_label_really_is_zero_inflated(self) -> None:
        """Guards the fixture: without sparsity these tests prove nothing."""
        data = build_training_frame(self._sparse_stats(), min_gameweek=4)
        zeros = (data.frame["label_goals_scored"] == 0).sum() / data.rows
        assert zeros > 0.7, f"fixture is only {zeros:.0%} zeros; not sparse enough to test"

    def test_a_sparse_count_model_does_not_output_a_constant(self) -> None:
        """The bug itself: every regression collapsed to exactly zero."""
        models, data = self._fit()
        predictions = models.predict(data.frame.head(200))
        for label in ("label_goals_scored", "label_minutes", "label_saves"):
            if label not in predictions:
                continue
            values = predictions[label]
            assert float(values.std()) > 1e-6, (
                f"{label} produced a constant. A model that cannot distinguish "
                "players contributes only an offset to expected points."
            )

    def test_predictions_track_the_mean_not_the_median(self) -> None:
        """Expected points needs E[X]. The median of a rare event is zero."""
        models, data = self._fit()
        frame = data.frame
        predictions = models.predict(frame)
        actual = float(sum(frame["label_goals_scored"].to_list()) / frame.height)
        predicted = float(predictions["label_goals_scored"].mean())
        assert predicted > actual * 0.4, (
            f"predicted mean {predicted:.4f} against actual {actual:.4f}: the model "
            "is fitting the median, so the component is effectively absent"
        )

    def test_degenerate_models_are_reported_as_such(self) -> None:
        """The metric must not applaud a constant."""
        models, _ = self._fit()
        assert models.degenerate_labels() == [], (
            f"degenerate components: {models.degenerate_labels()}"
        )

    def test_skill_is_zeroed_for_a_constant_model(self) -> None:
        """A constant predictor has no skill however small its error.

        Built directly rather than by fitting, because the whole point is that
        a constant model can score highly on MAE.
        """
        from xg_alonso.prediction.trained import FoldReport

        constant = FoldReport(
            label="label_goals_scored",
            fold_index=0,
            train_rows=1000,
            validate_rows=200,
            mae=0.04,
            baseline_mae=0.078,  # a genuinely better MAE than predicting the mean
            prediction_std=0.0,  # ...but the output never varies
            predicted_mean=0.0,
            actual_mean=0.04,
        )
        assert constant.skill > 0.4, "MAE skill alone would rate this highly"
        assert constant.degenerate
        models = ComponentModels(reports=[constant])
        assert models.skill_by_label()["label_goals_scored"] == 0.0
        assert models.degenerate_labels() == ["label_goals_scored"]

    def test_mean_bias_is_reported(self) -> None:
        """A component biased low drags the whole assembled total down."""
        models, _ = self._fit()
        bias = models.mean_bias_by_label()
        assert bias
        for label, value in bias.items():
            assert abs(value) < 5.0, f"{label} mean bias {value:.3f} is implausible"
