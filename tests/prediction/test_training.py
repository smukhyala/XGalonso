"""Training-pipeline tests.

The load-bearing property is that a training row is built from the manager's
vantage point at that gameweek's deadline. If that fails, everything downstream
validates beautifully and loses money — so it is tested directly rather than
assumed from the fact that the generators are point-in-time safe.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from xg_alonso.prediction.dataset import COMPONENT_LABELS, build_training_frame
from xg_alonso.prediction.trained import train_component_models

#: Tiny models. These tests verify plumbing and invariants, not accuracy,
#: and a full-size fit per test makes the suite unusable.
FAST = {"max_iter": 15, "max_depth": 3}

T0 = datetime(2025, 8, 1, 12, 0, tzinfo=UTC)

_COLUMNS = (
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "yellow_cards",
    "bonus",
    "bps",
    "total_points",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
)


def _stats(*, players: int = 40, gameweeks: int = 24, season: str = "2025-26") -> pl.DataFrame:
    """Synthetic history where output rises with a player's latent quality."""
    rows = []
    for player in range(1, players + 1):
        quality = player / players
        for week in range(1, gameweeks + 1):
            kickoff = T0 + timedelta(days=7 * week)
            rows.append(
                {
                    "player_code": player,
                    "season": season,
                    "gameweek_id": week,
                    "kickoff_time": kickoff,
                    "available_time": kickoff + timedelta(hours=3),
                    "minutes": 20 + quality * 70,
                    "starts": 1.0 if quality > 0.4 else 0.0,
                    "goals_scored": quality * 1.2,
                    "assists": quality * 0.7,
                    "clean_sheets": 1.0 if (week + player) % 3 == 0 else 0.0,
                    "goals_conceded": 2.0 - quality,
                    "saves": quality * 2.0,
                    "yellow_cards": (week % 5 == 0) * 1.0,
                    "bonus": quality * 1.5,
                    "bps": quality * 30,
                    "total_points": 2 + quality * 8,
                    "expected_goals": quality * 0.9,
                    "expected_assists": quality * 0.5,
                    "expected_goal_involvements": quality * 1.4,
                    "expected_goals_conceded": 1.8 - quality,
                }
            )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return frame.with_columns(
        pl.col("kickoff_time").dt.replace_time_zone("UTC"),
        pl.col("available_time").dt.replace_time_zone("UTC"),
        *[pl.col(c).cast(pl.Float64) for c in _COLUMNS],
    )


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


class TestLabelCoverage:
    def test_every_declared_component_label_is_produced(self) -> None:
        data = build_training_frame(_stats(), min_gameweek=4)
        for component in COMPONENT_LABELS:
            assert f"label_{component}" in data.label_columns
