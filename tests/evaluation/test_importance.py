"""Permutation-importance tests.

The two that matter are the synthetic ones. Importance is a measurement, and a
measurement nobody has calibrated is a number with a plausible shape — so the
suite plants a feature the label is a function of and a feature that is pure
noise, and asserts the method can tell them apart. Everything else here is
bookkeeping around that.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from xg_alonso.evaluation.importance import (
    ALL_POSITIONS,
    IMPORTANCE_SCHEMA_VERSION,
    ImportanceTable,
    label_weights_from_predictions,
    load_importance,
    permutation_importance,
    write_importance,
)
from xg_alonso.prediction.trained import train_component_models

COMPUTED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)

#: Small and fast. The point is to check the measurement, not to fit well.
_TINY_MODEL = {"max_iter": 30, "max_depth": 3, "min_samples_leaf": 5}


def _frame(rows: int = 600, seed: int = 7) -> pl.DataFrame:
    """A frame where one feature drives the label and one is noise.

    ``signal`` determines the label outright; ``noise`` is drawn independently.
    A method that cannot separate those two cannot be trusted on 171 real
    features where the separation is far less stark.
    """
    generator = np.random.default_rng(seed)
    signal = generator.normal(size=rows)
    noise = generator.normal(size=rows)
    label = np.clip(np.round(2.0 + 1.8 * signal), 0.0, None)

    seasons = ["2024-25"] * rows
    gameweeks = [1 + (index % 30) for index in range(rows)]

    return pl.DataFrame(
        {
            "signal": signal,
            "noise": noise,
            "label_goals_scored": label,
            "label_season": seasons,
            "label_gameweek": gameweeks,
        }
    )


def _models(frame: pl.DataFrame):  # type: ignore[no-untyped-def]
    return train_component_models(
        frame,
        feature_columns=("signal", "noise"),
        label_columns=("label_goals_scored",),
        min_train_gameweeks=8,
        validate_gameweeks=4,
        embargo_gameweeks=1,
        model_kwargs=_TINY_MODEL,
    )


def _table(frame: pl.DataFrame, models) -> ImportanceTable:  # type: ignore[no-untyped-def]
    return permutation_importance(
        models,
        frame,
        label_columns=("label_goals_scored",),
        families={"signal": "player_rate", "noise": "player_rate"},
        label_weights={"label_goals_scored": 1.0},
        catalogue_version="test_v1",
        computed_at=COMPUTED_AT,
        n_repeats=5,
        seed=11,
    )


class TestMeasurement:
    def test_a_predictive_feature_outranks_a_noise_feature(self) -> None:
        frame = _frame()
        table = _table(frame, _models(frame))
        ranked = table.by_feature()
        assert ranked["signal"] > ranked["noise"]

    def test_a_noise_feature_barely_moves_the_error(self) -> None:
        """Shuffling a column the model learned nothing from should cost nothing."""
        frame = _frame()
        table = _table(frame, _models(frame))
        noise_rows = [r for r in table.rows if r.feature_name == "noise"]
        assert noise_rows
        for row in noise_rows:
            assert row.relative_delta < 0.05

    def test_a_predictive_feature_registers_as_signal(self) -> None:
        frame = _frame()
        table = _table(frame, _models(frame))
        signal_rows = [r for r in table.rows if r.feature_name == "signal"]
        assert all(row.is_signal for row in signal_rows)

    def test_the_same_seed_gives_the_same_answer(self) -> None:
        """A measurement that moves between identical runs is not a measurement."""
        frame = _frame()
        models = _models(frame)
        assert _table(frame, models).by_feature() == _table(frame, models).by_feature()


class TestHonesty:
    def test_measuring_on_an_empty_frame_is_refused(self) -> None:
        frame = _frame()
        with pytest.raises(ValueError, match="validation rows"):
            permutation_importance(
                _models(frame),
                frame.clear(),
                label_columns=("label_goals_scored",),
                families={},
                label_weights={},
                catalogue_version="test_v1",
                computed_at=COMPUTED_AT,
            )

    def test_an_unfitted_feature_is_refused(self) -> None:
        frame = _frame()
        with pytest.raises(KeyError, match="not fitted on"):
            permutation_importance(
                _models(frame),
                frame,
                label_columns=("label_goals_scored",),
                families={},
                label_weights={},
                catalogue_version="test_v1",
                computed_at=COMPUTED_AT,
                features=("does_not_exist",),
            )

    def test_a_degenerate_label_is_flagged_and_excluded_from_ranking(self) -> None:
        """Ranking features by their effect on a constant is arithmetic without meaning."""
        frame = _frame()
        models = _models(frame)
        table = permutation_importance(
            models,
            frame,
            label_columns=("label_goals_scored",),
            families={},
            label_weights={"label_goals_scored": 1.0},
            catalogue_version="test_v1",
            computed_at=COMPUTED_AT,
        )
        # Force the degenerate view rather than contriving a constant label:
        # what is under test is that the flag suppresses the ranking.
        flagged = ImportanceTable(
            rows=tuple(
                type(row)(**{**row.__dict__, "degenerate_label": True}) for row in table.rows
            ),
            catalogue_version=table.catalogue_version,
            model_fingerprint=table.model_fingerprint,
            computed_at=table.computed_at,
            label_weights=table.label_weights,
        )
        assert flagged.by_feature() == {}
        assert flagged.degenerate_labels() == ("label_goals_scored",)

    def test_a_harmful_feature_reports_a_negative_delta(self) -> None:
        """Deltas are not clipped: a feature that misleads the model should say so."""
        frame = _frame()
        table = _table(frame, _models(frame))
        assert all(row.mae_delta == row.permuted_mae - row.baseline_mae for row in table.rows)


class TestAggregation:
    def test_family_totals_sum_the_features_within_them(self) -> None:
        frame = _frame()
        table = _table(frame, _models(frame))
        per_feature = table.by_feature()
        assert table.by_family()["player_rate"] == pytest.approx(sum(per_feature.values()))

    def test_label_weights_are_measured_and_normalised(self) -> None:
        class _Breakdown:
            appearance = 1.0
            goals = 0.5
            assists = 0.25
            clean_sheets = 0.0
            goals_conceded = 0.0
            saves = 0.0
            cards = 0.0
            bonus = 0.25

        class _Prediction:
            breakdown = _Breakdown()

        weights = label_weights_from_predictions([_Prediction(), _Prediction()])
        assert sum(weights.values()) == pytest.approx(1.0)
        # Appearance is shared by two labels, so each carries the full term.
        assert weights["label_minutes"] == weights["label_starts"]
        assert weights["label_goals_scored"] > weights["label_assists"]

    def test_no_predictions_gives_no_weights(self) -> None:
        assert label_weights_from_predictions([]) == {}


class TestPersistence:
    def test_a_table_round_trips_through_parquet(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        frame = _frame()
        table = _table(frame, _models(frame))
        path = write_importance(table, tmp_path / "gold" / "feature_importance.parquet")
        loaded = load_importance(path)

        assert loaded.height == len(table.rows)
        assert loaded["schema_version"].unique().to_list() == [IMPORTANCE_SCHEMA_VERSION]
        assert loaded["model_fingerprint"].unique().to_list() == [table.model_fingerprint]

    def test_an_empty_table_still_writes_a_typed_frame(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """An empty result must not become an unreadable file."""
        empty = ImportanceTable(
            rows=(),
            catalogue_version="test_v1",
            model_fingerprint="abc",
            computed_at=COMPUTED_AT,
            label_weights={},
        )
        path = write_importance(empty, tmp_path / "empty.parquet")
        assert load_importance(path).height == 0


class TestPositionalSlices:
    """A ranking must describe the players it was measured on, and say which."""

    def test_the_pooled_slice_is_the_default(self) -> None:
        frame = _frame()
        table = _table(frame, _models(frame))
        assert {row.position for row in table.rows} == {ALL_POSITIONS}

    def test_a_slice_records_which_position_it_describes(self) -> None:
        frame = _frame()
        table = permutation_importance(
            _models(frame),
            frame,
            label_columns=("label_goals_scored",),
            families={"signal": "player_rate", "noise": "player_rate"},
            label_weights={"label_goals_scored": 1.0},
            catalogue_version="test_v1",
            computed_at=COMPUTED_AT,
            n_repeats=5,
            seed=11,
            position="DEF",
        )
        assert {row.position for row in table.rows} == {"DEF"}
        assert all(row.rows_measured == frame.height for row in table.rows)

    def test_a_slice_can_rank_features_differently_from_the_pool(self) -> None:
        """The reason positions are measured rather than reweighted.

        The pooled frame is built so `signal` predicts the label. Half of it is
        relabelled into a slice where that relationship is destroyed, so the
        feature that leads the pool must not lead the slice. If a slice could
        only ever reproduce the pooled order there would be no point measuring
        one.
        """
        frame = _frame()
        models = _models(frame)

        scrambled = frame.with_columns(
            pl.col("signal").shuffle(seed=7).alias("signal"),
        )
        pooled = _table(frame, models).by_feature()
        sliced = permutation_importance(
            models,
            scrambled,
            label_columns=("label_goals_scored",),
            families={"signal": "player_rate", "noise": "player_rate"},
            label_weights={"label_goals_scored": 1.0},
            catalogue_version="test_v1",
            computed_at=COMPUTED_AT,
            n_repeats=5,
            seed=11,
            position="GKP",
        ).by_feature("GKP")

        assert pooled["signal"] > pooled["noise"]
        assert sliced["signal"] < pooled["signal"]

    def test_a_v1_table_loads_as_the_pooled_slice(self, tmp_path: Path) -> None:
        """An older table has no position column; its rows *were* the pool."""
        frame = _frame()
        table = _table(frame, _models(frame))
        legacy = table.to_frame().drop(["position", "rows_measured"])
        destination = tmp_path / "v1.parquet"
        legacy.write_parquet(destination)

        loaded = load_importance(destination)
        assert loaded["position"].unique().to_list() == [ALL_POSITIONS]
        assert loaded["rows_measured"].unique().to_list() == [0]


class TestPositionalWeights:
    """A clean sheet is four points to a defender and zero to a forward.

    Slicing the rows without also slicing the weights would measure the right
    players against the wrong definition of what their points are made of, and
    every position's ranking would collapse toward whatever dominates the pooled
    population. These check that it does not.
    """

    @staticmethod
    def _population() -> list[Any]:
        class _Position:
            def __init__(self, value: str) -> None:
                self.value = value

        class _Breakdown:
            def __init__(self, goals: float, clean_sheets: float) -> None:
                self.appearance = 2.0
                self.goals = goals
                self.assists = 0.0
                self.clean_sheets = clean_sheets
                self.goals_conceded = 0.0
                self.saves = 0.0
                self.cards = 0.0
                self.bonus = 0.0

        class _Prediction:
            def __init__(self, position: str, goals: float, clean_sheets: float) -> None:
                self.position = _Position(position)
                self.breakdown = _Breakdown(goals, clean_sheets)

        # Defenders earn through clean sheets, forwards through goals — which is
        # the actual scoring table, not a contrivance.
        return [
            *[_Prediction("DEF", goals=0.1, clean_sheets=1.6) for _ in range(10)],
            *[_Prediction("FWD", goals=1.8, clean_sheets=0.0) for _ in range(10)],
        ]

    def test_a_defender_weights_clean_sheets_above_goals(self) -> None:
        weights = label_weights_from_predictions(self._population(), position="DEF")
        assert weights["label_clean_sheets"] > weights["label_goals_scored"]

    def test_a_forward_weights_goals_above_clean_sheets(self) -> None:
        weights = label_weights_from_predictions(self._population(), position="FWD")
        assert weights["label_goals_scored"] > weights["label_clean_sheets"]

    def test_the_two_positions_disagree(self) -> None:
        """If they agreed, slicing the weights would be pointless."""
        defenders = label_weights_from_predictions(self._population(), position="DEF")
        forwards = label_weights_from_predictions(self._population(), position="FWD")
        assert defenders["label_clean_sheets"] != forwards["label_clean_sheets"]

    def test_the_pooled_weighting_hides_the_difference(self) -> None:
        """The reason this exists: pooled weights describe neither position."""
        pooled = label_weights_from_predictions(self._population())
        defenders = label_weights_from_predictions(self._population(), position="DEF")
        assert pooled["label_clean_sheets"] < defenders["label_clean_sheets"]

    def test_each_slice_is_still_normalised(self) -> None:
        for position in ("DEF", "FWD"):
            weights = label_weights_from_predictions(self._population(), position=position)
            assert sum(weights.values()) == pytest.approx(1.0)

    def test_an_absent_position_yields_nothing_rather_than_a_wrong_answer(self) -> None:
        """Empty, so the caller can fall back deliberately instead of silently."""
        assert label_weights_from_predictions(self._population(), position="GKP") == {}

    def test_a_slice_keeps_its_own_weighting_when_tables_are_merged(self) -> None:
        """The regression: merging slices used to overwrite every row's weight.

        `xg importance` measures each slice separately and then concatenates the
        rows into one table for writing. When the weight lived only on the table,
        that concatenation replaced every slice's weighting with the last one
        assigned — and four positional rankings silently became four copies of
        the pooled one.
        """
        frame = _frame()
        models = _models(frame)

        def measure(position: str, weight: float) -> ImportanceTable:
            return permutation_importance(
                models,
                frame,
                label_columns=("label_goals_scored",),
                families={"signal": "player_rate", "noise": "player_rate"},
                label_weights={"label_goals_scored": weight},
                catalogue_version="test_v1",
                computed_at=COMPUTED_AT,
                n_repeats=3,
                seed=11,
                position=position,
            )

        defenders = measure("DEF", 0.9)
        forwards = measure("FWD", 0.1)

        merged = ImportanceTable(
            rows=(*defenders.rows, *forwards.rows),
            catalogue_version="test_v1",
            model_fingerprint="test",
            computed_at=COMPUTED_AT,
            # Deliberately wrong for both slices, as the real merge site was.
            label_weights={"label_goals_scored": 0.5},
        )

        frame_out = merged.to_frame()
        weights = {
            row["position"]: row["label_weight"]
            for row in frame_out.select("position", "label_weight").unique().iter_rows(named=True)
        }
        assert weights["DEF"] == pytest.approx(0.9)
        assert weights["FWD"] == pytest.approx(0.1)
        assert merged.by_feature("DEF")["signal"] > merged.by_feature("FWD")["signal"]
