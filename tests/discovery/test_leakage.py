"""Point-in-time safety of discovered features, and of the machinery around them.

**Every positive test here is paired with a negative control.** A leakage harness
that never fails is indistinguishable from a broken one, and the broken version
is more dangerous than no harness at all because it manufactures confidence. So
each group proves both that the real thing is clean *and* that a deliberately
leaky twin is caught.

Four distinct leaks are covered, because they are four different mistakes:

1. a feature program reading records not yet available
2. a feature reading the target it is meant to predict
3. an embedding or cluster model fitted on rows it is then applied to
4. a fold whose validation window overlaps its training window
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from tests.discovery.conftest import T0, make_entities, make_history
from xg_alonso.contracts.folds import walk_forward_folds
from xg_alonso.discovery.clusters import fit_clusters
from xg_alonso.discovery.compile import CompileContext, compile_program, validate_program
from xg_alonso.discovery.dsl import (
    Arith,
    ArithOp,
    EwmMean,
    FeatureProgram,
    GroupKey,
    GroupRel,
    GroupRelOp,
    Lag,
    Rolling,
    RollingAgg,
    ShrunkRate,
    Source,
    TimeSince,
    Trend,
)
from xg_alonso.discovery.embeddings import fit_embedding
from xg_alonso.discovery.harness import HarnessConfig, _folds
from xg_alonso.features.generators import stage_window
from xg_alonso.features.leakage import (
    assert_detects_leakage,
    assert_no_leakage,
    find_leakage,
    make_future_records,
)

pytestmark = pytest.mark.leakage


def _programs() -> list[FeatureProgram]:
    """One program per temporal primitive. Each must be independently clean."""
    return [
        FeatureProgram(
            name="p_shrunk",
            root=ShrunkRate(numerator="expected_goals", denominator="minutes", window=5),
        ),
        FeatureProgram(name="p_rolling", root=Rolling(child=Source(column="minutes"), window=5)),
        FeatureProgram(
            name="p_std",
            root=Rolling(
                child=Source(column="total_points"), window=5, agg=RollingAgg.STD, min_periods=3
            ),
        ),
        FeatureProgram(name="p_lag", root=Lag(child=Source(column="total_points"), periods=1)),
        FeatureProgram(
            name="p_ewm", root=EwmMean(child=Source(column="bps"), window=10, halflife=3.0)
        ),
        FeatureProgram(name="p_trend", root=Trend(child=Source(column="threat"), window=5)),
        FeatureProgram(name="p_since", root=TimeSince(event_column="minutes")),
        FeatureProgram(
            name="p_group",
            root=GroupRel(
                op=GroupRelOp.ZSCORE,
                by=GroupKey.POSITION,
                child=Rolling(child=Source(column="total_points"), window=5),
            ),
        ),
        FeatureProgram(
            name="p_interaction",
            root=Arith(
                op=ArithOp.MUL,
                left=ShrunkRate(numerator="expected_goals", denominator="minutes", window=5),
                right=TimeSince(event_column="minutes"),
            ),
        ),
    ]


def _builder(program: FeatureProgram):  # type: ignore[no-untyped-def]
    def build(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
        return compile_program(
            program, entities, CompileContext(player_stats=source, stage=stage_window)
        )

    return build


class TestProgramsCannotSeeTheFuture:
    """Rebuild with future-stamped records appended; nothing may move."""

    @pytest.mark.parametrize("program", _programs(), ids=lambda p: p.name)
    def test_program_is_point_in_time_safe(self, program: FeatureProgram) -> None:
        history = make_history()
        entities = make_entities()
        future = make_future_records(
            history, after=T0 + timedelta(days=400), entity_keys=("player_code",), rows=6
        )
        assert_no_leakage(
            _builder(program),
            entities=entities,
            source=history,
            future_records=future,
            compare_columns=[program.name],
        )

    def test_the_harness_catches_a_leaky_builder(self) -> None:
        """The negative control. Without this, every result above is worthless."""
        history = make_history()
        entities = make_entities()
        future = make_future_records(
            history, after=T0 + timedelta(days=400), entity_keys=("player_code",), rows=6
        )

        def leaky(ents: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            # Aggregates the WHOLE history, ignoring available_time entirely —
            # the single most common leak in fantasy-sports modelling.
            everything = source.group_by("player_code").agg(
                pl.col("total_points").mean().alias("p_rolling")
            )
            return ents.join(everything, on="player_code", how="left")

        assert_detects_leakage(leaky, entities=entities, source=history, future_records=future)

    def test_leakage_is_reported_per_column(self) -> None:
        history = make_history()
        entities = make_entities()
        future = make_future_records(
            history, after=T0 + timedelta(days=400), entity_keys=("player_code",), rows=6
        )

        def leaky(ents: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            totals = source.group_by("player_code").agg(
                pl.col("threat").sum().alias("leaked_column")
            )
            return ents.join(totals, on="player_code", how="left")

        assert find_leakage(leaky, entities=entities, source=history, future_records=future) == [
            "leaked_column"
        ]

    def test_an_empty_future_set_proves_nothing_and_says_so(self) -> None:
        history = make_history()
        with pytest.raises(ValueError, match="proves nothing"):
            find_leakage(
                _builder(_programs()[0]),
                entities=make_entities(),
                source=history,
                future_records=history.head(0),
            )


class TestRollingWindowsStopAtTheCutoff:
    """The window is recomputed from each row's own vantage point."""

    def test_a_lag_reads_the_last_visible_match_only(self) -> None:
        history = make_history(players=(1,), rows=6)
        # Cutoff between gameweek 3 and gameweek 4.
        cutoff = history.filter(pl.col("gameweek_id") == 3)["available_time"][0] + timedelta(
            hours=1
        )
        entities = pl.DataFrame(
            {"player_code": [1], "prediction_timestamp": [cutoff], "position": ["MID"]}
        )
        program = FeatureProgram(
            name="last_points", root=Lag(child=Source(column="total_points"), periods=1)
        )
        out = compile_program(
            program, entities, CompileContext(player_stats=history, stage=stage_window)
        )
        expected = history.filter(pl.col("gameweek_id") == 3)["total_points"][0]
        assert out["last_points"][0] == pytest.approx(float(expected))

    def test_a_rolling_mean_uses_only_visible_rows(self) -> None:
        history = make_history(players=(1,), rows=8)
        cutoff = history.filter(pl.col("gameweek_id") == 4)["available_time"][0] + timedelta(
            hours=1
        )
        entities = pl.DataFrame(
            {"player_code": [1], "prediction_timestamp": [cutoff], "position": ["MID"]}
        )
        program = FeatureProgram(
            name="mean_3", root=Rolling(child=Source(column="total_points"), window=3)
        )
        out = compile_program(
            program, entities, CompileContext(player_stats=history, stage=stage_window)
        )
        visible = history.filter(pl.col("gameweek_id").is_in([2, 3, 4]))["total_points"].to_list()
        assert out["mean_3"][0] == pytest.approx(sum(visible) / 3)


class TestTargetLeakage:
    """A feature may not read the label it is predicting."""

    def test_reading_a_label_column_is_refused_statically(self) -> None:
        program = FeatureProgram(
            name="cheat", root=Rolling(child=Source(column="label_total_points"), window=3)
        )
        issues = validate_program(
            program,
            available_columns=("label_total_points", "minutes"),
            forbidden_columns=("label_total_points",),
        )
        assert any(i.code == "target_leakage" for i in issues)

    def test_a_legitimate_column_is_not_flagged(self) -> None:
        program = FeatureProgram(
            name="fine", root=Rolling(child=Source(column="minutes"), window=3)
        )
        issues = validate_program(
            program,
            available_columns=("label_total_points", "minutes"),
            forbidden_columns=("label_total_points",),
        )
        assert issues == []

    def test_an_unknown_column_is_refused(self) -> None:
        program = FeatureProgram(
            name="ghost", root=Rolling(child=Source(column="pressing_intensity"), window=3)
        )
        issues = validate_program(program, available_columns=("minutes",))
        assert any(i.code == "unknown_column" for i in issues)


class TestModelsAreFittedOnTrainingRowsOnly:
    """Scalers, projections and clusters must never see the rows they score."""

    def _rows(self, *, drifted: bool = False) -> pl.DataFrame:
        """Two structurally different populations.

        **A location shift would not work here**, and finding that out was the
        point of writing the control. Clustering happens in a standardised space,
        so adding a constant to every column changes nothing after z-scoring —
        both the positive test and its control would have passed vacuously, one
        of them for entirely the wrong reason.

        So the drift changes *shape*: different spreads, and a reversed
        relationship between minutes and threat. That is a population a refitted
        model genuinely partitions differently.
        """
        generator = np.random.default_rng(11 if not drifted else 99)
        minutes = generator.normal(60, 10 if not drifted else 30, 120)
        threat = (
            minutes * 0.5 + generator.normal(0, 3, 120)
            if not drifted
            else -minutes * 0.9 + generator.normal(0, 12, 120)
        )
        return pl.DataFrame(
            {
                "player_code": list(range(1, 121)),
                "position": ["MID"] * 120,
                "minutes_mean_5": minutes,
                "total_points_mean_5": generator.normal(4, 2 if not drifted else 9, 120),
                "threat_per90_5": threat,
                "selected_mean_5": generator.normal(1000, 300 if not drifted else 20, 120),
                "expected_goals_per90_5": generator.normal(0.3, 0.1, 120),
            }
        )

    def test_an_embedding_does_not_move_when_new_rows_are_scored(self) -> None:
        train = self._rows()
        model = fit_embedding(train, n_components=3, min_rows=10)
        before = model.transform(train)

        # Score a structurally different population. The stored mean, scale and
        # loadings must be untouched, so the training vectors are identical.
        model.transform(self._rows(drifted=True))
        assert np.allclose(before, model.transform(train))

        # And the control: refitting on that population really does differ.
        refitted = fit_embedding(self._rows(drifted=True), n_components=3, min_rows=10)
        assert not np.allclose(model.mean, refitted.mean) or not np.allclose(
            model.scale, refitted.scale
        )

    def test_a_cluster_model_does_not_refit_on_the_rows_it_scores(self) -> None:
        train = self._rows()
        model = fit_clusters(train, k=3, seed=5)
        centroids = model.centroids.copy()

        model.assign(self._rows(drifted=True))

        assert np.allclose(centroids, model.centroids)
        # And the training assignment is unchanged by having scored the drift.
        first, _, _ = model.assign(train)
        second, _, _ = model.assign(train)
        assert np.array_equal(first, second)

    def test_negative_control_a_refit_would_move_the_centroids(self) -> None:
        """The control: fitting on the drifted rows *does* change the model.

        Without this the test above could pass because the drift was too small to
        matter, rather than because the API prevents refitting.
        """
        fitted_on_train = fit_clusters(self._rows(), k=3, seed=5)
        fitted_on_drift = fit_clusters(self._rows(drifted=True), k=3, seed=5)
        assert not np.allclose(fitted_on_train.centroids, fitted_on_drift.centroids)


class TestFoldIsolation:
    """Validation always lies strictly in the future of training."""

    def test_no_fold_overlaps_its_own_training_window(self, training_frame: pl.DataFrame) -> None:
        splits = _folds(training_frame, HarnessConfig(min_train_gameweeks=8, max_folds=4))
        assert splits, "fixture should produce at least one usable fold"
        for split in splits:
            train_weeks = set(split.train["__timeline"].to_list())
            validate_weeks = set(split.validate["__timeline"].to_list())
            assert not (train_weeks & validate_weeks)
            assert min(validate_weeks) > max(train_weeks)

    def test_the_embargo_is_respected(self, training_frame: pl.DataFrame) -> None:
        splits = _folds(
            training_frame, HarnessConfig(min_train_gameweeks=8, embargo_gameweeks=2, max_folds=3)
        )
        for split in splits:
            gap = min(split.validate["__timeline"].to_list()) - max(
                split.train["__timeline"].to_list()
            )
            assert gap > 2

    def test_holdout_seasons_never_appear_in_any_fold(self, training_frame: pl.DataFrame) -> None:
        splits = _folds(
            training_frame,
            HarnessConfig(min_train_gameweeks=8, holdout_seasons=("2025-26",), max_folds=4),
        )
        for split in splits:
            for frame in (split.train, split.validate):
                assert "2025-26" not in set(frame["label_season"].to_list())

    def test_the_only_fold_constructor_refuses_a_shuffled_sequence(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            walk_forward_folds(
                gameweeks=[3, 1, 2, 4, 5, 6, 7, 8],  # type: ignore[list-item]
                min_train_gameweeks=3,
            )

    def test_a_season_boundary_does_not_wrap_backwards(self, training_frame: pl.DataFrame) -> None:
        """GW3 of a later season must sort after GW24 of an earlier one."""
        splits = _folds(training_frame, HarnessConfig(min_train_gameweeks=20, max_folds=6))
        for split in splits:
            latest_train_season = max(split.train["label_season"].to_list())
            earliest_validate_season = min(split.validate["label_season"].to_list())
            assert earliest_validate_season >= latest_train_season


class TestGroupPrimitivesDoNotPoolAcrossCutoffs:
    """A group-relative value compares players at the *same* deadline."""

    def test_group_statistics_use_only_the_current_batch(self) -> None:
        history = make_history(players=(1, 2, 3), rows=10)
        entities = make_entities(players=(1, 2, 3))
        program = FeatureProgram(
            name="rank_in_position",
            root=GroupRel(
                op=GroupRelOp.RANK,
                by=GroupKey.ALL,
                child=Rolling(child=Source(column="total_points"), window=5),
            ),
        )
        ctx = CompileContext(player_stats=history, stage=stage_window)
        full = compile_program(program, entities, ctx)

        # The same players scored in a two-player batch must rank against each
        # other, not against an absent third.
        pair = compile_program(program, make_entities(players=(1, 2)), ctx)
        assert pair["rank_in_position"].to_list() == [0.0, 1.0]
        assert full["rank_in_position"].to_list() == [0.0, 0.5, 1.0]

    def test_shrunk_rate_refuses_a_pooled_prior_across_cutoffs(self) -> None:
        """The shipped generator refuses this case outright; the DSL inherits it."""
        from xg_alonso.features.generators import shrunk_rate_as_of

        history = make_history()
        mixed = pl.concat([make_entities(days=100), make_entities(days=300)], how="vertical")
        with pytest.raises(ValueError, match="distinct prediction timestamps"):
            shrunk_rate_as_of(
                mixed,
                history,
                entity_keys=["player_code"],
                numerator="expected_goals",
                denominator="minutes",
                window=5,
                prior_strength=3.0,
                output_name="rate",
            )
