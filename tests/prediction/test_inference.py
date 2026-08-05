"""Inference and persistence tests.

The overlap guard gets the most attention here. Evaluating a model on data it
trained on produces a result that looks entirely reasonable and means nothing,
so the check has to be mechanical rather than a matter of remembering.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from tests.conftest import BOOTSTRAP_FIXTURE

from conftest import FAST
from conftest import synthetic_stats as _stats
from xg_alonso.contracts.identifiers import GameweekId
from xg_alonso.contracts.prediction import Position
from xg_alonso.domain.scoring import ScoringRules
from xg_alonso.prediction.dataset import build_training_frame
from xg_alonso.prediction.inference import (
    SavedModel,
    load_models,
    model_summary,
    predict_with_models,
    save_models,
)
from xg_alonso.prediction.trained import ComponentModels, train_component_models

#: What the ``fitted`` fixture yields: models, a feature frame, and the
#: seasons they were trained on.
Fitted = tuple[ComponentModels, pl.DataFrame, tuple[str, ...]]

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def rules() -> ScoringRules:
    payload = json.loads(BOOTSTRAP_FIXTURE.read_text())
    return ScoringRules.from_bootstrap(
        payload, version="2026-27", source_sha256="a" * 64, fetched_at=NOW
    )


@pytest.fixture(scope="module")
def fitted() -> Fitted:
    data = build_training_frame(_stats(players=30, gameweeks=20), min_gameweek=4)
    models = train_component_models(
        data.frame,
        feature_columns=data.feature_columns,
        label_columns=data.label_columns,
        min_train_gameweeks=8,
        model_kwargs=FAST,
    )
    frame = data.frame.head(40).with_columns(pl.lit("MID").alias("position"))
    return models, frame, data.seasons


class TestPersistence:
    def test_round_trip(self, tmp_path: Path, fitted: Fitted) -> None:
        models, _, seasons = fitted
        saved = SavedModel(
            models=models,
            trained_seasons=seasons,
            trained_gameweeks=(5, 6, 7),
            saved_at=NOW,
        )
        path = tmp_path / "m.pkl"
        save_models(saved, path)
        loaded = load_models(path)
        assert loaded.trained_seasons == seasons
        assert loaded.models.fingerprint() == models.fingerprint()

    def test_loading_a_foreign_object_is_rejected(self, tmp_path: Path) -> None:
        """Unpickling arbitrary objects and hoping is not a loading strategy."""
        import pickle

        path = tmp_path / "wrong.pkl"
        path.write_bytes(pickle.dumps({"not": "a model"}))
        with pytest.raises(TypeError, match="does not contain a SavedModel"):
            load_models(path)


class TestOverlapGuard:
    def _saved(self, seasons: tuple[str, ...], gameweeks: tuple[int, ...]) -> SavedModel:
        return SavedModel(
            models=ComponentModels(),
            trained_seasons=seasons,
            trained_gameweeks=gameweeks,
            saved_at=NOW,
        )

    def test_same_season_and_gameweeks_overlaps(self) -> None:
        saved = self._saved(("2024-25",), (6, 7, 8, 9))
        assert saved.overlaps("2024-25", (8, 9, 10))

    def test_different_season_does_not_overlap(self) -> None:
        """The intended workflow: train on last season, evaluate on this one."""
        saved = self._saved(("2024-25",), tuple(range(1, 39)))
        assert not saved.overlaps("2025-26", tuple(range(1, 39)))

    def test_same_season_disjoint_gameweeks_does_not_overlap(self) -> None:
        saved = self._saved(("2024-25",), (1, 2, 3, 4, 5))
        assert not saved.overlaps("2024-25", (20, 21, 22))

    def test_a_single_shared_gameweek_is_enough_to_overlap(self) -> None:
        saved = self._saved(("2024-25",), (1, 2, 3))
        assert saved.overlaps("2024-25", (3, 4, 5))


class TestPrediction:
    def test_it_produces_one_prediction_per_row(self, fitted: Fitted, rules: ScoringRules) -> None:
        models, frame, _ = fitted
        predictions = predict_with_models(
            frame,
            models=models,
            rules=rules,
            from_gameweek=GameweekId(10),
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="t",
            code_version="v",
            feature_set_version="catalogue_v1",
        )
        assert len(predictions) == frame.height

    def test_points_are_assembled_through_the_domain(
        self, fitted: Fitted, rules: ScoringRules
    ) -> None:
        """Nothing in the model layer knows what a goal is worth."""
        models, frame, _ = fitted
        prediction = predict_with_models(
            frame,
            models=models,
            rules=rules,
            from_gameweek=GameweekId(10),
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="t",
            code_version="v",
            feature_set_version="catalogue_v1",
        )[0]
        assert prediction.breakdown.total == pytest.approx(prediction.expected_points)
        assert prediction.scoring_rules_version == rules.version

    def test_minutes_are_reconciled_into_a_valid_distribution(
        self, fitted: Fitted, rules: ScoringRules
    ) -> None:
        """The two minutes models are independent and can disagree.

        The contract rejects an incoherent distribution, so the reconciliation
        has to happen before construction rather than at validation time.
        """
        models, frame, _ = fitted
        for prediction in predict_with_models(
            frame,
            models=models,
            rules=rules,
            from_gameweek=GameweekId(10),
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="t",
            code_version="v",
            feature_set_version="catalogue_v1",
        ):
            minutes = prediction.components.minutes
            assert minutes.p_start <= minutes.p_appearance + 1e-9
            assert minutes.p_60_plus <= minutes.p_appearance + 1e-9
            assert 0.0 <= minutes.expected_minutes <= 120.0

    def test_provenance_is_complete(self, fitted: Fitted, rules: ScoringRules) -> None:
        models, frame, _ = fitted
        prediction = predict_with_models(
            frame,
            models=models,
            rules=rules,
            from_gameweek=GameweekId(10),
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="t",
            code_version="v",
            feature_set_version="catalogue_v1",
        )[0]
        assert len(prediction.provenance.model_artifact_sha256) == 64
        assert prediction.provenance.feature_set_version == "catalogue_v1"
        assert prediction.provenance.data_cutoff == NOW

    def test_unknown_positions_are_skipped_not_guessed(
        self, fitted: Fitted, rules: ScoringRules
    ) -> None:
        models, frame, _ = fitted
        mixed = frame.with_columns(
            pl.when(pl.int_range(pl.len()) < 5)
            .then(pl.lit("AM"))  # Assistant Manager: not a footballer
            .otherwise(pl.lit("MID"))
            .alias("position")
        )
        predictions = predict_with_models(
            mixed,
            models=models,
            rules=rules,
            from_gameweek=GameweekId(10),
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="t",
            code_version="v",
            feature_set_version="catalogue_v1",
        )
        assert len(predictions) == mixed.height - 5
        assert all(p.position is Position.MID for p in predictions)

    def test_an_empty_frame_yields_nothing(self, fitted: Fitted, rules: ScoringRules) -> None:
        models, frame, _ = fitted
        assert (
            predict_with_models(
                frame.head(0),
                models=models,
                rules=rules,
                from_gameweek=GameweekId(10),
                data_cutoff=NOW,
                predicted_at=NOW,
                run_id="t",
                code_version="v",
                feature_set_version="catalogue_v1",
            )
            == []
        )

    def test_a_missing_required_column_fails_loudly(
        self, fitted: Fitted, rules: ScoringRules
    ) -> None:
        models, frame, _ = fitted
        with pytest.raises(KeyError, match="position"):
            predict_with_models(
                frame.drop("position"),
                models=models,
                rules=rules,
                from_gameweek=GameweekId(10),
                data_cutoff=NOW,
                predicted_at=NOW,
                run_id="t",
                code_version="v",
                feature_set_version="catalogue_v1",
            )


class TestSummary:
    def test_it_reports_what_the_model_was_trained_on(self, fitted: Fitted) -> None:
        models, _, seasons = fitted
        summary = model_summary(
            SavedModel(
                models=models,
                trained_seasons=seasons,
                trained_gameweeks=(5, 6),
                saved_at=NOW,
            )
        )
        assert summary["seasons"] == list(seasons)
        assert summary["labels"]
        assert len(summary["fingerprint"]) == 12


class TestSkillWeightedShrinkage:
    """Shrinkage pulls weak components toward the constant they barely improve on."""

    def test_shrinkage_moves_predictions_toward_the_population_mean(self, fitted: Fitted) -> None:
        models, frame, _ = fitted
        raw = models.predict(frame, shrink_by_skill=False)
        shrunk = models.predict(frame, shrink_by_skill=True)

        moved = 0
        for label in raw:
            mean = models.label_means.get(label)
            if mean is None:
                continue
            raw_spread = float(raw[label].std())
            shrunk_spread = float(shrunk[label].std())
            if raw_spread > 1e-9:
                assert shrunk_spread <= raw_spread + 1e-9, f"{label} spread grew under shrinkage"
                moved += 1
        assert moved > 0, "no component was affected, so the switch does nothing"

    def test_spread_is_scaled_by_exactly_the_measured_skill(self, fitted: Fitted) -> None:
        """The blend has an exact property, so assert it rather than a tendency.

        ``shrunk = w * raw + (1 - w) * mean`` is affine, so the spread scales by
        exactly ``w``. Checking that pins the arithmetic without depending on two
        labels happening to differ in skill.
        """
        models, frame, _ = fitted
        skill = models.skill_by_label()
        raw = models.predict(frame, shrink_by_skill=False)
        shrunk = models.predict(frame, shrink_by_skill=True)

        checked = 0
        for label, values in raw.items():
            spread = float(values.std())
            if spread <= 1e-6 or label not in models.label_means:
                continue
            weight = max(0.0, min(1.0, skill.get(label, 0.0)))
            assert float(shrunk[label].std()) == pytest.approx(weight * spread, rel=1e-6), (
                f"{label} spread did not scale by its skill"
            )
            checked += 1
        assert checked > 0, "no label had enough spread to check"

    def test_a_zero_skill_component_collapses_to_its_mean(self, fitted: Fitted) -> None:
        """A component that beats nothing should contribute nothing but a constant."""
        models, frame, _ = fitted
        label = next(iter(models.models))
        stripped = ComponentModels(
            models={label: models.models[label]},
            feature_columns=models.feature_columns,
            reports=[],  # no reports means zero measured skill
            trained_on_rows=models.trained_on_rows,
            label_means={label: 4.2},
        )
        shrunk = stripped.predict(frame, shrink_by_skill=True)[label]
        assert float(shrunk.std()) == pytest.approx(0.0, abs=1e-9)
        assert float(shrunk.mean()) == pytest.approx(4.2)

    def test_label_means_are_recorded_for_every_label(self, fitted: Fitted) -> None:
        """A prediction is only as useful as the fallback it can blend with."""
        models, _, _ = fitted
        for label in models.models:
            assert label in models.label_means

    def test_shrinkage_is_off_by_default(self, fitted: Fitted) -> None:
        """Behaviour that changes results belongs behind an explicit switch."""
        models, frame, _ = fitted
        default = models.predict(frame)
        explicit = models.predict(frame, shrink_by_skill=False)
        for label in default:
            assert (default[label] == explicit[label]).all()

    def test_a_stale_artifact_is_rejected_at_load(self, tmp_path: Path) -> None:
        """Pickle restores objects missing fields added since they were written.

        Without this check the failure surfaces much later, inside a prediction
        loop, as an AttributeError with no indication of the cause.
        """
        import pickle

        stale = SavedModel(
            models=ComponentModels(),
            trained_seasons=("2024-25",),
            trained_gameweeks=(1, 2),
            saved_at=NOW,
        )
        del stale.models.label_means

        path = tmp_path / "stale.pkl"
        path.write_bytes(pickle.dumps(stale))
        with pytest.raises(ValueError, match="older version"):
            load_models(path)
