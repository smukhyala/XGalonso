"""Freeze checks, each paired with a negative control.

Every check here guards a failure that produces a *better*-looking number, so
none of them will ever be caught by a result that seems wrong. That is exactly
why each needs a control proving it fires: a freeze harness that never fails is
indistinguishable from one that is broken, and the broken version is worse than
none because it manufactures confidence.

The pattern is the one `features/leakage.py::assert_detects_leakage` already
established for the leakage harness.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from xg_alonso.contracts.evaluation import (
    EvaluationWindow,
    ExperimentConfig,
    ModelSpec,
    PolicyKind,
    PolicyParameters,
    PolicySpec,
    SquadCohortSpec,
    SquadSource,
)
from xg_alonso.contracts.folds import WalkForwardFold
from xg_alonso.contracts.identifiers import GameweekId
from xg_alonso.evaluation.frozen import FreezeViolation, assert_frozen
from xg_alonso.prediction.inference import SavedModel
from xg_alonso.prediction.trained import ComponentModels

NOW = datetime(2026, 7, 1, tzinfo=UTC)


def _saved(
    *,
    seasons: tuple[str, ...] = ("2022-23",),
    gameweeks: tuple[int, ...] = (4, 5, 6),
    columns: tuple[str, ...] = ("a", "b"),
    folds: tuple[WalkForwardFold, ...] = (),
) -> SavedModel:
    return SavedModel(
        models=ComponentModels(feature_columns=columns, folds=folds),
        trained_seasons=seasons,
        trained_gameweeks=gameweeks,
        saved_at=NOW,
    )


def _config(**overrides: object) -> ExperimentConfig:
    base = {
        "name": "t",
        "windows": (EvaluationWindow(season="2024-25", start_gameweeks=(6,), end_gameweek=12),),
        "squads": (SquadCohortSpec(source=SquadSource.MOST_EXPENSIVE_LEGAL),),
        "models": (ModelSpec(name="trained"),),
        "policies": (PolicySpec(name="p", selector=PolicyKind.MODEL, model="trained"),),
    }
    return ExperimentConfig(**{**base, **overrides})  # type: ignore[arg-type]


class TestACleanConfigPasses:
    def test_every_check_runs_and_reports(self) -> None:
        checks = assert_frozen(_config(), models={"trained": _saved()})

        assert len(checks) == 8
        assert all(c.passed for c in checks)

    def test_the_record_is_returned_even_on_success(self) -> None:
        """ "We checked and it was fine" is only auditable if it is written down."""
        checks = assert_frozen(_config(), models={"trained": _saved()})
        assert {c.name for c in checks} == {
            "train_seasons_precede_test",
            "no_model_overlap_with_evaluation_period",
            "architecture_and_hyperparameters_frozen",
            "feature_selection_frozen",
            "calibration_not_fitted_on_test",
            "thresholds_not_selected_on_test",
            "folds_are_walk_forward",
            "inputs_are_pinned",
        }

    def test_a_closed_form_model_has_nothing_to_freeze(self) -> None:
        checks = assert_frozen(_config(), models={"trained": None})
        assert all(c.passed for c in checks)


class TestCheck1TrainSeasonsPrecedeTest:
    def test_training_on_the_evaluated_season_fires(self) -> None:
        with pytest.raises(FreezeViolation, match="train_seasons_precede_test"):
            assert_frozen(_config(), models={"trained": _saved(seasons=("2024-25",))})

    def test_training_on_a_later_season_fires(self) -> None:
        """Strictly stronger than SavedModel.overlaps, which passes this."""
        model = _saved(seasons=("2025-26",))
        assert not model.overlaps("2024-25", (6, 7, 8))  # the weaker check is happy

        with pytest.raises(FreezeViolation, match="train_seasons_precede_test"):
            assert_frozen(_config(), models={"trained": model})

    def test_training_on_an_earlier_season_passes(self) -> None:
        checks = assert_frozen(_config(), models={"trained": _saved(seasons=("2023-24",))})
        assert next(c for c in checks if c.name == "train_seasons_precede_test").passed


class TestCheck2NoGameweekOverlap:
    def test_an_overlapping_gameweek_within_the_season_fires(self) -> None:
        with pytest.raises(FreezeViolation):
            assert_frozen(
                _config(),
                models={"trained": _saved(seasons=("2024-25",), gameweeks=(8, 9))},
            )


class TestCheck3ArchitectureFrozen:
    def test_a_fingerprint_mismatch_fires(self) -> None:
        with pytest.raises(FreezeViolation, match="architecture_and_hyperparameters_frozen"):
            assert_frozen(
                _config(models=(ModelSpec(name="trained", expected_fingerprint="f" * 64),)),
                models={"trained": _saved()},
            )

    def test_a_matching_fingerprint_passes(self) -> None:
        saved = _saved()
        checks = assert_frozen(
            _config(
                models=(ModelSpec(name="trained", expected_fingerprint=saved.models.fingerprint()),)
            ),
            models={"trained": saved},
        )
        assert next(c for c in checks if c.name.startswith("architecture")).passed

    def test_pinning_nothing_does_not_claim_to_be_frozen(self) -> None:
        checks = assert_frozen(_config(), models={"trained": _saved()})
        assert next(c for c in checks if c.name.startswith("architecture")).passed


class TestCheck4FeatureSelectionFrozen:
    def test_a_widened_feature_set_fires(self) -> None:
        import hashlib

        pinned = hashlib.sha256(b"a|b").hexdigest()
        with pytest.raises(FreezeViolation, match="feature_selection_frozen"):
            assert_frozen(
                _config(models=(ModelSpec(name="trained", expected_feature_columns_hash=pinned),)),
                models={"trained": _saved(columns=("a", "b", "discovered_c"))},
            )

    def test_the_pinned_feature_set_passes(self) -> None:
        import hashlib

        pinned = hashlib.sha256(b"a|b").hexdigest()
        checks = assert_frozen(
            _config(models=(ModelSpec(name="trained", expected_feature_columns_hash=pinned),)),
            models={"trained": _saved(columns=("a", "b"))},
        )
        assert next(c for c in checks if c.name == "feature_selection_frozen").passed


class TestCheck5CalibrationNotFittedOnTest:
    def test_evaluating_on_the_calibration_season_fires(self) -> None:
        """This fires today for any 2025-26 run with calibration on."""
        with pytest.raises(FreezeViolation, match="calibration_not_fitted_on_test"):
            assert_frozen(
                _config(
                    windows=(
                        EvaluationWindow(season="2025-26", start_gameweeks=(6,), end_gameweek=12),
                    ),
                    models=(ModelSpec(name="trained", apply_price_calibration=True),),
                ),
                models={"trained": _saved()},
            )

    def test_a_different_season_with_calibration_passes(self) -> None:
        checks = assert_frozen(
            _config(models=(ModelSpec(name="trained", apply_price_calibration=True),)),
            models={"trained": _saved()},
        )
        assert next(c for c in checks if c.name == "calibration_not_fitted_on_test").passed

    def test_the_calibration_season_without_calibration_passes(self) -> None:
        """The overlap only matters if a model actually applies the correction."""
        checks = assert_frozen(
            _config(
                windows=(EvaluationWindow(season="2025-26", start_gameweeks=(6,), end_gameweek=12),)
            ),
            models={"trained": _saved()},
        )
        assert next(c for c in checks if c.name == "calibration_not_fitted_on_test").passed


class TestCheck6ThresholdsNotSelectedOnTest:
    def test_tuning_on_the_evaluated_season_fires(self) -> None:
        with pytest.raises(FreezeViolation, match="thresholds_not_selected_on_test"):
            assert_frozen(
                _config(parameters=PolicyParameters(tuned_on_seasons=("2024-25",))),
                models={"trained": _saved()},
            )

    def test_tuning_on_a_different_season_passes(self) -> None:
        checks = assert_frozen(
            _config(parameters=PolicyParameters(tuned_on_seasons=("2022-23",))),
            models={"trained": _saved()},
        )
        assert next(c for c in checks if c.name == "thresholds_not_selected_on_test").passed


class TestCheck7FoldsAreWalkForward:
    def test_a_backwards_fold_fires(self) -> None:
        """Re-asserted on the loaded object rather than trusting the pickle."""
        legal = WalkForwardFold(
            fold_index=0,
            train_start=GameweekId(1),
            train_end=GameweekId(10),
            validate_start=GameweekId(11),
            validate_end=GameweekId(12),
            embargo_gameweeks=0,
        )
        # Bypass the constructor the way a stale pickle would.
        leaky = legal.model_construct(
            fold_index=0,
            train_start=GameweekId(1),
            train_end=GameweekId(10),
            validate_start=GameweekId(8),
            validate_end=GameweekId(12),
            embargo_gameweeks=0,
        )
        with pytest.raises(FreezeViolation, match="folds_are_walk_forward"):
            assert_frozen(_config(), models={"trained": _saved(folds=(leaky,))})

    def test_a_forward_fold_passes(self) -> None:
        legal = WalkForwardFold(
            fold_index=0,
            train_start=GameweekId(1),
            train_end=GameweekId(10),
            validate_start=GameweekId(12),
            validate_end=GameweekId(13),
            embargo_gameweeks=1,
        )
        checks = assert_frozen(_config(), models={"trained": _saved(folds=(legal,))})
        assert next(c for c in checks if c.name == "folds_are_walk_forward").passed


class TestCheck8InputsArePinned:
    def test_a_changed_data_manifest_fires(self) -> None:
        with pytest.raises(FreezeViolation, match="inputs_are_pinned"):
            assert_frozen(
                _config(data_manifest_sha256="a" * 64),
                models={"trained": _saved()},
                data_manifest_sha256="b" * 64,
            )

    def test_a_matching_manifest_passes(self) -> None:
        checks = assert_frozen(
            _config(data_manifest_sha256="a" * 64),
            models={"trained": _saved()},
            data_manifest_sha256="a" * 64,
        )
        assert next(c for c in checks if c.name == "inputs_are_pinned").passed

    def test_pinning_nothing_is_not_a_failure(self) -> None:
        """A first run has not produced the hashes it wants pinned yet."""
        checks = assert_frozen(_config(), models={"trained": _saved()})
        assert next(c for c in checks if c.name == "inputs_are_pinned").passed
