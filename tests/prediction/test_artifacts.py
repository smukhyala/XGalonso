"""Artifact manifests and the compatibility gate.

The case that matters most is the one with no natural exception. A *missing*
column raises, eventually. A mismatched arity raises, from inside
scikit-learn. A **reordered** tuple of the right length raises nothing at all —
it feeds every column to the wrong tree splits and returns plausible, wrong
numbers. It is checked explicitly because nothing else will ever notice it.

The second theme is that a stale artifact is not a broken one. Six of the eight
models on disk were fitted before the catalogue grew and still predict
correctly, so "the active build supplies columns this model never saw" is a
warning. A gate on ordered equality would have refused them all on day one.
"""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xg_alonso.contracts.artifacts import (
    ARTIFACT_MANIFEST_VERSION,
    ArtifactManifest,
    ArtifactStatus,
    CompatibilityReason,
    Severity,
)
from xg_alonso.prediction.artifacts import (
    ActiveSchema,
    ArtifactCompatibilityError,
    check_compatibility,
    manifest_path_for,
    read_manifest,
    write_manifest,
)
from xg_alonso.prediction.dataset import TrainingData
from xg_alonso.prediction.trained import ComponentModels

NOW = datetime(2026, 7, 1, tzinfo=UTC)


def _manifest(names: tuple[str, ...], **overrides: object) -> ArtifactManifest:
    base: dict[str, object] = {
        "created_at": NOW,
        "feature_names": names,
        "feature_count": len(names),
        "feature_catalogue_version": "catalogue_v2",
        "feature_catalogue_hash": "a" * 64,
        "rules_snapshot_hash": "b" * 64,
    }
    return ArtifactManifest(**{**base, **overrides})  # type: ignore[arg-type]


def _active(names: tuple[str, ...], **overrides: object) -> ActiveSchema:
    base: dict[str, object] = {
        "feature_names": names,
        "catalogue_version": "catalogue_v2",
        "catalogue_hash": "a" * 64,
        "rules_snapshot_hash": "b" * 64,
    }
    return ActiveSchema(**{**base, **overrides})  # type: ignore[arg-type]


class TestTheManifestValidatesItself:
    def test_a_count_disagreeing_with_the_names_is_refused(self) -> None:
        with pytest.raises(ValueError, match="disagrees"):
            ArtifactManifest(created_at=NOW, feature_names=("a", "b"), feature_count=5)

    def test_a_duplicate_feature_name_is_refused(self) -> None:
        """A repeated column makes the ordered schema ambiguous."""
        with pytest.raises(ValueError, match="ambiguous"):
            ArtifactManifest(created_at=NOW, feature_names=("a", "a"), feature_count=2)

    def test_a_naive_timestamp_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            ArtifactManifest(created_at=datetime(2026, 7, 1))  # noqa: DTZ001

    def test_it_round_trips_through_json_preserving_order(self, tmp_path: Path) -> None:
        original = _manifest(("z", "a", "m"))
        path = tmp_path / "model.pkl"
        write_manifest(path, original)
        restored = read_manifest(path)

        assert restored is not None
        assert restored.feature_names == ("z", "a", "m")


class TestSatisfiabilityNotEquality:
    def test_a_subset_artifact_loads(self) -> None:
        """Six of the eight real artifacts are subsets and predict correctly."""
        verdict = check_compatibility(
            _manifest(("a", "b")),
            active=_active(("a", "b", "c", "d")),
            artifact_path=Path("m.pkl"),
        )
        assert verdict.compatible
        assert verdict.status is ArtifactStatus.COMPATIBLE

    def test_unexpected_features_warn_rather_than_block(self) -> None:
        verdict = check_compatibility(
            _manifest(("a",)),
            active=_active(("a", "b", "c")),
            artifact_path=Path("m.pkl"),
        )
        finding = next(
            f for f in verdict.findings if f.reason is CompatibilityReason.UNEXPECTED_FEATURES
        )
        assert finding.severity is Severity.WARNING
        assert verdict.compatible

    def test_a_missing_feature_blocks_and_names_it(self) -> None:
        verdict = check_compatibility(
            _manifest(("a", "gone")),
            active=_active(("a", "b")),
            artifact_path=Path("m.pkl"),
        )
        finding = next(
            f for f in verdict.findings if f.reason is CompatibilityReason.MISSING_FEATURES
        )
        assert finding.severity is Severity.BLOCKING
        assert "gone" in finding.detail
        assert "xg train" in finding.detail
        assert verdict.schema_diff is not None
        assert verdict.schema_diff.missing == ("gone",)

    def test_a_missing_feature_is_migratable_not_archival(self) -> None:
        """Its columns can be regenerated: a retrain, not a rewrite."""
        verdict = check_compatibility(
            _manifest(("gone",)), active=_active(("a",)), artifact_path=Path("m.pkl")
        )
        assert verdict.status is ArtifactStatus.MIGRATABLE

    def test_an_extension_feature_satisfies_the_need(self) -> None:
        verdict = check_compatibility(
            _manifest(("a", "discovered_x")),
            active=_active(("a",), extension_features=("discovered_x",)),
            artifact_path=Path("m.pkl"),
        )
        assert verdict.compatible


class TestTheReorderedSchema:
    def test_a_reordered_fitted_schema_blocks(self) -> None:
        """The one failure with no natural exception anywhere."""
        verdict = check_compatibility(
            _manifest(("a", "b", "c")),
            active=_active(("a", "b", "c")),
            artifact_path=Path("m.pkl"),
            artifact_feature_columns=("c", "b", "a"),
        )
        finding = next(
            f for f in verdict.findings if f.reason is CompatibilityReason.REORDERED_FEATURES
        )
        assert finding.severity is Severity.BLOCKING
        assert "plausible, wrong numbers" in finding.detail

    def test_a_matching_order_passes(self) -> None:
        verdict = check_compatibility(
            _manifest(("a", "b", "c")),
            active=_active(("a", "b", "c")),
            artifact_path=Path("m.pkl"),
            artifact_feature_columns=("a", "b", "c"),
        )
        assert verdict.compatible

    def test_a_count_mismatch_blocks_before_sklearn_sees_it(self) -> None:
        verdict = check_compatibility(
            _manifest(("a", "b")),
            active=_active(("a", "b", "c")),
            artifact_path=Path("m.pkl"),
            artifact_feature_columns=("a", "b", "c"),
        )
        finding = next(
            f for f in verdict.findings if f.reason is CompatibilityReason.FEATURE_COUNT_MISMATCH
        )
        assert finding.severity is Severity.BLOCKING

    def test_an_estimator_arity_mismatch_blocks(self) -> None:
        verdict = check_compatibility(
            _manifest(("a", "b")),
            active=_active(("a", "b")),
            artifact_path=Path("m.pkl"),
            estimator_arities={"label_minutes": 7},
        )
        assert any(
            f.reason is CompatibilityReason.ESTIMATOR_ARITY_MISMATCH for f in verdict.blocking
        )


class TestProvenanceChecks:
    def test_a_rules_change_blocks(self) -> None:
        """Components are priced with the *active* rules, not the fitted ones."""
        verdict = check_compatibility(
            _manifest(("a",), rules_snapshot_hash="c" * 64),
            active=_active(("a",)),
            artifact_path=Path("m.pkl"),
        )
        finding = next(
            f for f in verdict.findings if f.reason is CompatibilityReason.RULES_HASH_MISMATCH
        )
        assert finding.severity is Severity.BLOCKING

    def test_a_catalogue_change_warns_while_the_schema_is_satisfiable(self) -> None:
        verdict = check_compatibility(
            _manifest(("a",), feature_catalogue_hash="c" * 64),
            active=_active(("a",)),
            artifact_path=Path("m.pkl"),
        )
        finding = next(
            f for f in verdict.findings if f.reason is CompatibilityReason.CATALOGUE_HASH_MISMATCH
        )
        assert finding.severity is Severity.WARNING
        assert verdict.compatible

    def test_a_catalogue_change_blocks_when_the_schema_is_not_satisfiable(self) -> None:
        verdict = check_compatibility(
            _manifest(("gone",), feature_catalogue_hash="c" * 64),
            active=_active(("a",)),
            artifact_path=Path("m.pkl"),
        )
        assert any(
            f.reason is CompatibilityReason.CATALOGUE_HASH_MISMATCH
            and f.severity is Severity.BLOCKING
            for f in verdict.findings
        )

    def test_a_payload_digest_mismatch_blocks(self) -> None:
        """One of the two files was modified or half-written."""
        verdict = check_compatibility(
            _manifest(("a",), payload_sha256="d" * 64),
            active=_active(("a",)),
            artifact_path=Path("m.pkl"),
            payload_sha256="e" * 64,
        )
        assert any(
            f.reason is CompatibilityReason.PAYLOAD_DIGEST_MISMATCH for f in verdict.blocking
        )


class TestManifestAbsence:
    def test_no_manifest_is_unverified_not_broken(self) -> None:
        """Every artifact on disk predates manifests and still works."""
        verdict = check_compatibility(
            None,
            active=_active(("a", "b")),
            artifact_path=Path("m.pkl"),
            artifact_feature_columns=("a",),
        )
        assert verdict.status is ArtifactStatus.UNVERIFIED
        assert verdict.compatible
        assert any(f.reason is CompatibilityReason.MANIFEST_ABSENT for f in verdict.findings)

    def test_a_future_manifest_version_names_both_versions(self, tmp_path: Path) -> None:
        """`extra="forbid"` would otherwise surface this as a pydantic dump."""
        path = tmp_path / "m.pkl"
        manifest_path_for(path).write_text(
            json.dumps({"artifact_version": "artifact_manifest_v99", "created_at": NOW.isoformat()})
        )
        with pytest.raises(ArtifactCompatibilityError) as exc:
            read_manifest(path)

        message = str(exc.value)
        assert "artifact_manifest_v99" in message
        assert ARTIFACT_MANIFEST_VERSION in message


class TestTheErrorExplainsItself:
    def test_it_names_the_artifact_and_the_fix(self) -> None:
        verdict = check_compatibility(
            _manifest(("gone",)),
            active=_active(("a",)),
            artifact_path=Path("/models/holdout.pkl"),
        )
        text = ArtifactCompatibilityError(verdict).args[0]

        assert "holdout.pkl" in text
        assert "BLOCKING" in text
        assert "xg train" in text

    def test_blocking_findings_are_listed_first(self) -> None:
        verdict = check_compatibility(
            _manifest(("gone",)), active=_active(("a", "b")), artifact_path=Path("m.pkl")
        )
        lines = verdict.explain().splitlines()[1:]
        assert "BLOCKING" in lines[0]


class TestTheGateRunsBeforeUnpickling:
    def test_an_incompatible_manifest_refuses_without_opening_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole guarantee: a KeyError must never be the first failure."""
        from xg_alonso.prediction import inference

        path = tmp_path / "m.pkl"
        path.write_bytes(pickle.dumps({"not": "a SavedModel"}))
        write_manifest(path, _manifest(("gone",)))

        def _explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("pickle.load was called despite a blocking manifest")

        monkeypatch.setattr(inference, "load_quietly", _explode)

        with pytest.raises(ArtifactCompatibilityError, match="missing_features"):
            inference.load_models(path, active=_active(("a",)))

    def test_a_foreign_object_still_raises_type_error(self, tmp_path: Path) -> None:
        """Unpickling arbitrary objects and hoping is not a loading strategy."""
        from xg_alonso.prediction.inference import load_models

        path = tmp_path / "m.pkl"
        path.write_bytes(pickle.dumps({"not": "a SavedModel"}))

        with pytest.raises(TypeError, match="does not contain a SavedModel"):
            load_models(path)


class TestRoundTrip:
    """Train, save, reload, predict — and get the identical numbers back."""

    @staticmethod
    def _fitted() -> tuple[ComponentModels, TrainingData]:
        from conftest import FAST, synthetic_stats
        from xg_alonso.prediction.dataset import build_training_frame
        from xg_alonso.prediction.trained import train_component_models

        data = build_training_frame(synthetic_stats(players=14, gameweeks=16), min_gameweek=2)
        models = train_component_models(
            data.frame,
            feature_columns=data.feature_columns,
            label_columns=data.label_columns[:2],
            min_train_gameweeks=4,
            validate_gameweeks=2,
            model_kwargs=FAST,
        )
        return models, data

    def test_a_saved_model_reloads_and_predicts_identically(self, tmp_path: Path) -> None:
        """Exact, not approximate: both calls hit the same fitted trees.

        This is the regression that catches a reordered or silently truncated
        schema, which produces plausible numbers rather than an error.
        """
        import numpy as np

        from xg_alonso.prediction.inference import SavedModel, load_models, save_models

        models, data = self._fitted()
        before = models.predict(data.frame)

        path = tmp_path / "m.pkl"
        save_models(
            SavedModel(
                models=models,
                trained_seasons=data.seasons,
                trained_gameweeks=data.gameweeks,
                saved_at=NOW,
            ),
            path,
        )
        after = load_models(path).models.predict(data.frame)

        assert set(before) == set(after)
        for label in before:
            np.testing.assert_array_equal(before[label], after[label])

    def test_saving_with_a_manifest_writes_a_readable_sidecar(self, tmp_path: Path) -> None:
        """Provenance must be readable without executing the artifact."""
        from xg_alonso.prediction.inference import SavedModel, save_models

        models, data = self._fitted()
        path = tmp_path / "m.pkl"
        save_models(
            SavedModel(
                models=models,
                trained_seasons=data.seasons,
                trained_gameweeks=data.gameweeks,
                saved_at=NOW,
            ),
            path,
            manifest=_manifest(tuple(models.feature_columns)),
        )

        sidecar = manifest_path_for(path)
        assert sidecar.exists()
        restored = read_manifest(path)
        assert restored is not None
        assert restored.feature_names == tuple(models.feature_columns)
        # And it carries the digest of the bytes actually written.
        assert restored.payload_sha256
        assert restored.payload_bytes == path.stat().st_size

    def test_a_tampered_artifact_is_refused(self, tmp_path: Path) -> None:
        from xg_alonso.prediction.inference import SavedModel, load_models, save_models

        models, data = self._fitted()
        path = tmp_path / "m.pkl"
        save_models(
            SavedModel(
                models=models,
                trained_seasons=data.seasons,
                trained_gameweeks=data.gameweeks,
                saved_at=NOW,
            ),
            path,
            manifest=_manifest(tuple(models.feature_columns)),
        )
        path.write_bytes(path.read_bytes() + b"\x00")

        active = _active(tuple(models.feature_columns))
        with pytest.raises(ArtifactCompatibilityError, match="payload_digest_mismatch"):
            load_models(path, active=active)
