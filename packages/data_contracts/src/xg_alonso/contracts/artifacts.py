"""What a saved model must be able to say about itself.

**The failure this prevents does not raise.** ``trained.py::_matrix`` selects
the artifact's own column tuple out of whatever frame it is handed, so three
different mismatches produce three very different symptoms:

- a *missing* column raises ``KeyError`` from inside a prediction loop, far
  from the cause;
- an *extra* column in the frame is harmless, which is why most artifacts here
  still work despite being years of features out of date;
- a *reordered* tuple of the right length raises nothing at all. It feeds every
  column to the wrong tree splits and returns plausible, wrong numbers.

Only the third is genuinely dangerous, and it is the only one with no natural
exception. That is the reason this module exists.

**Compatibility is satisfiability, not equality.** Measured against the eight
artifacts on disk: seven need only columns the active build supplies, and one
needs five it does not. A gate built on ordered equality would classify six
working models as incompatible on the day it shipped, because they were fitted
before the catalogue grew. The question is *can the active build supply, by
name, every column this artifact was fitted on* — with catalogue-hash equality
reported alongside as staleness rather than as a refusal.

**Contracts carries provenance, it never computes it.** The catalogue hash
comes from ``features``, the rules hash from ``domain``, the training-data hash
from ``storage`` — all of which sit above this layer. Each is an opaque string
here, the same arrangement ``contracts/discovery.py::ExperimentManifest``
already uses, and the assembly happens in ``cli`` where every source is in
scope.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ARTIFACT_MANIFEST_VERSION",
    "ArtifactCompatibility",
    "ArtifactIncompatibility",
    "ArtifactIndexEntry",
    "ArtifactManifest",
    "ArtifactStatus",
    "CompatibilityReason",
    "ComponentMetrics",
    "EstimatorConfig",
    "FeatureSchemaDiff",
    "RuntimeVersions",
    "Severity",
]

ARTIFACT_MANIFEST_VERSION: Final[str] = "artifact_manifest_v1"


class ArtifactStatus(StrEnum):
    COMPATIBLE = "compatible"
    MIGRATABLE = "incompatible_but_migratable"
    """Its missing features can be regenerated — a retrain, not a rewrite."""
    ARCHIVAL = "incompatible_and_archival"
    """Nothing can produce what it needs. Kept for provenance, never loaded."""
    CORRUPT = "corrupt_or_incomplete"
    UNVERIFIED = "unverified"
    """No manifest. Schema checks ran; provenance checks could not."""


class CompatibilityReason(StrEnum):
    MANIFEST_ABSENT = "manifest_absent"
    MANIFEST_UNREADABLE = "manifest_unreadable"
    ARTIFACT_VERSION_UNSUPPORTED = "artifact_version_unsupported"
    MISSING_FEATURES = "missing_features"
    UNEXPECTED_FEATURES = "unexpected_features"
    REORDERED_FEATURES = "reordered_features"
    FEATURE_COUNT_MISMATCH = "feature_count_mismatch"
    ESTIMATOR_ARITY_MISMATCH = "estimator_arity_mismatch"
    CATALOGUE_VERSION_MISMATCH = "catalogue_version_mismatch"
    CATALOGUE_HASH_MISMATCH = "catalogue_hash_mismatch"
    RULES_HASH_MISMATCH = "rules_hash_mismatch"
    PAYLOAD_DIGEST_MISMATCH = "payload_digest_mismatch"
    RUNTIME_VERSION_DRIFT = "runtime_version_drift"


class Severity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RuntimeVersions(_Frozen):
    """What the artifact was pickled under. Drift here is a warning, not a bar."""

    python: str = ""
    numpy: str = ""
    polars: str = ""
    scikit_learn: str = ""
    pydantic: str = ""


class EstimatorConfig(_Frozen):
    """One label's estimator as configured — not its fitted trees."""

    label: str
    estimator_class: str
    loss: str = ""
    random_state: int | None = None
    hyperparameters: tuple[tuple[str, str], ...] = ()


class ComponentMetrics(_Frozen):
    """Per-label out-of-sample summary, kept structurally rather than printed."""

    label: str
    mean_skill: float
    mean_bias: float
    degenerate: bool
    label_mean: float
    folds: int = Field(ge=0)


class ArtifactManifest(_Frozen):
    """Everything needed to decide whether a saved model may be used."""

    artifact_version: str = ARTIFACT_MANIFEST_VERSION
    artifact_format: Literal["pickle"] = "pickle"
    model_type: str = ""
    model_name: str = ""
    model_version: str = ""
    objective_id: str = ""
    """Non-empty means it needs discovered features, not just the catalogue."""
    created_at: datetime

    git_commit: str = ""
    git_dirty: bool = False
    code_version: str = ""
    runtime: RuntimeVersions = RuntimeVersions()

    feature_catalogue_version: str = ""
    feature_catalogue_hash: str = ""
    feature_names: tuple[str, ...] = ()
    """Ordered, exactly as fitted. The order is the dangerous part."""
    feature_count: int = Field(default=0, ge=0)
    dropped_features: tuple[str, ...] = ()
    extension_features: tuple[str, ...] = ()

    rules_snapshot_hash: str = ""
    scoring_rules_version: str = ""
    training_data_manifest_hash: str = ""
    training_seasons: tuple[str, ...] = ()
    training_gameweeks: tuple[int, ...] = ()
    training_rows: int = Field(default=0, ge=0)
    training_start_time: datetime | None = None
    training_end_time: datetime | None = None
    training_cutoff: datetime | None = None

    estimators: tuple[EstimatorConfig, ...] = ()
    component_metrics: tuple[ComponentMetrics, ...] = ()
    label_columns: tuple[str, ...] = ()

    model_fingerprint: str = ""
    payload_sha256: str = ""
    payload_bytes: int = Field(default=0, ge=0)

    @field_validator("created_at", "training_start_time", "training_end_time", "training_cutoff")
    @classmethod
    def _tz_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _schema_is_self_consistent(self) -> ArtifactManifest:
        if self.feature_count != len(self.feature_names):
            raise ValueError(
                f"feature_count {self.feature_count} disagrees with "
                f"{len(self.feature_names)} feature names"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(
                "duplicate feature names: a repeated column makes the ordered "
                "schema ambiguous, and ordering is the whole point of recording it"
            )
        return self

    @property
    def reproducible(self) -> bool:
        return bool(self.code_version) and not self.git_dirty


class FeatureSchemaDiff(_Frozen):
    """The ordered comparison, reported in full rather than as a boolean."""

    expected_order: tuple[str, ...] = ()
    """What the artifact will select, in the order it will select it."""
    active_order: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    reordered: tuple[str, ...] = ()

    @property
    def satisfiable(self) -> bool:
        """Whether the active build can supply everything the artifact needs."""
        return not self.missing


class ArtifactIncompatibility(_Frozen):
    reason: CompatibilityReason
    severity: Severity
    detail: str
    features: tuple[str, ...] = ()


class ArtifactCompatibility(_Frozen):
    """The verdict on one artifact, with everything behind it."""

    artifact_path: Path
    status: ArtifactStatus
    checked_at: datetime
    manifest: ArtifactManifest | None = None
    schema_diff: FeatureSchemaDiff | None = None
    findings: tuple[ArtifactIncompatibility, ...] = ()

    @property
    def blocking(self) -> tuple[ArtifactIncompatibility, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.BLOCKING)

    @property
    def compatible(self) -> bool:
        return not self.blocking

    def explain(self) -> str:
        """Human-first, blocking findings first, always naming the artifact."""
        lines = [f"{self.artifact_path.name}: {self.status}"]
        for finding in sorted(self.findings, key=lambda f: f.severity is not Severity.BLOCKING):
            mark = "BLOCKING" if finding.severity is Severity.BLOCKING else "warning"
            lines.append(f"  [{mark}] {finding.reason}: {finding.detail}")
        return "\n".join(lines)


class ArtifactIndexEntry(_Frozen):
    """One append-only event about one artifact.

    Status changes are appended, never edited, so "no deletion, provenance
    preserved" is a property of the format rather than a rule somebody has to
    remember. Same idiom as ``storage/bronze.py``'s manifest.
    """

    artifact_id: str
    """``payload_sha256[:16]`` — stable across moves and renames."""
    path: Path
    previous_path: Path | None = None
    model_name: str = ""
    objective_id: str = ""
    status: ArtifactStatus
    reasons: tuple[CompatibilityReason, ...] = ()
    payload_sha256: str = ""
    feature_count: int = Field(default=0, ge=0)
    feature_catalogue_version: str = ""
    recorded_at: datetime
    note: str = ""

    @field_validator("recorded_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value
