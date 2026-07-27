"""Timestamps and provenance — the machinery that makes results reproducible.

The four-timestamp rule is the foundation of point-in-time correctness. Every
source record carries all four, and feature joins filter on ``available_time``:

- ``event_time``     — when the underlying real-world event happened
- ``observed_time``  — when we collected it
- ``available_time`` — when the information first became publicly usable
- ``processed_time`` — when our pipeline transformed it

A feature may use a record only when ``available_time <= prediction_timestamp``.

**`available_time` is a lower bound for HTTP sources.** We derive it from the
response `Date` header minus `Age`, which tells us the origin published *at or
before* that instant. Deriving it that way admits data slightly earlier than
reality, so it is a lower bound, not an observation — record it, and treat the
gap as leakage exposure rather than as conservatism.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "FOUR_TIMESTAMP_FIELDS",
    "PredictionProvenance",
    "RunManifest",
    "SourceTimestamps",
    "TimeSource",
    "utc_now",
]


def utc_now() -> datetime:
    """Timezone-aware UTC now.

    Use this rather than ``datetime.utcnow()``, which returns a naive datetime
    and silently compares wrong against tz-aware timestamps. Ruff's ``DTZ003``
    enforces this repo-wide.
    """
    return datetime.now(UTC)


FOUR_TIMESTAMP_FIELDS: tuple[str, ...] = (
    "event_time",
    "observed_time",
    "available_time",
    "processed_time",
)


class TimeSource(StrEnum):
    """How ``available_time`` was determined, so its reliability is auditable."""

    HTTP_DATE_MINUS_AGE = "http_date_minus_age"
    """Derived from response headers. A lower bound on true availability."""

    DEADLINE = "deadline"
    """Pinned to an official gameweek deadline. Exact."""

    FIXTURE_KICKOFF = "fixture_kickoff"
    """Pinned to a fixture kickoff time. Exact."""

    FIXTURE_FINALIZATION = "fixture_finalization"
    """Available once a fixture's results are final."""

    ARCHIVE_DECLARED = "archive_declared"
    """Taken from a historical archive's own declaration. Trust with care."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceTimestamps(_Frozen):
    """The four timestamps every ingested record must carry."""

    event_time: datetime
    observed_time: datetime
    available_time: datetime
    processed_time: datetime
    time_source: TimeSource

    @field_validator("event_time", "observed_time", "available_time", "processed_time")
    @classmethod
    def _require_tz_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware; naive datetimes are rejected")
        return value.astimezone(UTC)

    def is_available_at(self, cutoff: datetime) -> bool:
        """Whether this record may be used for a prediction made at ``cutoff``."""
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        return self.available_time <= cutoff


class RunManifest(_Frozen):
    """Identifies one execution of a pipeline, so its outputs can be reproduced.

    A run with uncommitted changes (``git_dirty``) may never promote an artifact
    to production — its outputs cannot be reproduced from the recorded commit.
    """

    run_id: str = Field(description="Unique id for this execution")
    pipeline: str = Field(description="Which pipeline ran, e.g. 'ingest.bootstrap'")
    started_at: datetime
    git_commit: str = Field(description="Full commit SHA of the running code")
    git_dirty: bool = Field(description="True when the working tree had uncommitted changes")
    config_hash: str = Field(description="Hash of the resolved configuration")

    @field_validator("started_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        return value.astimezone(UTC)

    @property
    def promotable(self) -> bool:
        """Whether artifacts from this run may be promoted to production."""
        return not self.git_dirty


class PredictionProvenance(_Frozen):
    """Everything needed to reproduce a single prediction.

    Every field is required. A closed-form baseline has no trained artifact on
    disk, so it supplies ``model_artifact_sha256`` as the hash of its estimator
    source module — the field stays non-null and the "zero nulls" acceptance
    criterion stays satisfiable.
    """

    model_name: str
    model_version: str
    model_artifact_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="SHA-256 of the model artifact, or of the estimator source for closed-form models",
    )
    feature_set_name: str
    feature_set_version: str
    embedding_version: str | None = Field(
        default=None, description="Set only when embeddings contributed"
    )
    data_cutoff: datetime = Field(
        description="No input to this prediction had available_time after this instant"
    )
    predicted_at: datetime
    run_id: str
    code_version: str = Field(description="Commit SHA of the code that produced this")

    @field_validator("data_cutoff", "predicted_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("provenance timestamps must be timezone-aware")
        return value.astimezone(UTC)
