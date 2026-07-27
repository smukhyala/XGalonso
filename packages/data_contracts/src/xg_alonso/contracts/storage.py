"""Storage protocols — the single definition of how anything reaches persistence.

During planning, four packages independently implemented their own storage and
each imported ``duckdb`` directly. That would have failed the import boundary on
day one and left four places to change when the backend changes.

These protocols are the seam. ``xg_alonso.storage`` provides the only
implementation and is the only package permitted to import a database driver
(enforced by ``.importlinter``). Decision D2 fixes the current backend as DuckDB
plus Parquet; keeping it behind these protocols is what makes that choice
reversible rather than load-bearing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from xg_alonso.contracts.provenance import SourceTimestamps

__all__ = [
    "BronzeSnapshotStore",
    "SnapshotRef",
    "TableStore",
]


class SnapshotRef(BaseModel):
    """A pointer to one immutable bronze snapshot.

    Bronze is append-only. A snapshot is identified by its content hash, so
    re-ingesting identical bytes is a no-op rather than a duplicate — which is
    what makes ingestion safely resumable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(description="Logical source name, e.g. 'fpl.bootstrap_static'")
    path: Path = Field(description="Location of the stored payload")
    content_sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    timestamps: SourceTimestamps
    run_id: str
    http_status: int | None = Field(default=None, description="Set for HTTP sources")


@runtime_checkable
class BronzeSnapshotStore(Protocol):
    """Immutable, append-only raw payload storage.

    There is deliberately no update or delete. Raw data is never overwritten;
    correcting a bad ingest means writing a new snapshot, not mutating an old
    one, so any past feature build stays reproducible.
    """

    def write(
        self,
        *,
        source: str,
        payload: bytes,
        timestamps: SourceTimestamps,
        run_id: str,
        http_status: int | None = None,
    ) -> SnapshotRef:
        """Persist a raw payload and return its reference.

        Returns the existing reference unchanged if identical bytes were already
        stored for this source, making ingestion idempotent.
        """
        ...

    def read(self, ref: SnapshotRef) -> bytes:
        """Read back a stored payload."""
        ...

    def latest(self, source: str, *, as_of: datetime | None = None) -> SnapshotRef | None:
        """Most recent snapshot for a source, optionally as of a point in time.

        Passing ``as_of`` is how a historical rebuild sees only what was
        available then.
        """
        ...

    def list_snapshots(self, source: str) -> list[SnapshotRef]:
        """Every snapshot for a source, oldest first."""
        ...


@runtime_checkable
class TableStore(Protocol):
    """Canonical (silver) and model-ready (gold) tabular storage."""

    def write_table(self, name: str, frame: pl.DataFrame, *, run_id: str) -> None:
        """Replace a table's contents. Silver and gold are derived and rebuildable."""
        ...

    def append_table(self, name: str, frame: pl.DataFrame, *, run_id: str) -> None:
        """Append rows to a table, creating it when absent."""
        ...

    def read_table(self, name: str) -> pl.DataFrame:
        """Read a whole table."""
        ...

    def query(self, sql: str, **params: object) -> pl.DataFrame:
        """Run a parameterised read-only query.

        Parameters are bound by the adapter, never interpolated into the string.
        """
        ...

    def table_exists(self, name: str) -> bool: ...

    def execute(self, sql: str) -> None:
        """Run DDL. Used by migrations only."""
        ...
