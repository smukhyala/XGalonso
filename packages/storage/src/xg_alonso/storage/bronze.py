"""Immutable bronze snapshot storage on the local filesystem.

Bronze is append-only. There is no update and no delete, because
``CLAUDE.md`` is unambiguous: raw data is never overwritten. Correcting a bad
ingest means writing a *new* snapshot, which keeps every past feature build
reproducible from the bytes it actually saw.

Layout::

    .data/bronze/<source>/_manifest.jsonl        append-only index
    .data/bronze/<source>/<YYYY-MM-DD>/<sha>.json.gz

The manifest is a plain JSONL sidecar rather than a database table on purpose:
bronze must be readable and auditable without booting any query engine, and a
corrupted database must never cost us the raw record of what an API returned.

Snapshots are keyed by content hash, so re-ingesting identical bytes is a no-op
rather than a duplicate. That is what makes ingestion safely resumable — a
half-finished run can simply be run again.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

from xg_alonso.contracts.provenance import SourceTimestamps
from xg_alonso.contracts.storage import SnapshotRef

__all__ = ["FileSystemBronzeStore"]

_MANIFEST_NAME = "_manifest.jsonl"


class FileSystemBronzeStore:
    """Filesystem-backed :class:`~xg_alonso.contracts.storage.BronzeSnapshotStore`."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def _source_dir(self, source: str) -> Path:
        # Source names are dotted (``fpl.bootstrap_static``); keep them flat so a
        # directory listing shows every source at a glance.
        return self._root / source

    def _manifest_path(self, source: str) -> Path:
        return self._source_dir(source) / _MANIFEST_NAME

    def write(
        self,
        *,
        source: str,
        payload: bytes,
        timestamps: SourceTimestamps,
        run_id: str,
        http_status: int | None = None,
    ) -> SnapshotRef:
        """Persist a raw payload, or return the existing reference if unchanged."""
        digest = hashlib.sha256(payload).hexdigest()

        existing = self._find_by_hash(source, digest)
        if existing is not None:
            return existing

        day = timestamps.observed_time.strftime("%Y-%m-%d")
        target_dir = self._source_dir(source) / day
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{digest}.json.gz"

        # mtime=0 keeps the gzip container byte-identical for identical input,
        # so a rerun-equality check compares payloads rather than clock noise.
        # GzipFile does not close a fileobj handed to it, so the raw handle owns
        # its own context — otherwise the descriptor leaks on every write.
        tmp = path.with_suffix(".gz.tmp")
        with (
            tmp.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle,
        ):
            handle.write(payload)
        tmp.replace(path)

        ref = SnapshotRef(
            source=source,
            path=path,
            content_sha256=digest,
            byte_size=len(payload),
            timestamps=timestamps,
            run_id=run_id,
            http_status=http_status,
        )
        self._append_manifest(source, ref)
        return ref

    def read(self, ref: SnapshotRef) -> bytes:
        """Read back a stored payload, verifying it has not been tampered with."""
        with gzip.open(ref.path, "rb") as handle:
            payload: bytes = handle.read()

        actual = hashlib.sha256(payload).hexdigest()
        if actual != ref.content_sha256:
            raise ValueError(
                f"bronze snapshot {ref.path} has hash {actual} but its manifest "
                f"records {ref.content_sha256}; raw data must never change after write"
            )
        return payload

    def latest(self, source: str, *, as_of: datetime | None = None) -> SnapshotRef | None:
        """Most recent snapshot, optionally restricted to what was available then.

        Passing ``as_of`` is how a historical rebuild sees only the data that
        existed at that instant, which is what makes backfills honest.
        """
        candidates = self.list_snapshots(source)
        if as_of is not None:
            if as_of.tzinfo is None:
                raise ValueError("as_of must be timezone-aware")
            candidates = [c for c in candidates if c.timestamps.available_time <= as_of]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.timestamps.available_time)

    def list_snapshots(self, source: str) -> list[SnapshotRef]:
        """Every snapshot for a source, oldest observation first."""
        manifest = self._manifest_path(source)
        if not manifest.exists():
            return []

        refs: list[SnapshotRef] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    refs.append(SnapshotRef.model_validate_json(stripped))
        refs.sort(key=lambda r: (r.timestamps.observed_time, r.content_sha256))
        return refs

    def _find_by_hash(self, source: str, digest: str) -> SnapshotRef | None:
        return next(
            (r for r in self.list_snapshots(source) if r.content_sha256 == digest),
            None,
        )

    def _append_manifest(self, source: str, ref: SnapshotRef) -> None:
        manifest = self._manifest_path(source)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(json.loads(ref.model_dump_json()), sort_keys=True, separators=(",", ":"))
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
