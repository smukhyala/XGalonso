"""Storage adapter tests.

The properties under test are the ones the rest of the system relies on:
bronze is immutable and idempotent, and the table store cannot be talked into
executing an injected identifier.
"""

from __future__ import annotations

import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from xg_alonso.contracts import (
    BronzeSnapshotStore,
    SourceTimestamps,
    TableStore,
    TimeSource,
)
from xg_alonso.storage import FileSystemBronzeStore
from xg_alonso.storage.duckdb_store import DuckDBTableStore
from xg_alonso.storage.parquet_store import ParquetTableStore

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _timestamps(offset_seconds: int = 0) -> SourceTimestamps:
    moment = NOW + timedelta(seconds=offset_seconds)
    return SourceTimestamps(
        event_time=moment,
        observed_time=moment,
        available_time=moment,
        processed_time=moment,
        time_source=TimeSource.HTTP_DATE_MINUS_AGE,
    )


class TestBronzeStore:
    def test_satisfies_the_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FileSystemBronzeStore(tmp_path), BronzeSnapshotStore)

    def test_round_trip(self, tmp_path: Path) -> None:
        store = FileSystemBronzeStore(tmp_path)
        payload = b'{"hello": "world"}'
        ref = store.write(
            source="fpl.bootstrap_static",
            payload=payload,
            timestamps=_timestamps(),
            run_id="r1",
            http_status=200,
        )
        assert store.read(ref) == payload
        assert ref.byte_size == len(payload)

    def test_identical_bytes_are_idempotent(self, tmp_path: Path) -> None:
        """Re-ingesting the same payload must not create a duplicate.

        This is what makes a half-finished ingest safe to simply run again.
        """
        store = FileSystemBronzeStore(tmp_path)
        payload = b'{"a": 1}'

        first = store.write(source="s", payload=payload, timestamps=_timestamps(), run_id="r1")
        second = store.write(source="s", payload=payload, timestamps=_timestamps(60), run_id="r2")

        assert first.content_sha256 == second.content_sha256
        assert first.path == second.path
        assert len(store.list_snapshots("s")) == 1, "identical bytes must not duplicate"

    def test_different_bytes_create_separate_snapshots(self, tmp_path: Path) -> None:
        store = FileSystemBronzeStore(tmp_path)
        store.write(source="s", payload=b'{"a": 1}', timestamps=_timestamps(), run_id="r1")
        store.write(source="s", payload=b'{"a": 2}', timestamps=_timestamps(60), run_id="r2")
        assert len(store.list_snapshots("s")) == 2

    def test_tampering_is_detected_on_read(self, tmp_path: Path) -> None:
        """Raw data must never change after write, so a changed file must fail loudly."""
        store = FileSystemBronzeStore(tmp_path)
        ref = store.write(
            source="s", payload=b'{"real": true}', timestamps=_timestamps(), run_id="r1"
        )

        with (
            ref.path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh,
        ):
            fh.write(b'{"real": false}')

        with pytest.raises(ValueError, match="must never change after write"):
            store.read(ref)

    def test_latest_respects_as_of(self, tmp_path: Path) -> None:
        """A historical rebuild must see only what existed at that instant."""
        store = FileSystemBronzeStore(tmp_path)
        store.write(source="s", payload=b"1", timestamps=_timestamps(0), run_id="r1")
        store.write(source="s", payload=b"2", timestamps=_timestamps(3600), run_id="r2")

        latest = store.latest("s")
        assert latest is not None
        assert store.read(latest) == b"2"

        earlier = store.latest("s", as_of=NOW + timedelta(seconds=60))
        assert earlier is not None
        assert store.read(earlier) == b"1", "as_of must exclude later snapshots"

    def test_latest_returns_none_for_unknown_source(self, tmp_path: Path) -> None:
        assert FileSystemBronzeStore(tmp_path).latest("never-ingested") is None

    def test_naive_as_of_rejected(self, tmp_path: Path) -> None:
        store = FileSystemBronzeStore(tmp_path)
        with pytest.raises(ValueError, match="timezone-aware"):
            store.latest("s", as_of=datetime(2026, 7, 27, 12, 0))  # noqa: DTZ001

    def test_compression_is_deterministic(self, tmp_path: Path) -> None:
        """Byte-identical output for identical input keeps rerun-equality honest."""
        a = FileSystemBronzeStore(tmp_path / "a")
        b = FileSystemBronzeStore(tmp_path / "b")
        payload = b'{"deterministic": true}'
        ref_a = a.write(source="s", payload=payload, timestamps=_timestamps(), run_id="r1")
        ref_b = b.write(source="s", payload=payload, timestamps=_timestamps(), run_id="r2")
        assert ref_a.path.read_bytes() == ref_b.path.read_bytes()


class TestDuckDBTableStore:
    def test_satisfies_the_protocol(self) -> None:
        with DuckDBTableStore() as store:
            assert isinstance(store, TableStore)

    def test_write_and_read(self) -> None:
        frame = pl.DataFrame({"player_code": [1, 2], "points": [5.0, 7.5]})
        with DuckDBTableStore() as store:
            store.write_table("players", frame, run_id="r1")
            assert store.read_table("players").sort("player_code").equals(frame)

    def test_write_table_replaces(self) -> None:
        with DuckDBTableStore() as store:
            store.write_table("t", pl.DataFrame({"a": [1, 2, 3]}), run_id="r1")
            store.write_table("t", pl.DataFrame({"a": [9]}), run_id="r2")
            assert store.read_table("t").height == 1

    def test_append_accumulates_and_creates(self) -> None:
        with DuckDBTableStore() as store:
            store.append_table("t", pl.DataFrame({"a": [1]}), run_id="r1")
            store.append_table("t", pl.DataFrame({"a": [2]}), run_id="r2")
            assert sorted(store.read_table("t")["a"].to_list()) == [1, 2]

    def test_missing_table_raises_keyerror(self) -> None:
        with DuckDBTableStore() as store, pytest.raises(KeyError, match="does not exist"):
            store.read_table("nope")

    @pytest.mark.parametrize(
        "bad",
        ["players; DROP TABLE players", "players--", "a b", "", "1players", "tbl'"],
    )
    def test_injected_identifiers_rejected(self, bad: str) -> None:
        """Table names cannot be bound as parameters, so they must be validated."""
        with DuckDBTableStore() as store, pytest.raises(ValueError, match="table name"):
            store.table_exists(bad)

    def test_query_binds_parameters(self) -> None:
        with DuckDBTableStore() as store:
            store.write_table(
                "t", pl.DataFrame({"code": [1, 2, 3], "v": [10, 20, 30]}), run_id="r1"
            )
            result = store.query("SELECT v FROM t WHERE code = $code", code=2)
            assert result["v"].to_list() == [20]

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "xg.duckdb"
        with DuckDBTableStore(db) as store:
            store.write_table("t", pl.DataFrame({"a": [1]}), run_id="r1")
        with DuckDBTableStore(db) as reopened:
            assert reopened.read_table("t")["a"].to_list() == [1]


class TestParquetTableStore:
    """The driver-free TableStore the composition root actually uses.

    ``.importlinter`` forbids ``xg_alonso.cli`` and ``xg_alonso.api`` from
    reaching ``duckdb``, so without this the ``TableStore`` protocol would be
    usable only from inside the storage package.
    """

    def test_write_then_read_round_trips(self, tmp_path: Path) -> None:
        store = ParquetTableStore(tmp_path)
        frame = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        store.write_table("things", frame, run_id="r1")
        assert store.read_table("things").equals(frame)

    def test_write_replaces_rather_than_appends(self, tmp_path: Path) -> None:
        store = ParquetTableStore(tmp_path)
        store.write_table("things", pl.DataFrame({"a": [1, 2]}), run_id="r1")
        store.write_table("things", pl.DataFrame({"a": [9]}), run_id="r2")
        assert store.read_table("things")["a"].to_list() == [9]

    def test_append_creates_then_accumulates(self, tmp_path: Path) -> None:
        store = ParquetTableStore(tmp_path)
        store.append_table("log", pl.DataFrame({"a": [1]}), run_id="r1")
        store.append_table("log", pl.DataFrame({"a": [2]}), run_id="r2")
        assert store.read_table("log")["a"].to_list() == [1, 2]

    def test_append_tolerates_a_schema_that_gained_a_column(self, tmp_path: Path) -> None:
        """A history row genuinely lacks a field added later; that is not an error."""
        store = ParquetTableStore(tmp_path)
        store.append_table("log", pl.DataFrame({"a": [1]}), run_id="r1")
        store.append_table("log", pl.DataFrame({"a": [2], "b": ["new"]}), run_id="r2")
        out = store.read_table("log")
        assert out.height == 2
        assert "b" in out.columns

    def test_reading_a_missing_table_raises(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="does not exist"):
            ParquetTableStore(tmp_path).read_table("absent")

    def test_a_path_traversing_name_is_rejected(self, tmp_path: Path) -> None:
        """A table name becomes a filename, so escaping the directory is refused."""
        store = ParquetTableStore(tmp_path)
        for name in ("../escape", "a/b", "with space", ""):
            with pytest.raises(ValueError, match="table name"):
                store.table_exists(name)

    def test_sql_is_refused_rather_than_faked(self, tmp_path: Path) -> None:
        store = ParquetTableStore(tmp_path)
        with pytest.raises(NotImplementedError, match="what a driver is for"):
            store.query("SELECT 1")
        with pytest.raises(NotImplementedError, match="no DDL"):
            store.execute("CREATE TABLE t (a INT)")

    def test_it_satisfies_the_table_store_protocol(self, tmp_path: Path) -> None:
        assert isinstance(ParquetTableStore(tmp_path), TableStore)

    def test_tables_are_listed(self, tmp_path: Path) -> None:
        store = ParquetTableStore(tmp_path)
        store.write_table("beta", pl.DataFrame({"a": [1]}), run_id="r")
        store.write_table("alpha", pl.DataFrame({"a": [1]}), run_id="r")
        assert store.list_tables() == ["alpha", "beta"]
