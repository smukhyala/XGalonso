"""DuckDB-backed table storage for the canonical (silver) and model-ready (gold) layers.

This module is the **only** place in the codebase that imports ``duckdb``,
enforced by the ``duckdb-isolation`` contract in ``.importlinter``. During
planning, four separate packages each reached for the driver directly; the gate
exists so that decision D2 stays reversible instead of becoming load-bearing.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import duckdb
import polars as pl

__all__ = ["DuckDBTableStore"]

# DuckDB accepts these unquoted; anything else must be rejected rather than
# interpolated, since table names cannot be passed as bind parameters.
_SAFE_IDENTIFIER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _validate_identifier(name: str) -> str:
    """Reject anything that is not a plain identifier.

    Table names cannot be bound as parameters, so they are concatenated into
    SQL. Validating them here is what keeps that concatenation safe.
    """
    if not name:
        raise ValueError("table name must not be empty")
    if name[0].isdigit():
        raise ValueError(f"table name must not start with a digit: {name!r}")
    illegal = set(name) - _SAFE_IDENTIFIER
    if illegal:
        raise ValueError(
            f"table name {name!r} contains disallowed characters {sorted(illegal)}; "
            "only letters, digits and underscores are permitted"
        )
    return name


class DuckDBTableStore:
    """DuckDB implementation of :class:`~xg_alonso.contracts.storage.TableStore`."""

    def __init__(self, database: Path | str = ":memory:") -> None:
        self._database = str(database)
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(self._database)

    def __enter__(self) -> DuckDBTableStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- TableStore -------------------------------------------------------

    def write_table(self, name: str, frame: pl.DataFrame, *, run_id: str) -> None:
        """Replace a table's contents. Silver and gold are derived and rebuildable."""
        table = _validate_identifier(name)
        self._conn.register("_incoming", frame)
        try:
            self._conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _incoming")
        finally:
            self._conn.unregister("_incoming")

    def append_table(self, name: str, frame: pl.DataFrame, *, run_id: str) -> None:
        """Append rows, creating the table when it does not yet exist."""
        table = _validate_identifier(name)
        self._conn.register("_incoming", frame)
        try:
            if self.table_exists(table):
                self._conn.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
            else:
                self._conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _incoming")
        finally:
            self._conn.unregister("_incoming")

    def read_table(self, name: str) -> pl.DataFrame:
        table = _validate_identifier(name)
        if not self.table_exists(table):
            raise KeyError(f"table {table!r} does not exist")
        return self._conn.execute(f"SELECT * FROM {table}").pl()

    def query(self, sql: str, **params: object) -> pl.DataFrame:
        """Run a parameterised read-only query.

        Parameters are bound by DuckDB using ``$name`` placeholders — never
        interpolated into the SQL string.
        """
        if params:
            return self._conn.execute(sql, params).pl()
        return self._conn.execute(sql).pl()

    def table_exists(self, name: str) -> bool:
        table = _validate_identifier(name)
        result = self._conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        return bool(result and result[0] > 0)

    def execute(self, sql: str) -> None:
        """Run DDL. Used by migrations only."""
        self._conn.execute(sql)

    # -- convenience ------------------------------------------------------

    def list_tables(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [str(r[0]) for r in rows]
