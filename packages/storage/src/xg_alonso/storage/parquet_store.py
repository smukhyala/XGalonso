"""Parquet-backed table storage — the driver-free implementation of ``TableStore``.

**Why this exists alongside the DuckDB one.** ``.importlinter`` forbids
``xg_alonso.cli`` and ``xg_alonso.api`` from reaching ``duckdb``, transitively
included, so that decision D2 stays reversible rather than load-bearing. That
boundary is correct and worth keeping — but it also means the composition root
cannot construct a :class:`~xg_alonso.storage.duckdb_store.DuckDBTableStore`, and
a registry only usable from inside the storage package is not a registry.

D2 names "DuckDB **and Parquet**". This is the Parquet half: one file per table,
the same :class:`~xg_alonso.contracts.storage.TableStore` protocol, and no
database driver anywhere in the import graph. An app can therefore persist
without a driver, and anything wanting SQL still reaches for DuckDB explicitly.

The trade is deliberate. ``query`` is not supported here — SQL is exactly what a
driver is for — and every consumer in the discovery registry reads whole tables
and filters in Polars, so nothing loses a capability it was using.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import polars as pl

__all__ = ["ParquetTableStore"]

#: Characters permitted in a table name. A table name becomes a filename, so
#: anything that could escape the directory is rejected rather than sanitised —
#: the same reasoning ``DuckDBTableStore`` applies to SQL identifiers.
_SAFE_IDENTIFIER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _validate_identifier(name: str) -> str:
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


class ParquetTableStore:
    """One Parquet file per table, under a root directory.

    Implements :class:`~xg_alonso.contracts.storage.TableStore` except for
    :meth:`query`, which raises. See the module docstring.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> ParquetTableStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """No handle to release. Present so callers can treat both stores alike."""

    def _path(self, name: str) -> Path:
        return self._root / f"{_validate_identifier(name)}.parquet"

    # -- TableStore -------------------------------------------------------

    def write_table(self, name: str, frame: pl.DataFrame, *, run_id: str) -> None:
        """Replace a table's contents. Silver and gold are derived and rebuildable."""
        del run_id
        frame.write_parquet(self._path(name))

    def append_table(self, name: str, frame: pl.DataFrame, *, run_id: str) -> None:
        """Append rows, creating the table when absent.

        Read-concat-rewrite rather than a true append. At the scale this holds —
        an evaluation history, not a fact table — that is simpler and safe.

        ``diagonal_relaxed`` rather than ``vertical_relaxed``: the latter aligns
        differing *dtypes* but still fails on differing column *counts*, and a
        schema that gained a field is the normal case here. Older rows get null
        for the new column, which is honest — those rows genuinely lack it.
        """
        del run_id
        path = self._path(name)
        if path.exists():
            frame = pl.concat([pl.read_parquet(path), frame], how="diagonal_relaxed")
        frame.write_parquet(path)

    def read_table(self, name: str) -> pl.DataFrame:
        path = self._path(name)
        if not path.exists():
            raise KeyError(f"table {name!r} does not exist")
        return pl.read_parquet(path)

    def query(self, sql: str, **params: object) -> pl.DataFrame:
        """Not supported. Use :class:`DuckDBTableStore` when SQL is needed."""
        del sql, params
        raise NotImplementedError(
            "ParquetTableStore does not run SQL — that is what a driver is for. "
            "Read the table and filter it in Polars, or use DuckDBTableStore."
        )

    def table_exists(self, name: str) -> bool:
        return self._path(name).exists()

    def execute(self, sql: str) -> None:
        """Not supported. There is no schema to migrate: Parquet carries its own."""
        del sql
        raise NotImplementedError("ParquetTableStore has no DDL; each file carries its schema")

    # -- convenience ------------------------------------------------------

    def list_tables(self) -> list[str]:
        return sorted(path.stem for path in self._root.glob("*.parquet"))
