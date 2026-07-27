"""Storage adapters — the only package permitted to import a database driver.

Implements the protocols declared in :mod:`xg_alonso.contracts.storage`.

:class:`~xg_alonso.storage.bronze.FileSystemBronzeStore` is exported here
because it needs nothing beyond the standard library.

:class:`~xg_alonso.storage.duckdb_store.DuckDBTableStore` is deliberately **not**
re-exported. Importing it here would mean every consumer of ``xg_alonso.storage``
transitively imported ``duckdb``, which would quietly defeat the
``duckdb-isolation`` contract that exists to keep decision D2 reversible. Code
that genuinely wants DuckDB reaches for it explicitly::

    from xg_alonso.storage.duckdb_store import DuckDBTableStore
"""

from xg_alonso.storage.bronze import FileSystemBronzeStore

__all__ = ["FileSystemBronzeStore"]
