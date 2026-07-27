"""Storage adapters — the only package permitted to import a database driver.

Implements the protocols declared in :mod:`xg_alonso.contracts.storage`:

- :class:`FileSystemBronzeStore` — immutable, append-only raw snapshots
- :class:`DuckDBTableStore` — canonical (silver) and model-ready (gold) tables

The ``duckdb-isolation`` contract in ``.importlinter`` fails the build if any
other package imports the driver, which is what keeps decision D2 (DuckDB plus
Parquet) a reversible choice rather than a structural commitment.
"""

from xg_alonso.storage.bronze import FileSystemBronzeStore
from xg_alonso.storage.duckdb_store import DuckDBTableStore

__all__ = ["DuckDBTableStore", "FileSystemBronzeStore"]
