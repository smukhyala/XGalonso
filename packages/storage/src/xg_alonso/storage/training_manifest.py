"""What data a model was trained on, as a hash that does not lie.

**Not the file bytes.** Parquet is not byte-stable: rewriting identical rows
changes compression blocks and embedded metadata, so hashing the file would
make the digest move on every ``xg backfill`` even when nothing about the data
changed. It is also 2.9 MB to re-read on every model load.

So the silver side is a *content summary* — row count, ordered schema, season
and gameweek coverage, the latest ``available_time``, the player count. Those
move when the data moves and stay put when it is merely rewritten. The bronze
side needs no summary: snapshots are already content-addressed by construction,
so their existing hashes are used directly.

**Recorded, never blocking.** A model trained on last week's snapshot is a
legitimate model. This answers "which data?", it does not gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

__all__ = ["TRAINING_MANIFEST_VERSION", "training_data_manifest_hash"]

TRAINING_MANIFEST_VERSION = "training_data_v1"


def training_data_manifest_hash(
    *,
    silver_path: Path,
    bronze_hashes: Sequence[str] = (),
    cutoff: datetime | None = None,
) -> tuple[str, dict[str, Any]]:
    """Hash the training inputs, and return the record it was built from.

    Args:
        silver_path: The canonical stats table.
        bronze_hashes: Content hashes of the raw snapshots that produced it.
            Already byte-exact, so they are used as-is.
        cutoff: Recorded so a later reader knows what the summary excluded.

    Returns:
        ``(digest, record)``. The record is stored beside the digest so a
        mismatch can be explained rather than merely detected.
    """
    frame = pl.read_parquet(silver_path)
    seasons = (
        sorted({str(s) for s in frame["season"].unique()}) if "season" in frame.columns else []
    )
    gameweeks = (
        sorted({int(g) for g in frame["gameweek_id"].unique().drop_nulls()})
        if "gameweek_id" in frame.columns
        else []
    )
    latest = (
        frame["available_time"].max()
        if "available_time" in frame.columns and frame.height
        else None
    )
    # Polars types a max() as a union covering temporal columns. This one is a
    # timestamp; the narrowing is stated once rather than at the use site.
    latest_iso = latest.isoformat() if isinstance(latest, datetime) else None

    record: dict[str, Any] = {
        "manifest_version": TRAINING_MANIFEST_VERSION,
        "silver": {
            "name": silver_path.name,
            "rows": frame.height,
            "schema": [[c, str(d)] for c, d in zip(frame.columns, frame.dtypes, strict=True)],
            "seasons": seasons,
            "gameweeks": gameweeks,
            "players": (frame["player_code"].n_unique() if "player_code" in frame.columns else 0),
            "max_available_time": latest_iso,
        },
        "bronze": sorted(bronze_hashes),
        "cutoff": cutoff.isoformat() if cutoff else None,
    }
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, record
