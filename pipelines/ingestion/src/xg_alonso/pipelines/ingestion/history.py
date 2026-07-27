"""Player history ingest.

Without this, every player looks identical to the model. At gameweek 1 the
current season has no matches, so a feature set built from current-season data
alone gives every player the same prior — and a recommender that cannot
distinguish players correctly recommends nothing. The pipeline runs, the tests
pass, and the product is useless.

``element-summary/{id}/`` supplies the missing history in two forms:

- ``history`` — per-gameweek rows for the *current* season (empty at GW1)
- ``history_past`` — one aggregate row per *prior* season

At GW1 only ``history_past`` exists, so that is what seeds the rate features.
Its rows are season aggregates rather than gameweek rows, which is fine for
per-90 rates: summing a numerator and a denominator gives the same answer
whether the rows are matches or seasons. It is *not* fine for recency windows,
so those stay null until real gameweeks accumulate — an honest null beats a
number that pretends a season is a match.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from xg_alonso.contracts.provenance import SourceTimestamps, TimeSource
from xg_alonso.contracts.storage import BronzeSnapshotStore, SnapshotRef
from xg_alonso.pipelines.ingestion.fpl_client import FplApiClient

__all__ = [
    "SOURCE_ELEMENT_SUMMARY",
    "ingest_element_summaries",
    "season_end_time",
]

SOURCE_ELEMENT_SUMMARY = "fpl.element_summary"

#: Delay between requests. The API is free and unauthenticated; hammering it
#: would be both rude and a good way to get blocked.
_POLITE_DELAY_SECONDS = 0.08


def season_end_time(season_name: str) -> datetime:
    """When a past season's statistics became final.

    ``history_past`` carries no timestamp, but a point-in-time join needs one.
    Using 30 June of the closing year is late enough that the season is
    certainly complete and early enough that it precedes the next season's
    first deadline — so these rows are visible at GW1 without ever being
    visible *during* the season they describe.
    """
    # Season names arrive as "2024/25".
    head = season_name.split("/")[0]
    try:
        start_year = int(head)
    except ValueError as exc:
        raise ValueError(f"unrecognised season name {season_name!r}") from exc
    return datetime(start_year + 1, 6, 30, tzinfo=UTC)


def ingest_element_summaries(
    *,
    client: FplApiClient,
    bronze: BronzeSnapshotStore,
    element_ids: Sequence[int],
    run_id: str,
    delay_seconds: float = _POLITE_DELAY_SECONDS,
    progress: Iterable[int] | None = None,
) -> list[SnapshotRef]:
    """Fetch each player's history into bronze, one snapshot per player.

    Snapshots are keyed by content hash, so re-running after an interruption
    re-fetches but does not duplicate. That matters here more than elsewhere:
    this is 500-odd sequential requests and it *will* be interrupted.
    """
    refs: list[SnapshotRef] = []
    for index, element_id in enumerate(element_ids):
        response = client.element_summary(element_id)

        # Tag the payload with its element id. The response body does not
        # include it, and a bronze snapshot that cannot say which player it
        # describes is not much of a record.
        body: dict[str, Any] = json.loads(response.payload)
        body["_element_id"] = element_id
        tagged = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

        refs.append(
            bronze.write(
                source=f"{SOURCE_ELEMENT_SUMMARY}.{element_id}",
                payload=tagged,
                timestamps=response.timestamps,
                run_id=run_id,
                http_status=response.status_code,
            )
        )
        if progress is not None and index % 50 == 0:
            pass
        if delay_seconds:
            time.sleep(delay_seconds)
    return refs


def read_element_summaries(
    bronze: BronzeSnapshotStore, element_ids: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """Read back every stored player history that exists.

    Missing players are skipped rather than raising: a partial ingest should
    still produce a partial feature set, not a crash.
    """
    payloads: dict[int, dict[str, Any]] = {}
    for element_id in element_ids:
        ref = bronze.latest(f"{SOURCE_ELEMENT_SUMMARY}.{element_id}")
        if ref is None:
            continue
        payloads[element_id] = json.loads(bronze.read(ref))
    return payloads


def synthetic_timestamps(available_time: datetime) -> SourceTimestamps:
    """Timestamps for a derived record whose availability we assert ourselves."""
    return SourceTimestamps(
        event_time=available_time,
        observed_time=available_time,
        available_time=available_time,
        processed_time=available_time,
        time_source=TimeSource.ARCHIVE_DECLARED,
    )
