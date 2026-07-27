"""Historical backfill from the community archive.

Decision D6 restricts us to the official FPL API, and this stays inside that
line: the archive is a verbatim mirror of official API responses, collected
gameweek by gameweek and published free. It exists because the live API is
*current-season only* — ``element-summary`` returns per-gameweek rows for this
season and nothing but rank and points for prior ones, so there is no official
route to the history a model needs.

**Verified against the archive on 2026-07-27:**

- ``merged_gw.csv`` exists for 2022-23 through 2025-26.
- The expected-goals family is present in all four, which is what makes
  2022/23 the correct backfill floor (D7) rather than an arbitrary cutoff.
- ``defensive_contribution`` appears **only** in 2025-26. Earlier seasons get
  null, never zero: zero would assert the player made no defensive
  contributions, which is a claim the data does not support.
- ``players_raw.csv`` carries both ``id`` (reissued each season) and ``code``
  (stable), which is what allows a player to be followed across seasons.

Provenance is recorded per file, and the current season is reconcilable against
the live API — the archive is trusted, but not blindly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import httpx

from xg_alonso.contracts.identifiers import Season
from xg_alonso.contracts.provenance import SourceTimestamps, TimeSource, utc_now
from xg_alonso.contracts.storage import BronzeSnapshotStore, SnapshotRef

__all__ = [
    "ARCHIVE_BASE_URL",
    "BACKFILL_SEASONS",
    "SOURCE_ARCHIVE_GW",
    "SOURCE_ARCHIVE_PLAYERS",
    "ArchiveFile",
    "fetch_archive_season",
    "seasons_with_defensive_contributions",
]

ARCHIVE_BASE_URL: Final[str] = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

SOURCE_ARCHIVE_GW = "archive.merged_gw"
SOURCE_ARCHIVE_PLAYERS = "archive.players_raw"

#: The D7 range. 2022/23 is the earliest season with expected goals in the
#: official payload, so earlier seasons would supply history the feature set
#: cannot actually use.
BACKFILL_SEASONS: Final[tuple[str, ...]] = ("2022-23", "2023-24", "2024-25", "2025-26")

#: Seasons whose rows carry defensive-contribution counts. Verified by
#: column-diffing the archive; every earlier season must stay null.
_DEFENSIVE_SEASONS: Final[frozenset[str]] = frozenset({"2025-26", "2026-27"})


def seasons_with_defensive_contributions() -> frozenset[str]:
    """Seasons where ``defensive_contribution`` is populated.

    FPL introduced the statistic in 2025/26. A model trained across the full
    backfill therefore has one season of it, not four — which is why the
    baseline leaves the term at zero rather than fitting it on a single season.
    """
    return _DEFENSIVE_SEASONS


@dataclass(frozen=True)
class ArchiveFile:
    """One fetched archive file, with the season it belongs to."""

    season: Season
    source: str
    ref: SnapshotRef


def _timestamps(observed: datetime) -> SourceTimestamps:
    """Timestamps for an archive fetch.

    ``available_time`` is the fetch time, not the season's end: this is when
    *we* could have read the file. Per-row availability is set later during
    normalization, from each fixture's kickoff, which is the timestamp that
    actually governs point-in-time correctness.
    """
    return SourceTimestamps(
        event_time=observed,
        observed_time=observed,
        available_time=observed,
        processed_time=observed,
        time_source=TimeSource.ARCHIVE_DECLARED,
    )


def fetch_archive_season(
    season: str,
    *,
    bronze: BronzeSnapshotStore,
    run_id: str,
    client: httpx.Client | None = None,
    timeout: float = 120.0,
) -> list[ArchiveFile]:
    """Fetch one season's gameweek and player files into bronze.

    Raises:
        httpx.HTTPStatusError: if either file is unavailable. A partial season
            is worse than a missing one — it would look complete downstream.
    """
    owns_client = client is None
    http = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "xg-alonso/0.1 (+research)"},
    )

    try:
        fetched: list[ArchiveFile] = []
        for source, path in (
            (SOURCE_ARCHIVE_GW, f"{season}/gws/merged_gw.csv"),
            (SOURCE_ARCHIVE_PLAYERS, f"{season}/players_raw.csv"),
        ):
            response = http.get(f"{ARCHIVE_BASE_URL}/{path}")
            response.raise_for_status()
            ref = bronze.write(
                source=f"{source}.{season}",
                payload=response.content,
                timestamps=_timestamps(utc_now()),
                run_id=run_id,
                http_status=response.status_code,
            )
            fetched.append(ArchiveFile(season=Season(season), source=source, ref=ref))
        return fetched
    finally:
        if owns_client:
            http.close()


def season_boundary(season: str) -> tuple[datetime, datetime]:
    """Approximate start and end instants for a season, for sanity checks.

    Used to reject rows whose kickoff falls outside their declared season, which
    is the cheapest way to catch a mis-parsed or mis-labelled archive file.
    """
    start_year = int(season.split("-")[0])
    return (
        datetime(start_year, 7, 1, tzinfo=UTC),
        datetime(start_year + 1, 7, 31, tzinfo=UTC),
    )
