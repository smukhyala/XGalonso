"""Team-match event counts from football-data.co.uk.

**Why this source.** The official FPL API publishes per-player expected goals
and its own composite indices, but nothing about what a *team* did in a match:
how many shots it took, how many were on target, how many corners it won, how
many fouls it gave away. Those are the raw event counts the discovery engine
wants to form hypotheses over, and no official endpoint carries them.

**Why not the obvious source.** Understat publishes shot-level data with xG per
shot, which is strictly richer. It also publishes::

    User-agent: *
    Disallow: /

so it is out, and the refusal is enforced by :mod:`.robots` rather than
remembered. FBref sits behind an interactive bot challenge, which would mean
evading a control rather than passing one. football-data.co.uk publishes
``User-agent: * / Disallow:`` — explicit permission for everything — and serves
static CSV. It is the source that said yes.

**What it costs.** Team-match totals, not shot-level detail. We learn that a
team took 18 shots and 7 were on target; we do not learn where they were taken
from or what each was worth. That is a real reduction in resolution and it is
recorded here rather than glossed: the shot-location features in the original
brief are not buildable from this source, and claiming otherwise would put a
hole in the feature registry's lineage.

**Verified against the source on 2026-07-29:** ``E0.csv`` exists for 2223,
2324, 2425 and 2526 with 380 matches each, and the ``HS/AS/HST/AST/HF/AF/HC/AC``
family is present in all four.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import httpx

from xg_alonso.contracts.identifiers import Season
from xg_alonso.contracts.provenance import SourceTimestamps, TimeSource, utc_now
from xg_alonso.contracts.storage import BronzeSnapshotStore, SnapshotRef
from xg_alonso.pipelines.ingestion.fpl_client import guard_offline
from xg_alonso.pipelines.ingestion.robots import USER_AGENT, Pacer, RobotsGate

__all__ = [
    "DIVISIONS",
    "FOOTBALL_DATA_BASE_URL",
    "REQUIRED_COLUMNS",
    "SOURCE_MATCH_EVENTS",
    "MatchEventsFile",
    "division_name",
    "fetch_match_events_season",
    "fetch_match_events_seasons",
    "season_code",
]

FOOTBALL_DATA_BASE_URL: Final[str] = "https://www.football-data.co.uk/mmz4281"

SOURCE_MATCH_EVENTS: Final[str] = "football_data.match_events"

#: Division codes we know how to read. ``E0`` is the Premier League and is the
#: only one ingested by default.
#:
#: ``E1`` (Championship) is supported because a promoted side otherwise arrives
#: with no history at all — Coventry City and Hull City are in the 2026/27 FPL
#: bootstrap and played no Premier League football in the backfill window. It is
#: off by default and every row carries its ``division``, because pooling
#: Championship shot counts with Premier League ones as though they were the
#: same quantity is a modelling decision, not a data-loading one.
DIVISIONS: Final[dict[str, str]] = {
    "E0": "Premier League",
    "E1": "Championship",
}

#: Columns the parser depends on. Absence is a hard error, not a null column:
#: a season silently missing ``HST`` would train a model on a feature that is
#: null for one season and populated for three, which looks like signal.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
)


def division_name(division: str) -> str:
    """Human name for a division code.

    Raises:
        KeyError: for a code we have not verified against the source.
    """
    return DIVISIONS[division]


def season_code(season: str) -> str:
    """Convert ``"2022-23"`` to the source's ``"2223"`` path segment.

    Raises:
        ValueError: if the season is not in ``YYYY-YY`` form. The source's path
            is positional, so a malformed season would silently fetch a
            different year rather than 404.
    """
    parts = season.split("-")
    expected_parts = 2
    if len(parts) != expected_parts or len(parts[0]) != 4 or len(parts[1]) != 2:
        message = f"expected a season like '2022-23', got {season!r}"
        raise ValueError(message)
    if not parts[0].isdigit() or not parts[1].isdigit():
        message = f"expected a season like '2022-23', got {season!r}"
        raise ValueError(message)

    start_short = parts[0][2:]
    if (int(start_short) + 1) % 100 != int(parts[1]):
        message = (
            f"season {season!r} is not a consecutive pair of years; "
            "the source path is positional so this would fetch the wrong season"
        )
        raise ValueError(message)
    return f"{start_short}{parts[1]}"


@dataclass(frozen=True)
class MatchEventsFile:
    """One fetched season-division file."""

    season: Season
    division: str
    ref: SnapshotRef


def _timestamps(observed: datetime) -> SourceTimestamps:
    """Timestamps for a season-file fetch.

    ``available_time`` is the fetch time — when *we* could have read the file —
    exactly as for the archive. Per-row availability is far more important here
    and is set during normalization from each match's kickoff, because these are
    post-match statistics: a row whose availability was left at the file's fetch
    time would make a completed season look available from its first minute.
    """
    return SourceTimestamps(
        event_time=observed,
        observed_time=observed,
        available_time=observed,
        processed_time=observed,
        time_source=TimeSource.ARCHIVE_DECLARED,
    )


def fetch_match_events_season(
    season: str,
    *,
    bronze: BronzeSnapshotStore,
    run_id: str,
    division: str = "E0",
    client: httpx.Client | None = None,
    gate: RobotsGate | None = None,
    pacer: Pacer | None = None,
    timeout: float = 60.0,
) -> MatchEventsFile:
    """Fetch one season-division CSV into bronze.

    The robots gate runs before the request, not after: a source that refuses us
    should cost one ``robots.txt`` read, not a download we then decide to
    discard.

    Raises:
        KeyError: for an unrecognised division code.
        OfflineError: when ``XG_ALONSO_OFFLINE`` is set.
        RobotsDisallowedError: if the origin refuses automated access.
        RobotsUnavailableError: if consent could not be established.
        httpx.HTTPStatusError: on a non-2xx response.
    """
    division_name(division)  # reject unknown codes before touching the network
    code = season_code(season)
    url = f"{FOOTBALL_DATA_BASE_URL}/{code}/{division}.csv"

    guard_offline(url)

    owns_client = client is None
    http = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv, */*"},
    )

    try:
        checker = gate or RobotsGate(client=http)
        checker.require(url)

        (pacer or Pacer(checker.crawl_delay(url))).wait()

        response = http.get(url)
        response.raise_for_status()

        ref = bronze.write(
            source=f"{SOURCE_MATCH_EVENTS}.{division}.{season}",
            payload=response.content,
            timestamps=_timestamps(utc_now()),
            run_id=run_id,
            http_status=response.status_code,
        )
        return MatchEventsFile(season=Season(season), division=division, ref=ref)
    finally:
        if owns_client:
            http.close()


def fetch_match_events_seasons(
    seasons: tuple[str, ...],
    *,
    bronze: BronzeSnapshotStore,
    run_id: str,
    division: str = "E0",
    client: httpx.Client | None = None,
    timeout: float = 60.0,
) -> list[MatchEventsFile]:
    """Fetch several seasons, sharing one client, one robots check and one pacer.

    Sharing the pacer is the point: a per-call pacer would let a four-season
    backfill issue four immediate requests, which is exactly the burst the delay
    exists to prevent.
    """
    owns_client = client is None
    http = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv, */*"},
    )
    try:
        gate = RobotsGate(client=http)
        sample_url = f"{FOOTBALL_DATA_BASE_URL}/0000/{division}.csv"
        pacer = Pacer(gate.crawl_delay(sample_url))
        return [
            fetch_match_events_season(
                season,
                bronze=bronze,
                run_id=run_id,
                division=division,
                client=http,
                gate=gate,
                pacer=pacer,
                timeout=timeout,
            )
            for season in seasons
        ]
    finally:
        if owns_client:
            http.close()
