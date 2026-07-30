"""Official FPL source adapters writing immutable bronze snapshots.

Decision D6 pins us to free sources, primarily the official FPL API: no paid
providers, zero budget. That constraint is less limiting than it sounds — the
API publishes ``expected_goals``, ``expected_assists``,
``expected_goal_involvements`` and ``expected_goals_conceded`` per player per
gameweek from 2022/23 onward, which covers most of the underlying-stats
taxonomy with licensed data and no terms-of-service exposure.

D6 was relaxed on 2026-07-29 to permit fetching free public match data that the
official API does not publish (see ``docs/match_event_data.md``). Permission to
fetch is not permission to fetch anything: :mod:`.robots` refuses any origin
whose ``robots.txt`` disallows us, which is why Understat — the richer source —
is not used here.

This is the only package permitted to import ``httpx``. Everything downstream
reads bronze snapshots, so a feature build is reproducible offline and a test
that reaches for the network fails loudly instead of flaking.
"""

from xg_alonso.pipelines.ingestion.archive import (
    ARCHIVE_BASE_URL,
    BACKFILL_SEASONS,
    SOURCE_ARCHIVE_GW,
    SOURCE_ARCHIVE_PLAYERS,
    SOURCE_ARCHIVE_TEAMS,
    ArchiveFile,
    fetch_archive_season,
    fetch_archive_teams,
    seasons_with_defensive_contributions,
)
from xg_alonso.pipelines.ingestion.bootstrap import (
    SOURCE_BOOTSTRAP,
    SOURCE_FIXTURES,
    IngestResult,
    PreseasonWarning,
    detect_preseason_hazards,
    git_manifest,
    ingest_bootstrap,
    load_local_payload,
    load_rules_from_snapshot,
    read_snapshot_payload,
)
from xg_alonso.pipelines.ingestion.changes import (
    MATERIAL_OWNERSHIP,
    diff_bootstrap,
    elements_by_code,
)
from xg_alonso.pipelines.ingestion.fpl_client import (
    FPL_BASE_URL,
    FplApiClient,
    FplResponse,
    OfflineError,
    derive_available_time,
    guard_offline,
)
from xg_alonso.pipelines.ingestion.history import (
    SOURCE_ELEMENT_SUMMARY,
    ingest_element_summaries,
    read_element_summaries,
    season_end_time,
)
from xg_alonso.pipelines.ingestion.match_events import (
    DIVISIONS,
    FOOTBALL_DATA_BASE_URL,
    REQUIRED_COLUMNS,
    SOURCE_MATCH_EVENTS,
    MatchEventsFile,
    division_name,
    fetch_match_events_season,
    fetch_match_events_seasons,
    season_code,
)
from xg_alonso.pipelines.ingestion.robots import (
    USER_AGENT,
    Pacer,
    RobotsDisallowedError,
    RobotsGate,
    RobotsUnavailableError,
)

__all__ = [
    "ARCHIVE_BASE_URL",
    "BACKFILL_SEASONS",
    "DIVISIONS",
    "FOOTBALL_DATA_BASE_URL",
    "FPL_BASE_URL",
    "MATERIAL_OWNERSHIP",
    "REQUIRED_COLUMNS",
    "SOURCE_ARCHIVE_GW",
    "SOURCE_ARCHIVE_PLAYERS",
    "SOURCE_ARCHIVE_TEAMS",
    "SOURCE_BOOTSTRAP",
    "SOURCE_ELEMENT_SUMMARY",
    "SOURCE_FIXTURES",
    "SOURCE_MATCH_EVENTS",
    "USER_AGENT",
    "ArchiveFile",
    "FplApiClient",
    "FplResponse",
    "IngestResult",
    "MatchEventsFile",
    "OfflineError",
    "Pacer",
    "PreseasonWarning",
    "RobotsDisallowedError",
    "RobotsGate",
    "RobotsUnavailableError",
    "derive_available_time",
    "detect_preseason_hazards",
    "diff_bootstrap",
    "division_name",
    "elements_by_code",
    "fetch_archive_season",
    "fetch_archive_teams",
    "fetch_match_events_season",
    "fetch_match_events_seasons",
    "git_manifest",
    "guard_offline",
    "ingest_bootstrap",
    "ingest_element_summaries",
    "load_local_payload",
    "load_rules_from_snapshot",
    "read_element_summaries",
    "read_snapshot_payload",
    "season_code",
    "season_end_time",
    "seasons_with_defensive_contributions",
]
