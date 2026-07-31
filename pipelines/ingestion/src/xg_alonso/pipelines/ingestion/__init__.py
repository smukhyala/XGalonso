"""Official FPL source adapters writing immutable bronze snapshots.

Decision D6 restricts us to the official FPL API: no scraping, no paid
providers, zero budget. That constraint is less limiting than it sounds — the
API publishes ``expected_goals``, ``expected_assists``,
``expected_goal_involvements`` and ``expected_goals_conceded`` per player per
gameweek from 2022/23 onward, which covers most of the underlying-stats
taxonomy with licensed data and no terms-of-service exposure.

This is the only package permitted to import ``httpx``. Everything downstream
reads bronze snapshots, so a feature build is reproducible offline and a test
that reaches for the network fails loudly instead of flaking.
"""

from xg_alonso.pipelines.ingestion.archive import (
    ARCHIVE_BASE_URL,
    BACKFILL_SEASONS,
    SOURCE_ARCHIVE_GW,
    SOURCE_ARCHIVE_PLAYERS,
    ArchiveFile,
    fetch_archive_season,
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

__all__ = [
    "ARCHIVE_BASE_URL",
    "BACKFILL_SEASONS",
    "FPL_BASE_URL",
    "MATERIAL_OWNERSHIP",
    "SOURCE_ARCHIVE_GW",
    "SOURCE_ARCHIVE_PLAYERS",
    "SOURCE_BOOTSTRAP",
    "SOURCE_ELEMENT_SUMMARY",
    "SOURCE_FIXTURES",
    "ArchiveFile",
    "FplApiClient",
    "FplResponse",
    "IngestResult",
    "OfflineError",
    "PreseasonWarning",
    "derive_available_time",
    "detect_preseason_hazards",
    "diff_bootstrap",
    "elements_by_code",
    "fetch_archive_season",
    "git_manifest",
    "guard_offline",
    "ingest_bootstrap",
    "ingest_element_summaries",
    "load_local_payload",
    "load_rules_from_snapshot",
    "read_element_summaries",
    "read_snapshot_payload",
    "season_end_time",
    "seasons_with_defensive_contributions",
]
