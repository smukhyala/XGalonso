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
from xg_alonso.pipelines.ingestion.fpl_client import (
    FPL_BASE_URL,
    FplApiClient,
    FplResponse,
    OfflineError,
    derive_available_time,
)
from xg_alonso.pipelines.ingestion.history import (
    SOURCE_ELEMENT_SUMMARY,
    ingest_element_summaries,
    read_element_summaries,
    season_end_time,
)

__all__ = [
    "FPL_BASE_URL",
    "SOURCE_BOOTSTRAP",
    "SOURCE_ELEMENT_SUMMARY",
    "SOURCE_FIXTURES",
    "FplApiClient",
    "FplResponse",
    "IngestResult",
    "OfflineError",
    "PreseasonWarning",
    "derive_available_time",
    "detect_preseason_hazards",
    "git_manifest",
    "ingest_bootstrap",
    "ingest_element_summaries",
    "load_local_payload",
    "load_rules_from_snapshot",
    "read_element_summaries",
    "read_snapshot_payload",
    "season_end_time",
]
