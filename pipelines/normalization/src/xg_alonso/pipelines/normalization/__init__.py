"""Bronze to silver normalization.

Turns raw FPL payloads into canonical, point-in-time-safe tables. Two invariants
are established here and relied upon everywhere downstream:

1. **Every row carries ``available_time``.** Without it a table cannot take part
   in a point-in-time join, so feature generation would have no way to know what
   was knowable when.
2. **Players are keyed on ``player_code``.** FPL reissues ``element_id`` every
   season; translating once at this boundary means nothing downstream can
   accidentally join across seasons on an id that was reused.
"""

from xg_alonso.pipelines.normalization.archive import (
    ArchiveNormalizationResult,
    normalize_archive_season,
)
from xg_alonso.pipelines.normalization.normalize import (
    build_element_to_code_map,
    normalize_element_summary,
    normalize_fixtures,
    normalize_gameweeks,
    normalize_history_past,
    normalize_players,
    normalize_teams,
)
from xg_alonso.pipelines.normalization.schema import (
    FIXTURES_SCHEMA,
    GAMEWEEKS_SCHEMA,
    PLAYER_GAMEWEEK_STATS_SCHEMA,
    PLAYERS_SCHEMA,
    SILVER_TABLES,
    TEAMS_SCHEMA,
    conform,
    empty_frame,
)

__all__ = [
    "FIXTURES_SCHEMA",
    "GAMEWEEKS_SCHEMA",
    "PLAYERS_SCHEMA",
    "PLAYER_GAMEWEEK_STATS_SCHEMA",
    "SILVER_TABLES",
    "TEAMS_SCHEMA",
    "ArchiveNormalizationResult",
    "build_element_to_code_map",
    "conform",
    "empty_frame",
    "normalize_archive_season",
    "normalize_element_summary",
    "normalize_fixtures",
    "normalize_gameweeks",
    "normalize_history_past",
    "normalize_players",
    "normalize_teams",
]
