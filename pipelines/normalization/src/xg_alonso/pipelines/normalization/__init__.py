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
    build_players_history,
    build_teams_history,
    normalize_archive_season,
)
from xg_alonso.pipelines.normalization.match_events import (
    PUBLICATION_LAG,
    SOURCE_TEAM_ALIASES,
    MatchEventsReport,
    TeamMappingError,
    TeamNameMapping,
    map_team_names,
    normalize_match_events,
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
    TEAM_MATCH_EVENTS_SCHEMA,
    TEAMS_SCHEMA,
    conform,
    empty_frame,
)

__all__ = [
    "FIXTURES_SCHEMA",
    "GAMEWEEKS_SCHEMA",
    "PLAYERS_SCHEMA",
    "PLAYER_GAMEWEEK_STATS_SCHEMA",
    "PUBLICATION_LAG",
    "SILVER_TABLES",
    "SOURCE_TEAM_ALIASES",
    "TEAMS_SCHEMA",
    "TEAM_MATCH_EVENTS_SCHEMA",
    "ArchiveNormalizationResult",
    "MatchEventsReport",
    "TeamMappingError",
    "TeamNameMapping",
    "build_element_to_code_map",
    "build_players_history",
    "build_teams_history",
    "conform",
    "empty_frame",
    "map_team_names",
    "normalize_archive_season",
    "normalize_element_summary",
    "normalize_fixtures",
    "normalize_gameweeks",
    "normalize_history_past",
    "normalize_match_events",
    "normalize_players",
    "normalize_teams",
]
