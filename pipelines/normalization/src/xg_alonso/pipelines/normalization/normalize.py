"""Bronze to silver: turn raw FPL payloads into canonical tables.

Every row carries ``available_time``, inherited from the snapshot it came from.
That single column is what makes point-in-time joins possible downstream — a
silver table without it is unusable for feature generation, so it is added here
rather than left to callers.

Element ids are translated to stable player codes at this boundary. Doing it
once, here, means nothing downstream ever has to know that FPL reissues ids.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import polars as pl

from xg_alonso.contracts.identifiers import Season
from xg_alonso.pipelines.normalization.schema import (
    FIXTURES_SCHEMA,
    GAMEWEEKS_SCHEMA,
    PLAYER_GAMEWEEK_STATS_SCHEMA,
    PLAYERS_SCHEMA,
    TEAMS_SCHEMA,
    conform,
    empty_frame,
)

__all__ = [
    "build_element_to_code_map",
    "normalize_element_summary",
    "normalize_fixtures",
    "normalize_gameweeks",
    "normalize_history_past",
    "normalize_players",
    "normalize_teams",
]

_POSITION_BY_ELEMENT_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _parse_utc(expr: pl.Expr) -> pl.Expr:
    """Parse an ISO-8601 string into a UTC-aware datetime."""
    return expr.str.to_datetime(time_zone="UTC", strict=False)


def build_element_to_code_map(payload: dict[str, Any]) -> dict[int, int]:
    """Map this season's element ids to stable player codes.

    Needed because ``element-summary`` and ``picks`` speak element ids while
    every silver table is keyed on ``player_code``.
    """
    return {int(e["id"]): int(e["code"]) for e in payload.get("elements", [])}


def normalize_players(
    payload: dict[str, Any], *, season: Season, available_time: datetime
) -> pl.DataFrame:
    """Canonical player rows from a ``bootstrap-static`` payload."""
    elements = payload.get("elements") or []
    if not elements:
        return empty_frame(PLAYERS_SCHEMA)

    frame = pl.DataFrame(
        [
            {
                "player_code": e.get("code"),
                "element_id": e.get("id"),
                "web_name": e.get("web_name"),
                "first_name": e.get("first_name"),
                "second_name": e.get("second_name"),
                "position": _POSITION_BY_ELEMENT_TYPE.get(e.get("element_type", 0)),
                "team_id": e.get("team"),
                "team_code": e.get("team_code"),
                "current_price": e.get("now_cost"),
                "status": e.get("status"),
                "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
            }
            for e in elements
        ],
        infer_schema_length=None,
    ).with_columns(
        pl.lit(str(season)).alias("season"),
        pl.lit(available_time).cast(PLAYERS_SCHEMA["available_time"]).alias("available_time"),
    )
    return conform(frame, PLAYERS_SCHEMA)


def normalize_teams(
    payload: dict[str, Any], *, season: Season, available_time: datetime
) -> pl.DataFrame:
    """Canonical club rows from a ``bootstrap-static`` payload.

    The ``strength_*`` fields are deliberately not carried through. They read 0
    or null in preseason and switch numeric scale once the season starts, so a
    silver column holding them would be a trap: present, plausible, and wrong.
    Team strength is derived from expected goals instead.
    """
    teams = payload.get("teams") or []
    if not teams:
        return empty_frame(TEAMS_SCHEMA)

    frame = pl.DataFrame(
        [
            {
                "team_id": t.get("id"),
                "team_code": t.get("code"),
                "name": t.get("name"),
                "short_name": t.get("short_name"),
            }
            for t in teams
        ],
        infer_schema_length=None,
    ).with_columns(
        pl.lit(str(season)).alias("season"),
        pl.lit(available_time).cast(TEAMS_SCHEMA["available_time"]).alias("available_time"),
    )
    return conform(frame, TEAMS_SCHEMA)


def normalize_gameweeks(
    payload: dict[str, Any], *, season: Season, available_time: datetime
) -> pl.DataFrame:
    """Canonical gameweek rows, including deadlines.

    Deadlines matter beyond scheduling: they are the natural prediction cutoff,
    so a feature build for gameweek *N* uses only data available at gameweek
    *N*'s deadline.
    """
    events = payload.get("events") or []
    if not events:
        return empty_frame(GAMEWEEKS_SCHEMA)

    frame = pl.DataFrame(
        [
            {
                "gameweek_id": e.get("id"),
                "deadline_time": e.get("deadline_time"),
                "finished": e.get("finished"),
                "is_current": e.get("is_current"),
                "is_next": e.get("is_next"),
            }
            for e in events
        ],
        infer_schema_length=None,
    ).with_columns(
        _parse_utc(pl.col("deadline_time")).alias("deadline_time"),
        pl.lit(str(season)).alias("season"),
        pl.lit(available_time).cast(GAMEWEEKS_SCHEMA["available_time"]).alias("available_time"),
    )
    return conform(frame, GAMEWEEKS_SCHEMA)


def normalize_fixtures(
    payload: list[dict[str, Any]], *, season: Season, available_time: datetime
) -> pl.DataFrame:
    """Canonical fixture rows from the ``fixtures`` payload."""
    if not payload:
        return empty_frame(FIXTURES_SCHEMA)

    frame = pl.DataFrame(
        [
            {
                "fixture_id": f.get("id"),
                "gameweek_id": f.get("event"),
                "home_team_id": f.get("team_h"),
                "away_team_id": f.get("team_a"),
                "kickoff_time": f.get("kickoff_time"),
                "finished": f.get("finished"),
                "home_score": f.get("team_h_score"),
                "away_score": f.get("team_a_score"),
                "home_difficulty": f.get("team_h_difficulty"),
                "away_difficulty": f.get("team_a_difficulty"),
            }
            for f in payload
        ],
        infer_schema_length=None,
    ).with_columns(
        _parse_utc(pl.col("kickoff_time")).alias("kickoff_time"),
        pl.lit(str(season)).alias("season"),
        pl.lit(available_time).cast(FIXTURES_SCHEMA["available_time"]).alias("available_time"),
    )
    return conform(frame, FIXTURES_SCHEMA)


def normalize_element_summary(
    payload: dict[str, Any],
    *,
    player_code: int,
    season: Season,
    available_time: datetime,
) -> pl.DataFrame:
    """Per-gameweek stats for one player from ``element-summary/{id}/``.

    ``available_time`` is set to each fixture's kickoff plus a settlement delay
    rather than to the snapshot time. A gameweek's statistics are not usable the
    instant the match starts, and stamping them with the fetch time would make
    every historical row look as though it had been available since the moment
    we happened to download it — which is leakage dressed up as bookkeeping.
    """
    history = payload.get("history") or []
    if not history:
        return empty_frame(PLAYER_GAMEWEEK_STATS_SCHEMA)

    frame = pl.DataFrame(
        [
            {
                "gameweek_id": h.get("round"),
                "fixture_id": h.get("fixture"),
                "opponent_team_id": h.get("opponent_team"),
                "was_home": h.get("was_home"),
                "minutes": h.get("minutes"),
                "starts": h.get("starts"),
                "goals_scored": h.get("goals_scored"),
                "assists": h.get("assists"),
                "clean_sheets": h.get("clean_sheets"),
                "goals_conceded": h.get("goals_conceded"),
                "saves": h.get("saves"),
                "yellow_cards": h.get("yellow_cards"),
                "red_cards": h.get("red_cards"),
                "own_goals": h.get("own_goals"),
                "penalties_saved": h.get("penalties_saved"),
                "penalties_missed": h.get("penalties_missed"),
                "bonus": h.get("bonus"),
                "bps": h.get("bps"),
                "total_points": h.get("total_points"),
                "expected_goals": h.get("expected_goals"),
                "expected_assists": h.get("expected_assists"),
                "expected_goal_involvements": h.get("expected_goal_involvements"),
                "expected_goals_conceded": h.get("expected_goals_conceded"),
                "defensive_contribution": h.get("defensive_contribution"),
                "value": h.get("value"),
                "kickoff_time": h.get("kickoff_time"),
            }
            for h in history
        ],
        infer_schema_length=None,
    ).with_columns(
        _parse_utc(pl.col("kickoff_time")).alias("kickoff_time"),
        pl.lit(player_code).alias("player_code"),
        pl.lit(str(season)).alias("season"),
    )

    # Results settle a few hours after kickoff — bonus points in particular are
    # provisional until the match is finalised.
    settled = pl.col("kickoff_time") + pl.duration(hours=3)
    frame = frame.with_columns(
        pl.when(settled.is_null())
        .then(pl.lit(available_time).cast(PLAYER_GAMEWEEK_STATS_SCHEMA["available_time"]))
        .otherwise(settled)
        .alias("available_time")
    )
    return conform(frame, PLAYER_GAMEWEEK_STATS_SCHEMA)


def normalize_history_past(
    payload: dict[str, Any],
    *,
    player_code: int,
    season_end_lookup: Callable[[str], datetime],
) -> pl.DataFrame:
    """Prior-season aggregates from ``element-summary``'s ``history_past``.

    One row per past season, shaped like ``player_gameweek_stats`` so the same
    generators work over it. ``gameweek_id`` is null because these are seasons,
    not gameweeks — which is exactly what stops a recency window from silently
    treating a whole season as one recent match.

    ``available_time`` is the season's end, so these rows are visible when
    predicting the following season and invisible during the season itself.
    """
    past = payload.get("history_past") or []
    if not past:
        return empty_frame(PLAYER_GAMEWEEK_STATS_SCHEMA)

    rows = []
    for entry in past:
        season_name = str(entry.get("season_name", ""))
        try:
            available = season_end_lookup(season_name)
        except ValueError:
            continue
        rows.append(
            {
                "player_code": player_code,
                "gameweek_id": None,
                "season": season_name,
                "fixture_id": None,
                "opponent_team_id": None,
                "was_home": None,
                "minutes": entry.get("minutes"),
                "starts": entry.get("starts"),
                "goals_scored": entry.get("goals_scored"),
                "assists": entry.get("assists"),
                "clean_sheets": entry.get("clean_sheets"),
                "goals_conceded": entry.get("goals_conceded"),
                "saves": entry.get("saves"),
                "yellow_cards": entry.get("yellow_cards"),
                "red_cards": entry.get("red_cards"),
                "own_goals": entry.get("own_goals"),
                "penalties_saved": entry.get("penalties_saved"),
                "penalties_missed": entry.get("penalties_missed"),
                "bonus": entry.get("bonus"),
                "bps": entry.get("bps"),
                "total_points": entry.get("total_points"),
                "expected_goals": entry.get("expected_goals"),
                "expected_assists": entry.get("expected_assists"),
                "expected_goal_involvements": entry.get("expected_goal_involvements"),
                "expected_goals_conceded": entry.get("expected_goals_conceded"),
                "defensive_contribution": entry.get("defensive_contribution"),
                "value": entry.get("end_cost"),
                "kickoff_time": available,
                "available_time": available,
            }
        )

    if not rows:
        return empty_frame(PLAYER_GAMEWEEK_STATS_SCHEMA)
    return conform(pl.DataFrame(rows, infer_schema_length=None), PLAYER_GAMEWEEK_STATS_SCHEMA)
