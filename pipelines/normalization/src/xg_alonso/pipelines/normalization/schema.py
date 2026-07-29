"""Canonical silver table definitions.

Silver is derived and rebuildable — bronze is the only thing that must never be
lost. So these tables are replaced wholesale on each normalization run rather
than mutated in place, which removes a whole class of partial-update bug.

**Identity.** ``player_code`` is the primary key everywhere, never
``element_id``. FPL reissues element ids each season, so a table keyed on them
silently joins one season's Salah to another season's someone-else. The stable
``code`` field is kept as the canonical identity and the per-season element id is
retained only for calling back into the API.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = [
    "FIXTURES_SCHEMA",
    "GAMEWEEKS_SCHEMA",
    "PLAYERS_SCHEMA",
    "PLAYER_GAMEWEEK_STATS_SCHEMA",
    "SILVER_TABLES",
    "TEAMS_SCHEMA",
    "TEAM_MATCH_EVENTS_SCHEMA",
]

_UTC = pl.Datetime(time_unit="us", time_zone="UTC")

PLAYERS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "player_code": pl.Int64(),  # stable across seasons; the canonical identity
    "element_id": pl.Int64(),  # per-season id, reissued annually
    "season": pl.Utf8(),
    "web_name": pl.Utf8(),
    "first_name": pl.Utf8(),
    "second_name": pl.Utf8(),
    "position": pl.Utf8(),  # GKP / DEF / MID / FWD
    "team_id": pl.Int64(),
    "team_code": pl.Int64(),
    "current_price": pl.Int64(),  # tenths of a million
    "status": pl.Utf8(),  # a=available, i=injured, s=suspended, u=unavailable, d=doubtful
    "chance_of_playing_next_round": pl.Int64(),
    # Share of managers owning the player, as a percentage. Published on every
    # element and previously dropped.
    #
    # It is not a nicety: an objective that seeks differentials or protects a
    # rank is *defined* by ownership, and without this column those objectives
    # would silently score every player identically on the term that
    # distinguishes them. Nullable, because a preseason payload can omit it.
    #
    # **Not effective ownership.** EO folds in captaincy and is what actually
    # moves rank; FPL does not publish it. Treated as a proxy everywhere.
    "selected_by_percent": pl.Float64(),
    "available_time": _UTC,
}

TEAMS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "team_id": pl.Int64(),
    "team_code": pl.Int64(),  # stable across seasons
    "season": pl.Utf8(),
    "name": pl.Utf8(),
    "short_name": pl.Utf8(),
    "available_time": _UTC,
}

GAMEWEEKS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "gameweek_id": pl.Int64(),
    "season": pl.Utf8(),
    "deadline_time": _UTC,
    "finished": pl.Boolean(),
    "is_current": pl.Boolean(),
    "is_next": pl.Boolean(),
    "available_time": _UTC,
}

FIXTURES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "fixture_id": pl.Int64(),
    "gameweek_id": pl.Int64(),
    "season": pl.Utf8(),
    "home_team_id": pl.Int64(),
    "away_team_id": pl.Int64(),
    "kickoff_time": _UTC,
    "finished": pl.Boolean(),
    "home_score": pl.Int64(),
    "away_score": pl.Int64(),
    "home_difficulty": pl.Int64(),
    "away_difficulty": pl.Int64(),
    "available_time": _UTC,
}

PLAYER_GAMEWEEK_STATS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "player_code": pl.Int64(),
    "gameweek_id": pl.Int64(),
    "season": pl.Utf8(),
    "fixture_id": pl.Int64(),
    "opponent_team_id": pl.Int64(),
    "was_home": pl.Boolean(),
    "minutes": pl.Int64(),
    "starts": pl.Int64(),
    "goals_scored": pl.Int64(),
    "assists": pl.Int64(),
    "clean_sheets": pl.Int64(),
    "goals_conceded": pl.Int64(),
    "saves": pl.Int64(),
    "yellow_cards": pl.Int64(),
    "red_cards": pl.Int64(),
    "own_goals": pl.Int64(),
    "penalties_saved": pl.Int64(),
    "penalties_missed": pl.Int64(),
    "bonus": pl.Int64(),
    "bps": pl.Int64(),
    "total_points": pl.Int64(),
    # Published by the official API from 2022/23 onward, which is what makes
    # decision D7's backfill floor 2022/23 rather than earlier.
    "expected_goals": pl.Float64(),
    "expected_assists": pl.Float64(),
    "expected_goal_involvements": pl.Float64(),
    "expected_goals_conceded": pl.Float64(),
    # Defensive contributions exist only from 2025/26. Older rows are null, not
    # zero — zero would assert the player made none, which is a different claim.
    "defensive_contribution": pl.Int64(),
    "value": pl.Int64(),  # price at the time, in tenths of a million
    # --- market state -----------------------------------------------------
    # Ownership and transfer flow. These are the only fields here that cannot
    # be reconstructed after the fact: they describe what the market believed
    # at a moment in time, and the API publishes only the current value. A day
    # not captured is a day that does not exist, which is why they are carried
    # even though the price model itself is deferred (D11).
    "selected": pl.Int64(),  # managers owning this player
    "transfers_in": pl.Int64(),
    "transfers_out": pl.Int64(),
    "transfers_balance": pl.Int64(),
    # --- FPL's own composite indices --------------------------------------
    # Opta-derived proxies the official API publishes. `threat` stands in for
    # shot volume and `creativity` for chance creation — the two families the
    # brief wanted from event data that D6 otherwise puts out of reach.
    "influence": pl.Float64(),
    "creativity": pl.Float64(),
    "threat": pl.Float64(),
    "ict_index": pl.Float64(),
    "kickoff_time": _UTC,
    "available_time": _UTC,
}

TEAM_MATCH_EVENTS_SCHEMA: Final[dict[str, pl.DataType]] = {
    # One row per team per match, so a team's own row carries both what it did
    # and what was done to it. The alternative — one row per match with home_*
    # and away_* columns — would force every downstream feature to branch on
    # which side a player's team was, which is where sign errors live.
    "team_code": pl.Int64(),  # stable FPL club identity, never the per-season id
    "opponent_team_code": pl.Int64(),
    "season": pl.Utf8(),
    "division": pl.Utf8(),  # E0 = Premier League, E1 = Championship. Never pool blindly.
    "kickoff_time": _UTC,
    "was_home": pl.Boolean(),
    "goals_for": pl.Int64(),
    "goals_against": pl.Int64(),
    # The reason this table exists. The official API publishes no team-level
    # event counts at all, so shot volume, shot quality proxy (on-target share)
    # and set-piece volume are unreachable without an outside source.
    "shots": pl.Int64(),
    "shots_against": pl.Int64(),
    "shots_on_target": pl.Int64(),
    "shots_on_target_against": pl.Int64(),
    "corners": pl.Int64(),
    "corners_against": pl.Int64(),
    "fouls_committed": pl.Int64(),
    "fouls_suffered": pl.Int64(),
    "yellow_cards": pl.Int64(),
    "red_cards": pl.Int64(),
    "available_time": _UTC,
}

#: Table name to schema. The normalization run replaces each of these wholesale.
SILVER_TABLES: Final[dict[str, dict[str, pl.DataType]]] = {
    "players": PLAYERS_SCHEMA,
    "teams": TEAMS_SCHEMA,
    "gameweeks": GAMEWEEKS_SCHEMA,
    "fixtures": FIXTURES_SCHEMA,
    "player_gameweek_stats": PLAYER_GAMEWEEK_STATS_SCHEMA,
    "team_match_events": TEAM_MATCH_EVENTS_SCHEMA,
}


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """An empty frame with the right columns and dtypes.

    Returned when a source has no rows, so downstream code sees a consistent
    shape instead of having to branch on emptiness.
    """
    return pl.DataFrame(schema=schema)


def conform(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Cast a frame to a canonical schema, adding missing columns as nulls.

    Missing columns become null rather than raising, because source coverage
    genuinely varies by season — ``defensive_contribution`` does not exist
    before 2025/26. Unexpected *extra* columns are dropped: silver is a
    contract, and letting arbitrary upstream fields through would make every
    downstream consumer depend on whatever the API happened to return.
    """
    if frame.is_empty():
        return empty_frame(schema)

    expressions = []
    for name, dtype in schema.items():
        if name in frame.columns:
            expressions.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            expressions.append(pl.lit(None, dtype=dtype).alias(name))
    return frame.select(expressions)
