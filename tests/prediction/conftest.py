"""Shared fixtures for the prediction tests.

The synthetic history lives here rather than in a test module so both the
training and inference suites can use it without importing each other.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

#: Tiny models. These tests verify plumbing and invariants, not accuracy, and a
#: full-size fit per test makes the suite unusable.
FAST: dict[str, int] = {"max_iter": 15, "max_depth": 3}

T0 = datetime(2025, 8, 1, 12, 0, tzinfo=UTC)

SOURCE_COLUMNS = (
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "yellow_cards",
    "bonus",
    "bps",
    "total_points",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "influence",
    "creativity",
    "threat",
    "ict_index",
)


def synthetic_stats(
    *, players: int = 40, gameweeks: int = 24, season: str = "2025-26"
) -> pl.DataFrame:
    """Synthetic history where output rises with a player's latent quality."""
    rows = []
    for player in range(1, players + 1):
        quality = player / players
        for week in range(1, gameweeks + 1):
            kickoff = T0 + timedelta(days=7 * week)
            rows.append(
                {
                    "player_code": player,
                    "season": season,
                    "gameweek_id": week,
                    # Rotate opponents so the fixture varies across gameweeks,
                    # which the opponent features need in order to mean anything.
                    "opponent_team_id": ((week + player) % 6) + 1,
                    "was_home": (week + player) % 2 == 0,
                    "kickoff_time": kickoff,
                    "available_time": kickoff + timedelta(hours=3),
                    "minutes": 20 + quality * 70,
                    "starts": 1.0 if quality > 0.4 else 0.0,
                    "goals_scored": quality * 1.2,
                    "assists": quality * 0.7,
                    "clean_sheets": 1.0 if (week + player) % 3 == 0 else 0.0,
                    "goals_conceded": 2.0 - quality,
                    "saves": quality * 2.0,
                    "yellow_cards": (week % 5 == 0) * 1.0,
                    "bonus": quality * 1.5,
                    "bps": quality * 30,
                    "total_points": 2 + quality * 8,
                    "expected_goals": quality * 0.9,
                    "expected_assists": quality * 0.5,
                    "expected_goal_involvements": quality * 1.4,
                    "expected_goals_conceded": 1.8 - quality,
                    "selected": 100000 * quality + week,
                    "transfers_in": 5000 * quality + week * 3,
                    "transfers_out": 3000 * (1 - quality) + week * 2,
                    "transfers_balance": 2000 * quality - week,
                    "influence": quality * 40 + week * 0.3,
                    "creativity": quality * 30 + week * 0.2,
                    "threat": quality * 50 + week * 0.4,
                    "ict_index": quality * 12 + week * 0.1,
                }
            )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return frame.with_columns(
        pl.col("kickoff_time").dt.replace_time_zone("UTC"),
        pl.col("available_time").dt.replace_time_zone("UTC"),
        *[pl.col(c).cast(pl.Float64) for c in SOURCE_COLUMNS],
    )
