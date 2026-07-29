"""Shared fixtures for the discovery tests.

Synthetic data with a *known* structure, so a test can assert an exact value
rather than "something plausible happened". The generators mirror the shape of
``player_gameweek_stats`` closely enough that a program written against the real
schema runs unchanged here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

T0 = datetime(2025, 8, 1, 12, 0, tzinfo=UTC)

#: Columns a program may read. Mirrors the real silver schema.
SOURCE_COLUMNS: tuple[str, ...] = (
    "player_code",
    "season",
    "gameweek_id",
    "opponent_team_id",
    "was_home",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "total_points",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "threat",
    "creativity",
    "influence",
    "ict_index",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "kickoff_time",
    "available_time",
)


def make_history(*, players: tuple[int, ...] = (1, 2, 3), rows: int = 12) -> pl.DataFrame:
    """Deterministic history, one row per player per gameweek.

    Values are scaled *by player* rather than offset by them. An additive offset
    leaves every standard deviation and every trend identical across players,
    which makes a test that should distinguish them pass for the wrong reason —
    a lesson ``tests/features/test_catalogue.py`` already records.
    """
    records: list[dict[str, object]] = []
    for player in players:
        for week in range(1, rows + 1):
            kickoff = T0 + timedelta(days=7 * week)
            base = float(player) * (1.0 + 0.25 * week)
            records.append(
                {
                    "player_code": player,
                    "season": "2025-26",
                    "gameweek_id": week,
                    "opponent_team_id": 1 + (week % 4),
                    "was_home": week % 2 == 0,
                    "minutes": min(90, 30 * player),
                    "starts": 1 if player > 1 else 0,
                    "goals_scored": week % (player + 1),
                    "assists": (week + player) % 3,
                    "clean_sheets": week % 2,
                    "goals_conceded": player,
                    "saves": player * 2,
                    "bonus": week % 3,
                    "bps": base * 3.0,
                    "total_points": base,
                    "expected_goals": base * 0.05,
                    "expected_assists": base * 0.03,
                    "expected_goal_involvements": base * 0.08,
                    "expected_goals_conceded": base * 0.04,
                    "threat": base * 10.0,
                    "creativity": base * 8.0,
                    "influence": base * 6.0,
                    "ict_index": base * 2.0,
                    "value": 50 + player * 5 + week,
                    "selected": 1000 * player + 10 * week,
                    "transfers_in": 100 * week,
                    "transfers_out": 50 * week,
                    "transfers_balance": 50 * week,
                    "kickoff_time": kickoff,
                    "available_time": kickoff + timedelta(hours=2),
                }
            )
    return pl.DataFrame(records)


def make_entities(
    *, players: tuple[int, ...] = (1, 2, 3), days: int = 200, positions: bool = True
) -> pl.DataFrame:
    """Prediction rows, all sharing one cutoff well after the history."""
    cutoff = T0 + timedelta(days=days)
    frame = pl.DataFrame(
        {
            "player_code": list(players),
            "opponent_team_id": [2] * len(players),
            "was_home": [True] * len(players),
            "prediction_timestamp": [cutoff] * len(players),
            "is_home": [1.0] * len(players),
            "opponent_conceded_xg_mean_5": [1.2] * len(players),
            "opponent_conceded_goals_mean_5": [1.1] * len(players),
        }
    )
    if positions:
        order = ("MID", "FWD", "DEF", "GKP")
        frame = frame.with_columns(
            pl.Series("position", [order[i % len(order)] for i in range(len(players))]),
            pl.Series("team_name", [f"Club {1 + i % 2}" for i in range(len(players))]),
        )
    return frame


@pytest.fixture
def history() -> pl.DataFrame:
    return make_history()


@pytest.fixture
def entities() -> pl.DataFrame:
    return make_entities()


@pytest.fixture
def training_frame() -> pl.DataFrame:
    """A small labelled frame with a *planted* signal.

    ``label_total_points`` depends on ``signal`` and not on ``noise``, so a
    harness that cannot tell them apart is broken. Twelve gameweeks across two
    seasons, enough for several walk-forward folds.
    """
    import numpy as np

    generator = np.random.default_rng(20260727)
    rows: list[dict[str, object]] = []
    for season_index, season in enumerate(("2024-25", "2025-26")):
        for week in range(1, 25):
            for player in range(1, 41):
                signal = float(generator.normal(2.0 + 0.05 * player, 1.0))
                noise = float(generator.normal(0.0, 1.0))
                rows.append(
                    {
                        "player_code": player,
                        "label_season": season,
                        "label_gameweek": week,
                        "position": ("GKP", "DEF", "MID", "FWD")[player % 4],
                        "anchor": signal * 0.5 + float(generator.normal(0, 0.5)),
                        "signal": signal,
                        "noise": noise,
                        "label_total_points": 2.0 * signal
                        + 0.3 * season_index
                        + float(generator.normal(0, 0.8)),
                    }
                )
    return pl.DataFrame(rows)
