"""Opponent-strength features.

The catalogue up to now described only the player. That is a real gap: a
defender's clean-sheet chance is mostly a property of who his team is playing,
and the evidence says so — ``clean_sheets`` is by far the weakest trained label
(+14% skill against +50% for attacking returns), and the trained model's worst
transfer decisions are all defenders.

**Deriving opponent strength without a team-id join.** Every player-gameweek row
already carries ``opponent_team_id``. So what a team concedes in a gameweek is
simply the sum of what the players facing them produced::

    conceded_xg[team, gw] = Σ expected_goals over rows where opponent_team_id == team

That inverts the table rather than joining it, which avoids depending on a
per-season club identity the stats table does not carry.

These features are point-in-time safe for the same reason the rest are: they go
through the same as-of staging, so a team's record is only ever read as of the
prediction cutoff. The *fixture* itself — who a player faces — is legitimately
known before the deadline, since FPL publishes fixtures well in advance.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from xg_alonso.features.generators import ROW_ID, stage_window

__all__ = [
    "OPPONENT_FEATURES",
    "build_opponent_features",
    "build_opponent_strength",
]

#: Windows over which a team's recent record is summarised.
_OPPONENT_WINDOWS: Final[tuple[int, ...]] = (3, 5, 10)

#: What a team's defensive record is summarised by.
_OPPONENT_METRICS: Final[tuple[str, ...]] = (
    "conceded_xg",
    "conceded_goals",
    "conceded_points",
    "clean_sheets_against",
)

OPPONENT_FEATURES: Final[tuple[str, ...]] = (
    *(
        f"opponent_{metric}_mean_{window}"
        for metric in _OPPONENT_METRICS
        for window in _OPPONENT_WINDOWS
    ),
    "is_home",
)


def build_opponent_strength(player_stats: pl.DataFrame) -> pl.DataFrame:
    """What each team conceded, per gameweek, derived by inverting the stats table.

    Returns one row per ``(season, gameweek, team)`` describing how much the
    opposition produced against that team — the defensive record a fixture
    feature needs.
    """
    required = {"opponent_team_id", "season", "gameweek_id", "available_time"}
    missing = sorted(required - set(player_stats.columns))
    if missing:
        raise KeyError(f"player_stats is missing columns needed for opponent strength: {missing}")

    if player_stats.is_empty():
        return pl.DataFrame(
            schema={
                "opponent_team_id": pl.Int64(),
                "season": pl.Utf8(),
                "gameweek_id": pl.Int64(),
                "conceded_xg": pl.Float64(),
                "conceded_goals": pl.Float64(),
                "conceded_points": pl.Float64(),
                "clean_sheets_against": pl.Float64(),
                "available_time": player_stats.schema.get(
                    "available_time", pl.Datetime(time_unit="us", time_zone="UTC")
                ),
                "kickoff_time": player_stats.schema.get(
                    "kickoff_time", pl.Datetime(time_unit="us", time_zone="UTC")
                ),
            }
        )

    return (
        player_stats.filter(pl.col("opponent_team_id").is_not_null())
        .group_by(["opponent_team_id", "season", "gameweek_id"])
        .agg(
            pl.col("expected_goals").sum().alias("conceded_xg"),
            pl.col("goals_scored").sum().alias("conceded_goals"),
            pl.col("total_points").sum().alias("conceded_points"),
            # How often the opposition kept a clean sheet against this team —
            # i.e. how rarely this team scores.
            pl.col("clean_sheets").max().alias("clean_sheets_against"),
            pl.col("available_time").max().alias("available_time"),
            pl.col("kickoff_time").max().alias("kickoff_time"),
        )
        .sort(["opponent_team_id", "season", "gameweek_id"])
    )


def build_opponent_features(
    entities: pl.DataFrame,
    *,
    opponent_strength: pl.DataFrame,
    prediction_time_col: str = "prediction_timestamp",
) -> pl.DataFrame:
    """Attach the upcoming opponent's recent defensive record.

    ``entities`` must carry ``opponent_team_id`` — who the player faces — and
    may carry ``was_home``. Rows without an opponent (a blank gameweek, or a
    fixture not yet assigned) get nulls rather than a league average, because
    "no fixture" and "an average fixture" are different situations and a model
    should be able to tell them apart.
    """
    if "opponent_team_id" not in entities.columns:
        raise KeyError(
            "entities must carry opponent_team_id. Who a player faces is known "
            "before the deadline, so it is an input, not a leak."
        )

    frame = entities.with_row_index(ROW_ID)

    if opponent_strength.is_empty():
        return frame.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(n) for n in OPPONENT_FEATURES if n != "is_home"]
        ).drop(ROW_ID)

    for window in _OPPONENT_WINDOWS:
        staged = stage_window(
            entities,
            opponent_strength,
            entity_keys=["opponent_team_id"],
            prediction_time_col=prediction_time_col,
            available_time_col="available_time",
            window=window,
            order_col="kickoff_time",
        )
        summary = staged.group_by(ROW_ID).agg(
            [pl.col(m).mean().alias(f"opponent_{m}_mean_{window}") for m in _OPPONENT_METRICS]
        )
        frame = frame.join(summary, on=ROW_ID, how="left")

    frame = frame.sort(ROW_ID).drop(ROW_ID)

    # Home advantage is small but real, and free to include.
    if "was_home" in frame.columns:
        frame = frame.with_columns(
            pl.col("was_home").cast(pl.Float64, strict=False).alias("is_home")
        )
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("is_home"))

    return frame
