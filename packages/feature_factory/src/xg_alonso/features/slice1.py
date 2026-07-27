"""The slice-1 feature set.

Eight features, chosen against one hard constraint: **each must be non-null for
most players at gameweek 1 of a new season**. That rules out anything needing
current-season history, which at GW1 is empty. Each is therefore built from the
prior season's data, carried across the season boundary on ``player_code``.

Team strength is derived from expected goals rather than from the API's
``strength_attack_*`` and ``strength_defence_*`` fields. Those read 0 for all 20
clubs in preseason and switch numeric scale once the season starts, so a fixture
term built on them would be either constant or discontinuous. Deriving it from
xG costs one aggregation and works in both regimes.

The catalogue grows from here toward the 300-700 target, but every addition
earns its place by measurement, not by being cheap to compute.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from xg_alonso.features.generators import rolling_as_of, shrunk_rate_as_of

__all__ = [
    "SLICE1_FEATURES",
    "SLICE1_FEATURE_SET_VERSION",
    "build_slice1_features",
    "build_team_gameweek_stats",
]

SLICE1_FEATURE_SET_VERSION: Final[str] = "slice1_v1"

SLICE1_FEATURES: Final[tuple[str, ...]] = (
    "minutes_mean_5",
    "start_rate_5",
    "xg_per90_shrunk",
    "xa_per90_shrunk",
    "xgi_per90_shrunk",
    "points_per90_shrunk",
    "bonus_per90_shrunk",
    "team_xg_for_mean_5",
)

#: Prior weight in matches. Three full matches of prior is enough to tame a
#: 12-minute cameo without flattening a genuinely elite rate over a season.
_PRIOR_MATCHES: Final[float] = 3.0

#: Window for rate features. Ten appearances balances recency against the
#: sample size a per-90 rate needs to mean anything.
_RATE_WINDOW: Final[int] = 10


def build_team_gameweek_stats(player_stats: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Aggregate player rows into team-gameweek attacking and defensive totals.

    Derived from expected goals rather than the API's strength fields, which are
    unusable preseason. ``available_time`` is carried through as the *latest*
    contributing row: a team total is only knowable once its last player row is.

    **Grouping includes the season.** Gameweek numbers repeat every year, so
    grouping on ``(team_id, gameweek_id)`` alone silently merges GW1 of every
    backfilled season into one row — producing a "team xG" of roughly four
    matches' worth. The symptom is a plausible-looking number about four times
    too large, which is exactly the kind of error that survives review.
    """
    empty_schema = {
        "team_id": pl.Int64(),
        "season": pl.Utf8(),
        "gameweek_id": pl.Int64(),
        "team_xg": pl.Float64(),
        "team_xgc": pl.Float64(),
        "available_time": player_stats.schema.get(
            "available_time", pl.Datetime(time_unit="us", time_zone="UTC")
        ),
    }
    if player_stats.is_empty():
        return pl.DataFrame(schema=empty_schema)

    with_team = player_stats.join(
        players.select("player_code", "team_id"), on="player_code", how="inner"
    )
    if with_team.is_empty():
        return pl.DataFrame(schema=empty_schema)

    return (
        with_team.group_by(["team_id", "season", "gameweek_id"])
        .agg(
            pl.col("expected_goals").sum().alias("team_xg"),
            # Conceded is a team property already present on every player row,
            # so the max is the team value rather than a sum over the squad.
            pl.col("expected_goals_conceded").max().alias("team_xgc"),
            pl.col("available_time").max().alias("available_time"),
        )
        .sort(["team_id", "season", "gameweek_id"])
    )


def build_slice1_features(
    entities: pl.DataFrame,
    *,
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame | None = None,
    prediction_time_col: str = "prediction_timestamp",
) -> pl.DataFrame:
    """Build the slice-1 feature set for a frame of prediction rows.

    Args:
        entities: One row per prediction. Requires ``player_code`` and a cutoff;
            ``team_id`` is required for the team feature.
        player_stats: Canonical ``player_gameweek_stats``.
        team_stats: Output of :func:`build_team_gameweek_stats`. The team feature
            is skipped when absent.
        prediction_time_col: Cutoff column on ``entities``.

    Returns:
        ``entities`` plus the feature columns, in the original row order.
    """
    if "player_code" not in entities.columns:
        raise KeyError("entities must carry player_code")
    if prediction_time_col not in entities.columns:
        raise KeyError(f"entities must carry {prediction_time_col!r}")

    frame = entities

    # --- volume and role -------------------------------------------------
    frame = rolling_as_of(
        frame,
        player_stats,
        entity_keys=["player_code"],
        value_column="minutes",
        window=5,
        aggregation="mean",
        output_name="minutes_mean_5",
        prediction_time_col=prediction_time_col,
        order_col="kickoff_time",
        min_periods=1,
    )
    frame = rolling_as_of(
        frame,
        player_stats,
        entity_keys=["player_code"],
        value_column="starts",
        window=5,
        aggregation="mean",
        output_name="start_rate_5",
        prediction_time_col=prediction_time_col,
        order_col="kickoff_time",
        min_periods=1,
    )

    # --- shrunk per-90 rates ---------------------------------------------
    rates = (
        ("expected_goals", "xg_per90_shrunk"),
        ("expected_assists", "xa_per90_shrunk"),
        ("expected_goal_involvements", "xgi_per90_shrunk"),
        ("total_points", "points_per90_shrunk"),
        ("bonus", "bonus_per90_shrunk"),
    )
    for numerator, output in rates:
        frame = shrunk_rate_as_of(
            frame,
            player_stats,
            entity_keys=["player_code"],
            numerator=numerator,
            denominator="minutes",
            window=_RATE_WINDOW,
            prior_strength=_PRIOR_MATCHES,
            output_name=output,
            prediction_time_col=prediction_time_col,
            order_col="kickoff_time",
        )

    # --- team attacking strength, derived from xG ------------------------
    if team_stats is not None and not team_stats.is_empty() and "team_id" in frame.columns:
        frame = rolling_as_of(
            frame,
            team_stats,
            entity_keys=["team_id"],
            value_column="team_xg",
            window=5,
            aggregation="mean",
            output_name="team_xg_for_mean_5",
            prediction_time_col=prediction_time_col,
            order_col="gameweek_id",
            min_periods=1,
        )
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("team_xg_for_mean_5"))

    return frame
