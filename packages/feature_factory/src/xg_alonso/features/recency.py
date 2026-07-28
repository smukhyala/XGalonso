"""How stale a player's recent form is, made visible to the model.

**The failure this fixes.** At gameweek 1 of 2026/27 the model projected Erling
Haaland to play 27 minutes with a 27% chance of starting, after four seasons
averaging 2,750 minutes. The cause was not the model and not the data: his most
recent row genuinely was a zero, because he was rested for a dead rubber on the
final day, and ``minutes_mean_1`` — the single most important feature in the
whole catalogue by measured importance — reads exactly that one match.

In-season that feature is excellent. If a player did not play last week, he
probably is not starting this week. Across a season boundary it inverts: the
"last match" is three months old, from a fixture whose result did not matter,
and it says nothing at all about the coming season.

**Why the model could not have known.** Every rolling window is counted in
matches, so a gap of three days and a gap of three months look identical inside
it. The staleness was not merely unlearned, it was *unrepresentable*. Worse,
``build_training_frame`` skips the opening gameweeks by default, so the model
had never seen a season-opener row and was extrapolating across a regime it had
no examples of.

These features make the gap a number. A model given ``days_since_last_match``
can learn what every manager already knows: when the last match was eighty days
ago, believe the twenty-match average and not the one-match one.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = ["RECENCY_FEATURES", "build_recency_features"]

#: Days without a match beyond which recent-form windows describe a different
#: regime. Six weeks: longer than any in-season break including a winter pause,
#: shorter than a summer.
_STALE_DAYS: Final[float] = 42.0

#: Appearances averaged over to answer "how long does he last when he plays".
_APPEARANCE_WINDOWS: Final[tuple[int, ...]] = (3, 5, 10)

RECENCY_FEATURES: Final[tuple[str, ...]] = (
    "days_since_last_match",
    "matches_last_30_days",
    "matches_last_90_days",
    "form_is_stale",
    "minutes_last_30_days",
    "days_since_last_appearance",
    *(f"minutes_when_played_mean_{window}" for window in _APPEARANCE_WINDOWS),
    "appearance_rate_10",
)


def build_recency_features(
    entities: pl.DataFrame,
    *,
    player_stats: pl.DataFrame,
    prediction_time_col: str = "prediction_timestamp",
) -> pl.DataFrame:
    """Attach recency and staleness features to a frame of prediction rows.

    Args:
        entities: One row per prediction, carrying ``player_code`` and a cutoff.
        player_stats: Canonical ``player_gameweek_stats``.
        prediction_time_col: The cutoff column.

    Returns:
        ``entities`` plus one column per recency feature, in the original order.

    Point-in-time safe by the same rule as everything else: only rows whose
    ``available_time`` precedes the cutoff are visible, so the gap is measured
    against matches that had actually been played.
    """
    required = {"player_code", "kickoff_time", "available_time", "minutes"}
    missing = sorted(required - set(player_stats.columns))
    if missing:
        raise KeyError(f"player_stats is missing columns required for recency: {missing}")

    frame = entities.with_row_index("__recency_row")
    joined = frame.select(["__recency_row", "player_code", prediction_time_col]).join(
        player_stats.select(["player_code", "kickoff_time", "available_time", "minutes"]),
        on="player_code",
        how="left",
    )

    visible = joined.filter(
        pl.col("available_time").is_not_null()
        & (pl.col("available_time") < pl.col(prediction_time_col))
    ).with_columns(
        ((pl.col(prediction_time_col) - pl.col("kickoff_time")).dt.total_seconds() / 86400.0).alias(
            "__age_days"
        )
    )

    summary = visible.group_by("__recency_row").agg(
        pl.col("__age_days").min().alias("days_since_last_match"),
        (pl.col("__age_days") <= 30.0).sum().cast(pl.Float64).alias("matches_last_30_days"),
        (pl.col("__age_days") <= 90.0).sum().cast(pl.Float64).alias("matches_last_90_days"),
        pl.col("minutes")
        .filter(pl.col("__age_days") <= 30.0)
        .sum()
        .cast(pl.Float64)
        .alias("minutes_last_30_days"),
    )

    summary = summary.with_columns(
        (pl.col("days_since_last_match") >= _STALE_DAYS).cast(pl.Float64).alias("form_is_stale")
    )

    # --- appearances, as distinct from fixtures ---------------------------
    #
    # Every catalogue window counts *fixtures his club played*, so a match he
    # was rested for enters the average as a zero and is indistinguishable from
    # one he was dropped from. For a rotation risk that is the right reading.
    # For an established starter rested in a dead rubber it is exactly wrong,
    # and it was wrong by a lot: a striker with four seasons above 2,700 minutes
    # projected at 38 because two of his last five club fixtures — a double
    # gameweek second leg and the final day — were zeros.
    #
    # Splitting the question fixes it without discarding anything. "How long
    # does he last when he plays" and "how often does he play" are two facts,
    # and a single fixture-based mean silently multiplies them together.
    played = visible.filter(pl.col("minutes") > 0)
    ranked = played.with_columns(
        pl.col("__age_days").rank("ordinal").over("__recency_row").alias("__recency")
    )
    appearance = ranked.group_by("__recency_row").agg(
        pl.col("__age_days").min().alias("days_since_last_appearance"),
        *(
            pl.col("minutes")
            .filter(pl.col("__recency") <= window)
            .mean()
            .alias(f"minutes_when_played_mean_{window}")
            for window in _APPEARANCE_WINDOWS
        ),
    )
    summary = summary.join(appearance, on="__recency_row", how="left")

    # How often he features at all, over the same span the windows cover.
    recent_ten = visible.with_columns(
        pl.col("__age_days").rank("ordinal").over("__recency_row").alias("__recency")
    ).filter(pl.col("__recency") <= 10)
    rate = recent_ten.group_by("__recency_row").agg(
        (pl.col("minutes") > 0).mean().cast(pl.Float64).alias("appearance_rate_10")
    )
    summary = summary.join(rate, on="__recency_row", how="left")

    frame = frame.join(summary, on="__recency_row", how="left")
    frame = frame.with_columns(
        # A player with no visible history has infinitely stale form, not fresh
        # form. Filling the gap with zero would say he played yesterday, which is
        # the opposite of what an empty history means.
        pl.col("days_since_last_match").fill_null(999.0),
        pl.col("matches_last_30_days").fill_null(0.0),
        pl.col("matches_last_90_days").fill_null(0.0),
        pl.col("minutes_last_30_days").fill_null(0.0),
        pl.col("form_is_stale").fill_null(1.0),
        pl.col("appearance_rate_10").fill_null(0.0),
        # A player who has never appeared has no "minutes when played". Left
        # null rather than zeroed: zero would assert he plays no minutes when he
        # plays, which is not a statement the absence of data supports.
        pl.col("days_since_last_appearance").fill_null(999.0),
    )
    return frame.sort("__recency_row").drop("__recency_row")
