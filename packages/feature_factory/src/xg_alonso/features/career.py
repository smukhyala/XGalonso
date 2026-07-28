"""Career-length features: what a player has proven, not merely what he just did.

**The failure this fixes.** Every window in the catalogue is measured in
appearances, and the longest is twenty — about half a season. Inside that
window a player who has scored thirty league goals a season for four years and
a player who had one exceptional season look identical, because the evidence
that separates them is entirely outside the window. The model could not tell
Haaland from a one-season wonder, and neither could any explanation built on it.

**The statistical shape of the problem.** This is not a missing feature so much
as a missing *sample size*. One elite season is one observation of a player's
level; four elite seasons is four. A mean over one season and a mean over four
are not the same estimate even when they are the same number, and treating them
as such is what makes a breakout look like a certainty.

So the rate features here are shrunk toward the population mean **in proportion
to how many seasons of evidence exist**:

    shrunk = (observed x seasons + population x prior) / (seasons + prior)

A player with four seasons keeps almost all of his own rate. A player with one
keeps roughly half of it, with the rest pulled to average — which is exactly the
statement "we have seen this once and it might not repeat". Nothing here
predicts that a breakout will regress; it declines to assert that it will not.

**Point-in-time safety.** Career aggregates read only rows whose
``available_time`` precedes the prediction timestamp, the same rule the rest of
the factory obeys, so a career figure never contains a season that had not
finished being played.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

__all__ = [
    "CAREER_FEATURES",
    "CAREER_VERSION",
    "CareerMetric",
    "build_career_features",
]

CAREER_VERSION: Final[str] = "career_v1"

#: Seasons of evidence that a player's own rate is worth as much as the prior.
#:
#: Two. With one season observed the estimate sits halfway between the player
#: and the population; by three it is mostly his own. Chosen rather than fitted
#: because the quantity being expressed is an editorial judgement about how much
#: one season proves, and a fitted value would imply a precision the four
#: seasons of history available cannot support.
_PRIOR_SEASONS: Final[float] = 2.0

#: Minutes in a season before it counts as a season of evidence. Roughly ten
#: full matches — below that a player was injured, loaned out or a squad
#: filler, and folding it in as a season would punish him for not playing.
_SEASON_MINUTES_FLOOR: Final[int] = 900


@dataclass(frozen=True)
class CareerMetric:
    """One statistic tracked across seasons."""

    source: str
    name: str
    per_90: bool = True


#: What is worth knowing across seasons. Deliberately short: career features
#: answer "has he done this before", and that question needs output and
#: availability, not the full catalogue replayed at a longer window.
_METRICS: Final[tuple[CareerMetric, ...]] = (
    CareerMetric(source="goals_scored", name="goals"),
    CareerMetric(source="assists", name="assists"),
    CareerMetric(source="expected_goal_involvements", name="xgi"),
    CareerMetric(source="total_points", name="points"),
    CareerMetric(source="bps", name="bps"),
    CareerMetric(source="minutes", name="minutes", per_90=False),
)

CAREER_FEATURES: Final[tuple[str, ...]] = (
    "career_seasons",
    "career_minutes_total",
    "career_minutes_per_season",
    *(f"career_{metric.name}_per90" for metric in _METRICS if metric.per_90),
    *(f"career_{metric.name}_per90_shrunk" for metric in _METRICS if metric.per_90),
    *(f"career_{metric.name}_season_std" for metric in _METRICS if metric.per_90),
    "career_best_season_points",
    "career_points_consistency",
    "career_is_proven",
)


def _season_totals(player_stats: pl.DataFrame) -> pl.DataFrame:
    """Per player per season, from rows available before each cutoff.

    Aggregated to season level first so that a season counts once however many
    matches it contained. Summing raw matches instead would let a player with
    one long season outweigh another with three short ones, which is the
    opposite of what "how many seasons has he done it" means.
    """
    aggregations = [pl.col("minutes").sum().alias("season_minutes")]
    for metric in _METRICS:
        if metric.source != "minutes":
            aggregations.append(pl.col(metric.source).sum().alias(f"season_{metric.name}"))
    aggregations.append(pl.col("available_time").max().alias("season_available_time"))

    return player_stats.group_by(["player_code", "season"]).agg(aggregations)


def build_career_features(
    entities: pl.DataFrame,
    *,
    player_stats: pl.DataFrame,
    prediction_time_col: str = "prediction_timestamp",
    prior_seasons: float = _PRIOR_SEASONS,
) -> pl.DataFrame:
    """Attach career aggregates to a frame of prediction rows.

    Args:
        entities: One row per prediction, carrying ``player_code`` and a cutoff.
        player_stats: Canonical ``player_gameweek_stats``.
        prediction_time_col: The cutoff column.
        prior_seasons: Strength of the shrinkage prior, in seasons.

    Returns:
        ``entities`` plus one column per career feature, in the original order.
    """
    required = {"player_code", "season", "minutes", "available_time"}
    missing = sorted(required - set(player_stats.columns))
    if missing:
        raise KeyError(f"player_stats is missing columns required for career features: {missing}")

    seasons = _season_totals(player_stats).filter(pl.col("season_minutes") >= _SEASON_MINUTES_FLOOR)

    frame = entities.with_row_index("__career_row")
    joined = frame.select(["__career_row", "player_code", prediction_time_col]).join(
        seasons, on="player_code", how="left"
    )

    # Point-in-time: a season only counts once every match in it was available.
    visible = joined.filter(
        pl.col("season_available_time").is_not_null()
        & (pl.col("season_available_time") < pl.col(prediction_time_col))
    )

    per90_metrics = [metric for metric in _METRICS if metric.per_90]
    aggregations: list[pl.Expr] = [
        pl.len().alias("career_seasons"),
        pl.col("season_minutes").sum().alias("career_minutes_total"),
        pl.col("season_minutes").mean().alias("career_minutes_per_season"),
        pl.col("season_points").max().alias("career_best_season_points"),
    ]
    for metric in per90_metrics:
        # Career rate is the pooled total over pooled minutes, not the mean of
        # per-season rates: a season of 200 minutes should not weigh the same as
        # a season of 3000.
        aggregations.append(pl.col(f"season_{metric.name}").sum().alias(f"__total_{metric.name}"))
        aggregations.append(
            (pl.col(f"season_{metric.name}") / pl.col("season_minutes") * 90.0)
            .std()
            .alias(f"career_{metric.name}_season_std")
        )

    summary = visible.group_by("__career_row").agg(aggregations)

    rates: list[pl.Expr] = []
    for metric in per90_metrics:
        rate = pl.col(f"__total_{metric.name}") / pl.col("career_minutes_total") * 90.0
        rates.append(rate.alias(f"career_{metric.name}_per90"))
    summary = summary.with_columns(rates)

    # Shrink each rate toward the population mean by seasons observed. Computed
    # from this batch, which is the same population the percentiles elsewhere
    # rank against.
    # The population mean stays a Polars expression rather than being pulled
    # into Python. Broadcasting it keeps the whole shrinkage in one pass, and it
    # avoids converting a scalar whose type the frame does not promise.
    shrunk: list[pl.Expr] = []
    weight = pl.col("career_seasons").cast(pl.Float64)
    for metric in per90_metrics:
        column = f"career_{metric.name}_per90"
        centre = pl.col(column).mean().fill_null(0.0)
        shrunk.append(
            (
                (pl.col(column).fill_null(centre) * weight + centre * prior_seasons)
                / (weight + prior_seasons)
            ).alias(f"{column}_shrunk")
        )
    summary = summary.with_columns(shrunk)

    summary = summary.with_columns(
        # Consistency: season-to-season spread relative to the level itself, so
        # it is comparable between a striker and a defender. Inverted so that
        # larger is steadier, which matches every other feature's orientation.
        #
        # **Null below two seasons, not 1.0.** Filling the missing standard
        # deviation with zero made a single-season player score as perfectly
        # consistent — the most flattering possible reading of the least
        # evidence, and precisely the "one-season wonder looks proven" failure
        # these features exist to remove. Variance across seasons is undefined
        # with one season, so it is reported as unknown.
        pl.when(pl.col("career_seasons") >= 2)
        .then(
            1.0
            / (
                1.0
                + pl.col("career_points_season_std").fill_null(0.0)
                / (pl.col("career_points_per90").abs() + 1e-6)
            )
        )
        .otherwise(None)
        .alias("career_points_consistency"),
        # "Proven" is a blunt flag on purpose: enough seasons that the shrinkage
        # is barely doing anything, which is the honest threshold for treating a
        # player's own rate as his level.
        (pl.col("career_seasons") >= 3).cast(pl.Float64).alias("career_is_proven"),
    )

    summary = summary.select(["__career_row", *CAREER_FEATURES])
    frame = frame.join(summary, on="__career_row", how="left")

    # A player with no visible history has no career, which is different from a
    # career of zero. Seasons and minutes are genuinely zero; rates stay null so
    # the model branches on "unknown" rather than reading a confident nothing.
    frame = frame.with_columns(
        pl.col("career_seasons").fill_null(0),
        pl.col("career_minutes_total").fill_null(0),
        pl.col("career_is_proven").fill_null(0.0),
    )
    return frame.sort("__career_row").drop("__career_row")
