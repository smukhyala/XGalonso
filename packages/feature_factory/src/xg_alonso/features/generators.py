"""As-of aggregate generators.

A rolling feature is harder to get right than an as-of *join*, because it
aggregates many historical rows rather than picking one. The window must be
recomputed from each entity's own vantage point: "this player's last 5
appearances **as known at this prediction timestamp**", not "this player's last
5 appearances in the dataset".

The distinction is the whole ballgame. Computing the window over the full
dataset is the single most common leak in fantasy-sports modelling, and it is
invisible in every metric except live performance. Everything here recomputes
the window per entity row, and the leakage harness proves it.

Windows are ordered by ``available_time``, not by gameweek number. A postponed
fixture played out of order must land where it actually became known, and
gameweek ordering would silently place it too early.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

__all__ = [
    "ROW_ID",
    "rolling_as_of",
    "shrunk_rate_as_of",
    "stage_window",
]

ROW_ID = "__xg_entity_row"


def stage_window(
    entities: pl.DataFrame,
    source: pl.DataFrame,
    *,
    entity_keys: Sequence[str],
    prediction_time_col: str,
    available_time_col: str,
    window: int,
    order_col: str | None,
) -> pl.DataFrame:
    """Attach each entity row's visible history, newest first, capped at ``window``.

    This is the expensive step — a join plus a rank over every entity row's
    visible past. Exposed publicly so a caller building many features over the
    *same* window can stage once and aggregate many times, rather than paying
    for the join per feature.
    """
    keys = list(entity_keys)
    missing = [c for c in [*keys, prediction_time_col] if c not in entities.columns]
    if missing:
        raise KeyError(f"entities is missing required columns: {missing}")
    missing = [c for c in [*keys, available_time_col] if c not in source.columns]
    if missing:
        raise KeyError(f"source is missing required columns: {missing}")
    if window < 1:
        raise ValueError("window must be at least 1")

    ordering = order_col or available_time_col
    if ordering not in source.columns:
        raise KeyError(f"source is missing ordering column {ordering!r}")

    staged = entities.with_row_index(ROW_ID)

    # Only history that was publicly usable at this row's prediction time.
    joined = staged.join(source, on=keys, how="left").filter(
        pl.col(available_time_col).is_null()
        | (pl.col(available_time_col) <= pl.col(prediction_time_col))
    )

    # Rank newest-first within each entity row and keep the window. Ties break on
    # available_time then the ordering column so the result does not depend on
    # how the source frame happened to be assembled.
    ranked = joined.sort(
        [ROW_ID, available_time_col, ordering], descending=[False, True, True], nulls_last=True
    ).with_columns(pl.int_range(pl.len()).over(ROW_ID).alias("__xg_rank"))

    return ranked.filter(pl.col("__xg_rank") < window)


def rolling_as_of(
    entities: pl.DataFrame,
    source: pl.DataFrame,
    *,
    entity_keys: Sequence[str],
    value_column: str,
    window: int,
    aggregation: str = "mean",
    output_name: str | None = None,
    prediction_time_col: str = "prediction_timestamp",
    available_time_col: str = "available_time",
    order_col: str | None = None,
    min_periods: int = 1,
) -> pl.DataFrame:
    """Aggregate a player's most recent ``window`` visible records.

    Args:
        entities: Prediction rows carrying the entity keys and a cutoff.
        source: Historical records carrying ``available_time``.
        entity_keys: Columns identifying the entity, e.g. ``["player_code"]``.
        value_column: The column to aggregate.
        window: How many recent records to include.
        aggregation: ``mean``, ``sum``, ``median``, ``min``, ``max``, ``std`` or ``count``.
        output_name: Feature column name. Defaults to a descriptive name.
        prediction_time_col: Cutoff column on ``entities``.
        available_time_col: Publication-time column on ``source``.
        order_col: Tie-breaking order within equal availability times.
        min_periods: Below this many observations the result is null rather than
            a number computed from almost nothing. A "mean" over one appearance
            is noise wearing the costume of a statistic.

    Returns:
        ``entities`` plus one feature column, in the original row order.
    """
    name = output_name or f"{value_column}_{aggregation}_{window}"
    windowed = stage_window(
        entities,
        source,
        entity_keys=entity_keys,
        prediction_time_col=prediction_time_col,
        available_time_col=available_time_col,
        window=window,
        order_col=order_col,
    )

    aggregations = {
        "mean": pl.col(value_column).mean(),
        "sum": pl.col(value_column).sum(),
        "median": pl.col(value_column).median(),
        "min": pl.col(value_column).min(),
        "max": pl.col(value_column).max(),
        "std": pl.col(value_column).std(),
        "count": pl.col(value_column).count(),
    }
    if aggregation not in aggregations:
        raise ValueError(
            f"unsupported aggregation {aggregation!r}; expected one of {sorted(aggregations)}"
        )

    summary = windowed.group_by(ROW_ID).agg(
        aggregations[aggregation].alias(name),
        pl.col(value_column).count().alias("__xg_n"),
    )
    summary = summary.with_columns(
        pl.when(pl.col("__xg_n") >= min_periods).then(pl.col(name)).otherwise(None).alias(name)
    ).drop("__xg_n")

    return (
        entities.with_row_index(ROW_ID)
        .join(summary, on=ROW_ID, how="left")
        .sort(ROW_ID)
        .drop(ROW_ID)
    )


def shrunk_rate_as_of(
    entities: pl.DataFrame,
    source: pl.DataFrame,
    *,
    entity_keys: Sequence[str],
    numerator: str,
    denominator: str,
    window: int,
    prior_strength: float,
    output_name: str,
    prior_value: float | None = None,
    denominator_scale: float = 90.0,
    prediction_time_col: str = "prediction_timestamp",
    available_time_col: str = "available_time",
    order_col: str | None = None,
) -> pl.DataFrame:
    """A per-90 rate, shrunk toward a population prior by sample size.

    Raw per-90 rates are useless at low minutes: a substitute who scores in his
    only 12 minutes reads as 7.5 goals per 90. Empirical-Bayes shrinkage pulls
    thin samples toward the population mean and leaves rich ones alone::

        posterior = (observed_total + prior_strength * prior_rate * scale)
                    / (observed_denominator + prior_strength * scale)

    With no observations the result is exactly the prior; as the denominator
    grows the prior's influence vanishes. Both properties are asserted in the
    tests, because a shrinkage function that silently stops shrinking looks
    perfectly healthy right up until it recommends the substitute.

    Args:
        prior_strength: Prior weight, expressed in units of ``denominator_scale``
            (so ``3.0`` means the prior is worth three full matches).
        prior_value: Population rate per ``denominator_scale``. Computed from the
            visible source when omitted.
        denominator_scale: Usually 90, for per-90 rates.
    """
    if prior_strength < 0:
        raise ValueError("prior_strength must not be negative")

    windowed = stage_window(
        entities,
        source,
        entity_keys=entity_keys,
        prediction_time_col=prediction_time_col,
        available_time_col=available_time_col,
        window=window,
        order_col=order_col,
    )

    if prior_value is None:
        # The pooled prior is computed from already-filtered rows, so it cannot
        # see past any single entity's own cutoff. It CAN, however, see past an
        # *earlier* entity's cutoff when the frame mixes several — entity A at
        # gameweek 3 would borrow strength from entity B's gameweek 30 window.
        # That is a real cross-row leak and it is invisible to a harness whose
        # entities share one cutoff, so refuse the case outright rather than
        # letting a backfill quietly produce it.
        distinct_cutoffs = entities[prediction_time_col].n_unique()
        if distinct_cutoffs > 1:
            raise ValueError(
                f"entities span {distinct_cutoffs} distinct prediction timestamps, so a "
                "pooled empirical prior would leak later rows' data into earlier rows. "
                "Pass an explicit prior_value, or build one cutoff at a time."
            )
        total_num = windowed[numerator].sum()
        total_den = windowed[denominator].sum()
        prior_value = (
            float(total_num) / float(total_den) * denominator_scale
            if total_den and float(total_den) > 0
            else 0.0
        )

    prior_mass = prior_strength * denominator_scale

    summary = windowed.group_by(ROW_ID).agg(
        pl.col(numerator).sum().alias("__xg_num"),
        pl.col(denominator).sum().alias("__xg_den"),
    )
    summary = summary.with_columns(
        (
            (pl.col("__xg_num").fill_null(0) + prior_value * prior_strength)
            / (pl.col("__xg_den").fill_null(0) + prior_mass)
            * denominator_scale
        ).alias(output_name)
    ).drop("__xg_num", "__xg_den")

    joined = (
        entities.with_row_index(ROW_ID)
        .join(summary, on=ROW_ID, how="left")
        .sort(ROW_ID)
        .drop(ROW_ID)
    )
    # Entities with no visible history at all still get the prior, so cold-start
    # players are ranked rather than dropped.
    return joined.with_columns(pl.col(output_name).fill_null(prior_value))
