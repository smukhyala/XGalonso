"""Point-in-time joins — the single highest-risk component in the platform.

The Feature Factory specification calls leakage "the most serious implementation
risk", and it is right. Leakage does not crash anything; it produces a model
that validates beautifully and loses money. Every guard here exists because the
failure it prevents is silent.

**The rule.** A feature computed for a prediction at time *T* may use a source
record only when ``available_time <= T``. Not ``event_time`` — a match that was
played is not the same as a match whose statistics had been published.

**Why ``maintain_order=True`` appears below.** Polars' ``unique()`` does not
guarantee row order without it, so ``keep="last"`` after a sort is not
deterministic. A tie-breaking bug of that shape passes locally by luck and fails
in CI — or worse, passes everywhere and silently picks a different record per
run, which would defeat rerun-equality without ever failing a test.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

__all__ = [
    "AVAILABLE_TIME",
    "PREDICTION_TIME",
    "as_of_join",
    "filter_available",
    "point_in_time_join",
]

AVAILABLE_TIME = "available_time"
PREDICTION_TIME = "prediction_timestamp"


def filter_available(
    source: pl.DataFrame,
    *,
    cutoff: datetime,
    available_time_col: str = AVAILABLE_TIME,
) -> pl.DataFrame:
    """Keep only rows that were publicly usable at ``cutoff``.

    The comparison is inclusive: a record that became available exactly at the
    prediction instant was available.
    """
    if available_time_col not in source.columns:
        raise KeyError(
            f"source is missing {available_time_col!r}; every ingested record must "
            "carry the four timestamps or point-in-time correctness cannot be checked"
        )
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return source.filter(pl.col(available_time_col) <= cutoff)


def as_of_join(
    entities: pl.DataFrame,
    source: pl.DataFrame,
    *,
    entity_keys: Sequence[str],
    prediction_time_col: str = PREDICTION_TIME,
    available_time_col: str = AVAILABLE_TIME,
    suffix: str = "_src",
) -> pl.DataFrame:
    """Attach each entity's most recent source record as of its prediction time.

    For every entity row this selects the single source row with the greatest
    ``available_time`` that does not exceed the entity's prediction timestamp.
    Entities with no qualifying record keep null source columns rather than
    being dropped — a missing observation is information, and silently losing
    the row would bias every downstream aggregate.

    Args:
        entities: Rows to enrich. Must contain ``entity_keys`` and the prediction
            time column.
        source: Timestamped source records carrying ``available_time``.
        entity_keys: Columns identifying the entity, e.g. ``["player_code"]``.
        prediction_time_col: Column on ``entities`` holding the cutoff.
        available_time_col: Column on ``source`` holding publication time.
        suffix: Applied to overlapping non-key column names.

    Returns:
        One row per input entity row, in the input's original order.
    """
    keys = list(entity_keys)
    _require_columns(entities, [*keys, prediction_time_col], "entities")
    _require_columns(source, [*keys, available_time_col], "source")

    _require_tz_aware(entities, prediction_time_col)
    _require_tz_aware(source, available_time_col)

    # Preserve the caller's row order across the join. join_asof requires sorted
    # inputs, so without an explicit marker the output order is the sorted order.
    order_col = "__xg_row_order"
    if order_col in entities.columns:
        raise ValueError(f"{order_col!r} is reserved for internal ordering")
    staged = entities.with_row_index(order_col)

    overlapping = (set(source.columns) & set(staged.columns)) - set(keys) - {order_col}
    prepared = source.rename({c: f"{c}{suffix}" for c in overlapping})
    source_time = (
        f"{available_time_col}{suffix}" if available_time_col in overlapping else available_time_col
    )

    # join_asof needs both frames sorted on the ordering column. Sorting by the
    # keys as well makes the result independent of input order.
    left = staged.sort([prediction_time_col, *keys])
    right = prepared.sort([source_time, *keys])

    joined = left.join_asof(
        right,
        left_on=prediction_time_col,
        right_on=source_time,
        by=keys,
        strategy="backward",  # most recent record at or before the cutoff
    )

    return joined.sort(order_col).drop(order_col)


def point_in_time_join(
    entities: pl.DataFrame,
    source: pl.DataFrame,
    *,
    entity_keys: Sequence[str],
    prediction_time_col: str = PREDICTION_TIME,
    available_time_col: str = AVAILABLE_TIME,
    value_columns: Sequence[str] | None = None,
    deduplicate: bool = True,
) -> pl.DataFrame:
    """Point-in-time-safe join, with deterministic tie-breaking.

    This is :func:`as_of_join` plus an explicit resolution for records sharing
    an ``available_time``. Ties are real — a single API snapshot stamps every
    row it contains with one publication instant — so leaving the choice to the
    engine would make feature values depend on input ordering.

    Args:
        entities: Rows to enrich.
        source: Timestamped source records.
        entity_keys: Columns identifying the entity.
        prediction_time_col: Cutoff column on ``entities``.
        available_time_col: Publication-time column on ``source``.
        value_columns: Restrict the joined columns to these, plus keys and time.
        deduplicate: Collapse ties to one record per (key, available_time).

    Returns:
        One row per input entity row, in the input's original order.
    """
    keys = list(entity_keys)
    prepared = source

    if value_columns is not None:
        keep = [*keys, available_time_col, *value_columns]
        missing = [c for c in keep if c not in prepared.columns]
        if missing:
            raise KeyError(f"source is missing requested columns: {missing}")
        prepared = prepared.select(keep)

    if deduplicate:
        # Ties must break on *content*, not on position. Sorting by the keys and
        # timestamp alone leaves tied rows in input order, so keep="last" would
        # pick whichever row happened to arrive last — making feature values
        # depend on how the source frame was assembled. Extending the sort to
        # every remaining column makes the winner a function of the row set
        # alone, so any permutation of the same records yields the same answer.
        #
        # maintain_order=True is still required: without it Polars does not
        # guarantee unique() respects the preceding sort at all.
        tie_breakers = [c for c in prepared.columns if c not in {*keys, available_time_col}]
        prepared = prepared.sort([*keys, available_time_col, *tie_breakers]).unique(
            subset=[*keys, available_time_col], keep="last", maintain_order=True
        )

    return as_of_join(
        entities,
        prepared,
        entity_keys=keys,
        prediction_time_col=prediction_time_col,
        available_time_col=available_time_col,
    )


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def _require_tz_aware(frame: pl.DataFrame, column: str) -> None:
    dtype = frame.schema[column]
    if not isinstance(dtype, pl.Datetime):
        raise TypeError(f"{column!r} must be a Datetime column, got {dtype}")
    if dtype.time_zone is None:
        raise TypeError(
            f"{column!r} must be timezone-aware. Naive timestamps compare "
            "incorrectly against UTC data and are a common source of leakage."
        )
