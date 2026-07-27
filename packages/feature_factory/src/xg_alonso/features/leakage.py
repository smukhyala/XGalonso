"""The leakage harness.

A feature leaks when its value at time *T* changes once data that only became
available after *T* is added. That is exactly testable: build features, append
future records, rebuild, and compare. If anything moved, the feature saw the
future.

**The negative control matters as much as the test.** A leakage harness that
never fails is indistinguishable from one that is broken, and the broken version
is more dangerous than no harness at all because it manufactures confidence. So
:func:`assert_detects_leakage` runs a deliberately leaky builder through the
same machinery and asserts that it *is* caught. Every generator gets both.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

import polars as pl

__all__ = [
    "FeatureBuilder",
    "LeakageDetected",
    "assert_detects_leakage",
    "assert_no_leakage",
    "find_leakage",
]

FeatureBuilder = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]
"""``(entities, source) -> features``. Deterministic, with no hidden state."""


class LeakageDetected(AssertionError):  # noqa: N818 - reads as a test assertion, not an error
    """Raised when adding future-only records changes a past feature value."""


def find_leakage(
    builder: FeatureBuilder,
    *,
    entities: pl.DataFrame,
    source: pl.DataFrame,
    future_records: pl.DataFrame,
    compare_columns: Sequence[str] | None = None,
) -> list[str]:
    """Return the feature columns that future data changed. Empty means clean.

    Args:
        builder: Builds features from entities and source records.
        entities: Prediction rows, all with cutoffs before ``future_records``.
        source: Records available at prediction time.
        future_records: Records whose ``available_time`` is after every cutoff.
            Adding these must not change any output.
        compare_columns: Restrict the comparison to these columns.

    Returns:
        Names of columns whose values differ between the two builds.
    """
    if future_records.is_empty():
        raise ValueError(
            "future_records is empty, so this proves nothing. The harness needs "
            "records that would change the answer if they were wrongly visible."
        )

    before = builder(entities, source)
    after = builder(entities, pl.concat([source, future_records], how="vertical_relaxed"))

    if before.height != after.height:
        return [f"<row count changed: {before.height} -> {after.height}>"]

    columns = list(compare_columns) if compare_columns else list(before.columns)
    leaked: list[str] = []

    for column in columns:
        if column not in after.columns:
            leaked.append(f"<{column} missing after rebuild>")
            continue
        lhs, rhs = before[column], after[column]
        if not lhs.equals(rhs, null_equal=True, check_dtypes=False):
            leaked.append(column)

    return leaked


def assert_no_leakage(
    builder: FeatureBuilder,
    *,
    entities: pl.DataFrame,
    source: pl.DataFrame,
    future_records: pl.DataFrame,
    compare_columns: Sequence[str] | None = None,
) -> None:
    """Assert that future data does not change any past feature value.

    Raises:
        LeakageDetected: naming every column that moved.
    """
    leaked = find_leakage(
        builder,
        entities=entities,
        source=source,
        future_records=future_records,
        compare_columns=compare_columns,
    )
    if leaked:
        raise LeakageDetected(
            "adding records that were not yet available changed these feature "
            f"columns: {leaked}. A feature may only use information whose "
            "available_time precedes the prediction timestamp."
        )


def assert_detects_leakage(
    leaky_builder: FeatureBuilder,
    *,
    entities: pl.DataFrame,
    source: pl.DataFrame,
    future_records: pl.DataFrame,
) -> None:
    """Negative control: assert the harness catches a knowingly leaky builder.

    Without this, a harness that silently stopped comparing would keep passing
    and every generator would appear clean. Run it alongside the positive test.

    Raises:
        AssertionError: if the leaky builder is *not* caught, which means the
            harness has lost its power to detect anything.
    """
    leaked = find_leakage(
        leaky_builder, entities=entities, source=source, future_records=future_records
    )
    if not leaked:
        raise AssertionError(
            "the negative control passed the leakage harness. A builder that "
            "deliberately reads future data was reported as clean, so the harness "
            "is broken and every 'clean' result from it is meaningless."
        )


def make_future_records(
    source: pl.DataFrame,
    *,
    after: datetime,
    entity_keys: Sequence[str] = (),
    available_time_col: str = "available_time",
    rows: int = 3,
) -> pl.DataFrame:
    """Build plausible records stamped after a cutoff, for use as the future set.

    Values are perturbed rather than copied, so a builder that wrongly consumes
    them produces a visibly different answer.

    ``entity_keys`` must name the identifier columns. Perturbing those would
    describe *different* entities, and future records about players nobody is
    predicting cannot change anyone's features — the harness would then pass
    every builder, including a leaky one. Keys are held fixed so the synthetic
    records land on the entities under test.
    """
    if source.is_empty():
        raise ValueError("cannot synthesise future records from an empty source")
    if after.tzinfo is None:
        raise ValueError("after must be timezone-aware")

    missing = [k for k in entity_keys if k not in source.columns]
    if missing:
        raise KeyError(f"source is missing entity key columns: {missing}")

    protected = {available_time_col, *entity_keys}
    template = source.head(rows)
    numeric = [
        c
        for c, dtype in zip(template.columns, template.dtypes, strict=True)
        if dtype.is_numeric() and c not in protected
    ]
    if not numeric:
        raise ValueError(
            "no perturbable numeric columns outside the entity keys, so the future "
            "records would be indistinguishable from the originals and prove nothing"
        )

    return template.with_columns(
        pl.lit(after).cast(template.schema[available_time_col]).alias(available_time_col),
        *[(pl.col(c) * 10 + 999).alias(c) for c in numeric],
    )
