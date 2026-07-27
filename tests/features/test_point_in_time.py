"""Point-in-time join and leakage-harness tests.

Marked ``leakage`` so ``make leakage`` can run them alone. These are the tests
that stop a silently-wrong model from shipping, so they get their own gate.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from xg_alonso.features.leakage import (
    LeakageDetected,
    assert_detects_leakage,
    assert_no_leakage,
    find_leakage,
    make_future_records,
)
from xg_alonso.features.point_in_time import (
    as_of_join,
    filter_available,
    point_in_time_join,
)

T0 = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _entities() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_code": [1, 2, 1],
            "prediction_timestamp": [
                T0 + timedelta(days=3),
                T0 + timedelta(days=3),
                T0 + timedelta(days=10),
            ],
        }
    ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))


def _source() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_code": [1, 1, 2, 1],
            "available_time": [
                T0,
                T0 + timedelta(days=2),
                T0 + timedelta(days=1),
                T0 + timedelta(days=8),
            ],
            "xg": [0.1, 0.5, 0.3, 0.9],
        }
    ).with_columns(pl.col("available_time").dt.replace_time_zone("UTC"))


class TestFilterAvailable:
    def test_cutoff_is_inclusive(self) -> None:
        """A record available exactly at the prediction instant was available."""
        result = filter_available(_source(), cutoff=T0)
        assert result.height == 1

    def test_missing_timestamp_column_is_a_hard_error(self) -> None:
        with pytest.raises(KeyError, match="four timestamps"):
            filter_available(pl.DataFrame({"a": [1]}), cutoff=T0)

    def test_naive_cutoff_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            filter_available(_source(), cutoff=datetime(2026, 8, 1, 12))  # noqa: DTZ001


class TestAsOfJoin:
    def test_selects_the_most_recent_available_record(self) -> None:
        result = as_of_join(_entities(), _source(), entity_keys=["player_code"])
        assert result["xg"].to_list() == [0.5, 0.3, 0.9]

    def test_preserves_input_row_order(self) -> None:
        """Callers index into the result positionally; reordering corrupts that."""
        entities = _entities()
        result = as_of_join(entities, _source(), entity_keys=["player_code"])
        assert result["player_code"].to_list() == entities["player_code"].to_list()

    def test_entities_without_a_record_keep_nulls(self) -> None:
        """Dropping the row would bias every downstream aggregate."""
        entities = pl.DataFrame(
            {"player_code": [99], "prediction_timestamp": [T0 + timedelta(days=3)]}
        ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))
        result = as_of_join(entities, _source(), entity_keys=["player_code"])
        assert result.height == 1
        assert result["xg"].to_list() == [None]

    def test_naive_timestamps_rejected(self) -> None:
        naive = _entities().with_columns(pl.col("prediction_timestamp").dt.replace_time_zone(None))
        with pytest.raises(TypeError, match="timezone-aware"):
            as_of_join(naive, _source(), entity_keys=["player_code"])

    def test_future_records_are_never_selected(self) -> None:
        source = _source()
        future = source.with_columns(
            pl.col("available_time") + pl.duration(days=365),
            pl.lit(99.0).alias("xg"),
        )
        combined = pl.concat([source, future])
        result = as_of_join(_entities(), combined, entity_keys=["player_code"])
        assert 99.0 not in result["xg"].to_list()


class TestDeterministicTieBreaking:
    def test_ties_resolve_identically_regardless_of_input_order(self) -> None:
        """One API snapshot stamps many rows with the same publication instant.

        Without maintain_order=True the winner among tied records can vary
        between runs, which would silently defeat rerun-equality.
        """
        tied = pl.DataFrame(
            {
                "player_code": [1, 1, 1],
                "available_time": [T0, T0, T0],
                "xg": [0.1, 0.2, 0.3],
            }
        ).with_columns(pl.col("available_time").dt.replace_time_zone("UTC"))

        entities = pl.DataFrame(
            {"player_code": [1], "prediction_timestamp": [T0 + timedelta(days=1)]}
        ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))

        results = set()
        for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0]):
            permuted = pl.DataFrame(
                {c: [tied[c][i] for i in order] for c in tied.columns}
            ).with_columns(pl.col("available_time").dt.replace_time_zone("UTC"))
            results.add(
                point_in_time_join(entities, permuted, entity_keys=["player_code"])["xg"].item()
            )

        assert len(results) == 1, f"tie-breaking is order-dependent: got {results}"

    def test_repeated_runs_agree(self) -> None:
        entities, source = _entities(), _source()
        runs = {
            tuple(point_in_time_join(entities, source, entity_keys=["player_code"])["xg"])
            for _ in range(5)
        }
        assert len(runs) == 1


class TestValueColumnSelection:
    def test_restricting_columns_drops_the_rest(self) -> None:
        source = _source().with_columns(pl.lit(1.0).alias("unwanted"))
        result = point_in_time_join(
            _entities(), source, entity_keys=["player_code"], value_columns=["xg"]
        )
        assert "unwanted" not in result.columns
        assert "xg" in result.columns

    def test_requesting_a_missing_column_raises(self) -> None:
        with pytest.raises(KeyError, match="missing requested columns"):
            point_in_time_join(
                _entities(), _source(), entity_keys=["player_code"], value_columns=["nope"]
            )


# ---------------------------------------------------------------------------
# The leakage harness itself
# ---------------------------------------------------------------------------


def _clean_builder(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    """Correct: only records available at the prediction time."""
    return point_in_time_join(entities, source, entity_keys=["player_code"])


def _leaky_builder(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    """Deliberately wrong: a season-to-date mean over every record, past and future.

    This is the most common real leak — an aggregate computed over the whole
    dataset rather than as of each prediction.
    """
    means = source.group_by("player_code").agg(pl.col("xg").mean().alias("xg"))
    return entities.join(means, on="player_code", how="left")


@pytest.mark.leakage
class TestLeakageHarness:
    def test_a_correct_builder_passes(self) -> None:
        source = _source()
        assert_no_leakage(
            _clean_builder,
            entities=_entities(),
            source=source,
            future_records=make_future_records(
                source, after=T0 + timedelta(days=400), entity_keys=["player_code"]
            ),
            compare_columns=["xg"],
        )

    def test_negative_control_the_harness_catches_a_known_leak(self) -> None:
        """Without this, a broken harness would pass everything silently."""
        source = _source()
        assert_detects_leakage(
            _leaky_builder,
            entities=_entities(),
            source=source,
            future_records=make_future_records(
                source, after=T0 + timedelta(days=400), entity_keys=["player_code"]
            ),
        )

    def test_a_leaky_builder_raises_with_the_offending_column(self) -> None:
        source = _source()
        with pytest.raises(LeakageDetected, match="xg"):
            assert_no_leakage(
                _leaky_builder,
                entities=_entities(),
                source=source,
                future_records=make_future_records(
                    source, after=T0 + timedelta(days=400), entity_keys=["player_code"]
                ),
                compare_columns=["xg"],
            )

    def test_empty_future_set_is_rejected(self) -> None:
        """A harness run with nothing to add proves nothing."""
        with pytest.raises(ValueError, match="proves nothing"):
            find_leakage(
                _clean_builder,
                entities=_entities(),
                source=_source(),
                future_records=_source().head(0),
            )

    def test_row_count_changes_are_reported(self) -> None:
        def per_source_row_builder(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            """Emits one row per source record, so the height tracks the input."""
            return source.select("player_code", "xg")

        leaked = find_leakage(
            per_source_row_builder,
            entities=_entities(),
            source=_source(),
            future_records=make_future_records(
                _source(), after=T0 + timedelta(days=400), entity_keys=["player_code"]
            ),
        )
        assert any("row count changed" in item for item in leaked)

    def test_future_records_are_perturbed_not_copied(self) -> None:
        """Identical copies would slip past a builder that wrongly reads them."""
        source = _source()
        future = make_future_records(
            source, after=T0 + timedelta(days=400), entity_keys=["player_code"]
        )
        assert future["xg"].to_list() != source.head(future.height)["xg"].to_list()
