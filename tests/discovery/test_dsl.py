"""The feature DSL: what it accepts, what it refuses, and how it versions.

The refusals matter more than the acceptances. A generator will eventually emit
every malformed program in this file, and the only thing standing between that
and a corrupted registry is that each one raises here.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from xg_alonso.discovery.dsl import (
    MAX_WINDOW,
    Arith,
    ArithOp,
    ClusterRel,
    ClusterRelOp,
    Const,
    EwmMean,
    FeatureProgram,
    GroupKey,
    GroupRel,
    GroupRelOp,
    Lag,
    Level,
    ProgramError,
    Rolling,
    RollingAgg,
    ShrunkRate,
    Source,
    SourceScope,
    TimeSince,
    Trend,
    Unary,
    UnaryOp,
    parse_program,
)


def _xg() -> ShrunkRate:
    return ShrunkRate(numerator="expected_goals", denominator="minutes", window=5)


class TestLevels:
    """The row/entity distinction, which is what stops meaningless expressions."""

    def test_history_source_is_row_level(self) -> None:
        assert Source(column="minutes").level is Level.ROW

    def test_entity_source_is_entity_level(self) -> None:
        assert Source(column="is_home", scope=SourceScope.ENTITY).level is Level.ENTITY

    def test_temporal_node_lifts_row_to_entity(self) -> None:
        assert Rolling(child=Source(column="minutes"), window=5).level is Level.ENTITY

    def test_mixing_levels_in_arithmetic_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="row-level"):
            Arith(
                op=ArithOp.MUL,
                left=Rolling(child=Source(column="minutes"), window=5),
                right=Source(column="minutes"),
            )

    def test_a_constant_combines_with_either_level(self) -> None:
        row = Arith(op=ArithOp.MUL, left=Source(column="minutes"), right=Const(value=2.0))
        entity = Arith(op=ArithOp.MUL, left=_xg(), right=Const(value=2.0))
        assert row.level is Level.ROW
        assert entity.level is Level.ENTITY

    def test_double_aggregation_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="row-level expression"):
            Rolling(child=Rolling(child=Source(column="minutes"), window=3), window=5)

    def test_program_root_must_be_entity_level(self) -> None:
        with pytest.raises(ValidationError, match="row-level root"):
            FeatureProgram(name="bad", root=Source(column="minutes"))


class TestRefusals:
    """Every construct the language deliberately cannot express."""

    def test_std_needs_at_least_two_observations(self) -> None:
        with pytest.raises(ValidationError, match="not a statistic"):
            Rolling(
                child=Source(column="total_points"),
                window=5,
                agg=RollingAgg.STD,
                min_periods=1,
            )

    def test_min_periods_above_window_is_uncomputable(self) -> None:
        with pytest.raises(ValidationError, match="never be computed"):
            Rolling(child=Source(column="minutes"), window=3, min_periods=5)

    def test_percentile_needs_a_quantile(self) -> None:
        with pytest.raises(ValidationError, match="needs a quantile"):
            Rolling(child=Source(column="bps"), window=5, agg=RollingAgg.PERCENTILE)

    def test_quantile_without_percentile_is_meaningless(self) -> None:
        with pytest.raises(ValidationError, match="meaningless"):
            Rolling(child=Source(column="bps"), window=5, agg=RollingAgg.MEAN, quantile=0.5)

    def test_identical_numerator_and_denominator_is_a_constant(self) -> None:
        with pytest.raises(ValidationError, match="constant wearing"):
            ShrunkRate(numerator="minutes", denominator="minutes", window=5)

    def test_window_beyond_the_cap_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Rolling(child=Source(column="minutes"), window=MAX_WINDOW + 1)

    def test_trend_needs_three_points(self) -> None:
        with pytest.raises(ValidationError):
            Trend(child=Source(column="minutes"), window=2)

    def test_batch_ranking_needs_an_entity_child(self) -> None:
        with pytest.raises(ValidationError, match="across the batch"):
            Unary(op=UnaryOp.ZSCORE, child=Source(column="minutes"))

    def test_clip_needs_a_bound(self) -> None:
        with pytest.raises(ValidationError, match="at least one bound"):
            Unary(op=UnaryOp.CLIP, child=_xg())

    def test_inverted_clip_bounds_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="exceeds upper"):
            Unary(op=UnaryOp.CLIP, child=_xg(), lower=5.0, upper=1.0)

    def test_gating_needs_a_cluster(self) -> None:
        with pytest.raises(ValidationError, match="needs a cluster_id"):
            ClusterRel(op=ClusterRelOp.GATE, child=_xg(), cluster_model_version="v1")

    def test_there_is_no_plain_division(self) -> None:
        """Division is unrepresentable, not merely discouraged."""
        assert not any(op.value == "div" for op in ArithOp)
        assert ArithOp.SAFE_DIV in set(ArithOp)

    def test_no_node_takes_a_forward_offset(self) -> None:
        """Every temporal primitive looks backwards. There is no lead."""
        with pytest.raises(ValidationError):
            Lag(child=Source(column="minutes"), periods=-1)


class TestVersioning:
    """Identity is semantic. A rename is not a new feature."""

    def test_the_same_program_hashes_the_same(self) -> None:
        left = FeatureProgram(name="a", root=_xg())
        right = FeatureProgram(name="a", root=_xg())
        assert left.version() == right.version()

    def test_the_name_is_excluded_from_the_hash(self) -> None:
        left = FeatureProgram(name="xg_five", root=_xg())
        right = FeatureProgram(name="something_entirely_different", root=_xg())
        assert left.version() == right.version()

    def test_a_changed_window_is_a_new_version(self) -> None:
        five = FeatureProgram(name="a", root=_xg())
        ten = FeatureProgram(
            name="a",
            root=ShrunkRate(numerator="expected_goals", denominator="minutes", window=10),
        )
        assert five.version() != ten.version()

    def test_the_hash_does_not_depend_on_key_order(self) -> None:
        program = FeatureProgram(name="a", root=_xg())
        shuffled = json.dumps(
            dict(reversed(list(json.loads(program.canonical_json()).items()))),
            separators=(",", ":"),
        )
        assert parse_program("a", shuffled).version() == program.version()


class TestRoundTrip:
    def test_serialise_and_parse_preserves_semantics(self) -> None:
        program = FeatureProgram(
            name="complex",
            root=Arith(
                op=ArithOp.MUL,
                left=_xg(),
                right=GroupRel(
                    op=GroupRelOp.ZSCORE,
                    by=GroupKey.POSITION,
                    child=EwmMean(child=Source(column="total_points"), window=10, halflife=3.0),
                ),
            ),
        )
        restored = parse_program("complex", program.canonical_json())
        assert restored.version() == program.version()
        assert restored.describe() == program.describe()

    def test_an_unknown_node_kind_is_a_parse_error(self) -> None:
        with pytest.raises(ProgramError, match="could not parse"):
            parse_program("x", '{"kind":"definitely_not_a_node"}')

    def test_malformed_json_is_a_parse_error(self) -> None:
        with pytest.raises(ProgramError):
            parse_program("x", "{not json")


class TestIntrospection:
    def test_columns_reports_every_source_read(self) -> None:
        program = FeatureProgram(
            name="p",
            root=Arith(op=ArithOp.MUL, left=_xg(), right=TimeSince(event_column="starts")),
        )
        assert program.columns() == ("expected_goals", "minutes", "starts")

    def test_max_window_is_the_widest_lookback(self) -> None:
        program = FeatureProgram(
            name="p",
            root=Arith(
                op=ArithOp.ADD,
                left=Rolling(child=Source(column="minutes"), window=3),
                right=Rolling(child=Source(column="bps"), window=20),
            ),
        )
        assert program.max_window() == 20

    def test_cluster_dependencies_are_reported(self) -> None:
        program = FeatureProgram(
            name="gated",
            root=ClusterRel(
                op=ClusterRelOp.GATE, child=_xg(), cluster_model_version="v7", cluster_id=2
            ),
        )
        assert program.cluster_versions() == ("v7",)

    def test_describe_is_readable(self) -> None:
        program = FeatureProgram(
            name="p",
            root=Arith(op=ArithOp.MUL, left=_xg(), right=TimeSince(event_column="minutes")),
        )
        assert program.describe() == (
            "(shrunk_rate_5(expected_goals/minutes) * days_since(minutes))"
        )
