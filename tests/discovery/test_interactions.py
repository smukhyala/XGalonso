"""Constructed interactions: inlining, canonical order, and the additive control.

Two tests here are really statements about safety rather than about behaviour.

:meth:`TestInliningMakesTheProofReal.test_a_leaky_component_makes_the_interaction_leaky`
is the important one. An interaction built by *referencing* its components'
materialised columns would sail through the leakage harness having proved
nothing — appending future records cannot change an already-computed number — and
the registry honours the resulting `LEAKAGE_PASSED` flag. Inlining forces the
harness to re-derive both halves from source, so a leaky component produces a
leaky interaction and the run notices.

:meth:`TestCanonicalOrder.test_the_same_pair_produces_one_version` guards a trap
that would be invisible in every report: `FeatureProgram.version()` hashes the
canonical JSON, so `MUL(a, b)` and `MUL(b, a)` hash differently and the same
interaction would register twice, each copy diluting the other's evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from xg_alonso.discovery.compile import CompileContext, compile_program
from xg_alonso.discovery.dsl import (
    MAX_DEPTH,
    MAX_NODES,
    Arith,
    ArithOp,
    FeatureProgram,
    Rolling,
    RollingAgg,
    ShrunkRate,
    Source,
)
from xg_alonso.discovery.interactions import (
    INTERACTION_FORMS,
    interaction_pairs,
    interaction_program,
)
from xg_alonso.features.generators import stage_window


def _rolling(name: str, column: str, window: int = 5) -> FeatureProgram:
    """A point-in-time rolling mean — the shape a real candidate has."""
    return FeatureProgram(
        name=name,
        root=Rolling(child=Source(column=column), window=window, agg=RollingAgg.MEAN),
    )


def _rate(name: str, numerator: str, denominator: str = "minutes") -> FeatureProgram:
    """A shrunk per-90 rate, the other common candidate shape."""
    return FeatureProgram(
        name=name,
        root=ShrunkRate(numerator=numerator, denominator=denominator, window=5),
    )


class TestInteractionConstruction:
    def test_a_product_inlines_both_component_trees(self) -> None:
        """The root must contain the components, not references to them."""
        left = _rolling("xg", "expected_goals")
        right = _rolling("mins", "minutes")
        program = interaction_program(left, right, form="mul")

        assert program is not None
        root = program.root
        assert isinstance(root, Arith)
        assert root.op is ArithOp.MUL
        # The children are the component *trees*, not `Source(name=...)` leaves
        # pointing at already-computed entity columns.
        assert root.left == left.root or root.left == right.root
        assert not isinstance(root.left, Source)
        assert not isinstance(root.right, Source)

    def test_both_forms_build(self) -> None:
        left = _rolling("xg", "expected_goals")
        right = _rolling("mins", "minutes")
        for form in INTERACTION_FORMS:
            assert interaction_program(left, right, form=form) is not None

    def test_an_unknown_form_is_refused(self) -> None:
        left = _rolling("a", "expected_goals")
        right = _rolling("b", "minutes")
        assert interaction_program(left, right, form="nonsense") is None

    def test_a_program_is_not_crossed_with_itself(self) -> None:
        same = _rolling("a", "expected_goals")
        assert interaction_program(same, same, form="mul") is None

    def test_safe_div_is_the_only_division(self) -> None:
        """The language has no plain division, so a zero denominator is
        unrepresentable rather than merely guarded against."""
        program = interaction_program(
            _rolling("a", "expected_goals"), _rolling("b", "minutes"), form="safe_div"
        )
        assert program is not None
        assert isinstance(program.root, Arith)
        assert program.root.op is ArithOp.SAFE_DIV
        assert program.root.epsilon > 0.0


class TestCanonicalOrder:
    def test_the_same_pair_produces_one_version(self) -> None:
        """`MUL(a, b)` and `MUL(b, a)` must not register as two features."""
        first = _rolling("alpha", "expected_goals")
        second = _rolling("beta", "minutes")

        forward = interaction_program(first, second, form="mul")
        backward = interaction_program(second, first, form="mul")
        assert forward is not None
        assert backward is not None
        assert forward.version() == backward.version()
        assert forward.name == backward.name

    def test_the_name_records_the_operand_order(self) -> None:
        program = interaction_program(
            _rolling("zeta", "expected_goals"), _rolling("alpha", "minutes"), form="mul"
        )
        assert program is not None
        assert program.name == "alpha__mul__zeta"


class TestPairGeneration:
    def test_it_caps_the_combinatorics(self) -> None:
        programs = [_rolling(f"f{i}", "expected_goals") for i in range(10)]
        pairs = interaction_pairs(programs, cap=4)
        # C(4,2) = 6 pairs, times two forms.
        assert len(pairs) == 6 * len(INTERACTION_FORMS)

    def test_it_is_deterministic(self) -> None:
        programs = [_rolling(f"f{i}", "expected_goals") for i in range(5)]
        first = [c.program.name for c in interaction_pairs(programs, cap=5)]
        second = [c.program.name for c in interaction_pairs(programs, cap=5)]
        assert first == second

    def test_components_are_reported_for_the_additive_control(self) -> None:
        programs = [_rolling("a", "expected_goals"), _rolling("b", "minutes")]
        pairs = interaction_pairs(programs, cap=2)
        assert pairs
        for candidate in pairs:
            assert set(candidate.components) == {"a", "b"}

    def test_a_single_program_yields_nothing(self) -> None:
        assert interaction_pairs([_rolling("only", "minutes")], cap=8) == []

    def test_a_zero_cap_yields_nothing(self) -> None:
        programs = [_rolling(f"f{i}", "minutes") for i in range(4)]
        assert interaction_pairs(programs, cap=0) == []


class TestNoTriples:
    """Nesting is prevented by construction, not by the depth limits."""

    def test_shallow_nesting_is_legal_so_the_caller_must_prevent_it(self) -> None:
        """The honest statement of where the guarantee actually comes from.

        `MAX_DEPTH` (8) and `MAX_NODES` (40) do not refuse a product of two
        shallow interactions — 11 nodes at depth 4 is comfortably legal. So the
        protection cannot be the validator; it is that `run_discovery` feeds
        `interaction_pairs` only the programs that survived round one, never its
        own output. This test exists to stop anyone re-deriving the wrong
        reassurance from a passing validator.
        """
        first = interaction_program(
            _rolling("a", "expected_goals"), _rolling("b", "minutes"), form="mul"
        )
        second = interaction_program(
            _rolling("c", "expected_assists"), _rolling("d", "starts"), form="mul"
        )
        assert first is not None
        assert second is not None

        nested = interaction_program(first, second, form="mul")
        assert nested is not None
        assert nested.root.node_count() <= MAX_NODES
        assert nested.root.depth() <= MAX_DEPTH

    def test_deep_components_do_overflow_the_limits(self) -> None:
        """The backstop still works where it applies."""
        deep = _rate("deep", "expected_goals")
        stacked = interaction_program(deep, _rolling("b", "minutes"), form="mul")
        assert stacked is not None
        for _ in range(6):
            nxt = interaction_program(stacked, _rolling("c", "starts"), form="mul")
            if nxt is None:
                break
            stacked = nxt
            if stacked.root.node_count() > MAX_NODES or stacked.root.depth() > MAX_DEPTH:
                break
        assert stacked.root.node_count() > MAX_NODES or stacked.root.depth() > MAX_DEPTH


class TestInliningMakesTheProofReal:
    """The safety property the whole module rests on."""

    def test_the_compiled_column_depends_on_the_source_not_a_cached_column(self) -> None:
        """Recomputing after the source changes must change the result.

        This is the mechanical form of "the leakage harness can see through the
        interaction". If the interaction referenced already-materialised entity
        columns, changing the underlying history would leave it untouched — and
        `find_leakage`, which works by appending future records and recomputing,
        would report clean having proved nothing.
        """
        entities = pl.DataFrame(
            {
                "player_code": [1, 2],
                "prediction_timestamp": [
                    datetime(2025, 1, 10, tzinfo=UTC),
                    datetime(2025, 1, 10, tzinfo=UTC),
                ],
            }
        ).with_columns(pl.col("prediction_timestamp").cast(pl.Datetime(time_unit="us")))

        def history(scale: float) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "player_code": [1, 1, 2, 2],
                    "expected_goals": [0.4 * scale, 0.6 * scale, 0.1, 0.3],
                    "minutes": [90.0, 90.0, 45.0, 45.0],
                    "kickoff_time": [
                        datetime(2025, 1, 1, tzinfo=UTC),
                        datetime(2025, 1, 5, tzinfo=UTC),
                        datetime(2025, 1, 1, tzinfo=UTC),
                        datetime(2025, 1, 5, tzinfo=UTC),
                    ],
                    "available_time": [
                        datetime(2025, 1, 1, tzinfo=UTC),
                        datetime(2025, 1, 5, tzinfo=UTC),
                        datetime(2025, 1, 1, tzinfo=UTC),
                        datetime(2025, 1, 5, tzinfo=UTC),
                    ],
                }
            ).with_columns(pl.col("available_time").cast(pl.Datetime(time_unit="us")))

        program = interaction_program(
            _rolling("xg", "expected_goals"), _rolling("mins", "minutes"), form="mul"
        )
        assert program is not None

        def value(scale: float) -> float:
            context = CompileContext(player_stats=history(scale), stage=stage_window)
            frame = compile_program(program, entities, context)
            return float(frame[program.name][0])

        assert value(1.0) != pytest.approx(value(5.0)), (
            "the interaction did not move when its source history changed, so the "
            "leakage harness would prove nothing about it"
        )
