"""A safe, serialisable feature-program language.

**Nothing here is Python source, and nothing here is ever ``eval``'d or
``exec``'d.** A feature program is a frozen expression tree of typed nodes. It
serialises to JSON, hashes to a version, reports its own dependencies, and is
compiled to a Polars plan by :mod:`xg_alonso.discovery.compile`. A generator —
whether a residual-driven search or, one day, a language model — can only ever
emit a tree, so the worst a bad proposal can do is fail validation.

This module is deliberately domain-free. It knows about columns, windows and
arithmetic; it knows nothing about football, and ``.importlinter`` enforces that
so the engine stays reusable.

## The two levels, and why the distinction is load-bearing

A feature is computed for a *prediction row* — one player at one deadline — from
many *history rows*. Those are different populations, and an expression that
silently mixes them is meaningless:

``ROW`` level
    An expression over the columns of a single historical record. ``minutes``,
    ``expected_goals / minutes``, ``log1p(bps)``. These live inside the lookback
    window and have one value *per historical match*.

``ENTITY`` level
    An expression over one prediction row. Produced by aggregating a ``ROW``
    expression across the window — that is what :class:`Rolling` and its
    siblings do — or read directly from a fixture attribute like ``is_home``.

The rule is mechanical: **a temporal node consumes ``ROW`` and produces
``ENTITY``.** Arithmetic may combine two expressions only at the same level. A
program's root must be ``ENTITY``, because that is the shape a feature column
has.

Without this, ``rolling_mean(minutes) * minutes`` type-checks as arithmetic and
means nothing — the left side is one number per player and the right is one per
match. Rejecting it costs nothing; allowing it produces a column that looks
computed and is not.

## What cannot be expressed

Some things are absent by construction rather than discouraged by convention:

- **There is no division.** Only :class:`safe_div <ArithOp>`, which carries an
  explicit epsilon. A divide-by-zero cannot be written.
- **There is no way to read the future.** Temporal nodes describe a lookback in
  *appearances already visible*; the compiler resolves them through the same
  ``stage_window`` the leakage harness already proves correct. No node takes a
  forward offset.
- **There is no raw column escape hatch.** Every leaf is a declared
  :class:`Source` whose name is checked against the real schema.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

__all__ = [
    "MAX_DEPTH",
    "MAX_NODES",
    "MAX_WINDOW",
    "Arith",
    "ArithOp",
    "ClusterRel",
    "ClusterRelOp",
    "Const",
    "EwmMean",
    "FeatureProgram",
    "GroupKey",
    "GroupRel",
    "GroupRelOp",
    "Lag",
    "Level",
    "Node",
    "ProgramError",
    "Rolling",
    "RollingAgg",
    "ShrunkRate",
    "Source",
    "SourceScope",
    "TimeSince",
    "Trend",
    "Unary",
    "UnaryOp",
    "parse_program",
    "program_hash",
]

#: Structural limits. Exceeding any of them is a static rejection rather than a
#: slow run: a program deep enough to be uninterpretable is not a feature, it is
#: a small model wearing one's clothes, and it belongs in the model layer where
#: it would at least be regularised.
MAX_DEPTH: Final[int] = 8
MAX_NODES: Final[int] = 40
MAX_WINDOW: Final[int] = 40


class ProgramError(ValueError):
    """A program is structurally invalid.

    Raised directly by :func:`parse_program` and by
    :func:`xg_alonso.discovery.compile.validate_program`, which are the two gates
    a generated program passes through.

    Node constructors raise it too, but Pydantic wraps any ``ValueError`` a
    validator raises inside a ``ValidationError``. Callers that build nodes by
    hand — tests, mostly — should therefore expect ``ValidationError``; callers
    that go through the gates get this type. The distinction is stated here
    rather than papered over, because a test that catches the wrong exception
    passes for the wrong reason.
    """


class Level(StrEnum):
    """Which population an expression is evaluated over. See the module docstring."""

    ROW = "row"
    ENTITY = "entity"


class SourceScope(StrEnum):
    """Where a leaf column is read from."""

    HISTORY = "history"
    """A column of the historical stats table. ``ROW`` level."""

    ENTITY = "entity"
    """A column already on the prediction row — the fixture, the price, a
    previously-built feature. ``ENTITY`` level."""


class RollingAgg(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    PERCENTILE = "percentile"


class ArithOp(StrEnum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    SAFE_DIV = "safe_div"
    """Division with an explicit epsilon in the denominator.

    The only division in the language. Plain division is unrepresentable, which
    is stronger than checking for it: a generator cannot emit a divide-by-zero
    because there is no node that would mean one.
    """
    MIN = "min"
    MAX = "max"


class UnaryOp(StrEnum):
    LOG1P = "log1p"
    NEG = "neg"
    ABS = "abs"
    CLIP = "clip"
    ZSCORE = "zscore"
    """Standardised across the batch. ``ENTITY`` level only."""
    PERCENTILE_RANK = "percentile_rank"
    """Rank within the batch, scaled to [0, 1]. ``ENTITY`` level only."""


class GroupKey(StrEnum):
    """What a group-relative expression is measured within."""

    POSITION = "position"
    TEAM = "team"
    OPPONENT = "opponent"
    ALL = "all"


class GroupRelOp(StrEnum):
    RANK = "rank"
    SHARE = "share"
    """This row's value divided by the group total. 'Share of team threat'."""
    DEV_FROM_MEAN = "dev_from_mean"
    ZSCORE = "zscore"


class ClusterRelOp(StrEnum):
    RANK = "rank"
    DEV_FROM_CENTROID = "dev_from_centroid"
    GATE = "gate"
    """Multiply by soft cluster membership.

    The gating primitive. A feature that matters only for one archetype is
    multiplied by the *probability* the player belongs to it, rather than
    switched on by a hard label. A player sitting between two clusters then
    contributes to both in proportion, which is what a boundary means.
    """


class _Node(BaseModel):
    """Base for every node. Frozen, closed, and self-describing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def children(self) -> tuple[Node, ...]:
        """Sub-expressions, in evaluation order."""
        return ()

    @property
    def level(self) -> Level:
        raise NotImplementedError  # pragma: no cover - every concrete node overrides

    def descendants(self) -> tuple[Node, ...]:
        """This node and everything beneath it, depth-first."""
        out: list[Node] = [self]  # type: ignore[list-item]
        for child in self.children():
            out.extend(child.descendants())
        return tuple(out)

    def depth(self) -> int:
        kids = self.children()
        return 1 + max((c.depth() for c in kids), default=0)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children())

    def columns(self) -> tuple[str, ...]:
        """Every source column this expression reads, de-duplicated and sorted.

        This is the lineage a registry stores and an availability check reads.
        """
        found: set[str] = set()
        for node in self.descendants():
            if isinstance(node, Source):
                found.add(node.column)
            elif isinstance(node, ShrunkRate):
                found.update({node.numerator, node.denominator})
            elif isinstance(node, TimeSince):
                found.add(node.event_column)
        return tuple(sorted(found))

    def max_window(self) -> int:
        """The widest lookback in the tree, in appearances. Zero when none."""
        widest = 0
        for node in self.descendants():
            window = getattr(node, "window", None)
            if isinstance(window, int):
                widest = max(widest, window)
        return widest


# --- leaves ------------------------------------------------------------------


class Source(_Node):
    """A raw column."""

    kind: Literal["source"] = "source"
    column: str = Field(min_length=1)
    scope: SourceScope = SourceScope.HISTORY

    @property
    def level(self) -> Level:
        return Level.ROW if self.scope is SourceScope.HISTORY else Level.ENTITY


class Const(_Node):
    """A literal.

    Level-polymorphic: a constant is meaningful against either population, so it
    reports ``ROW`` and the arithmetic validator treats it as compatible with
    both. Without that, ``rolling_mean(x) * 2`` would fail to type-check.
    """

    kind: Literal["const"] = "const"
    value: float

    @property
    def level(self) -> Level:
        return Level.ROW


# --- temporal: ROW -> ENTITY -------------------------------------------------


class _Temporal(_Node):
    """Consumes a ``ROW`` expression over the lookback window, yields ``ENTITY``."""

    child: Node

    def children(self) -> tuple[Node, ...]:
        return (self.child,)

    @property
    def level(self) -> Level:
        return Level.ENTITY

    @model_validator(mode="after")
    def _child_is_row_level(self) -> _Temporal:
        if self.child.level is not Level.ROW:
            raise ProgramError(
                f"{type(self).__name__} aggregates over history rows, so its child must be "
                f"a row-level expression; got an entity-level one. Aggregating an "
                f"already-aggregated value is either a mistake or a second window that "
                f"should be written as one."
            )
        return self


class Rolling(_Temporal):
    """Aggregate a row expression over the last ``window`` visible appearances."""

    kind: Literal["rolling"] = "rolling"
    window: int = Field(ge=1, le=MAX_WINDOW)
    agg: RollingAgg = RollingAgg.MEAN
    min_periods: int = Field(default=1, ge=1)
    quantile: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _percentile_has_a_quantile(self) -> Rolling:
        if self.agg is RollingAgg.PERCENTILE and self.quantile is None:
            raise ProgramError("a percentile aggregation needs a quantile")
        if self.agg is not RollingAgg.PERCENTILE and self.quantile is not None:
            raise ProgramError(f"quantile is meaningless for {self.agg.value}")
        if self.agg is RollingAgg.STD and self.min_periods < 2:
            raise ProgramError(
                "a standard deviation over fewer than two observations is not a "
                "statistic; set min_periods >= 2"
            )
        if self.min_periods > self.window:
            raise ProgramError(
                f"min_periods {self.min_periods} exceeds window {self.window}, so this "
                "feature can never be computed"
            )
        return self


class Lag(_Temporal):
    """The value ``periods`` appearances ago. ``periods=1`` is the last match."""

    kind: Literal["lag"] = "lag"
    periods: int = Field(default=1, ge=1, le=MAX_WINDOW)

    @property
    def window(self) -> int:
        return self.periods


class EwmMean(_Temporal):
    """Exponentially weighted mean over the window, newest observation heaviest."""

    kind: Literal["ewm_mean"] = "ewm_mean"
    window: int = Field(ge=2, le=MAX_WINDOW)
    halflife: float = Field(gt=0.0, description="Appearances over which weight halves")


class Trend(_Temporal):
    """Ordinary-least-squares slope across the window, oldest to newest.

    Positive means improving. Requires at least three points: a slope through two
    is a line through two, which describes the pair and predicts nothing.
    """

    kind: Literal["trend"] = "trend"
    window: int = Field(ge=3, le=MAX_WINDOW)


class ShrunkRate(_Node):
    """An empirical-Bayes per-unit rate, shrunk toward a prior by sample size.

    Mirrors :func:`xg_alonso.features.generators.shrunk_rate_as_of`. Present as a
    primitive rather than composed from ``safe_div`` because the shrinkage is the
    point: a raw ratio over a thin denominator is not a small-sample estimate of
    a rate, it is a different quantity that happens to share its units.
    """

    kind: Literal["shrunk_rate"] = "shrunk_rate"
    numerator: str = Field(min_length=1)
    denominator: str = Field(min_length=1)
    window: int = Field(ge=1, le=MAX_WINDOW)
    prior_strength: float = Field(default=3.0, ge=0.0)
    scale: float = Field(default=90.0, gt=0.0)

    @property
    def level(self) -> Level:
        return Level.ENTITY

    @model_validator(mode="after")
    def _distinct(self) -> ShrunkRate:
        if self.numerator == self.denominator:
            raise ProgramError(
                f"numerator and denominator are both {self.numerator!r}; this is a "
                "constant wearing a rate's costume"
            )
        return self


class TimeSince(_Node):
    """Days between the prediction cutoff and the last visible qualifying event.

    ``ENTITY`` level: it is a property of the prediction row, measured against
    the row's own cutoff, so it cannot be aggregated further.
    """

    kind: Literal["time_since"] = "time_since"
    event_column: str = Field(
        min_length=1, description="Column whose non-null, positive value marks the event"
    )
    require_positive: bool = Field(
        default=True,
        description=(
            "When true the event is 'this column exceeded zero'. That is the "
            "difference between 'days since his club last played' and 'days "
            "since he last actually appeared', which are different questions "
            "and were conflated in an earlier minutes model."
        ),
    )

    @property
    def level(self) -> Level:
        return Level.ENTITY


# --- combinators -------------------------------------------------------------


class Arith(_Node):
    """Binary arithmetic between two expressions at the same level."""

    kind: Literal["arith"] = "arith"
    op: ArithOp
    left: Node
    right: Node
    epsilon: float = Field(
        default=1e-6,
        gt=0.0,
        description="Added to the denominator of safe_div. Never zero.",
    )

    def children(self) -> tuple[Node, ...]:
        return (self.left, self.right)

    @property
    def level(self) -> Level:
        # A Const is level-polymorphic, so a mixed pair takes the non-constant
        # side's level. Two constants fold to ROW, which is harmless.
        if isinstance(self.left, Const):
            return self.right.level
        return self.left.level

    @model_validator(mode="after")
    def _levels_agree(self) -> Arith:
        left_const = isinstance(self.left, Const)
        right_const = isinstance(self.right, Const)
        if left_const or right_const:
            return self
        if self.left.level is not self.right.level:
            row_side = "left" if self.left.level is Level.ROW else "right"
            raise ProgramError(
                f"cannot combine a {self.left.level.value}-level left operand with a "
                f"{self.right.level.value}-level right one. The row-level side (the "
                f"{row_side}) has one value per historical match; the other has one per "
                "player, so the result would not describe anything. Aggregate the "
                "row-level side first."
            )
        return self


class Unary(_Node):
    """A single-argument transform."""

    kind: Literal["unary"] = "unary"
    op: UnaryOp
    child: Node
    lower: float | None = None
    upper: float | None = None

    def children(self) -> tuple[Node, ...]:
        return (self.child,)

    @property
    def level(self) -> Level:
        return self.child.level

    @model_validator(mode="after")
    def _check_operands(self) -> Unary:
        if self.op is UnaryOp.CLIP and self.lower is None and self.upper is None:
            raise ProgramError("clip needs at least one bound")
        if self.op is not UnaryOp.CLIP and (self.lower is not None or self.upper is not None):
            raise ProgramError(f"bounds are meaningless for {self.op.value}")
        if (
            self.op is UnaryOp.CLIP
            and self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ProgramError(f"clip lower {self.lower} exceeds upper {self.upper}")
        if self.op in {UnaryOp.ZSCORE, UnaryOp.PERCENTILE_RANK} and (
            self.child.level is not Level.ENTITY
        ):
            raise ProgramError(
                f"{self.op.value} ranks across the batch of predictions, so it needs "
                "an entity-level child; ranking history rows against each other "
                "would compare a player's own matches, which is a different question"
            )
        return self


class GroupRel(_Node):
    """An entity-level value expressed relative to its group.

    'Share of his team's threat', 'rank among midfielders'. The group is computed
    **within the prediction batch**, which is the correct scope: a batch is one
    deadline, and a player's standing among the players a manager is choosing
    between at that deadline is exactly the comparison being made.
    """

    kind: Literal["group_rel"] = "group_rel"
    op: GroupRelOp
    by: GroupKey
    child: Node

    def children(self) -> tuple[Node, ...]:
        return (self.child,)

    @property
    def level(self) -> Level:
        return Level.ENTITY

    @model_validator(mode="after")
    def _child_is_entity_level(self) -> GroupRel:
        if self.child.level is not Level.ENTITY:
            raise ProgramError(
                "a group-relative expression compares players to each other, so its "
                "child must already be one value per player"
            )
        return self


class ClusterRel(_Node):
    """An entity-level value expressed relative to a player's cluster.

    Carries the cluster model version explicitly. A cluster id means nothing
    without the model that produced it, and a feature that referenced 'cluster 3'
    across two model versions would silently change meaning between runs.
    """

    kind: Literal["cluster_rel"] = "cluster_rel"
    op: ClusterRelOp
    child: Node
    cluster_model_version: str = Field(min_length=1)
    cluster_id: int | None = Field(
        default=None,
        ge=0,
        description="Required for GATE — which cluster's membership multiplies the value",
    )

    def children(self) -> tuple[Node, ...]:
        return (self.child,)

    @property
    def level(self) -> Level:
        return Level.ENTITY

    @model_validator(mode="after")
    def _check(self) -> ClusterRel:
        if self.child.level is not Level.ENTITY:
            raise ProgramError("a cluster-relative expression needs an entity-level child")
        if self.op is ClusterRelOp.GATE and self.cluster_id is None:
            raise ProgramError("gating needs a cluster_id to take the membership of")
        return self


#: The recursive union. Discriminated on ``kind`` so a serialised program parses
#: back to exactly the node types it was written with, and an unknown ``kind``
#: is a parse error rather than a silently-dropped branch.
Node = Annotated[
    Union[  # noqa: UP007 - pydantic needs the explicit Union for discrimination
        Source,
        Const,
        Rolling,
        Lag,
        EwmMean,
        Trend,
        ShrunkRate,
        TimeSince,
        Arith,
        Unary,
        GroupRel,
        ClusterRel,
    ],
    Field(discriminator="kind"),
]

for _cls in (
    Rolling,
    Lag,
    EwmMean,
    Trend,
    Arith,
    Unary,
    GroupRel,
    ClusterRel,
):
    _cls.model_rebuild()

_NODE_ADAPTER: Final[TypeAdapter[Any]] = TypeAdapter(Node)


def _canonical(payload: Any) -> Any:
    """Recursively sort mapping keys so the JSON form is order-independent."""
    if isinstance(payload, dict):
        return {key: _canonical(payload[key]) for key in sorted(payload)}
    if isinstance(payload, list):
        return [_canonical(item) for item in payload]
    return payload


class FeatureProgram(BaseModel):
    """A named, versioned expression tree. The unit the registry stores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Column name the program produces")
    root: Node

    @model_validator(mode="after")
    def _root_is_entity_level(self) -> FeatureProgram:
        if self.root.level is not Level.ENTITY:
            raise ProgramError(
                f"program {self.name!r} has a row-level root, so it would produce one "
                "value per historical match rather than one per player. Wrap it in a "
                "temporal aggregation."
            )
        return self

    def canonical_json(self) -> str:
        """Key-sorted, whitespace-free JSON. The basis of the version hash."""
        payload = _canonical(self.root.model_dump(mode="json"))
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def version(self) -> str:
        """Content hash of the program's semantics.

        **Identity is semantic, not nominal.** Two differently-named programs
        that compute the same thing hash identically and the second is a
        duplicate; a program that is edited becomes a new version rather than
        silently changing what an old evaluation referred to. The name is
        excluded from the hash for exactly that reason.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def columns(self) -> tuple[str, ...]:
        return self.root.columns()

    def depth(self) -> int:
        return self.root.depth()

    def node_count(self) -> int:
        return self.root.node_count()

    def max_window(self) -> int:
        return self.root.max_window()

    def cluster_versions(self) -> tuple[str, ...]:
        """Cluster models this program depends on, if any."""
        return tuple(
            sorted(
                {
                    node.cluster_model_version
                    for node in self.root.descendants()
                    if isinstance(node, ClusterRel)
                }
            )
        )

    def describe(self) -> str:
        """A compact, readable rendering of the tree. For explanation surfaces."""
        return _describe(self.root)


def _describe(node: Node) -> str:
    """Render a node as a readable expression."""
    if isinstance(node, Source):
        return node.column
    if isinstance(node, Const):
        return f"{node.value:g}"
    if isinstance(node, Rolling):
        suffix = f",q={node.quantile:g}" if node.quantile is not None else ""
        return f"{node.agg.value}_{node.window}({_describe(node.child)}{suffix})"
    if isinstance(node, Lag):
        return f"lag_{node.periods}({_describe(node.child)})"
    if isinstance(node, EwmMean):
        return f"ewm_{node.window}(hl={node.halflife:g}, {_describe(node.child)})"
    if isinstance(node, Trend):
        return f"trend_{node.window}({_describe(node.child)})"
    if isinstance(node, ShrunkRate):
        return f"shrunk_rate_{node.window}({node.numerator}/{node.denominator})"
    if isinstance(node, TimeSince):
        return f"days_since({node.event_column})"
    if isinstance(node, Arith):
        symbols = {
            ArithOp.ADD: "+",
            ArithOp.SUB: "-",
            ArithOp.MUL: "*",
            ArithOp.SAFE_DIV: "/",
            ArithOp.MIN: "min",
            ArithOp.MAX: "max",
        }
        symbol = symbols[node.op]
        if node.op in {ArithOp.MIN, ArithOp.MAX}:
            return f"{symbol}({_describe(node.left)}, {_describe(node.right)})"
        return f"({_describe(node.left)} {symbol} {_describe(node.right)})"
    if isinstance(node, Unary):
        if node.op is UnaryOp.CLIP:
            return f"clip({_describe(node.child)}, {node.lower}, {node.upper})"
        return f"{node.op.value}({_describe(node.child)})"
    if isinstance(node, GroupRel):
        return f"{node.op.value}_by_{node.by.value}({_describe(node.child)})"
    if isinstance(node, ClusterRel):
        target = "" if node.cluster_id is None else f"#{node.cluster_id}"
        return f"cluster_{node.op.value}{target}({_describe(node.child)})"
    raise ProgramError(f"unrenderable node {type(node).__name__}")  # pragma: no cover


def parse_program(name: str, payload: str | dict[str, Any]) -> FeatureProgram:
    """Parse a serialised program.

    Raises:
        ProgramError: on anything malformed — an unknown node kind, a level
            mismatch, a missing field. Never returns a partially-understood
            tree, because a program that parsed "mostly" would compute something
            nobody wrote.
    """
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        root = _NODE_ADAPTER.validate_python(data)
        return FeatureProgram(name=name, root=root)
    except ProgramError:
        raise
    except Exception as exc:  # pydantic ValidationError and json errors alike
        raise ProgramError(f"could not parse program {name!r}: {exc}") from exc


def program_hash(program: FeatureProgram) -> str:
    """The program's semantic version. See :meth:`FeatureProgram.version`."""
    return program.version()
