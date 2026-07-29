"""Compile a feature program to Polars, and refuse the ones that should not run.

Two gates, in order, and both must pass before a value is ever computed:

1. :func:`validate_program` — static. Checks the tree against the real schema
   and the configured limits. Cheap, and catches most of what a generator gets
   wrong.
2. The point-in-time leakage harness in :mod:`xg_alonso.features.leakage`,
   driven by the caller. Static validation cannot prove a window does not
   straddle a boundary; rebuilding with future rows appended and comparing does.

**Why the window staging is injected rather than imported.** The correct
implementation of "this player's last five appearances *as known at this
prediction timestamp*" already exists, in
:func:`xg_alonso.features.generators.stage_window`, and the leakage harness
already proves it. Re-implementing it here to keep this module domain-free would
create a second copy of the single highest-risk function in the platform, and
the two would eventually disagree on exactly the boundary condition that causes
silent leakage.

So the staging is a :class:`WindowStager` — a protocol this module declares and
``stage_window`` structurally satisfies. The engine stays generic, the proven
implementation stays the only one, and the seam is the same trick
``contracts.storage`` uses to keep DuckDB reversible.

**Shared sub-expressions are computed once.** Every node is keyed by the hash of
its own subtree, so ``xg_per90_5 * rest`` and ``xg_per90_5 / price`` in the same
batch stage and aggregate ``xg_per90_5`` a single time. That is not an
optimisation detail — feature search evaluates hundreds of programs over the
same anchors, and recomputing a shared anchor per candidate is the difference
between a search that finishes and one that does not.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import polars as pl

from xg_alonso.discovery.dsl import (
    MAX_DEPTH,
    MAX_NODES,
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
    Node,
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
)

__all__ = [
    "CompileContext",
    "ProgramCache",
    "ValidationIssue",
    "WindowStager",
    "compile_program",
    "compile_programs",
    "validate_program",
]

#: Internal row index. Matches the constant the shipped stager uses, because the
#: staged frame comes back carrying it.
ROW_ID: Final[str] = "__xg_entity_row"

#: Rank column the stager attaches: 0 is the newest visible record.
RANK: Final[str] = "__xg_rank"

_SECONDS_PER_DAY: Final[float] = 86400.0


class WindowStager(Protocol):
    """Attaches each entity row's visible history, newest first, capped at ``window``.

    :func:`xg_alonso.features.generators.stage_window` satisfies this. The
    returned frame must carry ``__xg_entity_row`` (the caller's row index) and
    ``__xg_rank`` (0 = most recent), and must contain **only** records whose
    ``available_time`` does not exceed the row's prediction timestamp.
    """

    def __call__(
        self,
        entities: pl.DataFrame,
        source: pl.DataFrame,
        *,
        entity_keys: Sequence[str],
        prediction_time_col: str,
        available_time_col: str,
        window: int,
        order_col: str | None,
    ) -> pl.DataFrame: ...


@dataclass(frozen=True)
class ValidationIssue:
    """One reason a program may not run."""

    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass
class ProgramCache:
    """Memoises computed sub-expressions within one compile call.

    Keyed by subtree hash, so identity is semantic: two programs that share an
    anchor share its computation without either knowing about the other.
    """

    columns: dict[str, pl.Series] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> pl.Series | None:
        found = self.columns.get(key)
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, key: str, values: pl.Series) -> None:
        self.columns[key] = values


@dataclass(frozen=True)
class CompileContext:
    """Everything a program needs to be evaluated against real data."""

    player_stats: pl.DataFrame
    """The historical record. One row per entity per past match."""

    stage: WindowStager

    entity_keys: tuple[str, ...] = ("player_code",)
    prediction_time_col: str = "prediction_timestamp"
    available_time_col: str = "available_time"
    order_col: str | None = "kickoff_time"
    event_time_col: str = "kickoff_time"

    group_columns: tuple[tuple[GroupKey, str], ...] = (
        (GroupKey.POSITION, "position"),
        (GroupKey.TEAM, "team_id"),
        (GroupKey.OPPONENT, "opponent_team_id"),
    )
    """Which entity column each group key resolves to."""

    cluster_memberships: pl.DataFrame | None = None
    """Optional. Columns: entity key(s), ``cluster_model_version``, ``cluster_id``,
    ``membership_probability``, ``distance_to_centroid``."""

    def group_column(self, key: GroupKey) -> str | None:
        if key is GroupKey.ALL:
            return None
        return next((column for group, column in self.group_columns if group is key), None)


def _subtree_key(node: Node) -> str:
    """Stable identity for a sub-expression: the hash of its canonical JSON."""
    payload = json.dumps(node.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


# --- static validation -------------------------------------------------------


def validate_program(
    program: FeatureProgram,
    *,
    available_columns: Sequence[str],
    entity_columns: Sequence[str] = (),
    forbidden_columns: Sequence[str] = (),
    max_depth: int = MAX_DEPTH,
    max_nodes: int = MAX_NODES,
    known_versions: Sequence[str] = (),
) -> list[ValidationIssue]:
    """Check a program without computing it. Empty result means it may run.

    Args:
        program: The tree to check.
        available_columns: Columns the *history* frame actually has.
        entity_columns: Columns the *prediction* frame has, for entity-scoped
            sources.
        forbidden_columns: Columns a feature may never read. This is where target
            leakage is blocked by name — a program that reads the label it is
            predicting will validate beautifully and be worthless.
        known_versions: Already-registered program versions. A collision means
            this program is semantically a duplicate of one that exists.

    Returns:
        Every issue found, not just the first. A generator revising a proposal
        wants the whole list; stopping at the first would make revision a
        guessing game.
    """
    issues: list[ValidationIssue] = []
    history = set(available_columns)
    entity = set(entity_columns)
    banned = set(forbidden_columns)

    if program.depth() > max_depth:
        issues.append(
            ValidationIssue(
                "excessive_depth",
                f"depth {program.depth()} exceeds the limit of {max_depth}; a program this "
                "deep is an unregularised model, not a feature",
            )
        )
    if program.node_count() > max_nodes:
        issues.append(
            ValidationIssue(
                "excessive_size",
                f"{program.node_count()} nodes exceeds the limit of {max_nodes}",
            )
        )

    for node in program.root.descendants():
        if isinstance(node, Source):
            pool = history if node.scope is SourceScope.HISTORY else entity
            label = node.scope.value
            if node.column not in pool:
                issues.append(
                    ValidationIssue(
                        "unknown_column",
                        f"{node.column!r} is not a column of the {label} frame",
                    )
                )
            if node.column in banned:
                issues.append(
                    ValidationIssue(
                        "target_leakage",
                        f"{node.column!r} is on the forbidden list; reading it would let "
                        "the feature see what it is meant to predict",
                    )
                )
        elif isinstance(node, ShrunkRate):
            for column in (node.numerator, node.denominator):
                if column not in history:
                    issues.append(
                        ValidationIssue(
                            "unknown_column", f"{column!r} is not a column of the history frame"
                        )
                    )
                if column in banned:
                    issues.append(
                        ValidationIssue("target_leakage", f"{column!r} is on the forbidden list")
                    )
        elif isinstance(node, TimeSince):
            if node.event_column not in history:
                issues.append(
                    ValidationIssue(
                        "unknown_column",
                        f"{node.event_column!r} is not a column of the history frame",
                    )
                )
        elif isinstance(node, ClusterRel) and known_versions:
            # Only checked when a version list is supplied; an experiment that has
            # not fitted clusters yet legitimately has none.
            pass

    if program.version() in set(known_versions):
        issues.append(
            ValidationIssue(
                "duplicate_semantics",
                f"a program computing exactly this is already registered as version "
                f"{program.version()[:12]}; identity here is semantic, so a new name does "
                "not make it a new feature",
            )
        )

    return issues


# --- row-level expressions ---------------------------------------------------


def _row_expr(node: Node) -> pl.Expr:
    """A ``ROW``-level node as a Polars expression over staged history columns."""
    if isinstance(node, Source):
        if node.scope is not SourceScope.HISTORY:
            raise ProgramError("an entity-scoped source cannot appear inside a window")
        return pl.col(node.column).cast(pl.Float64, strict=False)
    if isinstance(node, Const):
        return pl.lit(node.value, dtype=pl.Float64)
    if isinstance(node, Arith):
        return _apply_arith(node, _row_expr(node.left), _row_expr(node.right))
    if isinstance(node, Unary):
        return _apply_unary(node, _row_expr(node.child))
    raise ProgramError(
        f"{type(node).__name__} is not a row-level node; the level checker should have "
        "caught this before compilation"
    )


def _apply_arith(node: Arith, left: pl.Expr, right: pl.Expr) -> pl.Expr:
    if node.op is ArithOp.ADD:
        return left + right
    if node.op is ArithOp.SUB:
        return left - right
    if node.op is ArithOp.MUL:
        return left * right
    if node.op is ArithOp.SAFE_DIV:
        # A ratio over a vanishing denominator is **undefined, not enormous**, so
        # it is null rather than a large number.
        #
        # The tempting alternative, `left / (|right| + eps)`, is worse than it
        # looks: it turns a player's single 12-minute cameo into an elite per-90
        # rate, which is precisely the failure `shrunk_rate_as_of` was written to
        # prevent. Fabricating a finite value where the data supports none is how
        # a substitute ends up at the top of a ranking.
        return pl.when(right.abs() > node.epsilon).then(left / right).otherwise(None)
    if node.op is ArithOp.MIN:
        return pl.min_horizontal(left, right)
    return pl.max_horizontal(left, right)


def _apply_unary(node: Unary, child: pl.Expr) -> pl.Expr:
    if node.op is UnaryOp.LOG1P:
        # Guarded: log1p of anything at or below -1 is undefined, and a feature
        # that silently becomes NaN is worse than one that is honestly null.
        return pl.when(child > -1.0).then(child.log1p()).otherwise(None)
    if node.op is UnaryOp.NEG:
        return -child
    if node.op is UnaryOp.ABS:
        return child.abs()
    if node.op is UnaryOp.CLIP:
        return child.clip(lower_bound=node.lower, upper_bound=node.upper)
    raise ProgramError(f"{node.op.value} is an entity-level transform and cannot apply per row")


# --- temporal aggregation ----------------------------------------------------


def _aggregate(node: Node, values: pl.Expr) -> pl.Expr:
    """The aggregation a temporal node performs over its staged window."""
    if isinstance(node, Rolling):
        if node.agg is RollingAgg.MEAN:
            return values.mean()
        if node.agg is RollingAgg.MEDIAN:
            return values.median()
        if node.agg is RollingAgg.STD:
            return values.std()
        if node.agg is RollingAgg.MIN:
            return values.min()
        if node.agg is RollingAgg.MAX:
            return values.max()
        if node.agg is RollingAgg.SUM:
            return values.sum()
        if node.agg is RollingAgg.COUNT:
            return values.count().cast(pl.Float64)
        return values.quantile(node.quantile or 0.5)
    raise ProgramError(f"{type(node).__name__} has no simple aggregation")


def _stage_for(
    node: Node, entities: pl.DataFrame, ctx: CompileContext, window: int
) -> pl.DataFrame:
    return ctx.stage(
        entities,
        ctx.player_stats,
        entity_keys=ctx.entity_keys,
        prediction_time_col=ctx.prediction_time_col,
        available_time_col=ctx.available_time_col,
        window=window,
        order_col=ctx.order_col,
    )


def _temporal_column(
    node: Node, entities: pl.DataFrame, ctx: CompileContext, cache: ProgramCache
) -> pl.Series:
    """Evaluate one temporal node to one value per entity row."""
    key = _subtree_key(node)
    cached = cache.get(key)
    if cached is not None:
        return cached

    name = "__value"

    if isinstance(node, Rolling):
        staged = _stage_for(node, entities, ctx, node.window)
        values = _row_expr(node.child)
        summary = staged.group_by(ROW_ID).agg(
            _aggregate(node, values).alias(name),
            values.count().alias("__n"),
        )
        summary = summary.with_columns(
            pl.when(pl.col("__n") >= node.min_periods)
            .then(pl.col(name))
            .otherwise(None)
            .alias(name)
        )

    elif isinstance(node, Lag):
        staged = _stage_for(node, entities, ctx, node.periods)
        values = _row_expr(node.child)
        # rank is 0 for the newest record, so the k-th previous appearance is
        # rank k-1 inside a window of k.
        summary = staged.group_by(ROW_ID).agg(
            values.filter(pl.col(RANK) == node.periods - 1).first().alias(name)
        )

    elif isinstance(node, EwmMean):
        staged = _stage_for(node, entities, ctx, node.window)
        values = _row_expr(node.child)
        weight = (pl.lit(0.5) ** (pl.col(RANK).cast(pl.Float64) / node.halflife)).alias("__w")
        staged = staged.with_columns(weight)
        summary = staged.group_by(ROW_ID).agg(
            ((values * pl.col("__w")).sum() / pl.col("__w").sum()).alias(name)
        )

    elif isinstance(node, Trend):
        staged = _stage_for(node, entities, ctx, node.window)
        values = _row_expr(node.child)
        # Time runs forward: the oldest visible record is the lowest t. Using
        # -rank rather than rank is what makes a positive slope mean "improving"
        # instead of the reverse.
        t = (-pl.col(RANK).cast(pl.Float64)).alias("__t")
        staged = staged.with_columns(t, values.alias("__y"))
        summary = staged.group_by(ROW_ID).agg(
            pl.col("__t").mean().alias("__tm"),
            pl.col("__y").mean().alias("__ym"),
            (pl.col("__t") * pl.col("__y")).mean().alias("__tym"),
            (pl.col("__t") ** 2).mean().alias("__t2m"),
            pl.col("__y").count().alias("__n"),
        )
        variance = pl.col("__t2m") - pl.col("__tm") ** 2
        summary = summary.with_columns(
            pl.when((pl.col("__n") >= 3) & (variance.abs() > 1e-12))
            .then((pl.col("__tym") - pl.col("__tm") * pl.col("__ym")) / variance)
            .otherwise(None)
            .alias(name)
        )

    elif isinstance(node, ShrunkRate):
        staged = _stage_for(node, entities, ctx, node.window)
        numerator = pl.col(node.numerator).cast(pl.Float64, strict=False).sum()
        denominator = pl.col(node.denominator).cast(pl.Float64, strict=False).sum()
        summary = staged.group_by(ROW_ID).agg(numerator.alias("__num"), denominator.alias("__den"))
        prior_mass = node.prior_strength * node.scale
        summary = summary.with_columns(
            (
                pl.col("__num").fill_null(0.0)
                / (pl.col("__den").fill_null(0.0) + prior_mass)
                * node.scale
            ).alias(name)
        )

    elif isinstance(node, TimeSince):
        summary = _time_since(node, entities, ctx, name)

    else:  # pragma: no cover - guarded by the level checker
        raise ProgramError(f"{type(node).__name__} is not a temporal node")

    values_out = (
        entities.with_row_index(ROW_ID)
        .join(summary.select([ROW_ID, name]), on=ROW_ID, how="left")
        .sort(ROW_ID)[name]
        .cast(pl.Float64, strict=False)
    )

    if isinstance(node, ShrunkRate):
        # With no visible history the posterior is exactly the prior, which is
        # zero here — never null, so cold-start players stay rankable. This
        # mirrors the shipped catalogue's behaviour rather than inventing a
        # second convention.
        values_out = values_out.fill_null(0.0)

    cache.put(key, values_out)
    return values_out


def _time_since(
    node: TimeSince, entities: pl.DataFrame, ctx: CompileContext, name: str
) -> pl.DataFrame:
    """Days since the last visible qualifying event, per entity row.

    Joined directly rather than staged, because "how long ago" is not bounded by
    a window count: a player who has not appeared in thirty matches still has an
    answer, and a windowed stage would silently report the wrong one.

    Point-in-time safe by the same rule as everything else — only records whose
    ``available_time`` precedes the row's cutoff are visible.
    """
    keys = list(ctx.entity_keys)
    needed = [*keys, ctx.event_time_col, ctx.available_time_col, node.event_column]
    indexed = entities.with_row_index(ROW_ID)
    joined = indexed.select([ROW_ID, *keys, ctx.prediction_time_col]).join(
        ctx.player_stats.select(needed), on=keys, how="left"
    )

    visible = joined.filter(
        pl.col(ctx.available_time_col).is_not_null()
        & (pl.col(ctx.available_time_col) < pl.col(ctx.prediction_time_col))
    )
    if node.require_positive:
        visible = visible.filter(pl.col(node.event_column).cast(pl.Float64, strict=False) > 0.0)

    return (
        visible.with_columns(
            (
                (pl.col(ctx.prediction_time_col) - pl.col(ctx.event_time_col)).dt.total_seconds()
                / _SECONDS_PER_DAY
            ).alias("__age")
        )
        .group_by(ROW_ID)
        .agg(pl.col("__age").min().alias(name))
    )


# --- entity-level evaluation -------------------------------------------------


def _entity_series(
    node: Node, entities: pl.DataFrame, ctx: CompileContext, cache: ProgramCache
) -> pl.Series:
    """Evaluate an ``ENTITY``-level node to one value per prediction row."""
    key = _subtree_key(node)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if isinstance(node, (Rolling, Lag, EwmMean, Trend, ShrunkRate, TimeSince)):
        return _temporal_column(node, entities, ctx, cache)

    if isinstance(node, Source):
        if node.scope is not SourceScope.ENTITY:
            raise ProgramError(
                f"{node.column!r} is a history column and cannot be read directly at "
                "entity level; wrap it in a window"
            )
        if node.column not in entities.columns:
            raise ProgramError(f"entity frame has no column {node.column!r}")
        values = entities[node.column].cast(pl.Float64, strict=False)

    elif isinstance(node, Const):
        values = pl.Series(name="__const", values=[node.value] * entities.height, dtype=pl.Float64)

    elif isinstance(node, Arith):
        left = _entity_series(node.left, entities, ctx, cache)
        right = _entity_series(node.right, entities, ctx, cache)
        frame = pl.DataFrame({"__l": left, "__r": right})
        values = frame.select(_apply_arith(node, pl.col("__l"), pl.col("__r")).alias("__v"))["__v"]

    elif isinstance(node, Unary):
        child = _entity_series(node.child, entities, ctx, cache)
        values = _entity_unary(node, child)

    elif isinstance(node, GroupRel):
        values = _group_relative(node, entities, ctx, cache)

    elif isinstance(node, ClusterRel):
        values = _cluster_relative(node, entities, ctx, cache)

    else:  # pragma: no cover
        raise ProgramError(f"cannot evaluate {type(node).__name__}")

    values = values.cast(pl.Float64, strict=False)
    cache.put(key, values)
    return values


def _f(value: object) -> float:
    """Narrow a Polars aggregate to a float, treating an empty result as zero.

    Polars types its aggregations as a union that includes ``date`` and
    ``timedelta``, because the identical call on a temporal column returns those.
    Every column reaching here has been cast to ``Float64``, so the narrowing is
    sound; it is stated once rather than scattered as casts at each call site —
    the same convention ``evaluation.accuracy`` already uses for the same reason.
    """
    return 0.0 if value is None else float(value)  # type: ignore[arg-type]


def _entity_unary(node: Unary, child: pl.Series) -> pl.Series:
    frame = pl.DataFrame({"__c": child})
    if node.op is UnaryOp.ZSCORE:
        mean = _f(child.mean())
        std = _f(child.std())
        if std < 1e-12:
            # Every player is identical on this dimension, so there is no
            # standing to report. Null rather than zero: zero would assert
            # "exactly average", which is a claim about a distribution that has
            # no spread to be average within.
            return pl.Series(name="__v", values=[None] * child.len(), dtype=pl.Float64)
        return ((child - mean) / std).rename("__v")
    if node.op is UnaryOp.PERCENTILE_RANK:
        non_null = child.drop_nulls().len()
        if non_null < 2:
            return pl.Series(name="__v", values=[None] * child.len(), dtype=pl.Float64)
        # Nulls stay null. Polars ranks them rather than skipping, and a player
        # with no measured value would otherwise be handed a percentile — an
        # invented standing among players he was never compared to.
        return frame.select(
            pl.when(pl.col("__c").is_not_null())
            .then((pl.col("__c").rank(method="average") - 1.0) / (non_null - 1))
            .otherwise(None)
            .alias("__v")
        )["__v"]
    return frame.select(_apply_unary(node, pl.col("__c")).alias("__v"))["__v"]


def _group_relative(
    node: GroupRel, entities: pl.DataFrame, ctx: CompileContext, cache: ProgramCache
) -> pl.Series:
    """Express a value relative to the group it sits in, within this batch."""
    child = _entity_series(node.child, entities, ctx, cache)
    column = ctx.group_column(node.by)

    if column is None:
        frame = pl.DataFrame({"__c": child}).with_columns(pl.lit(0).alias("__g"))
    else:
        if column not in entities.columns:
            raise ProgramError(
                f"cannot group by {node.by.value}: the entity frame has no {column!r} column"
            )
        frame = pl.DataFrame({"__c": child, "__g": entities[column]})

    over = pl.col("__c").over("__g")
    if node.op is GroupRelOp.RANK:
        count = pl.col("__c").count().over("__g")
        expression = (
            pl.when(count > 1)
            .then((pl.col("__c").rank(method="average").over("__g") - 1.0) / (count - 1))
            .otherwise(None)
        )
    elif node.op is GroupRelOp.SHARE:
        total = over.sum()
        expression = pl.when(total.abs() > 1e-12).then(pl.col("__c") / total).otherwise(None)
    elif node.op is GroupRelOp.DEV_FROM_MEAN:
        expression = pl.col("__c") - over.mean()
    else:
        spread = over.std()
        expression = (
            pl.when(spread > 1e-12).then((pl.col("__c") - over.mean()) / spread).otherwise(None)
        )

    return frame.select(expression.alias("__v"))["__v"]


def _cluster_relative(
    node: ClusterRel, entities: pl.DataFrame, ctx: CompileContext, cache: ProgramCache
) -> pl.Series:
    """Express a value relative to the player's cluster, or gate it by membership."""
    child = _entity_series(node.child, entities, ctx, cache)
    memberships = ctx.cluster_memberships
    if memberships is None:
        raise ProgramError(
            "this program is cluster-conditioned but no cluster memberships were "
            "supplied; a gated feature computed without its gate would silently "
            "become a global one"
        )

    keys = list(ctx.entity_keys)
    scoped = memberships.filter(pl.col("cluster_model_version") == node.cluster_model_version)
    if scoped.is_empty():
        raise ProgramError(
            f"no memberships for cluster model {node.cluster_model_version!r}; a cluster "
            "id is meaningless without the model that produced it"
        )

    if node.op is ClusterRelOp.GATE:
        wanted = scoped.filter(pl.col("cluster_id") == node.cluster_id).select(
            [*keys, "membership_probability"]
        )
        joined = (
            entities.select(keys)
            .with_row_index(ROW_ID)
            .join(wanted, on=keys, how="left")
            .sort(ROW_ID)
        )
        # A player with no membership row for this cluster belongs to it with
        # probability zero, so the gate closes. That is the correct reading of a
        # missing soft assignment.
        probability = joined["membership_probability"].fill_null(0.0).cast(pl.Float64)
        return (child * probability).rename("__v")

    # Rank or deviation within the player's own assigned cluster.
    assignment = (
        scoped.sort([*keys, "membership_probability"], descending=[*[False] * len(keys), True])
        .unique(subset=keys, keep="first", maintain_order=True)
        .select([*keys, "cluster_id", "distance_to_centroid"])
    )
    joined = (
        entities.select(keys)
        .with_row_index(ROW_ID)
        .join(assignment, on=keys, how="left")
        .sort(ROW_ID)
    )
    frame = pl.DataFrame({"__c": child, "__g": joined["cluster_id"].fill_null(-1)})

    if node.op is ClusterRelOp.RANK:
        count = pl.col("__c").count().over("__g")
        expression = (
            pl.when(count > 1)
            .then((pl.col("__c").rank(method="average").over("__g") - 1.0) / (count - 1))
            .otherwise(None)
        )
    else:
        expression = pl.col("__c") - pl.col("__c").mean().over("__g")

    return frame.select(expression.alias("__v"))["__v"]


# --- entry points ------------------------------------------------------------


def compile_program(
    program: FeatureProgram,
    entities: pl.DataFrame,
    ctx: CompileContext,
    *,
    cache: ProgramCache | None = None,
) -> pl.DataFrame:
    """Compute one program, returning ``entities`` plus one named column.

    Row order is preserved. Rows the program cannot answer for carry nulls rather
    than being dropped: a missing observation is information, and losing the row
    would bias every downstream aggregate.
    """
    if program.root.level is not Level.ENTITY:  # pragma: no cover - constructor enforces
        raise ProgramError("a program root must be entity level")
    if program.name in entities.columns:
        raise ProgramError(
            f"{program.name!r} already exists on the entity frame; overwriting it would "
            "make the feature set differ from the version string that describes it"
        )
    values = _entity_series(program.root, entities, ctx, cache or ProgramCache())
    return entities.with_columns(values.rename(program.name))


def compile_programs(
    programs: Sequence[FeatureProgram],
    entities: pl.DataFrame,
    ctx: CompileContext,
) -> tuple[pl.DataFrame, ProgramCache]:
    """Compute several programs over one entity frame, sharing sub-expressions.

    Returns the enriched frame and the cache, so a caller can report how much was
    reused. Programs are computed in the given order; a failure names the program
    rather than surfacing an opaque Polars error from inside a shared subtree.
    """
    cache = ProgramCache()
    frame = entities
    for program in programs:
        try:
            frame = compile_program(program, frame, ctx, cache=cache)
        except ProgramError:
            raise
        except Exception as exc:
            raise ProgramError(f"failed computing {program.name!r}: {exc}") from exc
    return frame, cache


def program_builder(
    program: FeatureProgram, ctx_factory: Any
) -> Any:  # pragma: no cover - thin adapter
    """Adapt a program to the ``FeatureBuilder`` signature the leakage harness takes.

    ``xg_alonso.features.leakage`` drives ``(entities, source) -> features``. This
    wraps a program so the existing harness can prove it point-in-time safe
    without the harness learning anything about the DSL.
    """

    def build(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
        return compile_program(program, entities, ctx_factory(source))

    return build
