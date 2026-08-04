"""What a feature is worth to a *decision*, not to a regression.

:class:`~xg_alonso.discovery.utility.MetricRegistry` has existed since the
discovery loop was written, described in its own docstring as "the extension
point named in the brief", with **nothing registered in it**. The consequence
was quiet and serious: ``objective_gain`` fell back to ``predictive_gain``, so
every objective scored a candidate identically and the loop's central claim —
that different objectives value different features — was true of the design and
false of the running code. ``decision_gain`` and ``turnover_penalty`` were
passed as permanently ``unmeasured``. This module is what fills it.

Why it lives here and not beside the registry
---------------------------------------------

``utility.py`` is on the ``discovery-core-is-generic`` forbidden-source list in
``.importlinter``: the search engine may not know what a football is. That
contract is what makes the loop reusable, and it would be a poor trade to break
it for the sake of proximity. So the registry stays generic and the football
arrives from outside it — which is the extension story that module claimed and
had never once demonstrated. This module sits above ``evaluation`` and
``optimization`` in the layering and may import both.

What makes these decision metrics
---------------------------------

**Every one is computed on the predicted top-k, not on the whole pool.** That is
the difference between "how accurate is this model" and "how good are the picks
it leads to". A model 5% worse on MAE across six hundred players that gets the
top twenty right scores *better* here — which is CLAUDE.md's "a slightly worse
regression model that produces better transfer decisions is preferred" made
executable rather than asserted, and
``tests/discovery/test_decision_metrics.py`` pins exactly that case.

**And the top-k is drawn from the reachable rows.** When a
:class:`~xg_alonso.discovery.feasible.FeasiblePool` has been applied, the
predictions handed here already exclude players this manager cannot buy, so
"the best twenty" means "the best twenty you could actually own". Two managers
therefore get genuinely different ``objective_gain`` from an identical model and
an identical feature. That single composition is the technical content of the
whole context-conditioning claim.

Cost
----

Zero extra model fits. :func:`~xg_alonso.discovery.harness.fold_predictions`
reads the same cache the subgroup breakdown already populated. A decision metric
that required its own backtest would roughly double every run, and a metric too
expensive to run is an unmeasured one with extra steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Final

import numpy as np

from xg_alonso.contracts.objective import PrimaryMetric
from xg_alonso.discovery.harness import FoldPredictions
from xg_alonso.discovery.utility import MetricContext, MetricRegistry, rank_correlation

__all__ = [
    "DECISION_METRICS",
    "TOP_K",
    "build_metric_context",
    "decision_gain",
    "objective_gain",
    "register_decision_metrics",
    "turnover_penalty",
]

TOP_K: Final[int] = 20
"""How many players count as "the picks".

Roughly the size of the shortlist a manager actually considers: a squad is
fifteen, and the handful of realistic alternatives to the two or three players
they might move takes it to about twenty. Small enough that ranking errors
outside it genuinely do not matter, large enough not to be dominated by a single
lucky haul.
"""

_TEMPLATE_SHARE: Final[float] = 0.20
"""Ownership above which a player is "template" for differential purposes."""

_MIN_ROWS: Final[int] = 40
"""Below this a top-k statistic describes the sample, not the population."""


def _top_k(predicted: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Indices of the ``k`` highest predictions, best first."""
    if predicted.size == 0:
        return np.empty(0, dtype=np.int64)
    width = min(k, predicted.size)
    # `argpartition` then sort the survivors: O(n) rather than a full sort, and
    # the loop runs once per candidate per fold.
    cut = np.argpartition(-predicted, width - 1)[:width]
    return cut[np.argsort(-predicted[cut])]


def _ownership(context: MetricContext) -> np.ndarray | None:
    return context.extra("ownership")


# --- the seven metrics -------------------------------------------------------
#
# One per `PrimaryMetric`. Each returns a scalar where higher is better, so the
# registry never needs to know a metric's orientation.


def expected_points(context: MetricContext) -> float:
    """Realised points of the picks. The default question, asked of decisions."""
    picks = _top_k(context.predicted)
    if picks.size == 0:
        return 0.0
    return float(np.mean(context.actual[picks]))


def expected_rank_gain(context: MetricContext) -> float:
    """Ownership-weighted differential of the picks.

    **A proxy, not a rank** — the same proxy
    :class:`~xg_alonso.contracts.objective.PrimaryMetric` documents. Points
    scored by players the field does not own move a manager up; points from
    template players do not. Falls back to plain realised points when no
    ownership is supplied, because a silent zero would read as "every pick was
    template", which is a confident and usually wrong claim.
    """
    picks = _top_k(context.predicted)
    if picks.size == 0:
        return 0.0
    owned = _ownership(context)
    if owned is None:
        return float(np.mean(context.actual[picks]))
    share = np.clip(owned[picks], 0.0, 1.0)
    return float(np.mean(context.actual[picks] * (1.0 - share)))


def downside_protection(context: MetricContext) -> float:
    """The 10th percentile of the picks' realised points.

    Maximising the lower tail rather than the mean. A set of picks that averages
    well because two of them hauled while eight blanked is not what a manager
    protecting a rank asked for.
    """
    picks = _top_k(context.predicted)
    if picks.size == 0:
        return 0.0
    return float(np.percentile(context.actual[picks], 10))


def captaincy_upside(context: MetricContext) -> float:
    """Realised points of the single highest-predicted player, per gameweek.

    Averaged over gameweeks rather than pooled, because the armband is chosen
    once a week and a metric pooled across weeks would reward a model that ranks
    one enormous haul first and everything else badly.
    """
    gameweeks = context.groups.get("gameweek")
    if gameweeks is None:
        best = _top_k(context.predicted, 1)
        return float(context.actual[best][0]) if best.size else 0.0

    scores: list[float] = []
    for week in np.unique(gameweeks):
        rows = np.flatnonzero(gameweeks == week)
        if rows.size == 0:
            continue
        chosen = rows[int(np.argmax(context.predicted[rows]))]
        scores.append(float(context.actual[chosen]))
    return float(np.mean(scores)) if scores else 0.0


def differential_yield(context: MetricContext) -> float:
    """Realised points of the picks, counting only genuine differentials.

    Stricter than :func:`expected_rank_gain`: a template player contributes
    nothing at all rather than being discounted, because a manager asking for
    differentials is not asking for a slightly-less-owned template.
    """
    picks = _top_k(context.predicted)
    if picks.size == 0:
        return 0.0
    owned = _ownership(context)
    if owned is None:
        return float(np.mean(context.actual[picks]))
    differential = owned[picks] < _TEMPLATE_SHARE
    if not differential.any():
        return 0.0
    return float(np.mean(context.actual[picks][differential]))


def team_value_growth(context: MetricContext) -> float:
    """Agreement between the ranking and realised transfer momentum.

    **Momentum, not a price forecast** — decision D11 defers the price model and
    this scores the published leading indicator instead. Requires
    ``extras['transfer_momentum']``; without it there is nothing to agree with
    and the honest answer is zero rather than a substituted proxy.
    """
    momentum = context.extra("transfer_momentum")
    if momentum is None or momentum.size != context.predicted.size:
        return 0.0
    return float(rank_correlation(context.predicted, momentum))


def transfer_flexibility(context: MetricContext) -> float:
    """Negative churn of the ranking across gameweeks.

    A feature that reorders the pool every week makes the optimizer propose a
    transfer every week, and paying four points for noise is the most expensive
    way to use a model. Negated so that, like every other metric here, higher is
    better.
    """
    gameweeks = context.groups.get("gameweek")
    if gameweeks is None:
        return 0.0
    return -_ranking_churn(context.predicted, gameweeks, context.groups.get("entity"))


_METRIC_BY_PRIMARY: Final[dict[str, object]] = {
    PrimaryMetric.EXPECTED_POINTS.value: expected_points,
    PrimaryMetric.EXPECTED_RANK_GAIN.value: expected_rank_gain,
    PrimaryMetric.DOWNSIDE_PROTECTION.value: downside_protection,
    PrimaryMetric.CAPTAINCY_UPSIDE.value: captaincy_upside,
    PrimaryMetric.DIFFERENTIAL_YIELD.value: differential_yield,
    PrimaryMetric.TEAM_VALUE_GROWTH.value: team_value_growth,
    PrimaryMetric.TRANSFER_FLEXIBILITY.value: transfer_flexibility,
}


def register_decision_metrics(registry: MetricRegistry) -> None:
    """Register one metric per :class:`PrimaryMetric`.

    Raises if a name is already present — the registry refuses silent
    replacement, because swapping a metric would change what every past
    evaluation of that objective meant.
    """
    for name, metric in _METRIC_BY_PRIMARY.items():
        if registry.get(name) is None:
            registry.register(name, metric)  # type: ignore[arg-type]


DECISION_METRICS: Final[MetricRegistry] = MetricRegistry()
register_decision_metrics(DECISION_METRICS)

# Every `PrimaryMetric` must resolve, or an objective silently falls back to the
# predictive gain again — the exact defect this module exists to remove.
assert set(_METRIC_BY_PRIMARY) == {m.value for m in PrimaryMetric}


# --- ranking churn -----------------------------------------------------------


def _ranking_churn(values: np.ndarray, gameweeks: np.ndarray, entities: np.ndarray | None) -> float:
    """Mean one-minus-correlation of the ranking between consecutive gameweeks.

    **Aligned on entity, which the generic
    :func:`~xg_alonso.discovery.utility.turnover_score` cannot do.** That
    function correlates two arrays positionally, which is correct only when both
    hold the same entities in the same order. Consecutive FPL gameweeks do not:
    players are injured, rested and rotated, so the row sets differ. Feeding it
    raw per-gameweek slices would measure *roster* churn and report it as
    *ranking* churn — a real defect that surfaces only when someone actually
    calls it, which until now nobody had.

    Aligning here on the players present in both weeks measures the thing the
    name claims. Without an entity column there is nothing to align on, and the
    honest answer is zero rather than a positional comparison of unrelated rows.
    """
    if entities is None or gameweeks.size == 0:
        return 0.0

    order = sorted({int(week) for week in np.unique(gameweeks)})
    churns: list[float] = []
    for previous, current in pairwise(order):
        left = {
            int(entity): float(value)
            for entity, value in zip(
                entities[gameweeks == previous], values[gameweeks == previous], strict=True
            )
        }
        right = {
            int(entity): float(value)
            for entity, value in zip(
                entities[gameweeks == current], values[gameweeks == current], strict=True
            )
        }
        shared = sorted(set(left) & set(right))
        if len(shared) < 2:
            continue
        correlation = rank_correlation(
            np.array([left[e] for e in shared]), np.array([right[e] for e in shared])
        )
        churns.append((1.0 - correlation) / 2.0)
    return float(np.mean(churns)) if churns else 0.0


# --- building a context from cached predictions ------------------------------


def build_metric_context(folds: Sequence[FoldPredictions]) -> MetricContext | None:
    """Pool per-fold predictions into one context, carrying the FPL extras.

    Returns ``None`` when too few rows survive to say anything, which the
    caller reports as "not measured" rather than as a measured zero.
    """
    if not folds:
        return None

    predicted = np.concatenate([f.predicted for f in folds])
    actual = np.concatenate([f.truth for f in folds])
    if predicted.size < _MIN_ROWS:
        return None

    def gather(column: str) -> np.ndarray | None:
        parts = [f.column(column) for f in folds]
        if any(part is None for part in parts):
            return None
        stacked = np.concatenate([part for part in parts if part is not None])
        return stacked if stacked.size == predicted.size else None

    extras: dict[str, np.ndarray] = {}
    ownership = gather("selected_mean_5")
    if ownership is not None:
        # `selected_mean_5` is a percentage in the source data; the metrics want
        # a share. Converted once, here, rather than in each metric.
        extras["ownership"] = np.asarray(ownership, dtype=np.float64) / 100.0
    momentum = gather("transfers_balance_mean_5")
    if momentum is not None:
        extras["transfer_momentum"] = np.asarray(momentum, dtype=np.float64)

    groups: dict[str, np.ndarray] = {}
    # A gameweek must be unique across seasons or week 3 of two seasons pools
    # into one, so the fold index disambiguates.
    weeks = gather("label_gameweek")
    if weeks is not None:
        offsets = np.concatenate([np.full(f.predicted.size, f.fold_index * 1000) for f in folds])
        groups["gameweek"] = np.asarray(weeks, dtype=np.int64) + offsets
    entity = gather("player_code")
    if entity is not None:
        groups["entity"] = np.asarray(entity, dtype=np.int64)

    return MetricContext(predicted=predicted, actual=actual, extras=extras, groups=groups)


# --- the three terms the loop was missing ------------------------------------


def _relative(candidate: float, baseline: float) -> float:
    """Candidate over baseline, as a signed relative change."""
    if abs(baseline) < 1e-12:
        return 0.0
    return (candidate - baseline) / abs(baseline)


def objective_gain(
    metric_name: str,
    baseline: MetricContext | None,
    candidate: MetricContext | None,
    *,
    registry: MetricRegistry = DECISION_METRICS,
) -> float | None:
    """How much the candidate improved *this objective's own* metric.

    ``None`` when it could not be measured, so the caller can report it as
    unmeasured rather than as zero. That distinction is the whole reason
    :attr:`~xg_alonso.discovery.utility.UtilityBreakdown.unmeasured` exists.
    """
    metric = registry.get(metric_name)
    if metric is None or baseline is None or candidate is None:
        return None
    return _relative(metric(candidate), metric(baseline))


def decision_gain(baseline: MetricContext | None, candidate: MetricContext | None) -> float | None:
    """Improvement in the realised points of the picks the model would make.

    Deliberately **not** a policy backtest. Simulating squads through
    :func:`~xg_alonso.evaluation.walk_forward` would mean hundreds of solves per
    candidate feature, inside a loop already fitting nine models per candidate —
    it would make ``xg discover`` unusable, and an unusable metric is an
    unmeasured one with extra steps.

    What it does measure is a concrete pick from the manager's own option set,
    scored against what actually happened. It ignores hit costs and squad
    legality beyond the reachability mask, which is stated here rather than
    implied by the name.
    """
    if baseline is None or candidate is None:
        return None
    return _relative(expected_points(candidate), expected_points(baseline))


def turnover_penalty(candidate: MetricContext | None) -> float | None:
    """How much the candidate's ranking churns week to week, in [0, 1]."""
    if candidate is None:
        return None
    gameweeks = candidate.groups.get("gameweek")
    if gameweeks is None:
        return None
    return _ranking_churn(candidate.predicted, gameweeks, candidate.groups.get("entity"))
