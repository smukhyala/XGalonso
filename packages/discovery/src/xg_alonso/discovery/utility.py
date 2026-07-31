"""Scoring a feature under an objective, rather than against a global metric.

The premise of this package in one function. ``feature_utility`` combines what a
feature does to *predictions*, what it does to *decisions*, and what it does to
the specific quantity the manager cares about, then charges it for complexity,
missingness, churn and leakage risk::

    utility =  w_prediction      * predictive_gain
             + w_decision        * decision_quality_gain
             + w_objective       * objective_specific_gain
             + w_stability       * temporal_stability
             + w_complementarity * conditional_incremental_gain
             - w_complexity      * complexity
             - w_missingness     * missingness
             - w_turnover        * turnover
             - w_leakage         * leakage_risk

**Why not just RMSE.** `CLAUDE.md` is explicit that a slightly worse regression
model producing better transfer decisions is preferred. A global error metric
cannot express that, and it is actively misleading for two of the objectives here:
a feature that sharpens the mean while flattening the tail improves RMSE and
makes a rank-chaser strictly worse off, because the tail is the only thing that
moves rank.

**Every term is returned, not just the total.** :class:`UtilityBreakdown` carries
each contribution, so "why was this accepted" has an arithmetic answer rather
than a narrative one. A scalar that cannot be decomposed is a scalar nobody can
argue with.

**New objectives do not touch this module.** An objective supplies its own
:class:`ObjectiveMetric` through :class:`MetricRegistry`; the search engine never
learns what the metric means.

This module is domain-free, and ``.importlinter`` keeps it that way.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Final, Protocol

import numpy as np

from xg_alonso.contracts.discovery import FoldMetrics
from xg_alonso.contracts.objective import UtilityWeights

__all__ = [
    "MetricContext",
    "MetricRegistry",
    "ObjectiveMetric",
    "UtilityBreakdown",
    "calibration_error",
    "feature_utility",
    "log_loss",
    "mae",
    "poisson_deviance",
    "rank_correlation",
    "rmse",
    "stability_score",
    "top_k_precision",
    "turnover_score",
]

#: Below this spread a vector is one number repeated and every metric over it is
#: uninterpretable. Matches the constant `evaluation.accuracy` uses, for the same
#: reason: a constant predictor must never be able to score well.
DEGENERATE_SD: Final[float] = 1e-6


# --- predictive metrics ------------------------------------------------------


def _clean(predicted: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either side is missing. Never impute."""
    mask = np.isfinite(predicted) & np.isfinite(actual)
    return predicted[mask], actual[mask]


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    p, a = _clean(predicted, actual)
    return float(np.mean(np.abs(p - a))) if p.size else float("nan")


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    p, a = _clean(predicted, actual)
    return float(np.sqrt(np.mean((p - a) ** 2))) if p.size else float("nan")


def poisson_deviance(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean Poisson deviance. The natural error for a count.

    Used rather than MAE wherever the label is a zero-inflated count, because MAE
    on such a label is minimised by predicting zero — the failure that once let a
    constant-output model score +48% "skill" in this repository.
    """
    p, a = _clean(predicted, actual)
    if not p.size:
        return float("nan")
    p = np.clip(p, 1e-9, None)
    a = np.clip(a, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(a > 0, a * np.log(a / p), 0.0)
    return float(2.0 * np.mean(term - (a - p)))


def log_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Binary cross-entropy, for probability outputs."""
    p, a = _clean(predicted, actual)
    if not p.size:
        return float("nan")
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    a = (a > 0).astype(np.float64)
    return float(-np.mean(a * np.log(p) + (1.0 - a) * np.log(1.0 - p)))


def calibration_error(predicted: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float:
    """Expected calibration error over equal-count bins.

    A model can rank perfectly and still be systematically overconfident, which
    matters here because the optimizer subtracts a hit cost from a predicted
    gain: a projection that says +5 and delivers +1 spends transfers it should
    not have.
    """
    p, a = _clean(predicted, actual)
    if p.size < bins:
        return float("nan")
    order = np.argsort(p)
    p, a = p[order], a[order]
    chunks = np.array_split(np.arange(p.size), bins)
    total = 0.0
    for chunk in chunks:
        if chunk.size:
            total += chunk.size * abs(float(np.mean(p[chunk])) - float(np.mean(a[chunk])))
    return total / p.size


def rank_correlation(
    predicted: Sequence[float] | np.ndarray, actual: Sequence[float] | np.ndarray
) -> float:
    """Spearman rank correlation, ties shared.

    **A deliberate second implementation.** ``evaluation.accuracy.spearman``
    computes the same quantity over Polars series, and this package may not
    import ``evaluation`` — the boundary that keeps the search engine reusable
    outside football. Rather than weaken the boundary or route a numeric helper
    through the contracts layer, the function is restated here over numpy and
    ``tests/discovery/test_utility.py`` asserts the two agree on shared input.
    A mirrored implementation with a drift test is honest; an undetected fork is
    not.
    """
    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if p.size < 2 or p.size != a.size:
        return 0.0
    p, a = _clean(p, a)
    if p.size < 2:
        return 0.0
    rp, ra = _average_ranks(p), _average_ranks(a)
    sp, sa = float(np.std(rp)), float(np.std(ra))
    if sp < DEGENERATE_SD or sa < DEGENERATE_SD:
        return 0.0
    return float(np.mean((rp - rp.mean()) * (ra - ra.mean())) / (sp * sa))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, 1-based."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Share ranks within each run of equal values.
    sorted_values = values[order]
    start = 0
    for index in range(1, sorted_values.size + 1):
        if index == sorted_values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = float(np.mean(ranks[order[start:index]]))
            start = index
    return ranks


def top_k_precision(predicted: np.ndarray, actual: np.ndarray, *, k: int) -> float:
    """Share of the predicted top ``k`` that were genuinely in the actual top ``k``.

    The metric closest to the product: a transfer promotes a player on the
    strength of their rank, so what matters is whether the highly-ranked players
    delivered, not whether their point estimates were close.
    """
    p, a = _clean(predicted, actual)
    k = min(k, p.size)
    if k == 0:
        return 0.0
    picked = set(np.argsort(-p, kind="stable")[:k].tolist())
    best = set(np.argsort(-a, kind="stable")[:k].tolist())
    return len(picked & best) / k


# --- stability and churn -----------------------------------------------------


def stability_score(folds: Sequence[FoldMetrics]) -> float:
    """How consistently a feature helped across folds, in [0, 1].

    The win rate, scaled down when the improvements themselves are erratic::

        stability = win_rate * (1 - min(1, spread / (|mean| + eps)))

    Both halves are needed. A feature that improved four folds of five has a
    0.8 win rate whether the gains were (0.01, 0.01, 0.01, 0.01) or
    (0.001, 0.001, 0.001, 0.4) — and the second is one lucky fold wearing a
    pattern's clothes. Penalising the spread separates them.

    Returns 0.0 for fewer than two folds: a single fold has no consistency to
    measure, and reporting 1.0 would let a one-fold result look maximally stable.
    """
    if len(folds) < 2:
        return 0.0
    improvements = np.array([f.relative_improvement for f in folds], dtype=np.float64)
    win_rate = float(np.mean(improvements > 0.0))
    if win_rate == 0.0:
        return 0.0
    centre = float(np.mean(improvements))
    spread = float(np.std(improvements))
    dispersion = spread / (abs(centre) + 1e-9)
    return float(win_rate * max(0.0, 1.0 - min(1.0, dispersion)))


def turnover_score(values_by_period: Sequence[np.ndarray]) -> float:
    """How much a feature's ranking of the pool churns between periods, in [0, 1].

    Zero means the ordering is unchanged week to week; one means it is
    reshuffled. High turnover is not automatically bad — form genuinely moves —
    but a feature that reorders the entire pool every week will make the
    optimizer propose a transfer every week, and paying four points for noise is
    the most expensive way to use a model.
    """
    usable = [v for v in values_by_period if v.size > 1]
    if len(usable) < 2:
        return 0.0
    churns: list[float] = []
    for previous, current in pairwise(usable):
        width = min(previous.size, current.size)
        correlation = rank_correlation(previous[:width], current[:width])
        churns.append((1.0 - correlation) / 2.0)
    return float(np.mean(churns)) if churns else 0.0


# --- objective-specific metrics ---------------------------------------------


@dataclass(frozen=True)
class MetricContext:
    """Everything an objective metric may read.

    Deliberately generic: arrays and named extras, no football types. An FPL
    metric reads ``extras["ownership"]``; a metric for some other domain reads
    whatever that domain put there. The engine never inspects the contents.
    """

    predicted: np.ndarray
    actual: np.ndarray
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)
    groups: Mapping[str, np.ndarray] = field(default_factory=dict)
    scalars: Mapping[str, float] = field(default_factory=dict)

    def extra(self, name: str) -> np.ndarray | None:
        return self.extras.get(name)

    def require(self, name: str) -> np.ndarray:
        found = self.extras.get(name)
        if found is None:
            raise KeyError(
                f"metric needs {name!r} but the context does not carry it; a metric "
                "that silently substitutes a default measures something other than "
                "what it claims"
            )
        return found


class ObjectiveMetric(Protocol):
    """A scalar an objective wants maximised. Higher is always better."""

    name: str

    def __call__(self, context: MetricContext) -> float: ...


@dataclass
class MetricRegistry:
    """Objective id or primary-metric name to its scoring function.

    The extension point named in the brief: a new objective registers a metric
    here and the search engine is untouched.
    """

    metrics: dict[str, Callable[[MetricContext], float]] = field(default_factory=dict)

    def register(self, name: str, metric: Callable[[MetricContext], float]) -> None:
        if name in self.metrics:
            raise ValueError(
                f"a metric named {name!r} is already registered; silently replacing it "
                "would change what every past evaluation of that objective meant"
            )
        self.metrics[name] = metric

    def get(self, name: str) -> Callable[[MetricContext], float] | None:
        return self.metrics.get(name)

    def score(self, name: str, context: MetricContext) -> float:
        metric = self.get(name)
        if metric is None:
            raise KeyError(f"no metric registered for {name!r}")
        return metric(context)


# --- the utility score -------------------------------------------------------


@dataclass(frozen=True)
class UtilityBreakdown:
    """Every term of the utility score, and the total.

    Returned instead of a bare float so an acceptance decision can be explained
    arithmetically. "Accepted with utility 0.31" says nothing; "accepted because
    +0.28 came from decision quality and the complexity charge was only -0.02"
    can be checked, and disagreed with.
    """

    predictive_gain: float = 0.0
    decision_gain: float = 0.0
    objective_gain: float = 0.0
    stability: float = 0.0
    complementarity: float = 0.0
    complexity_penalty: float = 0.0
    missingness_penalty: float = 0.0
    turnover_penalty: float = 0.0
    leakage_penalty: float = 0.0

    unmeasured: tuple[str, ...] = ()
    """Terms this caller could not supply a value for.

    A term left at its default contributes exactly zero, which in a breakdown
    is indistinguishable from a term that was measured and found to be zero.
    The two mean very different things — "decision quality did not improve" is
    a result, "decision quality was never measured" is a gap — so a caller that
    cannot supply a term names it here and :meth:`contributions` omits it
    rather than reporting a zero that reads like evidence.
    """

    @property
    def total(self) -> float:
        return (
            self.predictive_gain
            + self.decision_gain
            + self.objective_gain
            + self.stability
            + self.complementarity
            - self.complexity_penalty
            - self.missingness_penalty
            - self.turnover_penalty
            - self.leakage_penalty
        )

    def contributions(self) -> tuple[tuple[str, float], ...]:
        """Named terms, largest absolute contribution first.

        Terms named in :attr:`unmeasured` are omitted, so a breakdown never
        presents an unmeasured term as a measured zero.
        """
        terms = (
            ("predictive_gain", self.predictive_gain),
            ("decision_gain", self.decision_gain),
            ("objective_gain", self.objective_gain),
            ("stability", self.stability),
            ("complementarity", self.complementarity),
            ("complexity_penalty", -self.complexity_penalty),
            ("missingness_penalty", -self.missingness_penalty),
            ("turnover_penalty", -self.turnover_penalty),
            ("leakage_penalty", -self.leakage_penalty),
        )
        skip = {name.removesuffix("_penalty") for name in self.unmeasured} | set(self.unmeasured)
        kept = tuple(
            pair
            for pair in terms
            if pair[0] not in skip and pair[0].removesuffix("_penalty") not in skip
        )
        return tuple(sorted(kept, key=lambda pair: -abs(pair[1])))

    def explain(self) -> str:
        parts = [f"{name} {value:+.4f}" for name, value in self.contributions() if value]
        text = f"utility {self.total:+.4f} = " + " ".join(parts) if parts else "utility 0"
        if self.unmeasured:
            text += f" (not measured: {', '.join(sorted(self.unmeasured))})"
        return text


def feature_utility(
    *,
    weights: UtilityWeights,
    predictive_gain: float = 0.0,
    decision_gain: float = 0.0,
    objective_gain: float = 0.0,
    folds: Sequence[FoldMetrics] = (),
    complementarity_gain: float = 0.0,
    complexity: int = 1,
    missingness: float = 0.0,
    turnover: float = 0.0,
    leakage_risk: float = 0.0,
    unmeasured: Sequence[str] = (),
) -> UtilityBreakdown:
    """Score one candidate feature under one objective's weights.

    Args:
        predictive_gain: Relative improvement in the predictive metric. Positive
            is better, already oriented.
        decision_gain: Relative improvement in realised decision quality.
        objective_gain: Relative improvement in the objective's own metric.
        folds: Per-fold results, used for the stability term.
        complementarity_gain: Improvement over the *required* feature set alone.
            This is the term that makes "find something that adds to xG" a
            different question from "find something good".
        complexity: Node count of the program.
        missingness: Share of rows with no value, in [0, 1].
        turnover: Ranking churn, in [0, 1].
        leakage_risk: In [0, 1]. Any non-zero value should normally sink the
            candidate — the default leakage weight is an order of magnitude
            above the others precisely so it does.
        unmeasured: Names of terms this caller could not supply. They are
            excluded from the reported breakdown rather than shown as zero,
            because an unmeasured term and a measured zero are different
            claims and only one of them is evidence.

    Complexity is charged on ``log1p(nodes)`` rather than linearly. The
    difference between a two-node and a four-node program is a real jump in
    interpretability; the difference between twenty and twenty-two is not, and a
    linear charge would rank those two as far apart as the first pair.
    """
    return UtilityBreakdown(
        predictive_gain=weights.prediction * predictive_gain,
        decision_gain=weights.decision * decision_gain,
        objective_gain=weights.objective * objective_gain,
        stability=weights.stability * stability_score(folds),
        complementarity=weights.complementarity * complementarity_gain,
        complexity_penalty=weights.complexity * math.log1p(max(0, complexity - 1)),
        missingness_penalty=weights.missingness * max(0.0, min(1.0, missingness)),
        turnover_penalty=weights.turnover * max(0.0, min(1.0, turnover)),
        leakage_penalty=weights.leakage * max(0.0, min(1.0, leakage_risk)),
        unmeasured=tuple(unmeasured),
    )
