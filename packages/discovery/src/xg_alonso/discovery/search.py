"""Bounded search for features that add something to what you already have.

The question this answers is not "which features are good" but **"given that xG
stays in the model, what else carries information xG does not?"** Those are
different questions with different answers, and the second is the one a manager
who trusts a metric actually asks.

**Pairwise correlation does not answer it.** Two features can correlate at 0.9
and still carry different information in the tail, and two features can correlate
at 0.0 and be jointly useless. What matters is whether ``Model(R + {f})`` beats
``Model(R)`` out of sample — a question only a fitted model on held-out folds can
answer, which is why every scorer here is a walk-forward evaluation rather than a
statistic over a correlation matrix.

**The search is bounded, never combinatorial.** With even fifty candidates an
unconstrained subset search is 2^50 evaluations, each of which fits nine models
over several folds. Two strategies, both with hard caps:

- :func:`greedy_forward` — add the single best candidate, repeat. Cheap, and
  correct when candidates are roughly independent.
- :func:`beam_search` — carry the best ``k`` partial sets forward. Finds
  **bundles**: pairs that are individually unimpressive and jointly strong, which
  greedy selection structurally cannot see because it never keeps a candidate
  that did not win on its own.

Both stop on a minimum-gain threshold as well as a size cap, so a search that has
stopped finding anything stops running.

Domain-free by contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from xg_alonso.contracts.discovery import FoldMetrics

__all__ = [
    "ScoreResult",
    "SearchResult",
    "SearchStep",
    "SubsetScorer",
    "beam_search",
    "greedy_forward",
]


@dataclass(frozen=True)
class ScoreResult:
    """What one feature set scored, out of sample."""

    metric: float
    """The headline number. Orientation is carried by ``lower_is_better``."""

    folds: tuple[FoldMetrics, ...] = ()
    lower_is_better: bool = True
    detail: dict[str, float] = field(default_factory=dict)

    def improvement_over(self, baseline: ScoreResult) -> float:
        """Relative improvement over a baseline, oriented so positive is better."""
        delta = baseline.metric - self.metric
        if not self.lower_is_better:
            delta = -delta
        if abs(baseline.metric) < 1e-12:
            return 0.0
        return delta / abs(baseline.metric)


class SubsetScorer(Protocol):
    """Fits and evaluates one feature set out of sample.

    The engine never learns what the features mean, how the model is fitted, or
    what the metric measures. That is what makes the search reusable.
    """

    def __call__(self, features: tuple[str, ...]) -> ScoreResult: ...


@dataclass(frozen=True)
class SearchStep:
    """One accepted addition, and what it bought."""

    added: tuple[str, ...]
    features: tuple[str, ...]
    score: ScoreResult
    gain: float

    def describe(self) -> str:
        names = " + ".join(self.added)
        return f"{names}: {self.gain:+.4%} over the previous set"


@dataclass(frozen=True)
class SearchResult:
    """The chosen set, the path that produced it, and what was passed over."""

    required: tuple[str, ...]
    selected: tuple[str, ...]
    baseline: ScoreResult
    final: ScoreResult
    steps: tuple[SearchStep, ...] = ()
    rejected: tuple[tuple[str, float], ...] = ()
    """Candidates that were evaluated and not chosen, with their best measured gain.

    Retained deliberately. A search that reports only its winners cannot be
    distinguished from one that never looked, and "we tried transfer momentum and
    it added nothing" is a finding worth keeping.
    """

    evaluations: int = 0
    truncated: bool = False
    """True when a cap stopped the search rather than the gain threshold.

    Surfaced because a truncated search has not shown that nothing more exists —
    it has shown that it stopped looking, and reporting the two identically would
    read as completeness.
    """

    @property
    def total_gain(self) -> float:
        return self.final.improvement_over(self.baseline)

    @property
    def discovered(self) -> tuple[str, ...]:
        """Selected features that were not already required."""
        return tuple(f for f in self.selected if f not in set(self.required))


def greedy_forward(
    *,
    required: Sequence[str],
    candidates: Sequence[str],
    scorer: Callable[[tuple[str, ...]], ScoreResult],
    max_features: int = 5,
    min_gain: float = 0.002,
    max_evaluations: int = 400,
    on_step: Callable[[SearchStep], None] | None = None,
) -> SearchResult:
    """Add the best remaining candidate until nothing clears ``min_gain``.

    Args:
        required: Features that must stay in every set. Never dropped, never
            evaluated for removal — a required feature is the user restating the
            problem, not offering an exchange rate.
        candidates: Features to consider adding.
        scorer: Fits and scores one set out of sample.
        max_features: Cap on discovered additions.
        min_gain: Relative improvement a candidate must deliver to be added.
        max_evaluations: Hard budget on scorer calls.

    Returns:
        The selection, every step, and every rejected candidate's best gain.
    """
    required_set = tuple(dict.fromkeys(required))
    pool = [c for c in dict.fromkeys(candidates) if c not in set(required_set)]

    baseline = scorer(required_set)
    evaluations = 1
    current = required_set
    current_score = baseline
    steps: list[SearchStep] = []
    best_seen: dict[str, float] = {}
    truncated = False

    while pool and len(steps) < max_features:
        best_name: str | None = None
        best_score: ScoreResult | None = None
        best_gain = -float("inf")

        for candidate in list(pool):
            if evaluations >= max_evaluations:
                truncated = True
                break
            trial = scorer((*current, candidate))
            evaluations += 1
            gain = trial.improvement_over(current_score)
            best_seen[candidate] = max(best_seen.get(candidate, -float("inf")), gain)
            if gain > best_gain:
                best_name, best_score, best_gain = candidate, trial, gain

        if truncated or best_name is None or best_score is None or best_gain < min_gain:
            break

        current = (*current, best_name)
        current_score = best_score
        pool.remove(best_name)
        step = SearchStep(added=(best_name,), features=current, score=best_score, gain=best_gain)
        steps.append(step)
        if on_step is not None:
            on_step(step)

    if len(steps) >= max_features and pool:
        truncated = True

    rejected = tuple(
        sorted(
            ((name, gain) for name, gain in best_seen.items() if name not in set(current)),
            key=lambda pair: -pair[1],
        )
    )
    return SearchResult(
        required=required_set,
        selected=current,
        baseline=baseline,
        final=current_score,
        steps=tuple(steps),
        rejected=rejected,
        evaluations=evaluations,
        truncated=truncated,
    )


def beam_search(
    *,
    required: Sequence[str],
    candidates: Sequence[str],
    scorer: Callable[[tuple[str, ...]], ScoreResult],
    beam_width: int = 3,
    max_features: int = 4,
    min_gain: float = 0.002,
    bundle_size: int = 1,
    max_evaluations: int = 600,
) -> SearchResult:
    """Carry the best ``beam_width`` partial sets forward at each depth.

    ``bundle_size`` above one adds candidates in groups, which is the only way to
    find a **bundle** — two features that are individually flat and jointly
    informative. Greedy selection cannot find one by construction: it discards
    each half before ever seeing them together.

    Bundles are expensive (``C(n, k)`` trials per level), so ``bundle_size`` is
    capped at 2 and the evaluation budget is enforced. When the budget stops the
    search, ``truncated`` says so rather than the result implying exhaustiveness.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if not 1 <= bundle_size <= 2:
        raise ValueError(
            "bundle_size above 2 makes each level combinatorial in the candidate "
            "count; if you need triples, run a second pass over the pairs this finds"
        )

    required_set = tuple(dict.fromkeys(required))
    pool = [c for c in dict.fromkeys(candidates) if c not in set(required_set)]

    baseline = scorer(required_set)
    evaluations = 1
    truncated = False
    best_seen: dict[str, float] = {}

    # Each beam entry: (features, score, steps)
    beam: list[tuple[tuple[str, ...], ScoreResult, tuple[SearchStep, ...]]] = [
        (required_set, baseline, ())
    ]
    best_entry = beam[0]

    for _ in range(max_features):
        expansions: list[tuple[tuple[str, ...], ScoreResult, tuple[SearchStep, ...], float]] = []

        for features, score, steps in beam:
            remaining = [c for c in pool if c not in set(features)]
            groups = _groups(remaining, bundle_size)
            for group in groups:
                if evaluations >= max_evaluations:
                    truncated = True
                    break
                trial = scorer((*features, *group))
                evaluations += 1
                gain = trial.improvement_over(score)
                for name in group:
                    best_seen[name] = max(best_seen.get(name, -float("inf")), gain)
                if gain < min_gain:
                    continue
                step = SearchStep(added=group, features=(*features, *group), score=trial, gain=gain)
                expansions.append(((*features, *group), trial, (*steps, step), gain))
            if truncated:
                break

        if not expansions:
            break

        # Rank by absolute quality against the baseline, not by the last step's
        # gain: a path that gained a lot from a weak position is not better than
        # one that gained less from a strong one.
        expansions.sort(key=lambda entry: -entry[1].improvement_over(baseline))
        beam = [(features, score, steps) for features, score, steps, _ in expansions[:beam_width]]
        candidate_best = max(beam, key=lambda entry: entry[1].improvement_over(baseline))
        if candidate_best[1].improvement_over(baseline) > best_entry[1].improvement_over(baseline):
            best_entry = candidate_best
        if truncated:
            break

    features, score, steps = best_entry
    rejected = tuple(
        sorted(
            ((name, gain) for name, gain in best_seen.items() if name not in set(features)),
            key=lambda pair: -pair[1],
        )
    )
    return SearchResult(
        required=required_set,
        selected=features,
        baseline=baseline,
        final=score,
        steps=steps,
        rejected=rejected,
        evaluations=evaluations,
        truncated=truncated,
    )


def _groups(names: Sequence[str], size: int) -> list[tuple[str, ...]]:
    """Candidate additions of the given size, in a deterministic order."""
    if size == 1:
        return [(name,) for name in names]
    out: list[tuple[str, ...]] = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            out.append((first, second))
    return out
