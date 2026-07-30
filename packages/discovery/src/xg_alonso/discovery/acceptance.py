"""When a candidate feature has earned its place, and when it has not.

**A feature is not accepted because it improved one fold.** That single rule is
most of what separates a discovery loop from a random-number generator with good
manners: run enough candidates against enough folds and some will win by chance,
and a policy that accepts on a mean improvement will faithfully collect them.

The gates below are all configurable and all default to something defensible.
Every rejection carries a reason, because an unexplained rejection is
indistinguishable from a crash, and a loop nobody can audit is one nobody should
trust.

**Order matters.** Leakage is checked first and is fatal — a leaking feature's
measured value is fiction, so there is nothing to weigh it against. Data
sufficiency comes next, because a verdict on three folds is not a weak verdict,
it is an absent one, and calling it "rejected" would teach the memory layer the
wrong lesson about a hypothesis that was never really tested.

This module is domain-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from xg_alonso.contracts.discovery import (
    AcceptanceStatus,
    ComplementarityClass,
    FeatureEvaluation,
    SubgroupResult,
)

__all__ = [
    "DEFAULT_POLICY",
    "STRICT_POLICY",
    "AcceptancePolicy",
    "AcceptanceVerdict",
    "classify_complementarity",
    "decide",
]


@dataclass(frozen=True)
class AcceptancePolicy:
    """Configurable thresholds a candidate must clear."""

    min_folds: int = 3
    """Below this the result is INSUFFICIENT_DATA, never REJECTED."""

    min_utility: float = 0.0
    """Mean objective utility. Zero means 'must not be actively harmful'."""

    min_fold_win_rate: float = 0.6
    """Share of folds that must improve. 0.6 of five folds is three."""

    min_incremental_value: float = 0.002
    """Relative improvement below which a gain is inside the noise.

    Deliberately not zero. A feature that improves MAE by 0.02% is not worth a
    column, the compute to build it, or the interpretability it costs.
    """

    max_recent_degradation: float = 0.01
    """How much the most recent fold may worsen before the candidate is refused.

    Checked separately from the average because a feature that helped for two
    seasons and stopped helping this one is the most dangerous kind: the model
    leans on it hardest exactly when it has stopped working.
    """

    max_missingness: float = 0.5
    max_turnover: float = 0.9
    max_complexity: int = 40

    min_subgroup_rows: int = 200
    """Below this a subgroup gain is not evidence.

    A spectacular improvement on eleven rows is the commonest way a discovery
    loop fools itself, and gating on sample size is the cheapest defence.
    """

    experimental_utility: float = 0.0
    """Utility above which a near-miss is ACCEPTED_EXPERIMENTALLY rather than rejected."""

    experimental_fold_win_rate: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_fold_win_rate <= 1.0:
            raise ValueError("min_fold_win_rate is a share and must lie in [0, 1]")
        if self.min_folds < 2:
            raise ValueError(
                "at least two folds are needed to say anything about consistency; "
                "a one-fold acceptance policy is a coin flip with a threshold"
            )


#: The shipped default. Tuned to be strict enough that a random feature is very
#: unlikely to pass, and loose enough that a real but modest signal can.
DEFAULT_POLICY: Final[AcceptancePolicy] = AcceptancePolicy()

#: For promoting a feature into the production feature set.
STRICT_POLICY: Final[AcceptancePolicy] = AcceptancePolicy(
    min_folds=5,
    min_utility=0.01,
    min_fold_win_rate=0.8,
    min_incremental_value=0.005,
    max_recent_degradation=0.0,
    max_missingness=0.3,
    min_subgroup_rows=500,
    experimental_utility=0.005,
)


@dataclass(frozen=True)
class AcceptanceVerdict:
    """The decision, and every criterion that produced it."""

    status: AcceptanceStatus
    reason: str
    failed_gates: tuple[str, ...] = ()
    passed_gates: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return self.status.is_usable


def classify_complementarity(
    evaluation: FeatureEvaluation,
    *,
    policy: AcceptancePolicy = DEFAULT_POLICY,
    global_gain: float | None = None,
    objective_gain: float | None = None,
) -> ComplementarityClass:
    """What kind of value a candidate adds, given the required features.

    The distinction is the product, not a nuance. A feature that helps everywhere
    is a model improvement. One that helps only under a particular objective is
    evidence that the objective layer is doing something real. One that helps
    only for certain clusters should be *gated* rather than added globally, and
    adding it globally would dilute it into noise for everyone else.

    Args:
        global_gain: Improvement measured without objective conditioning.
        objective_gain: Improvement measured under the objective.
    """
    if not evaluation.leakage_passed:
        return ComplementarityClass.LEAKAGE_SUSPECTED
    if len(evaluation.folds) < policy.min_folds:
        return ComplementarityClass.INSUFFICIENT_HISTORY

    meaningful = policy.min_incremental_value

    if evaluation.incremental_value >= meaningful:
        if evaluation.stability < 0.3:
            return ComplementarityClass.UNSTABLE
        return ComplementarityClass.GLOBALLY_COMPLEMENTARY

    # Not globally useful. Is it useful somewhere specific?
    strong_clusters = [
        result
        for result in evaluation.cluster_results
        if result.n >= policy.min_subgroup_rows and result.relative_improvement >= meaningful
    ]
    if strong_clusters:
        return ComplementarityClass.CLUSTER_SPECIFIC

    strong_subgroups = [
        result
        for result in evaluation.subgroup_results
        if result.n >= policy.min_subgroup_rows and result.relative_improvement >= meaningful
    ]
    if strong_subgroups:
        return ComplementarityClass.CLUSTER_SPECIFIC

    if (
        objective_gain is not None
        and global_gain is not None
        and objective_gain >= meaningful
        and global_gain < meaningful
    ):
        return ComplementarityClass.OBJECTIVE_SPECIFIC

    if evaluation.stability < 0.3 and evaluation.fold_win_rate >= 0.4:
        return ComplementarityClass.UNSTABLE

    return ComplementarityClass.REDUNDANT


def decide(
    evaluation: FeatureEvaluation,
    *,
    policy: AcceptancePolicy = DEFAULT_POLICY,
    complexity: int = 1,
) -> AcceptanceVerdict:
    """Apply the policy to a measured evaluation.

    Gate order is deliberate: fatal problems first, then sufficiency, then
    quality. A candidate that leaks is never described as "low utility", because
    its utility was never measured on anything real.
    """
    passed: list[str] = []
    failed: list[str] = []

    # 1. Fatal. A leaking feature has no measured value to weigh.
    if not evaluation.leakage_passed:
        return AcceptanceVerdict(
            status=AcceptanceStatus.REJECTED,
            reason=(
                "failed the point-in-time leakage harness, so every metric measured "
                "for it describes a model that could see the future"
            ),
            failed_gates=("leakage",),
        )
    if not evaluation.leakage_checks:
        return AcceptanceVerdict(
            status=AcceptanceStatus.REJECTED,
            reason=(
                "no leakage checks were run. An unchecked feature is not a clean one; "
                "absence of evidence was recorded as evidence of absence"
            ),
            failed_gates=("leakage_unchecked",),
        )
    passed.append("leakage")

    # 2. Sufficiency. Not enough evidence is a distinct answer from a bad result.
    if len(evaluation.folds) < policy.min_folds:
        return AcceptanceVerdict(
            status=AcceptanceStatus.INSUFFICIENT_DATA,
            reason=(
                f"measured on {len(evaluation.folds)} folds, below the minimum of "
                f"{policy.min_folds}; this is an absent verdict, not a negative one"
            ),
            failed_gates=("min_folds",),
        )
    passed.append("min_folds")

    # 3. Quality gates. Collected rather than short-circuited, so a revision has
    #    the whole list to work from.
    if evaluation.missingness > policy.max_missingness:
        failed.append("missingness")
    else:
        passed.append("missingness")

    if evaluation.turnover > policy.max_turnover:
        failed.append("turnover")
    else:
        passed.append("turnover")

    if complexity > policy.max_complexity:
        failed.append("complexity")
    else:
        passed.append("complexity")

    if evaluation.utility < policy.min_utility:
        failed.append("utility")
    else:
        passed.append("utility")

    if evaluation.fold_win_rate < policy.min_fold_win_rate:
        failed.append("fold_win_rate")
    else:
        passed.append("fold_win_rate")

    if evaluation.incremental_value < policy.min_incremental_value:
        failed.append("incremental_value")
    else:
        passed.append("incremental_value")

    recent = _recent_degradation(evaluation)
    if recent > policy.max_recent_degradation:
        failed.append("recent_degradation")
    else:
        passed.append("recent_degradation")

    if evaluation.complementarity is ComplementarityClass.REDUNDANT:
        failed.append("redundant")
    else:
        passed.append("complementarity")

    if not failed:
        return AcceptanceVerdict(
            status=AcceptanceStatus.ACCEPTED,
            reason=(
                f"improved {evaluation.folds_improved} of {len(evaluation.folds)} folds "
                f"({evaluation.fold_win_rate:.0%}) with utility {evaluation.utility:+.4f} "
                f"and stability {evaluation.stability:.2f}"
            ),
            passed_gates=tuple(passed),
        )

    # A near miss is worth retesting rather than discarding, but only when the
    # failures are about strength rather than about validity.
    structural = {"missingness", "turnover", "complexity", "recent_degradation"}
    if (
        not (set(failed) & structural)
        and evaluation.utility >= policy.experimental_utility
        and evaluation.fold_win_rate >= policy.experimental_fold_win_rate
    ):
        return AcceptanceVerdict(
            status=AcceptanceStatus.ACCEPTED_EXPERIMENTALLY,
            reason=(
                f"cleared every structural gate but fell short on {', '.join(sorted(failed))}; "
                f"utility {evaluation.utility:+.4f} over {len(evaluation.folds)} folds is "
                "worth carrying as experimental rather than discarding"
            ),
            failed_gates=tuple(failed),
            passed_gates=tuple(passed),
        )

    if evaluation.complementarity is ComplementarityClass.UNSTABLE:
        return AcceptanceVerdict(
            status=AcceptanceStatus.REVISE,
            reason=(
                f"showed real gains on {evaluation.folds_improved} folds but with "
                f"stability {evaluation.stability:.2f}; the mechanism may be right and "
                "the transformation wrong, so this is worth revising rather than rejecting"
            ),
            failed_gates=tuple(failed),
            passed_gates=tuple(passed),
        )

    return AcceptanceVerdict(
        status=AcceptanceStatus.REJECTED,
        reason=f"failed {', '.join(sorted(failed))}",
        failed_gates=tuple(failed),
        passed_gates=tuple(passed),
    )


def _recent_degradation(evaluation: FeatureEvaluation) -> float:
    """How much worse the most recent fold was. Zero when it improved."""
    if not evaluation.folds:
        return 0.0
    latest = max(evaluation.folds, key=lambda f: f.fold_index)
    return max(0.0, -latest.relative_improvement)


def subgroup_is_credible(
    result: SubgroupResult, *, policy: AcceptancePolicy = DEFAULT_POLICY
) -> bool:
    """Whether a subgroup finding rests on enough rows to be worth reporting."""
    return result.n >= policy.min_subgroup_rows
