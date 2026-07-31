"""Distributions, paired comparisons, and what the unit of analysis has to be.

**The unit is the starting condition** — one ``(season, start_gameweek,
squad_id)`` — and not the run, and not the gameweek. Both alternatives are
wrong in ways that inflate confidence rather than reduce it:

- *Not the run.* A stochastic policy has replicates and a deterministic one has
  a single value. Pairing at run level broadcasts that single value against
  every replicate and silently multiplies ``n``. Replicates are reduced to
  their mean first, so every policy contributes exactly one value per
  condition and the pairing is exact; the spread across replicates is reported
  separately as ``seed_sensitivity_sd`` rather than being folded in.
- *Not the gameweek.* Gameweeks inside a run are serially dependent through
  squad divergence — the trap ``decision_delta`` exists to avoid. Gameweek
  series feed descriptive metrics like drawdown and never a confidence
  interval.

**"Never a single aggregate number without its distribution" is enforced
structurally, not by discipline.** The report writer renders
:class:`Aggregate`, which cannot be constructed without its percentiles, so a
naked mean has no path to a report even by accident.

**No p-values and no significance language.** `docs/backtesting_and_leakage.md`
states the position; intervals, proportions and effect sizes say what was
measured without implying a decision procedure nobody agreed to.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from xg_alonso.contracts.evaluation import BootstrapUnit
from xg_alonso.contracts.seeds import ROOT_SEED, derive_seed

__all__ = [
    "Aggregate",
    "BootstrapInterval",
    "PairedComparison",
    "aggregate",
    "bootstrap_mean",
    "paired_comparison",
]


@dataclass(frozen=True)
class Aggregate:
    """A distribution. There is deliberately no way to build a bare mean."""

    n: int
    mean: float
    median: float
    sd: float
    p5: float
    p25: float
    p75: float
    p95: float
    worst: float
    best: float
    worst_label: str = ""
    best_label: str = ""

    def row(self) -> str:
        return (
            f"n={self.n:<4} mean={self.mean:>+8.2f}  median={self.median:>+8.2f}  "
            f"sd={self.sd:>7.2f}  p5={self.p5:>+8.2f}  p95={self.p95:>+8.2f}  "
            f"worst={self.worst:>+8.2f}  best={self.best:>+8.2f}"
        )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile over an already-sorted sequence."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(values: Sequence[float], *, labels: Sequence[str] = ()) -> Aggregate:
    """Summarise a sample as a distribution.

    Args:
        labels: Names parallel to ``values``, so the worst and best runs can be
            pointed at rather than merely quantified. A number nobody can
            trace to a run is a number nobody can investigate.
    """
    if not values:
        raise ValueError("cannot summarise an empty sample; report n=0 rather than a zero")

    ordered = sorted(values)
    paired = sorted(zip(values, labels or [""] * len(values), strict=True))
    return Aggregate(
        n=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        sd=statistics.pstdev(values) if len(values) > 1 else 0.0,
        p5=_percentile(ordered, 0.05),
        p25=_percentile(ordered, 0.25),
        p75=_percentile(ordered, 0.75),
        p95=_percentile(ordered, 0.95),
        worst=ordered[0],
        best=ordered[-1],
        worst_label=paired[0][1],
        best_label=paired[-1][1],
    )


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile bootstrap interval, with what it was resampled over."""

    point: float
    low: float
    high: float
    confidence: float
    resamples: int
    unit: BootstrapUnit
    seed: int
    n_units: int
    caveat: str = ""

    def row(self) -> str:
        text = f"{self.point:+.2f} [{self.low:+.2f}, {self.high:+.2f}] ({self.confidence:.0%})"
        return text + (f"  — {self.caveat}" if self.caveat else "")


def bootstrap_mean(
    values: Sequence[float],
    *,
    blocks: Sequence[str] | None = None,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = ROOT_SEED,
    unit: BootstrapUnit = BootstrapUnit.SEASON_BLOCK,
) -> BootstrapInterval:
    """Percentile bootstrap over the mean.

    Args:
        blocks: A grouping label per value, normally the season. Conditions
            inside one season share the same actual outcomes — a good gameweek
            lifts every squad at once — so resampling them independently would
            treat correlated observations as independent and produce an
            interval far narrower than the evidence supports.
        unit: ``SEASON_BLOCK`` resamples blocks then draws within them.
            ``CONDITION`` resamples values directly and is used only when there
            are too few blocks, with the degradation recorded as a caveat.

    With three seasons the block interval will be visibly wide. That is the
    correct answer, and it is precisely what a single-season headline conceals.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")

    point = statistics.fmean(values)
    caveat = ""
    grouped: dict[str, list[float]] = {}
    if blocks is not None:
        for value, block in zip(values, blocks, strict=True):
            grouped.setdefault(block, []).append(value)

    effective = unit
    if unit is BootstrapUnit.SEASON_BLOCK and len(grouped) < 3:
        effective = BootstrapUnit.CONDITION
        caveat = (
            f"only {len(grouped)} block(s); season-block resampling needs at least three, "
            "so conditions were resampled directly and the interval is narrower than "
            "the dependence between them warrants"
        )

    rng = random.Random(seed)
    means: list[float] = []
    if effective is BootstrapUnit.SEASON_BLOCK:
        keys = list(grouped)
        for _ in range(resamples):
            drawn: list[float] = []
            for _ in range(len(keys)):
                block = grouped[rng.choice(keys)]
                drawn.extend(rng.choice(block) for _ in range(len(block)))
            means.append(statistics.fmean(drawn))
    else:
        size = len(values)
        for _ in range(resamples):
            means.append(statistics.fmean([rng.choice(values) for _ in range(size)]))

    means.sort()
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        point=point,
        low=_percentile(means, tail),
        high=_percentile(means, 1.0 - tail),
        confidence=confidence,
        resamples=resamples,
        unit=effective,
        seed=seed,
        n_units=len(grouped) if grouped else len(values),
        caveat=caveat,
    )


@dataclass(frozen=True)
class PairedComparison:
    """One policy against one baseline, over identical starting conditions."""

    policy: str
    baseline: str
    metric: str
    n_conditions: int
    differences: Aggregate
    mean_difference: BootstrapInterval
    probability_of_outperforming: float
    effect_size_dz: float | None
    conditions_won: int
    conditions_lost: int
    conditions_tied: int
    seed_sensitivity_sd: float | None = None
    diagnostic: bool = False

    def row(self) -> str:
        label = f"{self.policy} vs {self.baseline}"
        flag = "  [diagnostic — an upper bound, not a baseline]" if self.diagnostic else ""
        return (
            f"{label:<32} {self.mean_difference.row()}  "
            f"won {self.conditions_won}/{self.n_conditions} "
            f"({self.probability_of_outperforming:.0%})"
            + (f"  d_z={self.effect_size_dz:+.2f}" if self.effect_size_dz is not None else "")
            + flag
        )


def paired_comparison(
    policy_values: Mapping[str, Sequence[float]],
    baseline_values: Mapping[str, Sequence[float]],
    *,
    policy: str,
    baseline: str,
    metric: str,
    blocks: Mapping[str, str] | None = None,
    resamples: int = 10_000,
    confidence: float = 0.95,
    unit: BootstrapUnit = BootstrapUnit.SEASON_BLOCK,
    diagnostic: bool = False,
) -> PairedComparison:
    """Compare two policies over the conditions they both ran.

    Args:
        policy_values: Condition id to that policy's replicate values.
        baseline_values: The same, for the baseline.
        blocks: Condition id to its block label, normally the season.

    Replicates are averaged **before** pairing. A deterministic policy has one
    value and a stochastic one has several; pairing them directly would
    broadcast the single value and inflate ``n`` by the replicate count.
    """
    shared = sorted(set(policy_values) & set(baseline_values))
    if not shared:
        raise ValueError(
            f"{policy} and {baseline} share no starting conditions, so there is "
            "nothing to pair; an unpaired comparison across different squads "
            "would measure the squads"
        )

    differences: list[float] = []
    spreads: list[float] = []
    for condition in shared:
        policy_runs = list(policy_values[condition])
        baseline_runs = list(baseline_values[condition])
        differences.append(statistics.fmean(policy_runs) - statistics.fmean(baseline_runs))
        if len(policy_runs) > 1:
            spreads.append(statistics.pstdev(policy_runs))

    won = sum(1 for d in differences if d > 0)
    lost = sum(1 for d in differences if d < 0)
    tied = len(differences) - won - lost

    spread = statistics.pstdev(differences) if len(differences) > 1 else 0.0
    return PairedComparison(
        policy=policy,
        baseline=baseline,
        metric=metric,
        n_conditions=len(shared),
        differences=aggregate(differences, labels=shared),
        mean_difference=bootstrap_mean(
            differences,
            blocks=[blocks[c] for c in shared] if blocks else None,
            resamples=resamples,
            confidence=confidence,
            seed=derive_seed(ROOT_SEED, "bootstrap", metric, policy, baseline),
            unit=unit,
        ),
        probability_of_outperforming=won / len(differences),
        effect_size_dz=(statistics.fmean(differences) / spread) if spread > 0 else None,
        conditions_won=won,
        conditions_lost=lost,
        conditions_tied=tied,
        seed_sensitivity_sd=statistics.fmean(spreads) if spreads else None,
        diagnostic=diagnostic,
    )
