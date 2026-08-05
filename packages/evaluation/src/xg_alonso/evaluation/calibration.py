"""Does a predictive distribution tell the truth about its own uncertainty?

``accuracy.py`` scores point estimates and says so: every dataclass in it is a
scalar prediction against a scalar outcome. This is the other question — given a
whole distribution over a player's points, is that distribution *right*? It has
different inputs (a PMF per row), different metrics, and a different degeneracy
check, so it is a separate module rather than fields bolted onto the other that
would be ``None`` half the time. It reuses that module's :data:`PRICE_BANDS`,
:data:`DEGENERATE_SD` and narrowing helpers, and the two reports are printed
together by one command.

**What this exists to catch.** The shipped ``expected_points_sd`` is
``max(0.5, |total| * minutes_sd/90 + 0.8)`` — invented, never coverage-checked,
and measurably anti-correlated with real dispersion: it peaks on 30-60 minute
cameos and is *lowest* on nailed starters, where week-to-week variance is
highest. Coverage on starters is roughly 38% at a nominal 80%. Nothing in the
codebase could report that, which is why this module ships before anything
changes ``sd``: **the strongest negative control for a calibration check is the
code that ships today, and if the check does not flag it the check is broken.**

Three things about the metrics are consequences of FPL points being a
*discrete, zero-inflated* quantity, and getting any of them wrong makes a
working model look broken:

1. **The PIT must be randomised.** Roughly 60% of player-gameweeks are exactly
   zero. The naive PIT ``F(y)`` is degenerate on a lattice — it can never be
   uniform, and its histogram shows a spike no amount of recalibration removes.
   :func:`randomised_pit` uses ``u = F(y-1) + v*p(y)`` with ``v ~ U(0,1)``, and
   ``v`` is drawn from a seed derived by name so the report is reproducible.
2. **Coverage is reported but is not the primary metric.** For a player with
   ``p_zero = 0.9`` the 80% central interval is ``[0, 0]`` and covers 90%.
   That over-coverage is correct and unavoidable, so ``mean_width`` travels
   beside every coverage number and the model is selected on CRPS.
3. **A pooled number is never enough.** The zero atom drags every marginal
   statistic toward looking fine: the incumbent scores ~78% marginal coverage
   at a nominal 80% while scoring ~39% on players who actually started. The
   slices here are therefore structural, not optional — there is no constructor
   that yields only ``overall``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.seeds import ROOT_SEED, derive_seed
from xg_alonso.evaluation.accuracy import DEGENERATE_SD, PRICE_BANDS, _f, _i

__all__ = [
    "DEFAULT_NOMINAL_LEVELS",
    "MINUTES_BANDS",
    "CalibrationReport",
    "CalibrationSlice",
    "CoverageBand",
    "Distribution",
    "frame_from_distributions",
    "gaussian_pmf",
    "inflate_pmf",
    "point_mass_pmf",
    "randomised_pit",
    "score_distributions",
]

#: Central-interval levels a human reads. 0.5 is included because a median
#: interval failing is a different diagnosis from a tail failing.
DEFAULT_NOMINAL_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)

#: Minutes bands, upper bound exclusive. The split that matters is 0 / cameo /
#: starter: dispersion differs by a factor of six across them, and the incumbent
#: `sd` is *inverted* over exactly this axis.
MINUTES_BANDS: tuple[tuple[str, int, int], ...] = (
    ("did not play", 0, 1),
    ("cameo (1-59)", 1, 60),
    ("started (60+)", 60, 1000),
)

#: Quantile levels for the pinball loss. Deliberately asymmetric toward the
#: right tail, which is where a captain decision lives.
_PINBALL_LEVELS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass(frozen=True)
class Distribution:
    """One player's predictive PMF over consecutive integers.

    A lattice, not a quantile grid: realised FPL points *is* an integer on a
    small bounded support, so this representation is exact, compact, and closed
    under the mixture and convolution operations the prediction layer performs.
    """

    support_min: int
    pmf: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.pmf:
            raise ValueError("a distribution needs at least one mass")
        total = math.fsum(self.pmf)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"pmf sums to {total!r}, not 1")
        if any(p < -1e-12 for p in self.pmf):
            raise ValueError("a negative mass is not a probability")

    @property
    def support(self) -> range:
        return range(self.support_min, self.support_min + len(self.pmf))

    @property
    def mean(self) -> float:
        return math.fsum(k * p for k, p in zip(self.support, self.pmf, strict=True))

    @property
    def sd(self) -> float:
        mean = self.mean
        var = math.fsum(((k - mean) ** 2) * p for k, p in zip(self.support, self.pmf, strict=True))
        return math.sqrt(max(0.0, var))

    def cdf_at(self, value: int) -> float:
        """``P(X <= value)``, saturating outside the support."""
        if value < self.support_min:
            return 0.0
        index = min(value - self.support_min, len(self.pmf) - 1)
        return min(1.0, math.fsum(self.pmf[: index + 1]))

    def mass_at(self, value: int) -> float:
        if value not in self.support:
            return 0.0
        return self.pmf[value - self.support_min]

    def interval(self, level: float) -> tuple[int, int]:
        """The central interval at ``level``, as inclusive integer bounds.

        Central rather than shortest: it is what "80% interval" conventionally
        means, and a shortest interval on a lattice with a 60% atom collapses to
        ``[0, 0]`` for most players, which measures the atom rather than the
        forecast.
        """
        tail = (1.0 - level) / 2.0
        lower = self.support_min
        for k in self.support:
            if self.cdf_at(k) >= tail - 1e-12:
                lower = k
                break
        upper = self.support[-1]
        for k in self.support:
            if self.cdf_at(k) >= 1.0 - tail - 1e-12:
                upper = k
                break
        return lower, max(lower, upper)


def randomised_pit(distribution: Distribution, actual: int, draw: float) -> float:
    """``u = F(y-1) + v * p(y)`` — the PIT that can be uniform on a lattice.

    The naive ``F(y)`` cannot: on a discrete support it takes finitely many
    values, and with ~60% of outcomes at exactly zero its histogram is a spike
    that survives any recalibration. Randomising within the mass at the realised
    outcome restores uniformity *if and only if* the forecast is calibrated,
    which is the property the whole check rests on.

    Args:
        draw: A ``U(0,1)`` variate. Supplied rather than drawn here so the
            caller owns reproducibility — see :func:`score_distributions`.
    """
    below = distribution.cdf_at(actual - 1)
    return below + draw * distribution.mass_at(actual)


def _crps(distribution: Distribution, actual: int) -> float:
    """Exact CRPS on a lattice: ``sum_k (F(k) - 1{y <= k})^2``.

    Proper, computed in closed form from the PMF rather than sampled, and the
    metric a distributional forecast is selected on. Coverage is what a human
    reads; this is what decides.
    """
    total = 0.0
    running = 0.0
    for k, p in zip(distribution.support, distribution.pmf, strict=True):
        running += p
        total += (running - (1.0 if actual <= k else 0.0)) ** 2
    # Outcomes beyond the support contribute (1 - 1)^2 = 0 above and
    # (0 - 0)^2 = 0 below, so the sum is complete.
    return total


def _pinball(distribution: Distribution, actual: int, level: float) -> float:
    """Quantile loss at ``level``, using the distribution's own quantile."""
    q = distribution.support_min
    for k in distribution.support:
        if distribution.cdf_at(k) >= level - 1e-12:
            q = k
            break
    delta = actual - q
    return level * delta if delta >= 0 else (level - 1.0) * delta


def inflate_pmf(distribution: Distribution, factor: float) -> Distribution:
    """Widen or narrow a distribution about its mean by a known factor.

    The first negative control. A check that reports the same dispersion ratio
    at ``factor=0.5`` and ``factor=2.0`` is not a check. Implemented by
    resampling the CDF on a rescaled axis, which preserves the mean to within
    lattice rounding and moves the spread by construction.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    mean = distribution.mean
    support = list(distribution.support)
    low = math.floor(mean + (support[0] - mean) * factor)
    high = math.ceil(mean + (support[-1] - mean) * factor)
    masses: list[float] = []
    for k in range(low, high + 1):
        # Map this lattice point back onto the original axis and read its mass.
        source = mean + (k - mean) / factor
        masses.append(max(0.0, _interpolated_mass(distribution, source) / factor))
    total = math.fsum(masses)
    if total <= 0:
        return distribution
    return Distribution(support_min=low, pmf=tuple(m / total for m in masses))


def _interpolated_mass(distribution: Distribution, position: float) -> float:
    """Linear interpolation of the PMF between lattice points."""
    lower = math.floor(position)
    weight = position - lower
    return (
        distribution.mass_at(int(lower)) * (1.0 - weight)
        + distribution.mass_at(int(lower) + 1) * weight
    )


def gaussian_pmf(
    mean: float, sd: float, *, support_min: int = -5, support_max: int = 30
) -> Distribution:
    """A normal discretised onto the points lattice.

    The second and strongest negative control: built from the *incumbent*
    ``expected_points_sd``, this is what the shipped code is really claiming.
    Feeding it through :func:`score_distributions` must reproduce the measured
    under-coverage, and if it does not, the check is broken.
    """
    width = max(sd, 1e-6)
    masses = []
    for k in range(support_min, support_max + 1):
        # Mass of the unit cell centred on k.
        masses.append(_normal_cdf((k + 0.5 - mean) / width) - _normal_cdf((k - 0.5 - mean) / width))
    total = math.fsum(masses)
    if total <= 0:
        return point_mass_pmf(round(mean))
    return Distribution(support_min=support_min, pmf=tuple(m / total for m in masses))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def point_mass_pmf(value: int) -> Distribution:
    """All mass on one outcome — the third negative control.

    Zero width, so coverage is near zero except by luck. Mirrors
    ``AccuracySlice.degenerate``: every metric computed from it is flagged
    uninterpretable rather than reported as excellent.
    """
    return Distribution(support_min=value, pmf=(1.0,))


@dataclass(frozen=True)
class CoverageBand:
    """Empirical coverage at one nominal level, with the width that bought it."""

    nominal: float
    empirical: float
    low: float
    high: float
    mean_width: float
    """Average interval width. A check that covers by being wide is not a check."""

    @property
    def gap(self) -> float:
        """Signed shortfall. Negative means the intervals are too narrow."""
        return self.empirical - self.nominal

    def row(self) -> str:
        return (
            f"    {self.nominal:>4.0%} -> {self.empirical:>6.1%} "
            f"[{self.low:>5.1%}, {self.high:>5.1%}]  width={self.mean_width:>5.2f}  "
            f"gap={self.gap:>+6.1%}"
        )


@dataclass(frozen=True)
class CalibrationSlice:
    """Distributional quality over one subset of forecasts."""

    name: str
    n: int
    coverage: tuple[CoverageBand, ...] = ()
    pit_histogram: tuple[float, ...] = ()
    pit_uniformity: float = 0.0
    """Cramér-von Mises statistic against U(0,1). Larger is worse."""
    crps: float = 0.0
    pinball: tuple[tuple[float, float], ...] = ()
    mean_sd: float = 0.0
    """Mean of the predicted standard deviations."""
    residual_sd: float = 0.0
    """Realised spread of (actual - predicted mean)."""

    @property
    def dispersion_ratio(self) -> float:
        """``residual_sd / mean_sd``. One is calibrated; the headline number.

        Today, for players who started, this is roughly 3.2 — the forecast
        claims about a third of the spread the outcomes actually have.
        """
        if self.mean_sd < DEGENERATE_SD:
            return 0.0
        return self.residual_sd / self.mean_sd

    @property
    def degenerate(self) -> bool:
        """Whether the forecasts carry no width to assess."""
        return self.mean_sd < DEGENERATE_SD

    def row(self) -> str:
        flag = "  DEGENERATE" if self.degenerate else ""
        return (
            f"{self.name:<26} n={self.n:>6}  crps={self.crps:>6.3f}  "
            f"disp={self.dispersion_ratio:>5.2f}  pit_cvm={self.pit_uniformity:>6.3f}{flag}"
        )

    def detail(self) -> str:
        lines = [self.row()]
        lines += [b.row() for b in self.coverage]
        return "\n".join(lines)


@dataclass(frozen=True)
class CalibrationReport:
    """Pooled and sliced, never pooled alone.

    There is deliberately no constructor that yields ``overall`` by itself. The
    zero atom makes every marginal statistic look acceptable while the forecast
    fails badly on the players a manager actually chooses between, so the slices
    are part of the type rather than an option a caller can omit.
    """

    overall: CalibrationSlice
    appeared_only: CalibrationSlice
    """Mandatory, not optional.

    ``accuracy.py`` already argues this for point accuracy — scoring every
    player pools in hundreds who never played, whose zero is trivially
    predictable — and it applies with more force to coverage, where a
    ``[0, 0]`` interval around a near-certain zero covers ~100%.
    """
    by_position: tuple[CalibrationSlice, ...] = ()
    by_price_band: tuple[CalibrationSlice, ...] = ()
    by_minutes_band: tuple[CalibrationSlice, ...] = ()
    by_predicted_decile: tuple[CalibrationSlice, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def conditionally_inconsistent(self) -> bool:
        """Whether the dispersion ratio varies materially across slices.

        A forecast can be calibrated on average while being overconfident on
        premiums and underconfident on cheap players. Pooled metrics cannot show
        that; this is a hard finding, not a warning.
        """
        ratios = [
            s.dispersion_ratio
            for group in (self.by_position, self.by_price_band, self.by_minutes_band)
            for s in group
            if s.n > 0 and not s.degenerate
        ]
        if len(ratios) < 2:
            return False
        return max(ratios) - min(ratios) > 0.5

    def summary(self) -> str:
        lines = ["Predictive distributions vs actual", ""]
        lines.append(self.overall.detail())
        lines += ["", self.appeared_only.detail()]
        for title, group in (
            ("By position", self.by_position),
            ("By price band", self.by_price_band),
            ("By minutes", self.by_minutes_band),
            ("By predicted decile", self.by_predicted_decile),
        ):
            if group:
                lines += ["", title] + [s.row() for s in group]
        if self.conditionally_inconsistent:
            lines += [
                "",
                "  ! dispersion ratio varies by more than 0.5 across slices: this "
                "forecast is conditionally miscalibrated even where the pooled "
                "number looks acceptable",
            ]
        if self.warnings:
            lines += ["", "Warnings"] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def _cramer_von_mises(values: Sequence[float]) -> float:
    """Distance from U(0,1). Zero is perfect uniformity."""
    n = len(values)
    if n == 0:
        return 0.0
    ordered = sorted(values)
    total = math.fsum((u - (2 * i + 1) / (2 * n)) ** 2 for i, u in enumerate(ordered))
    return total + 1.0 / (12 * n)


def _histogram(values: Sequence[float], bins: int = 20) -> tuple[float, ...]:
    if not values:
        return tuple(0.0 for _ in range(bins))
    counts = [0] * bins
    for u in values:
        index = min(bins - 1, max(0, int(u * bins)))
        counts[index] += 1
    return tuple(c / len(values) for c in counts)


def _score_slice(
    name: str,
    rows: Sequence[tuple[Distribution, int, float]],
    *,
    nominal_levels: Sequence[float],
) -> CalibrationSlice:
    """Score one subset. Rows are ``(distribution, actual, pit_draw)``."""
    if not rows:
        return CalibrationSlice(name=name, n=0)

    pits: list[float] = []
    crps_values: list[float] = []
    sds: list[float] = []
    residuals: list[float] = []
    covered: dict[float, list[int]] = {lvl: [] for lvl in nominal_levels}
    widths: dict[float, list[int]] = {lvl: [] for lvl in nominal_levels}
    pinball_totals: dict[float, list[float]] = {lvl: [] for lvl in _PINBALL_LEVELS}

    for distribution, actual, draw in rows:
        pits.append(randomised_pit(distribution, actual, draw))
        crps_values.append(_crps(distribution, actual))
        sds.append(distribution.sd)
        residuals.append(actual - distribution.mean)
        for level in nominal_levels:
            low, high = distribution.interval(level)
            covered[level].append(1 if low <= actual <= high else 0)
            widths[level].append(high - low + 1)
        for level in _PINBALL_LEVELS:
            pinball_totals[level].append(_pinball(distribution, actual, level))

    n = len(rows)
    bands = []
    for level in nominal_levels:
        hits = covered[level]
        empirical = sum(hits) / n
        # Normal-approximation band on a proportion. A full bootstrap here would
        # resample rows that are not independent across a gameweek; the interval
        # is indicative and the report says so rather than implying precision.
        half = 1.96 * math.sqrt(max(0.0, empirical * (1 - empirical) / n))
        bands.append(
            CoverageBand(
                nominal=level,
                empirical=empirical,
                low=max(0.0, empirical - half),
                high=min(1.0, empirical + half),
                mean_width=sum(widths[level]) / n,
            )
        )

    mean_residual = math.fsum(residuals) / n
    residual_sd = math.sqrt(max(0.0, math.fsum((r - mean_residual) ** 2 for r in residuals) / n))

    return CalibrationSlice(
        name=name,
        n=n,
        coverage=tuple(bands),
        pit_histogram=_histogram(pits),
        pit_uniformity=_cramer_von_mises(pits),
        crps=math.fsum(crps_values) / n,
        pinball=tuple((lvl, math.fsum(v) / n) for lvl, v in pinball_totals.items()),
        mean_sd=math.fsum(sds) / n,
        residual_sd=residual_sd,
    )


def _distributions_from(frame: pl.DataFrame) -> list[Distribution]:
    return [
        Distribution(support_min=int(lo), pmf=tuple(float(p) for p in pmf))
        for lo, pmf in zip(frame["pmf_support_min"], frame["pmf"], strict=True)
    ]


def score_distributions(
    frame: pl.DataFrame,
    *,
    nominal_levels: Sequence[float] = DEFAULT_NOMINAL_LEVELS,
    seed: int = ROOT_SEED,
    label: str = "distributions",
) -> CalibrationReport:
    """Score predictive distributions against what happened.

    Args:
        frame: One row per forecast. Required columns ``actual`` (int),
            ``pmf_support_min`` (int) and ``pmf`` (list of float). Optional
            ``position``, ``price``, ``minutes`` and ``predicted`` each unlock
            a slice; a missing one is omitted rather than faked.
        seed: Root for the PIT randomisation. Derived by name, so the report is
            reproducible across runs and machines — a seeded draw inside a
            metric would otherwise make the number move between runs.
        label: Names the pooled slice in the report.

    Raises:
        ValueError: if a required column is absent. A calibration report built
            from a frame missing its outcomes is worse than no report.
    """
    required = {"actual", "pmf_support_min", "pmf"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    if frame.is_empty():
        empty = CalibrationSlice(name=label, n=0)
        return CalibrationReport(
            overall=empty,
            appeared_only=CalibrationSlice(name="appeared only", n=0),
            warnings=("no rows to score",),
        )

    distributions = _distributions_from(frame)
    actuals = [int(v) for v in frame["actual"]]

    # One draw per row, from a seed derived by name. Not `random` and not a
    # bare literal: `contracts/seeds.py` exists because eight copies of one
    # seed was considered a defect worth a module.
    generator = np.random.default_rng(derive_seed(seed, "calibration", "pit", label))
    draws = [float(v) for v in generator.random(len(actuals))]

    rows = list(zip(distributions, actuals, draws, strict=True))
    warnings: list[str] = []

    overall = _score_slice(label, rows, nominal_levels=nominal_levels)

    if "minutes" in frame.columns:
        played = [r for r, m in zip(rows, frame["minutes"], strict=True) if _i(m) > 0]
    else:
        played = [r for r, a in zip(rows, actuals, strict=True) if a != 0]
        warnings.append(
            "no `minutes` column, so `appeared only` falls back to a non-zero "
            "score, which is a proxy and not the same population"
        )
    appeared = _score_slice("appeared only", played, nominal_levels=nominal_levels)

    by_position: list[CalibrationSlice] = []
    if "position" in frame.columns:
        for position in Position:
            subset = [
                r for r, p in zip(rows, frame["position"], strict=True) if str(p) == position.value
            ]
            if subset:
                by_position.append(
                    _score_slice(position.value, subset, nominal_levels=nominal_levels)
                )

    by_price: list[CalibrationSlice] = []
    if "price" in frame.columns:
        for name, low, high in PRICE_BANDS:
            subset = [r for r, v in zip(rows, frame["price"], strict=True) if low <= _i(v) < high]
            if subset:
                by_price.append(_score_slice(name, subset, nominal_levels=nominal_levels))

    by_minutes: list[CalibrationSlice] = []
    if "minutes" in frame.columns:
        for name, low, high in MINUTES_BANDS:
            subset = [r for r, v in zip(rows, frame["minutes"], strict=True) if low <= _i(v) < high]
            if subset:
                by_minutes.append(_score_slice(name, subset, nominal_levels=nominal_levels))

    by_decile: list[CalibrationSlice] = []
    if "predicted" in frame.columns and frame.height >= 10:
        # Ranked by the *prediction*, never the outcome. Catches the conditional
        # failure a pooled number hides: overconfident on premiums, under on
        # cheap players, with the average landing on acceptable.
        order = sorted(range(len(rows)), key=lambda i: _f(frame["predicted"][i]))
        size = max(1, len(order) // 10)
        for d in range(10):
            chunk = order[d * size : (d + 1) * size] if d < 9 else order[9 * size :]
            if chunk:
                by_decile.append(
                    _score_slice(
                        f"decile {d + 1}", [rows[i] for i in chunk], nominal_levels=nominal_levels
                    )
                )

    if overall.degenerate:
        warnings.append(
            "the forecasts have effectively no width, so every metric here is "
            "uninterpretable rather than excellent"
        )

    return CalibrationReport(
        overall=overall,
        appeared_only=appeared,
        by_position=tuple(by_position),
        by_price_band=tuple(by_price),
        by_minutes_band=tuple(by_minutes),
        by_predicted_decile=tuple(by_decile),
        warnings=tuple(warnings),
    )


def frame_from_distributions(
    distributions: Sequence[Distribution],
    actuals: Sequence[int],
    *,
    extra: Mapping[str, Sequence[object]] | None = None,
) -> pl.DataFrame:
    """Assemble a scoring frame. Convenience for callers and tests alike."""
    data: dict[str, object] = {
        "actual": list(actuals),
        "pmf_support_min": [d.support_min for d in distributions],
        "pmf": [list(d.pmf) for d in distributions],
    }
    for key, values in (extra or {}).items():
        data[key] = list(values)
    return pl.DataFrame(data)
