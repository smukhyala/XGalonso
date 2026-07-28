"""Score assembled expected points against what actually happened.

Every other accuracy measure in this repository scores a *component* — minutes,
goals, clean sheets — in isolation. None of them scores the number a manager
actually reads. `expected_points` is the product of the whole chain: component
models, the pinned scoring rules, and the assembly step. A chain can be sound at
every link and still be wrong end to end, and until this module existed nothing
would have caught that.

**Every metric here is paired with a degeneracy check**, which is the lesson the
constant-output bug taught at some cost: a model that predicted `0.0000` for
every player scored `+48%` MAE skill, because zero genuinely does minimise MAE
on a label that is 96% zeros. An accuracy number that cannot tell a constant
predictor from a working one is not an accuracy number. So each slice carries
its prediction spread and a `degenerate` flag, and `skill` is forced to zero
when the flag is set however flattering the error looks.

Ranking is reported alongside error for the same reason. Transfers are chosen by
comparing players, so a model can carry a large constant bias and still make
every decision correctly — or be well calibrated on average and rank nobody
correctly. `spearman` and `top_k` measure the property the optimizer actually
consumes; `mae` and `bias` measure the property the interface displays.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import SupportsFloat, SupportsInt, cast

import polars as pl

__all__ = [
    "PRICE_BANDS",
    "AccuracyReport",
    "AccuracySlice",
    "TopKResult",
    "score_predictions",
    "spearman",
]

# Below this spread the predictions are effectively one number repeated, and
# every error metric computed from them is uninterpretable.
DEGENERATE_SD = 1e-6

# Bands in tenths of a million, upper bound exclusive. These are the tiers a
# manager actually shops in — budget enabler, mid-price, premium — and errors
# behave differently in each, so a single pooled number hides the failure.
PRICE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("budget (<£5.5m)", 0, 55),
    ("mid (£5.5-8.0m)", 55, 80),
    ("premium (£8.0-11.0m)", 80, 110),
    ("elite (£11.0m+)", 110, 10_000),
)

TOP_K_VALUES: tuple[int, ...] = (10, 20, 50)


def _f(value: object) -> float:
    """Narrow a Polars aggregate to a float, treating an empty result as zero.

    Polars types its aggregations as a union that includes `date` and
    `timedelta`, because the identical call on a temporal column returns those.
    Every column this module touches is numeric, so the narrowing is sound; it
    is stated once here rather than scattered as casts at each call site.
    """
    return 0.0 if value is None else float(cast(SupportsFloat, value))


def _i(value: object) -> int:
    """The integer counterpart of :func:`_f`, for row counts."""
    return 0 if value is None else int(cast(SupportsInt, value))


def spearman(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Rank correlation between predictions and outcomes.

    Implemented directly rather than pulled from scipy: it is Pearson on average
    ranks, ties shared, and adding a dependency for fifteen lines would be the
    wrong trade. Returns 0.0 when either side has no rank variation, which is
    the honest answer — a constant vector ranks nothing.
    """
    n = len(predicted)
    if n < 2 or len(actual) != n:
        return 0.0

    frame = pl.DataFrame({"predicted": list(predicted), "actual": list(actual)}).with_columns(
        pl.col("predicted").rank(method="average").alias("rank_predicted"),
        pl.col("actual").rank(method="average").alias("rank_actual"),
    )

    rank_predicted = frame["rank_predicted"]
    rank_actual = frame["rank_actual"]
    sd_predicted = _f(rank_predicted.std(ddof=0))
    sd_actual = _f(rank_actual.std(ddof=0))
    if sd_predicted < DEGENERATE_SD or sd_actual < DEGENERATE_SD:
        return 0.0

    mean_predicted = _f(rank_predicted.mean())
    mean_actual = _f(rank_actual.mean())
    covariance = _f(((rank_predicted - mean_predicted) * (rank_actual - mean_actual)).mean())
    return covariance / (sd_predicted * sd_actual)


@dataclass(frozen=True)
class AccuracySlice:
    """Error and ranking quality over one subset of predictions."""

    name: str
    n: int
    mae: float
    rmse: float
    bias: float
    """Mean signed error. Positive means the model over-predicts this slice."""
    spearman: float
    prediction_sd: float
    baseline_mae: float
    """MAE of predicting this slice's mean outcome for every player."""

    @property
    def degenerate(self) -> bool:
        """Whether the predictions carry no information to measure."""
        return self.prediction_sd < DEGENERATE_SD

    @property
    def skill(self) -> float:
        """Fractional MAE improvement over predicting the mean. Zero if degenerate.

        The guard is the point. A constant predictor can beat the mean on a
        zero-inflated label, and reporting that as skill is how a broken model
        gets promoted.
        """
        if self.degenerate or self.baseline_mae < DEGENERATE_SD:
            return 0.0
        return 1.0 - self.mae / self.baseline_mae

    def row(self) -> str:
        flag = "  DEGENERATE" if self.degenerate else ""
        return (
            f"{self.name:<24} n={self.n:>6}  mae={self.mae:>6.3f}  rmse={self.rmse:>6.3f}  "
            f"bias={self.bias:>+6.3f}  rho={self.spearman:>+5.2f}  "
            f"skill={self.skill:>+6.1%}{flag}"
        )


@dataclass(frozen=True)
class TopKResult:
    """How good the players the model liked most turned out to be.

    This is the metric closest to the product. A transfer promotes somebody into
    the squad on the strength of their rank, so what matters is whether the
    highly-ranked players actually delivered — not whether their point estimates
    were close.
    """

    k: int
    precision: float
    """Share of the predicted top k that were also in the actual top k."""
    mean_actual: float
    """Mean actual points scored by the predicted top k."""
    mean_overall: float
    """Mean actual points across every scored player."""

    @property
    def lift(self) -> float:
        """How many times the field average the picked players returned."""
        if self.mean_overall < DEGENERATE_SD:
            return 0.0
        return self.mean_actual / self.mean_overall

    def row(self) -> str:
        return (
            f"top-{self.k:<3} precision={self.precision:>5.1%}  "
            f"mean actual={self.mean_actual:>5.2f} vs field {self.mean_overall:>5.2f}  "
            f"lift={self.lift:>4.2f}x"
        )


@dataclass(frozen=True)
class AccuracyReport:
    """The full picture: pooled, sliced, and ranked."""

    overall: AccuracySlice
    by_position: tuple[AccuracySlice, ...] = ()
    by_price_band: tuple[AccuracySlice, ...] = ()
    top_k: tuple[TopKResult, ...] = ()
    appeared_only: AccuracySlice | None = None
    """The same metrics restricted to players who actually took the field.

    Scoring every player pools in hundreds who never played, whose zero is
    trivially predictable. That inflates the headline and hides whether the
    model is any good at the population a manager chooses between.
    """
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        lines = ["Assembled expected points vs actual", "", self.overall.row()]
        if self.appeared_only is not None:
            lines.append(self.appeared_only.row())
        if self.by_position:
            lines += ["", "By position"] + [s.row() for s in self.by_position]
        if self.by_price_band:
            lines += ["", "By price band"] + [s.row() for s in self.by_price_band]
        if self.top_k:
            lines += ["", "Ranking"] + [t.row() for t in self.top_k]
        if self.warnings:
            lines += ["", "Warnings"] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def _slice(frame: pl.DataFrame, name: str) -> AccuracySlice:
    """Compute one slice's metrics. Assumes `predicted` and `actual` columns."""
    n = frame.height
    if n == 0:
        return AccuracySlice(
            name=name,
            n=0,
            mae=0.0,
            rmse=0.0,
            bias=0.0,
            spearman=0.0,
            prediction_sd=0.0,
            baseline_mae=0.0,
        )

    predicted = frame["predicted"]
    actual = frame["actual"]
    error = predicted - actual
    mean_actual = _f(actual.mean())

    return AccuracySlice(
        name=name,
        n=n,
        mae=_f(error.abs().mean()),
        rmse=math.sqrt(_f((error**2).mean())),
        bias=_f(error.mean()),
        spearman=spearman(predicted.to_list(), actual.to_list()),
        prediction_sd=_f(predicted.std(ddof=0)),
        baseline_mae=_f((actual - mean_actual).abs().mean()),
    )


def _top_k_within(frame: pl.DataFrame, k: int) -> TopKResult:
    """Rank by prediction, then check the outcome. Never the other way round."""
    n = frame.height
    k = min(k, n)
    if k == 0:
        return TopKResult(k=0, precision=0.0, mean_actual=0.0, mean_overall=0.0)

    # Ties are broken on player_code so the result does not depend on row order.
    by_predicted = frame.sort(["predicted", "player_code"], descending=[True, False]).head(k)
    by_actual = frame.sort(["actual", "player_code"], descending=[True, False]).head(k)

    picked = set(by_predicted["player_code"].to_list())
    truly_best = set(by_actual["player_code"].to_list())

    return TopKResult(
        k=k,
        precision=len(picked & truly_best) / k,
        mean_actual=_f(by_predicted["actual"].mean()),
        mean_overall=_f(frame["actual"].mean()),
    )


def _top_k(frame: pl.DataFrame, k: int) -> TopKResult:
    """Top-k measured *within* a gameweek, then averaged across gameweeks.

    Pooling every gameweek first would ask a question nobody faces: it ranks
    the best player-gameweeks of an entire season against each other, so a
    model could score well by knowing which *weeks* were high-scoring while
    ranking players within a week no better than chance. A manager picks among
    players for one deadline, so that is the unit the metric has to use.

    Falls back to a single pooled ranking when the frame carries no gameweek
    column, which is the right behaviour for a caller scoring one gameweek.
    """
    if "gameweek_id" not in frame.columns:
        return _top_k_within(frame, k)

    per_gameweek = [
        _top_k_within(group, k)
        for (_,), group in frame.group_by(["gameweek_id"], maintain_order=True)
        if group.height
    ]
    if not per_gameweek:
        return TopKResult(k=0, precision=0.0, mean_actual=0.0, mean_overall=0.0)

    weeks = len(per_gameweek)
    return TopKResult(
        k=max(r.k for r in per_gameweek),
        precision=sum(r.precision for r in per_gameweek) / weeks,
        mean_actual=sum(r.mean_actual for r in per_gameweek) / weeks,
        mean_overall=sum(r.mean_overall for r in per_gameweek) / weeks,
    )


def score_predictions(frame: pl.DataFrame) -> AccuracyReport:
    """Score assembled expected points against actual points.

    ``frame`` must carry ``player_code``, ``predicted`` and ``actual``. Optional
    ``position``, ``price`` and ``minutes`` columns unlock the corresponding
    breakdowns rather than being required — a caller with only the two core
    columns still gets the pooled numbers.

    Rows where either side is null are dropped, and the count is reported as a
    warning rather than passed over silently: a model that declines to predict
    for half the league is not more accurate, it is less useful.
    """
    required = {"player_code", "predicted", "actual"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"score_predictions needs columns {sorted(missing)}")

    warnings: list[str] = []
    before = frame.height
    scored = frame.drop_nulls(subset=["predicted", "actual"])
    dropped = before - scored.height
    if dropped:
        warnings.append(f"{dropped} of {before} rows had a null prediction or outcome")

    if scored.is_empty():
        return AccuracyReport(
            overall=_slice(scored, "overall"), warnings=(*warnings, "no scorable rows")
        )

    overall = _slice(scored, "overall")
    if overall.degenerate:
        warnings.append(
            "predictions are constant to within 1e-6 — every error metric below is "
            "uninterpretable and skill has been forced to zero"
        )

    by_position: list[AccuracySlice] = []
    if "position" in scored.columns:
        for position in ("GKP", "DEF", "MID", "FWD"):
            subset = scored.filter(pl.col("position") == position)
            if subset.height:
                by_position.append(_slice(subset, position))

    by_price_band: list[AccuracySlice] = []
    if "price" in scored.columns:
        for label, low, high in PRICE_BANDS:
            subset = scored.filter((pl.col("price") >= low) & (pl.col("price") < high))
            if subset.height:
                by_price_band.append(_slice(subset, label))

    appeared_only: AccuracySlice | None = None
    if "minutes" in scored.columns:
        played = scored.filter(pl.col("minutes") > 0)
        if played.height:
            appeared_only = _slice(played, "played only")

    # Each k is clamped to the population it will actually rank within — one
    # gameweek when grouped — then de-duplicated, so a pool smaller than the
    # largest k does not report the same row three times.
    if "gameweek_id" in scored.columns:
        population = _i(scored.group_by("gameweek_id").len()["len"].max())
    else:
        population = scored.height

    effective_k: list[int] = []
    for k in TOP_K_VALUES:
        clamped = min(k, population)
        if clamped and clamped not in effective_k:
            effective_k.append(clamped)
    top_k = tuple(_top_k(scored, k) for k in effective_k)

    return AccuracyReport(
        overall=overall,
        by_position=tuple(by_position),
        by_price_band=tuple(by_price_band),
        top_k=top_k,
        appeared_only=appeared_only,
        warnings=tuple(warnings),
    )
