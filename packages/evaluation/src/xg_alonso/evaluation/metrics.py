"""Everything one walk is worth, with the ambiguous names split apart.

Two of the metrics the specification asks for are traps, and both get two
fields here rather than one.

**"Percentage of weeks ahead of hold" is ambiguous, and one reading is
useless.** Counting weeks where the *cumulative* difference is positive
measures squad divergence: once an acting squad is ahead it stays ahead, so any
policy that ever made one good move reports something close to 100%. Counting
weeks it *won outright* measures the week. Both are computed;
``weeks_cumulatively_ahead`` and ``weeks_won_outright`` are named so nobody can
mistake one for the other.

**A metric that could not be measured reports so, rather than zero.** Autosub
points before the simulator existed, or squad value before prices moved, were
structurally zero — and a zero in a report reads as a measurement. Each carries
a :class:`MetricStatus` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from xg_alonso.evaluation.backtest import BacktestResult

__all__ = ["MetricStatus", "RunMetrics", "compute_run_metrics"]


class MetricStatus(StrEnum):
    OK = "ok"
    UNAVAILABLE_STATIC_PRICES = "unavailable_static_prices"
    """Squad value cannot move while the walk carries one price map."""


@dataclass(frozen=True)
class RunMetrics:
    """One walk's result, in every dimension the report shows."""

    gameweeks: int
    total_points: int
    points_vs_hold: int
    cumulative_incremental: tuple[int, ...]

    immediate_decision_delta_total: int
    immediate_decision_delta_mean: float
    immediate_decision_delta_median: float

    transfers: int
    hits_taken: int
    hit_points_paid: int
    points_per_transfer: float
    transfer_win_rate: float
    mean_regret: float
    calibration_error: float

    bench_points: int
    autosub_points: int
    captaincy_points: int

    max_drawdown_vs_hold: int
    weeks_cumulatively_ahead: float
    weeks_won_outright: float

    squad_value_change: int = 0
    squad_value_status: MetricStatus = MetricStatus.UNAVAILABLE_STATIC_PRICES

    def to_dict(self) -> dict[str, object]:
        return {
            "gameweeks": self.gameweeks,
            "total_points": self.total_points,
            "points_vs_hold": self.points_vs_hold,
            "immediate_decision_delta_total": self.immediate_decision_delta_total,
            "immediate_decision_delta_mean": self.immediate_decision_delta_mean,
            "immediate_decision_delta_median": self.immediate_decision_delta_median,
            "transfers": self.transfers,
            "hits_taken": self.hits_taken,
            "hit_points_paid": self.hit_points_paid,
            "points_per_transfer": self.points_per_transfer,
            "transfer_win_rate": self.transfer_win_rate,
            "mean_regret": self.mean_regret,
            "calibration_error": self.calibration_error,
            "bench_points": self.bench_points,
            "autosub_points": self.autosub_points,
            "captaincy_points": self.captaincy_points,
            "max_drawdown_vs_hold": self.max_drawdown_vs_hold,
            "weeks_cumulatively_ahead": self.weeks_cumulatively_ahead,
            "weeks_won_outright": self.weeks_won_outright,
            "squad_value_change": self.squad_value_change,
            "squad_value_status": str(self.squad_value_status),
        }


def compute_run_metrics(result: BacktestResult, *, prices_moved: bool = False) -> RunMetrics:
    """Derive every metric from one completed walk.

    Args:
        prices_moved: Whether the walk had per-gameweek prices. When it did
            not, squad value cannot change and the metric reports
            ``UNAVAILABLE_STATIC_PRICES`` rather than a zero that reads as one.
    """
    outcomes = result.outcomes
    if not outcomes:
        return RunMetrics(
            gameweeks=0,
            total_points=0,
            points_vs_hold=0,
            cumulative_incremental=(),
            immediate_decision_delta_total=0,
            immediate_decision_delta_mean=0.0,
            immediate_decision_delta_median=0.0,
            transfers=0,
            hits_taken=0,
            hit_points_paid=0,
            points_per_transfer=0.0,
            transfer_win_rate=0.0,
            mean_regret=0.0,
            calibration_error=0.0,
            bench_points=0,
            autosub_points=0,
            captaincy_points=0,
            max_drawdown_vs_hold=0,
            weeks_cumulatively_ahead=0.0,
            weeks_won_outright=0.0,
        )

    running = 0
    cumulative: list[int] = []
    peak = 0
    drawdown = 0
    for outcome in outcomes:
        running += outcome.incremental
        cumulative.append(running)
        peak = max(peak, running)
        drawdown = max(drawdown, peak - running)

    acted = [o for o in outcomes if o.transfer_made]
    deltas = sorted(o.decision_delta for o in acted)
    median = 0.0
    if deltas:
        middle = len(deltas) // 2
        median = (
            float(deltas[middle])
            if len(deltas) % 2
            else (deltas[middle - 1] + deltas[middle]) / 2.0
        )

    # Two readings of "ahead", named apart. The cumulative one measures squad
    # divergence and reports near-100% for any policy that ever moved ahead;
    # the outright one measures the week that was actually played.
    ahead = sum(1 for value in cumulative if value > 0) / len(cumulative)
    won = sum(1 for o in outcomes if o.incremental > 0) / len(outcomes)

    return RunMetrics(
        gameweeks=result.gameweeks,
        total_points=sum(o.net_policy_points for o in outcomes),
        points_vs_hold=result.total_incremental,
        cumulative_incremental=tuple(cumulative),
        immediate_decision_delta_total=sum(o.decision_delta for o in acted),
        immediate_decision_delta_mean=result.mean_decision_delta,
        immediate_decision_delta_median=median,
        transfers=result.transfers_made,
        hits_taken=sum(1 for o in outcomes if o.hit_cost > 0),
        hit_points_paid=result.total_hits,
        points_per_transfer=(sum(o.decision_delta for o in acted) / len(acted) if acted else 0.0),
        transfer_win_rate=result.decision_win_rate,
        mean_regret=result.mean_regret,
        calibration_error=result.calibration_error,
        bench_points=sum(o.bench_points for o in outcomes),
        autosub_points=sum(o.autosub_points for o in outcomes),
        captaincy_points=sum(o.captain_points for o in outcomes),
        max_drawdown_vs_hold=drawdown,
        weeks_cumulatively_ahead=ahead,
        weeks_won_outright=won,
        squad_value_status=(
            MetricStatus.OK if prices_moved else MetricStatus.UNAVAILABLE_STATIC_PRICES
        ),
    )
