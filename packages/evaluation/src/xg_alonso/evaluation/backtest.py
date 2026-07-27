"""Walk-forward policy backtesting.

`CLAUDE.md` is explicit that model metrics are intermediate and the real question
is decision quality. This module answers that question directly: **does acting on
the recommendations beat holding?**

The design is a *policy* backtest, not a prediction backtest. Two squads start
identical and walk the season together — one takes every recommendation, one
never transfers. Both are scored on the same actual outcomes. The difference is
the product metric, and unlike a points-MAE it cannot be improved by a model
that ranks well while recommending badly.

**No random splits, ever.** The walk is strictly forward: at gameweek *N* the
features see only data with ``available_time`` before gameweek *N*'s deadline,
and the score comes from gameweek *N*'s actual results, which are by
construction not yet visible. That ordering is the whole point, so the deadline
is derived from real kickoff times rather than assumed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import polars as pl

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, Season, TenthsOfMillion
from xg_alonso.contracts.prediction import PlayerPrediction
from xg_alonso.contracts.recommendation import TransferRecommendation
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.optimization.lineup import best_starting_xi

__all__ = [
    "BacktestResult",
    "GameweekOutcome",
    "actual_points",
    "apply_transfer",
    "gameweek_deadlines",
    "score_squad",
    "walk_forward",
]

#: FPL deadlines fall roughly 90 minutes before the first kickoff of a gameweek.
#: Subtracting it keeps the cutoff conservative: using the kickoff itself would
#: admit up to an hour and a half of team news the manager could not have had.
_DEADLINE_MARGIN = timedelta(minutes=90)


@dataclass(frozen=True)
class GameweekOutcome:
    """What happened in one gameweek, for both policies."""

    season: Season
    gameweek: GameweekId
    policy_points: int
    hold_points: int
    hit_cost: int
    transfer_made: bool
    player_out: PlayerCode | None
    player_in: PlayerCode | None
    predicted_gain: float
    decision_delta: int = 0
    """Points the incoming player outscored the outgoing one by, this gameweek.

    This isolates *this week's* decision. ``incremental`` cannot: it compares two
    squads that have been diverging for weeks, so once the acting squad is ahead
    it wins every gameweek regardless of what was decided this one. A per-week
    win rate built on ``incremental`` therefore measures squad divergence and
    reports near-100% for any policy that ever made a good move.
    """

    @property
    def net_policy_points(self) -> int:
        """Points after the transfer hit — what the manager actually banks."""
        return self.policy_points - self.hit_cost

    @property
    def incremental(self) -> int:
        """The headline number: points gained over holding, net of hits."""
        return self.net_policy_points - self.hold_points


@dataclass
class BacktestResult:
    """A completed walk, with the decision-quality metrics that matter."""

    outcomes: list[GameweekOutcome] = field(default_factory=list)

    @property
    def gameweeks(self) -> int:
        return len(self.outcomes)

    @property
    def total_incremental(self) -> int:
        """Total points gained over the hold baseline. The headline metric."""
        return sum(o.incremental for o in self.outcomes)

    @property
    def transfers_made(self) -> int:
        return sum(1 for o in self.outcomes if o.transfer_made)

    @property
    def total_hits(self) -> int:
        return sum(o.hit_cost for o in self.outcomes)

    @property
    def decision_win_rate(self) -> float:
        """Share of transfers where the incoming player outscored the outgoing one.

        The honest per-decision measure. A coin-flipping policy lands near 50%;
        anything much above that is genuine skill, and anything near 100% is a
        bug in the harness rather than a very good model.
        """
        acted = [o for o in self.outcomes if o.transfer_made]
        if not acted:
            return 0.0
        return sum(1 for o in acted if o.decision_delta > 0) / len(acted)

    @property
    def mean_decision_delta(self) -> float:
        """Average points gained per transfer, in the week it was made."""
        acted = [o for o in self.outcomes if o.transfer_made]
        if not acted:
            return 0.0
        return sum(o.decision_delta for o in acted) / len(acted)

    @property
    def mean_regret(self) -> float:
        """Average loss on transfers that turned out worse than holding.

        Reported separately from the mean gain because the two are not
        symmetric in practice: one badly-timed hit can erase several good weeks.
        """
        losses = [
            -o.decision_delta for o in self.outcomes if o.transfer_made and o.decision_delta < 0
        ]
        if not losses:
            return 0.0
        return sum(losses) / len(losses)

    @property
    def calibration_error(self) -> float:
        """Mean absolute gap between predicted and realised gain.

        A model that predicts +5 and delivers +1 every week is systematically
        overconfident even when it is directionally right, and only this metric
        catches that.
        """
        acted = [o for o in self.outcomes if o.transfer_made]
        if not acted:
            return 0.0
        return sum(abs(o.predicted_gain - o.decision_delta) for o in acted) / len(acted)

    def summary(self) -> str:
        """A short, honest report."""
        if not self.outcomes:
            return "No gameweeks were evaluated."
        return "\n".join(
            [
                f"Gameweeks evaluated      {self.gameweeks}",
                f"Transfers made           {self.transfers_made}",
                f"Points paid in hits      {self.total_hits}",
                "",
                f"Season vs hold           {self.total_incremental:+d} pts",
                "",
                "Per-decision (isolates each transfer from squad drift):",
                f"  Transfers that gained  {self.decision_win_rate:.0%}",
                f"  Mean gain per transfer {self.mean_decision_delta:+.2f} pts",
                f"  Mean loss when wrong   {self.mean_regret:.2f} pts",
                f"  Prediction error       {self.calibration_error:.2f} pts",
            ]
        )


def gameweek_deadlines(player_stats: pl.DataFrame) -> pl.DataFrame:
    """Derive each gameweek's deadline from its first kickoff.

    Historical deadlines are not in the archive, but kickoff times are, and the
    deadline is a fixed offset before the first match. Deriving it keeps the
    backtest honest about what a manager could have known.
    """
    return (
        player_stats.filter(pl.col("kickoff_time").is_not_null())
        .group_by(["season", "gameweek_id"])
        .agg((pl.col("kickoff_time").min() - _DEADLINE_MARGIN).alias("deadline"))
        .sort(["season", "gameweek_id"])
    )


def actual_points(
    player_stats: pl.DataFrame, *, season: Season, gameweek: GameweekId
) -> dict[PlayerCode, int]:
    """What each player actually scored in one gameweek.

    A player with two fixtures in a gameweek scores in both, so points are
    summed rather than taken from a single row.
    """
    rows = player_stats.filter(
        (pl.col("season") == str(season)) & (pl.col("gameweek_id") == int(gameweek))
    )
    if rows.is_empty():
        return {}
    totals = rows.group_by("player_code").agg(pl.col("total_points").sum().alias("points"))
    return {
        PlayerCode(int(r["player_code"])): int(r["points"] or 0)
        for r in totals.iter_rows(named=True)
    }


def score_squad(
    squad: SquadState,
    points: dict[PlayerCode, int],
    *,
    predictions: dict[PlayerCode, PlayerPrediction] | None = None,
    rules: SquadRules | None = None,
) -> int:
    """Score a squad against actual results.

    **The XI is chosen from predictions, then scored against outcomes** — that
    temporal order is the whole point. Picking the eleven that happened to do
    best would measure hindsight, so the selection sees only what was knowable
    at the deadline and the scoring sees only what happened after.

    When ``predictions`` and ``rules`` are supplied the best legal XI is chosen
    and its captain doubled. Without them it falls back to the squad's stored
    slots, which is what the earlier version did unconditionally — and which
    scored a fixed eleven with whatever sat in slot 1 as captain.

    Autosubs are not modelled: a benched player never replaces a starter who
    fails to play. That understates every squad identically, so a comparison
    between policies stays fair.
    """
    if predictions is not None and rules is not None:
        selection = best_starting_xi(squad.picks, predictions, rules)
        starters = selection.starters
        captain = selection.captain
    else:
        starters = squad.starters
        captain = next((p.player_code for p in squad.picks if p.is_captain), None)

    total = 0
    for pick in starters:
        scored = points.get(pick.player_code, 0)
        total += scored * (2 if pick.player_code == captain else 1)
    return total


def apply_transfer(
    squad: SquadState,
    recommendation: TransferRecommendation,
    *,
    prices: dict[PlayerCode, TenthsOfMillion],
    positions: dict[PlayerCode, str],
    teams: dict[PlayerCode, int],
) -> SquadState:
    """Return the squad that results from acting on a recommendation.

    A hold returns the squad unchanged. Free transfers accumulate up to the cap
    when nothing is done, which is what makes rolling a real alternative rather
    than a wasted week.
    """
    from xg_alonso.contracts.identifiers import TeamId
    from xg_alonso.contracts.prediction import Position

    if recommendation.package.is_hold:
        return squad.model_copy(update={"free_transfers": min(5, squad.free_transfers + 1)})

    move = recommendation.package.moves[0]
    outgoing = squad.by_code(move.player_out)
    if outgoing is None:
        raise KeyError(f"cannot sell {move.player_out}: not in the squad")

    incoming = SquadPick(
        player_code=move.player_in,
        position=Position(positions[move.player_in]),
        team_id=TeamId(teams[move.player_in]),
        purchase_price=move.purchase_price,
        current_price=prices[move.player_in],
        selling_price=move.purchase_price,
        squad_slot=outgoing.squad_slot,
        is_captain=outgoing.is_captain,
        is_vice_captain=outgoing.is_vice_captain,
    )

    picks = tuple(incoming if p.player_code == move.player_out else p for p in squad.picks)
    return squad.model_copy(
        update={
            "picks": picks,
            "bank": recommendation.package.bank_after,
            "free_transfers": max(0, squad.free_transfers - 1),
        }
    )


RecommendFn = Callable[
    [SquadState, GameweekId, Season],
    tuple[TransferRecommendation, dict[PlayerCode, PlayerPrediction]],
]
"""``(squad, gameweek, season) -> (recommendation, predictions)``."""


def walk_forward(
    *,
    initial_squad: SquadState,
    season: Season,
    gameweeks: Sequence[GameweekId],
    recommend_fn: RecommendFn,
    player_stats: pl.DataFrame,
    prices: dict[PlayerCode, TenthsOfMillion],
    positions: dict[PlayerCode, str],
    teams: dict[PlayerCode, int],
    rules: SquadRules | None = None,
) -> BacktestResult:
    """Walk a season, comparing an acting policy against never transferring.

    Both squads start identical. The acting squad takes every recommendation;
    the hold squad never changes. Both are scored on the same actual results.

    Gameweeks with no recorded outcome are skipped rather than scored as zero —
    a blank gameweek and a gameweek nobody played are different things.
    """
    result = BacktestResult()
    acting = initial_squad
    holding = initial_squad

    for gameweek in gameweeks:
        outcomes = actual_points(player_stats, season=season, gameweek=gameweek)
        if not outcomes:
            continue

        recommendation, predictions = recommend_fn(acting, gameweek, season)

        # Score with the squad as it stands *for* this gameweek — the transfer
        # is made before the deadline, so the incoming player plays this week.
        acting_after = apply_transfer(
            acting, recommendation, prices=prices, positions=positions, teams=teams
        )

        move = recommendation.package.moves[0] if recommendation.package.moves else None
        decision_delta = (
            outcomes.get(move.player_in, 0) - outcomes.get(move.player_out, 0) if move else 0
        )

        result.outcomes.append(
            GameweekOutcome(
                season=season,
                gameweek=gameweek,
                policy_points=score_squad(
                    acting_after, outcomes, predictions=predictions, rules=rules
                ),
                hold_points=score_squad(holding, outcomes, predictions=predictions, rules=rules),
                hit_cost=recommendation.package.hit_cost,
                transfer_made=not recommendation.package.is_hold,
                player_out=(
                    recommendation.package.moves[0].player_out
                    if recommendation.package.moves
                    else None
                ),
                player_in=(
                    recommendation.package.moves[0].player_in
                    if recommendation.package.moves
                    else None
                ),
                predicted_gain=recommendation.expected_points_gain,
                decision_delta=decision_delta,
            )
        )
        acting = acting_after
        holding = holding.model_copy(update={"free_transfers": min(5, holding.free_transfers + 1)})

    return result
