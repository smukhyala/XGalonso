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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import polars as pl

from xg_alonso.contracts.constraints import SquadViolation
from xg_alonso.contracts.identifiers import (
    GameweekId,
    PlayerCode,
    Season,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.recommendation import TransferMove, TransferRecommendation
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.constraints import check_squad
from xg_alonso.domain.pricing import selling_price
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.transfers import accrue, settle_gameweek
from xg_alonso.evaluation.simulator import simulate_squad

__all__ = [
    "BacktestResult",
    "GameweekOutcome",
    "actual_fixture_counts",
    "actual_minutes",
    "actual_points",
    "actual_prices",
    "apply_transfer",
    "gameweek_deadlines",
    "price_at_deadline",
    "refusals",
    "reprice_squad",
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


def actual_minutes(
    player_stats: pl.DataFrame, *, season: Season, gameweek: GameweekId
) -> dict[PlayerCode, int]:
    """Minutes each player played in one gameweek.

    Summed across fixtures, not maximised: a player who played 20 minutes in one
    leg of a double and none in the other *did* play, and must not be
    substituted. The mirror of :func:`actual_points`, and the input the autosub
    simulator needs.
    """
    rows = player_stats.filter(
        (pl.col("season") == str(season)) & (pl.col("gameweek_id") == int(gameweek))
    )
    if rows.is_empty():
        return {}
    totals = rows.group_by("player_code").agg(pl.col("minutes").sum().alias("minutes"))
    return {
        PlayerCode(int(r["player_code"])): int(r["minutes"] or 0)
        for r in totals.iter_rows(named=True)
    }


def actual_fixture_counts(
    player_stats: pl.DataFrame, *, season: Season, gameweek: GameweekId
) -> dict[PlayerCode, int]:
    """How many fixtures each player's club had. Zero is a blank, two a double."""
    rows = player_stats.filter(
        (pl.col("season") == str(season)) & (pl.col("gameweek_id") == int(gameweek))
    )
    if rows.is_empty():
        return {}
    counts = rows.group_by("player_code").agg(pl.len().alias("fixtures"))
    return {
        PlayerCode(int(r["player_code"])): int(r["fixtures"]) for r in counts.iter_rows(named=True)
    }


def actual_prices(
    player_stats: pl.DataFrame, *, season: Season, gameweek: GameweekId
) -> dict[PlayerCode, TenthsOfMillion]:
    """Listed price at a gameweek, from the ``value`` column.

    **Lagged by the caller, not here.** ``value`` for gameweek *N* is recorded
    alongside gameweek *N*'s result, which is after its deadline. Gameweek
    *N-1*'s value is provably knowable at *N*'s deadline; *N*'s may not be. See
    :func:`price_at_deadline`.

    Without this the walk used one static price map for the whole season, so no
    price ever moved and squad value was a constant by construction.
    """
    rows = player_stats.filter(
        (pl.col("season") == str(season)) & (pl.col("gameweek_id") == int(gameweek))
    )
    if rows.is_empty():
        return {}
    latest = rows.group_by("player_code").agg(pl.col("value").last().alias("value"))
    return {
        PlayerCode(int(r["player_code"])): TenthsOfMillion(int(r["value"]))
        for r in latest.iter_rows(named=True)
        if r["value"] is not None
    }


def price_at_deadline(
    player_stats: pl.DataFrame, *, season: Season, gameweek: GameweekId
) -> dict[PlayerCode, TenthsOfMillion]:
    """Prices a manager could have seen at this gameweek's deadline.

    The previous gameweek's listed value. At gameweek 1 there is no previous
    week, so the map is empty and the caller keeps whatever opening prices it
    already had.
    """
    if int(gameweek) <= 1:
        return {}
    return actual_prices(player_stats, season=season, gameweek=GameweekId(int(gameweek) - 1))


def reprice_squad(
    squad: SquadState, *, prices: Mapping[PlayerCode, TenthsOfMillion], rules: SquadRules
) -> SquadState:
    """Update every pick's current and selling price to today's market.

    Runs *before* the recommendation, because prices move before the deadline
    and a manager decides against the prices they can see. Selling price is
    recomputed through :func:`~xg_alonso.domain.pricing.selling_price`, so the
    sell-on fee and its round-down are applied in exactly one place.

    A player absent from ``prices`` keeps the price he had, which is the honest
    reading of a missing row: not that he became free, but that nothing new was
    published about him.
    """
    repriced = []
    for pick in squad.picks:
        current = prices.get(pick.player_code, pick.current_price)
        repriced.append(
            pick.model_copy(
                update={
                    "current_price": current,
                    "selling_price": selling_price(
                        purchase_price=pick.purchase_price,
                        current_price=current,
                        rules=rules,
                    ),
                }
            )
        )
    return squad.model_copy(update={"picks": tuple(repriced)})


def score_squad(
    squad: SquadState,
    points: dict[PlayerCode, int],
    *,
    predictions: dict[PlayerCode, PlayerPrediction] | None = None,
    rules: SquadRules | None = None,
    minutes: Mapping[PlayerCode, int] | None = None,
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

    Supplying ``minutes`` enables the game's own mechanics: substitutes come on
    for starters who did not play, and the vice-captain inherits the armband
    when the captain does not. Omitting it reproduces the outcome-blind
    behaviour exactly — no substitutions, and the captain doubled whether or not
    he played — which is what every caller got before autosubs were modelled.

    The decomposed result is available from
    :func:`~xg_alonso.evaluation.simulator.simulate_squad`; this returns only
    the total.
    """
    return simulate_squad(
        squad,
        points,
        predictions=predictions,
        rules=rules,
        minutes=minutes,
    ).total


def apply_transfer(
    squad: SquadState,
    recommendation: TransferRecommendation,
    *,
    prices: dict[PlayerCode, TenthsOfMillion],
    positions: dict[PlayerCode, str],
    teams: dict[PlayerCode, int],
    rules: SquadRules,
) -> SquadState:
    """Return the squad that results from acting on a recommendation.

    A hold returns the squad unchanged. The allowance is settled through
    `domain.transfers`, so the cap and the per-transfer charge come from the
    pinned snapshot rather than from a literal here — and a transferring week
    now accrues its `+1` like any other, instead of sliding toward a permanent
    hit as `max(0, ft - 1)` did.
    """
    if recommendation.package.is_hold:
        return squad.model_copy(
            update={"free_transfers": accrue(squad.free_transfers, rules=rules)}
        )

    move = recommendation.package.moves[0]
    outgoing = squad.by_code(move.player_out)
    if outgoing is None:
        raise KeyError(f"cannot sell {move.player_out}: not in the squad")

    incoming = SquadPick(
        player_code=move.player_in,
        position=Position(positions[move.player_in]),
        team_id=TeamId(teams[move.player_in]),
        purchase_price=move.purchase_price,
        # At the moment of purchase these are the same number by definition.
        # Reading `current_price` from a static map let the two diverge, and a
        # `SquadPick` whose selling price sits outside its own price band is a
        # contract violation waiting for the next reprice.
        current_price=move.purchase_price,
        selling_price=move.purchase_price,
        squad_slot=outgoing.squad_slot,
        is_captain=outgoing.is_captain,
        is_vice_captain=outgoing.is_vice_captain,
    )

    picks = tuple(incoming if p.player_code == move.player_out else p for p in squad.picks)
    bank_after = recommendation.package.bank_after

    # Legality is checked before the move is committed, never after. A backtest
    # that silently applied an unaffordable or quota-breaking transfer was
    # measuring a policy the game would have refused to run.
    violations = _refusals(
        picks,
        outgoing=outgoing,
        move=move,
        bank_before=squad.bank,
        bank_after=bank_after,
        rules=rules,
    )
    if violations:
        # The transfer never happened, so the allowance accrues as it would in
        # any quiet week.
        return squad.model_copy(
            update={"free_transfers": accrue(squad.free_transfers, rules=rules)}
        )

    ledger = settle_gameweek(
        free_transfers=squad.free_transfers,
        transfers_made=len(recommendation.package.moves),
        rules=rules,
    )
    return squad.model_copy(
        update={
            "picks": picks,
            "bank": bank_after,
            "free_transfers": ledger.free_transfers_after,
        }
    )


def refusals(
    squad: SquadState,
    recommendation: TransferRecommendation,
    *,
    positions: Mapping[PlayerCode, str],
    teams: Mapping[PlayerCode, int],
    rules: SquadRules,
) -> list[SquadViolation]:
    """Why this recommendation would be refused, or an empty list.

    Exposed alongside :func:`apply_transfer` so a caller can report *why* a
    move did not happen. ``apply_transfer`` itself returns the unchanged squad,
    because a refused transfer and a hold produce the same squad — but they are
    not the same event, and a simulator needs to say which.
    """
    if recommendation.package.is_hold:
        return []
    move = recommendation.package.moves[0]
    outgoing = squad.by_code(move.player_out)
    if outgoing is None:
        return [SquadViolation(rule="not_owned", detail=f"{move.player_out} is not in the squad")]
    incoming = SquadPick(
        player_code=move.player_in,
        position=Position(positions[move.player_in]),
        team_id=TeamId(teams[move.player_in]),
        purchase_price=move.purchase_price,
        current_price=move.purchase_price,
        selling_price=move.purchase_price,
        squad_slot=outgoing.squad_slot,
    )
    picks = tuple(incoming if p.player_code == move.player_out else p for p in squad.picks)
    return _refusals(
        picks,
        outgoing=outgoing,
        move=move,
        bank_before=squad.bank,
        bank_after=recommendation.package.bank_after,
        rules=rules,
    )


def _refusals(
    picks: tuple[SquadPick, ...],
    *,
    outgoing: SquadPick,
    move: TransferMove,
    bank_before: TenthsOfMillion,
    bank_after: TenthsOfMillion,
    rules: SquadRules,
) -> list[SquadViolation]:
    """Affordability, then squad legality. Both before anything is committed."""
    violations: list[SquadViolation] = []

    # `TransferPackage` already refuses a negative `bank_after`, so a *self-
    # consistent* package cannot describe an unaffordable move. What it cannot
    # check is whether its `bank_before` matches the squad it is applied to — a
    # recommendation computed against a stale bank passes its own validator and
    # is still unaffordable in fact. That is what this compares.
    affordable = TenthsOfMillion(outgoing.selling_price + bank_before)
    if move.purchase_price > affordable:
        violations.append(
            SquadViolation(
                rule="budget",
                detail=(
                    f"{move.player_in} costs {move.purchase_price} but selling "
                    f"{move.player_out} for {outgoing.selling_price} plus a bank of "
                    f"{bank_before} affords only {affordable}"
                ),
            )
        )

    # The squad's own value is the ceiling: an appreciating squad is legal, and
    # the opening budget stopped applying the moment it was assembled.
    budget = TenthsOfMillion(sum(p.selling_price for p in picks) + bank_after)
    violations.extend(check_squad(picks, rules=rules, bank=bank_after, budget=budget))
    return violations


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
    rules: SquadRules,
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

        # Both squads are scored through the identical path, with the identical
        # minutes, so autosubs and vice-captaincy help the policy and the
        # baseline on the same terms.
        played = actual_minutes(player_stats, season=season, gameweek=gameweek)

        recommendation, predictions = recommend_fn(acting, gameweek, season)

        # Score with the squad as it stands *for* this gameweek — the transfer
        # is made before the deadline, so the incoming player plays this week.
        acting_after = apply_transfer(
            acting, recommendation, prices=prices, positions=positions, teams=teams, rules=rules
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
                    acting_after,
                    outcomes,
                    predictions=predictions,
                    rules=rules,
                    minutes=played,
                ),
                hold_points=score_squad(
                    holding, outcomes, predictions=predictions, rules=rules, minutes=played
                ),
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
        holding = holding.model_copy(
            update={"free_transfers": accrue(holding.free_transfers, rules=rules)}
        )

    return result
