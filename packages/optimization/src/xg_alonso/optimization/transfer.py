"""Single-transfer optimization.

Per `CLAUDE.md`, the optimizer is the product and predictions merely feed it. So
this module owns the decision, and its output is judged against one question:
**does it beat holding?**

**Why exhaustive search rather than MILP.** A single transfer is one of roughly
15 outgoing players times a few hundred affordable replacements — tens of
thousands of candidates, each evaluated with integer arithmetic and a few
comparisons. That runs in well under a second. A solver would add a dependency,
a modelling layer and a class of "infeasible" failures that are hard to explain
to a user, in exchange for nothing at this size. MILP earns its place when
multi-transfer packages arrive and the combinatorics stop being trivial.

Every candidate is checked against the real constraints before it is scored, so
an illegal transfer can never be ranked, let alone recommended. A recommendation
the game would reject costs the user a transfer to discover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.reason_codes import Reason, ReasonCode, ReasonPolarity
from xg_alonso.contracts.recommendation import (
    BaselineComparison,
    TransferMove,
    TransferPackage,
    TransferRecommendation,
)
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.optimization.lineup import starting_xi_points

__all__ = [
    "HOLD_BASELINE",
    "Candidate",
    "TransferCandidate",
    "best_single_transfer",
    "hold_expected_points",
    "rank_single_transfers",
]

HOLD_BASELINE: Final[str] = "hold"

#: How much of a point of predicted gain must survive the uncertainty penalty
#: before a transfer is worth recommending. Set above zero because a marginal
#: gain inside the noise is not a reason to spend a transfer.
_MIN_NET_GAIN: Final[float] = 0.15

#: Weight on prediction uncertainty in the objective. A transfer that gains 1.0
#: expected points with wide variance is worse than one gaining 0.9 with narrow.
_RISK_WEIGHT: Final[float] = 0.25


@dataclass(frozen=True)
class Candidate:
    """A player who could be bought, with everything needed to price the move."""

    player_code: PlayerCode
    position: Position
    team_id: TeamId
    price: TenthsOfMillion
    prediction: PlayerPrediction


@dataclass(frozen=True)
class TransferCandidate:
    """One evaluated out-for-in move."""

    out_pick: SquadPick
    incoming: Candidate
    out_prediction: PlayerPrediction
    gross_gain: float
    hit_cost: int
    risk_penalty: float

    @property
    def net_gain(self) -> float:
        """Expected points gained after the hit and the uncertainty penalty."""
        return self.gross_gain - self.hit_cost - self.risk_penalty

    @property
    def bank_after(self) -> TenthsOfMillion:
        return TenthsOfMillion(self.out_pick.selling_price - self.incoming.price)


def hold_expected_points(
    squad: SquadState,
    predictions: dict[PlayerCode, PlayerPrediction],
    rules: SquadRules,
) -> float:
    """Expected points from doing nothing — the baseline everything is measured against.

    Scored as the **best legal XI with the captain doubled**, not as the squad's
    stored slot order. Two earlier versions of this got it wrong in ways that
    compounded:

    - Summing the stored starters credited whatever eleven the payload happened
      to name, so a transfer that improved a bench player looked like a full
      gain even though it changed nothing that scores.
    - No captain was ever chosen, so the largest single term in FPL scoring —
      a doubled return — was absent from every measurement.
    """
    return starting_xi_points(squad.picks, predictions, rules)


def _club_counts(picks: Sequence[SquadPick]) -> dict[TeamId, int]:
    counts: dict[TeamId, int] = {}
    for pick in picks:
        counts[pick.team_id] = counts.get(pick.team_id, 0) + 1
    return counts


def rank_single_transfers(
    squad: SquadState,
    *,
    candidates: Sequence[Candidate],
    predictions: dict[PlayerCode, PlayerPrediction],
    rules: SquadRules,
) -> list[TransferCandidate]:
    """Every legal single transfer, best net gain first.

    Legality is enforced before scoring, so an illegal move never enters the
    ranking. Checks applied: position match, affordability against the selling
    price plus bank, the three-per-club limit, and no buying a player already
    owned.
    """
    owned = {pick.player_code for pick in squad.picks}
    counts = _club_counts(squad.picks)
    hit_cost = 0 if squad.free_transfers >= 1 else rules.hit_cost_per_transfer

    # The bar every candidate must clear: what this squad scores untouched.
    hold_points = starting_xi_points(squad.picks, predictions, rules)

    evaluated: list[TransferCandidate] = []

    for pick in squad.picks:
        out_prediction = predictions.get(pick.player_code)
        if out_prediction is None:
            continue

        budget = TenthsOfMillion(pick.selling_price + squad.bank)

        for candidate in candidates:
            if candidate.player_code in owned:
                continue
            if candidate.position is not pick.position:
                continue
            if candidate.price > budget:
                continue

            # Selling one player from a club frees a slot in that club.
            club_after = counts.get(candidate.team_id, 0) + (
                -1 if candidate.team_id == pick.team_id else 0
            )
            if club_after >= rules.max_per_club:
                continue

            # Score the squad this move would actually produce, then re-pick
            # the XI. A straight difference between the two players is wrong
            # whenever either sits on the bench: replacing a benched player with
            # a slightly better benched player gains nothing at all.
            after = [p for p in squad.picks if p.player_code != pick.player_code]
            after.append(
                pick.model_copy(
                    update={
                        "player_code": candidate.player_code,
                        "position": candidate.position,
                        "team_id": candidate.team_id,
                        "purchase_price": candidate.price,
                        "current_price": candidate.price,
                        "selling_price": candidate.price,
                    }
                )
            )
            gross = starting_xi_points(after, predictions, rules) - hold_points

            # Combine the two uncertainties; a swap inherits both.
            risk = _RISK_WEIGHT * (
                candidate.prediction.expected_points_sd + out_prediction.expected_points_sd
            )

            evaluated.append(
                TransferCandidate(
                    out_pick=pick,
                    incoming=candidate,
                    out_prediction=out_prediction,
                    gross_gain=gross,
                    hit_cost=hit_cost,
                    risk_penalty=risk,
                )
            )

    # Sort by net gain, breaking ties on player code so the ranking is stable
    # across runs rather than dependent on candidate iteration order.
    evaluated.sort(key=lambda c: (-c.net_gain, c.incoming.player_code, c.out_pick.player_code))
    return evaluated


def _build_reasons(candidate: TransferCandidate) -> tuple[Reason, ...]:
    """Ground the recommendation in the evidence that actually drove it.

    Each reason is built from values already used in the decision, so the
    explanation cannot drift from the arithmetic it explains.
    """
    reasons: list[Reason] = []
    incoming = candidate.incoming.prediction
    outgoing = candidate.out_prediction

    in_minutes = incoming.components.minutes
    out_minutes = outgoing.components.minutes

    if in_minutes.p_start >= 0.6:
        reasons.append(
            Reason(
                code=ReasonCode.EXPECTED_MINUTES_SECURE,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=candidate.incoming.player_code,
                evidence={
                    "p_start": in_minutes.p_start,
                    "expected_minutes": in_minutes.expected_minutes,
                },
                weight=abs(candidate.gross_gain) * 0.4,
            )
        )

    if out_minutes.p_start < in_minutes.p_start:
        reasons.append(
            Reason(
                code=ReasonCode.EXPECTED_MINUTES_DECLINE,
                polarity=ReasonPolarity.SUPPORTS_OUT,
                subject=candidate.out_pick.player_code,
                evidence={
                    "p_start": out_minutes.p_start,
                    "expected_minutes": out_minutes.expected_minutes,
                },
                weight=abs(candidate.gross_gain) * 0.3,
            )
        )

    in_xgi = incoming.components.goals + incoming.components.assists
    out_xgi = outgoing.components.goals + outgoing.components.assists
    if in_xgi > out_xgi:
        reasons.append(
            Reason(
                code=ReasonCode.UNDERLYING_STATS_IMPROVING,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=candidate.incoming.player_code,
                evidence={"recent_xgi": in_xgi, "baseline_xgi": out_xgi},
                weight=abs(candidate.gross_gain) * 0.5,
            )
        )

    return tuple(reasons)


def best_single_transfer(
    squad: SquadState,
    *,
    candidates: Sequence[Candidate],
    predictions: dict[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    entry_id: EntryId,
    gameweek: GameweekId,
    generated_at: datetime,
    run_id: str,
    optimizer_config_hash: str,
    horizon_gameweeks: int = 1,
) -> TransferRecommendation:
    """The best single transfer, or an explicit hold when none clears the bar.

    Returning a hold is a real answer, not a failure. Most gameweeks the correct
    move is to keep the transfer, and a tool that always finds something to do
    is a tool that costs its user points.
    """
    baseline = hold_expected_points(squad, predictions, rules)
    ranked = rank_single_transfers(
        squad, candidates=candidates, predictions=predictions, rules=rules
    )

    best = ranked[0] if ranked else None

    if best is None or best.net_gain < _MIN_NET_GAIN:
        package = TransferPackage(
            moves=(),
            transfers_used=0,
            free_transfers_available=squad.free_transfers,
            hit_cost=0,
            bank_before=squad.bank,
            bank_after=squad.bank,
        )
        return TransferRecommendation(
            entry_id=entry_id,
            gameweek=gameweek,
            package=package,
            comparison=BaselineComparison(
                baseline_name=HOLD_BASELINE,
                baseline_expected_points=baseline,
                candidate_expected_points=baseline,
                horizon_gameweeks=horizon_gameweeks,
            ),
            reasons=(),
            expected_points_gain=0.0,
            risk_score=0.0,
            generated_at=generated_at,
            run_id=run_id,
            optimizer_config_hash=optimizer_config_hash,
        )

    move = TransferMove(
        player_out=best.out_pick.player_code,
        player_in=best.incoming.player_code,
        selling_price=best.out_pick.selling_price,
        purchase_price=best.incoming.price,
    )
    package = TransferPackage(
        moves=(move,),
        transfers_used=1,
        free_transfers_available=squad.free_transfers,
        hit_cost=best.hit_cost,
        bank_before=squad.bank,
        bank_after=TenthsOfMillion(squad.bank - move.net_cost),
    )

    return TransferRecommendation(
        entry_id=entry_id,
        gameweek=gameweek,
        package=package,
        comparison=BaselineComparison(
            baseline_name=HOLD_BASELINE,
            baseline_expected_points=baseline,
            candidate_expected_points=baseline + best.gross_gain - best.hit_cost,
            horizon_gameweeks=horizon_gameweeks,
        ),
        reasons=_build_reasons(best),
        expected_points_gain=round(best.gross_gain - best.hit_cost, 6),
        expected_value_gain=TenthsOfMillion(0),
        risk_score=round(best.risk_penalty, 6),
        generated_at=generated_at,
        run_id=run_id,
        optimizer_config_hash=optimizer_config_hash,
    )
