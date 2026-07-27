"""Transfer policies, for measuring whether the model has any skill.

Beating "never transfer" is a low bar. A squad that starts weak improves under
almost any policy, so a backtest that only compares against holding cannot tell
model skill apart from regression to competence.

These policies all choose from the **same set of legal transfers**, computed by
the same code. The only thing that varies is which candidate gets picked, so any
difference in outcome is selection skill and nothing else — not a different
budget, not a different legality check, not luck about what was affordable.

If the model cannot beat :func:`random_policy`, it is not a model. If it cannot
beat :func:`highest_form_policy`, it is not worth the machinery.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from datetime import datetime

from xg_alonso.contracts.identifiers import EntryId, GameweekId, PlayerCode, TenthsOfMillion
from xg_alonso.contracts.prediction import PlayerPrediction
from xg_alonso.contracts.reason_codes import Reason, ReasonCode, ReasonPolarity
from xg_alonso.contracts.recommendation import (
    BaselineComparison,
    TransferMove,
    TransferPackage,
    TransferRecommendation,
)
from xg_alonso.contracts.squad import SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.optimization.transfer import (
    HOLD_BASELINE,
    Candidate,
    TransferCandidate,
    hold_expected_points,
    rank_single_transfers,
)

__all__ = [
    "POLICIES",
    "PolicyName",
    "highest_form_policy",
    "hold_policy",
    "model_policy",
    "most_expensive_policy",
    "random_policy",
    "run_policy",
]

PolicyName = str

#: A policy picks one legal transfer, or ``None`` to hold.
Selector = Callable[[Sequence[TransferCandidate], random.Random], TransferCandidate | None]

#: Minimum modelled gain before the model policy will spend a transfer. Matches
#: the optimizer's own threshold so the two agree about when acting is worth it.
_MIN_NET_GAIN = 0.15


def model_policy(
    candidates: Sequence[TransferCandidate], rng: random.Random
) -> TransferCandidate | None:
    """The system under test: highest modelled net gain, if it clears the bar."""
    del rng
    if not candidates:
        return None
    best = candidates[0]  # already sorted by net gain
    return best if best.net_gain >= _MIN_NET_GAIN else None


def random_policy(
    candidates: Sequence[TransferCandidate], rng: random.Random
) -> TransferCandidate | None:
    """A uniformly random legal transfer — the floor any model must clear.

    Seeded, so a backtest remains reproducible. An unseeded control would make
    the comparison unrepeatable, which defeats the purpose of having one.
    """
    return rng.choice(list(candidates)) if candidates else None


def highest_form_policy(
    candidates: Sequence[TransferCandidate], rng: random.Random
) -> TransferCandidate | None:
    """Buy whoever is projected highest, ignoring who is sold and at what cost.

    This is the heuristic most managers actually use, and it is a genuinely hard
    baseline: it captures most of the signal in "good players score points"
    while ignoring value, risk and opportunity cost — which is precisely what
    the optimizer is supposed to add.
    """
    del rng
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.incoming.prediction.expected_points)


def most_expensive_policy(
    candidates: Sequence[TransferCandidate], rng: random.Random
) -> TransferCandidate | None:
    """Buy the most expensive affordable player. A pure price heuristic.

    Included because price encodes crowd wisdom about quality, so it is a
    surprisingly strong control — and a model that merely relearns price has
    added nothing.
    """
    del rng
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.incoming.price)


def hold_policy(
    candidates: Sequence[TransferCandidate], rng: random.Random
) -> TransferCandidate | None:
    """Never transfer. The reference baseline."""
    del candidates, rng
    return None


POLICIES: dict[PolicyName, Selector] = {
    "model": model_policy,
    "highest_form": highest_form_policy,
    "most_expensive": most_expensive_policy,
    "random": random_policy,
    "hold": hold_policy,
}


def run_policy(
    squad: SquadState,
    *,
    selector: Selector,
    candidates: Sequence[Candidate],
    predictions: dict[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    entry_id: EntryId,
    gameweek: GameweekId,
    generated_at: datetime,
    run_id: str,
    rng: random.Random,
    policy_name: PolicyName = "model",
) -> TransferRecommendation:
    """Apply one policy and wrap its choice as a recommendation.

    Legality is established once, by :func:`rank_single_transfers`, before any
    policy sees the options. Every policy therefore chooses from an identical
    legal set — the comparison isolates selection and nothing else.
    """
    baseline = hold_expected_points(squad, predictions, rules)
    legal = rank_single_transfers(
        squad, candidates=candidates, predictions=predictions, rules=rules
    )
    chosen = selector(legal, rng)

    if chosen is None:
        return TransferRecommendation(
            entry_id=entry_id,
            gameweek=gameweek,
            package=TransferPackage(
                moves=(),
                transfers_used=0,
                free_transfers_available=squad.free_transfers,
                hit_cost=0,
                bank_before=squad.bank,
                bank_after=squad.bank,
            ),
            comparison=BaselineComparison(
                baseline_name=HOLD_BASELINE,
                baseline_expected_points=baseline,
                candidate_expected_points=baseline,
                horizon_gameweeks=1,
            ),
            reasons=(),
            expected_points_gain=0.0,
            risk_score=0.0,
            generated_at=generated_at,
            run_id=run_id,
            optimizer_config_hash=policy_name,
        )

    move = TransferMove(
        player_out=chosen.out_pick.player_code,
        player_in=chosen.incoming.player_code,
        selling_price=chosen.out_pick.selling_price,
        purchase_price=chosen.incoming.price,
    )
    package = TransferPackage(
        moves=(move,),
        transfers_used=1,
        free_transfers_available=squad.free_transfers,
        hit_cost=chosen.hit_cost,
        bank_before=squad.bank,
        bank_after=TenthsOfMillion(squad.bank - move.net_cost),
    )

    # Every recommendation must carry grounded evidence, including the controls —
    # otherwise a policy could dodge the contract that keeps prose honest.
    minutes = chosen.incoming.prediction.components.minutes
    reason = Reason(
        code=ReasonCode.EXPECTED_MINUTES_SECURE,
        polarity=ReasonPolarity.SUPPORTS_IN,
        subject=chosen.incoming.player_code,
        evidence={
            "p_start": minutes.p_start,
            "expected_minutes": minutes.expected_minutes,
        },
        weight=max(0.0, chosen.gross_gain),
    )

    return TransferRecommendation(
        entry_id=entry_id,
        gameweek=gameweek,
        package=package,
        comparison=BaselineComparison(
            baseline_name=HOLD_BASELINE,
            baseline_expected_points=baseline,
            candidate_expected_points=baseline + chosen.gross_gain - chosen.hit_cost,
            horizon_gameweeks=1,
        ),
        reasons=(reason,),
        expected_points_gain=round(chosen.gross_gain - chosen.hit_cost, 6),
        risk_score=round(chosen.risk_penalty, 6),
        generated_at=generated_at,
        run_id=run_id,
        optimizer_config_hash=policy_name,
    )
