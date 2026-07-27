"""The decision layer.

Predictions estimate reality; optimization chooses actions. Keeping the two
apart is why a slightly worse regression that produces better transfers is the
better model.

Slice 1 supports single transfers, evaluated exhaustively against every legal
alternative and measured against the hold baseline. Holding is a real answer:
most gameweeks the correct move is to keep the transfer.
"""

from xg_alonso.optimization.lineup import (
    CAPTAIN_MULTIPLIER,
    XiSelection,
    best_starting_xi,
    legal_formations,
    starting_xi_points,
)
from xg_alonso.optimization.squad_builder import SquadCandidate, build_squad
from xg_alonso.optimization.transfer import (
    HOLD_BASELINE,
    Candidate,
    TransferCandidate,
    best_single_transfer,
    hold_expected_points,
    rank_single_transfers,
)

__all__ = [
    "CAPTAIN_MULTIPLIER",
    "HOLD_BASELINE",
    "Candidate",
    "SquadCandidate",
    "TransferCandidate",
    "XiSelection",
    "best_single_transfer",
    "best_starting_xi",
    "build_squad",
    "hold_expected_points",
    "legal_formations",
    "rank_single_transfers",
    "starting_xi_points",
]
