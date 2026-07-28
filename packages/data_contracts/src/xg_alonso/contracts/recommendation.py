"""Recommendation output — the product.

Per `CLAUDE.md`, the optimizer is the product and predictions merely feed it.
Everything here is what a user actually receives, and every field exists so the
recommendation can be audited after the fact: what was proposed, what it was
compared against, what evidence supported it, and what inputs produced it.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.identifiers import EntryId, GameweekId, PlayerCode, TenthsOfMillion
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.reason_codes import Reason

__all__ = [
    "BaselineComparison",
    "PlayerBestMove",
    "TransferBoard",
    "TransferMove",
    "TransferOption",
    "TransferPackage",
    "TransferRecommendation",
]


class TransferMove(BaseModel):
    """One player out, one player in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_out: PlayerCode
    player_in: PlayerCode
    selling_price: TenthsOfMillion = Field(description="Realised on the outgoing player")
    purchase_price: TenthsOfMillion = Field(description="Paid for the incoming player")

    @property
    def net_cost(self) -> TenthsOfMillion:
        """Money required. Negative means the move frees funds."""
        return TenthsOfMillion(self.purchase_price - self.selling_price)


class TransferPackage(BaseModel):
    """A set of moves evaluated as one decision.

    Moves are evaluated together because their feasibility is joint: two
    individually affordable transfers can be unaffordable as a pair, and two
    individually legal ones can together breach the three-per-club limit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    moves: tuple[TransferMove, ...]
    transfers_used: int = Field(ge=0)
    free_transfers_available: int = Field(ge=0, le=5)
    hit_cost: int = Field(ge=0, description="Points deducted: 4 per transfer beyond the free ones")
    bank_before: TenthsOfMillion = Field(ge=0)
    bank_after: TenthsOfMillion = Field(ge=0)

    @model_validator(mode="after")
    def _check_accounting(self) -> TransferPackage:
        if self.transfers_used != len(self.moves):
            raise ValueError(
                f"transfers_used ({self.transfers_used}) must equal the number of "
                f"moves ({len(self.moves)})"
            )
        paid = max(0, self.transfers_used - self.free_transfers_available)
        if self.hit_cost != paid * 4:
            raise ValueError(
                f"hit_cost {self.hit_cost} disagrees with {paid} paid transfers at 4 points each"
            )
        net = sum(m.net_cost for m in self.moves)
        if self.bank_after != self.bank_before - net:
            raise ValueError(
                f"bank does not reconcile: {self.bank_before} - {net} != {self.bank_after}"
            )
        return self

    @property
    def is_hold(self) -> bool:
        return len(self.moves) == 0


class BaselineComparison(BaseModel):
    """What this recommendation is measured against.

    Hold is the headline baseline. A recommendation that cannot beat doing
    nothing is not a recommendation, and reporting a raw expected-points figure
    without this comparison is how tools flatter themselves.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_name: str = Field(description="'hold' for the do-nothing baseline")
    baseline_expected_points: float
    candidate_expected_points: float
    horizon_gameweeks: int = Field(ge=1)

    @property
    def delta(self) -> float:
        """Incremental expected points over the baseline, net of any hit."""
        return self.candidate_expected_points - self.baseline_expected_points

    @property
    def beats_baseline(self) -> bool:
        return self.delta > 0


class TransferOption(BaseModel):
    """One evaluated move, carried to the surface rather than discarded.

    The optimizer has always scored every legal transfer and returned them
    ranked. Only the first survived to the product, so a user saw one option
    with no sense of what it beat — which makes a recommendation impossible to
    disagree with intelligently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    move: TransferMove
    gross_gain: float = Field(description="Expected points gained before hit and risk")
    net_gain: float = Field(description="After the hit cost and the uncertainty penalty")
    hit_cost: int = Field(ge=0)
    risk_penalty: float = Field(ge=0.0)
    bank_after: TenthsOfMillion
    reasons: tuple[Reason, ...] = ()

    @property
    def rank_key(self) -> tuple[float, int, int]:
        """Stable ordering: best net gain first, ties broken on player codes."""
        return (-self.net_gain, int(self.move.player_in), int(self.move.player_out))


class PlayerBestMove(BaseModel):
    """The best legal move for one squad member, or a grounded reason there is none.

    Every squad member appears, including those with nothing worth doing. A
    player who is simply absent from the output reads as one the system did not
    consider, and that ambiguity is what makes a recommendation feel arbitrary:
    the question "why not him?" has no answer anywhere on the screen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_out: PlayerCode
    position: Position
    legal_replacements: int = Field(
        ge=0, description="How many players could legally have taken this slot"
    )
    option: TransferOption | None = None
    reasons: tuple[Reason, ...] = Field(
        default=(),
        description="Why there is no move, when there is none. Empty when one exists.",
    )

    @model_validator(mode="after")
    def _absence_is_explained(self) -> PlayerBestMove:
        if self.option is None and not self.reasons:
            raise ValueError(
                f"player {self.player_out} has no move and no reason for it; "
                "an unexplained absence is indistinguishable from an oversight"
            )
        if self.option is not None and self.option.move.player_out != self.player_out:
            raise ValueError(
                f"move sells {self.option.move.player_out} but this entry is for "
                f"{self.player_out}"
            )
        return self


class TransferBoard(BaseModel):
    """Every move worth showing, and the accounting behind the search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top: tuple[TransferOption, ...] = Field(description="Globally best moves, ranked")
    by_player: tuple[PlayerBestMove, ...] = Field(
        description="One entry per squad member, in squad order"
    )
    candidates_considered: int = Field(ge=0, description="Buyable players evaluated")
    legal_moves: int = Field(ge=0, description="Moves that passed every constraint")

    @model_validator(mode="after")
    def _board_is_consistent(self) -> TransferBoard:
        seen = [entry.player_out for entry in self.by_player]
        if len(seen) != len(set(seen)):
            raise ValueError("by_player lists the same player twice")
        if self.legal_moves > 0 and not self.top:
            raise ValueError(
                f"{self.legal_moves} legal moves were found but none were returned; "
                "a silently truncated board reads as an empty market"
            )
        return self


class TransferRecommendation(BaseModel):
    """A complete, auditable recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: EntryId
    gameweek: GameweekId
    package: TransferPackage
    comparison: BaselineComparison
    reasons: tuple[Reason, ...]

    expected_points_gain: float = Field(description="Net of the hit cost")
    expected_value_gain: TenthsOfMillion = Field(
        default=TenthsOfMillion(0), description="Projected squad-value movement"
    )
    risk_score: float = Field(ge=0.0, description="Uncertainty in the gain, in expected points")

    generated_at: datetime
    run_id: str
    optimizer_config_hash: str = Field(description="Which objective weights produced this")

    board: TransferBoard | None = Field(
        default=None,
        description=(
            "The alternatives this recommendation was chosen from. Optional so "
            "existing callers keep working, but the API populates it: a "
            "recommendation shown without its runners-up cannot be argued with."
        ),
    )

    @model_validator(mode="after")
    def _check_grounded(self) -> TransferRecommendation:
        if not self.package.is_hold and not self.reasons:
            raise ValueError(
                "a recommendation that proposes a transfer must carry at least one "
                "grounded reason; unexplained recommendations are not shippable"
            )
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        return self

    def render_reasons(self) -> tuple[str, ...]:
        """Grounded prose for each reason, strongest first."""
        ordered = sorted(self.reasons, key=lambda r: r.weight, reverse=True)
        return tuple(r.render() for r in ordered)
