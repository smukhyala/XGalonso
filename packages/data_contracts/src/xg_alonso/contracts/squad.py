"""Squad state — what a manager owns, at what prices, with what flexibility.

Purchase price is not published by the public API. Decision D3 reconstructs it
from ``entry/{id}/transfers/``, which does expose ``element_in_cost`` per move.
Selling price then follows FPL's rule: profit is halved and rounded *down* to
the nearest 0.1, which is why all money here is integer tenths of a million.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import Position

__all__ = [
    "ChipState",
    "ChipStatus",
    "SquadPick",
    "SquadState",
]


class ChipStatus(StrEnum):
    """Availability of a chip. Modelled per D5 so chips are not blocked later."""

    AVAILABLE = "available"
    PLAYED = "played"
    UNAVAILABLE = "unavailable"


class ChipState(BaseModel):
    """Which chips remain. No chip *logic* ships in the MVP — only this state.

    Note the wildcard windows: two wildcards exist, GW2-19 and GW20-38, so a
    wildcard is not available in GW1 at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    wildcard: ChipStatus = ChipStatus.AVAILABLE
    free_hit: ChipStatus = ChipStatus.AVAILABLE
    bench_boost: ChipStatus = ChipStatus.AVAILABLE
    triple_captain: ChipStatus = ChipStatus.AVAILABLE


class SquadPick(BaseModel):
    """One owned player, with the prices that determine what a sale realises."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_code: PlayerCode
    position: Position
    team_id: TeamId
    purchase_price: TenthsOfMillion = Field(description="What the manager paid")
    current_price: TenthsOfMillion = Field(description="Today's listed price")
    selling_price: TenthsOfMillion = Field(
        description="What a sale realises now: purchase price plus half the profit, rounded down"
    )
    squad_slot: int = Field(ge=1, le=15, description="1-11 start, 12-15 bench order")
    is_captain: bool = False
    is_vice_captain: bool = False

    @model_validator(mode="after")
    def _selling_price_within_bounds(self) -> SquadPick:
        low, high = (
            min(self.purchase_price, self.current_price),
            max(self.purchase_price, self.current_price),
        )
        if not low <= self.selling_price <= high:
            raise ValueError(
                f"selling_price {self.selling_price} must lie between purchase "
                f"{self.purchase_price} and current {self.current_price}: a manager "
                "never realises more than the current price nor less than they paid"
            )
        return self

    @property
    def is_starter(self) -> bool:
        return self.squad_slot <= 11


class SquadState(BaseModel):
    """A complete squad at a point in time — the optimizer's starting position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: EntryId
    gameweek: GameweekId
    picks: tuple[SquadPick, ...]
    bank: TenthsOfMillion = Field(ge=0, description="Uncommitted money")
    free_transfers: int = Field(
        ge=0,
        description=(
            "Allowance carried into this gameweek. The cap is a game rule and "
            "lives in SquadRules.max_free_transfers, which is derived from the "
            "pinned snapshot; a literal here contradicted it."
        ),
    )
    chips: ChipState = Field(default_factory=ChipState)

    @model_validator(mode="after")
    def _check_squad_shape(self) -> SquadState:
        if len(self.picks) != 15:
            raise ValueError(f"a squad holds exactly 15 players, got {len(self.picks)}")
        codes = [p.player_code for p in self.picks]
        if len(set(codes)) != 15:
            raise ValueError("a squad may not contain the same player twice")
        slots = sorted(p.squad_slot for p in self.picks)
        if slots != list(range(1, 16)):
            raise ValueError("squad slots must be exactly 1 through 15")
        if sum(p.is_captain for p in self.picks) > 1:
            raise ValueError("at most one captain")
        if sum(p.is_vice_captain for p in self.picks) > 1:
            raise ValueError("at most one vice-captain")
        return self

    @property
    def squad_value(self) -> TenthsOfMillion:
        """Total realisable value: what every player would fetch, plus the bank."""
        return TenthsOfMillion(sum(p.selling_price for p in self.picks) + self.bank)

    @property
    def starters(self) -> tuple[SquadPick, ...]:
        return tuple(sorted((p for p in self.picks if p.is_starter), key=lambda p: p.squad_slot))

    @property
    def bench(self) -> tuple[SquadPick, ...]:
        return tuple(
            sorted((p for p in self.picks if not p.is_starter), key=lambda p: p.squad_slot)
        )

    def by_code(self, code: PlayerCode) -> SquadPick | None:
        return next((p for p in self.picks if p.player_code == code), None)
