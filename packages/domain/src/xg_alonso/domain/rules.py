"""Squad and transfer rules, read from a pinned API snapshot.

Same principle as :mod:`xg_alonso.domain.scoring`: FPL publishes these in
``game_config.rules`` and ``element_types``, so we read them instead of typing
them. It matters more than it looks — FPL changed free-transfer accumulation
recently, and a hardcoded "you may bank at most 2" would still look plausible
while quietly producing illegal recommendations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.identifiers import TenthsOfMillion
from xg_alonso.contracts.prediction import Position

__all__ = ["PositionRule", "SquadRules"]


class PositionRule(BaseModel):
    """Squad and starting-XI quotas for one position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: Position
    squad_select: int = Field(gt=0, description="Exact number required in the 15")
    min_play: int = Field(ge=0, description="Minimum in the starting XI")
    max_play: int = Field(ge=0, description="Maximum in the starting XI")

    @model_validator(mode="after")
    def _check_bounds(self) -> PositionRule:
        if self.min_play > self.max_play:
            raise ValueError(f"{self.position}: min_play exceeds max_play")
        if self.max_play > self.squad_select:
            raise ValueError(
                f"{self.position}: cannot start {self.max_play} from a squad of {self.squad_select}"
            )
        return self


class SquadRules(BaseModel):
    """The constraint set every recommendation must satisfy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    source_sha256: str = Field(min_length=64, max_length=64)

    squad_size: int = Field(gt=0)
    starting_size: int = Field(gt=0)
    max_per_club: int = Field(gt=0)
    total_budget: TenthsOfMillion
    sell_on_fee: float = Field(
        ge=0.0, le=1.0, description="Fraction of profit lost on sale; 0.5 means half"
    )
    sell_at_purchase_price: bool = Field(
        description="When true, profit is not realised at all on sale"
    )
    max_extra_free_transfers: int = Field(
        ge=0, description="Free transfers carried beyond the standard one"
    )
    transfers_cap: int = Field(gt=0, description="Maximum transfers in a single gameweek")
    hit_cost_per_transfer: int = Field(
        default=4, gt=0, description="VERIFY: points deducted per transfer beyond the free ones"
    )

    positions: tuple[PositionRule, ...]

    @model_validator(mode="after")
    def _check_internally_consistent(self) -> SquadRules:
        total = sum(p.squad_select for p in self.positions)
        if total != self.squad_size:
            raise ValueError(
                f"position quotas sum to {total} but the squad holds {self.squad_size}"
            )
        if sum(p.min_play for p in self.positions) > self.starting_size:
            raise ValueError("minimum starters exceed the starting-XI size")
        if sum(p.max_play for p in self.positions) < self.starting_size:
            raise ValueError("maximum starters cannot fill the starting XI")
        return self

    @property
    def max_free_transfers(self) -> int:
        """Free transfers a manager may hold: the standard one plus any carried."""
        return 1 + self.max_extra_free_transfers

    def rule_for(self, position: Position) -> PositionRule:
        for rule in self.positions:
            if rule.position is position:
                return rule
        raise KeyError(f"no rule for position {position}")

    @classmethod
    def from_bootstrap(
        cls, payload: dict[str, Any], *, version: str, source_sha256: str
    ) -> SquadRules:
        """Parse squad and transfer rules from a ``bootstrap-static`` payload."""
        game_config = payload.get("game_config")
        if not isinstance(game_config, dict) or "rules" not in game_config:
            raise KeyError(
                "bootstrap-static payload has no game_config.rules block; "
                "constraints must never be assumed"
            )
        rules: dict[str, Any] = game_config["rules"]

        element_types = payload.get("element_types")
        if not isinstance(element_types, list) or not element_types:
            raise KeyError("bootstrap-static payload has no element_types")

        positions: list[PositionRule] = []
        for entry in element_types:
            short = entry.get("singular_name_short")
            if short not in Position.__members__:
                continue
            positions.append(
                PositionRule(
                    position=Position(short),
                    squad_select=int(entry["squad_select"]),
                    min_play=int(entry["squad_min_play"]),
                    max_play=int(entry["squad_max_play"]),
                )
            )

        return cls(
            version=version,
            source_sha256=source_sha256,
            squad_size=int(rules["squad_squadsize"]),
            starting_size=int(rules["squad_squadplay"]),
            max_per_club=int(rules["squad_team_limit"]),
            total_budget=TenthsOfMillion(int(rules["squad_total_spend"])),
            sell_on_fee=float(rules["transfers_sell_on_fee"]),
            sell_at_purchase_price=bool(rules["element_sell_at_purchase_price"]),
            max_extra_free_transfers=int(rules["max_extra_free_transfers"]),
            transfers_cap=int(rules["transfers_cap"]),
            positions=tuple(sorted(positions, key=lambda p: list(Position).index(p.position))),
        )
