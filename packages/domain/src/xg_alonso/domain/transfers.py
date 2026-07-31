"""Free transfers: what a gameweek costs, and what is left for the next one.

**Why this is a module and not four lines in the optimizer.** The rule was
spread across five places and none of them agreed with the pinned snapshot:
``SquadState`` and ``TransferPackage`` each capped the allowance at a literal
``5``, ``backtest.py`` accrued with ``min(5, ft + 1)`` twice, and
``TransferPackage`` charged a literal ``4`` per paid transfer — while
``SquadRules.max_free_transfers`` and ``SquadRules.hit_cost_per_transfer``,
both derived from ``game_config``, were never consulted. FPL changed
free-transfer accumulation recently. A hardcoded cap would have kept looking
plausible while quietly producing illegal recommendations, which is the exact
failure the no-literals rule exists to prevent.

**The order of operations matters and is easy to get backwards.** Spend first,
charge for the excess, *then* accrue. The ``+1`` arrives even when the
allowance was spent to zero, so a manager who transfers every week sits at a
steady one rather than sliding into a permanent hit. Accruing before spending
would give a free transfer that the same gameweek then consumes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.constraints import SquadViolation
from xg_alonso.contracts.recommendation import TransferPackage
from xg_alonso.domain.rules import SquadRules

__all__ = ["TransferLedger", "accrue", "check_transfer_package", "settle_gameweek"]


class TransferLedger(BaseModel):
    """What one gameweek's transfer decision cost, and what it leaves behind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    free_transfers_before: int = Field(ge=0)
    transfers_made: int = Field(ge=0)
    free_transfers_used: int = Field(ge=0)
    paid_transfers: int = Field(ge=0)
    hit_cost: int = Field(ge=0)
    free_transfers_after: int = Field(ge=0)

    @model_validator(mode="after")
    def _spending_adds_up(self) -> TransferLedger:
        if self.free_transfers_used + self.paid_transfers != self.transfers_made:
            raise ValueError(
                f"{self.free_transfers_used} free plus {self.paid_transfers} paid "
                f"does not equal {self.transfers_made} transfers made"
            )
        if self.free_transfers_used > self.free_transfers_before:
            raise ValueError(
                f"{self.free_transfers_used} free transfers used but only "
                f"{self.free_transfers_before} were available"
            )
        return self


def settle_gameweek(
    *, free_transfers: int, transfers_made: int, rules: SquadRules
) -> TransferLedger:
    """Spend the allowance, charge for the excess, then accrue for next week.

    Args:
        free_transfers: The allowance carried into this gameweek.
        transfers_made: How many transfers were actually made.
        rules: Supplies the cap, the per-transfer charge and the per-gameweek
            limit, all read from the pinned snapshot.

    Raises:
        ValueError: if more transfers were made than the game permits in one
            gameweek. Returning a ledger for an impossible week would let a
            backtest measure a policy FPL would have refused.
    """
    if free_transfers < 0 or transfers_made < 0:
        raise ValueError("free transfers and transfers made must not be negative")
    if transfers_made > rules.transfers_cap:
        raise ValueError(
            f"{transfers_made} transfers exceeds the cap of {rules.transfers_cap} "
            "for a single gameweek"
        )

    used = min(free_transfers, transfers_made)
    paid = transfers_made - used
    remaining = free_transfers - used

    return TransferLedger(
        free_transfers_before=free_transfers,
        transfers_made=transfers_made,
        free_transfers_used=used,
        paid_transfers=paid,
        hit_cost=paid * rules.hit_cost_per_transfer,
        # The +1 lands even when the allowance was spent to zero.
        free_transfers_after=min(rules.max_free_transfers, remaining + 1),
    )


def accrue(free_transfers: int, *, rules: SquadRules) -> int:
    """The allowance after a gameweek in which nothing was done."""
    return settle_gameweek(
        free_transfers=free_transfers, transfers_made=0, rules=rules
    ).free_transfers_after


def check_transfer_package(package: TransferPackage, *, rules: SquadRules) -> list[SquadViolation]:
    """Check a package's hit cost against the rules, exactly.

    ``TransferPackage`` keeps only the *shape* of the accounting — that a charge
    exists exactly when a paid transfer does, and divides evenly among them —
    because ``contracts`` may not import ``domain`` and therefore cannot read
    ``hit_cost_per_transfer``. The exact value is checked here, where the rules
    live. A package charging eight points for one paid transfer satisfies the
    contract and fails this.
    """
    violations: list[SquadViolation] = []

    paid = max(0, package.transfers_used - package.free_transfers_available)
    expected = paid * rules.hit_cost_per_transfer
    if package.hit_cost != expected:
        violations.append(
            SquadViolation(
                rule="hit_cost",
                detail=(
                    f"hit_cost {package.hit_cost} should be {expected}: {paid} paid "
                    f"transfers at {rules.hit_cost_per_transfer} points each"
                ),
            )
        )

    if package.free_transfers_available > rules.max_free_transfers:
        violations.append(
            SquadViolation(
                rule="max_free_transfers",
                detail=(
                    f"{package.free_transfers_available} free transfers exceeds the "
                    f"cap of {rules.max_free_transfers}"
                ),
            )
        )

    if package.transfers_used > rules.transfers_cap:
        violations.append(
            SquadViolation(
                rule="transfers_cap",
                detail=(
                    f"{package.transfers_used} transfers exceeds the cap of "
                    f"{rules.transfers_cap} for a single gameweek"
                ),
            )
        )

    return violations
