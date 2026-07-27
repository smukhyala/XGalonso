"""Reconstruct purchase prices from public transfer history.

Decision D3 keeps us on public endpoints, and the public API does not publish
what a manager paid for a player — that lives behind the authenticated
``my-team/`` endpoint. But ``entry/{id}/transfers/`` *does* publish
``element_in_cost`` for every move, so purchase prices can be rebuilt exactly by
replaying the transfer log over the opening squad.

**Verification status.** As of 2026-07-27 this is implemented against the
documented field names but is **not verified against live data**, because the
2026/27 season has not started: ``entry/{id}/transfers/`` returns ``[]`` for
every entry, ``entry/{id}/event/{gw}/picks/`` returns 404 before a deadline, and
``entry/{id}/history/`` exposes only rank and points for past seasons. The
parser is therefore written to tolerate unknown keys and to report what it could
not resolve, rather than to assume its own correctness. See
:func:`reconstruct_purchase_prices` for what happens when the log is incomplete.

Getting this wrong corrupts every budget-constrained recommendation, and it
fails quietly: an off-by-one-tenth selling price produces transfers the game
simply refuses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from xg_alonso.contracts.identifiers import PlayerElementId, TenthsOfMillion

__all__ = [
    "PurchasePriceResult",
    "TransferRecord",
    "parse_transfer_log",
    "reconstruct_purchase_prices",
]


@dataclass(frozen=True)
class TransferRecord:
    """One move from ``entry/{id}/transfers/``.

    Unknown keys from the payload are preserved in ``extra`` rather than
    discarded, so a schema change shows up in a diff instead of vanishing.
    """

    event: int
    element_in: PlayerElementId
    element_in_cost: TenthsOfMillion
    element_out: PlayerElementId
    element_out_cost: TenthsOfMillion
    time: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PurchasePriceResult:
    """Reconstructed prices, plus an honest account of what could not be resolved."""

    purchase_prices: dict[PlayerElementId, TenthsOfMillion]
    unresolved: tuple[PlayerElementId, ...]
    transfers_applied: int

    @property
    def complete(self) -> bool:
        """Whether every player's purchase price was established from evidence.

        When this is ``False`` the caller is working with assumed prices and
        should say so, rather than presenting an approximate budget as exact.
        """
        return not self.unresolved


_REQUIRED_KEYS = ("element_in", "element_in_cost", "element_out", "element_out_cost", "event")


def parse_transfer_log(payload: Sequence[dict[str, Any]]) -> list[TransferRecord]:
    """Parse the transfers payload defensively.

    Raises:
        KeyError: if a documented field is missing. Failing loudly is correct
            here — silently skipping malformed rows would produce a plausible
            but wrong budget, which is worse than no answer.
    """
    records: list[TransferRecord] = []
    for index, row in enumerate(payload):
        missing = [k for k in _REQUIRED_KEYS if k not in row]
        if missing:
            raise KeyError(
                f"transfer row {index} is missing {missing}. The endpoint shape has "
                "changed, or this is not an entry/{id}/transfers/ payload. Purchase "
                "prices must not be guessed."
            )
        known = {*_REQUIRED_KEYS, "time", "entry"}
        records.append(
            TransferRecord(
                event=int(row["event"]),
                element_in=PlayerElementId(int(row["element_in"])),
                element_in_cost=TenthsOfMillion(int(row["element_in_cost"])),
                element_out=PlayerElementId(int(row["element_out"])),
                element_out_cost=TenthsOfMillion(int(row["element_out_cost"])),
                time=row.get("time"),
                extra={k: v for k, v in row.items() if k not in known},
            )
        )
    return records


def reconstruct_purchase_prices(
    *,
    current_squad: Sequence[PlayerElementId],
    transfers: Sequence[TransferRecord],
    opening_prices: dict[PlayerElementId, TenthsOfMillion] | None = None,
) -> PurchasePriceResult:
    """Replay the transfer log to recover what the manager paid for each player.

    A player's purchase price is the ``element_in_cost`` of the most recent
    transfer that brought them in. A player never transferred in was part of the
    opening squad, so their purchase price is their opening price.

    Args:
        current_squad: Element ids the manager holds now.
        transfers: The transfer log, in any order — it is sorted here.
        opening_prices: Prices at the start of the season, for players who were
            never transferred in. Omitted entries end up in ``unresolved``.

    Returns:
        The prices that could be established, and the players that could not be.
        Players are reported rather than defaulted, because a silently assumed
        price is indistinguishable from a known one at the point of use.
    """
    opening_prices = opening_prices or {}
    held = set(current_squad)

    # Replay chronologically so a player transferred in, out, and in again ends
    # up with the price from the *last* purchase.
    ordered = sorted(transfers, key=lambda t: (t.event, t.time or ""))

    paid: dict[PlayerElementId, TenthsOfMillion] = {}
    for transfer in ordered:
        paid[transfer.element_in] = transfer.element_in_cost
        # Selling clears the record; buying back later re-establishes it.
        paid.pop(transfer.element_out, None)

    resolved: dict[PlayerElementId, TenthsOfMillion] = {}
    unresolved: list[PlayerElementId] = []
    for element_id in current_squad:
        if element_id in paid:
            resolved[element_id] = paid[element_id]
        elif element_id in opening_prices:
            resolved[element_id] = opening_prices[element_id]
        else:
            unresolved.append(element_id)

    del held
    return PurchasePriceResult(
        purchase_prices=resolved,
        unresolved=tuple(unresolved),
        transfers_applied=len(ordered),
    )
