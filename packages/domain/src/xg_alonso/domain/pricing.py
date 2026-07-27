"""Selling price and squad-value arithmetic.

Getting this wrong corrupts every budget-constrained recommendation, and it
fails quietly: an off-by-one-tenth selling price yields transfers the game will
simply refuse. All arithmetic is integer, in tenths of a million.

**The rule.** A manager keeps only part of a player's price rise. With
``sell_on_fee = 0.5``, half the profit is retained and the result is rounded
*down* to the nearest 0.1. A price *fall* is absorbed in full — a player bought
at 7.0 and now worth 6.5 sells for 6.5, not 6.75.
"""

from __future__ import annotations

from xg_alonso.contracts.identifiers import TenthsOfMillion
from xg_alonso.domain.rules import SquadRules

__all__ = ["selling_price", "squad_value"]


def selling_price(
    *,
    purchase_price: TenthsOfMillion,
    current_price: TenthsOfMillion,
    rules: SquadRules,
) -> TenthsOfMillion:
    """What a sale realises now.

    Args:
        purchase_price: What the manager paid, in tenths of a million.
        current_price: The player's current listed price.
        rules: Supplies ``sell_on_fee`` and ``sell_at_purchase_price``.

    Returns:
        The realisable price, in tenths of a million.
    """
    if purchase_price < 0 or current_price < 0:
        raise ValueError("prices must not be negative")

    if rules.sell_at_purchase_price:
        return purchase_price

    profit = current_price - purchase_price
    if profit <= 0:
        # Losses are absorbed in full; there is nothing to share.
        return current_price

    # Integer floor division keeps the "rounded down to the nearest 0.1" rule
    # exact. Doing this in floating point is how tools end up one tenth short.
    retained = int(profit * (1.0 - rules.sell_on_fee) + 1e-9)
    return TenthsOfMillion(purchase_price + retained)


def squad_value(
    *,
    selling_prices: tuple[TenthsOfMillion, ...],
    bank: TenthsOfMillion,
) -> TenthsOfMillion:
    """Total realisable value: what every player would fetch, plus the bank.

    This is deliberately *not* the sum of current prices. Unrealised profit that
    the sell-on fee would take is not money the manager can spend, and treating
    it as though it were produces recommendations that cannot be executed.
    """
    return TenthsOfMillion(sum(selling_prices) + bank)
