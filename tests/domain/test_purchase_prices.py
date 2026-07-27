"""Purchase-price reconstruction tests.

Includes a live contract test, marked ``network`` and skipped by default, that
will confirm the endpoint's real field names once the 2026/27 season produces
transfers. Until then the shape is documented-but-unverified, and this file is
where that gap is tracked rather than forgotten.
"""

from __future__ import annotations

from typing import Any

import pytest

from xg_alonso.contracts.identifiers import PlayerElementId, TenthsOfMillion
from xg_alonso.domain.purchase_prices import (
    parse_transfer_log,
    reconstruct_purchase_prices,
)


def _transfer(event: int, in_id: int, in_cost: int, out_id: int, out_cost: int) -> dict[str, Any]:
    return {
        "element_in": in_id,
        "element_in_cost": in_cost,
        "element_out": out_id,
        "element_out_cost": out_cost,
        "event": event,
        "entry": 1,
        "time": f"2026-09-{event:02d}T10:00:00Z",
    }


class TestParsing:
    def test_documented_fields_parse(self) -> None:
        records = parse_transfer_log([_transfer(2, 10, 75, 20, 60)])
        assert records[0].element_in == 10
        assert records[0].element_in_cost == 75

    def test_unknown_keys_are_preserved_not_dropped(self) -> None:
        """A schema change should surface in a diff, not vanish silently."""
        row = _transfer(2, 10, 75, 20, 60)
        row["some_new_field"] = "surprise"
        records = parse_transfer_log([row])
        assert records[0].extra == {"some_new_field": "surprise"}

    def test_missing_field_fails_loudly(self) -> None:
        """A plausible-but-wrong budget is worse than no answer."""
        row = _transfer(2, 10, 75, 20, 60)
        del row["element_in_cost"]
        with pytest.raises(KeyError, match="must not be guessed"):
            parse_transfer_log([row])

    def test_empty_log_is_valid(self) -> None:
        """Exactly what the live endpoint returns before a season starts."""
        assert parse_transfer_log([]) == []


class TestReconstruction:
    def test_transferred_in_player_uses_the_paid_price(self) -> None:
        result = reconstruct_purchase_prices(
            current_squad=[PlayerElementId(10)],
            transfers=parse_transfer_log([_transfer(2, 10, 75, 20, 60)]),
        )
        assert result.purchase_prices[PlayerElementId(10)] == 75
        assert result.complete

    def test_never_transferred_player_uses_the_opening_price(self) -> None:
        result = reconstruct_purchase_prices(
            current_squad=[PlayerElementId(5)],
            transfers=[],
            opening_prices={PlayerElementId(5): TenthsOfMillion(50)},
        )
        assert result.purchase_prices[PlayerElementId(5)] == 50

    def test_rebought_player_uses_the_most_recent_purchase(self) -> None:
        """Sold at one price and bought back at another; the later one counts."""
        log = parse_transfer_log(
            [
                _transfer(2, 10, 75, 20, 60),  # buy 10 at 7.5
                _transfer(5, 20, 61, 10, 78),  # sell 10, buy 20 back
                _transfer(9, 10, 82, 30, 55),  # buy 10 again at 8.2
            ]
        )
        result = reconstruct_purchase_prices(current_squad=[PlayerElementId(10)], transfers=log)
        assert result.purchase_prices[PlayerElementId(10)] == 82

    def test_out_of_order_log_is_sorted(self) -> None:
        """The endpoint's ordering is not guaranteed, so it must not matter."""
        rows = [_transfer(9, 10, 82, 30, 55), _transfer(2, 10, 75, 20, 60)]
        forward = reconstruct_purchase_prices(
            current_squad=[PlayerElementId(10)], transfers=parse_transfer_log(rows)
        )
        backward = reconstruct_purchase_prices(
            current_squad=[PlayerElementId(10)],
            transfers=parse_transfer_log(list(reversed(rows))),
        )
        assert forward.purchase_prices == backward.purchase_prices == {10: 82}

    def test_unknown_player_is_reported_not_defaulted(self) -> None:
        """A silently assumed price is indistinguishable from a known one."""
        result = reconstruct_purchase_prices(
            current_squad=[PlayerElementId(99)], transfers=[], opening_prices={}
        )
        assert result.unresolved == (PlayerElementId(99),)
        assert not result.complete
        assert PlayerElementId(99) not in result.purchase_prices

    def test_sold_player_is_not_reported(self) -> None:
        result = reconstruct_purchase_prices(
            current_squad=[PlayerElementId(20)],
            transfers=parse_transfer_log([_transfer(2, 20, 60, 10, 75)]),
        )
        assert PlayerElementId(10) not in result.purchase_prices


@pytest.mark.network
class TestLiveContract:
    """Confirms the real endpoint shape. Skipped until transfers exist.

    This is the outstanding verification for decision D3. Run it with
    ``pytest -m network`` once the season is under way; if it fails, the parser
    above is wrong and every reconstructed budget is suspect.
    """

    def test_transfers_endpoint_matches_the_documented_shape(self) -> None:
        import httpx

        response = httpx.get(
            "https://fantasy.premierleague.com/api/entry/1/transfers/", timeout=20.0
        )
        response.raise_for_status()
        payload = response.json()

        if not payload:
            pytest.skip(
                "entry/1/transfers/ is empty — the season has not produced transfers "
                "yet, so the field names remain unverified (see D3)."
            )

        # The assertion that actually matters: our parser accepts real data.
        records = parse_transfer_log(payload)
        assert records
        assert all(r.element_in_cost > 0 for r in records)
