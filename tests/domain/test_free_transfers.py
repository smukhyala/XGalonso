"""The free-transfer allowance, charged and accrued from the pinned rules.

The rule was previously spread across five places, none of which read the
snapshot: two contracts capped the allowance at a literal 5, the backtest
accrued with ``min(5, ft + 1)`` twice, and the package charged a literal 4 per
paid transfer. FPL has changed free-transfer accumulation before, and every one
of those literals would have kept looking plausible while producing illegal
recommendations.

So every assertion here reads ``rules.max_free_transfers``,
``rules.hit_cost_per_transfer`` or ``rules.transfers_cap``. The one test that
states a bare number says in its name that it is pinning a ``VERIFY`` constant
rather than reading a published one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from xg_alonso.contracts.identifiers import PlayerCode, TenthsOfMillion
from xg_alonso.contracts.recommendation import TransferMove, TransferPackage
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.transfers import accrue, check_transfer_package, settle_gameweek

FIXTURE = Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def rules(payload: dict[str, Any]) -> SquadRules:
    return SquadRules.from_bootstrap(payload, version="2026-27", source_sha256="b" * 64)


def _move() -> TransferMove:
    return TransferMove(
        player_out=PlayerCode(1),
        player_in=PlayerCode(2),
        selling_price=TenthsOfMillion(50),
        purchase_price=TenthsOfMillion(50),
    )


class TestScenario7TwoSavedFreeTransfers:
    def test_a_quiet_week_banks_one(self, rules: SquadRules) -> None:
        ledger = settle_gameweek(free_transfers=1, transfers_made=0, rules=rules)
        assert ledger.free_transfers_after == 2
        assert ledger.hit_cost == 0
        assert ledger.paid_transfers == 0

    def test_spending_two_banked_transfers_costs_nothing(self, rules: SquadRules) -> None:
        ledger = settle_gameweek(free_transfers=2, transfers_made=2, rules=rules)
        assert ledger.free_transfers_used == 2
        assert ledger.paid_transfers == 0
        assert ledger.hit_cost == 0
        # Spent to zero, but the weekly +1 still arrives.
        assert ledger.free_transfers_after == 1

    def test_accrual_converges_to_the_cap_and_stays(self, rules: SquadRules) -> None:
        allowance = 1
        for _ in range(20):
            allowance = accrue(allowance, rules=rules)
        assert allowance == rules.max_free_transfers
        assert accrue(allowance, rules=rules) == rules.max_free_transfers


class TestScenario8TransferHit:
    def test_one_paid_transfer_is_charged_once(self, rules: SquadRules) -> None:
        ledger = settle_gameweek(free_transfers=1, transfers_made=2, rules=rules)
        assert ledger.free_transfers_used == 1
        assert ledger.paid_transfers == 1
        assert ledger.hit_cost == rules.hit_cost_per_transfer
        assert ledger.free_transfers_after == 1

    def test_a_transferring_week_still_accrues(self, rules: SquadRules) -> None:
        """The old `max(0, ft - 1)` slid a weekly transferrer into a permanent hit."""
        allowance = 1
        for _ in range(5):
            allowance = settle_gameweek(
                free_transfers=allowance, transfers_made=1, rules=rules
            ).free_transfers_after
        assert allowance == 1

    def test_exceeding_the_gameweek_cap_is_refused(self, rules: SquadRules) -> None:
        with pytest.raises(ValueError, match="exceeds the cap"):
            settle_gameweek(free_transfers=1, transfers_made=rules.transfers_cap + 1, rules=rules)

    def test_the_per_transfer_charge_is_a_verify_constant(self, rules: SquadRules) -> None:
        """FPL does not publish this one, so it is asserted rather than read."""
        assert rules.hit_cost_per_transfer == 4


class TestTheContractKeepsOnlyTheShape:
    """`contracts` may not import `domain`, so it cannot know the exact charge."""

    def test_a_charge_with_no_paid_transfer_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="hit_cost"):
            TransferPackage(
                moves=(_move(),),
                transfers_used=1,
                free_transfers_available=1,
                hit_cost=4,
                bank_before=TenthsOfMillion(10),
                bank_after=TenthsOfMillion(10),
            )

    def test_no_charge_with_a_paid_transfer_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="hit_cost"):
            TransferPackage(
                moves=(_move(), _move()),
                transfers_used=2,
                free_transfers_available=1,
                hit_cost=0,
                bank_before=TenthsOfMillion(10),
                bank_after=TenthsOfMillion(10),
            )

    def test_an_uneven_charge_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="hit_cost"):
            TransferPackage(
                moves=(_move(), _move(), _move()),
                transfers_used=3,
                free_transfers_available=1,
                hit_cost=5,  # two paid transfers cannot share five points evenly
                bank_before=TenthsOfMillion(10),
                bank_after=TenthsOfMillion(10),
            )

    def test_the_contract_permits_a_wrong_but_even_charge(self, rules: SquadRules) -> None:
        """Which is exactly the gap `check_transfer_package` exists to close."""
        package = TransferPackage(
            moves=(_move(), _move()),
            transfers_used=2,
            free_transfers_available=1,
            hit_cost=8,  # even over one paid transfer, but twice the real charge
            bank_before=TenthsOfMillion(10),
            bank_after=TenthsOfMillion(10),
        )
        violations = check_transfer_package(package, rules=rules)
        assert [v.rule for v in violations] == ["hit_cost"]
        assert str(rules.hit_cost_per_transfer) in violations[0].detail

    def test_a_correct_package_passes_both(self, rules: SquadRules) -> None:
        package = TransferPackage(
            moves=(_move(), _move()),
            transfers_used=2,
            free_transfers_available=1,
            hit_cost=rules.hit_cost_per_transfer,
            bank_before=TenthsOfMillion(10),
            bank_after=TenthsOfMillion(10),
        )
        assert check_transfer_package(package, rules=rules) == []

    def test_an_allowance_beyond_the_cap_is_caught_by_the_rules(self, rules: SquadRules) -> None:
        """The contract no longer bounds this — the rules do."""
        package = TransferPackage(
            moves=(),
            transfers_used=0,
            free_transfers_available=rules.max_free_transfers + 1,
            hit_cost=0,
            bank_before=TenthsOfMillion(10),
            bank_after=TenthsOfMillion(10),
        )
        assert [v.rule for v in check_transfer_package(package, rules=rules)] == [
            "max_free_transfers"
        ]


class TestTheLedgerCannotContradictItself:
    def test_spending_more_free_transfers_than_held_is_refused(self, rules: SquadRules) -> None:
        from xg_alonso.domain.transfers import TransferLedger

        with pytest.raises(ValidationError, match="only 1 were available"):
            TransferLedger(
                free_transfers_before=1,
                transfers_made=2,
                free_transfers_used=2,
                paid_transfers=0,
                hit_cost=0,
                free_transfers_after=1,
            )

    def test_free_plus_paid_must_equal_transfers_made(self) -> None:
        from xg_alonso.domain.transfers import TransferLedger

        with pytest.raises(ValidationError, match="does not equal"):
            TransferLedger(
                free_transfers_before=2,
                transfers_made=3,
                free_transfers_used=1,
                paid_transfers=1,
                hit_cost=4,
                free_transfers_after=1,
            )
