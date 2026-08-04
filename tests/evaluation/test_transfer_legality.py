"""Prices that move, and transfers the game would refuse.

Two things were silently wrong before this. ``walk_forward`` carried one static
price map for the whole season, so no price ever moved and squad value was a
constant by construction. And ``apply_transfer`` committed whatever it was
handed — an unaffordable move or one breaching the three-per-club limit was
applied without complaint, so a backtest could measure a policy FPL would have
refused to run.

Scenarios 5, 6, 13 and 14 from the simulation plan.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    Season,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.reason_codes import Reason, ReasonCode, ReasonPolarity
from xg_alonso.contracts.recommendation import (
    BaselineComparison,
    TransferMove,
    TransferPackage,
    TransferRecommendation,
)
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.constraints import check_squad
from xg_alonso.domain.rules import SquadRules
from xg_alonso.evaluation.backtest import (
    actual_minutes,
    actual_prices,
    apply_transfer,
    price_at_deadline,
    refusals,
    reprice_squad,
)

NOW = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def rules(squad_rules: SquadRules) -> SquadRules:
    """The pinned squad constraints, built once in the root conftest."""
    return squad_rules


def _squad(rules: SquadRules, *, bank: int = 0) -> SquadState:
    """A legal fifteen, three clubs deep, every player at 5.0m."""
    shape = [
        (Position.GKP, rules.rule_for(Position.GKP).squad_select),
        (Position.DEF, rules.rule_for(Position.DEF).squad_select),
        (Position.MID, rules.rule_for(Position.MID).squad_select),
        (Position.FWD, rules.rule_for(Position.FWD).squad_select),
    ]
    picks: list[SquadPick] = []
    slot = 1
    for position, count in shape:
        for _ in range(count):
            picks.append(
                SquadPick(
                    player_code=PlayerCode(slot),
                    position=position,
                    team_id=TeamId(1 + (slot - 1) // rules.max_per_club),
                    purchase_price=TenthsOfMillion(50),
                    current_price=TenthsOfMillion(50),
                    selling_price=TenthsOfMillion(50),
                    squad_slot=slot,
                )
            )
            slot += 1
    return SquadState(
        entry_id=EntryId(1),
        gameweek=GameweekId(5),
        picks=tuple(picks),
        bank=TenthsOfMillion(bank),
        free_transfers=1,
    )


def _recommendation(
    out: int, into: int, *, sell: int, buy: int, bank: int
) -> TransferRecommendation:
    move = TransferMove(
        player_out=PlayerCode(out),
        player_in=PlayerCode(into),
        selling_price=TenthsOfMillion(sell),
        purchase_price=TenthsOfMillion(buy),
    )
    return TransferRecommendation(
        entry_id=EntryId(1),
        gameweek=GameweekId(5),
        package=TransferPackage(
            moves=(move,),
            transfers_used=1,
            free_transfers_available=1,
            hit_cost=0,
            bank_before=TenthsOfMillion(bank),
            bank_after=TenthsOfMillion(bank - (buy - sell)),
        ),
        comparison=BaselineComparison(
            baseline_name="hold",
            baseline_expected_points=40.0,
            candidate_expected_points=42.0,
            horizon_gameweeks=1,
        ),
        reasons=(
            Reason(
                code=ReasonCode.EXPECTED_MINUTES_SECURE,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=PlayerCode(into),
                evidence={"p_start": 0.9, "expected_minutes": 85.0},
                weight=1.0,
            ),
        ),
        expected_points_gain=2.0,
        risk_score=0.5,
        generated_at=NOW,
        run_id="t",
        optimizer_config_hash="h",
    )


class TestScenario5PriceRise:
    def test_half_the_profit_is_retained_and_rounded_down(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        risen = reprice_squad(squad, prices={PlayerCode(1): TenthsOfMillion(57)}, rules=rules)
        pick = risen.by_code(PlayerCode(1))
        assert pick is not None
        assert pick.current_price == 57
        # Bought at 50, worth 57: 7 profit, half retained, rounded down.
        assert pick.selling_price == 53

    def test_an_appreciating_squad_stays_legal(self, rules: SquadRules) -> None:
        """The opening budget stopped applying the moment the squad was bought."""
        squad = _squad(rules)
        risen = reprice_squad(
            squad,
            prices={p.player_code: TenthsOfMillion(90) for p in squad.picks},
            rules=rules,
        )
        own_value = TenthsOfMillion(sum(p.selling_price for p in risen.picks) + risen.bank)

        assert own_value > rules.total_budget
        assert check_squad(risen.picks, rules=rules, bank=risen.bank, budget=own_value) == []
        # And the default still enforces the purchase cap, so nothing else changed.
        assert check_squad(risen.picks, rules=rules, bank=risen.bank) != []


class TestScenario6PriceFall:
    def test_a_loss_is_absorbed_in_full(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        fallen = reprice_squad(squad, prices={PlayerCode(1): TenthsOfMillion(47)}, rules=rules)
        pick = fallen.by_code(PlayerCode(1))
        assert pick is not None
        assert pick.current_price == 47
        assert pick.selling_price == 47

    def test_squad_value_falls_by_the_full_amount(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        before = squad.squad_value
        fallen = reprice_squad(squad, prices={PlayerCode(1): TenthsOfMillion(47)}, rules=rules)
        assert before - fallen.squad_value == 3

    def test_an_unpriced_player_keeps_his_price(self, rules: SquadRules) -> None:
        """A missing row means nothing was published, not that he became free."""
        squad = _squad(rules)
        same = reprice_squad(squad, prices={}, rules=rules)
        assert same.squad_value == squad.squad_value


class TestScenario13RejectedOnSellingValue:
    def test_a_move_priced_against_a_stale_bank_is_refused(self, rules: SquadRules) -> None:
        """`TransferPackage` cannot catch this one, and that is the point.

        Its own validator refuses a negative `bank_after`, so a self-consistent
        package can never describe an unaffordable move. What it cannot know is
        whether its `bank_before` matches the squad it is applied to — a
        recommendation computed against a stale bank passes validation and is
        still unaffordable in fact.
        """
        squad = _squad(rules, bank=0)
        recommendation = _recommendation(1, 999, sell=50, buy=80, bank=30)

        violations = refusals(
            squad,
            recommendation,
            positions={PlayerCode(999): Position.GKP.value},
            teams={PlayerCode(999): 20},
            rules=rules,
        )
        after = apply_transfer(
            squad,
            recommendation,
            prices={PlayerCode(999): TenthsOfMillion(80)},
            positions={PlayerCode(999): Position.GKP.value},
            teams={PlayerCode(999): 20},
            rules=rules,
        )

        assert [v.rule for v in violations] == ["budget"]
        assert after.picks == squad.picks
        assert after.by_code(PlayerCode(999)) is None
        # The transfer never happened, so the allowance accrued.
        assert after.free_transfers == 2

    def test_the_same_move_succeeds_with_enough_in_the_bank(self, rules: SquadRules) -> None:
        squad = _squad(rules, bank=30)
        recommendation = _recommendation(1, 999, sell=50, buy=80, bank=30)

        after = apply_transfer(
            squad,
            recommendation,
            prices={PlayerCode(999): TenthsOfMillion(80)},
            positions={PlayerCode(999): Position.GKP.value},
            teams={PlayerCode(999): 20},
            rules=rules,
        )

        assert after.by_code(PlayerCode(999)) is not None
        assert after.by_code(PlayerCode(1)) is None


class TestScenario14RejectedOnClubQuota:
    def test_a_fourth_player_from_one_club_is_refused(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        # Players 1-3 are club 1. Selling a club-2 player to buy a club-1 player
        # would make four.
        recommendation = _recommendation(4, 999, sell=50, buy=50, bank=0)

        violations = refusals(
            squad,
            recommendation,
            positions={PlayerCode(999): Position.DEF.value},
            teams={PlayerCode(999): 1},
            rules=rules,
        )
        after = apply_transfer(
            squad,
            recommendation,
            prices={PlayerCode(999): TenthsOfMillion(50)},
            positions={PlayerCode(999): Position.DEF.value},
            teams={PlayerCode(999): 1},
            rules=rules,
        )

        assert any(v.rule == "max_per_club" for v in violations)
        assert after.picks == squad.picks

    def test_selling_within_the_same_club_frees_the_slot(self, rules: SquadRules) -> None:
        """The contrast case. An over-eager check would refuse this too."""
        squad = _squad(rules)
        club_one = [p for p in squad.picks if int(p.team_id) == 1]
        assert len(club_one) == rules.max_per_club

        outgoing = club_one[0]
        recommendation = _recommendation(int(outgoing.player_code), 999, sell=50, buy=50, bank=0)

        violations = refusals(
            squad,
            recommendation,
            positions={PlayerCode(999): outgoing.position.value},
            teams={PlayerCode(999): 1},
            rules=rules,
        )
        after = apply_transfer(
            squad,
            recommendation,
            prices={PlayerCode(999): TenthsOfMillion(50)},
            positions={PlayerCode(999): outgoing.position.value},
            teams={PlayerCode(999): 1},
            rules=rules,
        )

        assert violations == []
        assert after.by_code(PlayerCode(999)) is not None


class TestOutcomeExtraction:
    @staticmethod
    def _stats() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_code": [1, 1, 2, 3],
                "season": ["2024-25"] * 4,
                "gameweek_id": [7, 7, 7, 7],
                "minutes": [20, 0, 90, 0],
                "total_points": [2, 1, 6, 0],
                "value": [55, 55, 70, 45],
            }
        )

    def test_minutes_are_summed_across_a_double(self) -> None:
        """20 and 0 is a player who played, and must not be substituted."""
        minutes = actual_minutes(self._stats(), season=Season("2024-25"), gameweek=GameweekId(7))
        assert minutes[PlayerCode(1)] == 20

    def test_a_player_who_did_not_feature_reads_zero(self) -> None:
        minutes = actual_minutes(self._stats(), season=Season("2024-25"), gameweek=GameweekId(7))
        assert minutes[PlayerCode(3)] == 0

    def test_prices_come_from_the_value_column(self) -> None:
        prices = actual_prices(self._stats(), season=Season("2024-25"), gameweek=GameweekId(7))
        assert prices[PlayerCode(2)] == 70

    def test_the_deadline_price_is_the_previous_gameweek(self) -> None:
        """GW7's value is recorded with GW7's result, which is after its deadline."""
        stats = pl.concat(
            [
                self._stats(),
                self._stats().with_columns(
                    pl.lit(6, dtype=pl.Int64).alias("gameweek_id"),
                    pl.lit(99, dtype=pl.Int64).alias("value"),
                ),
            ]
        )
        prices = price_at_deadline(stats, season=Season("2024-25"), gameweek=GameweekId(7))
        assert prices[PlayerCode(2)] == 99

    def test_gameweek_one_has_no_previous_price(self) -> None:
        assert (
            price_at_deadline(self._stats(), season=Season("2024-25"), gameweek=GameweekId(1)) == {}
        )
