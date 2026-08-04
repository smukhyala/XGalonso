"""Scenario 15: the squad stays legal through every weekly transition.

The other fourteen scenarios each isolate one mechanic. This one runs them
together — hold, applied transfer, refused transfer, blank, double, prices
moving both ways — and asserts after **every** gameweek that the squad is still
something the game would accept.

The invariants are deliberately the ones that a partially-correct
implementation satisfies most of the time: slot integrity, one captain, a
non-negative bank, an allowance inside the rules. Each is cheap to check and
each has a plausible bug that would leave it true in nine weeks out of ten.

Also covers the prediction half of scenario 10: two fixtures in a gameweek must
*sum*, and the frozen `breakdown.total == expected_points` validator is what
makes the aggregate self-checking.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.contracts.reason_codes import Reason, ReasonCode, ReasonPolarity
from xg_alonso.contracts.recommendation import (
    BaselineComparison,
    TransferMove,
    TransferPackage,
    TransferRecommendation,
)
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.constraints import check_squad, check_starting_xi
from xg_alonso.domain.rules import SquadRules
from xg_alonso.evaluation.backtest import apply_transfer, reprice_squad
from xg_alonso.prediction.gameweek import (
    blank_prediction,
    collapse_by_player,
    combine_gameweek_fixtures,
)

NOW = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def rules(squad_rules: SquadRules) -> SquadRules:
    """The pinned squad constraints, built once in the root conftest."""
    return squad_rules


def _prediction(code: int, points: float, *, position: Position = Position.MID) -> PlayerPrediction:
    return PlayerPrediction(
        player_code=PlayerCode(code),
        position=position,
        from_gameweek=GameweekId(34),
        horizon_gameweeks=1,
        components=ComponentExpectations(
            minutes=MinutesPrediction(
                p_appearance=0.95,
                p_start=0.9,
                expected_minutes=85.0,
                p_60_plus=0.85,
                minutes_sd=10.0,
            ),
            goals=0.3,
            assists=0.2,
            clean_sheet_probability=0.0,
            goals_conceded=0.0,
            saves=0.0,
            yellow_cards=0.0,
            red_cards=0.0,
            own_goals=0.0,
            penalties_saved=0.0,
            penalties_missed=0.0,
            defensive_contribution_probability=0.0,
            bonus=0.0,
        ),
        breakdown=PointsBreakdown(
            appearance=points,
            goals=0.0,
            assists=0.0,
            clean_sheets=0.0,
            goals_conceded=0.0,
            saves=0.0,
            cards=0.0,
            own_goals=0.0,
            penalties=0.0,
            defensive_contribution=0.0,
            bonus=0.0,
        ),
        expected_points=points,
        expected_points_sd=2.0,
        scoring_rules_version="test",
        provenance=PredictionProvenance(
            model_name="m",
            model_version="1",
            model_artifact_sha256="c" * 64,
            feature_set_name="f",
            feature_set_version="1",
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="r",
            code_version="v",
        ),
    )


class TestScenario10PredictionsSumAcrossADouble:
    def test_points_and_the_breakdown_both_sum(self) -> None:
        legs = [_prediction(1, 4.0), _prediction(1, 3.0)]
        combined = combine_gameweek_fixtures(legs)

        assert combined.expected_points == pytest.approx(7.0)
        # The frozen validator would have rejected construction otherwise, which
        # is what makes the aggregate self-checking rather than merely checked.
        assert combined.breakdown.total == pytest.approx(combined.expected_points)

    def test_uncertainty_adds_in_quadrature(self) -> None:
        """Independent matches, unlike the horizon's multiplicative inflation."""
        legs = [_prediction(1, 4.0), _prediction(1, 3.0)]
        combined = combine_gameweek_fixtures(legs)
        assert combined.expected_points_sd == pytest.approx((2.0**2 + 2.0**2) ** 0.5)

    def test_two_rows_collapse_to_one_entry(self) -> None:
        legs = [_prediction(1, 4.0), _prediction(1, 3.0), _prediction(2, 5.0)]
        collapsed = collapse_by_player(legs)

        assert len(collapsed) == 2
        assert collapsed[PlayerCode(1)].expected_points == pytest.approx(7.0)
        assert collapsed[PlayerCode(2)].expected_points == pytest.approx(5.0)

    def test_a_blank_scores_nothing_at_all(self) -> None:
        """Zero, not a small number: a blanking player must be uncaptainable."""
        blank = blank_prediction(
            player_code=PlayerCode(9), position=Position.FWD, template=_prediction(1, 4.0)
        )
        assert blank.expected_points == 0.0
        assert blank.components.minutes.expected_minutes == 0.0
        assert blank.components.minutes.p_start == 0.0

    def test_rows_for_two_players_are_refused(self) -> None:
        with pytest.raises(ValueError, match="2 players"):
            combine_gameweek_fixtures([_prediction(1, 4.0), _prediction(2, 3.0)])

    def test_an_empty_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="blank_prediction"):
            combine_gameweek_fixtures([])


def _squad(rules: SquadRules) -> SquadState:
    layout = (
        [Position.GKP]
        + [Position.DEF] * 4
        + [Position.MID] * 4
        + [Position.FWD] * 2
        + [Position.MID, Position.DEF, Position.FWD, Position.GKP]
    )
    picks = [
        SquadPick(
            player_code=PlayerCode(i + 1),
            position=position,
            team_id=TeamId(1 + i // rules.max_per_club),
            purchase_price=TenthsOfMillion(50),
            current_price=TenthsOfMillion(50),
            selling_price=TenthsOfMillion(50),
            squad_slot=i + 1,
            is_captain=(i == 4),
            is_vice_captain=(i == 9),
        )
        for i, position in enumerate(layout)
    ]
    return SquadState(
        entry_id=EntryId(1),
        gameweek=GameweekId(5),
        picks=tuple(picks),
        bank=TenthsOfMillion(20),
        free_transfers=1,
    )


def _hold() -> TransferRecommendation:
    return TransferRecommendation(
        entry_id=EntryId(1),
        gameweek=GameweekId(5),
        package=TransferPackage(
            moves=(),
            transfers_used=0,
            free_transfers_available=1,
            hit_cost=0,
            bank_before=TenthsOfMillion(20),
            bank_after=TenthsOfMillion(20),
        ),
        comparison=BaselineComparison(
            baseline_name="hold",
            baseline_expected_points=40.0,
            candidate_expected_points=40.0,
            horizon_gameweeks=1,
        ),
        reasons=(),
        expected_points_gain=0.0,
        risk_score=0.0,
        generated_at=NOW,
        run_id="t",
        optimizer_config_hash="h",
    )


def _transfer(out: int, into: int, *, sell: int, buy: int, bank: int) -> TransferRecommendation:
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


class TestScenario15SquadStaysLegalThroughEveryTransition:
    def test_a_five_week_walk_never_leaves_an_illegal_squad(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        opening_value = squad.squad_value

        # hold, applied transfer, refused transfer (unaffordable), hold, transfer
        week: list[tuple[TransferRecommendation, dict[PlayerCode, TenthsOfMillion]]] = [
            (_hold(), {}),
            (
                _transfer(6, 900, sell=50, buy=60, bank=20),
                {PlayerCode(1): TenthsOfMillion(56)},
            ),
            (
                # priced against a bank the squad no longer has
                _transfer(7, 901, sell=50, buy=90, bank=60),
                {PlayerCode(2): TenthsOfMillion(44)},
            ),
            (_hold(), {PlayerCode(3): TenthsOfMillion(62)}),
            (
                _transfer(8, 902, sell=50, buy=50, bank=10),
                {PlayerCode(4): TenthsOfMillion(48)},
            ),
        ]

        positions = {
            PlayerCode(900): Position.MID.value,
            PlayerCode(901): Position.MID.value,
            PlayerCode(902): Position.MID.value,
        }
        teams = {PlayerCode(900): 18, PlayerCode(901): 19, PlayerCode(902): 20}

        for recommendation, prices in week:
            squad = reprice_squad(squad, prices=prices, rules=rules)
            squad = apply_transfer(
                squad,
                recommendation,
                prices={},
                positions=positions,
                teams=teams,
                rules=rules,
            )

            # The squad's own value is the ceiling; the opening cap stopped
            # applying the moment it was assembled.
            ceiling = TenthsOfMillion(max(int(opening_value), int(squad.squad_value)))
            assert check_squad(squad.picks, rules=rules, bank=squad.bank, budget=ceiling) == []
            assert check_starting_xi(squad.starters, rules=rules) == []
            assert 0 <= squad.free_transfers <= rules.max_free_transfers
            assert squad.bank >= 0
            assert sorted(p.squad_slot for p in squad.picks) == list(range(1, 16))
            assert sum(p.is_captain for p in squad.picks) == 1
            assert sum(p.is_vice_captain for p in squad.picks) == 1
            assert len({p.player_code for p in squad.picks}) == rules.squad_size

    def test_a_refused_transfer_leaves_the_squad_byte_identical(self, rules: SquadRules) -> None:
        squad = _squad(rules)
        before = squad.picks

        after = apply_transfer(
            squad,
            _transfer(6, 900, sell=50, buy=90, bank=60),  # bank is really 20
            prices={},
            positions={PlayerCode(900): Position.MID.value},
            teams={PlayerCode(900): 18},
            rules=rules,
        )

        assert after.picks == before
        assert after.free_transfers == 2  # accrued, because nothing was spent
