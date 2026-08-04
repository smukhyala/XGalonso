"""Backtest tests.

The metrics here are the ones the project is ultimately judged on, so they are
tested for the ways they can *flatter* the system, not only for arithmetic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest
from tests.conftest import BOOTSTRAP_FIXTURE

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    Season,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.recommendation import (
    BaselineComparison,
    TransferMove,
    TransferPackage,
    TransferRecommendation,
)
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.evaluation import (
    BacktestResult,
    GameweekOutcome,
    actual_points,
    apply_transfer,
    gameweek_deadlines,
    score_squad,
)
from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA, conform

NOW = datetime(2025, 8, 15, 12, 0, tzinfo=UTC)


def _rules() -> SquadRules:
    """Rules from the pinned snapshot, so a cap is never a literal in a test."""
    payload = json.loads(BOOTSTRAP_FIXTURE.read_text())
    return SquadRules.from_bootstrap(payload, version="2026-27", source_sha256="b" * 64)


SEASON = Season("2025-26")


def _pick(code: int, slot: int, position: Position = Position.MID, *, captain: bool = False):  # type: ignore[no-untyped-def]
    return SquadPick(
        player_code=PlayerCode(code),
        position=position,
        team_id=TeamId(1 + code % 10),
        purchase_price=TenthsOfMillion(50),
        current_price=TenthsOfMillion(50),
        selling_price=TenthsOfMillion(50),
        squad_slot=slot,
        is_captain=captain,
    )


def _squad() -> SquadState:
    layout = [
        (Position.GKP, 1),
        (Position.DEF, 4),
        (Position.MID, 4),
        (Position.FWD, 2),
        (Position.GKP, 1),
        (Position.DEF, 1),
        (Position.MID, 1),
        (Position.FWD, 1),
    ]
    picks, code, slot = [], 100, 1
    for position, count in layout:
        for _ in range(count):
            picks.append(_pick(code, slot, position, captain=slot == 1))
            code += 1
            slot += 1
    return SquadState(
        entry_id=EntryId(1),
        gameweek=GameweekId(5),
        picks=tuple(picks),
        bank=TenthsOfMillion(10),
        free_transfers=1,
    )


class TestScoring:
    def test_only_the_starting_xi_scores(self) -> None:
        squad = _squad()
        points = dict.fromkeys((p.player_code for p in squad.picks), 5)
        # 11 starters at 5, captain doubled: 11*5 + 5 = 60.
        assert score_squad(squad, points) == 60

    def test_captain_is_doubled(self) -> None:
        squad = _squad()
        captain = next(p for p in squad.picks if p.is_captain)
        assert score_squad(squad, {captain.player_code: 10}) == 20

    def test_missing_players_score_zero_not_error(self) -> None:
        """A player with no row simply did not feature."""
        assert score_squad(_squad(), {}) == 0

    def test_double_gameweeks_are_summed(self) -> None:
        stats = conform(
            pl.DataFrame(
                [
                    {
                        "player_code": 1,
                        "season": str(SEASON),
                        "gameweek_id": 7,
                        "total_points": 6,
                        "kickoff_time": NOW,
                        "available_time": NOW,
                    },
                    {
                        "player_code": 1,
                        "season": str(SEASON),
                        "gameweek_id": 7,
                        "total_points": 9,
                        "kickoff_time": NOW,
                        "available_time": NOW,
                    },
                ],
                infer_schema_length=None,
            ),
            PLAYER_GAMEWEEK_STATS_SCHEMA,
        )
        points = actual_points(stats, season=SEASON, gameweek=GameweekId(7))
        assert points[PlayerCode(1)] == 15, "two fixtures in one gameweek both count"


class TestDeadlines:
    def test_deadline_precedes_the_first_kickoff(self) -> None:
        """Using the kickoff itself would admit team news nobody had."""
        stats = conform(
            pl.DataFrame(
                [
                    {
                        "player_code": 1,
                        "season": str(SEASON),
                        "gameweek_id": 3,
                        "kickoff_time": datetime(2025, 9, 13, 14, 0, tzinfo=UTC),
                        "available_time": NOW,
                    },
                    {
                        "player_code": 2,
                        "season": str(SEASON),
                        "gameweek_id": 3,
                        "kickoff_time": datetime(2025, 9, 13, 11, 30, tzinfo=UTC),
                        "available_time": NOW,
                    },
                ],
                infer_schema_length=None,
            ),
            PLAYER_GAMEWEEK_STATS_SCHEMA,
        )
        deadline = gameweek_deadlines(stats)["deadline"][0]
        assert deadline == datetime(2025, 9, 13, 10, 0, tzinfo=UTC)


class TestApplyTransfer:
    def _recommendation(self, out_code: int, in_code: int) -> TransferRecommendation:
        move = TransferMove(
            player_out=PlayerCode(out_code),
            player_in=PlayerCode(in_code),
            selling_price=TenthsOfMillion(50),
            purchase_price=TenthsOfMillion(55),
        )
        package = TransferPackage(
            moves=(move,),
            transfers_used=1,
            free_transfers_available=1,
            hit_cost=0,
            bank_before=TenthsOfMillion(10),
            bank_after=TenthsOfMillion(5),
        )
        from xg_alonso.contracts.reason_codes import Reason, ReasonCode, ReasonPolarity

        return TransferRecommendation(
            entry_id=EntryId(1),
            gameweek=GameweekId(5),
            package=package,
            comparison=BaselineComparison(
                baseline_name="hold",
                baseline_expected_points=40.0,
                candidate_expected_points=44.0,
                horizon_gameweeks=1,
            ),
            reasons=(
                Reason(
                    code=ReasonCode.EXPECTED_MINUTES_SECURE,
                    polarity=ReasonPolarity.SUPPORTS_IN,
                    subject=PlayerCode(in_code),
                    evidence={"p_start": 0.9, "expected_minutes": 85.0},
                    weight=1.0,
                ),
            ),
            expected_points_gain=4.0,
            risk_score=0.5,
            generated_at=NOW,
            run_id="t",
            optimizer_config_hash="h",
        )

    def test_transfer_swaps_the_player_and_keeps_the_slot(self) -> None:
        squad = _squad()
        outgoing = squad.picks[3]
        after = apply_transfer(
            squad,
            self._recommendation(int(outgoing.player_code), 999),
            prices={PlayerCode(999): TenthsOfMillion(55)},
            positions={PlayerCode(999): outgoing.position.value},
            teams={PlayerCode(999): 20},
            rules=_rules(),
        )
        assert after.by_code(PlayerCode(999)) is not None
        assert after.by_code(outgoing.player_code) is None
        assert after.by_code(PlayerCode(999)).squad_slot == outgoing.squad_slot  # type: ignore[union-attr]

    def test_holding_banks_a_free_transfer(self) -> None:
        """Rolling must be a real alternative, not a wasted week."""
        squad = _squad()
        hold = TransferRecommendation(
            entry_id=EntryId(1),
            gameweek=GameweekId(5),
            package=TransferPackage(
                moves=(),
                transfers_used=0,
                free_transfers_available=1,
                hit_cost=0,
                bank_before=TenthsOfMillion(10),
                bank_after=TenthsOfMillion(10),
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
        after = apply_transfer(squad, hold, prices={}, positions={}, teams={}, rules=_rules())
        assert after.free_transfers == 2

    def test_free_transfers_cap_at_the_rule(self) -> None:
        """The cap comes from the payload, not from a literal 5."""
        rules = _rules()
        squad = _squad().model_copy(update={"free_transfers": rules.max_free_transfers})
        hold = TransferRecommendation(
            entry_id=EntryId(1),
            gameweek=GameweekId(5),
            package=TransferPackage(
                moves=(),
                transfers_used=0,
                free_transfers_available=rules.max_free_transfers,
                hit_cost=0,
                bank_before=TenthsOfMillion(10),
                bank_after=TenthsOfMillion(10),
            ),
            comparison=BaselineComparison(
                baseline_name="hold",
                baseline_expected_points=1.0,
                candidate_expected_points=1.0,
                horizon_gameweeks=1,
            ),
            reasons=(),
            expected_points_gain=0.0,
            risk_score=0.0,
            generated_at=NOW,
            run_id="t",
            optimizer_config_hash="h",
        )
        after = apply_transfer(squad, hold, prices={}, positions={}, teams={}, rules=rules)
        assert after.free_transfers == rules.max_free_transfers

    def test_selling_a_player_not_owned_fails(self) -> None:
        with pytest.raises(KeyError, match="not in the squad"):
            apply_transfer(
                _squad(),
                self._recommendation(9999, 888),
                prices={PlayerCode(888): TenthsOfMillion(55)},
                positions={PlayerCode(888): "MID"},
                teams={PlayerCode(888): 20},
                rules=_rules(),
            )


class TestMetricsDoNotFlatter:
    """The per-decision metric must isolate a transfer from squad drift."""

    def _outcome(self, gw: int, *, incremental: int, decision: int, made: bool = True):  # type: ignore[no-untyped-def]
        return GameweekOutcome(
            season=SEASON,
            gameweek=GameweekId(gw),
            policy_points=50 + incremental,
            hold_points=50,
            hit_cost=0,
            transfer_made=made,
            player_out=PlayerCode(1) if made else None,
            player_in=PlayerCode(2) if made else None,
            predicted_gain=2.0,
            decision_delta=decision,
        )

    def test_win_rate_uses_the_decision_not_the_squad_gap(self) -> None:
        """A squad ahead on aggregate must not report a 100% win rate.

        This is the bug the metric was rewritten to prevent: comparing two
        diverging squads makes every week a 'win' once the acting squad leads,
        regardless of what was decided that week.
        """
        result = BacktestResult(
            outcomes=[
                self._outcome(6, incremental=20, decision=5),
                self._outcome(7, incremental=20, decision=-3),
                self._outcome(8, incremental=20, decision=-1),
                self._outcome(9, incremental=20, decision=4),
            ]
        )
        assert result.total_incremental == 80, "the squad is well ahead overall"
        assert result.decision_win_rate == 0.5, "but only half the decisions helped"

    def test_regret_counts_only_losing_decisions(self) -> None:
        result = BacktestResult(
            outcomes=[
                self._outcome(6, incremental=10, decision=6),
                self._outcome(7, incremental=10, decision=-4),
                self._outcome(8, incremental=10, decision=-2),
            ]
        )
        assert result.mean_regret == pytest.approx(3.0)

    def test_holds_are_excluded_from_the_win_rate(self) -> None:
        """Counting holds would inflate the rate with decisions that did nothing."""
        result = BacktestResult(
            outcomes=[
                self._outcome(6, incremental=0, decision=0, made=False),
                self._outcome(7, incremental=5, decision=5),
            ]
        )
        assert result.decision_win_rate == 1.0
        assert result.transfers_made == 1

    def test_calibration_compares_prediction_to_the_decision(self) -> None:
        result = BacktestResult(outcomes=[self._outcome(6, incremental=50, decision=3)])
        # predicted +2.0 against a realised +3 decision.
        assert result.calibration_error == pytest.approx(1.0)

    def test_hits_reduce_the_incremental(self) -> None:
        outcome = GameweekOutcome(
            season=SEASON,
            gameweek=GameweekId(6),
            policy_points=60,
            hold_points=50,
            hit_cost=4,
            transfer_made=True,
            player_out=PlayerCode(1),
            player_in=PlayerCode(2),
            predicted_gain=8.0,
            decision_delta=10,
        )
        assert outcome.net_policy_points == 56
        assert outcome.incremental == 6

    def test_empty_result_reports_honestly(self) -> None:
        assert "No gameweeks" in BacktestResult().summary()
