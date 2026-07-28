"""Transfer-board tests.

The board exists because the optimizer scored the whole market and the product
showed one row of it. Two properties keep that fix honest: every option on the
board must be a legal move, and every squad member must appear — including the
ones with nothing worth doing, since a player who is merely absent is
indistinguishable from a player who was never considered.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from xg_alonso.contracts.evidence import FeatureEvidence, FeatureValue
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
from xg_alonso.contracts.reason_codes import ReasonCode
from xg_alonso.contracts.recommendation import TransferBoard
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import PositionRule, SquadRules
from xg_alonso.explanations.reasons import PopulationStats
from xg_alonso.optimization.transfer import (
    Candidate,
    best_single_transfer,
    build_transfer_board,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)

_LAYOUT = (
    [(Position.GKP, code) for code in (1, 2)]
    + [(Position.DEF, code) for code in (3, 4, 5, 6, 7)]
    + [(Position.MID, code) for code in (8, 9, 10, 11, 12)]
    + [(Position.FWD, code) for code in (13, 14, 15)]
)


def _rules() -> SquadRules:
    return SquadRules(
        version="test",
        source_sha256="a" * 64,
        squad_size=15,
        starting_size=11,
        max_per_club=3,
        total_budget=TenthsOfMillion(1000),
        sell_on_fee=0.5,
        sell_at_purchase_price=False,
        max_extra_free_transfers=4,
        transfers_cap=20,
        positions=(
            PositionRule(position=Position.GKP, squad_select=2, min_play=1, max_play=1),
            PositionRule(position=Position.DEF, squad_select=5, min_play=3, max_play=5),
            PositionRule(position=Position.MID, squad_select=5, min_play=2, max_play=5),
            PositionRule(position=Position.FWD, squad_select=3, min_play=1, max_play=3),
        ),
    )


def _evidence(xg: float, xa: float) -> FeatureEvidence:
    return FeatureEvidence(
        values=(
            FeatureValue(
                name="expected_goals_per90_5",
                label="expected goals per 90",
                family="player_rate",
                value=xg,
                percentile=min(0.99, xg),
            ),
            FeatureValue(
                name="expected_assists_per90_5",
                label="expected assists per 90",
                family="player_rate",
                value=xa,
                percentile=min(0.99, xa),
            ),
        )
    )


def _prediction(
    code: int,
    points: float,
    position: Position,
    *,
    xg: float = 0.2,
    xa: float = 0.1,
    p_start: float = 0.9,
) -> PlayerPrediction:
    minutes = MinutesPrediction(
        p_appearance=1.0,
        p_start=p_start,
        expected_minutes=85.0 * p_start,
        p_60_plus=p_start * 0.9,
        minutes_sd=8.0,
    )
    breakdown = PointsBreakdown(
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
    )
    return PlayerPrediction(
        player_code=PlayerCode(code),
        position=position,
        from_gameweek=GameweekId(1),
        horizon_gameweeks=1,
        components=ComponentExpectations(
            minutes=minutes,
            goals=xg,
            assists=xa,
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
        breakdown=breakdown,
        expected_points=points,
        expected_points_sd=0.5,
        scoring_rules_version="test",
        provenance=PredictionProvenance(
            model_name="t",
            model_version="1",
            model_artifact_sha256="b" * 64,
            feature_set_name="f",
            feature_set_version="1",
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="r",
            code_version="c",
        ),
        feature_evidence=_evidence(xg, xa),
    )


def _squad(bank: int = 20, free_transfers: int = 1) -> SquadState:
    picks = [
        SquadPick(
            player_code=PlayerCode(code),
            position=position,
            team_id=TeamId(1 + code % 8),
            purchase_price=TenthsOfMillion(50),
            current_price=TenthsOfMillion(50),
            selling_price=TenthsOfMillion(50),
            squad_slot=index + 1,
        )
        for index, (position, code) in enumerate(_LAYOUT)
    ]
    return SquadState(
        entry_id=EntryId(1),
        gameweek=GameweekId(1),
        picks=tuple(picks),
        bank=TenthsOfMillion(bank),
        free_transfers=free_transfers,
    )


def _owned_predictions() -> dict[PlayerCode, PlayerPrediction]:
    return {PlayerCode(code): _prediction(code, 2.0, position) for position, code in _LAYOUT}


def _market(*, better_forward: bool = True) -> list[Candidate]:
    """A market of unowned players, one clearly better forward among them."""
    candidates: list[Candidate] = []
    code = 100
    for position in (Position.GKP, Position.DEF, Position.MID, Position.FWD):
        for offset in range(3):
            points = 2.0 + offset * 0.1
            if position is Position.FWD and offset == 2 and better_forward:
                points = 6.0
            candidates.append(
                Candidate(
                    player_code=PlayerCode(code),
                    position=position,
                    team_id=TeamId(20),
                    price=TenthsOfMillion(60),
                    prediction=_prediction(code, points, position, xg=0.8 if points > 5 else 0.1),
                )
            )
            code += 1
    return candidates


def _board(*, better_forward: bool = True) -> TransferBoard:
    predictions = _owned_predictions()
    candidates = _market(better_forward=better_forward)
    for candidate in candidates:
        predictions[candidate.player_code] = candidate.prediction
    return build_transfer_board(
        _squad(),
        candidates=candidates,
        predictions=predictions,
        rules=_rules(),
        population=PopulationStats.from_predictions(list(predictions.values())),
    )


class TestCoverage:
    def test_every_squad_member_appears_exactly_once(self) -> None:
        """A player who is simply missing reads as one the system overlooked."""
        board = _board()
        listed = [entry.player_out for entry in board.by_player]
        assert len(listed) == 15
        assert sorted(int(c) for c in listed) == [
            code for _, code in sorted(_LAYOUT, key=lambda p: p[1])
        ]

    def test_a_player_with_no_move_still_carries_a_reason(self) -> None:
        board = _board(better_forward=False)
        without = [entry for entry in board.by_player if entry.option is None]
        assert without, "expected at least one player with nothing worth doing"
        for entry in without:
            assert entry.reasons

    def test_the_absence_reason_names_the_position_constraint(self) -> None:
        """This is the answer to 'why wasn't the defender considered?'"""
        board = _board(better_forward=False)
        entry = next(e for e in board.by_player if e.option is None)
        codes = {reason.code for reason in entry.reasons}
        assert ReasonCode.POSITION_LOCKED in codes
        rendered = " ".join(reason.render() for reason in entry.reasons)
        assert entry.position.value in rendered


class TestLegality:
    def test_every_option_swaps_like_for_like(self) -> None:
        board = _board()
        positions = {int(code): position for position, code in _LAYOUT}
        market = {int(c.player_code): c.position for c in _market()}
        for option in board.top:
            out_position = positions[int(option.move.player_out)]
            in_position = market[int(option.move.player_in)]
            assert out_position is in_position

    def test_no_option_buys_a_player_already_owned(self) -> None:
        board = _board()
        owned = {code for _, code in _LAYOUT}
        for option in board.top:
            assert int(option.move.player_in) not in owned

    def test_options_are_ranked_by_net_gain(self) -> None:
        board = _board()
        gains = [option.net_gain for option in board.top]
        assert gains == sorted(gains, reverse=True)

    def test_by_player_entries_sell_the_player_they_describe(self) -> None:
        board = _board()
        for entry in board.by_player:
            if entry.option is not None:
                assert entry.option.move.player_out == entry.player_out


class TestAccounting:
    def test_the_board_reports_what_it_searched(self) -> None:
        board = _board()
        assert board.candidates_considered == len(_market())
        assert board.legal_moves > 0

    def test_the_board_is_capped_but_not_silently_empty(self) -> None:
        board = _board()
        assert 0 < len(board.top) <= 8
        assert board.legal_moves >= len(board.top)


class TestReasons:
    def test_the_headline_move_cites_a_named_statistic(self) -> None:
        """The defect this work exists to fix: no feature was ever named."""
        predictions = _owned_predictions()
        candidates = _market()
        for candidate in candidates:
            predictions[candidate.player_code] = candidate.prediction

        recommendation = best_single_transfer(
            _squad(),
            candidates=candidates,
            predictions=predictions,
            rules=_rules(),
            entry_id=EntryId(1),
            gameweek=GameweekId(1),
            generated_at=NOW,
            run_id="r",
            optimizer_config_hash="h",
            population=PopulationStats.from_predictions(list(predictions.values())),
        )

        assert not recommendation.package.is_hold
        codes = {reason.code for reason in recommendation.reasons}
        assert ReasonCode.XG_RATE_HIGHER in codes, (
            "a transfer justified without naming an underlying statistic is the "
            "defect this vocabulary was expanded to remove"
        )

    def test_every_reason_names_its_subject(self) -> None:
        """Unattributed reasons read as self-contradictory on screen."""
        board = _board()
        for option in board.top:
            for reason in option.reasons:
                assert reason.subject in (option.move.player_in, option.move.player_out)

    def test_the_recommendation_carries_the_board_it_was_chosen_from(self) -> None:
        predictions = _owned_predictions()
        candidates = _market()
        for candidate in candidates:
            predictions[candidate.player_code] = candidate.prediction

        recommendation = best_single_transfer(
            _squad(),
            candidates=candidates,
            predictions=predictions,
            rules=_rules(),
            entry_id=EntryId(1),
            gameweek=GameweekId(1),
            generated_at=NOW,
            run_id="r",
            optimizer_config_hash="h",
        )
        assert recommendation.board is not None
        assert len(recommendation.board.by_player) == 15

    def test_a_hold_still_shows_what_it_beat(self) -> None:
        """'Nothing was good enough' and 'nothing was considered' must not look alike."""
        predictions = _owned_predictions()
        candidates = _market(better_forward=False)
        for candidate in candidates:
            predictions[candidate.player_code] = candidate.prediction

        recommendation = best_single_transfer(
            _squad(),
            candidates=candidates,
            predictions=predictions,
            rules=_rules(),
            entry_id=EntryId(1),
            gameweek=GameweekId(1),
            generated_at=NOW,
            run_id="r",
            optimizer_config_hash="h",
        )
        assert recommendation.package.is_hold
        assert recommendation.board is not None
        assert recommendation.board.legal_moves > 0


class TestContractGuards:
    def test_an_unexplained_absence_is_rejected(self) -> None:
        from xg_alonso.contracts.recommendation import PlayerBestMove

        with pytest.raises(ValueError, match="no move and no reason"):
            PlayerBestMove(
                player_out=PlayerCode(1),
                position=Position.MID,
                legal_replacements=0,
            )
