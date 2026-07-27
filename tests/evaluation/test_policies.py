"""Policy tests.

The controls exist to make the backtest falsifiable, so they are tested for the
properties that make them valid controls — identical legal option sets,
reproducible randomness, and selection rules that genuinely differ.
"""

from __future__ import annotations

import random
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
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.evaluation import POLICIES, run_policy
from xg_alonso.optimization.transfer import Candidate

NOW = datetime(2025, 9, 1, 12, 0, tzinfo=UTC)


def _rules() -> SquadRules:
    from xg_alonso.domain.rules import PositionRule

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


def _prediction(code: int, points: float) -> PlayerPrediction:
    minutes = MinutesPrediction(
        p_appearance=1.0, p_start=0.9, expected_minutes=85.0, p_60_plus=0.85, minutes_sd=10.0
    )
    components = ComponentExpectations(
        minutes=minutes,
        goals=0.0,
        assists=0.0,
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
        position=Position.MID,
        from_gameweek=GameweekId(6),
        horizon_gameweeks=1,
        components=components,
        breakdown=breakdown,
        expected_points=points,
        expected_points_sd=0.5,
        scoring_rules_version="test",
        provenance=PredictionProvenance(
            model_name="m",
            model_version="1",
            model_artifact_sha256="b" * 64,
            feature_set_name="f",
            feature_set_version="1",
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="r",
            code_version="c",
        ),
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
    picks, code, slot = [], 1, 1
    for position, count in layout:
        for _ in range(count):
            picks.append(
                SquadPick(
                    player_code=PlayerCode(code),
                    position=position,
                    team_id=TeamId(1 + code % 8),
                    purchase_price=TenthsOfMillion(50),
                    current_price=TenthsOfMillion(50),
                    selling_price=TenthsOfMillion(50),
                    squad_slot=slot,
                    is_captain=slot == 1,
                )
            )
            code += 1
            slot += 1
    return SquadState(
        entry_id=EntryId(1),
        gameweek=GameweekId(6),
        picks=tuple(picks),
        bank=TenthsOfMillion(200),
        free_transfers=1,
    )


def _setup() -> tuple[SquadState, list[Candidate], dict[PlayerCode, PlayerPrediction]]:
    """A squad of low scorers, plus three distinguishable buy options."""
    squad = _squad()
    predictions = {p.player_code: _prediction(int(p.player_code), 1.0) for p in squad.picks}

    options = [
        # (code, price, expected points) — cheap-and-best vs expensive-and-worst,
        # so price and projection disagree and the policies must diverge.
        (100, TenthsOfMillion(60), 9.0),
        (101, TenthsOfMillion(120), 4.0),
        (102, TenthsOfMillion(80), 6.0),
    ]
    candidates = []
    for code, price, points in options:
        prediction = _prediction(code, points)
        predictions[PlayerCode(code)] = prediction
        candidates.append(
            Candidate(
                player_code=PlayerCode(code),
                position=Position.MID,
                team_id=TeamId(20),
                price=price,
                prediction=prediction,
            )
        )
    return squad, candidates, predictions


def _run(policy: str, seed: int = 1) -> object:
    squad, candidates, predictions = _setup()
    return run_policy(
        squad,
        selector=POLICIES[policy],
        candidates=candidates,
        predictions=predictions,
        rules=_rules(),
        entry_id=EntryId(1),
        gameweek=GameweekId(6),
        generated_at=NOW,
        run_id="t",
        rng=random.Random(seed),
        policy_name=policy,
    )


class TestPoliciesDiffer:
    def test_model_picks_the_best_projected_net_of_cost(self) -> None:
        result = _run("model")
        assert result.package.moves[0].player_in == PlayerCode(100)  # type: ignore[attr-defined]

    def test_highest_form_picks_the_best_projection(self) -> None:
        result = _run("highest_form")
        assert result.package.moves[0].player_in == PlayerCode(100)  # type: ignore[attr-defined]

    def test_most_expensive_picks_by_price_not_projection(self) -> None:
        """The control must genuinely ignore the model, or it is not a control."""
        result = _run("most_expensive")
        assert result.package.moves[0].player_in == PlayerCode(101)  # type: ignore[attr-defined]

    def test_hold_never_transfers(self) -> None:
        assert _run("hold").package.is_hold  # type: ignore[attr-defined]

    def test_every_named_policy_runs(self) -> None:
        for name in POLICIES:
            assert _run(name) is not None


class TestControlsAreValid:
    def test_random_is_reproducible_for_a_given_seed(self) -> None:
        """An unseeded control makes the comparison unrepeatable."""
        first = _run("random", seed=7)
        second = _run("random", seed=7)
        assert first.package.moves[0].player_in == second.package.moves[0].player_in  # type: ignore[attr-defined]

    def test_random_explores_more_than_one_option(self) -> None:
        picks = {
            _run("random", seed=s).package.moves[0].player_in  # type: ignore[attr-defined]
            for s in range(30)
        }
        assert len(picks) > 1, "a random control that always picks the same option is not random"

    def test_every_policy_produces_a_legal_affordable_move(self) -> None:
        """All policies choose from the same legality-filtered set."""
        squad, _, _ = _setup()
        for name in POLICIES:
            result = _run(name)
            if result.package.is_hold:  # type: ignore[attr-defined]
                continue
            move = result.package.moves[0]  # type: ignore[attr-defined]
            outgoing = squad.by_code(move.player_out)
            assert outgoing is not None
            assert move.purchase_price <= outgoing.selling_price + squad.bank
            assert result.package.bank_after >= 0  # type: ignore[attr-defined]

    def test_every_policy_carries_grounded_evidence(self) -> None:
        """A control must not dodge the contract that keeps prose honest."""
        for name in POLICIES:
            result = _run(name)
            if not result.package.is_hold:  # type: ignore[attr-defined]
                assert result.reasons, f"{name} produced an unexplained transfer"  # type: ignore[attr-defined]
                assert result.render_reasons()  # type: ignore[attr-defined]

    def test_model_holds_when_nothing_clears_the_bar(self) -> None:
        """Selectivity is the skill: knowing when not to spend a transfer."""
        squad = _squad()
        predictions = {p.player_code: _prediction(int(p.player_code), 5.0) for p in squad.picks}
        marginal = _prediction(200, 5.01)
        predictions[PlayerCode(200)] = marginal
        candidates = [
            Candidate(
                player_code=PlayerCode(200),
                position=Position.MID,
                team_id=TeamId(20),
                price=TenthsOfMillion(60),
                prediction=marginal,
            )
        ]
        result = run_policy(
            squad,
            selector=POLICIES["model"],
            candidates=candidates,
            predictions=predictions,
            rules=_rules(),
            entry_id=EntryId(1),
            gameweek=GameweekId(6),
            generated_at=NOW,
            run_id="t",
            rng=random.Random(1),
            policy_name="model",
        )
        assert result.package.is_hold, "a gain inside the noise is not worth a transfer"

    def test_controls_act_even_when_the_model_would_hold(self) -> None:
        """That difference is what the backtest measures."""
        squad = _squad()
        predictions = {p.player_code: _prediction(int(p.player_code), 5.0) for p in squad.picks}
        marginal = _prediction(200, 5.01)
        predictions[PlayerCode(200)] = marginal
        candidates = [
            Candidate(
                player_code=PlayerCode(200),
                position=Position.MID,
                team_id=TeamId(20),
                price=TenthsOfMillion(60),
                prediction=marginal,
            )
        ]
        for name in ("highest_form", "most_expensive", "random"):
            result = run_policy(
                squad,
                selector=POLICIES[name],
                candidates=candidates,
                predictions=predictions,
                rules=_rules(),
                entry_id=EntryId(1),
                gameweek=GameweekId(6),
                generated_at=NOW,
                run_id="t",
                rng=random.Random(1),
                policy_name=name,
            )
            assert not result.package.is_hold, f"{name} should transfer indiscriminately"


class TestAccountingHolds:
    def test_bank_reconciles_after_every_policy(self) -> None:
        squad, _, _ = _setup()
        for name in POLICIES:
            result = _run(name)
            package = result.package  # type: ignore[attr-defined]
            if package.is_hold:
                assert package.bank_after == squad.bank
            else:
                assert package.bank_after == squad.bank - package.moves[0].net_cost

    @pytest.mark.parametrize("name", ["model", "highest_form", "most_expensive", "random"])
    def test_free_transfer_means_no_hit(self, name: str) -> None:
        assert _run(name).package.hit_cost == 0  # type: ignore[attr-defined]
