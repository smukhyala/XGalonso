"""Tests for the four frozen contracts.

These are the invariants that stop a whole class of bug from reaching the rest
of the system, so each one is tested for the failure it exists to prevent —
not merely for the happy path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from xg_alonso.contracts import (
    BaselineComparison,
    ComponentExpectations,
    EntryId,
    GameweekId,
    MinutesPrediction,
    PlayerCode,
    PlayerPrediction,
    PointsBreakdown,
    Position,
    PredictionProvenance,
    Reason,
    ReasonCode,
    ReasonPolarity,
    RunManifest,
    SourceTimestamps,
    SquadPick,
    SquadState,
    TeamId,
    TenthsOfMillion,
    TimeSource,
    TransferMove,
    TransferPackage,
    TransferRecommendation,
    format_money,
    parse_season,
    walk_forward_folds,
)

NOW = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Identifiers and money
# ---------------------------------------------------------------------------


class TestMoneyAndSeasons:
    def test_money_formats_as_millions(self) -> None:
        assert format_money(TenthsOfMillion(55)) == "£5.5m"
        assert format_money(TenthsOfMillion(1000)) == "£100.0m"

    def test_valid_season_parses(self) -> None:
        assert parse_season("2025-26") == "2025-26"
        assert parse_season("2099-00") == "2099-00"

    @pytest.mark.parametrize("bad", ["2025", "2025-2026", "25-26", "2025-27", "abcd-ef"])
    def test_malformed_season_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="season"):
            parse_season(bad)


# ---------------------------------------------------------------------------
# Contract 1 — the walk-forward fold. Random splits must be impossible.
# ---------------------------------------------------------------------------


class TestWalkForwardFolds:
    def test_folds_are_strictly_forward_in_time(self) -> None:
        folds = walk_forward_folds(
            gameweeks=[GameweekId(i) for i in range(1, 11)],
            min_train_gameweeks=5,
        )
        assert folds
        for fold in folds:
            assert fold.validate_start > fold.train_end, (
                "validation must lie strictly after training"
            )

    def test_shuffled_input_is_rejected(self) -> None:
        """An unsorted sequence is the usual symptom of a shuffled split."""
        shuffled = [GameweekId(i) for i in [3, 1, 5, 2, 4, 8, 6, 7]]
        with pytest.raises(ValueError, match="ascending"):
            walk_forward_folds(gameweeks=shuffled, min_train_gameweeks=3)

    def test_duplicate_gameweeks_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            walk_forward_folds(
                gameweeks=[GameweekId(i) for i in [1, 2, 2, 3, 4, 5]],
                min_train_gameweeks=2,
            )

    def test_embargo_creates_a_real_gap(self) -> None:
        folds = walk_forward_folds(
            gameweeks=[GameweekId(i) for i in range(1, 21)],
            min_train_gameweeks=5,
            embargo_gameweeks=2,
        )
        for fold in folds:
            gap = fold.validate_start - fold.train_end - 1
            assert gap == 2, f"expected a 2-gameweek embargo, got {gap}"

    def test_expanding_window_grows_and_sliding_does_not(self) -> None:
        gws = [GameweekId(i) for i in range(1, 16)]
        expanding = walk_forward_folds(gameweeks=gws, min_train_gameweeks=5, expanding=True)
        sliding = walk_forward_folds(gameweeks=gws, min_train_gameweeks=5, expanding=False)

        assert all(f.train_start == 1 for f in expanding)
        widths = {f.train_end - f.train_start for f in sliding}
        assert widths == {4}, "a sliding window keeps a constant width"

    def test_too_few_gameweeks_raises(self) -> None:
        with pytest.raises(ValueError, match="need at least"):
            walk_forward_folds(gameweeks=[GameweekId(1), GameweekId(2)], min_train_gameweeks=5)


# ---------------------------------------------------------------------------
# Contract 2 — reason codes. Prose may never reference absent evidence.
# ---------------------------------------------------------------------------


class TestReasonGrounding:
    def test_reason_renders_from_its_evidence(self) -> None:
        reason = Reason(
            code=ReasonCode.EXPECTED_MINUTES_SECURE,
            polarity=ReasonPolarity.SUPPORTS_IN,
            subject=PlayerCode(154561),
            evidence={"p_start": 0.92, "expected_minutes": 84.0},
            weight=1.4,
        )
        rendered = reason.render()
        assert "92%" in rendered
        assert "84" in rendered

    def test_missing_evidence_is_rejected_at_construction(self) -> None:
        """The no-fabrication guarantee: an ungroundable reason cannot exist."""
        with pytest.raises(ValidationError, match="expected_minutes"):
            Reason(
                code=ReasonCode.EXPECTED_MINUTES_SECURE,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=PlayerCode(154561),
                evidence={"p_start": 0.92},  # expected_minutes absent
                weight=1.4,
            )

    def test_every_reason_code_has_a_template(self) -> None:
        from xg_alonso.contracts import REASON_TEMPLATES

        missing = [c for c in ReasonCode if c not in REASON_TEMPLATES]
        assert not missing, f"reason codes without templates: {missing}"


# ---------------------------------------------------------------------------
# Contract 3 — the prediction schema.
# ---------------------------------------------------------------------------


def _provenance() -> PredictionProvenance:
    return PredictionProvenance(
        model_name="naive_pp90",
        model_version="1",
        model_artifact_sha256="a" * 64,
        feature_set_name="slice1",
        feature_set_version="1",
        data_cutoff=NOW,
        predicted_at=NOW,
        run_id="run-1",
        code_version="deadbeef",
    )


def _components(**overrides: float) -> ComponentExpectations:
    base: dict[str, object] = {
        "minutes": MinutesPrediction(
            p_appearance=0.95,
            p_start=0.9,
            expected_minutes=80.0,
            p_60_plus=0.85,
            minutes_sd=12.0,
        ),
        "goals": 0.4,
        "assists": 0.2,
        "clean_sheet_probability": 0.3,
        "goals_conceded": 1.2,
        "saves": 0.0,
        "yellow_cards": 0.15,
        "red_cards": 0.01,
        "own_goals": 0.0,
        "penalties_saved": 0.0,
        "penalties_missed": 0.02,
        "defensive_contribution_probability": 0.25,
        "bonus": 0.6,
    }
    base.update(overrides)
    return ComponentExpectations(**base)  # type: ignore[arg-type]


class TestPredictionSchema:
    def test_p60_cannot_exceed_p_appearance(self) -> None:
        """A player cannot reach 60 minutes without playing at all."""
        with pytest.raises(ValidationError, match="p_60_plus"):
            MinutesPrediction(
                p_appearance=0.5,
                p_start=0.4,
                expected_minutes=70.0,
                p_60_plus=0.9,
                minutes_sd=10.0,
            )

    def test_starting_cannot_exceed_appearing(self) -> None:
        with pytest.raises(ValidationError, match="p_start"):
            MinutesPrediction(
                p_appearance=0.3,
                p_start=0.8,
                expected_minutes=70.0,
                p_60_plus=0.2,
                minutes_sd=10.0,
            )

    def test_a_substitute_may_still_reach_sixty_minutes(self) -> None:
        """An early substitution can pass 60 minutes, so p_start must not bound p_60_plus."""
        prediction = MinutesPrediction(
            p_appearance=0.9,
            p_start=0.1,
            expected_minutes=45.0,
            p_60_plus=0.3,
            minutes_sd=20.0,
        )
        assert prediction.p_60_plus > prediction.p_start

    def test_breakdown_must_reconcile_with_total(self) -> None:
        """An explanation must never disagree with the number it explains."""
        breakdown = PointsBreakdown(
            appearance=2.0,
            goals=2.0,
            assists=0.6,
            clean_sheets=0.3,
            goals_conceded=-0.2,
            saves=0.0,
            cards=-0.2,
            own_goals=0.0,
            penalties=0.0,
            defensive_contribution=0.5,
            bonus=0.6,
        )
        with pytest.raises(ValidationError, match="breakdown sums"):
            PlayerPrediction(
                player_code=PlayerCode(1),
                position=Position.MID,
                from_gameweek=GameweekId(1),
                horizon_gameweeks=1,
                components=_components(),
                breakdown=breakdown,
                expected_points=99.0,  # deliberately inconsistent
                expected_points_sd=1.0,
                scoring_rules_version="2026-27",
                provenance=_provenance(),
            )

    def test_consistent_prediction_is_accepted(self) -> None:
        breakdown = PointsBreakdown(
            appearance=2.0,
            goals=2.0,
            assists=0.6,
            clean_sheets=0.3,
            goals_conceded=-0.2,
            saves=0.0,
            cards=-0.2,
            own_goals=0.0,
            penalties=0.0,
            defensive_contribution=0.5,
            bonus=0.6,
        )
        prediction = PlayerPrediction(
            player_code=PlayerCode(1),
            position=Position.MID,
            from_gameweek=GameweekId(1),
            horizon_gameweeks=1,
            components=_components(),
            breakdown=breakdown,
            expected_points=breakdown.total,
            expected_points_sd=1.0,
            scoring_rules_version="2026-27",
            provenance=_provenance(),
        )
        assert prediction.expected_points == pytest.approx(5.6)


# ---------------------------------------------------------------------------
# Provenance — every field required, naive datetimes rejected.
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            SourceTimestamps(
                event_time=datetime(2026, 8, 21, 17, 30),  # noqa: DTZ001 - deliberately naive
                observed_time=NOW,
                available_time=NOW,
                processed_time=NOW,
                time_source=TimeSource.DEADLINE,
            )

    def test_availability_gate(self) -> None:
        ts = SourceTimestamps(
            event_time=NOW,
            observed_time=NOW,
            available_time=NOW,
            processed_time=NOW,
            time_source=TimeSource.DEADLINE,
        )
        assert ts.is_available_at(NOW)
        assert ts.is_available_at(NOW + timedelta(seconds=1))
        assert not ts.is_available_at(NOW - timedelta(seconds=1))

    def test_dirty_run_is_not_promotable(self) -> None:
        """A run with uncommitted changes cannot be reproduced from its commit."""
        dirty = RunManifest(
            run_id="r1",
            pipeline="ingest.bootstrap",
            started_at=NOW,
            git_commit="a" * 40,
            git_dirty=True,
            config_hash="c" * 16,
        )
        assert not dirty.promotable

    def test_missing_provenance_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionProvenance(  # type: ignore[call-arg]
                model_name="m",
                model_version="1",
                feature_set_name="f",
                feature_set_version="1",
                data_cutoff=NOW,
                predicted_at=NOW,
                run_id="r",
                code_version="v",
            )


# ---------------------------------------------------------------------------
# Squad and recommendation accounting.
# ---------------------------------------------------------------------------


def _pick(slot: int, code: int, *, purchase: int = 50, current: int = 50) -> SquadPick:
    return SquadPick(
        player_code=PlayerCode(code),
        position=Position.MID,
        team_id=TeamId(1 + (code % 20)),
        purchase_price=TenthsOfMillion(purchase),
        current_price=TenthsOfMillion(current),
        selling_price=TenthsOfMillion(min(purchase, current)),
        squad_slot=slot,
    )


def _squad(**overrides: object) -> SquadState:
    base: dict[str, object] = {
        "entry_id": EntryId(1),
        "gameweek": GameweekId(1),
        "picks": tuple(_pick(i, 1000 + i) for i in range(1, 16)),
        "bank": TenthsOfMillion(5),
        "free_transfers": 1,
    }
    base.update(overrides)
    return SquadState(**base)  # type: ignore[arg-type]


class TestSquadState:
    def test_valid_squad_reports_value_and_split(self) -> None:
        squad = _squad()
        assert len(squad.starters) == 11
        assert len(squad.bench) == 4
        assert squad.squad_value == 15 * 50 + 5

    def test_wrong_size_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exactly 15"):
            _squad(picks=tuple(_pick(i, 1000 + i) for i in range(1, 15)))

    def test_duplicate_player_rejected(self) -> None:
        picks = [_pick(i, 1000 + i) for i in range(1, 15)]
        picks.append(_pick(15, 1001))  # duplicate of slot 1
        with pytest.raises(ValidationError, match="same player twice"):
            _squad(picks=tuple(picks))

    def test_selling_price_outside_bounds_rejected(self) -> None:
        """A manager never realises more than the current price."""
        with pytest.raises(ValidationError, match="selling_price"):
            SquadPick(
                player_code=PlayerCode(1),
                position=Position.FWD,
                team_id=TeamId(1),
                purchase_price=TenthsOfMillion(50),
                current_price=TenthsOfMillion(55),
                selling_price=TenthsOfMillion(60),
                squad_slot=1,
            )


class TestTransferAccounting:
    def test_hit_cost_must_match_paid_transfers(self) -> None:
        move = TransferMove(
            player_out=PlayerCode(1),
            player_in=PlayerCode(2),
            selling_price=TenthsOfMillion(50),
            purchase_price=TenthsOfMillion(55),
        )
        with pytest.raises(ValidationError, match="hit_cost"):
            TransferPackage(
                moves=(move, move),
                transfers_used=2,
                free_transfers_available=1,
                hit_cost=0,  # one transfer is paid, so this should be 4
                bank_before=TenthsOfMillion(10),
                bank_after=TenthsOfMillion(0),
            )

    def test_bank_must_reconcile(self) -> None:
        move = TransferMove(
            player_out=PlayerCode(1),
            player_in=PlayerCode(2),
            selling_price=TenthsOfMillion(50),
            purchase_price=TenthsOfMillion(55),
        )
        with pytest.raises(ValidationError, match="bank does not reconcile"):
            TransferPackage(
                moves=(move,),
                transfers_used=1,
                free_transfers_available=1,
                hit_cost=0,
                bank_before=TenthsOfMillion(10),
                bank_after=TenthsOfMillion(10),  # should be 5
            )

    def test_transfer_requires_a_grounded_reason(self) -> None:
        move = TransferMove(
            player_out=PlayerCode(1),
            player_in=PlayerCode(2),
            selling_price=TenthsOfMillion(50),
            purchase_price=TenthsOfMillion(50),
        )
        package = TransferPackage(
            moves=(move,),
            transfers_used=1,
            free_transfers_available=1,
            hit_cost=0,
            bank_before=TenthsOfMillion(10),
            bank_after=TenthsOfMillion(10),
        )
        comparison = BaselineComparison(
            baseline_name="hold",
            baseline_expected_points=40.0,
            candidate_expected_points=42.0,
            horizon_gameweeks=1,
        )
        with pytest.raises(ValidationError, match="grounded reason"):
            TransferRecommendation(
                entry_id=EntryId(1),
                gameweek=GameweekId(1),
                package=package,
                comparison=comparison,
                reasons=(),
                expected_points_gain=2.0,
                risk_score=0.5,
                generated_at=NOW,
                run_id="r",
                optimizer_config_hash="h",
            )

    def test_hold_needs_no_reason(self) -> None:
        package = TransferPackage(
            moves=(),
            transfers_used=0,
            free_transfers_available=1,
            hit_cost=0,
            bank_before=TenthsOfMillion(10),
            bank_after=TenthsOfMillion(10),
        )
        rec = TransferRecommendation(
            entry_id=EntryId(1),
            gameweek=GameweekId(1),
            package=package,
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
            run_id="r",
            optimizer_config_hash="h",
        )
        assert rec.package.is_hold
        assert not rec.comparison.beats_baseline
