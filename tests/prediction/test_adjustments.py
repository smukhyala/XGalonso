"""The single post-processing path every surface shares.

Three surfaces once applied three different sets of adjustments to the same
prediction: ``cli.main._predict_all`` tempered by availability, the API
corrected the price-band bias, and ``cli.pipeline.recommend`` did neither. Same
team id, same deadline, three different projections — and no test anywhere
compared two surfaces against each other, which is why it survived.

These tests pin the sequence itself. The cross-surface agreement test is the
one that matters: it fails if any surface grows or loses an adjustment without
the others following.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from tests.conftest import REPO_ROOT

from xg_alonso.contracts.form import (
    FormDirection,
    FormSignal,
    FormStrength,
    SignalSet,
)
from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TenthsOfMillion
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.prediction.adjustments import ADJUSTMENT_ORDER, adjust_predictions

_CUTOFF = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)


def _prediction(code: int = 1, points: float = 6.0) -> PlayerPrediction:
    return PlayerPrediction(
        player_code=PlayerCode(code),
        position=Position.MID,
        from_gameweek=GameweekId(1),
        horizon_gameweeks=1,
        components=ComponentExpectations(
            minutes=MinutesPrediction(
                p_appearance=0.95,
                p_start=0.90,
                expected_minutes=80.0,
                p_60_plus=0.85,
                minutes_sd=10.0,
            ),
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
            model_name="test",
            model_version="1",
            model_artifact_sha256="b" * 64,
            feature_set_name="f",
            feature_set_version="1",
            data_cutoff=_CUTOFF,
            predicted_at=_CUTOFF,
            run_id="r",
            code_version="c",
        ),
    )


def _signals(code: int) -> SignalSet:
    return SignalSet(
        signals=(
            FormSignal(
                player_code=PlayerCode(code),
                direction=FormDirection.NEGATIVE,
                strength=FormStrength.CLEAR,
                summary="Struggled badly and lost his place.",
                sources=("https://example.com/report",),
                observed_at=datetime(2026, 8, 1, tzinfo=UTC),
                expires_at=datetime(2026, 9, 1, tzinfo=UTC),
            ),
        )
    )


class TestEachAdjustmentIsOptional:
    """Skipping one must be a caller's explicit choice, not an accident."""

    def test_no_arguments_returns_the_predictions_unchanged(self) -> None:
        raw = [_prediction()]
        assert adjust_predictions(raw) == raw

    def test_availability_alone_reduces_a_doubtful_player(self) -> None:
        raw = [_prediction(points=6.0)]
        out = adjust_predictions(raw, chances={PlayerCode(1): 25})
        assert out[0].expected_points < raw[0].expected_points

    def test_a_fully_fit_player_is_untouched_by_availability(self) -> None:
        raw = [_prediction(points=6.0)]
        out = adjust_predictions(raw, chances={PlayerCode(1): 100})
        assert out[0].expected_points == pytest.approx(raw[0].expected_points)

    def test_a_null_chance_means_fit_not_missing(self) -> None:
        """A null in the payload is FPL saying "no doubt", not "no data"."""
        raw = [_prediction(points=6.0)]
        out = adjust_predictions(raw, chances={PlayerCode(1): None})
        assert out[0].expected_points == pytest.approx(raw[0].expected_points)

    def test_signals_need_a_cutoff_to_apply(self) -> None:
        """Without an `at` there is no way to know a signal was live."""
        raw = [_prediction(points=6.0)]
        out = adjust_predictions(raw, signals=_signals(1))
        assert out[0].expected_points == pytest.approx(raw[0].expected_points)

    def test_a_live_negative_signal_lowers_the_projection(self) -> None:
        raw = [_prediction(points=6.0)]
        out = adjust_predictions(raw, signals=_signals(1), at=datetime(2026, 8, 15, tzinfo=UTC))
        assert out[0].expected_points < raw[0].expected_points


class TestTheSequenceIsFixed:
    def test_the_declared_order_is_the_applied_order(self) -> None:
        assert ADJUSTMENT_ORDER == ("price_calibration", "availability", "form_signals")

    def test_adjustments_compose_rather_than_overwrite(self) -> None:
        """Two live adjustments must both land, not the last one only."""
        raw = [_prediction(points=6.0)]
        availability_only = adjust_predictions(raw, chances={PlayerCode(1): 50})
        both = adjust_predictions(
            raw,
            chances={PlayerCode(1): 50},
            signals=_signals(1),
            at=datetime(2026, 8, 15, tzinfo=UTC),
        )
        assert both[0].expected_points < availability_only[0].expected_points

    def test_the_input_list_is_never_mutated(self) -> None:
        raw = [_prediction(points=6.0)]
        before = raw[0].expected_points
        adjust_predictions(raw, chances={PlayerCode(1): 25})
        assert raw[0].expected_points == before

    def test_the_breakdown_still_sums_to_the_total(self) -> None:
        """The contract's own invariant, after every adjustment has run.

        `PlayerPrediction` validates that `breakdown.total == expected_points`,
        so an adjustment that scaled the total without rescaling the breakdown
        would raise here rather than ship a prediction that cannot explain
        itself.
        """
        out = adjust_predictions(
            [_prediction(points=6.0)],
            prices={PlayerCode(1): TenthsOfMillion(120)},
            chances={PlayerCode(1): 50},
            signals=_signals(1),
            at=datetime(2026, 8, 15, tzinfo=UTC),
        )
        assert out[0].breakdown.total == pytest.approx(out[0].expected_points, abs=1e-6)


class TestEverySurfaceAppliesTheSameAdjustments:
    """The regression the three surfaces spent a release disagreeing over.

    Asserted against the source rather than by booting three stacks: the point
    is that no surface post-processes predictions on its own, which is a
    property of the call graph and is exactly what drifted before.
    """

    @staticmethod
    def _source(relative: str) -> str:
        return (REPO_ROOT / relative).read_text()

    @pytest.mark.parametrize(
        "module",
        [
            "apps/cli/src/xg_alonso/cli/pipeline.py",
            "apps/cli/src/xg_alonso/cli/main.py",
            "apps/api/src/xg_alonso/api/service.py",
        ],
    )
    def test_the_surface_routes_through_the_shared_path(self, module: str) -> None:
        assert "adjust_predictions" in self._source(module), (
            f"{module} does not call adjust_predictions, so it can drift away "
            "from the other surfaces without any test noticing"
        )

    @pytest.mark.parametrize(
        "module",
        [
            "apps/cli/src/xg_alonso/cli/pipeline.py",
            "apps/cli/src/xg_alonso/cli/main.py",
            "apps/api/src/xg_alonso/api/service.py",
        ],
    )
    def test_no_surface_applies_an_adjustment_directly(self, module: str) -> None:
        source = self._source(module)
        for leaked in ("apply_availability(", "apply_price_calibration(", "apply_form_signals("):
            assert leaked not in source, (
                f"{module} calls {leaked} directly. Adjustments belong in "
                "prediction.adjustments, or the surfaces disagree again."
            )
