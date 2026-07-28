"""External form signal tests.

This is the one channel in the system where a human sentence reaches a
projection, so the tests are mostly about what a signal is *not* allowed to do.
An unbounded text input into a prediction is precisely the arrangement the
reason-code module exists to prevent, and the value of the feature depends
entirely on the bound holding.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from xg_alonso.contracts.form import (
    FORM_SIGNAL_CLAMP,
    FormDirection,
    FormSignal,
    FormStrength,
    SignalSet,
)
from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.prediction.form import apply_form_signals, form_reason, load_signals

NOW = datetime(2026, 8, 1, tzinfo=UTC)
SOURCE = "https://example.com/report"


def _signal(
    *,
    code: int = 1,
    direction: FormDirection = FormDirection.NEGATIVE,
    strength: FormStrength = FormStrength.CLEAR,
    observed: datetime = NOW,
    expires: datetime | None = None,
) -> FormSignal:
    return FormSignal(
        player_code=PlayerCode(code),
        direction=direction,
        strength=strength,
        summary="Struggled at the World Cup and lost his place in the side.",
        sources=(SOURCE,),
        observed_at=observed,
        expires_at=expires or observed + timedelta(days=30),
    )


def _prediction(code: int = 1, points: float = 5.0) -> PlayerPrediction:
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
        position=Position.FWD,
        from_gameweek=1,
        horizon_gameweeks=1,
        components=ComponentExpectations(
            minutes=MinutesPrediction(
                p_appearance=1.0,
                p_start=0.9,
                expected_minutes=85.0,
                p_60_plus=0.8,
                minutes_sd=8.0,
            ),
            goals=0.4,
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
        breakdown=breakdown,
        expected_points=points,
        expected_points_sd=1.0,
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
    )


class TestBounds:
    def test_no_signal_can_exceed_the_clamp(self) -> None:
        """The bound is the whole reason this channel is allowed to exist."""
        for direction in FormDirection:
            for strength in FormStrength:
                multiplier = _signal(direction=direction, strength=strength).multiplier
                assert 1.0 - FORM_SIGNAL_CLAMP <= multiplier <= 1.0 + FORM_SIGNAL_CLAMP

    def test_direction_decides_the_sign(self) -> None:
        assert _signal(direction=FormDirection.NEGATIVE).multiplier < 1.0
        assert _signal(direction=FormDirection.POSITIVE).multiplier > 1.0

    def test_strength_orders_the_magnitudes(self) -> None:
        slight = _signal(strength=FormStrength.SLIGHT).multiplier
        clear = _signal(strength=FormStrength.CLEAR).multiplier
        strong = _signal(strength=FormStrength.STRONG).multiplier
        assert strong < clear < slight < 1.0


class TestProvenance:
    def test_a_signal_without_a_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1 item"):
            FormSignal(
                player_code=PlayerCode(1),
                direction=FormDirection.NEGATIVE,
                strength=FormStrength.CLEAR,
                summary="He looked out of sorts, apparently.",
                sources=(),
                observed_at=NOW,
                expires_at=NOW + timedelta(days=7),
            )

    def test_a_source_that_is_not_a_url_is_refused(self) -> None:
        """A claim nobody can check is a rumour, and rumours do not move numbers."""
        with pytest.raises(ValueError, match="not a URL"):
            FormSignal(
                player_code=PlayerCode(1),
                direction=FormDirection.NEGATIVE,
                strength=FormStrength.CLEAR,
                summary="Struggled badly in the tournament by all accounts.",
                sources=("someone told me",),
                observed_at=NOW,
                expires_at=NOW + timedelta(days=7),
            )

    def test_a_signal_must_expire_after_it_was_observed(self) -> None:
        with pytest.raises(ValueError, match="expire after"):
            _signal(expires=NOW - timedelta(days=1))


class TestExpiry:
    def test_a_live_signal_applies(self) -> None:
        live = SignalSet(signals=(_signal(),)).live(NOW + timedelta(days=1))
        assert PlayerCode(1) in live

    def test_an_expired_signal_is_dropped(self) -> None:
        """Stale form information is worse than none: it still looks current."""
        live = SignalSet(signals=(_signal(),)).live(NOW + timedelta(days=90))
        assert live == {}

    def test_a_future_signal_is_not_applied_early(self) -> None:
        live = SignalSet(signals=(_signal(),)).live(NOW - timedelta(days=1))
        assert live == {}

    def test_the_strongest_signal_wins_rather_than_compounding(self) -> None:
        """Two reports of one slump are not twice the evidence."""
        pair = SignalSet(
            signals=(
                _signal(strength=FormStrength.SLIGHT),
                _signal(strength=FormStrength.STRONG),
            )
        )
        live = pair.live(NOW + timedelta(days=1))
        assert live[PlayerCode(1)].strength is FormStrength.STRONG


class TestApplication:
    def test_a_negative_signal_lowers_the_projection(self) -> None:
        adjusted = apply_form_signals(
            [_prediction(points=5.0)],
            SignalSet(signals=(_signal(),)),
            at=NOW + timedelta(days=1),
        )
        assert adjusted[0].expected_points < 5.0

    def test_the_breakdown_still_sums_to_the_total(self) -> None:
        """The contract enforces this, so a partial rescale would not construct."""
        adjusted = apply_form_signals(
            [_prediction()], SignalSet(signals=(_signal(),)), at=NOW + timedelta(days=1)
        )
        assert adjusted[0].breakdown.total == pytest.approx(adjusted[0].expected_points)

    def test_acting_on_outside_information_widens_uncertainty(self) -> None:
        """A signal that made a projection more confident would be the wrong shape."""
        before = _prediction()
        after = apply_form_signals(
            [before], SignalSet(signals=(_signal(),)), at=NOW + timedelta(days=1)
        )[0]
        assert after.expected_points_sd > before.expected_points_sd

    def test_players_without_a_signal_are_untouched(self) -> None:
        original = _prediction(code=99)
        adjusted = apply_form_signals(
            [original], SignalSet(signals=(_signal(code=1),)), at=NOW + timedelta(days=1)
        )
        assert adjusted[0] == original

    def test_no_signals_changes_nothing(self) -> None:
        original = [_prediction()]
        assert apply_form_signals(original, SignalSet(), at=NOW) == original


class TestReason:
    def test_the_reason_cites_the_source_and_the_shift(self) -> None:
        rendered = form_reason(_signal(), PlayerCode(1), weight=1.0).render()
        assert SOURCE in rendered
        assert "World Cup" in rendered
        assert "10%" in rendered

    def test_a_negative_signal_argues_for_selling(self) -> None:
        assert form_reason(_signal(), PlayerCode(1), weight=1.0).polarity.value == "supports_out"


class TestLoading:
    def test_a_missing_file_yields_no_signals(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Outside information is optional; its absence must not change behaviour."""
        assert load_signals(tmp_path / "nothing.json").signals == ()

    def test_signals_round_trip_through_json(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "form_signals.json"
        path.write_text(json.dumps({"signals": [json.loads(_signal().model_dump_json())]}))
        loaded = load_signals(path)
        assert len(loaded.signals) == 1
        assert loaded.signals[0].sources == (SOURCE,)

    def test_a_malformed_signal_is_rejected_rather_than_skipped(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Silently dropping a bad signal would hide that the file is wrong."""
        path = tmp_path / "form_signals.json"
        path.write_text(json.dumps({"signals": [{"player_code": 1}]}))
        with pytest.raises(ValueError, match="validation error"):
            load_signals(path)
