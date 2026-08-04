"""Price-band calibration tests.

The correction exists because the model under-projects expensive players more
than cheap ones, which makes an optimizer bank money rather than upgrade. Its
danger is the mirror image: a correction applied where the measurement does not
support one would move a whole tier on the strength of a few dozen matches.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from tests.conftest import BOOTSTRAP_FIXTURE

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TenthsOfMillion
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.domain.scoring import ScoringRules, assemble_points
from xg_alonso.prediction.calibration import (
    PRICE_BAND_BIAS,
    apply_price_calibration,
    correction_for,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
RULES = ScoringRules.from_bootstrap(
    json.loads(BOOTSTRAP_FIXTURE.read_text()),
    version="2026-27",
    source_sha256="f" * 64,
    fetched_at=NOW,
)


def make_prediction(
    code: int = 1,
    position: Position = Position.MID,
    *,
    p_start: float = 0.9,
    goals: float = 0.22,
    assists: float = 0.14,
) -> PlayerPrediction:
    """A realistic prediction, assembled through the real scoring rules.

    Built with `assemble_points` rather than hand-written totals so the tests
    exercise the same conversion the product uses — a hand-built breakdown would
    make the reconciliation test pass by construction.
    """
    minutes = MinutesPrediction(
        p_appearance=min(1.0, p_start + 0.06),
        p_start=p_start,
        expected_minutes=88.0 * p_start,
        p_60_plus=p_start * 0.92,
        minutes_sd=9.0,
    )
    components = ComponentExpectations(
        minutes=minutes,
        goals=goals,
        assists=assists,
        clean_sheet_probability=0.28 if position in (Position.GKP, Position.DEF) else 0.05,
        goals_conceded=1.2,
        saves=2.6 if position is Position.GKP else 0.0,
        yellow_cards=0.12,
        red_cards=0.0,
        own_goals=0.0,
        penalties_saved=0.0,
        penalties_missed=0.0,
        defensive_contribution_probability=0.3,
        bonus=0.45,
    )
    breakdown = assemble_points(components, position, RULES)
    return PlayerPrediction(
        player_code=PlayerCode(code),
        position=position,
        from_gameweek=GameweekId(1),
        horizon_gameweeks=1,
        components=components,
        breakdown=breakdown,
        expected_points=breakdown.total,
        expected_points_sd=1.1,
        scoring_rules_version=RULES.version,
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


class TestBands:
    def test_every_price_falls_in_exactly_one_band(self) -> None:
        for price in (40, 54, 55, 79, 80, 109, 110, 155):
            assert correction_for(price) is not None

    def test_bands_do_not_overlap(self) -> None:
        for price in (40, 60, 90, 130):
            matches = [b for b in PRICE_BAND_BIAS if b.lower <= price < b.upper]
            assert len(matches) == 1


class TestEvidenceThreshold:
    def test_a_thin_sample_is_not_acted_on(self) -> None:
        """The elite band's bias is measured on 64 rows and has the wrong sign.

        Correcting by it would be fitting noise; the honest reading of 64
        observations is that we do not know.
        """
        elite = next(b for b in PRICE_BAND_BIAS if b.label == "elite")
        assert not elite.is_trustworthy
        assert elite.adjustment == 0.0

    def test_a_well_measured_band_is_corrected(self) -> None:
        premium = next(b for b in PRICE_BAND_BIAS if b.label == "premium")
        assert premium.is_trustworthy
        assert premium.adjustment == pytest.approx(0.351)

    def test_over_projection_is_never_inflated_further(self) -> None:
        """The correction only ever adds back what was measured as missing."""
        for band in PRICE_BAND_BIAS:
            if band.bias >= 0:
                assert band.adjustment == 0.0


class TestUncertainty:
    def test_a_band_the_model_cannot_order_is_widened(self) -> None:
        """Rank correlation of 0.31 means it can barely tell elites apart."""
        elite = next(b for b in PRICE_BAND_BIAS if b.label == "elite")
        assert elite.uncertainty_multiplier > 1.0

    def test_a_well_ordered_band_is_left_alone(self) -> None:
        budget = next(b for b in PRICE_BAND_BIAS if b.label == "budget")
        assert budget.uncertainty_multiplier == 1.0


class TestApplication:
    def test_a_premium_gains_more_than_a_budget_player(self) -> None:
        """The whole point: the gap between the tiers was the error."""
        cheap = make_prediction(code=1)
        dear = make_prediction(code=2)
        prices = {PlayerCode(1): TenthsOfMillion(45), PlayerCode(2): TenthsOfMillion(95)}

        out = apply_price_calibration([cheap, dear], prices)
        cheap_gain = out[0].expected_points - cheap.expected_points
        dear_gain = out[1].expected_points - dear.expected_points
        assert dear_gain > cheap_gain

    def test_the_breakdown_still_sums_to_the_total(self) -> None:
        prediction = make_prediction(code=1)
        out = apply_price_calibration([prediction], {PlayerCode(1): TenthsOfMillion(95)})[0]
        assert out.breakdown.total == pytest.approx(out.expected_points)

    def test_an_unpriced_player_is_untouched(self) -> None:
        prediction = make_prediction(code=1)
        assert apply_price_calibration([prediction], {})[0] == prediction

    def test_an_elite_player_gains_nothing_but_widens(self) -> None:
        prediction = make_prediction(code=1)
        out = apply_price_calibration([prediction], {PlayerCode(1): TenthsOfMillion(130)})[0]
        assert out.expected_points == pytest.approx(prediction.expected_points)
        assert out.expected_points_sd > prediction.expected_points_sd
