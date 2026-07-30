"""Tests for applying FPL's published chance of playing.

The distinction under test is between this and a form signal. A form signal is
somebody's reading of a match report and is clamped to ±15% because it is
judgement. What FPL publishes here is the game's own statement about a player,
and clamping it would substitute our caution for their fact — a 25% chance
means a 75% reduction and is applied as one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.prediction.availability import (
    apply_availability,
    availability_factor,
)

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
            feature_set_version="test_v1",
            data_cutoff=_CUTOFF,
            predicted_at=_CUTOFF,
            run_id="test",
            code_version="test",
        ),
    )


class TestFactor:
    @pytest.mark.parametrize(
        ("chance", "expected"), [(100, 1.0), (75, 0.75), (50, 0.5), (25, 0.25), (0, 0.0)]
    )
    def test_a_percentage_becomes_a_factor(self, chance: int, expected: float) -> None:
        assert availability_factor(chance) == expected

    def test_an_unstated_chance_means_fully_fit(self) -> None:
        """FPL clears the field for a healthy player.

        Reading `None` as unknown would discount every fit player in the game.
        """
        assert availability_factor(None) == 1.0

    def test_a_nonsense_value_is_clamped_not_trusted(self) -> None:
        """A projection silently multiplied by 1.4 is not a failure anybody
        would notice."""
        assert availability_factor(140) == 1.0
        assert availability_factor(-10) == 0.0


class TestApplication:
    def test_a_seventy_five_percent_player_loses_a_quarter(self) -> None:
        """The exact gap: a doubtful player was scored as fully fit."""
        adjusted = apply_availability([_prediction(points=6.0)], {PlayerCode(1): 75})[0]
        assert adjusted.expected_points == pytest.approx(4.5)

    def test_it_is_not_clamped_like_a_form_signal(self) -> None:
        """A form signal moves ±15% because it is judgement. This is fact."""
        adjusted = apply_availability([_prediction(points=6.0)], {PlayerCode(1): 25})[0]
        assert adjusted.expected_points == pytest.approx(1.5)

    def test_a_zero_percent_player_projects_zero(self) -> None:
        adjusted = apply_availability([_prediction(points=6.0)], {PlayerCode(1): 0})[0]
        assert adjusted.expected_points == pytest.approx(0.0)

    def test_a_fully_fit_player_is_untouched(self) -> None:
        original = _prediction()
        assert apply_availability([original], {PlayerCode(1): 100})[0] == original

    def test_an_unlisted_player_is_untouched(self) -> None:
        """Absent means "not told", not "discount him"."""
        original = _prediction()
        assert apply_availability([original], {})[0] == original


class TestMinutes:
    def test_minutes_scale_with_the_probability(self) -> None:
        """He is the same footballer, less likely to be on the pitch."""
        adjusted = apply_availability([_prediction()], {PlayerCode(1): 50})[0]
        minutes = adjusted.components.minutes

        assert minutes.expected_minutes == pytest.approx(40.0)
        assert minutes.p_start == pytest.approx(0.45)
        assert minutes.p_appearance == pytest.approx(0.475)

    def test_the_contract_invariants_survive(self) -> None:
        """Starting implies appearing, and so does lasting an hour."""
        for chance in (75, 50, 25, 10):
            minutes = apply_availability([_prediction()], {PlayerCode(1): chance})[
                0
            ].components.minutes
            assert minutes.p_start <= minutes.p_appearance
            assert minutes.p_60_plus <= minutes.p_appearance

    def test_uncertainty_widens_rather_than_narrows(self) -> None:
        """A coin-flip on whether he plays is the widest outcome there is."""
        original = _prediction()
        adjusted = apply_availability([original], {PlayerCode(1): 50})[0]

        assert adjusted.expected_points_sd > original.expected_points_sd
        assert adjusted.components.minutes.minutes_sd > original.components.minutes.minutes_sd

    def test_the_widest_spread_is_at_the_coin_flip(self) -> None:
        spreads = {
            chance: apply_availability([_prediction()], {PlayerCode(1): chance})[
                0
            ].components.minutes.minutes_sd
            for chance in (10, 50, 90)
        }
        assert spreads[50] > spreads[10]
        assert spreads[50] > spreads[90]


class TestBreakdown:
    def test_the_breakdown_still_sums_to_the_total(self) -> None:
        """The contract rejects a breakdown that disagrees with its own total,
        which is what stops an explanation drifting from its number."""
        adjusted = apply_availability([_prediction(points=6.0)], {PlayerCode(1): 75})[0]
        assert adjusted.breakdown.total == pytest.approx(adjusted.expected_points)

    def test_negative_terms_scale_too(self) -> None:
        """Half as likely to appear is half as likely to be booked.

        Leaving cards untouched would make a doubtful player look *worse* than
        his availability implies rather than merely less valuable.
        """
        base = _prediction(points=6.0)
        booked = base.model_copy(
            update={
                "breakdown": base.breakdown.model_copy(update={"appearance": 7.0, "cards": -1.0}),
            }
        )
        adjusted = apply_availability([booked], {PlayerCode(1): 50})[0]

        assert adjusted.breakdown.cards == pytest.approx(-0.5)
        assert adjusted.breakdown.appearance == pytest.approx(3.5)
