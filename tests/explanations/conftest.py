"""Shared fixtures for explanation tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tests.conftest import BOOTSTRAP_FIXTURE
from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.domain.scoring import ScoringRules, assemble_points

NOW = datetime(2026, 8, 1, tzinfo=UTC)


RULES = ScoringRules.from_bootstrap(
    json.loads(BOOTSTRAP_FIXTURE.read_text()),
    version="2026-27",
    source_sha256="f" * 64,
    fetched_at=NOW,
)


@pytest.fixture(scope="session")
def rules() -> ScoringRules:
    """The pinned scoring snapshot, shared by every explanation test."""
    return RULES


@pytest.fixture(scope="session")
def prediction_factory():  # type: ignore[no-untyped-def]
    """A builder, so a test can vary position and rates without re-plumbing."""
    return make_prediction


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
