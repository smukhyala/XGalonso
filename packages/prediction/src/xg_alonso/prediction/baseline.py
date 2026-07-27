"""The slice-1 expected-points baseline.

Deliberately closed-form. It needs no scikit-learn, no training run and no
stored artifact, which means the vertical slice can prove every contract
boundary — data, features, prediction, domain, optimization, explanation — before
any modelling question is opened. Swapping in a trained model later changes this
module and nothing else.

The shape follows decision D8. Rather than predicting points, it predicts
**components**, then hands them to :func:`~xg_alonso.domain.assemble_points`,
which is the only code that knows what a component is worth. So this estimator
never encodes a scoring rule, and a rules change re-scores it for free.

Its accuracy is not the point. Its job is to be defensible, reproducible and
honest about its own uncertainty, so that the optimizer above it can be judged
on whether it beats holding.
"""

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime
from typing import Final

import polars as pl

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.domain.scoring import ScoringRules, assemble_points

__all__ = [
    "BASELINE_NAME",
    "BASELINE_VERSION",
    "estimator_fingerprint",
    "predict_frame",
    "predict_player",
]

BASELINE_NAME: Final[str] = "naive_pp90"
BASELINE_VERSION: Final[str] = "1"

#: Minutes below which a player is treated as a rotation risk rather than a starter.
_FULL_MATCH: Final[float] = 90.0

#: League-average team xG per gameweek, used to centre the fixture adjustment.
#: Recomputed from the data when available; this is only the cold-start fallback.
_DEFAULT_TEAM_XG: Final[float] = 1.4

#: How far the fixture term may move a prediction. Capped because the underlying
#: signal is a five-match mean of a noisy quantity, and an uncapped multiplier
#: would let one blowout dominate a transfer recommendation.
_FIXTURE_CLAMP: Final[tuple[float, float]] = (0.8, 1.25)


def estimator_fingerprint() -> str:
    """SHA-256 of this module's source.

    A closed-form estimator has no artifact on disk, but provenance still
    requires an artifact hash — and making the field nullable would let a real
    model ship without one. Hashing the source keeps the field non-null and
    genuinely identifying: change the formula and the fingerprint changes.
    """
    source = inspect.getsource(inspect.getmodule(estimator_fingerprint) or __import__(__name__))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _expected_minutes(minutes_mean: float | None, start_rate: float | None) -> MinutesPrediction:
    """Turn recent volume and role into a minutes distribution.

    Minutes dominate FPL scoring, so this is the load-bearing estimate. A player
    with no history gets a low, uncertain prior rather than zero — zero would
    assert he cannot play, which is a stronger claim than "we do not know".
    """
    if minutes_mean is None:
        # Cold start: assume a fringe squad player, and say so with wide variance.
        return MinutesPrediction(
            p_appearance=0.35, p_start=0.15, expected_minutes=18.0, p_60_plus=0.12, minutes_sd=25.0
        )

    mean = max(0.0, min(float(minutes_mean), _FULL_MATCH))
    rate = 0.0 if start_rate is None else max(0.0, min(float(start_rate), 1.0))

    # Appearance probability rises with observed minutes; a player averaging a
    # full match essentially always appears.
    p_appearance = min(1.0, mean / 70.0) if mean < 70.0 else 1.0
    p_start = min(rate, p_appearance)
    # Reaching 60 minutes tracks starting, discounted for early substitutions.
    p_60 = min(p_appearance, p_start * 0.9 + max(0.0, (mean - 60.0) / 30.0) * 0.1)

    # Uncertainty peaks for genuine rotation risks and shrinks at both extremes.
    sd = 30.0 * (1.0 - abs(mean - 45.0) / 45.0) + 6.0

    return MinutesPrediction(
        p_appearance=round(p_appearance, 6),
        p_start=round(p_start, 6),
        expected_minutes=round(mean, 6),
        p_60_plus=round(max(0.0, min(p_60, p_appearance)), 6),
        minutes_sd=round(sd, 6),
    )


def _fixture_multiplier(team_xg: float | None, league_mean: float) -> float:
    """Scale attacking output by the team's recent expected goals.

    Built from xG rather than the API's ``strength_attack_*`` fields, which read
    0 for every club preseason and change scale mid-season.
    """
    if team_xg is None or league_mean <= 0:
        return 1.0
    raw = float(team_xg) / league_mean
    low, high = _FIXTURE_CLAMP
    return max(low, min(raw, high))


def predict_player(
    *,
    player_code: PlayerCode,
    position: Position,
    from_gameweek: GameweekId,
    features: dict[str, float | None],
    rules: ScoringRules,
    provenance: PredictionProvenance,
    horizon_gameweeks: int = 1,
    league_mean_team_xg: float = _DEFAULT_TEAM_XG,
) -> PlayerPrediction:
    """Predict one player's components and assemble them into expected points."""
    minutes = _expected_minutes(features.get("minutes_mean_5"), features.get("start_rate_5"))
    fixture = _fixture_multiplier(features.get("team_xg_for_mean_5"), league_mean_team_xg)

    # Per-90 rates scale to the minutes we actually expect, times the horizon.
    scale = (minutes.expected_minutes / _FULL_MATCH) * horizon_gameweeks

    def rate(name: str) -> float:
        value = features.get(name)
        return 0.0 if value is None else max(0.0, float(value))

    goals = rate("xg_per90_shrunk") * scale * fixture
    assists = rate("xa_per90_shrunk") * scale * fixture
    bonus = rate("bonus_per90_shrunk") * scale

    # Clean sheets and goals conceded are team properties. Without an opponent
    # model in slice 1, both come from league-average rates rather than being
    # invented per player — an honest constant beats a fabricated signal.
    clean_sheet = 0.28 * minutes.p_60_plus * horizon_gameweeks
    conceded = 1.35 * (minutes.expected_minutes / _FULL_MATCH) * horizon_gameweeks
    saves = (2.6 * scale) if position is Position.GKP else 0.0

    components = ComponentExpectations(
        minutes=minutes,
        goals=goals,
        assists=assists,
        clean_sheet_probability=min(1.0, clean_sheet),
        goals_conceded=conceded,
        saves=saves,
        yellow_cards=0.12 * scale,
        red_cards=0.004 * scale,
        own_goals=0.0,
        penalties_saved=0.0,
        penalties_missed=0.0,
        # No defensive-contribution model in slice 1: the statistic exists for a
        # single season, which is not enough to fit anything trustworthy. Left
        # at zero so the term is visibly absent rather than quietly guessed.
        defensive_contribution_probability=0.0,
        bonus=bonus,
    )

    breakdown = assemble_points(components, position, rules)

    # Uncertainty is driven by minutes, which dominate the variance of the total.
    minutes_share = minutes.minutes_sd / _FULL_MATCH
    sd = max(0.5, abs(breakdown.total) * minutes_share + 0.8)

    return PlayerPrediction(
        player_code=player_code,
        position=position,
        from_gameweek=from_gameweek,
        horizon_gameweeks=horizon_gameweeks,
        components=components,
        breakdown=breakdown,
        expected_points=breakdown.total,
        expected_points_sd=round(sd, 6),
        scoring_rules_version=rules.version,
        provenance=provenance,
    )


def predict_frame(
    features: pl.DataFrame,
    *,
    rules: ScoringRules,
    from_gameweek: GameweekId,
    data_cutoff: datetime,
    predicted_at: datetime,
    run_id: str,
    code_version: str,
    feature_set_version: str,
    horizon_gameweeks: int = 1,
) -> list[PlayerPrediction]:
    """Predict every row of a feature frame.

    Requires ``player_code`` and ``position``. The league-mean team xG used to
    centre the fixture term is computed from this frame when available, so the
    adjustment is relative to the league as observed rather than to a constant.
    """
    for required in ("player_code", "position"):
        if required not in features.columns:
            raise KeyError(f"feature frame must carry {required!r}")

    league_mean = _DEFAULT_TEAM_XG
    if "team_xg_for_mean_5" in features.columns:
        observed = features["team_xg_for_mean_5"].drop_nulls()
        if observed.len() > 0:
            mean_value = observed.mean()
            if isinstance(mean_value, (int, float)) and mean_value > 0:
                league_mean = float(mean_value)

    provenance = PredictionProvenance(
        model_name=BASELINE_NAME,
        model_version=BASELINE_VERSION,
        model_artifact_sha256=estimator_fingerprint(),
        feature_set_name="slice1",
        feature_set_version=feature_set_version,
        data_cutoff=data_cutoff,
        predicted_at=predicted_at,
        run_id=run_id,
        code_version=code_version,
    )

    predictions: list[PlayerPrediction] = []
    for row in features.iter_rows(named=True):
        position = row.get("position")
        if not isinstance(position, str) or position not in Position.__members__:
            continue
        predictions.append(
            predict_player(
                player_code=PlayerCode(int(row["player_code"])),
                position=Position(position),
                from_gameweek=from_gameweek,
                features={k: row.get(k) for k in row},
                rules=rules,
                provenance=provenance,
                horizon_gameweeks=horizon_gameweeks,
                league_mean_team_xg=league_mean,
            )
        )
    return predictions
