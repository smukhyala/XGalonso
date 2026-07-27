"""Turn trained component models into predictions, and persist them.

The mapping from model outputs to :class:`ComponentExpectations` is where the
trained path rejoins the closed-form one: both produce components, and both
hand them to :func:`~xg_alonso.domain.assemble_points`. Nothing here knows what
a goal is worth, so swapping estimators never touches scoring.

**Persistence carries provenance.** A saved model records the seasons and
gameweeks it was fitted on, so a later run can check it was not trained on the
period it is about to be evaluated over — the mistake that makes a backtest
meaningless while looking entirely normal.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
from xg_alonso.prediction.trained import (
    TRAINED_MODEL_NAME,
    TRAINED_MODEL_VERSION,
    ComponentModels,
)

__all__ = [
    "SavedModel",
    "load_models",
    "predict_with_models",
    "save_models",
]

_FULL_MATCH = 90.0


@dataclass(frozen=True)
class SavedModel:
    """A persisted model plus the provenance needed to use it safely."""

    models: ComponentModels
    trained_seasons: tuple[str, ...]
    trained_gameweeks: tuple[int, ...]
    saved_at: datetime

    def overlaps(self, season: str, gameweeks: tuple[int, ...]) -> bool:
        """Whether this model was fitted on any of the given gameweeks.

        A backtest over a period the model trained on measures memorisation,
        not skill, and the result looks perfectly reasonable — so the check is
        mechanical rather than left to whoever runs it.
        """
        if season not in self.trained_seasons:
            return False
        return bool(set(gameweeks) & set(self.trained_gameweeks))


def save_models(saved: SavedModel, path: Path) -> None:
    """Persist a fitted model and its provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(saved, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_models(path: Path) -> SavedModel:
    """Load a persisted model.

    Raises:
        TypeError: if the file does not hold a :class:`SavedModel`. Unpickling
            arbitrary objects and hoping is not a loading strategy.
    """
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, SavedModel):
        raise TypeError(f"{path} does not contain a SavedModel (got {type(loaded).__name__})")
    return loaded


def _minutes_from(expected_minutes: float, p_start: float) -> MinutesPrediction:
    """Assemble a coherent minutes distribution from two model outputs.

    The models predict expected minutes and start probability independently, so
    they can disagree — a player can come back with 80 expected minutes and a
    0.2 start probability. The contract's invariants would reject that, so the
    quantities are reconciled here rather than at validation time.
    """
    mean = max(0.0, min(expected_minutes, _FULL_MATCH))
    start = max(0.0, min(p_start, 1.0))

    # Appearing is implied by the minutes actually expected, and by starting.
    p_appearance = max(start, min(1.0, mean / 70.0) if mean < 70.0 else 1.0)
    p_60 = min(p_appearance, start * 0.9 + max(0.0, (mean - 60.0) / 30.0) * 0.1)
    sd = 30.0 * (1.0 - abs(mean - 45.0) / 45.0) + 6.0

    return MinutesPrediction(
        p_appearance=round(p_appearance, 6),
        p_start=round(min(start, p_appearance), 6),
        expected_minutes=round(mean, 6),
        p_60_plus=round(max(0.0, p_60), 6),
        minutes_sd=round(sd, 6),
    )


def predict_with_models(
    features: pl.DataFrame,
    *,
    models: ComponentModels,
    rules: ScoringRules,
    from_gameweek: GameweekId,
    data_cutoff: datetime,
    predicted_at: datetime,
    run_id: str,
    code_version: str,
    feature_set_version: str,
    horizon_gameweeks: int = 1,
) -> list[PlayerPrediction]:
    """Predict components with the trained models and assemble them into points.

    Requires ``player_code`` and ``position`` on the feature frame. Rows whose
    position is unrecognised are skipped rather than guessed.
    """
    for required in ("player_code", "position"):
        if required not in features.columns:
            raise KeyError(f"feature frame must carry {required!r}")
    if features.is_empty():
        return []

    predicted = models.predict(features)

    def column(label: str, default: float = 0.0) -> list[float]:
        values = predicted.get(label)
        if values is None:
            return [default] * features.height
        return [float(v) for v in values]

    minutes = column("label_minutes")
    starts = column("label_starts")
    goals = column("label_goals_scored")
    assists = column("label_assists")
    clean_sheets = column("label_clean_sheets")
    conceded = column("label_goals_conceded")
    saves = column("label_saves")
    bonus = column("label_bonus")
    yellows = column("label_yellow_cards")

    provenance = PredictionProvenance(
        model_name=TRAINED_MODEL_NAME,
        model_version=TRAINED_MODEL_VERSION,
        model_artifact_sha256=models.fingerprint(),
        feature_set_name="catalogue",
        feature_set_version=feature_set_version,
        data_cutoff=data_cutoff,
        predicted_at=predicted_at,
        run_id=run_id,
        code_version=code_version,
    )

    out: list[PlayerPrediction] = []
    for index, row in enumerate(features.iter_rows(named=True)):
        position = row.get("position")
        if not isinstance(position, str) or position not in Position.__members__:
            continue

        minute_prediction = _minutes_from(minutes[index], starts[index])
        scale = horizon_gameweeks

        components = ComponentExpectations(
            minutes=minute_prediction,
            goals=max(0.0, goals[index]) * scale,
            assists=max(0.0, assists[index]) * scale,
            # The clean-sheet model predicts the event; FPL pays it only to
            # players who last 60 minutes, so the two are combined here to keep
            # the assembly a plain expectation.
            clean_sheet_probability=min(
                1.0, max(0.0, clean_sheets[index]) * minute_prediction.p_60_plus
            ),
            goals_conceded=max(0.0, conceded[index]) * scale,
            saves=max(0.0, saves[index]) * scale,
            yellow_cards=max(0.0, yellows[index]) * scale,
            red_cards=0.0,
            own_goals=0.0,
            penalties_saved=0.0,
            penalties_missed=0.0,
            # Not modelled: the statistic exists for one season only, which is
            # not enough to fit. Left visibly at zero rather than guessed.
            defensive_contribution_probability=0.0,
            bonus=max(0.0, bonus[index]) * scale,
        )

        breakdown = assemble_points(components, Position(position), rules)
        sd = max(
            0.5,
            abs(breakdown.total) * (minute_prediction.minutes_sd / _FULL_MATCH) + 0.8,
        )

        out.append(
            PlayerPrediction(
                player_code=PlayerCode(int(row["player_code"])),
                position=Position(position),
                from_gameweek=from_gameweek,
                horizon_gameweeks=horizon_gameweeks,
                components=components,
                breakdown=breakdown,
                expected_points=breakdown.total,
                expected_points_sd=round(sd, 6),
                scoring_rules_version=rules.version,
                provenance=provenance,
            )
        )
    return out


def model_summary(saved: SavedModel) -> dict[str, Any]:
    """A compact description of a saved model, for reporting."""
    return {
        "name": TRAINED_MODEL_NAME,
        "version": TRAINED_MODEL_VERSION,
        "fingerprint": saved.models.fingerprint()[:12],
        "rows": saved.models.trained_on_rows,
        "folds": len(saved.models.folds),
        "seasons": list(saved.trained_seasons),
        "labels": sorted(saved.models.models),
        "skill": saved.models.skill_by_label(),
    }
