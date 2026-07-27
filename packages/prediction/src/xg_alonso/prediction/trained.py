"""Walk-forward-trained component models.

Follows decision D8: nine small models predict *components* — minutes, starts,
goals, assists, clean sheets, goals conceded, saves, bonus, cards — and the
domain layer prices them. Nothing here knows what a goal is worth, so a scoring
change re-prices existing predictions without retraining anything.

**Training is strictly walk-forward.** Folds come from the frozen
:func:`~xg_alonso.contracts.folds.walk_forward_folds` constructor, which cannot
shuffle. A model is fitted only on gameweeks strictly before the ones it is
evaluated on, with an embargo between them so a rolling feature's lookback
window cannot straddle the boundary.

``HistGradientBoosting`` is used because it handles missing values natively.
That matters more than it sounds: a third of the catalogue is null for players
without enough history, and imputing those would invent a past the player did
not have.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from xg_alonso.contracts.folds import WalkForwardFold, walk_forward_folds
from xg_alonso.contracts.identifiers import GameweekId

__all__ = [
    "TRAINED_MODEL_NAME",
    "TRAINED_MODEL_VERSION",
    "ComponentModels",
    "FoldReport",
    "train_component_models",
]

TRAINED_MODEL_NAME: Final[str] = "component_hgb"
TRAINED_MODEL_VERSION: Final[str] = "1"

#: Labels modelled as probabilities rather than counts. Everything else is a
#: regression on a raw count.
_BINARY_LABELS: Final[frozenset[str]] = frozenset({"label_clean_sheets", "label_starts"})

#: Deliberately small trees. The dataset is tens of thousands of rows with
#: heavily correlated features, and a deeper model memorises player identity
#: rather than learning form.
_REGRESSOR_KWARGS: Final[dict[str, Any]] = {
    "max_iter": 200,
    "max_depth": 4,
    "learning_rate": 0.06,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 20260727,
}


@dataclass(frozen=True)
class FoldReport:
    """Out-of-sample error for one label on one fold."""

    label: str
    fold_index: int
    train_rows: int
    validate_rows: int
    mae: float
    baseline_mae: float

    @property
    def skill(self) -> float:
        """Fractional improvement over predicting the training mean.

        Negative means the model is worse than a constant, which is the only
        result that unambiguously says a label is not learnable from these
        features.
        """
        if self.baseline_mae <= 0:
            return 0.0
        return 1.0 - (self.mae / self.baseline_mae)


@dataclass
class ComponentModels:
    """Fitted component models, plus the evidence they were evaluated on."""

    models: dict[str, Any] = field(default_factory=dict)
    feature_columns: tuple[str, ...] = ()
    reports: list[FoldReport] = field(default_factory=list)
    trained_on_rows: int = 0
    folds: tuple[WalkForwardFold, ...] = ()

    def fingerprint(self) -> str:
        """A stable hash identifying this fitted artifact for provenance."""
        parts = [
            TRAINED_MODEL_NAME,
            TRAINED_MODEL_VERSION,
            ",".join(sorted(self.models)),
            ",".join(self.feature_columns),
            str(self.trained_on_rows),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def skill_by_label(self) -> dict[str, float]:
        """Mean out-of-sample skill per label, worst first when sorted."""
        by_label: dict[str, list[float]] = {}
        for report in self.reports:
            by_label.setdefault(report.label, []).append(report.skill)
        return {k: sum(v) / len(v) for k, v in by_label.items()}

    def predict(self, features: pl.DataFrame) -> dict[str, np.ndarray]:
        """Predict every component for a feature frame."""
        matrix = _matrix(features, self.feature_columns)
        out: dict[str, np.ndarray] = {}
        for label, model in self.models.items():
            if label in _BINARY_LABELS:
                out[label] = model.predict_proba(matrix)[:, 1]
            else:
                out[label] = np.clip(model.predict(matrix), 0.0, None)
        return out


def _matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Feature matrix with nulls preserved as NaN.

    HistGradientBoosting treats NaN as its own branch, so a player without
    history is modelled as *unknown* rather than as an invented average.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"feature frame is missing columns: {missing[:8]}")
    return frame.select(columns).to_numpy().astype(np.float64)


def train_component_models(
    training: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    min_train_gameweeks: int = 8,
    embargo_gameweeks: int = 1,
    validate_gameweeks: int = 4,
    model_kwargs: dict[str, Any] | None = None,
) -> ComponentModels:
    """Fit component models with walk-forward evaluation, then refit on all data.

    Folds establish honest out-of-sample error; the shipped models are then
    refitted on everything, because a model used for the *next* gameweek should
    have seen every gameweek before it.

    Args:
        training: Output of :func:`~xg_alonso.prediction.dataset.build_training_frame`.
        feature_columns: Which columns are inputs.
        label_columns: Which columns are targets.
        min_train_gameweeks: Minimum training window before the first fold.
        embargo_gameweeks: Gap between training and validation, so a rolling
            window cannot straddle the boundary.
        validate_gameweeks: Validation window length.
        model_kwargs: Overrides for the estimator hyperparameters. Behaviour
            that changes between experiments belongs in configuration, not in a
            constant — and it lets tests fit small, fast models.

    Raises:
        ValueError: if there is not enough history for a single fold. Training
            without out-of-sample evidence is not training, it is fitting.
    """
    if training.is_empty():
        raise ValueError("training frame is empty")

    # A globally increasing gameweek index across seasons, so folds never wrap
    # backwards over a season boundary.
    ordered_seasons = sorted(training["label_season"].unique().to_list())
    season_offset = {season: index * 100 for index, season in enumerate(ordered_seasons)}
    indexed = training.with_columns(
        (
            pl.col("label_season").replace_strict(season_offset, default=0)
            + pl.col("label_gameweek")
        ).alias("__timeline")
    )

    timeline = sorted({int(v) for v in indexed["__timeline"].unique()})
    folds = walk_forward_folds(
        gameweeks=[GameweekId(v) for v in timeline],
        min_train_gameweeks=min_train_gameweeks,
        validate_gameweeks=validate_gameweeks,
        embargo_gameweeks=embargo_gameweeks,
    )

    result = ComponentModels(feature_columns=feature_columns, folds=tuple(folds))

    for fold in folds:
        train_mask = indexed["__timeline"].is_between(fold.train_start, fold.train_end)
        validate_mask = indexed["__timeline"].is_between(fold.validate_start, fold.validate_end)
        train_rows = indexed.filter(train_mask)
        validate_rows = indexed.filter(validate_mask)
        if train_rows.is_empty() or validate_rows.is_empty():
            continue

        x_train = _matrix(train_rows, feature_columns)
        x_validate = _matrix(validate_rows, feature_columns)

        for label in label_columns:
            y_train = train_rows[label].to_numpy().astype(np.float64)
            y_validate = validate_rows[label].to_numpy().astype(np.float64)

            model = _fit(label, x_train, y_train, model_kwargs)
            if model is None:
                continue

            if label in _BINARY_LABELS:
                predicted = model.predict_proba(x_validate)[:, 1]
                truth = (y_validate > 0).astype(np.float64)
            else:
                predicted = np.clip(model.predict(x_validate), 0.0, None)
                truth = y_validate

            # Predicting the training mean is the bar any model must clear.
            # For a binary label that mean is the base rate of the event.
            if label in _BINARY_LABELS:
                constant = float(np.mean((y_train > 0).astype(np.float64)))
            else:
                constant = float(np.mean(y_train))
            result.reports.append(
                FoldReport(
                    label=label,
                    fold_index=fold.fold_index,
                    train_rows=train_rows.height,
                    validate_rows=validate_rows.height,
                    mae=float(np.mean(np.abs(predicted - truth))),
                    baseline_mae=float(np.mean(np.abs(constant - truth))),
                )
            )

    # Refit on everything for production use.
    x_all = _matrix(indexed, feature_columns)
    for label in label_columns:
        y_all = indexed[label].to_numpy().astype(np.float64)
        model = _fit(label, x_all, y_all, model_kwargs)
        if model is not None:
            result.models[label] = model
    result.trained_on_rows = indexed.height

    return result


def _fit(
    label: str, x: np.ndarray, y: np.ndarray, overrides: dict[str, Any] | None = None
) -> Any | None:
    """Fit one component model, or return ``None`` when the label is degenerate."""
    kwargs = {**_REGRESSOR_KWARGS, **(overrides or {})}
    if label in _BINARY_LABELS:
        binary = (y > 0).astype(np.int64)
        if len(np.unique(binary)) < 2:
            # A label that never varies cannot be classified, and forcing it
            # would produce a model that is confidently constant.
            return None
        model: Any = HistGradientBoostingClassifier(**kwargs)
        model.fit(x, binary)
        return model

    if float(np.std(y)) == 0.0:
        return None
    regressor = HistGradientBoostingRegressor(loss="absolute_error", **kwargs)
    regressor.fit(x, y)
    return regressor
