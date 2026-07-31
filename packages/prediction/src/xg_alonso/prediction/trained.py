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
from xg_alonso.contracts.prediction import MINUTES_STATES, MinutesState
from xg_alonso.contracts.seeds import ROOT_SEED

__all__ = [
    "TRAINED_MODEL_NAME",
    "TRAINED_MODEL_VERSION",
    "ComponentModels",
    "FoldReport",
    "StateFoldReport",
    "incumbent_state_probabilities",
    "train_component_models",
]

#: Ninety minutes. A fact about football, not an FPL rule, so it is not read
#: from the bootstrap payload — the payload does not publish it.
_FULL_MATCH: Final[float] = 90.0

TRAINED_MODEL_NAME: Final[str] = "component_hgb"
TRAINED_MODEL_VERSION: Final[str] = "1"

#: Labels modelled as probabilities rather than counts. Everything else is a
#: regression on a raw count.
_BINARY_LABELS: Final[frozenset[str]] = frozenset({"label_clean_sheets", "label_starts"})

#: Labels modelled as a distribution over mutually exclusive classes.
#:
#: **Why this is not two more binary heads.** ``inference.py::_minutes_from``
#: currently derives ``p_appearance`` and ``p_60_plus`` algebraically from an
#: expected-minutes regression and a start probability, because those two are
#: fitted independently and can contradict each other — 80 expected minutes
#: beside a 0.2 start probability is a shape the contract rejects. The
#: reconciliation exists to satisfy the validator, not because it estimates
#: anything, and it feeds the *appearance* term of ``assemble_points``, which is
#: the single largest term for most players.
#:
#: A three-class head estimates the whole distribution at once. The classes are
#: mutually exclusive and exhaustive by construction, so coherence is a property
#: of the model rather than a repair applied afterwards.
_MULTICLASS_LABELS: Final[frozenset[str]] = frozenset({"label_minutes_state"})

#: Count labels, fitted with Poisson loss.
#:
#: **The loss function is load-bearing here, not a tuning knob.** These labels
#: are 74-97% zeros, and ``absolute_error`` fits the conditional *median* — which
#: is zero everywhere on a distribution like that. A model trained that way
#: outputs a constant zero, carries no information, and still scores ~+50% on an
#: MAE-based skill metric, because predicting zero genuinely minimises MAE for a
#: rare event.
#:
#: Expected points needs E[X], the mean. Poisson loss fits the conditional mean
#: and is the natural family for a count, so it is what these use.
_COUNT_LABELS: Final[frozenset[str]] = frozenset(
    {
        "label_goals_scored",
        "label_assists",
        "label_saves",
        "label_goals_conceded",
        "label_bonus",
        "label_yellow_cards",
    }
)

#: Deliberately small trees. The dataset is tens of thousands of rows with
#: heavily correlated features, and a deeper model memorises player identity
#: rather than learning form.
_REGRESSOR_KWARGS: Final[dict[str, Any]] = {
    "max_iter": 200,
    "max_depth": 4,
    "learning_rate": 0.06,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": ROOT_SEED,
}


@dataclass(frozen=True)
class StateFoldReport:
    """Out-of-sample calibration for the minutes-state head on one fold.

    Deliberately **not** a :class:`FoldReport`. That one carries MAE against a
    mean-predicting baseline, which is the right measure for a count and a
    meaningless one for a distribution over three classes. Forcing this into
    the same shape would produce a record where half the fields are undefined
    and the two would be averaged together in every summary.

    The baseline here is the *incumbent*: the probabilities
    ``inference.py::_minutes_from`` derives today. Beating a base-rate constant
    would prove very little — the question is whether estimating the states
    directly is better than reconciling them from two point estimates.
    """

    fold_index: int
    train_rows: int
    validate_rows: int
    log_loss: float
    brier: float
    baseline_log_loss: float
    baseline_brier: float
    state_frequencies: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Realised class shares in the validation window, in canonical order."""

    @property
    def log_loss_gain(self) -> float:
        """Fractional reduction in log loss against the incumbent.

        Positive means the head is better. Zero when the baseline is already
        perfect, which cannot happen on real data but can on a fixture.
        """
        if self.baseline_log_loss <= 0.0:
            return 0.0
        return (self.baseline_log_loss - self.log_loss) / self.baseline_log_loss

    @property
    def brier_gain(self) -> float:
        """Fractional reduction in multiclass Brier score against the incumbent."""
        if self.baseline_brier <= 0.0:
            return 0.0
        return (self.baseline_brier - self.brier) / self.baseline_brier


@dataclass(frozen=True)
class FoldReport:
    """Out-of-sample error for one label on one fold."""

    label: str
    fold_index: int
    train_rows: int
    validate_rows: int
    mae: float
    baseline_mae: float
    prediction_std: float = 0.0
    predicted_mean: float = 0.0
    actual_mean: float = 0.0

    @property
    def skill(self) -> float:
        """Fractional improvement over predicting the training mean.

        **Read this together with :attr:`degenerate`.** On its own, MAE skill is
        not trustworthy for a rare event: a model that always outputs zero scores
        around +50% on a label that is 96% zeros, because zero really does
        minimise MAE there. The number is only meaningful once the prediction has
        been confirmed to vary.
        """
        if self.baseline_mae <= 0:
            return 0.0
        return 1.0 - (self.mae / self.baseline_mae)

    @property
    def degenerate(self) -> bool:
        """Whether the model output is effectively constant.

        A constant predictor carries no information regardless of its error, and
        it is exactly what an ill-chosen loss produces on a zero-inflated label.
        Detecting it is the check that MAE skill cannot provide.
        """
        return self.prediction_std < 1e-6

    @property
    def mean_bias(self) -> float:
        """How far the average prediction sits from the average outcome.

        Expected-points assembly sums components, so a component that is
        systematically low drags the whole total down. A model can have
        excellent MAE and still be badly biased in the mean — which is precisely
        the failure that matters here.
        """
        return self.predicted_mean - self.actual_mean


@dataclass
class ComponentModels:
    """Fitted component models, plus the evidence they were evaluated on."""

    models: dict[str, Any] = field(default_factory=dict)
    feature_columns: tuple[str, ...] = ()
    reports: list[FoldReport] = field(default_factory=list)
    trained_on_rows: int = 0
    folds: tuple[WalkForwardFold, ...] = ()
    dropped_features: tuple[str, ...] = ()
    """Declared features that were null for every training row.

    Recorded rather than silently discarded: a feature set that quietly shrinks
    is one whose version string no longer describes it.
    """

    estimator_kwargs: dict[str, Any] = field(default_factory=dict)
    """The hyperparameters these models were actually fitted with.

    Applied at fit time and never stored, so "was the architecture frozen
    between pinning and running" had no answer. A freeze check needs one.
    """

    state_reports: list[StateFoldReport] = field(default_factory=list)
    """Out-of-sample scores for the minutes-state head, if one was fitted.

    Appended with a default so artifacts pickled before the head existed
    unpickle into this class and fall through to an empty list.
    """

    label_means: dict[str, float] = field(default_factory=dict)
    """Population mean per label — what a component is shrunk *toward*.

    Kept alongside the models because a prediction is only as useful as the
    fallback it can be blended with, and recomputing the mean at inference time
    would use whatever data happened to be in the batch.
    """

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
        """Mean out-of-sample skill per label.

        A degenerate label reports zero skill regardless of its MAE, because a
        constant predictor has no skill however small its error.
        """
        by_label: dict[str, list[FoldReport]] = {}
        for report in self.reports:
            by_label.setdefault(report.label, []).append(report)
        return {
            label: (
                0.0
                if all(r.degenerate for r in reports)
                else sum(r.skill for r in reports) / len(reports)
            )
            for label, reports in by_label.items()
        }

    def degenerate_labels(self) -> list[str]:
        """Labels whose model output is effectively constant.

        These contribute nothing but an offset to the assembled points total,
        so any ranking built on them is driven by whatever else remains.
        """
        by_label: dict[str, list[FoldReport]] = {}
        for report in self.reports:
            by_label.setdefault(report.label, []).append(report)
        return sorted(
            label for label, reports in by_label.items() if all(r.degenerate for r in reports)
        )

    def mean_bias_by_label(self) -> dict[str, float]:
        """Average gap between predicted and actual means, per label."""
        by_label: dict[str, list[float]] = {}
        for report in self.reports:
            by_label.setdefault(report.label, []).append(report.mean_bias)
        return {k: sum(v) / len(v) for k, v in by_label.items()}

    def predict(
        self, features: pl.DataFrame, *, shrink_by_skill: bool = False
    ) -> dict[str, np.ndarray]:
        """Predict every component for a feature frame.

        Args:
            features: Rows to predict.
            shrink_by_skill: Blend each component toward its population mean in
                proportion to how badly it predicts out of sample.

        **Why shrinkage helps.** Components differ enormously in learnability —
        starts reach ~60% skill while clean sheets reach ~20% — but the
        assembled points total treats them as equally trustworthy. A weak signal
        used confidently is worse than a constant used honestly: it makes
        differences between players look real when they are noise, and the
        optimizer then spends transfers on them.

        Blending each prediction toward its mean with weight equal to measured
        skill lets a strong component speak nearly unaltered while a weak one is
        pulled most of the way back to the constant it barely improves on. Using
        ``skill`` as that weight is a deliberate approximation of the optimal
        shrinkage factor, not a derivation — its justification is that it is
        measured out of sample rather than assumed.
        """
        matrix = _matrix(features, self.feature_columns)
        skill = self.skill_by_label() if shrink_by_skill else {}

        out: dict[str, np.ndarray] = {}
        for label, model in self.models.items():
            if label in _MULTICLASS_LABELS:
                # Not a scalar per row, so it does not belong in this mapping.
                # `predict_minutes_state` returns it with its shape declared.
                continue
            if label in _BINARY_LABELS:
                raw = model.predict_proba(matrix)[:, 1]
            else:
                raw = np.clip(model.predict(matrix), 0.0, None)

            if shrink_by_skill:
                weight = max(0.0, min(1.0, skill.get(label, 0.0)))
                fallback = self.label_means.get(label, float(np.mean(raw)))
                raw = weight * raw + (1.0 - weight) * fallback

            out[label] = raw
        return out

    def predict_minutes_state(self, features: pl.DataFrame) -> np.ndarray | None:
        """Class probabilities over :data:`MINUTES_STATES`, or ``None`` if unfitted.

        Returns:
            An ``(n, 3)`` array whose columns are ``none``, ``short``, ``long``
            in that fixed order, each row summing to 1. ``None`` when this
            artifact predates the head — absent is honest, and a caller that
            silently substituted a uniform prior would be inventing the single
            most load-bearing quantity in the distribution.

        Kept out of :meth:`predict` because that returns one scalar per row per
        label, and quietly putting a two-dimensional array into it would break
        every consumer that assumes otherwise without changing a type.
        """
        model = self.models.get("label_minutes_state")
        if model is None:
            return None
        matrix = _matrix(features, self.feature_columns)
        return _state_probabilities(model, matrix)


def _state_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Expand a classifier's ``predict_proba`` to all three states.

    ``HistGradientBoostingClassifier`` emits one column per class it *saw*. An
    early walk-forward fold can contain no ``long`` rows at all, in which case
    the raw output has two columns and silently means something different. This
    places each column at its canonical index and leaves the unseen states at
    zero, so the shape is ``(n, 3)`` regardless.
    """
    raw = model.predict_proba(matrix)
    out = np.zeros((raw.shape[0], len(MINUTES_STATES)), dtype=np.float64)
    for column, class_index in enumerate(model.classes_):
        out[:, int(class_index)] = raw[:, column]
    return out


def usable_features(frame: pl.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    """Feature columns that have at least one value in this frame.

    **An entirely-null column crashes the estimator.** HistGradientBoosting bins
    each feature by taking midpoints between distinct values, and with no values
    at all the binner asks numpy for a two-wide sliding window over an empty
    array, which raises rather than degrading. The traceback names
    `_find_binning_thresholds` and says nothing about which column caused it.

    This is not hypothetical or new: `defensive_contribution` exists for one
    season only, so any training run that does not include 2025/26 already had
    an all-null column in the catalogue and would have hit exactly this. It
    surfaced when career features were added because those are null for players
    with no prior seasons.

    Dropping such a column loses nothing — a feature with no values cannot split
    anything — and the caller records which were dropped so a silently shrinking
    feature set stays visible.
    """
    return tuple(
        name
        for name in columns
        if name in frame.columns and frame[name].null_count() < frame.height
    )


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

    # A feature that is null for every row cannot split anything, and passing one
    # to the estimator raises from inside its binner rather than degrading.
    present = usable_features(training, feature_columns)
    dropped = [name for name in feature_columns if name not in set(present)]
    if dropped:
        # Not a warning to a log nobody reads: the fitted artifact records the
        # columns it actually used, so a later prediction cannot hand it a
        # feature set it was never fitted on.
        feature_columns = present

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

    result = ComponentModels(
        feature_columns=feature_columns, folds=tuple(folds), dropped_features=tuple(dropped)
    )

    for fold in folds:
        train_mask = indexed["__timeline"].is_between(fold.train_start, fold.train_end)
        validate_mask = indexed["__timeline"].is_between(fold.validate_start, fold.validate_end)
        train_rows = indexed.filter(train_mask)
        validate_rows = indexed.filter(validate_mask)
        if train_rows.is_empty() or validate_rows.is_empty():
            continue

        # Usability is a property of *this fold's* rows, not of the whole frame.
        # A career feature is null for every player before their first completed
        # season, and `defensive_contribution` is null for every season but one —
        # so an early fold can hold an all-null column while the full frame does
        # not. Checking globally and fitting per fold was the gap that let the
        # binner crash survive the first fix.
        fold_columns = usable_features(train_rows, feature_columns)
        if not fold_columns:
            continue
        x_train = _matrix(train_rows, fold_columns)
        x_validate = _matrix(validate_rows, fold_columns)

        fold_models: dict[str, Any] = {}
        for label in label_columns:
            y_train = train_rows[label].to_numpy().astype(np.float64)
            y_validate = validate_rows[label].to_numpy().astype(np.float64)

            model = _fit(label, x_train, y_train, model_kwargs)
            if model is None:
                continue
            fold_models[label] = model

            if label in _MULTICLASS_LABELS:
                # Scored separately: MAE against a mean-predicting constant is
                # not a measure of a distribution over three classes.
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
                    prediction_std=float(np.std(predicted)),
                    predicted_mean=float(np.mean(predicted)),
                    actual_mean=float(np.mean(truth)),
                )
            )

        state_report = _score_minutes_state(
            fold_models,
            validate_rows=validate_rows,
            x_validate=x_validate,
            fold=fold,
            train_rows=train_rows.height,
        )
        if state_report is not None:
            result.state_reports.append(state_report)

    # Refit on everything for production use.
    x_all = _matrix(indexed, feature_columns)
    for label in label_columns:
        y_all = indexed[label].to_numpy().astype(np.float64)
        model = _fit(label, x_all, y_all, model_kwargs)
        if model is not None:
            result.models[label] = model
        # Record what this component is shrunk toward. For a binary label the
        # mean is the base rate of the event, matching what the model predicts.
        result.label_means[label] = float(
            np.mean((y_all > 0).astype(np.float64)) if label in _BINARY_LABELS else np.mean(y_all)
        )
    result.trained_on_rows = indexed.height
    result.estimator_kwargs = {**_REGRESSOR_KWARGS, **(model_kwargs or {})}

    return result


def incumbent_state_probabilities(expected_minutes: np.ndarray, p_start: np.ndarray) -> np.ndarray:
    """The state distribution the shipped code derives today, vectorised.

    This is a deliberate, temporary duplicate of the reconciliation inside
    ``inference.py::_minutes_from`` — including its magic numbers, which are
    reproduced rather than cleaned up because the point is to reproduce the
    incumbent exactly. ``inference`` cannot be imported here (it imports this
    module), and the state head has to be scored against what it proposes to
    replace rather than against a base-rate constant.

    ``tests/prediction/test_minutes_state.py`` asserts elementwise agreement
    with ``_minutes_from``, so the duplication cannot drift. Both disappear in
    the task that wires the head into inference.

    Returns:
        An ``(n, 3)`` array over :data:`MINUTES_STATES`.
    """
    mean = np.clip(expected_minutes, 0.0, _FULL_MATCH)
    start = np.clip(p_start, 0.0, 1.0)

    p_appearance = np.maximum(mean / 70.0, 0.0)
    p_appearance = np.where(mean < 70.0, np.minimum(1.0, p_appearance), 1.0)
    p_appearance = np.maximum(start, p_appearance)

    p_60 = np.minimum(p_appearance, start * 0.9 + np.maximum(0.0, (mean - 60.0) / 30.0) * 0.1)
    p_60 = np.maximum(0.0, p_60)

    out = np.zeros((mean.shape[0], len(MINUTES_STATES)), dtype=np.float64)
    out[:, MinutesState.NONE.class_index] = 1.0 - p_appearance
    out[:, MinutesState.SHORT.class_index] = p_appearance - p_60
    out[:, MinutesState.LONG.class_index] = p_60
    return out


def _log_loss(probabilities: np.ndarray, truth: np.ndarray) -> float:
    """Mean negative log probability assigned to what actually happened.

    Clipped away from zero, because an unseen class carries probability exactly
    zero after :func:`_state_probabilities` expands the column set, and one such
    row would otherwise send the whole fold to infinity.
    """
    taken = probabilities[np.arange(truth.shape[0]), truth]
    return float(-np.mean(np.log(np.clip(taken, 1e-12, 1.0))))


def _brier(probabilities: np.ndarray, truth: np.ndarray) -> float:
    """Multiclass Brier score — squared error against the one-hot outcome."""
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(truth.shape[0]), truth] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def _score_minutes_state(
    fold_models: dict[str, Any],
    *,
    validate_rows: pl.DataFrame,
    x_validate: np.ndarray,
    fold: WalkForwardFold,
    train_rows: int,
) -> StateFoldReport | None:
    """Score the state head against the incumbent reconciliation, on one fold.

    Returns ``None`` — rather than a report full of zeros — whenever anything
    needed is missing: no state head, no state label, or no minutes/starts
    models to build the incumbent from. A fold that could not be compared did
    not produce a comparison.
    """
    model = fold_models.get("label_minutes_state")
    minutes_model = fold_models.get("label_minutes")
    starts_model = fold_models.get("label_starts")
    if model is None or minutes_model is None or starts_model is None:
        return None
    if "label_minutes_state" not in validate_rows.columns:
        return None

    truth = validate_rows["label_minutes_state"].to_numpy().astype(np.int64)
    predicted = _state_probabilities(model, x_validate)

    incumbent = incumbent_state_probabilities(
        np.clip(minutes_model.predict(x_validate), 0.0, None),
        starts_model.predict_proba(x_validate)[:, 1],
    )

    counts = np.bincount(truth, minlength=len(MINUTES_STATES)).astype(np.float64)
    shares = counts / max(1.0, float(truth.shape[0]))

    return StateFoldReport(
        fold_index=fold.fold_index,
        train_rows=train_rows,
        validate_rows=validate_rows.height,
        log_loss=_log_loss(predicted, truth),
        brier=_brier(predicted, truth),
        baseline_log_loss=_log_loss(incumbent, truth),
        baseline_brier=_brier(incumbent, truth),
        state_frequencies=(float(shares[0]), float(shares[1]), float(shares[2])),
    )


def _fit(
    label: str, x: np.ndarray, y: np.ndarray, overrides: dict[str, Any] | None = None
) -> Any | None:
    """Fit one component model, or return ``None`` when the label is degenerate."""
    kwargs = {**_REGRESSOR_KWARGS, **(overrides or {})}
    if label in _MULTICLASS_LABELS:
        classes = y.astype(np.int64)
        if len(np.unique(classes)) < 2:
            # One observed state cannot be a distribution over three.
            return None
        multiclass: Any = HistGradientBoostingClassifier(**kwargs)
        multiclass.fit(x, classes)
        return multiclass

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

    # Poisson for counts, squared error otherwise. Both fit the conditional
    # mean; absolute_error would fit the median and collapse to zero on these
    # zero-inflated labels.
    loss = "poisson" if label in _COUNT_LABELS else "squared_error"
    regressor = HistGradientBoostingRegressor(loss=loss, **kwargs)
    regressor.fit(x, np.clip(y, 0.0, None) if loss == "poisson" else y)
    return regressor
