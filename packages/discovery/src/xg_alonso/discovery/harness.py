"""Walk-forward measurement of what a feature set is worth.

Every number this package reports comes from here, and the ordering constraint
is absolute: **a model is fitted only on gameweeks strictly before the ones it is
scored on.** Folds come from the frozen
:func:`~xg_alonso.contracts.folds.walk_forward_folds`, which is structurally
incapable of shuffling — it validates that the gameweek sequence is ascending and
unique and raises otherwise.

Three properties are non-negotiable and all three are enforced rather than
trusted:

1. **No random splits.** The only fold constructor is the frozen one.
2. **Nothing is fitted on validation rows.** Scalers, projections, clusters and
   the estimator all see training rows only. The embedding and cluster APIs make
   this structural; here it is the caller's fold loop, and
   ``tests/discovery/test_fold_isolation.py`` asserts it.
3. **The final period is not optimised on.** ``holdout_seasons`` is excluded
   from every fold, so acceptance decisions cannot be tuned against the data
   that will later be used to claim the system works.

**Two controls, because a positive result needs something to be positive
against.** :func:`random_control` fits the same estimator on the required set
plus a *matched-complexity noise column*, and :func:`shuffled_control` plus a
permuted copy of the candidate itself. A candidate that does not beat both has
not been shown to carry information — it has been shown that adding any column
helps, which is a statement about the model's capacity, not about football.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from xg_alonso.contracts.discovery import FoldMetrics, SubgroupResult
from xg_alonso.contracts.folds import WalkForwardFold, walk_forward_folds
from xg_alonso.contracts.identifiers import GameweekId
from xg_alonso.discovery.search import ScoreResult

__all__ = [
    "DEFAULT_ESTIMATOR_KWARGS",
    "FoldSplit",
    "HarnessConfig",
    "PredictionCache",
    "build_timeline",
    "evaluate_feature_set",
    "make_scorer",
    "random_control",
    "shuffled_control",
    "subgroup_metrics",
]

#: Deliberately small trees, matching ``prediction.trained``. The dataset is
#: tens of thousands of rows with heavily correlated features, and a deeper model
#: memorises player identity rather than learning form. Matching the production
#: estimator also means a feature that helps here is a feature that helps there.
DEFAULT_ESTIMATOR_KWARGS: Final[dict[str, Any]] = {
    "max_iter": 120,
    "max_depth": 4,
    "learning_rate": 0.08,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 20260727,
}

#: The label a discovery run scores against by default.
#:
#: Total points rather than a single component. `CLAUDE.md` models components and
#: prices them, and that is right for *prediction* — but a discovered feature is
#: being judged on whether it improves the number a manager reads, and a feature
#: that sharpens goals while blunting minutes can improve a component and worsen
#: the total.
DEFAULT_TARGET: Final[str] = "label_total_points"


@dataclass
class PredictionCache:
    """Memoises ``(fold, feature set) -> validation predictions``.

    A discovery run fits the *same* models many times over. The baseline is
    refitted for every candidate's subgroup breakdown, and the position and
    cluster breakdowns of one candidate are identical fits differing only in how
    the resulting errors are grouped. Without this cache a run spends most of its
    time recomputing predictions it already had — measured at roughly a 4x
    saving on the shipped configuration.

    Keyed on the sorted feature set, so ``(a, b)`` and ``(b, a)`` are one entry.
    Nothing about the ordering of a feature list changes what is fitted.
    """

    entries: dict[tuple[int, tuple[str, ...]], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = (
        field(default_factory=dict)
    )
    hits: int = 0
    misses: int = 0

    def key(self, fold_index: int, columns: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        return (fold_index, tuple(sorted(set(columns))))


@dataclass(frozen=True)
class FoldSplit:
    """One fold's actual rows, already separated. Never overlapping."""

    fold: WalkForwardFold
    train: pl.DataFrame
    validate: pl.DataFrame

    @property
    def usable(self) -> bool:
        return not self.train.is_empty() and not self.validate.is_empty()


@dataclass
class HarnessConfig:
    """How a discovery evaluation is run."""

    target: str = DEFAULT_TARGET
    min_train_gameweeks: int = 12
    validate_gameweeks: int = 4
    embargo_gameweeks: int = 1
    """Skipped between training and validation, so a rolling window's lookback
    cannot straddle the boundary and carry validation information backwards."""

    holdout_seasons: tuple[str, ...] = ()
    """Excluded from every fold. The final test period is never optimised on."""

    max_folds: int = 6
    seed: int = 20260727
    estimator_kwargs: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ESTIMATOR_KWARGS))
    lower_is_better: bool = True


def build_timeline(
    frame: pl.DataFrame, *, season_col: str = "label_season", gameweek_col: str = "label_gameweek"
) -> pl.DataFrame:
    """Attach a globally increasing gameweek index across seasons.

    Without this a fold can wrap backwards over a season boundary — gameweek 3 of
    2024-25 sorts before gameweek 30 of 2023-24 on the raw number, and the
    resulting "walk-forward" split trains on the future. Mirrors what
    ``prediction.trained`` already does, for the same reason.
    """
    seasons = sorted(str(s) for s in frame[season_col].unique().to_list())
    offsets = {season: index * 100 for index, season in enumerate(seasons)}
    return frame.with_columns(
        (pl.col(season_col).replace_strict(offsets, default=0) + pl.col(gameweek_col)).alias(
            "__timeline"
        )
    )


def _folds(frame: pl.DataFrame, config: HarnessConfig) -> list[FoldSplit]:
    """Build fold splits from the frozen constructor, holdout excluded."""
    working = frame
    if config.holdout_seasons:
        working = working.filter(~pl.col("label_season").is_in(list(config.holdout_seasons)))
    if working.is_empty():
        raise ValueError(
            "excluding the holdout seasons left no rows; the whole dataset cannot "
            "be the final test period"
        )

    indexed = build_timeline(working)
    timeline = sorted({int(v) for v in indexed["__timeline"].unique()})

    folds = walk_forward_folds(
        gameweeks=[GameweekId(v) for v in timeline],
        min_train_gameweeks=config.min_train_gameweeks,
        validate_gameweeks=config.validate_gameweeks,
        embargo_gameweeks=config.embargo_gameweeks,
    )
    # Keep the most recent folds: they are the ones whose regime resembles the
    # period the model will actually run in.
    if len(folds) > config.max_folds:
        folds = folds[-config.max_folds :]

    splits: list[FoldSplit] = []
    for fold in folds:
        train = indexed.filter(pl.col("__timeline").is_between(fold.train_start, fold.train_end))
        validate = indexed.filter(
            pl.col("__timeline").is_between(fold.validate_start, fold.validate_end)
        )
        split = FoldSplit(fold=fold, train=train, validate=validate)
        if split.usable:
            splits.append(split)
    return splits


def _matrix(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Feature matrix with nulls preserved as NaN.

    HistGradientBoosting treats NaN as its own branch, so a player without
    history is modelled as *unknown* rather than as an invented average — the
    same choice ``prediction.trained`` makes and for the same reason.
    """
    height = frame.height
    out = np.full((height, len(columns)), np.nan, dtype=np.float64)
    for index, name in enumerate(columns):
        if name in frame.columns:
            out[:, index] = frame[name].cast(pl.Float64, strict=False).to_numpy()
    return out


def _usable(frame: pl.DataFrame, columns: Sequence[str]) -> tuple[str, ...]:
    """Columns with at least one value in this frame.

    An entirely-null column crashes HistGradientBoosting's binner rather than
    degrading, with a traceback that names ``_find_binning_thresholds`` and not
    the column. Checked per fold, because usability is a property of the fold's
    rows: ``defensive_contribution`` exists for one season only, so an early
    fold can hold an all-null column while the full frame does not.
    """
    return tuple(
        name
        for name in columns
        if name in frame.columns and frame[name].null_count() < frame.height
    )


def _fit_and_score(
    split: FoldSplit, columns: Sequence[str], config: HarnessConfig
) -> tuple[float, int] | None:
    """Fit on the training rows, score MAE on the validation rows."""
    usable = _usable(split.train, columns)
    if not usable:
        return None

    y_train = split.train[config.target].cast(pl.Float64, strict=False).to_numpy()
    y_validate = split.validate[config.target].cast(pl.Float64, strict=False).to_numpy()
    train_ok = np.isfinite(y_train)
    validate_ok = np.isfinite(y_validate)
    if int(train_ok.sum()) < 50 or int(validate_ok.sum()) < 20:
        return None
    if float(np.std(y_train[train_ok])) < 1e-9:
        return None

    x_train = _matrix(split.train, usable)[train_ok]
    x_validate = _matrix(split.validate, usable)[validate_ok]

    model = HistGradientBoostingRegressor(**config.estimator_kwargs)
    model.fit(x_train, y_train[train_ok])
    predicted = model.predict(x_validate)
    return float(np.mean(np.abs(predicted - y_validate[validate_ok]))), int(validate_ok.sum())


def evaluate_feature_set(
    frame: pl.DataFrame,
    *,
    baseline_columns: Sequence[str],
    candidate_columns: Sequence[str] = (),
    config: HarnessConfig | None = None,
    splits: Sequence[FoldSplit] | None = None,
) -> ScoreResult:
    """Score ``baseline + candidate`` out of sample, fold by fold.

    Args:
        frame: Training rows across every season, carrying the target.
        baseline_columns: The set to measure against — normally the manager's
            required features.
        candidate_columns: Added to the baseline. Empty measures the baseline.
        splits: Precomputed folds. Passed in when many candidates are scored over
            the same splits, which is the whole of a search.

    Returns:
        A :class:`~xg_alonso.discovery.search.ScoreResult` whose ``metric`` is the
        row-weighted mean validation MAE. Row-weighted rather than a plain
        average of fold means, so a fold covering twice as many players counts
        twice — an unweighted mean lets a thin fold move the headline.
    """
    settings = config or HarnessConfig()
    folds = list(splits) if splits is not None else _folds(frame, settings)
    columns = tuple(dict.fromkeys([*baseline_columns, *candidate_columns]))
    if not columns:
        raise ValueError("a feature set must contain at least one column")

    metrics: list[FoldMetrics] = []
    weighted_total = 0.0
    weight = 0

    for split in folds:
        scored = _fit_and_score(split, columns, settings)
        if scored is None:
            continue
        mae, rows = scored
        weighted_total += mae * rows
        weight += rows
        metrics.append(
            FoldMetrics(
                fold_index=split.fold.fold_index,
                train_rows=split.train.height,
                validate_rows=rows,
                baseline_metric=mae,
                candidate_metric=mae,
                metric_name="mae",
                lower_is_better=settings.lower_is_better,
            )
        )

    if weight == 0:
        return ScoreResult(metric=float("inf"), folds=(), lower_is_better=True)

    return ScoreResult(
        metric=weighted_total / weight,
        folds=tuple(metrics),
        lower_is_better=settings.lower_is_better,
        detail={"folds": float(len(metrics)), "validate_rows": float(weight)},
    )


def make_scorer(
    frame: pl.DataFrame, *, config: HarnessConfig | None = None
) -> tuple[Callable[[tuple[str, ...]], ScoreResult], list[FoldSplit]]:
    """A memoised scorer over fixed folds, plus the folds themselves.

    Folds are built **once**. A search evaluates hundreds of sets, and rebuilding
    the splits per call would dominate the runtime — and, worse, would leave open
    the possibility of two candidates being scored against different splits,
    which would make their gains incomparable.

    Memoised on the sorted column tuple, so ``(a, b)`` and ``(b, a)`` are one
    evaluation rather than two.
    """
    settings = config or HarnessConfig()
    splits = _folds(frame, settings)
    cache: dict[tuple[str, ...], ScoreResult] = {}

    def score(columns: tuple[str, ...]) -> ScoreResult:
        key = tuple(sorted(set(columns)))
        found = cache.get(key)
        if found is None:
            found = evaluate_feature_set(
                frame, baseline_columns=columns, config=settings, splits=splits
            )
            cache[key] = found
        return found

    return score, splits


def paired_fold_metrics(baseline: ScoreResult, candidate: ScoreResult) -> tuple[FoldMetrics, ...]:
    """Align two runs fold by fold so improvement is measured within a fold.

    Comparing pooled means would let a candidate look better merely by being
    evaluated on an easier set of folds. Pairing on ``fold_index`` removes that,
    and a fold present in only one run is dropped rather than matched to nothing.
    """
    by_index = {f.fold_index: f for f in baseline.folds}
    out: list[FoldMetrics] = []
    for fold in candidate.folds:
        base = by_index.get(fold.fold_index)
        if base is None:
            continue
        out.append(
            FoldMetrics(
                fold_index=fold.fold_index,
                train_rows=fold.train_rows,
                validate_rows=fold.validate_rows,
                baseline_metric=base.candidate_metric,
                candidate_metric=fold.candidate_metric,
                metric_name=fold.metric_name,
                lower_is_better=fold.lower_is_better,
            )
        )
    return tuple(out)


def random_control(
    frame: pl.DataFrame,
    *,
    baseline_columns: Sequence[str],
    config: HarnessConfig | None = None,
    splits: Sequence[FoldSplit] | None = None,
    seed: int = 20260727,
    repeats: int = 3,
) -> ScoreResult:
    """The baseline plus a matched-complexity **noise** column.

    The control a positive result needs. A gradient-boosted model handed one more
    column will often improve slightly whatever the column contains, so "adding
    this feature helped" is not evidence until it beats adding nothing in
    particular. Averaged over several draws, because a single lucky noise column
    is exactly the thing being controlled for.
    """
    settings = config or HarnessConfig()
    folds = list(splits) if splits is not None else _folds(frame, settings)
    generator = np.random.default_rng(seed)

    scores: list[ScoreResult] = []
    for draw in range(repeats):
        name = f"__control_noise_{draw}"
        noisy = [
            FoldSplit(
                fold=split.fold,
                train=split.train.with_columns(
                    pl.Series(name, generator.standard_normal(split.train.height))
                ),
                validate=split.validate.with_columns(
                    pl.Series(name, generator.standard_normal(split.validate.height))
                ),
            )
            for split in folds
        ]
        scores.append(
            evaluate_feature_set(
                frame,
                baseline_columns=baseline_columns,
                candidate_columns=(name,),
                config=settings,
                splits=noisy,
            )
        )

    usable = [s for s in scores if np.isfinite(s.metric)]
    if not usable:
        return ScoreResult(metric=float("inf"), folds=(), lower_is_better=True)
    return ScoreResult(
        metric=float(np.mean([s.metric for s in usable])),
        folds=usable[0].folds,
        lower_is_better=settings.lower_is_better,
        detail={"repeats": float(len(usable))},
    )


def shuffled_control(
    frame: pl.DataFrame,
    *,
    baseline_columns: Sequence[str],
    candidate: str,
    config: HarnessConfig | None = None,
    splits: Sequence[FoldSplit] | None = None,
    seed: int = 20260727,
) -> ScoreResult:
    """The baseline plus a **permuted copy of the candidate itself**.

    Stronger than a noise control: it preserves the candidate's exact marginal
    distribution — its scale, skew and missingness — and destroys only the
    association with the target. A candidate that beats this has been shown to
    carry information about the outcome specifically, rather than to have a
    convenient shape.

    Shuffled **within** each fold's rows, so the permutation cannot move a value
    from validation into training.
    """
    settings = config or HarnessConfig()
    folds = list(splits) if splits is not None else _folds(frame, settings)
    generator = np.random.default_rng(seed)
    name = f"__shuffled_{candidate}"

    permuted = []
    for split in folds:
        if candidate not in split.train.columns:
            continue
        permuted.append(
            FoldSplit(
                fold=split.fold,
                train=split.train.with_columns(
                    pl.Series(name, generator.permutation(split.train[candidate].to_numpy()))
                ),
                validate=split.validate.with_columns(
                    pl.Series(name, generator.permutation(split.validate[candidate].to_numpy()))
                ),
            )
        )
    if not permuted:
        return ScoreResult(metric=float("inf"), folds=(), lower_is_better=True)

    return evaluate_feature_set(
        frame,
        baseline_columns=baseline_columns,
        candidate_columns=(name,),
        config=settings,
        splits=permuted,
    )


def subgroup_metrics(
    frame: pl.DataFrame,
    *,
    baseline_columns: Sequence[str],
    candidate_columns: Sequence[str],
    segment_column: str,
    segment_kind: str,
    config: HarnessConfig | None = None,
    splits: Sequence[FoldSplit] | None = None,
    min_rows: int = 200,
    cache: PredictionCache | None = None,
) -> list[SubgroupResult]:
    """Where a candidate helps, measured **within** each segment.

    The model is fitted once on all training rows and scored separately per
    segment of the validation rows. That is the right decomposition: a gated
    feature should be judged by whether it improves the players it targets while
    not harming the rest, and fitting a separate model per segment would answer a
    different question — whether a segment-specific model is better — and would
    do it on a fraction of the data.

    Segments below ``min_rows`` are dropped rather than reported. A spectacular
    gain on eleven rows is the commonest way a discovery loop fools itself.
    """
    settings = config or HarnessConfig()
    folds = list(splits) if splits is not None else _folds(frame, settings)
    base = tuple(dict.fromkeys(baseline_columns))
    full = tuple(dict.fromkeys([*baseline_columns, *candidate_columns]))

    totals: dict[str, list[float]] = {}
    counts: dict[str, int] = {}

    for split in folds:
        if segment_column not in split.validate.columns:
            continue
        base_pred = _predictions(split, base, settings, cache)
        full_pred = _predictions(split, full, settings, cache)
        if base_pred is None or full_pred is None:
            continue
        truth, base_values, mask = base_pred
        _, full_values, _ = full_pred

        segments = split.validate[segment_column].to_list()
        kept = [str(segments[i]) for i, ok in enumerate(mask) if ok]
        base_error = np.abs(base_values - truth)
        full_error = np.abs(full_values - truth)

        for index, segment in enumerate(kept):
            entry = totals.setdefault(segment, [0.0, 0.0])
            entry[0] += float(base_error[index])
            entry[1] += float(full_error[index])
            counts[segment] = counts.get(segment, 0) + 1

    out: list[SubgroupResult] = []
    for segment, (base_sum, full_sum) in totals.items():
        n = counts[segment]
        if n < min_rows:
            continue
        out.append(
            SubgroupResult(
                segment_kind=segment_kind,
                segment=segment,
                n=n,
                baseline_metric=base_sum / n,
                candidate_metric=full_sum / n,
                lower_is_better=True,
            )
        )
    return sorted(out, key=lambda r: -r.relative_improvement)


def _predictions(
    split: FoldSplit,
    columns: Sequence[str],
    config: HarnessConfig,
    cache: PredictionCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit and predict, returning ``(truth, predicted, validation_mask)``."""
    if cache is not None:
        key = cache.key(split.fold.fold_index, columns)
        if key in cache.entries:
            cache.hits += 1
            return cache.entries[key]
        cache.misses += 1

    result = _predictions_uncached(split, columns, config)
    if cache is not None:
        cache.entries[cache.key(split.fold.fold_index, columns)] = result
    return result


def _predictions_uncached(
    split: FoldSplit, columns: Sequence[str], config: HarnessConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    usable = _usable(split.train, columns)
    if not usable:
        return None
    y_train = split.train[config.target].cast(pl.Float64, strict=False).to_numpy()
    y_validate = split.validate[config.target].cast(pl.Float64, strict=False).to_numpy()
    train_ok = np.isfinite(y_train)
    validate_ok = np.isfinite(y_validate)
    if int(train_ok.sum()) < 50 or int(validate_ok.sum()) < 20:
        return None
    if float(np.std(y_train[train_ok])) < 1e-9:
        return None

    model = HistGradientBoostingRegressor(**config.estimator_kwargs)
    model.fit(_matrix(split.train, usable)[train_ok], y_train[train_ok])
    predicted = model.predict(_matrix(split.validate, usable)[validate_ok])
    return y_validate[validate_ok], np.asarray(predicted, dtype=np.float64), validate_ok
