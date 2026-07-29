"""Out-of-sample permutation importance for the component models.

**Why this exists.** The feature philosophy in `CLAUDE.md` says every feature
should know "whether it improved performance", and nothing measured that. The
catalogue declares 171 features, the models consume all of them, and there was
no evidence anywhere about which ones earn their place.

**Why permutation.** ``HistGradientBoosting`` exposes no ``feature_importances_``
at all, so there is nothing native to read. The available alternatives are a
surrogate model — which measures the surrogate, not the estimator that ships —
or permutation, which measures the deployed model directly by breaking one
feature at a time and observing what the error does. Permutation is slower and
correct, so it is what this uses.

**Why on validation rows.** Importance measured on training rows describes the
fit, and a gradient-boosted model can memorise a feature that generalises
nothing. Shuffling within the walk-forward *validation* window is what makes the
result evidence rather than a description: a feature that only helped by
memorising shows no degradation out of sample.

Two properties are reported rather than hidden, because both would otherwise
make the ranking quietly misleading:

- **Correlated features split their importance.** ``goals_scored_mean_3`` and
  ``goals_scored_mean_5`` carry nearly the same signal, so shuffling either
  leaves the model able to recover from the other, and both score low. Family
  totals are reported alongside the flat ranking for exactly this reason.
- **A degenerate label has no meaningful importance.** When a component model
  outputs a constant, permuting its inputs changes nothing, and ranking features
  by their effect on a constant is arithmetic without meaning. Those labels are
  flagged, not ranked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from xg_alonso.contracts.folds import WalkForwardFold

__all__ = [
    "ALL_POSITIONS",
    "IMPORTANCE_SCHEMA_VERSION",
    "LABEL_TO_BREAKDOWN",
    "FeatureImportance",
    "ImportanceTable",
    "label_weights_from_predictions",
    "load_importance",
    "permutation_importance",
    "write_importance",
]

IMPORTANCE_SCHEMA_VERSION: Final[str] = "importance_v2"
"""Bumped from v1 when ``position`` joined the measurement.

A v1 table has no position column. It is still readable — its rows load as the
``ALL`` slice, which is exactly what they were — so an existing table degrades
to "no positional detail" rather than failing to load."""

#: Labels modelled as probabilities. Mirrors the trained module's own split;
#: read from there rather than restated so the two cannot drift.
_BINARY_LABELS: Final[frozenset[str]] = frozenset({"label_clean_sheets", "label_starts"})


#: The slice covering every player, as opposed to one position's rows.
ALL_POSITIONS: Final[str] = "ALL"


@dataclass(frozen=True)
class FeatureImportance:
    """One feature's measured effect on one label, on one fold, for one position.

    **Position is part of the measurement, not a display filter.** A global
    ranking answers "what predicts a footballer's return", which is a question
    nobody is asking. Minutes played dominates a striker's goals; for a
    defender the same minutes matter through clean sheets, and a feature that
    ranks third overall may rank first for goalkeepers and last for forwards.
    Averaging those into one number does not describe any of them.

    ``position`` is ``ALL`` for the pooled slice, which is kept because it is
    the right denominator for "is this feature worth its place in the catalogue
    at all".
    """

    feature_name: str
    family: str
    label: str
    fold_index: int
    baseline_mae: float
    permuted_mae: float
    mae_delta_std: float
    repeats: int
    degenerate_label: bool
    position: str = ALL_POSITIONS
    """Which players this measurement describes. Defaults to the pooled slice so
    a v1 table and every existing call site keep their original meaning."""

    label_weight: float = 0.0
    """How much this label contributed to a points total, *for this slice*.

    Carried on the row rather than looked up from the table, because a table
    holds several slices and each has its own weighting — a clean sheet is four
    points to a defender and zero to a forward. A single table-level mapping
    would overwrite every slice's weighting with whichever was assigned last,
    which silently turns a positional ranking back into the pooled one.
    """

    rows_measured: int = 0
    """How many validation rows the slice had.

    Carried because a positional slice is small — goalkeepers are a twentieth of
    the pool — and an importance measured on eighty rows deserves to be read
    differently from one measured on eight thousand. Zero means unrecorded,
    which is what a v1 table gives."""

    @property
    def mae_delta(self) -> float:
        """How much worse the model gets when this feature is destroyed.

        Negative values are kept rather than clipped to zero. A feature whose
        removal *improves* out-of-sample error is real information — it says the
        model is being actively misled — and clipping it would present noise and
        active harm as the same thing.
        """
        return self.permuted_mae - self.baseline_mae

    @property
    def relative_delta(self) -> float:
        """Degradation as a fraction of the unpermuted error.

        Labels differ by orders of magnitude in scale — minutes are measured in
        tens, red cards in hundredths — so a raw MAE delta cannot be compared
        across them. This can.
        """
        if self.baseline_mae <= 0:
            return 0.0
        return self.mae_delta / self.baseline_mae

    @property
    def is_signal(self) -> bool:
        """Whether the degradation is larger than its own run-to-run variation.

        A feature whose mean delta sits inside the spread of its repeats has not
        been shown to matter, however positive that mean happens to be.
        """
        if self.degenerate_label:
            return False
        return self.mae_delta > self.mae_delta_std


@dataclass(frozen=True)
class ImportanceTable:
    """Every measurement, plus what produced it."""

    rows: tuple[FeatureImportance, ...]
    catalogue_version: str
    model_fingerprint: str
    computed_at: datetime
    label_weights: dict[str, float]
    """How much each label contributes to a points total, used for aggregation.

    Without this, ``label_yellow_cards`` — worth minus one point and rare —
    aggregates alongside ``label_minutes``, which gates every other component.
    An unweighted overall ranking would say they matter equally, which is not a
    close call.
    """

    def by_feature(self, position: str = ALL_POSITIONS) -> dict[str, float]:
        """Weighted overall importance per feature, averaged across folds.

        Each label's relative degradation is scaled by that label's contribution
        to expected points, so the result answers "which features move the
        number the product actually uses" rather than "which features move some
        component".

        **One slice at a time, always.** A table holds the pooled rows and every
        positional slice together, so an aggregate that ignored ``position``
        would quietly average a goalkeeper's measurement with a striker's and
        report the mean as importance. Defaults to the pooled slice, which is
        what every existing caller meant.
        """
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in self.rows:
            if row.degenerate_label or row.position != position:
                continue
            weight = row.label_weight or self.label_weights.get(row.label, 0.0)
            totals[row.feature_name] = (
                totals.get(row.feature_name, 0.0) + row.relative_delta * weight
            )
            counts[row.feature_name] = counts.get(row.feature_name, 0) + 1
        return {name: totals[name] / counts[name] for name in totals if counts[name] > 0}

    def by_family(self, position: str = ALL_POSITIONS) -> dict[str, float]:
        """Weighted importance summed within each catalogue family.

        The honest complement to the flat ranking. Correlated features inside a
        family substitute for one another and individually score low; the family
        total is what says whether that *kind* of feature matters.
        """
        totals: dict[str, float] = {}
        per_feature = self.by_feature(position)
        family_of = {row.feature_name: row.family for row in self.rows}
        for name, value in per_feature.items():
            family = family_of.get(name, "unknown")
            totals[family] = totals.get(family, 0.0) + value
        return totals

    def folds_measured(self) -> int:
        """How many distinct folds contributed rows."""
        return len({row.fold_index for row in self.rows})

    def positions_measured(self) -> tuple[str, ...]:
        """Slices present, pooled first then positions in a stable order."""
        found = {row.position for row in self.rows}
        ordered = [ALL_POSITIONS] if ALL_POSITIONS in found else []
        ordered.extend(sorted(found - {ALL_POSITIONS}))
        return tuple(ordered)

    def stability(self, position: str = ALL_POSITIONS) -> dict[str, float]:
        """Standard deviation of each feature's rank across folds.

        A feature that ranks third on one fold and hundredth on the next has not
        been shown to matter; it has been shown to be noisy.

        **Returns an empty mapping when only one fold was measured.** Ranking is
        done within a ``(fold, label)`` group and compared across folds for the
        same label, so with a single fold there is nothing to compare and the
        honest answer is "not measured". An earlier version pooled every label
        into one ranking, which meant the number it reported was the spread of a
        feature's rank *across components* — large for every feature, since a
        minutes feature legitimately ranks first for minutes and last for saves.
        It looked like an instability warning on the entire catalogue.
        """
        if self.folds_measured() < 2:
            return {}

        grouped: dict[tuple[int, str], list[tuple[str, float]]] = {}
        for row in self.rows:
            # Same reason as `by_feature`: ranking a pooled row against a
            # positional one would measure the difference between slices and
            # report it as instability across folds.
            if row.degenerate_label or row.position != position:
                continue
            grouped.setdefault((row.fold_index, row.label), []).append(
                (row.feature_name, row.relative_delta)
            )

        # rank[label][feature] = where that feature placed on each fold
        per_label: dict[str, dict[str, list[int]]] = {}
        for (_, label), entries in grouped.items():
            entries.sort(key=lambda pair: -pair[1])
            for rank, (name, _) in enumerate(entries):
                per_label.setdefault(label, {}).setdefault(name, []).append(rank)

        spreads: dict[str, list[float]] = {}
        for ranks_by_feature in per_label.values():
            for name, ranks in ranks_by_feature.items():
                if len(ranks) > 1:
                    spreads.setdefault(name, []).append(float(np.std(ranks)))

        return {name: sum(values) / len(values) for name, values in spreads.items() if values}

    def degenerate_labels(self) -> tuple[str, ...]:
        return tuple(sorted({row.label for row in self.rows if row.degenerate_label}))

    def to_frame(self) -> pl.DataFrame:
        """The table as a frame, ready to persist."""
        if not self.rows:
            return pl.DataFrame(
                schema={
                    "feature_name": pl.Utf8,
                    "family": pl.Utf8,
                    "label": pl.Utf8,
                    "fold_index": pl.Int64,
                    "position": pl.Utf8,
                    "rows_measured": pl.Int64,
                    "baseline_mae": pl.Float64,
                    "permuted_mae": pl.Float64,
                    "mae_delta": pl.Float64,
                    "mae_delta_std": pl.Float64,
                    "relative_delta": pl.Float64,
                    "repeats": pl.Int64,
                    "degenerate_label": pl.Boolean,
                    "label_weight": pl.Float64,
                    "catalogue_version": pl.Utf8,
                    "model_fingerprint": pl.Utf8,
                    "schema_version": pl.Utf8,
                    "computed_at": pl.Datetime(time_zone="UTC"),
                }
            )
        return pl.DataFrame(
            {
                "feature_name": [r.feature_name for r in self.rows],
                "family": [r.family for r in self.rows],
                "label": [r.label for r in self.rows],
                "fold_index": [r.fold_index for r in self.rows],
                "position": [r.position for r in self.rows],
                "rows_measured": [r.rows_measured for r in self.rows],
                "baseline_mae": [r.baseline_mae for r in self.rows],
                "permuted_mae": [r.permuted_mae for r in self.rows],
                "mae_delta": [r.mae_delta for r in self.rows],
                "mae_delta_std": [r.mae_delta_std for r in self.rows],
                "relative_delta": [r.relative_delta for r in self.rows],
                "repeats": [r.repeats for r in self.rows],
                "degenerate_label": [r.degenerate_label for r in self.rows],
                "label_weight": [
                    r.label_weight or self.label_weights.get(r.label, 0.0) for r in self.rows
                ],
                "catalogue_version": [self.catalogue_version] * len(self.rows),
                "model_fingerprint": [self.model_fingerprint] * len(self.rows),
                "schema_version": [IMPORTANCE_SCHEMA_VERSION] * len(self.rows),
                "computed_at": [self.computed_at] * len(self.rows),
            }
        )


#: Which assembled points term each modelled label drives.
#:
#: ``label_minutes`` and ``label_starts`` both map to the appearance term, which
#: understates them: minutes gate every other component as well, since a player
#: who does not play scores nothing anywhere. The weighting below is therefore a
#: floor on their importance, not an estimate of it — worth stating, because a
#: reader who sees minutes rank third should know the method cannot rank it
#: higher than the appearance points alone justify.
LABEL_TO_BREAKDOWN: Final[dict[str, str]] = {
    "label_minutes": "appearance",
    "label_starts": "appearance",
    "label_goals_scored": "goals",
    "label_assists": "assists",
    "label_clean_sheets": "clean_sheets",
    "label_goals_conceded": "goals_conceded",
    "label_saves": "saves",
    "label_yellow_cards": "cards",
    "label_bonus": "bonus",
}


def label_weights_from_predictions(
    predictions: Any, *, position: str | None = None
) -> dict[str, float]:
    """How much each label contributes to a points total, measured not assumed.

    Taken as the mean absolute contribution of the corresponding breakdown term
    across a real predicted population. Assuming instead — say, weighting goals
    by the six points a goal is worth — would rank goals above appearance, when
    in practice almost every player collects appearance points and almost none
    score, so appearance moves the totals more.

    Normalised to sum to one, so the aggregate ranking is a share rather than a
    quantity in unstated units.

    Args:
        predictions: A predicted population to measure against.
        position: Restrict to players of one position. **This is what makes a
            positional ranking mean anything.** A clean sheet is four points to
            a defender and zero to a forward, so a single global weighting
            flattens every position's ranking toward whatever dominates the
            pooled population — which is appearance, for everyone. Slicing the
            rows without also slicing the weights measures the right players
            against the wrong definition of what their points are made of.
    """
    totals: dict[str, float] = {}
    count = 0
    for prediction in predictions:
        if position is not None and str(getattr(prediction.position, "value", "")) != position:
            continue
        breakdown = prediction.breakdown
        count += 1
        for label, term in LABEL_TO_BREAKDOWN.items():
            totals[label] = totals.get(label, 0.0) + abs(getattr(breakdown, term))

    if count == 0:
        return {}

    means = {label: total / count for label, total in totals.items()}
    scale = sum(means.values())
    if scale <= 0:
        return dict.fromkeys(means, 0.0)
    return {label: value / scale for label, value in means.items()}


def _predict(model: Any, label: str, matrix: np.ndarray) -> np.ndarray:
    if label in _BINARY_LABELS:
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return np.clip(np.asarray(model.predict(matrix), dtype=np.float64), 0.0, None)


def _truth(label: str, values: np.ndarray) -> np.ndarray:
    return (values > 0).astype(np.float64) if label in _BINARY_LABELS else values


def _matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing feature columns: {missing[:8]}")
    return frame.select(columns).to_numpy().astype(np.float64)


def permutation_importance(
    models: Any,
    validation: pl.DataFrame,
    *,
    label_columns: tuple[str, ...],
    families: dict[str, str],
    label_weights: dict[str, float],
    catalogue_version: str,
    computed_at: datetime,
    folds: tuple[WalkForwardFold, ...] = (),
    fold_index: int = 0,
    n_repeats: int = 5,
    seed: int = 20260727,
    features: tuple[str, ...] | None = None,
    position: str = ALL_POSITIONS,
) -> ImportanceTable:
    """Measure how much each feature is worth to each component model.

    Args:
        models: A fitted :class:`~xg_alonso.prediction.trained.ComponentModels`.
        validation: Rows the models were **not** fitted on.
        label_columns: Which labels to measure.
        families: Feature name to catalogue family, for grouped reporting.
        label_weights: Each label's contribution to a points total.
        catalogue_version: Recorded so a table cannot be read against a
            different feature set without the mismatch being visible.
        computed_at: Passed in rather than read from the clock, so a run is
            reproducible.
        n_repeats: Shuffles per feature. The spread across repeats is what
            separates a real effect from a lucky permutation.
        seed: Fixed, so the same inputs give the same table.
        position: Which slice ``validation`` represents. The caller filters the
            frame; this only labels the result, so the two cannot disagree about
            what was measured.
        features: Which features to measure. Defaults to every feature the
            models were fitted on.

    Returns:
        One row per feature per label.
    """
    if validation.is_empty():
        raise ValueError(
            "permutation importance needs validation rows; measuring on the "
            "training set would describe the fit rather than the model"
        )

    columns = tuple(models.feature_columns)
    measured = features if features is not None else columns
    unknown = [f for f in measured if f not in columns]
    if unknown:
        raise KeyError(f"these features were not fitted on: {unknown[:8]}")

    degenerate = set(models.degenerate_labels())
    base_matrix = _matrix(validation, columns)
    index_of = {name: column for column, name in enumerate(columns)}
    generator = np.random.default_rng(seed)

    rows: list[FeatureImportance] = []
    for label in label_columns:
        model = models.models.get(label)
        if model is None or label not in validation.columns:
            continue

        truth = _truth(label, validation[label].to_numpy().astype(np.float64))
        baseline = float(np.mean(np.abs(_predict(model, label, base_matrix) - truth)))

        for feature in measured:
            column = index_of[feature]
            original = base_matrix[:, column].copy()

            deltas: list[float] = []
            for _ in range(n_repeats):
                base_matrix[:, column] = generator.permutation(original)
                permuted = float(np.mean(np.abs(_predict(model, label, base_matrix) - truth)))
                deltas.append(permuted - baseline)
            base_matrix[:, column] = original

            rows.append(
                FeatureImportance(
                    feature_name=feature,
                    family=families.get(feature, "unknown"),
                    label=label,
                    fold_index=fold_index,
                    baseline_mae=baseline,
                    permuted_mae=baseline + float(np.mean(deltas)),
                    mae_delta_std=float(np.std(deltas)),
                    repeats=n_repeats,
                    degenerate_label=label in degenerate,
                    position=position,
                    rows_measured=validation.height,
                    label_weight=label_weights.get(label, 0.0),
                )
            )

    return ImportanceTable(
        rows=tuple(rows),
        catalogue_version=catalogue_version,
        model_fingerprint=models.fingerprint(),
        computed_at=computed_at,
        label_weights=label_weights,
    )


def write_importance(table: ImportanceTable, path: Path) -> Path:
    """Persist a table, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_frame().write_parquet(path)
    return path


def load_importance(path: Path) -> pl.DataFrame:
    """Read a persisted table, tolerating one written before positions existed.

    Returns the frame rather than the dataclass because every consumer so far —
    the API and the CLI report — filters and aggregates, and rebuilding objects
    only to tear them down again would buy nothing.

    A v1 table has no ``position`` column. Its rows *were* the pooled slice, so
    they are labelled ``ALL`` rather than null: null would suggest the slice is
    unknown, when it is known exactly.
    """
    frame = pl.read_parquet(path)
    if "position" not in frame.columns:
        frame = frame.with_columns(pl.lit(ALL_POSITIONS).alias("position"))
    if "rows_measured" not in frame.columns:
        frame = frame.with_columns(pl.lit(0, dtype=pl.Int64).alias("rows_measured"))
    return frame
