"""Persistent registries for hypotheses, features, evaluations and experiments.

Backed by the existing :class:`~xg_alonso.contracts.storage.TableStore`, which is
DuckDB today (decision D2) and stays replaceable because nothing here knows that.
Introducing SQLite alongside it would have added a second persistence story for
no benefit and quietly contradicted D2.

**Evaluations are append-only.** :meth:`DiscoveryRegistry.record_evaluation` uses
``append_table`` and there is deliberately no update or delete. Re-measuring a
feature adds a row; it never edits one. That is what makes the question "what did
we believe about this feature in October, and on what evidence" answerable at all
— and a feature-discovery system that cannot answer it is a system whose past
conclusions are unfalsifiable.

The same reasoning the bronze layer already applies to raw data, applied to
findings.

**Serialisation.** Nested structures — fold metrics, subgroup results, the
program tree — are stored as JSON text in a single column rather than exploded
into child tables. They are read back whole, never queried into, so a relational
decomposition would buy nothing and cost a migration every time a schema gains a
field.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

import polars as pl
from pydantic import BaseModel

from xg_alonso.contracts.discovery import (
    AcceptanceStatus,
    ClusterSummary,
    DiscoveredFeatureSpec,
    ExperimentManifest,
    FeatureEvaluation,
    FeatureHypothesis,
    HypothesisStatus,
    Lesson,
    PlayerClusterAssignment,
)
from xg_alonso.contracts.objective import ManagerObjective
from xg_alonso.contracts.storage import TableStore

__all__ = [
    "DISCOVERY_TABLES",
    "REGISTRY_SCHEMA_VERSION",
    "DiscoveryRegistry",
]

REGISTRY_SCHEMA_VERSION: Final[str] = "discovery_registry_v1"

HYPOTHESES: Final[str] = "discovery_hypotheses"
FEATURES: Final[str] = "discovery_features"
DEPENDENCIES: Final[str] = "discovery_feature_dependencies"
EVALUATIONS: Final[str] = "discovery_evaluations"
OBJECTIVES: Final[str] = "discovery_objectives"
CLUSTER_MODELS: Final[str] = "discovery_cluster_models"
CLUSTER_ASSIGNMENTS: Final[str] = "discovery_cluster_assignments"
EXPERIMENTS: Final[str] = "discovery_experiments"
LESSONS: Final[str] = "discovery_lessons"

DISCOVERY_TABLES: Final[tuple[str, ...]] = (
    HYPOTHESES,
    FEATURES,
    DEPENDENCIES,
    EVALUATIONS,
    OBJECTIVES,
    CLUSTER_MODELS,
    CLUSTER_ASSIGNMENTS,
    EXPERIMENTS,
    LESSONS,
)

#: Fields serialised as JSON text rather than as columns. Everything that is a
#: sequence of structures, plus anything whose shape may grow.
_JSON_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "folds",
        "baseline_metrics",
        "candidate_metrics",
        "subgroup_results",
        "cluster_results",
        "leakage_checks",
        "seeds",
        "fold_definitions",
        "metrics",
        "target_objectives",
        "applicable_positions",
        "applicable_clusters",
        "required_raw_fields",
        "input_columns",
        "objective_tags",
        "secondary_metrics",
        "supporting_evidence",
        "seasons",
        "dominant_features",
        "objective_weights",
    }
)


def _flatten(model: BaseModel) -> dict[str, Any]:
    """One row from a model: scalars as columns, structures as JSON text."""
    payload = model.model_dump(mode="json")
    row: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _JSON_FIELDS or isinstance(value, (list, dict, tuple)):
            row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        elif isinstance(value, bool):
            row[key] = value
        else:
            row[key] = value
    return row


def _inflate(row: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_flatten` — JSON columns back into structures."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in _JSON_FIELDS and isinstance(value, str):
            try:
                out[key] = json.loads(value)
            except json.JSONDecodeError:
                out[key] = value
        else:
            out[key] = value
    return out


class DiscoveryRegistry:
    """Read/write access to every discovery artifact.

    One object rather than nine so a caller has one place to look, and so the
    "did this table exist yet" question is answered once.
    """

    def __init__(self, store: TableStore) -> None:
        self._store = store

    # -- generic helpers --------------------------------------------------

    def _append(self, table: str, model: BaseModel, *, run_id: str = "discovery") -> None:
        frame = pl.DataFrame([_flatten(model)])
        if self._store.table_exists(table):
            existing = self._store.read_table(table)
            # Align columns so a model that gained a field does not fail the
            # append with an opaque arity error. New columns are added as nulls
            # to history, which is honest: those rows genuinely lack the value.
            for column in existing.columns:
                if column not in frame.columns:
                    frame = frame.with_columns(pl.lit(None).alias(column))
            for column in frame.columns:
                if column not in existing.columns:
                    existing = existing.with_columns(pl.lit(None).alias(column))
                    self._store.write_table(table, existing, run_id=run_id)
            frame = frame.select(existing.columns)
        self._store.append_table(table, frame, run_id=run_id)

    def _read(self, table: str) -> pl.DataFrame:
        if not self._store.table_exists(table):
            return pl.DataFrame()
        return self._store.read_table(table)

    def _upsert(self, table: str, model: BaseModel, key: str, *, run_id: str = "discovery") -> None:
        """Replace the row with this key, or append when absent.

        Used for mutable registries — a hypothesis's status changes as it moves
        through the loop. **Never used for evaluations**, whose whole value is
        that they cannot be rewritten.
        """
        row = _flatten(model)
        frame = pl.DataFrame([row])
        if not self._store.table_exists(table):
            self._store.append_table(table, frame, run_id=run_id)
            return
        existing = self._store.read_table(table)
        for column in existing.columns:
            if column not in frame.columns:
                frame = frame.with_columns(pl.lit(None).alias(column))
        frame = frame.select(existing.columns)
        kept = existing.filter(pl.col(key) != row[key])
        self._store.write_table(
            table, pl.concat([kept, frame], how="vertical_relaxed"), run_id=run_id
        )

    # -- objectives -------------------------------------------------------

    def register_objective(self, objective: ManagerObjective) -> None:
        self._upsert(OBJECTIVES, objective, key="id")

    def objectives(self) -> list[ManagerObjective]:
        frame = self._read(OBJECTIVES)
        if frame.is_empty():
            return []
        return [ManagerObjective(**_inflate(row)) for row in frame.iter_rows(named=True)]

    def objective(self, objective_id: str) -> ManagerObjective | None:
        return next((o for o in self.objectives() if o.id == objective_id), None)

    # -- hypotheses -------------------------------------------------------

    def register_hypothesis(self, hypothesis: FeatureHypothesis) -> None:
        self._upsert(HYPOTHESES, hypothesis, key="id")

    def hypotheses(self, *, status: HypothesisStatus | None = None) -> list[FeatureHypothesis]:
        frame = self._read(HYPOTHESES)
        if frame.is_empty():
            return []
        found = [FeatureHypothesis(**_inflate(row)) for row in frame.iter_rows(named=True)]
        if status is not None:
            found = [h for h in found if h.status is status]
        return found

    def hypothesis(self, hypothesis_id: str) -> FeatureHypothesis | None:
        return next((h for h in self.hypotheses() if h.id == hypothesis_id), None)

    def set_hypothesis_status(self, hypothesis_id: str, status: HypothesisStatus) -> None:
        found = self.hypothesis(hypothesis_id)
        if found is None:
            raise KeyError(f"no hypothesis {hypothesis_id!r}")
        self.register_hypothesis(found.model_copy(update={"status": status}))

    # -- features ---------------------------------------------------------

    def register_feature(self, spec: DiscoveredFeatureSpec) -> None:
        """Store a feature spec and its lineage.

        Refuses a spec that has not passed the leakage harness. This is the gate:
        static validation alone cannot prove a window does not straddle a fold
        boundary, so a program that was only statically checked must not become
        something a model can be fitted on.
        """
        if not spec.validation_status.is_registrable:
            raise ValueError(
                f"feature {spec.name!r} has validation status "
                f"{spec.validation_status.value}; only a program that has passed the "
                "point-in-time leakage harness may be registered, because a leaking "
                "feature's measured value is fiction"
            )
        self._upsert(FEATURES, spec, key="version")
        if spec.input_columns:
            rows = [
                {"feature_id": spec.id, "feature_version": spec.version, "source_column": column}
                for column in spec.input_columns
            ]
            self._store.append_table(DEPENDENCIES, pl.DataFrame(rows), run_id="discovery")

    def features(self) -> list[DiscoveredFeatureSpec]:
        frame = self._read(FEATURES)
        if frame.is_empty():
            return []
        return [DiscoveredFeatureSpec(**_inflate(row)) for row in frame.iter_rows(named=True)]

    def feature(self, version: str) -> DiscoveredFeatureSpec | None:
        return next((f for f in self.features() if f.version == version), None)

    def feature_versions(self) -> tuple[str, ...]:
        """Every registered version. Passed to the validator for duplicate detection."""
        return tuple(f.version for f in self.features())

    def dependencies(self, feature_version: str) -> tuple[str, ...]:
        frame = self._read(DEPENDENCIES)
        if frame.is_empty():
            return ()
        rows = frame.filter(pl.col("feature_version") == feature_version)
        return tuple(sorted({str(r["source_column"]) for r in rows.iter_rows(named=True)}))

    # -- evaluations (append-only) ----------------------------------------

    def record_evaluation(self, evaluation: FeatureEvaluation) -> None:
        """Append one evaluation. **There is no update and no delete.**"""
        self._append(EVALUATIONS, evaluation)

    def evaluations(
        self,
        *,
        feature_version: str | None = None,
        objective_id: str | None = None,
        experiment_id: str | None = None,
    ) -> list[FeatureEvaluation]:
        frame = self._read(EVALUATIONS)
        if frame.is_empty():
            return []
        found = [FeatureEvaluation(**_inflate(row)) for row in frame.iter_rows(named=True)]
        if feature_version is not None:
            found = [e for e in found if e.feature_version == feature_version]
        if objective_id is not None:
            found = [e for e in found if e.objective_id == objective_id]
        if experiment_id is not None:
            found = [e for e in found if e.experiment_id == experiment_id]
        return found

    def latest_evaluation(
        self, feature_version: str, objective_id: str
    ) -> FeatureEvaluation | None:
        found = self.evaluations(feature_version=feature_version, objective_id=objective_id)
        if not found:
            return None
        return max(found, key=lambda e: e.evaluated_at)

    def accepted_features(self, objective_id: str) -> list[DiscoveredFeatureSpec]:
        """Feature specs whose most recent verdict under this objective is usable.

        Reads the *latest* evaluation per feature rather than any accepting one.
        A feature accepted in March and rejected in May is rejected; taking the
        best result ever recorded would make the registry a highlight reel.
        """
        by_version = {f.version: f for f in self.features()}
        out: list[DiscoveredFeatureSpec] = []
        for version, spec in by_version.items():
            latest = self.latest_evaluation(version, objective_id)
            if latest is not None and latest.accepted.is_usable:
                out.append(spec)
        return out

    # -- clusters ---------------------------------------------------------

    def register_cluster_model(self, summaries: Sequence[ClusterSummary]) -> None:
        if not summaries:
            return
        rows = [_flatten(s) for s in summaries]
        self._store.append_table(CLUSTER_MODELS, pl.DataFrame(rows), run_id="discovery")

    def cluster_summaries(
        self, *, model_version: str | None = None, objective_id: str | None = None
    ) -> list[ClusterSummary]:
        frame = self._read(CLUSTER_MODELS)
        if frame.is_empty():
            return []
        found = [ClusterSummary(**_inflate(row)) for row in frame.iter_rows(named=True)]
        if model_version is not None:
            found = [c for c in found if c.cluster_model_version == model_version]
        if objective_id is not None:
            found = [c for c in found if c.objective_id == objective_id]
        return found

    def register_assignments(self, assignments: Sequence[PlayerClusterAssignment]) -> None:
        if not assignments:
            return
        rows = [_flatten(a) for a in assignments]
        self._store.append_table(CLUSTER_ASSIGNMENTS, pl.DataFrame(rows), run_id="discovery")

    def assignments_frame(
        self, *, model_version: str | None = None, objective_id: str | None = None
    ) -> pl.DataFrame:
        """Assignments as a frame, ready to join.

        Returned as a frame rather than as objects because every consumer joins
        or filters — rebuilding models only to tear them down again would buy
        nothing, which is the same reasoning ``load_importance`` already applies.
        """
        frame = self._read(CLUSTER_ASSIGNMENTS)
        if frame.is_empty():
            return frame
        if model_version is not None:
            frame = frame.filter(pl.col("cluster_model_version") == model_version)
        if objective_id is not None and "objective_id" in frame.columns:
            frame = frame.filter(pl.col("objective_id") == objective_id)
        return frame

    def cluster_history(
        self, player_code: int, *, model_version: str | None = None
    ) -> pl.DataFrame:
        """One player's cluster over time, oldest first.

        The transition history. A player who moves from "rotation option" to
        "nailed starter" has changed as an asset, and the gameweek he moved is
        the thing a manager wants to see.
        """
        frame = self.assignments_frame(model_version=model_version)
        if frame.is_empty():
            return frame
        return frame.filter(pl.col("player_code") == player_code).sort(["season", "gameweek"])

    # -- experiments ------------------------------------------------------

    def register_experiment(self, manifest: ExperimentManifest) -> None:
        self._upsert(EXPERIMENTS, manifest, key="experiment_id")

    def experiments(self) -> list[ExperimentManifest]:
        frame = self._read(EXPERIMENTS)
        if frame.is_empty():
            return []
        return [ExperimentManifest(**_inflate(row)) for row in frame.iter_rows(named=True)]

    def experiment(self, experiment_id: str) -> ExperimentManifest | None:
        return next((e for e in self.experiments() if e.experiment_id == experiment_id), None)

    # -- lessons ----------------------------------------------------------

    def record_lesson(self, lesson: Lesson) -> None:
        self._upsert(LESSONS, lesson, key="id")

    def lessons(self, *, family: str | None = None) -> list[Lesson]:
        frame = self._read(LESSONS)
        if frame.is_empty():
            return []
        found = [Lesson(**_inflate(row)) for row in frame.iter_rows(named=True)]
        if family is not None:
            found = [lesson for lesson in found if lesson.hypothesis_family == family]
        return found

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Row counts per table. Cheap enough to call from a health endpoint."""
        return {
            table: (self._read(table).height if self._store.table_exists(table) else 0)
            for table in DISCOVERY_TABLES
        }

    def acceptance_report(self, objective_id: str) -> pl.DataFrame:
        """Every feature's latest verdict under one objective, best utility first.

        The table a user reads to see what the loop concluded and why. Rejections
        are included with their reasons: a report that showed only what was
        accepted would hide the part that makes the accepted set credible.
        """
        specs = {f.version: f for f in self.features()}
        rows: list[dict[str, Any]] = []
        for version, spec in specs.items():
            latest = self.latest_evaluation(version, objective_id)
            if latest is None:
                continue
            rows.append(
                {
                    "feature": spec.name,
                    "version": version[:12],
                    "hypothesis_id": spec.hypothesis_id or "",
                    "status": latest.accepted.value,
                    "complementarity": latest.complementarity.value,
                    "utility": round(latest.utility, 5),
                    "incremental_value": round(latest.incremental_value, 5),
                    "folds": len(latest.folds),
                    "folds_improved": latest.folds_improved,
                    "fold_win_rate": round(latest.fold_win_rate, 3),
                    "stability": round(latest.stability, 3),
                    "missingness": round(latest.missingness, 3),
                    "leakage_passed": latest.leakage_passed,
                    "reason": latest.rejection_reason,
                }
            )
        if not rows:
            return pl.DataFrame(
                schema={
                    "feature": pl.Utf8,
                    "version": pl.Utf8,
                    "hypothesis_id": pl.Utf8,
                    "status": pl.Utf8,
                    "complementarity": pl.Utf8,
                    "utility": pl.Float64,
                    "incremental_value": pl.Float64,
                    "folds": pl.Int64,
                    "folds_improved": pl.Int64,
                    "fold_win_rate": pl.Float64,
                    "stability": pl.Float64,
                    "missingness": pl.Float64,
                    "leakage_passed": pl.Boolean,
                    "reason": pl.Utf8,
                }
            )
        order = {
            AcceptanceStatus.ACCEPTED.value: 0,
            AcceptanceStatus.ACCEPTED_EXPERIMENTALLY.value: 1,
            AcceptanceStatus.REVISE.value: 2,
            AcceptanceStatus.INSUFFICIENT_DATA.value: 3,
            AcceptanceStatus.REJECTED.value: 4,
            AcceptanceStatus.DEPRECATED.value: 5,
        }
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("status").replace_strict(order, default=9).alias("__order"))
            .sort(["__order", "utility"], descending=[False, True])
            .drop("__order")
        )
