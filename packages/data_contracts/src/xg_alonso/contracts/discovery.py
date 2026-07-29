"""The vocabulary of a feature-discovery experiment.

A discovery run is a scientific loop, not a search: a hypothesis is stated with
a falsification condition *before* it is tested, compiled into a program that
cannot read the future, measured against baselines on strictly forward folds,
and then accepted or rejected against criteria fixed in advance.

Every schema here exists to make one step of that loop auditable after the fact.
The two that matter most:

- :class:`FeatureHypothesis` carries ``falsification_condition``. A hypothesis
  that cannot say what would refute it is not a hypothesis, and the field is
  required rather than optional for exactly that reason.
- :class:`FeatureEvaluation` is **append-only**. It records fold-level detail
  alongside the summary, because a mean improvement built on one spectacular
  fold and four losses is a different finding from a consistent small gain, and
  a summary alone cannot tell them apart.

**Naming.** ``DiscoveredFeatureSpec`` is deliberately not ``FeatureSpec``:
:class:`xg_alonso.features.catalogue.FeatureSpec` already exists and means
something else — a *declared* catalogue entry written by a person. This is a
*discovered* program proposed by the loop. They are different concepts with
different lifecycles, and giving them one name would be the first step toward
giving them one confused implementation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.provenance import utc_now

__all__ = [
    "AcceptanceStatus",
    "ClusterSummary",
    "ComplementarityClass",
    "DiscoveredFeatureSpec",
    "EntityLevel",
    "ExperimentManifest",
    "ExperimentStage",
    "FeatureEvaluation",
    "FeatureHypothesis",
    "FoldMetrics",
    "GenerationSource",
    "HypothesisStatus",
    "LeakageRisk",
    "Lesson",
    "MissingValuePolicy",
    "PlayerClusterAssignment",
    "SubgroupResult",
    "ValidationStatus",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _require_tz(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


# --- hypotheses --------------------------------------------------------------


class LeakageRisk(StrEnum):
    """Declared, *a priori* leakage exposure of a hypothesis.

    Declared by the proposer and then checked mechanically. The declaration is
    kept even after the check because a proposer who repeatedly declares NONE and
    is repeatedly caught is a generator worth distrusting.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GenerationSource(StrEnum):
    """Where a hypothesis came from. Recorded so provenance is not folklore."""

    SEEDED = "seeded"
    """From the hand-written baseline library. A candidate, not a truth."""

    RESIDUAL = "residual"
    """Generated from measured model weakness on a segment."""

    IMPORTANCE = "importance"
    """Generated from the measured importance table."""

    COMPLEMENT = "complement"
    """Generated to complement a required feature."""

    REVISION = "revision"
    """A revision of an earlier hypothesis that was promising but unstable."""

    LLM = "llm"
    """Proposed by a language model.

    **No LLM client exists in this repository.** The value exists so that a
    proposal, if one is ever supplied through the adapter in
    ``xg_alonso.discovery.llm``, is permanently distinguishable from a measured
    one. An LLM rationale is a hypothesis, never a finding.
    """


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    COMPILED = "compiled"
    VALIDATED = "validated"
    """Passed static and leakage validation; not yet measured."""
    TESTED = "tested"
    ACCEPTED = "accepted"
    ACCEPTED_EXPERIMENTALLY = "accepted_experimentally"
    REJECTED = "rejected"
    REVISE = "revise"
    INSUFFICIENT_DATA = "insufficient_data"
    DEPRECATED = "deprecated"


class FeatureHypothesis(_Frozen):
    """A testable claim about football, stated before it is measured."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""

    football_rationale: str = Field(
        min_length=1,
        description=(
            "The mechanism claimed, in football terms. Required: a candidate "
            "with no stated mechanism is a column name, and the loop exists to "
            "avoid generating those."
        ),
    )

    target_objectives: tuple[str, ...] = ()
    """Objective ids this is expected to help. Empty means 'any'."""

    applicable_positions: tuple[Position, ...] = ()
    """Empty means every position."""

    applicable_clusters: tuple[int, ...] = ()
    """Cluster ids this targets, within ``cluster_model_version``. Empty means all."""

    cluster_model_version: str | None = None

    required_raw_fields: tuple[str, ...] = Field(
        default=(),
        description="Source columns the program will read. Checked against the actual schema.",
    )

    transformation_plan: str = ""
    """Prose description of the intended program, before compilation."""

    expected_relationship: str = Field(
        default="",
        description="The direction and shape expected, e.g. 'positive, strongest for high-minute forwards'",
    )

    falsification_condition: str = Field(
        min_length=1,
        description=(
            "What result would refute this. Required. A hypothesis that cannot "
            "be refuted cannot be tested, and recording the condition before the "
            "measurement is what stops it being rewritten afterwards."
        ),
    )

    leakage_risk: LeakageRisk = LeakageRisk.NONE
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_hypothesis_id: str | None = None
    generation_source: GenerationSource = GenerationSource.SEEDED
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        return _require_tz(value)

    @property
    def family(self) -> str:
        """Coarse grouping used by hypothesis memory.

        Derived from the id's first segment, so ``congestion.rest_x_minutes`` and
        ``congestion.short_rest_penalty`` share a family and their lessons pool.
        """
        return self.id.split(".", 1)[0]


# --- feature programs --------------------------------------------------------


class EntityLevel(StrEnum):
    """What one row of a computed feature describes."""

    PLAYER_GAMEWEEK = "player_gameweek"
    PLAYER_SEASON = "player_season"
    TEAM_GAMEWEEK = "team_gameweek"


class MissingValuePolicy(StrEnum):
    """What a feature does when it cannot be computed.

    ``NULL`` is the default and usually correct: ``HistGradientBoosting`` treats
    a null as its own branch, so "unknown" stays distinguishable from "zero".
    Filling with zero asserts a fact the data does not support — that a player
    with no history produced nothing — and that assertion is wrong for exactly
    the cold-start players a manager most needs ranked.
    """

    NULL = "null"
    ZERO = "zero"
    PRIOR = "prior"
    """Shrink to the population prior, as the catalogue's rates already do."""


class ValidationStatus(StrEnum):
    UNVALIDATED = "unvalidated"
    STATIC_PASSED = "static_passed"
    LEAKAGE_PASSED = "leakage_passed"
    """Passed both static validation and the point-in-time leakage harness."""
    REJECTED_STATIC = "rejected_static"
    REJECTED_LEAKAGE = "rejected_leakage"

    @property
    def is_computable(self) -> bool:
        """Whether a program in this state may be computed at all."""
        return self in {ValidationStatus.STATIC_PASSED, ValidationStatus.LEAKAGE_PASSED}

    @property
    def is_registrable(self) -> bool:
        """Whether a program may enter the registry as a usable feature.

        Only a program that has passed the leakage harness qualifies. Static
        validation alone is not enough: a program can be syntactically clean and
        still read a window that straddles a fold boundary.
        """
        return self is ValidationStatus.LEAKAGE_PASSED


class DiscoveredFeatureSpec(_Frozen):
    """One discovered feature: its program, its lineage, and its version."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, description="Column name the program produces")
    version: str = Field(
        min_length=8,
        description=(
            "Content hash of the canonical program JSON. Identity is semantic, "
            "not nominal: two programs that compute the same thing collide here "
            "and the second is rejected as a duplicate, and an edited program is "
            "a new version rather than a silent mutation of an old one."
        ),
    )
    description: str = ""

    program: str = Field(
        min_length=2,
        description=(
            "The feature program as canonical JSON. A serialized expression "
            "tree, never Python source — nothing in this system evals or execs."
        ),
    )

    input_columns: tuple[str, ...] = Field(
        default=(), description="Raw source columns this program reads. Its lineage."
    )
    output_type: str = Field(default="float")
    entity_level: EntityLevel = EntityLevel.PLAYER_GAMEWEEK

    temporal_window: int | None = Field(
        default=None, ge=1, description="Widest lookback window, in appearances"
    )
    minimum_history: int = Field(
        default=0, ge=0, description="Observations below which the value is withheld"
    )
    missing_value_policy: MissingValuePolicy = MissingValuePolicy.NULL

    applicable_positions: tuple[Position, ...] = ()
    applicable_clusters: tuple[int, ...] = ()
    cluster_model_version: str | None = None
    objective_tags: tuple[str, ...] = ()

    hypothesis_id: str | None = None
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    complexity: int = Field(
        default=1, ge=1, description="Node count of the program tree. Penalised in utility."
    )
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        return _require_tz(value)

    @property
    def is_gated(self) -> bool:
        """Whether this feature applies only to a subset of players."""
        return bool(self.applicable_positions or self.applicable_clusters)

    def qualified_name(self) -> str:
        """Name plus a short version, for use as a column in a versioned frame."""
        return f"{self.name}__{self.version[:8]}"


# --- evaluation --------------------------------------------------------------


class FoldMetrics(_Frozen):
    """One walk-forward fold's measurement of one candidate.

    Retained per fold rather than averaged away. A mean hides whether a gain was
    consistent or came from a single fold, and consistency is most of what
    separates a feature from a coincidence.
    """

    fold_index: int = Field(ge=0)
    train_rows: int = Field(ge=0)
    validate_rows: int = Field(ge=0)

    baseline_metric: float
    candidate_metric: float
    metric_name: str = "mae"
    lower_is_better: bool = True

    @property
    def improvement(self) -> float:
        """Signed improvement, oriented so positive always means better."""
        delta = self.baseline_metric - self.candidate_metric
        return delta if self.lower_is_better else -delta

    @property
    def relative_improvement(self) -> float:
        """Improvement as a fraction of the baseline, comparable across metrics."""
        if abs(self.baseline_metric) < 1e-12:
            return 0.0
        return self.improvement / abs(self.baseline_metric)

    @property
    def improved(self) -> bool:
        return self.improvement > 0.0


class ComplementarityClass(StrEnum):
    """What a candidate adds, given the features already required.

    The distinction between the first three is the product. A feature that helps
    globally is a model improvement; one that helps only under one objective is a
    *reason the objective layer exists*; one that helps only for some clusters is
    a gated feature. Collapsing them into "useful/not useful" throws away the
    finding.
    """

    GLOBALLY_COMPLEMENTARY = "globally_complementary"
    OBJECTIVE_SPECIFIC = "objective_specific"
    CLUSTER_SPECIFIC = "cluster_specific"
    REDUNDANT = "redundant"
    UNSTABLE = "unstable"
    LEAKAGE_SUSPECTED = "leakage_suspected"
    INSUFFICIENT_HISTORY = "insufficient_history"


class AcceptanceStatus(StrEnum):
    ACCEPTED = "accepted"
    ACCEPTED_EXPERIMENTALLY = "accepted_experimentally"
    REJECTED = "rejected"
    REVISE = "revise"
    INSUFFICIENT_DATA = "insufficient_data"
    DEPRECATED = "deprecated"

    @property
    def is_usable(self) -> bool:
        return self in {AcceptanceStatus.ACCEPTED, AcceptanceStatus.ACCEPTED_EXPERIMENTALLY}


class SubgroupResult(_Frozen):
    """A candidate's value inside one segment of the player pool.

    ``n`` is carried and checked. A spectacular gain on eleven rows is the most
    common way a discovery loop fools itself, and the acceptance policy refuses
    a subgroup claim below a configured floor for exactly that reason.
    """

    segment_kind: str = Field(description="'position' | 'cluster' | 'price_band' | 'ownership'")
    segment: str
    n: int = Field(ge=0)
    baseline_metric: float
    candidate_metric: float
    lower_is_better: bool = True

    @property
    def improvement(self) -> float:
        delta = self.baseline_metric - self.candidate_metric
        return delta if self.lower_is_better else -delta

    @property
    def relative_improvement(self) -> float:
        if abs(self.baseline_metric) < 1e-12:
            return 0.0
        return self.improvement / abs(self.baseline_metric)


class FeatureEvaluation(_Frozen):
    """The complete measured verdict on one candidate under one objective.

    Written append-only. Re-evaluating a feature adds a row; it never edits one,
    so the history of what was believed and when stays intact.
    """

    feature_id: str
    feature_version: str
    objective_id: str
    model_family: str = Field(default="component_hgb")

    backtest_start: int = Field(description="First gameweek on the global timeline")
    backtest_end: int
    folds: tuple[FoldMetrics, ...] = ()

    baseline_metrics: tuple[tuple[str, float], ...] = ()
    """Named metrics for the baseline feature set. A tuple of pairs rather than a
    mapping so the model stays hashable and genuinely frozen."""

    candidate_metrics: tuple[tuple[str, float], ...] = ()

    incremental_value: float = Field(
        default=0.0, description="Mean relative improvement over the baseline across folds"
    )
    permutation_importance: float | None = None
    stability: float = Field(
        default=0.0,
        description=(
            "Consistency of the improvement across folds, in [0, 1]. Computed as "
            "the share of folds that improved, penalised by the spread — a "
            "feature that wins every fold by a little scores above one that wins "
            "half of them by a lot."
        ),
    )
    turnover: float = Field(default=0.0, ge=0.0, description="Rank churn of the feature's values")
    missingness: float = Field(default=0.0, ge=0.0, le=1.0)

    leakage_checks: tuple[str, ...] = ()
    """Named checks that were run. Empty means none ran, which is a rejection."""

    leakage_passed: bool = False

    subgroup_results: tuple[SubgroupResult, ...] = ()
    cluster_results: tuple[SubgroupResult, ...] = ()

    complementarity: ComplementarityClass = ComplementarityClass.REDUNDANT
    utility: float = 0.0
    accepted: AcceptanceStatus = AcceptanceStatus.REJECTED
    rejection_reason: str = ""

    experiment_id: str = ""
    evaluated_at: datetime = Field(default_factory=utc_now)

    @field_validator("evaluated_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        return _require_tz(value)

    @model_validator(mode="after")
    def _rejection_is_explained(self) -> FeatureEvaluation:
        if self.accepted is AcceptanceStatus.REJECTED and not self.rejection_reason:
            raise ValueError(
                "a rejected candidate must carry a reason; an unexplained "
                "rejection is indistinguishable from a crash"
            )
        return self

    @property
    def folds_improved(self) -> int:
        return sum(1 for f in self.folds if f.improved)

    @property
    def fold_win_rate(self) -> float:
        if not self.folds:
            return 0.0
        return self.folds_improved / len(self.folds)

    @property
    def recent_fold_improved(self) -> bool:
        """Whether the most recent fold improved.

        Checked separately because a feature that helped for two seasons and
        stopped helping this one is worse than useless — the model would keep
        leaning on it precisely when it has stopped working.
        """
        if not self.folds:
            return False
        return max(self.folds, key=lambda f: f.fold_index).improved

    def metric(self, name: str, *, candidate: bool = True) -> float | None:
        source = self.candidate_metrics if candidate else self.baseline_metrics
        return next((value for key, value in source if key == name), None)


# --- clusters ----------------------------------------------------------------


class PlayerClusterAssignment(_Frozen):
    """Where one player sat, in one gameweek, under one cluster model.

    Keyed by gameweek because a cluster is not an identity. A midfielder pushed
    forward after an injury genuinely becomes a different kind of asset, and a
    model that assigns him a permanent label cannot notice.
    """

    player_code: PlayerCode
    gameweek: GameweekId
    season: str
    cluster_model_version: str
    cluster_id: int = Field(ge=0)
    membership_probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Soft membership. Used to gate features: a feature that matters for "
            "one archetype is multiplied by the probability the player belongs "
            "to it, rather than switched on by a hard label he may sit close to "
            "the boundary of."
        ),
    )
    distance_to_centroid: float = Field(default=0.0, ge=0.0)
    objective_id: str | None = Field(
        default=None,
        description="Set when this assignment came from an objective-conditioned model",
    )


class ClusterSummary(_Frozen):
    """What a cluster is, described by the dimensions it is extreme on.

    ``label`` may be a readable name, but ``dominant_features`` is the evidence
    and is always present. A label without its statistical basis is a story; the
    basis without a label is merely hard to read. Both ship.
    """

    cluster_model_version: str
    cluster_id: int = Field(ge=0)
    position: str | None = None
    objective_id: str | None = None

    size: int = Field(ge=0)
    label: str = ""
    dominant_features: tuple[tuple[str, float], ...] = ()
    """(dimension, standardised centroid) pairs, strongest first."""

    silhouette: float | None = None
    stability: float | None = Field(
        default=None, description="Share of members retained from the previous gameweek"
    )


# --- experiments -------------------------------------------------------------


class ExperimentStage(StrEnum):
    """Lifecycle of a discovery experiment.

    Exposed even though execution is currently synchronous. The stage vocabulary
    is what a job queue would report, so adding one later is a change of
    execution strategy rather than of interface.
    """

    QUEUED = "queued"
    VALIDATING = "validating"
    COMPUTING_FEATURES = "computing_features"
    BACKTESTING = "backtesting"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {ExperimentStage.COMPLETED, ExperimentStage.FAILED}


class ExperimentManifest(_Frozen):
    """Everything needed to reproduce one discovery experiment.

    The reproducibility claim this repository makes elsewhere — a prediction is
    reproducible from its provenance — extends here. If any field below is
    missing, the experiment is a result nobody can check.
    """

    experiment_id: str = Field(min_length=1)
    stage: ExperimentStage = ExperimentStage.QUEUED

    objective_id: str
    objective_version: str
    constraints_hash: str = Field(description="Hash of the serialized ManagerConstraints")
    beliefs_hash: str = Field(default="", description="Hash of the serialized beliefs")

    data_cutoff: datetime
    seasons: tuple[str, ...] = ()
    feature_set_version: str = ""
    cluster_model_version: str | None = None
    model_config_hash: str = ""

    seeds: tuple[tuple[str, int], ...] = ()
    """Named seeds. Every RNG in this system is seeded; recording which value
    went where is what makes 'deterministic' checkable rather than asserted."""

    fold_definitions: tuple[tuple[int, int, int, int, int], ...] = ()
    """(fold_index, train_start, train_end, validate_start, validate_end)."""

    hypotheses_proposed: int = Field(default=0, ge=0)
    features_compiled: int = Field(default=0, ge=0)
    features_accepted: int = Field(default=0, ge=0)
    features_rejected: int = Field(default=0, ge=0)

    metrics: tuple[tuple[str, float], ...] = ()
    code_version: str = Field(default="", description="Commit SHA of the code that ran this")
    git_dirty: bool = False
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    error: str = ""

    @field_validator("data_cutoff", "started_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        return _require_tz(value)

    @property
    def reproducible(self) -> bool:
        """Whether this experiment's outputs may be trusted as reproducible.

        A dirty working tree disqualifies a run, exactly as
        :attr:`xg_alonso.contracts.provenance.RunManifest.promotable` does: the
        recorded commit does not describe the code that ran.
        """
        return bool(self.code_version) and not self.git_dirty

    @model_validator(mode="after")
    def _failure_is_explained(self) -> ExperimentManifest:
        if self.stage is ExperimentStage.FAILED and not self.error:
            raise ValueError("a failed experiment must record why it failed")
        return self


class Lesson(_Frozen):
    """A compact, reusable finding about a family of hypotheses.

    This is the memory that makes the loop cumulative. Without it the generator
    re-proposes a rejected transformation every run, and the cost of a discovery
    system becomes linear in how many times it has already been run.
    """

    id: str = Field(min_length=1)
    hypothesis_family: str = Field(min_length=1)
    result: str = Field(min_length=1, description="What was found, in one or two sentences")
    failure_mode: str = ""
    promising_direction: str = ""
    supporting_evidence: tuple[str, ...] = Field(
        default=(), description="Evaluation ids this lesson was drawn from"
    )
    objective_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _tz_aware(cls, value: datetime) -> datetime:
        return _require_tz(value)
