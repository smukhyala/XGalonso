"""Shared schemas, protocols and reason codes.

This package is the vocabulary every other package speaks. It depends on no
other internal package, which is what keeps it a boundary rather than a second
implementation.

Four contracts here are *frozen* — during planning each was independently
designed several incompatible ways, so each now has exactly one definition:

- :mod:`~xg_alonso.contracts.prediction` — the prediction output shape
- :mod:`~xg_alonso.contracts.reason_codes` — the reason-code vocabulary
- :mod:`~xg_alonso.contracts.folds` — the walk-forward fold
- :mod:`~xg_alonso.contracts.storage` — the storage protocols

Two further modules describe the objective-conditioned discovery layer:

- :mod:`~xg_alonso.contracts.objective` — what a manager is trying to achieve,
  what they will not allow, and what they believe
- :mod:`~xg_alonso.contracts.discovery` — hypotheses, feature programs, their
  evaluations, clusters and experiment manifests
"""

from xg_alonso.contracts.constraints import SquadViolation
from xg_alonso.contracts.discovery import (
    AcceptanceStatus,
    ClusterSummary,
    ComplementarityClass,
    DiscoveredFeatureSpec,
    EntityLevel,
    ExperimentManifest,
    ExperimentStage,
    FeatureEvaluation,
    FeatureHypothesis,
    FoldMetrics,
    GenerationSource,
    HypothesisStatus,
    LeakageRisk,
    Lesson,
    MissingValuePolicy,
    PlayerClusterAssignment,
    SubgroupResult,
    ValidationStatus,
)
from xg_alonso.contracts.evidence import (
    EVIDENCE_PANEL_VERSION,
    EXPLANATORY_PANEL,
    FeatureEvidence,
    FeatureValue,
    PanelEntry,
    panel_feature_names,
)
from xg_alonso.contracts.folds import WalkForwardFold, walk_forward_folds
from xg_alonso.contracts.form import (
    FORM_SIGNAL_CLAMP,
    FormDirection,
    FormSignal,
    FormStrength,
    SignalSet,
)
from xg_alonso.contracts.identifiers import (
    EntryId,
    FixtureId,
    GameweekId,
    PlayerCode,
    PlayerElementId,
    Season,
    TeamCode,
    TeamId,
    TenthsOfMillion,
    format_money,
    parse_season,
)
from xg_alonso.contracts.objective import (
    BeliefEntity,
    BeliefProposition,
    ChipIntent,
    CompiledIntent,
    FeatureDiscoveryRequest,
    FieldConfidence,
    ManagerConstraints,
    ManagerObjective,
    ObjectiveBundle,
    OwnershipPreference,
    ParseSource,
    PrimaryMetric,
    RiskPreference,
    SquadArea,
    TeamQuota,
    UserBelief,
    UtilityWeights,
)
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    PointsBreakdown,
    Position,
)
from xg_alonso.contracts.provenance import (
    FOUR_TIMESTAMP_FIELDS,
    PredictionProvenance,
    RunManifest,
    SourceTimestamps,
    TimeSource,
    utc_now,
)
from xg_alonso.contracts.reason_codes import (
    REASON_TEMPLATES,
    Reason,
    ReasonCode,
    ReasonPolarity,
)
from xg_alonso.contracts.recommendation import (
    BaselineComparison,
    TransferMove,
    TransferPackage,
    TransferRecommendation,
)
from xg_alonso.contracts.schedule import (
    GameweekSlate,
    TeamFixture,
    blanking_teams,
    doubling_teams,
)
from xg_alonso.contracts.simulation import (
    AutosubResult,
    Captaincy,
    CaptainSource,
    GameweekSimulation,
    SeasonSimulation,
    SkippedSubstitution,
    SkipReason,
    Substitution,
    SubstitutionReason,
)
from xg_alonso.contracts.squad import ChipState, ChipStatus, SquadPick, SquadState
from xg_alonso.contracts.storage import BronzeSnapshotStore, SnapshotRef, TableStore

__all__ = [
    "EVIDENCE_PANEL_VERSION",
    "EXPLANATORY_PANEL",
    "FORM_SIGNAL_CLAMP",
    "FOUR_TIMESTAMP_FIELDS",
    "REASON_TEMPLATES",
    "AcceptanceStatus",
    "AutosubResult",
    "BaselineComparison",
    "BeliefEntity",
    "BeliefProposition",
    "BronzeSnapshotStore",
    "CaptainSource",
    "Captaincy",
    "ChipIntent",
    "ChipState",
    "ChipStatus",
    "ClusterSummary",
    "CompiledIntent",
    "ComplementarityClass",
    "ComponentExpectations",
    "DiscoveredFeatureSpec",
    "EntityLevel",
    "EntryId",
    "ExperimentManifest",
    "ExperimentStage",
    "FeatureDiscoveryRequest",
    "FeatureEvaluation",
    "FeatureEvidence",
    "FeatureHypothesis",
    "FeatureValue",
    "FieldConfidence",
    "FixtureId",
    "FoldMetrics",
    "FormDirection",
    "FormSignal",
    "FormStrength",
    "GameweekId",
    "GameweekSimulation",
    "GameweekSlate",
    "GenerationSource",
    "HypothesisStatus",
    "LeakageRisk",
    "Lesson",
    "ManagerConstraints",
    "ManagerObjective",
    "MinutesPrediction",
    "MissingValuePolicy",
    "ObjectiveBundle",
    "OwnershipPreference",
    "PanelEntry",
    "ParseSource",
    "PlayerClusterAssignment",
    "PlayerCode",
    "PlayerElementId",
    "PlayerPrediction",
    "PointsBreakdown",
    "Position",
    "PredictionProvenance",
    "PrimaryMetric",
    "Reason",
    "ReasonCode",
    "ReasonPolarity",
    "RiskPreference",
    "RunManifest",
    "Season",
    "SeasonSimulation",
    "SignalSet",
    "SkipReason",
    "SkippedSubstitution",
    "SnapshotRef",
    "SourceTimestamps",
    "SquadArea",
    "SquadPick",
    "SquadState",
    "SquadViolation",
    "SubgroupResult",
    "Substitution",
    "SubstitutionReason",
    "TableStore",
    "TeamCode",
    "TeamFixture",
    "TeamId",
    "TeamQuota",
    "TenthsOfMillion",
    "TimeSource",
    "TransferMove",
    "TransferPackage",
    "TransferRecommendation",
    "UserBelief",
    "UtilityWeights",
    "ValidationStatus",
    "WalkForwardFold",
    "blanking_teams",
    "doubling_teams",
    "format_money",
    "panel_feature_names",
    "parse_season",
    "utc_now",
    "walk_forward_folds",
]
