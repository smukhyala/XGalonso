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
"""

from xg_alonso.contracts.evidence import (
    EVIDENCE_PANEL_VERSION,
    EXPLANATORY_PANEL,
    FeatureEvidence,
    FeatureValue,
    PanelEntry,
    panel_feature_names,
)
from xg_alonso.contracts.folds import WalkForwardFold, walk_forward_folds
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
from xg_alonso.contracts.squad import ChipState, ChipStatus, SquadPick, SquadState
from xg_alonso.contracts.storage import BronzeSnapshotStore, SnapshotRef, TableStore

__all__ = [
    "EVIDENCE_PANEL_VERSION",
    "EXPLANATORY_PANEL",
    "FOUR_TIMESTAMP_FIELDS",
    "REASON_TEMPLATES",
    "BaselineComparison",
    "BronzeSnapshotStore",
    "ChipState",
    "ChipStatus",
    "ComponentExpectations",
    "EntryId",
    "FeatureEvidence",
    "FeatureValue",
    "FixtureId",
    "GameweekId",
    "MinutesPrediction",
    "PanelEntry",
    "PlayerCode",
    "PlayerElementId",
    "PlayerPrediction",
    "PointsBreakdown",
    "Position",
    "PredictionProvenance",
    "Reason",
    "ReasonCode",
    "ReasonPolarity",
    "RunManifest",
    "Season",
    "SnapshotRef",
    "SourceTimestamps",
    "SquadPick",
    "SquadState",
    "TableStore",
    "TeamCode",
    "TeamId",
    "TenthsOfMillion",
    "TimeSource",
    "TransferMove",
    "TransferPackage",
    "TransferRecommendation",
    "WalkForwardFold",
    "format_money",
    "panel_feature_names",
    "parse_season",
    "utc_now",
    "walk_forward_folds",
]
