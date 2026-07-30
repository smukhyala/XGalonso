"""Model training and inference.

Consumes approved feature sets, never raw data. Predicts **components** rather
than points, per decision D8: the domain layer owns scoring, so a rules change
re-scores existing predictions without retraining anything.

Slice 1 ships a closed-form baseline. It is not accurate; it is defensible,
reproducible, and honest about its uncertainty, which is what the optimizer
above it needs in order to be judged against holding.
"""

from xg_alonso.prediction.availability import CERTAIN, apply_availability, availability_factor
from xg_alonso.prediction.baseline import (
    BASELINE_NAME,
    BASELINE_VERSION,
    estimator_fingerprint,
    predict_frame,
    predict_player,
)
from xg_alonso.prediction.calibration import (
    CALIBRATION_VERSION,
    PRICE_BAND_BIAS,
    apply_price_calibration,
)
from xg_alonso.prediction.dataset import (
    COMPONENT_LABELS,
    TrainingData,
    build_training_frame,
)
from xg_alonso.prediction.evidence import attach_feature_evidence, build_feature_evidence
from xg_alonso.prediction.form import apply_form_signals, form_reason, load_signals
from xg_alonso.prediction.inference import (
    SavedModel,
    load_models,
    model_summary,
    predict_with_models,
    save_models,
)
from xg_alonso.prediction.refresh import (
    DEFAULT_REFRESH_BUDGET,
    RefreshPlan,
    RefreshRequest,
    plan_refresh,
)
from xg_alonso.prediction.trained import (
    TRAINED_MODEL_NAME,
    TRAINED_MODEL_VERSION,
    ComponentModels,
    FoldReport,
    train_component_models,
)

__all__ = [
    "BASELINE_NAME",
    "BASELINE_VERSION",
    "CALIBRATION_VERSION",
    "CERTAIN",
    "COMPONENT_LABELS",
    "DEFAULT_REFRESH_BUDGET",
    "PRICE_BAND_BIAS",
    "TRAINED_MODEL_NAME",
    "TRAINED_MODEL_VERSION",
    "ComponentModels",
    "FoldReport",
    "RefreshPlan",
    "RefreshRequest",
    "SavedModel",
    "TrainingData",
    "apply_availability",
    "apply_form_signals",
    "apply_price_calibration",
    "attach_feature_evidence",
    "availability_factor",
    "build_feature_evidence",
    "build_training_frame",
    "estimator_fingerprint",
    "form_reason",
    "load_models",
    "load_signals",
    "model_summary",
    "plan_refresh",
    "predict_frame",
    "predict_player",
    "predict_with_models",
    "save_models",
    "train_component_models",
]
