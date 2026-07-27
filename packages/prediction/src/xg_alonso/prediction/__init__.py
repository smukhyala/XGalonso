"""Model training and inference.

Consumes approved feature sets, never raw data. Predicts **components** rather
than points, per decision D8: the domain layer owns scoring, so a rules change
re-scores existing predictions without retraining anything.

Slice 1 ships a closed-form baseline. It is not accurate; it is defensible,
reproducible, and honest about its uncertainty, which is what the optimizer
above it needs in order to be judged against holding.
"""

from xg_alonso.prediction.baseline import (
    BASELINE_NAME,
    BASELINE_VERSION,
    estimator_fingerprint,
    predict_frame,
    predict_player,
)

__all__ = [
    "BASELINE_NAME",
    "BASELINE_VERSION",
    "estimator_fingerprint",
    "predict_frame",
    "predict_player",
]
