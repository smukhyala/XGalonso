"""Walk-forward evaluation.

Model metrics are intermediate. The question this package answers is the one
`CLAUDE.md` says matters: **does acting on the recommendations beat holding?**

A points-MAE can improve while decisions get worse, so the headline metric here
is a policy comparison — two identical squads walk the season, one transferring
and one not, both scored on the same actual results. The gap is the product.
"""

from xg_alonso.evaluation.accuracy import (
    PRICE_BANDS,
    AccuracyReport,
    AccuracySlice,
    TopKResult,
    score_predictions,
    spearman,
)
from xg_alonso.evaluation.backtest import (
    BacktestResult,
    GameweekOutcome,
    actual_points,
    apply_transfer,
    gameweek_deadlines,
    score_squad,
    walk_forward,
)
from xg_alonso.evaluation.importance import (
    IMPORTANCE_SCHEMA_VERSION,
    LABEL_TO_BREAKDOWN,
    FeatureImportance,
    ImportanceTable,
    label_weights_from_predictions,
    load_importance,
    permutation_importance,
    write_importance,
)
from xg_alonso.evaluation.policies import (
    POLICIES,
    PolicyName,
    highest_form_policy,
    hold_policy,
    model_policy,
    most_expensive_policy,
    random_policy,
    run_policy,
)
from xg_alonso.evaluation.report import (
    BacktestReport,
    compare_to_previous,
    load_reports,
    write_report,
)

__all__ = [
    "POLICIES",
    "IMPORTANCE_SCHEMA_VERSION",
    "LABEL_TO_BREAKDOWN",
    "PRICE_BANDS",
    "AccuracyReport",
    "AccuracySlice",
    "BacktestReport",
    "BacktestResult",
    "FeatureImportance",
    "GameweekOutcome",
    "ImportanceTable",
    "PolicyName",
    "TopKResult",
    "actual_points",
    "apply_transfer",
    "compare_to_previous",
    "gameweek_deadlines",
    "highest_form_policy",
    "label_weights_from_predictions",
    "hold_policy",
    "load_importance",
    "load_reports",
    "model_policy",
    "most_expensive_policy",
    "permutation_importance",
    "random_policy",
    "run_policy",
    "score_predictions",
    "score_squad",
    "spearman",
    "walk_forward",
    "write_importance",
    "write_report",
]
