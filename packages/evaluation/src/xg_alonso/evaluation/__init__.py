"""Walk-forward evaluation.

Model metrics are intermediate. The question this package answers is the one
`CLAUDE.md` says matters: **does acting on the recommendations beat holding?**

A points-MAE can improve while decisions get worse, so the headline metric here
is a policy comparison — two identical squads walk the season, one transferring
and one not, both scored on the same actual results. The gap is the product.
"""

from xg_alonso.evaluation.backtest import (
    BacktestResult,
    GameweekOutcome,
    actual_points,
    apply_transfer,
    gameweek_deadlines,
    score_squad,
    walk_forward,
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
    "BacktestReport",
    "BacktestResult",
    "GameweekOutcome",
    "PolicyName",
    "actual_points",
    "apply_transfer",
    "compare_to_previous",
    "gameweek_deadlines",
    "highest_form_policy",
    "hold_policy",
    "load_reports",
    "model_policy",
    "most_expensive_policy",
    "random_policy",
    "run_policy",
    "score_squad",
    "walk_forward",
    "write_report",
]
