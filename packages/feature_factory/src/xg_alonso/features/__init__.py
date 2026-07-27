"""Deterministic, point-in-time-safe candidate feature generation.

The Feature Factory generates candidates; it does not decide which are useful.
That judgement belongs to the Feature Scientist, which is deferred per D10.

Everything here obeys one rule: **a feature may use only information whose
``available_time`` precedes the prediction timestamp.** The rule is enforced
mechanically rather than by convention — :mod:`~xg_alonso.features.leakage`
rebuilds features with future records appended and fails if any value moved,
and its negative control proves the harness itself still has teeth.
"""

from xg_alonso.features.generators import rolling_as_of, shrunk_rate_as_of
from xg_alonso.features.leakage import (
    LeakageDetected,
    assert_detects_leakage,
    assert_no_leakage,
    find_leakage,
    make_future_records,
)
from xg_alonso.features.point_in_time import (
    as_of_join,
    filter_available,
    point_in_time_join,
)
from xg_alonso.features.slice1 import (
    SLICE1_FEATURE_SET_VERSION,
    SLICE1_FEATURES,
    build_slice1_features,
    build_team_gameweek_stats,
)

__all__ = [
    "SLICE1_FEATURES",
    "SLICE1_FEATURE_SET_VERSION",
    "LeakageDetected",
    "as_of_join",
    "assert_detects_leakage",
    "assert_no_leakage",
    "build_slice1_features",
    "build_team_gameweek_stats",
    "filter_available",
    "find_leakage",
    "make_future_records",
    "point_in_time_join",
    "rolling_as_of",
    "shrunk_rate_as_of",
]
