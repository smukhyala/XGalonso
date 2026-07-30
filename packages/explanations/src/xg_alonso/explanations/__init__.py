"""Converts structured evidence into user-facing reasoning.

An LLM may rewrite a grounded sentence for readability. It may never invent a
cause or a statistic — and here that is a structural guarantee rather than a
policy. A Reason cannot be constructed unless its evidence satisfies every
placeholder in its template, so no renderer is ever handed a gap to fill.
"""

from xg_alonso.explanations.context import (
    FixtureRun,
    PlayerContext,
    ScheduledFixture,
    SeasonLine,
    build_fixture_run,
    build_player_context,
    build_season_lines,
)
from xg_alonso.explanations.derivation import DerivationLine, PointsDerivation, derive_points
from xg_alonso.explanations.history import (
    HistoryNote,
    Meeting,
    build_history_notes,
)
from xg_alonso.explanations.lineup_diff import (
    LineupComparison,
    PlayerSwap,
    compare_lineups,
    selection_from_starters,
)
from xg_alonso.explanations.render import render_recommendation, render_squad_summary

__all__ = [
    "DerivationLine",
    "FixtureRun",
    "HistoryNote",
    "LineupComparison",
    "Meeting",
    "PlayerContext",
    "PlayerSwap",
    "PointsDerivation",
    "ScheduledFixture",
    "SeasonLine",
    "build_fixture_run",
    "build_history_notes",
    "build_player_context",
    "build_season_lines",
    "compare_lineups",
    "derive_points",
    "render_recommendation",
    "render_squad_summary",
    "selection_from_starters",
]
