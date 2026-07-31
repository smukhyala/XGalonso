"""Rule drift detection.

Constants are read from a pinned snapshot, which solves transcription errors but
introduces a new failure: the pin going stale. If FPL revalues an assist and we
keep scoring the old way, every prediction is quietly wrong and nothing crashes.

So every ingest re-reads the live rules and compares them against the pin. A
difference is not an error to swallow — it is a rules change that invalidates
stored predictions and must be surfaced before anything is recomputed.

These functions are pure comparisons over already-parsed objects, so the pipeline
does the I/O and the domain does the judging.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringRules

__all__ = ["RuleChange", "diff_scoring_rules", "diff_squad_rules"]


class RuleChange(BaseModel):
    """One constant that moved between the pinned snapshot and the live payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str
    pinned: str
    live: str

    def __str__(self) -> str:
        return f"{self.field}: pinned={self.pinned} live={self.live}"


def _compare(
    pinned: BaseModel, live: BaseModel, fields: tuple[str, ...], prefix: str = ""
) -> list[RuleChange]:
    changes: list[RuleChange] = []
    for name in fields:
        was = getattr(pinned, name)
        now = getattr(live, name)
        if was != now:
            changes.append(RuleChange(field=f"{prefix}{name}", pinned=repr(was), live=repr(now)))
    return changes


#: Scoring fields whose change invalidates every stored prediction.
_SCORING_FIELDS: tuple[str, ...] = (
    "goals_scored",
    "clean_sheets",
    "goals_conceded",
    "defensive_contribution",
    "assists",
    "saves",
    "bonus",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "short_play",
    "long_play",
)

#: Constraint fields whose change invalidates every stored recommendation.
_SQUAD_FIELDS: tuple[str, ...] = (
    "squad_size",
    "starting_size",
    "max_per_club",
    "total_budget",
    "sell_on_fee",
    "sell_at_purchase_price",
    "max_extra_free_transfers",
    "vice_captain_enabled",
    "transfers_cap",
    "positions",
)


def diff_scoring_rules(pinned: ScoringRules, live: ScoringRules) -> list[RuleChange]:
    """Scoring values that changed. Empty means the pin is still valid.

    ``version``, ``source_sha256`` and ``fetched_at`` are excluded — they differ
    by construction on every fetch and say nothing about the rules themselves.
    """
    return _compare(pinned, live, _SCORING_FIELDS)


def diff_squad_rules(pinned: SquadRules, live: SquadRules) -> list[RuleChange]:
    """Squad and transfer constants that changed. Empty means the pin is valid."""
    return _compare(pinned, live, _SQUAD_FIELDS)
