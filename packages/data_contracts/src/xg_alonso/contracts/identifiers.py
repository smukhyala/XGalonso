"""Identifier and unit types shared across every package.

Two conventions here are load-bearing and easy to get wrong:

**Player identity.** FPL reissues `element.id` every season, so it is useless as a
cross-season key. `element.code` is stable across seasons and is therefore the
canonical player identity. Anything that needs to join a player to their history
uses :class:`PlayerCode`, never the per-season element id.

**Money.** FPL prices are integers in tenths of a million: ``55`` means £5.5m.
Budgets, selling prices and hit costs are all exact integer arithmetic. Never
represent money as a float — the selling-price rule halves profits and rounds,
and float drift there produces recommendations that are one-tenth infeasible.
"""

from __future__ import annotations

from typing import NewType

__all__ = [
    "EntryId",
    "FixtureId",
    "GameweekId",
    "PlayerCode",
    "PlayerElementId",
    "Season",
    "TeamCode",
    "TeamId",
    "TenthsOfMillion",
    "format_money",
    "parse_season",
]

# --- Entity identifiers ----------------------------------------------------

PlayerCode = NewType("PlayerCode", int)
"""Stable cross-season player identity (`element.code`). The canonical player key."""

PlayerElementId = NewType("PlayerElementId", int)
"""Per-season FPL element id. Reissued each season — never use as a join key."""

TeamId = NewType("TeamId", int)
"""Per-season Premier League club id (1-20, reassigned each season)."""

TeamCode = NewType("TeamCode", int)
"""Stable cross-season club identity (`team.code`)."""

FixtureId = NewType("FixtureId", int)
GameweekId = NewType("GameweekId", int)

EntryId = NewType("EntryId", int)
"""A human manager's FPL team. Distinct from :class:`TeamId`, which is a club.

The public API uses `team_id` for both, which is ambiguous. This codebase does
not: an *entry* is a manager, a *team* is a club, a *player* is a footballer.
"""

Season = NewType("Season", str)
"""A season in `YYYY-YY` form, e.g. ``"2025-26"``."""


# --- Money -----------------------------------------------------------------

TenthsOfMillion = NewType("TenthsOfMillion", int)
"""Money in tenths of a million. ``55`` is £5.5m. Always an exact integer."""


def format_money(value: TenthsOfMillion) -> str:
    """Render money for display: ``55`` becomes ``"£5.5m"``."""
    return f"£{value / 10:.1f}m"


def parse_season(value: str) -> Season:
    """Validate and normalise a season string.

    Raises:
        ValueError: if the value is not `YYYY-YY` with consecutive years.
    """
    parts = value.split("-")
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
        raise ValueError(f"season must look like '2025-26', got {value!r}")
    try:
        start = int(parts[0])
        end_short = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"season must be numeric, got {value!r}") from exc
    if (start + 1) % 100 != end_short:
        raise ValueError(f"season years must be consecutive, got {value!r}")
    return Season(value)
