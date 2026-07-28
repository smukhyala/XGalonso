"""Decide which players are worth spending a lookup on, and when.

**The cost problem, stated plainly.** There are roughly 600 selectable players.
Searching for each of them every gameweek is 600 lookups a week, most of them
about players nobody will ever own, returning nothing. That is the naive design
and it is unaffordable in both money and latency, so this module exists to make
the refresh *sublinear in the squad* rather than linear in the league.

Three ideas do the work, in order of how much they save:

1. **Ask about teams, not players.** Team news is written per club — one article
   covers a manager's whole selection. Twenty clubs replaces six hundred
   players, a 30x reduction, and it is a better match for how the information
   is actually published.
2. **Only ask about players who can change a decision.** A player nobody owns,
   who no optimizer would buy, and whom the squad cannot afford, cannot alter
   the recommendation however out of form he is. Priority is computed from what
   the decision layer would actually consider.
3. **Only ask when the answer could have changed.** A signal already on file and
   not near expiry is not worth re-fetching, and a player nothing has ever been
   written about is worth asking less often than one in the middle of a story.

The result is a bounded work-list — never more than ``budget`` lookups — ordered
so that if the budget runs out mid-run, what was skipped is what mattered least.

This module performs **no I/O and knows no provider.** It decides *what* to ask
about. Whatever fetches the answers lives outside, which is what keeps the
prediction path free of a network dependency (D6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from xg_alonso.contracts.form import SignalSet
from xg_alonso.contracts.identifiers import PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import PlayerPrediction

__all__ = [
    "DEFAULT_REFRESH_BUDGET",
    "RefreshPlan",
    "RefreshRequest",
    "plan_refresh",
]

#: Lookups per run. Twenty is one per club, which covers the league exactly once
#: at the granularity team news is published in.
DEFAULT_REFRESH_BUDGET: Final[int] = 20

#: Refresh a signal once it is within this of expiring. Earlier is wasted work;
#: later leaves a window where the file is empty and the projection silently
#: reverts.
_STALE_WITHIN: Final[timedelta] = timedelta(days=3)

#: A player must project at least this to be worth asking about at all. Below it
#: he will not be started or bought, so no story about him changes a decision.
_RELEVANCE_FLOOR: Final[float] = 1.5


@dataclass(frozen=True)
class RefreshRequest:
    """One lookup worth making, and why it is worth making."""

    team_id: TeamId
    players: tuple[PlayerCode, ...]
    """Players on this club the decision layer would actually consider."""

    priority: float
    """Higher is more worth spending the budget on."""

    reason: str
    """Why this club made the list, for the run log."""


@dataclass(frozen=True)
class RefreshPlan:
    """A bounded work-list, plus what it deliberately left out."""

    requests: tuple[RefreshRequest, ...]
    skipped_teams: int
    covered_players: int
    total_players: int

    @property
    def lookups(self) -> int:
        return len(self.requests)

    def describe(self) -> str:
        """One line for the run log.

        States the coverage as a fraction rather than an absolute, because "20
        lookups" reads as thorough and "20 lookups covering 143 of 612 players"
        reads as what it is.
        """
        return (
            f"{self.lookups} lookups covering {self.covered_players} of "
            f"{self.total_players} players; {self.skipped_teams} clubs skipped as "
            "unable to change a decision"
        )


def plan_refresh(
    predictions: Sequence[PlayerPrediction],
    *,
    teams: Mapping[PlayerCode, TeamId],
    prices: Mapping[PlayerCode, TenthsOfMillion],
    signals: SignalSet,
    at: datetime,
    owned: frozenset[PlayerCode] = frozenset(),
    affordable_at: TenthsOfMillion | None = None,
    budget: int = DEFAULT_REFRESH_BUDGET,
    relevance_floor: float = _RELEVANCE_FLOOR,
) -> RefreshPlan:
    """Choose which clubs to look up, within a fixed budget.

    Args:
        predictions: Everyone the model has an opinion about.
        teams: Which club each player belongs to.
        prices: Current prices, used to drop players the squad cannot reach.
        signals: What is already on file, so live signals are not re-fetched.
        at: The moment to judge staleness against.
        owned: The manager's own players. Always worth asking about — a slump in
            someone you already own changes a decision immediately, where a
            slump in someone you do not merely changes a shortlist.
        affordable_at: The most the squad could spend on one player. Players
            above it cannot be bought, so a story about them changes nothing.
        budget: Maximum lookups.
        relevance_floor: Minimum projection before a player is worth asking
            about at all.

    Returns:
        The plan, ordered most valuable first and truncated to ``budget``.
    """
    live = signals.live(at)
    fresh_enough = {code for code, signal in live.items() if signal.expires_at - at > _STALE_WITHIN}

    by_team: dict[TeamId, list[tuple[PlayerCode, float]]] = {}
    considered = 0

    for prediction in predictions:
        code = prediction.player_code
        team = teams.get(code)
        if team is None:
            continue

        is_owned = code in owned
        if not is_owned:
            if prediction.expected_points < relevance_floor:
                continue
            price = prices.get(code)
            if affordable_at is not None and price is not None and price > affordable_at:
                continue

        considered += 1
        if code in fresh_enough:
            # Already covered and not near expiry. Counted as considered so the
            # coverage figure reflects the squad, not just this run's fetches.
            continue

        # Owned players are worth more than the same projection unowned, because
        # acting on them needs no transfer to become relevant.
        weight = prediction.expected_points * (2.0 if is_owned else 1.0)
        by_team.setdefault(team, []).append((code, weight))

    requests: list[RefreshRequest] = []
    for team, members in by_team.items():
        members.sort(key=lambda pair: -pair[1])
        owned_here = sum(1 for code, _ in members if code in owned)
        requests.append(
            RefreshRequest(
                team_id=team,
                players=tuple(code for code, _ in members),
                # Summed rather than averaged: a club with six relevant players
                # is a better use of one lookup than a club with one, and one
                # article covers all six either way.
                priority=sum(weight for _, weight in members),
                reason=(
                    f"{len(members)} relevant players"
                    + (f", {owned_here} owned" if owned_here else "")
                ),
            )
        )

    requests.sort(key=lambda request: (-request.priority, int(request.team_id)))
    kept = requests[:budget]

    return RefreshPlan(
        requests=tuple(kept),
        skipped_teams=len(requests) - len(kept),
        covered_players=sum(len(request.players) for request in kept),
        total_players=considered,
    )
