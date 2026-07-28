"""Qualitative history: what this player has actually done in this situation.

**What this adds that nothing else does.** Every other explanation in the system
is derived from the model — a projection, a component, a percentile. Those say
*how much* and never *when*. A manager weighing a captain does not think "0.61
expected goals per 90"; he thinks "he scored twice at Stamford Bridge last
season and he always turns up on the opening weekend". Those are checkable facts
about matches that happened, and the system held all of them and surfaced none.

**These are retrievals, not inferences.** Every sentence below is assembled from
rows in the canonical stats table, with the season, the gameweek and the score
line attached. Nothing is predicted here and nothing is smoothed: if a player
has faced this opponent twice, the note says twice and gives both results. That
makes it the one part of the explanation layer a user can verify against a
scoreboard, which is exactly why it is worth having next to the modelled parts.

**Point-in-time safety still applies.** Only rows whose ``available_time``
precedes the prediction cutoff are read, so a note never cites a match that had
not been played when the decision was made.

Three situations are covered, chosen because they are the three a manager
actually raises out loud:

- **This opponent.** What he has done against them, and at this venue.
- **This gameweek number.** Opening weekends and festive fixtures behave
  differently, and some players reliably turn up in them.
- **This venue.** Home and away are different games for some players.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import polars as pl

__all__ = [
    "HistoryNote",
    "Meeting",
    "build_history_notes",
]

#: Matches to cite individually before summarising instead. Three keeps the note
#: a sentence rather than a table; beyond that a reader stops reading and the
#: aggregate is the more useful form anyway.
_MAX_CITED: Final[int] = 3

#: Appearances required before a venue or gameweek split is worth stating. Two
#: matches is an anecdote, and presenting it as a pattern is the failure mode
#: this kind of note is most prone to.
_MIN_SAMPLE: Final[int] = 3


@dataclass(frozen=True)
class Meeting:
    """One past match, as it happened."""

    season: str
    gameweek: int
    was_home: bool
    minutes: int
    goals: int
    assists: int
    points: int

    @property
    def returned(self) -> bool:
        return self.goals > 0 or self.assists > 0

    def describe(self) -> str:
        """The score line, in the words a person would use."""
        where = "at home" if self.was_home else "away"
        if self.minutes == 0:
            return f"{self.season} ({where}): did not play"

        parts: list[str] = []
        if self.goals:
            parts.append(f"{self.goals} goal{'s' if self.goals > 1 else ''}")
        if self.assists:
            parts.append(f"{self.assists} assist{'s' if self.assists > 1 else ''}")
        outcome = " and ".join(parts) if parts else "no returns"
        return f"{self.season} ({where}): {outcome}, {self.points} pts"


@dataclass(frozen=True)
class HistoryNote:
    """One checkable statement about a player's record in this situation."""

    kind: str
    text: str
    strength: float
    """How notable this is, for ordering. Not a probability and not a weight in
    the model — nothing here feeds a projection."""

    meetings: tuple[Meeting, ...] = ()

    @property
    def is_positive(self) -> bool:
        return self.strength > 0


def _meetings(rows: list[dict[str, Any]]) -> tuple[Meeting, ...]:
    return tuple(
        Meeting(
            season=str(row["season"]),
            gameweek=int(row["gameweek_id"] or 0),
            was_home=bool(row["was_home"]),
            minutes=int(row["minutes"] or 0),
            goals=int(row["goals_scored"] or 0),
            assists=int(row["assists"] or 0),
            points=int(row["total_points"] or 0),
        )
        for row in rows
    )


def _opponent_note(played: tuple[Meeting, ...], opponent: str) -> HistoryNote | None:
    """What he has done against this club."""
    appearances = [m for m in played if m.minutes > 0]
    if not appearances:
        return None

    goals = sum(m.goals for m in appearances)
    assists = sum(m.assists for m in appearances)
    points = sum(m.points for m in appearances)
    average = points / len(appearances)

    recent = sorted(appearances, key=lambda m: (m.season, m.gameweek), reverse=True)[:_MAX_CITED]
    lines = "; ".join(m.describe() for m in recent)

    if goals or assists:
        involvement = []
        if goals:
            involvement.append(f"{goals} goal{'s' if goals > 1 else ''}")
        if assists:
            involvement.append(f"{assists} assist{'s' if assists > 1 else ''}")
        summary = (
            f"Against {opponent} he has {' and '.join(involvement)} in "
            f"{len(appearances)} appearance{'s' if len(appearances) > 1 else ''}, "
            f"averaging {average:.1f} points. Most recently — {lines}."
        )
    else:
        summary = (
            f"Against {opponent} he has no goals or assists in "
            f"{len(appearances)} appearance{'s' if len(appearances) > 1 else ''}, "
            f"averaging {average:.1f} points. Most recently — {lines}."
        )

    # Centred on two points a match, roughly a blank with an appearance.
    return HistoryNote(
        kind="opponent",
        text=summary,
        strength=average - 2.0,
        meetings=tuple(recent),
    )


def _gameweek_note(played: tuple[Meeting, ...], gameweek: int) -> HistoryNote | None:
    """What he has done in this gameweek number in previous seasons."""
    appearances = [m for m in played if m.minutes > 0]
    if len(appearances) < 2:
        return None

    goals = sum(m.goals for m in appearances)
    assists = sum(m.assists for m in appearances)
    points = sum(m.points for m in appearances)
    average = points / len(appearances)
    returning = sum(1 for m in appearances if m.returned)

    label = "opening weekend" if gameweek == 1 else f"gameweek {gameweek}"
    lines = "; ".join(
        m.describe() for m in sorted(appearances, key=lambda m: m.season, reverse=True)
    )

    summary = (
        f"On the {label} he has returned in {returning} of his last "
        f"{len(appearances)} seasons — {goals} goal{'s' if goals != 1 else ''} and "
        f"{assists} assist{'s' if assists != 1 else ''}, averaging {average:.1f} points. "
        f"{lines}."
    )

    # **The marker has to agree with the sentence.** Scoring this note on points
    # alone marked "returned in 0 of his last 4 seasons, averaging 2.5 points" as
    # a point in the player's favour, because 2.5 clears an appearance. The
    # sentence leads with the return count, so the strength must too: a record
    # with no returns in it is never encouraging, whatever the appearance points
    # add up to.
    strength = average - 2.0
    if returning == 0:
        strength = -max(abs(strength), 1.0)

    return HistoryNote(kind="gameweek", text=summary, strength=strength)


def _venue_note(played: tuple[Meeting, ...], was_home: bool) -> HistoryNote | None:
    """Whether this venue suits him, when there is enough evidence to say."""
    here = [m for m in played if m.minutes > 0 and m.was_home == was_home]
    there = [m for m in played if m.minutes > 0 and m.was_home != was_home]
    if len(here) < _MIN_SAMPLE or len(there) < _MIN_SAMPLE:
        return None

    home_rate = sum(m.points for m in here) / len(here)
    away_rate = sum(m.points for m in there) / len(there)
    gap = home_rate - away_rate
    # Below a point a match the split is noise dressed as a tendency.
    if abs(gap) < 1.0:
        return None

    venue = "at home" if was_home else "away"
    other = "away" if was_home else "at home"
    direction = "better" if gap > 0 else "worse"
    return HistoryNote(
        kind="venue",
        text=(
            f"He is {direction} {venue}: {home_rate:.1f} points a match across "
            f"{len(here)} appearances, against {away_rate:.1f} {other}."
        ),
        strength=gap,
    )


def build_history_notes(
    player_stats: pl.DataFrame,
    *,
    fixtures: dict[int, tuple[int, bool]],
    gameweek: int,
    team_names: dict[int, str],
    cutoff: Any,
    player_codes: list[int] | None = None,
) -> dict[int, list[HistoryNote]]:
    """Assemble history notes for every player, in one pass over the table.

    Args:
        player_stats: Canonical ``player_gameweek_stats``.
        fixtures: Player code to ``(opponent_team_id, was_home)`` for the match
            being predicted.
        gameweek: The gameweek number being predicted.
        team_names: Team id to display name.
        cutoff: Only rows available strictly before this are read.
        player_codes: Restrict to these players. All of them when omitted.

    Returns:
        Player code to notes, most notable first.

    Batched deliberately. Filtering the whole table once and grouping is a
    single scan; asking per player would be six hundred scans of the same
    hundred thousand rows to answer the same question.
    """
    required = {"player_code", "season", "gameweek_id", "opponent_team_id", "was_home"}
    missing = sorted(required - set(player_stats.columns))
    if missing:
        raise KeyError(f"player_stats is missing columns required for history: {missing}")

    visible = player_stats.filter(pl.col("available_time") < cutoff)
    if player_codes is not None:
        visible = visible.filter(pl.col("player_code").is_in(player_codes))
    if visible.is_empty():
        return {}

    wanted = list(fixtures) if player_codes is None else player_codes
    by_player: dict[int, list[HistoryNote]] = {}

    columns = [
        "player_code",
        "season",
        "gameweek_id",
        "opponent_team_id",
        "was_home",
        "minutes",
        "goals_scored",
        "assists",
        "total_points",
    ]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in visible.select(columns).iter_rows(named=True):
        grouped.setdefault(int(row["player_code"]), []).append(row)

    for code in wanted:
        fixture = fixtures.get(code)
        rows = grouped.get(code)
        if fixture is None or not rows:
            continue
        opponent_id, was_home = fixture
        opponent = team_names.get(opponent_id, "them")

        notes: list[HistoryNote] = []
        against = _meetings([r for r in rows if int(r["opponent_team_id"] or 0) == opponent_id])
        if against:
            note = _opponent_note(against, opponent)
            if note is not None:
                notes.append(note)

        same_week = _meetings([r for r in rows if int(r["gameweek_id"] or 0) == gameweek])
        if same_week:
            note = _gameweek_note(same_week, gameweek)
            if note is not None:
                notes.append(note)

        venue = _venue_note(_meetings(rows), was_home)
        if venue is not None:
            notes.append(venue)

        if notes:
            notes.sort(key=lambda n: -abs(n.strength))
            by_player[code] = notes

    return by_player
