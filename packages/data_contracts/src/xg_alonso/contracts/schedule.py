"""A gameweek's fixtures, so blanks and doubles stop being special cases.

**The bug this type exists to make impossible.** Fixtures were attached to
players by concatenating the home and away sides and then taking
``.unique(subset=["team_id"], keep="first")`` after a kickoff sort. That is one
fixture per club by construction, which has two consequences and neither is
visible in any output:

- a club playing twice in a gameweek silently loses its **second** fixture, so
  the biggest scoring weeks in the game are modelled as ordinary ones;
- a club playing *no* fixture silently inherits its next one, from a later
  gameweek, so a blank reads as a normal week against a real opponent.

Here a blank is simply the absence of a row and a double is two rows. Neither
is a branch, which is the point: code that asks :meth:`GameweekSlate.for_team`
gets zero, one or two fixtures and cannot accidentally assume one.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from xg_alonso.contracts.identifiers import FixtureId, GameweekId, TeamId

__all__ = ["GameweekSlate", "TeamFixture", "blanking_teams", "doubling_teams"]


class TeamFixture(BaseModel):
    """One club's side of one fixture. Both clubs appear, once each."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: FixtureId
    gameweek: GameweekId
    team_id: TeamId
    opponent_team_id: TeamId
    was_home: bool
    kickoff_time: datetime | None = None


class GameweekSlate(BaseModel):
    """Every fixture in one gameweek, queryable by club."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gameweek: GameweekId
    fixtures: tuple[TeamFixture, ...] = ()

    @model_validator(mode="after")
    def _one_gameweek_no_duplicates(self) -> GameweekSlate:
        for fixture in self.fixtures:
            if fixture.gameweek != self.gameweek:
                raise ValueError(
                    f"fixture {fixture.fixture_id} is in gameweek {fixture.gameweek}, "
                    f"but this slate is gameweek {self.gameweek}"
                )
            if fixture.team_id == fixture.opponent_team_id:
                raise ValueError(f"fixture {fixture.fixture_id} has a club playing itself")
        seen = [(f.fixture_id, f.team_id) for f in self.fixtures]
        if len(set(seen)) != len(seen):
            raise ValueError("a club appears twice on the same side of one fixture")
        return self

    def for_team(self, team_id: TeamId) -> tuple[TeamFixture, ...]:
        """This club's fixtures, earliest first.

        Ordered by kickoff then fixture id so a double gameweek's two legs have
        a stable index. A null kickoff sorts last rather than first: an
        unscheduled fixture is not the earliest one.
        """
        rows = [f for f in self.fixtures if f.team_id == team_id]
        return tuple(
            sorted(rows, key=lambda f: (f.kickoff_time is None, f.kickoff_time, int(f.fixture_id)))
        )

    def fixture_count(self, team_id: TeamId) -> int:
        return len(self.for_team(team_id))

    def is_blank(self, team_id: TeamId) -> bool:
        return self.fixture_count(team_id) == 0

    def is_double(self, team_id: TeamId) -> bool:
        return self.fixture_count(team_id) >= 2

    @property
    def teams(self) -> frozenset[TeamId]:
        """Clubs with at least one fixture. Says nothing about who is missing."""
        return frozenset(f.team_id for f in self.fixtures)


def blanking_teams(slate: GameweekSlate, league: Collection[TeamId]) -> tuple[TeamId, ...]:
    """Clubs in ``league`` with no fixture in this slate.

    A free function rather than a property because a slate does not know the
    league: absence is only meaningful against a roster, and inferring one from
    the fixtures present would make every club that blanks invisible.
    """
    return tuple(sorted(set(league) - slate.teams))


def doubling_teams(slate: GameweekSlate) -> tuple[TeamId, ...]:
    """Clubs with two or more fixtures. Derivable from the slate alone."""
    return tuple(sorted(t for t in slate.teams if slate.is_double(t)))
