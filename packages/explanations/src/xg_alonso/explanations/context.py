"""Where a player is coming from, and what he is walking into.

**The gap this fills.** Every number the product shows is about *next* week. A
projection, a component breakdown, a percentile — all of them describe a single
upcoming fixture, and a manager comparing two players on expected goals alone is
being asked to trust a decimal with no account of how it was arrived at. The
first two questions anyone actually asks are missing:

- *What did he do last season?* Eighteen goals in 2,800 minutes and eight goals
  in 900 are different players, and the projection alone does not say which one
  it is describing.
- *What is his opening run like?* A striker facing three of the promoted sides
  in the first five is not the same asset as one opening against last season's
  top four, however identical their per-match projections.

**Retrievals, not inferences.** Like :mod:`.history`, everything here is
assembled from rows that exist. A season line is a sum over played matches. A
fixture run is the fixture list with the difficulty the API itself publishes.
Nothing is modelled, nothing is smoothed, and every figure can be checked
against a scoreboard or a fixture list — which is the point, because this sits
beside modelled numbers and has to be distinguishable from them.

**Point-in-time safety still applies.** Season lines read only rows whose
``available_time`` precedes the cutoff, so a summary never counts a match that
had not been played when the decision was made. Fixture runs are the exception
and deliberately so: a *scheduled* fixture is knowable in advance, which is what
makes it usable. Only the schedule and the published difficulty are read, never
a result.

**Per-90 rates are withheld below a minutes floor.** A player with 90 minutes
and one goal has a rate of 1.00 per 90, and printing that beside a 0.61 earned
over 2,800 minutes invites exactly the wrong comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import polars as pl

__all__ = [
    "MIN_MINUTES_FOR_RATE",
    "FixtureRun",
    "PlayerContext",
    "ScheduledFixture",
    "SeasonLine",
    "build_fixture_run",
    "build_player_context",
    "build_season_lines",
]

#: Minutes a player must have played before a per-90 rate is reported.
#:
#: Roughly five full matches. Below it the rate is dominated by whether one shot
#: went in, and a headline rate computed from a cameo is the most misleading
#: figure this module could publish.
MIN_MINUTES_FOR_RATE: Final[int] = 450

#: Fixtures that constitute an "opening run". Five is what managers plan around
#: — long enough for a hard patch to matter, short enough to still be a decision
#: about the squad you start with rather than a season forecast.
DEFAULT_RUN_LENGTH: Final[int] = 5

#: FPL publishes difficulty on a 1-5 scale. Held as a constant because the
#: midpoint is what "an average run" is measured against, and hard-coding 3
#: in three places is how that quietly becomes 2.5 in one of them.
_DIFFICULTY_MIDPOINT: Final[float] = 3.0


def _count(value: int, noun: str) -> str:
    """``1 goal`` / ``4 goals``. These strings are read by people, not parsed."""
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


@dataclass(frozen=True)
class SeasonLine:
    """What a player produced in one season. Counted, not modelled."""

    season: str
    appearances: int
    minutes: int
    goals: int
    assists: int
    clean_sheets: int
    points: int
    expected_goals: float | None
    expected_assists: float | None

    @property
    def involvements(self) -> int:
        return self.goals + self.assists

    @property
    def per_90(self) -> float | None:
        """Goal involvements per 90 minutes, or ``None`` below the floor.

        ``None`` rather than a number, because the alternative is publishing a
        rate from a sample that cannot support one and trusting the reader to
        discount it. They will not.
        """
        if self.minutes < MIN_MINUTES_FOR_RATE:
            return None
        return round(self.involvements * 90.0 / self.minutes, 2)

    @property
    def points_per_appearance(self) -> float | None:
        if self.appearances <= 0:
            return None
        return round(self.points / self.appearances, 2)

    def sentence(self, position: str) -> str:
        """One line, framed for what this position is actually paid for.

        A defender's season is not a striker's season with fewer goals. Leading
        a centre-back's summary with his goal count describes the least
        important thing he did, so each position leads with the return that
        dominates its points.
        """
        minutes = f"{self.minutes:,} minutes"
        if position == "GKP":
            head = _count(self.clean_sheets, "clean sheet")
        elif position == "DEF":
            head = f"{_count(self.clean_sheets, 'clean sheet')}, " + _count(
                self.involvements, "attacking return"
            )
        elif position == "MID":
            head = f"{_count(self.goals, 'goal')} and {_count(self.assists, 'assist')}"
        else:
            head = _count(self.goals, "goal")

        rate = self.per_90
        tail = f", {rate:.2f} involvements per 90" if rate is not None else ""
        return f"{self.season}: {head} in {minutes} ({self.points} points){tail}"


@dataclass(frozen=True)
class ScheduledFixture:
    """One upcoming match, from the published schedule."""

    gameweek: int
    opponent: str
    is_home: bool
    difficulty: int | None

    @property
    def label(self) -> str:
        """``BUR (H)`` — the form every fixture ticker in the game uses."""
        return f"{self.opponent} ({'H' if self.is_home else 'A'})"


@dataclass(frozen=True)
class FixtureRun:
    """The next few fixtures, characterised."""

    fixtures: tuple[ScheduledFixture, ...]
    blanks: tuple[int, ...]
    doubles: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.fixtures)

    @property
    def home_count(self) -> int:
        return sum(1 for fixture in self.fixtures if fixture.is_home)

    @property
    def mean_difficulty(self) -> float | None:
        """Average published difficulty, or ``None`` when none is published.

        Preseason the API returns zeroes for the strength fields; a run of
        zeroes would average to 0.0 and read as the easiest schedule in the
        league, so unrated fixtures are excluded rather than counted.
        """
        rated = [f.difficulty for f in self.fixtures if f.difficulty]
        if not rated:
            return None
        return round(sum(rated) / len(rated), 2)

    def sentence(self) -> str:
        """A description a manager would recognise, hedged where it should be."""
        if not self.fixtures:
            return "No fixtures scheduled in this window."

        run = ", ".join(fixture.label for fixture in self.fixtures)
        parts = [f"Next {self.length}: {run}"]

        difficulty = self.mean_difficulty
        if difficulty is not None:
            # Deliberately three plain bands. The published difficulty is a
            # coarse 1-5 judgement, and dressing it up in decimals would imply
            # a precision the source does not have.
            if difficulty <= _DIFFICULTY_MIDPOINT - 0.5:
                verdict = "a kind run"
            elif difficulty >= _DIFFICULTY_MIDPOINT + 0.5:
                verdict = "a hard run"
            else:
                verdict = "an average run"
            parts.append(f"{verdict} at {difficulty} average difficulty")

        parts.append(f"{self.home_count} of {self.length} at home")

        if self.blanks:
            parts.append(f"blank in GW{', GW'.join(str(gw) for gw in self.blanks)}")
        if self.doubles:
            parts.append(f"double in GW{', GW'.join(str(gw) for gw in self.doubles)}")

        return "; ".join(parts) + "."


@dataclass(frozen=True)
class PlayerContext:
    """Everything behind and ahead of one player."""

    player_code: int
    position: str
    seasons: tuple[SeasonLine, ...]
    run: FixtureRun | None

    @property
    def last_season(self) -> SeasonLine | None:
        return self.seasons[-1] if self.seasons else None

    def sentences(self) -> tuple[str, ...]:
        lines = [season.sentence(self.position) for season in reversed(self.seasons)]
        if self.run is not None:
            lines.append(self.run.sentence())
        return tuple(lines)


def _as_int(frame: pl.DataFrame, column: str) -> pl.Expr:
    return (
        pl.col(column).cast(pl.Int64, strict=False).fill_null(0)
        if column in frame.columns
        else pl.lit(0, dtype=pl.Int64)
    )


def _as_float(frame: pl.DataFrame, column: str) -> pl.Expr:
    return (
        pl.col(column).cast(pl.Float64, strict=False)
        if column in frame.columns
        else pl.lit(None, dtype=pl.Float64)
    )


def build_season_lines(
    player_stats: pl.DataFrame,
    *,
    player_code: int,
    cutoff: datetime | None = None,
    max_seasons: int = 3,
) -> tuple[SeasonLine, ...]:
    """Summarise a player's completed seasons, most recent last.

    Args:
        player_stats: Canonical per-gameweek rows.
        player_code: The stable identity. Never ``element_id``, which FPL
            reissues each season — the whole point of a multi-season summary is
            that it follows one footballer.
        cutoff: Only rows available strictly before this instant are counted, so
            a summary cannot include a match that had not been played.
        max_seasons: How many seasons back to report.

    Returns:
        One line per season with at least one appearance. A season the player
        did not play is omitted rather than reported as zeroes, since zero
        appearances and zero goals are different claims.
    """
    if player_stats.is_empty() or "player_code" not in player_stats.columns:
        return ()

    rows = player_stats.filter(pl.col("player_code") == player_code)
    if cutoff is not None and "available_time" in rows.columns:
        rows = rows.filter(pl.col("available_time") < cutoff)
    if rows.is_empty():
        return ()

    # An appearance is a minute played. Counting selected-but-unused rows would
    # deflate every per-appearance figure by however long a player sat on a
    # bench, which is not what anyone means by "he averaged four points a game".
    played = rows.filter(_as_int(rows, "minutes") > 0)
    if played.is_empty():
        return ()

    grouped = (
        played.group_by("season")
        .agg(
            pl.len().alias("appearances"),
            _as_int(played, "minutes").sum().alias("minutes"),
            _as_int(played, "goals_scored").sum().alias("goals"),
            _as_int(played, "assists").sum().alias("assists"),
            _as_int(played, "clean_sheets").sum().alias("clean_sheets"),
            _as_int(played, "total_points").sum().alias("points"),
            _as_float(played, "expected_goals").sum().alias("expected_goals"),
            _as_float(played, "expected_assists").sum().alias("expected_assists"),
        )
        .sort("season")
        .tail(max_seasons)
    )

    lines: list[SeasonLine] = []
    for row in grouped.iter_rows(named=True):
        expected_goals = row["expected_goals"]
        expected_assists = row["expected_assists"]
        lines.append(
            SeasonLine(
                season=str(row["season"]),
                appearances=int(row["appearances"]),
                minutes=int(row["minutes"]),
                goals=int(row["goals"]),
                assists=int(row["assists"]),
                clean_sheets=int(row["clean_sheets"]),
                points=int(row["points"]),
                expected_goals=None if expected_goals is None else round(float(expected_goals), 2),
                expected_assists=(
                    None if expected_assists is None else round(float(expected_assists), 2)
                ),
            )
        )
    return tuple(lines)


def build_fixture_run(
    fixtures: pl.DataFrame,
    *,
    team_id: int,
    from_gameweek: int,
    team_names: dict[int, str],
    length: int = DEFAULT_RUN_LENGTH,
) -> FixtureRun:
    """Characterise a team's next few scheduled fixtures.

    Reads the schedule and the published difficulty only — never a result — so
    this is knowable in advance and safe to show before a deadline.

    Blanks and doubles are derived from the schedule itself rather than assumed:
    a gameweek in the window with no fixture is a blank, and one with two is a
    double. Both change a decision more than any difficulty average does, so
    they are surfaced separately instead of being averaged away.
    """
    empty = FixtureRun(fixtures=(), blanks=(), doubles=())
    if fixtures.is_empty():
        return empty

    required = {"gameweek_id", "home_team_id", "away_team_id"}
    if not required.issubset(fixtures.columns):
        return empty

    window = list(range(from_gameweek, from_gameweek + length))
    involved = fixtures.filter(
        ((pl.col("home_team_id") == team_id) | (pl.col("away_team_id") == team_id))
        & pl.col("gameweek_id").is_in(window)
    ).sort("gameweek_id")

    scheduled: list[ScheduledFixture] = []
    per_gameweek: dict[int, int] = dict.fromkeys(window, 0)

    for row in involved.iter_rows(named=True):
        gameweek = int(row["gameweek_id"])
        is_home = int(row["home_team_id"]) == team_id
        opponent_id = int(row["away_team_id"]) if is_home else int(row["home_team_id"])
        difficulty_column = "home_difficulty" if is_home else "away_difficulty"
        raw = row.get(difficulty_column)

        per_gameweek[gameweek] = per_gameweek.get(gameweek, 0) + 1
        scheduled.append(
            ScheduledFixture(
                gameweek=gameweek,
                opponent=team_names.get(opponent_id, str(opponent_id)),
                is_home=is_home,
                difficulty=None if raw is None else int(raw) or None,
            )
        )

    return FixtureRun(
        fixtures=tuple(scheduled),
        blanks=tuple(gw for gw in window if per_gameweek.get(gw, 0) == 0),
        doubles=tuple(gw for gw in window if per_gameweek.get(gw, 0) > 1),
    )


def build_player_context(
    *,
    player_code: int,
    position: str,
    team_id: int,
    player_stats: pl.DataFrame,
    fixtures: pl.DataFrame,
    team_names: dict[int, str],
    from_gameweek: int,
    cutoff: datetime | None = None,
    max_seasons: int = 3,
    run_length: int = DEFAULT_RUN_LENGTH,
) -> PlayerContext:
    """Assemble both halves of a player's context in one call."""
    return PlayerContext(
        player_code=player_code,
        position=position,
        seasons=build_season_lines(
            player_stats, player_code=player_code, cutoff=cutoff, max_seasons=max_seasons
        ),
        run=build_fixture_run(
            fixtures,
            team_id=team_id,
            from_gameweek=from_gameweek,
            team_names=team_names,
            length=run_length,
        ),
    )
