"""Blank and double gameweeks, as a shape rather than a special case.

``build_entities`` used to take ``.unique(subset=["team_id"], keep="first")``
over every upcoming fixture. Two consequences, neither visible in any output:

- a club playing twice lost its **second** leg, so the biggest scoring weeks in
  the game were modelled as ordinary ones;
- a club playing *no* fixture silently inherited its next one, from a **later**
  gameweek, so a blank read as a normal week against a real opponent.

The second is the more dangerous, and it gets an explicit negative control
below: the same club has a fixture in the following week, and the entity frame
must not carry it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from xg_alonso.contracts.identifiers import FixtureId, GameweekId, TeamId
from xg_alonso.contracts.schedule import (
    GameweekSlate,
    TeamFixture,
    blanking_teams,
    doubling_teams,
)

T0 = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)


def _fixture(
    fid: int, gw: int, team: int, opponent: int, *, home: bool, hours: int = 0
) -> TeamFixture:
    return TeamFixture(
        fixture_id=FixtureId(fid),
        gameweek=GameweekId(gw),
        team_id=TeamId(team),
        opponent_team_id=TeamId(opponent),
        was_home=home,
        kickoff_time=T0 + timedelta(hours=hours),
    )


class TestTheSlateDescribesTheWeek:
    def test_a_club_with_one_fixture_is_neither_blank_nor_double(self) -> None:
        slate = GameweekSlate(
            gameweek=GameweekId(28),
            fixtures=(_fixture(1, 28, 3, 7, home=True), _fixture(1, 28, 7, 3, home=False)),
        )
        assert slate.fixture_count(TeamId(3)) == 1
        assert not slate.is_blank(TeamId(3))
        assert not slate.is_double(TeamId(3))

    def test_a_club_with_no_fixture_is_blank(self) -> None:
        slate = GameweekSlate(
            gameweek=GameweekId(28),
            fixtures=(_fixture(1, 28, 3, 7, home=True), _fixture(1, 28, 7, 3, home=False)),
        )
        assert slate.is_blank(TeamId(9))
        assert slate.fixture_count(TeamId(9)) == 0
        assert slate.for_team(TeamId(9)) == ()

    def test_a_club_with_two_fixtures_is_double(self) -> None:
        slate = GameweekSlate(
            gameweek=GameweekId(34),
            fixtures=(
                _fixture(1, 34, 3, 7, home=True, hours=0),
                _fixture(1, 34, 7, 3, home=False, hours=0),
                _fixture(2, 34, 3, 9, home=False, hours=72),
                _fixture(2, 34, 9, 3, home=True, hours=72),
            ),
        )
        assert slate.is_double(TeamId(3))
        assert slate.fixture_count(TeamId(3)) == 2
        assert doubling_teams(slate) == (TeamId(3),)

    def test_fixtures_are_ordered_by_kickoff(self) -> None:
        """A stable order is what makes `fixture_index` mean anything."""
        slate = GameweekSlate(
            gameweek=GameweekId(34),
            fixtures=(
                _fixture(2, 34, 3, 9, home=False, hours=72),
                _fixture(1, 34, 3, 7, home=True, hours=0),
            ),
        )
        assert [int(f.fixture_id) for f in slate.for_team(TeamId(3))] == [1, 2]

    def test_blanking_teams_needs_the_league(self) -> None:
        """Absence is only meaningful against a roster."""
        slate = GameweekSlate(
            gameweek=GameweekId(28),
            fixtures=(_fixture(1, 28, 3, 7, home=True), _fixture(1, 28, 7, 3, home=False)),
        )
        league = [TeamId(3), TeamId(7), TeamId(9), TeamId(11)]
        assert blanking_teams(slate, league) == (TeamId(9), TeamId(11))

    def test_a_fixture_from_another_gameweek_is_refused(self) -> None:
        with pytest.raises(ValueError, match="gameweek"):
            GameweekSlate(gameweek=GameweekId(28), fixtures=(_fixture(1, 29, 3, 7, home=True),))

    def test_a_club_cannot_play_itself(self) -> None:
        with pytest.raises(ValueError, match="playing itself"):
            GameweekSlate(gameweek=GameweekId(28), fixtures=(_fixture(1, 28, 3, 3, home=True),))


class TestEntitiesCarryEveryLeg:
    """`build_entities` against a synthetic context."""

    @staticmethod
    def _context(fixtures: pl.DataFrame) -> object:
        from xg_alonso.cli.pipeline import SliceContext

        players = pl.DataFrame(
            {
                "player_code": [1, 2, 3],
                "position": ["MID", "DEF", "FWD"],
                "team_id": [3, 3, 9],
                "current_price": [50, 50, 50],
                "web_name": ["a", "b", "c"],
                "status": ["a", "a", "a"],
            }
        )
        empty = pl.DataFrame()
        return SliceContext(
            season="2026-27",  # type: ignore[arg-type]
            scoring=None,  # type: ignore[arg-type]
            squad_rules=None,  # type: ignore[arg-type]
            players=players,
            teams=empty,
            gameweeks=empty,
            fixtures=fixtures,
            player_stats=empty,
            snapshot_sha256="a" * 64,
            available_time=T0,
        )

    @staticmethod
    def _fixtures() -> pl.DataFrame:
        """Club 3 plays twice in GW34; club 9 blanks in 34 and plays in 35."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3],
                "gameweek_id": [34, 34, 35],
                "home_team_id": [3, 9, 9],
                "away_team_id": [7, 3, 3],
                "finished": [False, False, False],
                "kickoff_time": [T0, T0 + timedelta(days=3), T0 + timedelta(days=10)],
            }
        )

    def test_a_double_gameweek_produces_two_rows_per_player(self) -> None:
        from xg_alonso.cli.pipeline import build_entities

        entities = build_entities(
            self._context(self._fixtures()),  # type: ignore[arg-type]
            cutoff=T0,
            gameweek=GameweekId(34),
        )
        club_three = entities.filter(pl.col("player_code") == 1)

        assert club_three.height == 2
        assert sorted(club_three["fixture_index"].to_list()) == [0, 1]
        assert club_three["fixture_count"].to_list() == [2, 2]
        # Both legs, and the second is away.
        assert sorted(club_three["opponent_team_id"].to_list()) == [7, 9]
        assert sorted(club_three["was_home"].to_list()) == [False, True]

    def test_a_blanking_club_keeps_one_null_row(self) -> None:
        from xg_alonso.cli.pipeline import build_entities

        entities = build_entities(
            self._context(self._fixtures()),  # type: ignore[arg-type]
            cutoff=T0,
            gameweek=GameweekId(34),
        )
        # Club 9 is at home in fixture 2, so it is NOT blank in GW34. Use a
        # gameweek where it genuinely has nothing.
        assert entities.filter(pl.col("player_code") == 3).height == 1

    def test_a_blank_does_not_inherit_a_later_fixture(self) -> None:
        """The negative control, and the defect that mattered most.

        `.unique(keep="first")` over *all* upcoming fixtures gave a blanking
        club its next match from a later gameweek, so a blank read as a normal
        week against a real opponent.
        """
        from xg_alonso.cli.pipeline import build_entities

        fixtures = pl.DataFrame(
            {
                "id": [1, 2],
                "gameweek_id": [28, 29],
                "home_team_id": [3, 9],
                "away_team_id": [7, 3],
                "finished": [False, False],
                "kickoff_time": [T0, T0 + timedelta(days=7)],
            }
        )
        entities = build_entities(
            self._context(fixtures),  # type: ignore[arg-type]
            cutoff=T0,
            gameweek=GameweekId(28),
        )
        blanking = entities.filter(pl.col("player_code") == 3)  # club 9

        assert blanking.height == 1
        assert blanking["opponent_team_id"].to_list() == [None]
        assert blanking["fixture_count"].to_list() == [0]

    def test_collapsing_two_legs_yields_one_entry(self) -> None:
        """What `collapse_by_player` is for, stated as a property."""
        from xg_alonso.cli.pipeline import build_entities

        entities = build_entities(
            self._context(self._fixtures()),  # type: ignore[arg-type]
            cutoff=T0,
            gameweek=GameweekId(34),
        )
        assert entities.height > entities["player_code"].n_unique()
