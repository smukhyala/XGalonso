"""Tests for prior-season lines and upcoming fixture runs.

The load-bearing assertions are the ones about what is *withheld*: a per-90 rate
from a cameo and a difficulty average over unrated preseason fixtures are both
numbers that would look authoritative and mean nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from xg_alonso.explanations.context import (
    MIN_MINUTES_FOR_RATE,
    build_fixture_run,
    build_player_context,
    build_season_lines,
)

_UTC = pl.Datetime(time_unit="us", time_zone="UTC")


def _stats(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "player_code": pl.Int64(),
            "season": pl.Utf8(),
            "minutes": pl.Int64(),
            "goals_scored": pl.Int64(),
            "assists": pl.Int64(),
            "clean_sheets": pl.Int64(),
            "total_points": pl.Int64(),
            "expected_goals": pl.Float64(),
            "expected_assists": pl.Float64(),
            "available_time": _UTC,
        },
    )


def _match(
    *,
    code: int = 1,
    season: str = "2025-26",
    minutes: int = 90,
    goals: int = 0,
    assists: int = 0,
    clean_sheets: int = 0,
    points: int = 2,
    when: datetime | None = None,
) -> dict[str, object]:
    return {
        "player_code": code,
        "season": season,
        "minutes": minutes,
        "goals_scored": goals,
        "assists": assists,
        "clean_sheets": clean_sheets,
        "total_points": points,
        "expected_goals": 0.3,
        "expected_assists": 0.1,
        "available_time": when or datetime(2026, 1, 1, tzinfo=UTC),
    }


def _fixtures(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "gameweek_id": pl.Int64(),
            "home_team_id": pl.Int64(),
            "away_team_id": pl.Int64(),
            "home_difficulty": pl.Int64(),
            "away_difficulty": pl.Int64(),
        },
    )


class TestSeasonLines:
    def test_counts_only_matches_the_player_appeared_in(self) -> None:
        """An unused substitute is not an appearance worth four points of nothing."""
        frame = _stats(
            [
                _match(minutes=90, goals=1, points=8),
                _match(minutes=0, points=0),
                _match(minutes=45, points=2),
            ]
        )
        line = build_season_lines(frame, player_code=1)[0]
        assert line.appearances == 2
        assert line.minutes == 135
        assert line.points == 10

    def test_omits_a_season_the_player_did_not_play(self) -> None:
        """Zero appearances and zero goals are different claims."""
        frame = _stats(
            [
                _match(season="2024-25", minutes=0, points=0),
                _match(season="2025-26", minutes=900, goals=5, points=50),
            ]
        )
        seasons = [line.season for line in build_season_lines(frame, player_code=1)]
        assert seasons == ["2025-26"]

    def test_orders_seasons_oldest_first(self) -> None:
        frame = _stats(
            [
                _match(season="2023-24", minutes=900),
                _match(season="2025-26", minutes=900),
                _match(season="2024-25", minutes=900),
            ]
        )
        seasons = [line.season for line in build_season_lines(frame, player_code=1)]
        assert seasons == ["2023-24", "2024-25", "2025-26"]

    def test_withholds_a_rate_below_the_minutes_floor(self) -> None:
        """One goal in a cameo is a 1.00 per 90 that means nothing."""
        frame = _stats([_match(minutes=90, goals=1, points=8)])
        line = build_season_lines(frame, player_code=1)[0]
        assert line.minutes < MIN_MINUTES_FOR_RATE
        assert line.per_90 is None
        assert "per 90" not in line.sentence("FWD")

    def test_reports_a_rate_above_the_floor(self) -> None:
        frame = _stats([_match(minutes=900, goals=10, points=60)])
        line = build_season_lines(frame, player_code=1)[0]
        assert line.per_90 == 1.0
        assert "per 90" in line.sentence("FWD")

    def test_respects_the_point_in_time_cutoff(self) -> None:
        """A summary must not count a match that had not been played."""
        frame = _stats(
            [
                _match(minutes=90, goals=1, points=8, when=datetime(2026, 1, 1, tzinfo=UTC)),
                _match(minutes=90, goals=5, points=30, when=datetime(2026, 6, 1, tzinfo=UTC)),
            ]
        )
        line = build_season_lines(frame, player_code=1, cutoff=datetime(2026, 3, 1, tzinfo=UTC))[0]
        assert line.goals == 1, "the later match must not be counted"

    def test_ignores_other_players(self) -> None:
        frame = _stats([_match(code=1, goals=3, minutes=900), _match(code=2, goals=9, minutes=900)])
        assert build_season_lines(frame, player_code=1)[0].goals == 3

    def test_an_empty_frame_yields_nothing(self) -> None:
        assert build_season_lines(_stats([]), player_code=1) == ()

    def test_a_defender_leads_with_clean_sheets_not_goals(self) -> None:
        """The whole point: a centre-back's season is not a striker's with fewer goals."""
        frame = _stats([_match(minutes=3000, goals=2, clean_sheets=14, points=150)])
        line = build_season_lines(frame, player_code=1)[0]

        defender = line.sentence("DEF")
        forward = line.sentence("FWD")
        assert defender.index("clean sheet") < defender.index("attacking return")
        assert "clean sheet" not in forward
        assert defender != forward

    def test_a_keeper_is_not_described_by_his_attacking_returns(self) -> None:
        frame = _stats([_match(minutes=3000, clean_sheets=15, points=160)])
        assert "attacking return" not in build_season_lines(frame, player_code=1)[0].sentence("GKP")

    def test_singular_and_plural_read_correctly(self) -> None:
        frame = _stats([_match(minutes=100, goals=1, assists=1, points=10)])
        sentence = build_season_lines(frame, player_code=1)[0].sentence("MID")
        assert "1 goal and 1 assist" in sentence
        assert "1 goals" not in sentence


class TestFixtureRun:
    def _names(self) -> dict[int, str]:
        return {1: "ARS", 2: "BUR", 3: "LEE", 4: "MCI"}

    def test_reads_both_home_and_away_fixtures(self) -> None:
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": 1,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_difficulty": 2,
                    "away_difficulty": 4,
                },
                {
                    "gameweek_id": 2,
                    "home_team_id": 3,
                    "away_team_id": 1,
                    "home_difficulty": 3,
                    "away_difficulty": 3,
                },
            ]
        )
        run = build_fixture_run(fixtures, team_id=1, from_gameweek=1, team_names=self._names())

        assert [f.label for f in run.fixtures] == ["BUR (H)", "LEE (A)"]
        assert run.home_count == 1

    def test_takes_the_difficulty_from_the_right_side(self) -> None:
        """Arsenal at home reads `home_difficulty`, away reads `away_difficulty`."""
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": 1,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_difficulty": 2,
                    "away_difficulty": 5,
                }
            ]
        )
        run = build_fixture_run(fixtures, team_id=1, from_gameweek=1, team_names=self._names())
        assert run.fixtures[0].difficulty == 2

        away = build_fixture_run(fixtures, team_id=2, from_gameweek=1, team_names=self._names())
        assert away.fixtures[0].difficulty == 5

    def test_ignores_fixtures_outside_the_window(self) -> None:
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": gw,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_difficulty": 3,
                    "away_difficulty": 3,
                }
                for gw in (1, 2, 9)
            ]
        )
        run = build_fixture_run(
            fixtures, team_id=1, from_gameweek=1, team_names=self._names(), length=3
        )
        assert [f.gameweek for f in run.fixtures] == [1, 2]

    def test_finds_a_blank_gameweek(self) -> None:
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": 1,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_difficulty": 3,
                    "away_difficulty": 3,
                }
            ]
        )
        run = build_fixture_run(
            fixtures, team_id=1, from_gameweek=1, team_names=self._names(), length=2
        )
        assert run.blanks == (2,)
        assert "blank in GW2" in run.sentence()

    def test_finds_a_double_gameweek(self) -> None:
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": 1,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_difficulty": 3,
                    "away_difficulty": 3,
                },
                {
                    "gameweek_id": 1,
                    "home_team_id": 3,
                    "away_team_id": 1,
                    "home_difficulty": 3,
                    "away_difficulty": 3,
                },
            ]
        )
        run = build_fixture_run(
            fixtures, team_id=1, from_gameweek=1, team_names=self._names(), length=1
        )
        assert run.doubles == (1,)
        assert "double in GW1" in run.sentence()

    def test_unrated_preseason_fixtures_yield_no_average(self) -> None:
        """A run of zeroes must not read as the easiest schedule in the league."""
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": 1,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_difficulty": 0,
                    "away_difficulty": 0,
                }
            ]
        )
        run = build_fixture_run(fixtures, team_id=1, from_gameweek=1, team_names=self._names())
        assert run.mean_difficulty is None
        assert "difficulty" not in run.sentence()

    def test_describes_a_hard_run_as_hard(self) -> None:
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": gw,
                    "home_team_id": 1,
                    "away_team_id": 4,
                    "home_difficulty": 5,
                    "away_difficulty": 2,
                }
                for gw in (1, 2)
            ]
        )
        run = build_fixture_run(
            fixtures, team_id=1, from_gameweek=1, team_names=self._names(), length=2
        )
        assert run.mean_difficulty == 5.0
        assert "a hard run" in run.sentence()

    def test_describes_a_kind_run_as_kind(self) -> None:
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": gw,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_difficulty": 2,
                    "away_difficulty": 4,
                }
                for gw in (1, 2)
            ]
        )
        run = build_fixture_run(
            fixtures, team_id=1, from_gameweek=1, team_names=self._names(), length=2
        )
        assert "a kind run" in run.sentence()

    def test_an_empty_schedule_is_not_an_error(self) -> None:
        run = build_fixture_run(_fixtures([]), team_id=1, from_gameweek=1, team_names={})
        assert run.fixtures == ()
        assert "No fixtures" in run.sentence()


class TestPlayerContext:
    def test_assembles_both_halves(self) -> None:
        stats = _stats([_match(minutes=900, goals=5, points=50)])
        fixtures = _fixtures(
            [
                {
                    "gameweek_id": 1,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_difficulty": 3,
                    "away_difficulty": 3,
                }
            ]
        )
        context = build_player_context(
            player_code=1,
            position="FWD",
            team_id=1,
            player_stats=stats,
            fixtures=fixtures,
            team_names={1: "ARS", 2: "BUR"},
            from_gameweek=1,
        )
        assert context.last_season is not None
        assert context.last_season.goals == 5
        assert context.run is not None
        assert len(context.sentences()) == 2
