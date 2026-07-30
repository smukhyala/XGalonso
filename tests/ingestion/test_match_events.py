"""Tests for the match-event source, its robots gate and its normalization.

The gate tests carry a negative control: an origin that publishes Understat's
actual ``robots.txt`` must be refused. Without it, "the gate allows the source
we use" would pass just as well with a gate that allows everything.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from xg_alonso.contracts.identifiers import Season
from xg_alonso.pipelines.ingestion.fpl_client import OfflineError
from xg_alonso.pipelines.ingestion.match_events import (
    FOOTBALL_DATA_BASE_URL,
    REQUIRED_COLUMNS,
    SOURCE_MATCH_EVENTS,
    fetch_match_events_season,
    season_code,
)
from xg_alonso.pipelines.ingestion.robots import (
    Pacer,
    RobotsDisallowedError,
    RobotsGate,
    RobotsUnavailableError,
)
from xg_alonso.pipelines.normalization.match_events import (
    PUBLICATION_LAG,
    TeamMappingError,
    map_team_names,
    normalize_match_events,
)

_UNDERSTAT_ROBOTS = "User-agent: *\nDisallow: /\n"
_PERMISSIVE_ROBOTS = "User-agent: *\nDisallow:\n"

_HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR"


def _csv(*rows: str) -> bytes:
    return ("\n".join([_HEADER, *rows]) + "\n").encode()


#: One August fixture (British Summer Time) and one December fixture (GMT), so a
#: timezone bug cannot hide behind a single sample.
_SUMMER_ROW = "E0,16/08/2024,20:00,Man United,Fulham,1,0,14,10,5,2,12,10,7,8,2,3,0,0"
_WINTER_ROW = "E0,26/12/2024,15:00,Tottenham,Nott'm Forest,0,1,18,9,4,3,8,11,9,2,1,2,0,1"


def _teams() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team_code": [1, 54, 6, 17, 40, 49],
            "name": [
                "Man Utd",
                "Fulham",
                "Spurs",
                "Nott'm Forest",
                "Ipswich Town",
                "Sheffield Utd",
            ],
        }
    )


class TestSeasonCode:
    def test_converts_a_season_to_the_source_path(self) -> None:
        assert season_code("2022-23") == "2223"
        assert season_code("2025-26") == "2526"

    def test_handles_the_century_rollover(self) -> None:
        assert season_code("1999-00") == "9900"

    @pytest.mark.parametrize("bad", ["2022", "2022-2023", "22-23", "abcd-ef"])
    def test_rejects_malformed_seasons(self, bad: str) -> None:
        with pytest.raises(ValueError, match="season"):
            season_code(bad)

    def test_rejects_non_consecutive_years(self) -> None:
        """The source path is positional, so this would fetch a real wrong season."""
        with pytest.raises(ValueError, match="consecutive"):
            season_code("2022-25")


class TestRobotsGate:
    @respx.mock
    def test_refuses_an_origin_that_disallows_everything(self) -> None:
        """The negative control: Understat's actual robots.txt must be refused."""
        respx.get("https://understat.com/robots.txt").mock(
            return_value=httpx.Response(200, text=_UNDERSTAT_ROBOTS)
        )
        with httpx.Client() as client:
            gate = RobotsGate(client=client)
            assert gate.allows("https://understat.com/league/EPL/2024") is False
            with pytest.raises(RobotsDisallowedError, match="machine-readable refusal"):
                gate.require("https://understat.com/league/EPL/2024")

    @respx.mock
    def test_permits_an_origin_that_disallows_nothing(self) -> None:
        respx.get("https://www.football-data.co.uk/robots.txt").mock(
            return_value=httpx.Response(200, text=_PERMISSIVE_ROBOTS)
        )
        with httpx.Client() as client:
            gate = RobotsGate(client=client)
            assert gate.allows(f"{FOOTBALL_DATA_BASE_URL}/2425/E0.csv") is True
            gate.require(f"{FOOTBALL_DATA_BASE_URL}/2425/E0.csv")

    @respx.mock
    def test_treats_a_missing_robots_file_as_unrestricted(self) -> None:
        """RFC 9309: 4xx means no rules exist, not that nothing is allowed."""
        respx.get("https://example.test/robots.txt").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            assert RobotsGate(client=client).allows("https://example.test/x.csv") is True

    @respx.mock
    def test_treats_a_server_error_as_a_complete_disallow(self) -> None:
        respx.get("https://example.test/robots.txt").mock(return_value=httpx.Response(503))
        with (
            httpx.Client() as client,
            pytest.raises(RobotsUnavailableError, match="complete disallow"),
        ):
            RobotsGate(client=client).require("https://example.test/x.csv")

    @respx.mock
    def test_treats_an_unreachable_origin_as_a_complete_disallow(self) -> None:
        respx.get("https://example.test/robots.txt").mock(
            side_effect=httpx.ConnectError("no route")
        )
        with (
            httpx.Client() as client,
            pytest.raises(RobotsUnavailableError, match="complete disallow"),
        ):
            RobotsGate(client=client).require("https://example.test/x.csv")

    @respx.mock
    def test_fetches_robots_once_per_origin(self) -> None:
        route = respx.get("https://example.test/robots.txt").mock(
            return_value=httpx.Response(200, text=_PERMISSIVE_ROBOTS)
        )
        with httpx.Client() as client:
            gate = RobotsGate(client=client)
            for index in range(5):
                gate.require(f"https://example.test/{index}.csv")
        assert route.call_count == 1

    @respx.mock
    def test_honours_a_declared_crawl_delay(self) -> None:
        respx.get("https://example.test/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 7\nDisallow:\n")
        )
        with httpx.Client() as client:
            assert RobotsGate(client=client).crawl_delay("https://example.test/x") == 7.0

    @respx.mock
    def test_never_goes_faster_than_the_default_floor(self) -> None:
        """A source asking us to hurry does not make hurrying a good idea."""
        respx.get("https://example.test/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 0\nDisallow:\n")
        )
        with httpx.Client() as client:
            assert RobotsGate(client=client).crawl_delay("https://example.test/x") >= 1.0


class TestFetch:
    """Fetches here are respx-mocked, so `XG_ALONSO_OFFLINE` is cleared per test.

    The guard exists so a test reaching for the *real* network fails loudly
    instead of flaking. respx intercepts every request in these, and an
    unmocked one raises — so the guarantee the guard provides is already held by
    stricter means. `test_honours_offline_mode` sets it back on purpose.
    """

    @respx.mock
    def test_refuses_before_downloading_when_the_origin_says_no(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from xg_alonso.storage.bronze import FileSystemBronzeStore

        monkeypatch.delenv("XG_ALONSO_OFFLINE", raising=False)

        robots = respx.get("https://www.football-data.co.uk/robots.txt").mock(
            return_value=httpx.Response(200, text=_UNDERSTAT_ROBOTS)
        )
        payload = respx.get(f"{FOOTBALL_DATA_BASE_URL}/2425/E0.csv").mock(
            return_value=httpx.Response(200, content=_csv(_SUMMER_ROW))
        )

        with pytest.raises(RobotsDisallowedError):
            fetch_match_events_season(
                "2024-25",
                bronze=FileSystemBronzeStore(tmp_path),
                run_id="test",
                pacer=Pacer(0.0),
            )

        assert robots.call_count == 1
        assert payload.call_count == 0, "a refused source must cost no download"

    @respx.mock
    def test_writes_a_bronze_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from xg_alonso.storage.bronze import FileSystemBronzeStore

        monkeypatch.delenv("XG_ALONSO_OFFLINE", raising=False)

        respx.get("https://www.football-data.co.uk/robots.txt").mock(
            return_value=httpx.Response(200, text=_PERMISSIVE_ROBOTS)
        )
        body = _csv(_SUMMER_ROW, _WINTER_ROW)
        respx.get(f"{FOOTBALL_DATA_BASE_URL}/2425/E0.csv").mock(
            return_value=httpx.Response(200, content=body)
        )

        bronze = FileSystemBronzeStore(tmp_path)
        result = fetch_match_events_season(
            "2024-25", bronze=bronze, run_id="test", pacer=Pacer(0.0)
        )

        assert result.season == Season("2024-25")
        assert result.division == "E0"
        assert bronze.read(result.ref) == body
        assert bronze.latest(f"{SOURCE_MATCH_EVENTS}.E0.2024-25") is not None

    def test_rejects_an_unknown_division_before_touching_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from xg_alonso.storage.bronze import FileSystemBronzeStore

        monkeypatch.delenv("XG_ALONSO_OFFLINE", raising=False)

        with pytest.raises(KeyError):
            fetch_match_events_season(
                "2024-25",
                bronze=FileSystemBronzeStore(tmp_path),
                run_id="test",
                division="ZZ",
            )

    def test_honours_offline_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from xg_alonso.storage.bronze import FileSystemBronzeStore

        monkeypatch.setenv("XG_ALONSO_OFFLINE", "1")
        with pytest.raises(OfflineError):
            fetch_match_events_season(
                "2024-25", bronze=FileSystemBronzeStore(tmp_path), run_id="test"
            )


class TestTeamMapping:
    def test_matches_identical_names(self) -> None:
        mapping = map_team_names(["Fulham"], _teams())
        assert mapping.resolved == {"Fulham": 54}
        assert mapping.complete

    def test_matches_through_the_alias_table(self) -> None:
        mapping = map_team_names(["Man United", "Tottenham", "Sheffield United"], _teams())
        assert mapping.resolved == {
            "Man United": 1,
            "Tottenham": 6,
            "Sheffield United": 49,
        }

    def test_matches_a_unique_prefix(self) -> None:
        """'Ipswich' reaches 'Ipswich Town' without needing an alias entry."""
        assert map_team_names(["Ipswich"], _teams()).resolved == {"Ipswich": 40}

    def test_ignores_punctuation_differences(self) -> None:
        assert map_team_names(["Nottm Forest"], _teams()).resolved == {"Nottm Forest": 17}

    def test_refuses_to_guess_between_two_candidates(self) -> None:
        """A prefix matching two clubs must not silently pick one."""
        ambiguous = pl.DataFrame({"team_code": [7, 8], "name": ["Sheffield Utd", "Sheffield Wed"]})
        mapping = map_team_names(["Sheffield"], ambiguous)
        assert mapping.resolved == {}
        assert mapping.unmatched == ("Sheffield",)

    def test_reports_a_name_it_cannot_place(self) -> None:
        mapping = map_team_names(["Real Madrid"], _teams())
        assert mapping.unmatched == ("Real Madrid",)
        assert not mapping.complete


class TestNormalization:
    def _report(self, *rows: str, strict: bool = True):  # type: ignore[no-untyped-def]
        return normalize_match_events(
            _csv(*rows),
            season="2024-25",
            division="E0",
            teams=_teams(),
            required_columns=REQUIRED_COLUMNS,
            strict=strict,
        )

    def test_produces_two_rows_per_match(self) -> None:
        report = self._report(_SUMMER_ROW, _WINTER_ROW)
        assert report.matches == 2
        assert report.frame.height == 4
        assert report.complete

    def test_mirrors_the_two_perspectives_of_one_match(self) -> None:
        """The away row's `shots` must be the home row's `shots_against`."""
        frame = self._report(_SUMMER_ROW).frame
        home = frame.filter(pl.col("was_home")).row(0, named=True)
        away = frame.filter(~pl.col("was_home")).row(0, named=True)

        assert (home["shots"], home["shots_against"]) == (14, 10)
        assert (away["shots"], away["shots_against"]) == (10, 14)
        assert home["shots"] == away["shots_against"]
        assert home["goals_for"] == 1
        assert away["goals_against"] == 1
        assert home["fouls_committed"] == 12
        assert away["fouls_suffered"] == 12
        assert home["team_code"] == 1
        assert away["opponent_team_code"] == 1

    def test_converts_british_summer_time_to_utc(self) -> None:
        """An August 20:00 kickoff is 19:00 UTC, not 20:00."""
        frame = self._report(_SUMMER_ROW).frame
        assert frame["kickoff_time"][0] == datetime(2024, 8, 16, 19, 0, tzinfo=UTC)

    def test_leaves_winter_kickoffs_on_utc(self) -> None:
        frame = self._report(_WINTER_ROW).frame
        assert frame["kickoff_time"][0] == datetime(2024, 12, 26, 15, 0, tzinfo=UTC)

    def test_availability_trails_kickoff(self) -> None:
        """These are post-match statistics; availability at kickoff would leak."""
        frame = self._report(_SUMMER_ROW, _WINTER_ROW).frame
        gaps = (frame["available_time"] - frame["kickoff_time"]).unique().to_list()
        assert gaps == [PUBLICATION_LAG]
        assert all(
            available > kickoff
            for available, kickoff in zip(
                frame["available_time"], frame["kickoff_time"], strict=True
            )
        )

    def test_availability_lag_is_at_least_a_full_match(self) -> None:
        """A lag shorter than 2h would make a match readable before full time."""
        assert timedelta(hours=2) <= PUBLICATION_LAG

    def test_raises_on_an_unresolvable_team_when_strict(self) -> None:
        row = "E0,16/08/2024,20:00,Real Madrid,Fulham,1,0,14,10,5,2,12,10,7,8,2,3,0,0"
        with pytest.raises(TeamMappingError, match="Real Madrid"):
            self._report(row)

    def test_reports_rather_than_hides_a_drop_when_not_strict(self) -> None:
        row = "E0,16/08/2024,20:00,Real Madrid,Fulham,1,0,14,10,5,2,12,10,7,8,2,3,0,0"
        report = self._report(row, _SUMMER_ROW, strict=False)
        assert report.unmatched_teams == ("Real Madrid",)
        assert report.skipped_rows == 1
        assert not report.complete
        assert report.frame.height == 2

    def test_refuses_a_file_missing_a_required_column(self) -> None:
        """A null column downstream reads as a quiet team, not as absent data."""
        header = _HEADER.replace(",HST", "")
        body = header + "\nE0,16/08/2024,20:00,Man United,Fulham,1,0,14,10,2,12,10,7,8,2,3,0,0\n"
        with pytest.raises(ValueError, match="missing required columns"):
            normalize_match_events(
                body.encode(),
                season="2024-25",
                division="E0",
                teams=_teams(),
                required_columns=REQUIRED_COLUMNS,
            )

    def test_returns_an_empty_frame_with_the_right_shape(self) -> None:
        from xg_alonso.pipelines.normalization.schema import TEAM_MATCH_EVENTS_SCHEMA

        report = self._report()
        assert report.matches == 0
        assert report.frame.columns == list(TEAM_MATCH_EVENTS_SCHEMA)

    def test_conforms_to_the_silver_schema(self) -> None:
        from xg_alonso.pipelines.normalization.schema import TEAM_MATCH_EVENTS_SCHEMA

        frame = self._report(_SUMMER_ROW, _WINTER_ROW).frame
        assert dict(frame.schema) == TEAM_MATCH_EVENTS_SCHEMA
