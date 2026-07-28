"""Situational-history tests.

These notes are the one part of the explanation layer a user can check against a
scoreboard, so the properties that matter are that they cite real matches, never
cite one that had not been played, and decline to call two appearances a pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from xg_alonso.explanations.history import build_history_notes

T0 = datetime(2023, 8, 1, tzinfo=UTC)
LATER = datetime(2026, 8, 1, tzinfo=UTC)
TEAMS = {7: "Chelsea", 9: "Everton"}


def _stats(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _row(
    *,
    season: str,
    gameweek: int,
    opponent: int,
    home: bool,
    goals: int = 0,
    assists: int = 0,
    minutes: int = 90,
    points: int = 2,
    code: int = 1,
    offset_days: int = 0,
) -> dict[str, object]:
    return {
        "player_code": code,
        "season": season,
        "gameweek_id": gameweek,
        "opponent_team_id": opponent,
        "was_home": home,
        "minutes": minutes,
        "goals_scored": goals,
        "assists": assists,
        "total_points": points,
        "available_time": T0 + timedelta(days=offset_days),
    }


class TestOpponentRecord:
    def test_it_reports_what_he_did_against_this_club(self) -> None:
        stats = _stats(
            [
                _row(season="2024-25", gameweek=5, opponent=7, home=False, goals=2, points=13),
                _row(
                    season="2023-24",
                    gameweek=9,
                    opponent=7,
                    home=False,
                    goals=1,
                    assists=1,
                    points=12,
                ),
            ]
        )
        notes = build_history_notes(
            stats,
            fixtures={1: (7, False)},
            gameweek=1,
            team_names=TEAMS,
            cutoff=LATER,
        )
        text = next(n.text for n in notes[1] if n.kind == "opponent")
        assert "Chelsea" in text
        assert "3 goals" in text
        assert "2024-25" in text

    def test_a_barren_record_is_stated_plainly(self) -> None:
        """The note must be as willing to argue against a player as for him."""
        stats = _stats(
            [_row(season=f"202{i}-2{i + 1}", gameweek=4, opponent=7, home=True) for i in range(3)]
        )
        notes = build_history_notes(
            stats, fixtures={1: (7, True)}, gameweek=1, team_names=TEAMS, cutoff=LATER
        )
        note = next(n for n in notes[1] if n.kind == "opponent")
        assert "no goals or assists" in note.text
        assert not note.is_positive


class TestGameweekRecord:
    def test_a_strong_opening_weekend_is_reported(self) -> None:
        """The motivating example: some players reliably turn up in gameweek 1."""
        stats = _stats(
            [
                _row(
                    season=f"202{i}-2{i + 1}",
                    gameweek=1,
                    opponent=9,
                    home=False,
                    goals=2,
                    points=13,
                )
                for i in range(4)
            ]
        )
        notes = build_history_notes(
            stats, fixtures={1: (7, False)}, gameweek=1, team_names=TEAMS, cutoff=LATER
        )
        note = next(n for n in notes[1] if n.kind == "gameweek")
        assert "opening weekend" in note.text
        assert "4 of his last 4" in note.text
        assert note.is_positive

    def test_a_record_with_no_returns_is_never_marked_positive(self) -> None:
        """The marker must agree with the sentence beside it.

        Scoring on points alone marked "returned in 0 of his last 4 seasons"
        as encouraging, because the appearance points cleared the threshold.
        """
        stats = _stats(
            [
                _row(season=f"202{i}-2{i + 1}", gameweek=1, opponent=9, home=True, points=3)
                for i in range(4)
            ]
        )
        notes = build_history_notes(
            stats, fixtures={1: (7, True)}, gameweek=1, team_names=TEAMS, cutoff=LATER
        )
        note = next(n for n in notes[1] if n.kind == "gameweek")
        assert "returned in 0 of his last 4" in note.text
        assert not note.is_positive

    def test_a_single_season_is_not_a_pattern(self) -> None:
        stats = _stats([_row(season="2024-25", gameweek=1, opponent=9, home=True, goals=3)])
        notes = build_history_notes(
            stats, fixtures={1: (7, True)}, gameweek=1, team_names=TEAMS, cutoff=LATER
        )
        assert all(n.kind != "gameweek" for n in notes.get(1, []))


class TestVenue:
    def test_a_split_needs_evidence_on_both_sides(self) -> None:
        """Three home matches and one away is not a home/away tendency."""
        stats = _stats(
            [_row(season="2024-25", gameweek=g, opponent=9, home=True, points=9) for g in (1, 2, 3)]
            + [_row(season="2024-25", gameweek=4, opponent=9, home=False, points=1)]
        )
        notes = build_history_notes(
            stats, fixtures={1: (9, True)}, gameweek=1, team_names=TEAMS, cutoff=LATER
        )
        assert all(n.kind != "venue" for n in notes.get(1, []))

    def test_a_marginal_split_is_not_reported(self) -> None:
        """Below a point a match the difference is noise dressed as a tendency."""
        stats = _stats(
            [_row(season="2024-25", gameweek=g, opponent=9, home=True, points=4) for g in (1, 2, 3)]
            + [
                _row(season="2024-25", gameweek=g, opponent=9, home=False, points=4)
                for g in (4, 5, 6)
            ]
        )
        notes = build_history_notes(
            stats, fixtures={1: (9, True)}, gameweek=1, team_names=TEAMS, cutoff=LATER
        )
        assert all(n.kind != "venue" for n in notes.get(1, []))


class TestPointInTime:
    def test_a_match_after_the_cutoff_is_never_cited(self) -> None:
        """A note must not reference a result the decision could not have known."""
        stats = _stats(
            [
                _row(season="2024-25", gameweek=5, opponent=7, home=True, goals=3, offset_days=0),
                _row(
                    season="2025-26", gameweek=5, opponent=7, home=True, goals=3, offset_days=2000
                ),
            ]
        )
        notes = build_history_notes(
            stats,
            fixtures={1: (7, True)},
            gameweek=1,
            team_names=TEAMS,
            cutoff=T0 + timedelta(days=10),
        )
        text = next(n.text for n in notes[1] if n.kind == "opponent")
        assert "2025-26" not in text
        assert "3 goals" in text


class TestGuards:
    def test_a_player_with_no_history_gets_no_notes(self) -> None:
        stats = _stats([_row(season="2024-25", gameweek=1, opponent=9, home=True, code=2)])
        notes = build_history_notes(
            stats, fixtures={1: (7, True)}, gameweek=1, team_names=TEAMS, cutoff=LATER
        )
        assert 1 not in notes

    def test_missing_columns_fail_loudly(self) -> None:
        stats = _stats([_row(season="2024-25", gameweek=1, opponent=7, home=True)]).drop(
            "opponent_team_id"
        )
        with pytest.raises(KeyError, match="missing columns"):
            build_history_notes(
                stats, fixtures={1: (7, True)}, gameweek=1, team_names=TEAMS, cutoff=LATER
            )

    def test_notes_are_ordered_by_how_notable_they_are(self) -> None:
        stats = _stats(
            [_row(season="2024-25", gameweek=5, opponent=7, home=True, goals=3, points=17)]
            + [
                _row(season=f"202{i}-2{i + 1}", gameweek=1, opponent=9, home=True, points=2)
                for i in range(3)
            ]
        )
        notes = build_history_notes(
            stats, fixtures={1: (7, True)}, gameweek=1, team_names=TEAMS, cutoff=LATER
        )[1]
        strengths = [abs(n.strength) for n in notes]
        assert strengths == sorted(strengths, reverse=True)
