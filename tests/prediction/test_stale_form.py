"""The stale-form minutes floor.

A rule rather than a learned relationship, so the tests are about its bounds.
It exists because three feature-based attempts failed, and its danger is the
opposite of the bug it fixes: a floor that fires too readily would invent
minutes for players who do not play.
"""

from __future__ import annotations

from xg_alonso.prediction.inference import _established_minutes_floor


class TestFloor:
    def test_it_multiplies_how_often_by_how_long(self) -> None:
        """A player appearing 80% of the time and lasting 90 implies 72."""
        assert _established_minutes_floor(90.0, 0.8) == 72.0

    def test_a_fringe_player_gets_a_low_floor(self) -> None:
        """The floor must not rescue someone who genuinely does not play."""
        assert (_established_minutes_floor(60.0, 0.2) or 0.0) < 15.0

    def test_no_appearance_record_gives_no_floor(self) -> None:
        """Absent evidence must not become a confident number."""
        assert _established_minutes_floor(None, 0.9) is None
        assert _established_minutes_floor(90.0, None) is None

    def test_a_player_who_never_appears_gets_no_floor(self) -> None:
        assert _established_minutes_floor(90.0, 0.0) is None
        assert _established_minutes_floor(0.0, 0.9) is None
