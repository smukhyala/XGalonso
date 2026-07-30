"""Tests for detecting what changed between two official payloads.

The motivating case is asserted directly: a striker who picks up an injury FPL
has already published must come back as a surfaced event, because the failure
this module exists to prevent is a stale snapshot keeping him in the recommended
eleven. The rest guard the line between *change* and *significance* — a real
poll turned up ten changes of which one mattered, and reporting them as equals
would bury it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from xg_alonso.contracts.events import EventKind, Materiality, availability_direction
from xg_alonso.pipelines.ingestion.changes import diff_bootstrap

_NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


def _player(
    code: int = 1,
    *,
    name: str = "Player",
    status: str = "a",
    chance: int | None = None,
    news: str = "",
    news_added: str | None = None,
    cost: int = 50,
    owned: str = "0.5",
) -> dict[str, Any]:
    return {
        "code": code,
        "id": code,
        "web_name": name,
        "status": status,
        "chance_of_playing_next_round": chance,
        "news": news,
        "news_added": news_added,
        "now_cost": cost,
        "selected_by_percent": owned,
    }


def _payload(*players: dict[str, Any]) -> dict[str, Any]:
    return {"elements": list(players)}


class TestTheMotivatingCase:
    def test_a_published_injury_is_detected(self) -> None:
        """The exact failure: FPL said `i`, we kept saying `a`."""
        before = _payload(_player(560262, name="Kroupi.Jr", owned="13.4"))
        after = _payload(
            _player(
                560262,
                name="Kroupi.Jr",
                status="i",
                chance=0,
                news="Foot injury - Unknown return date",
                news_added="2026-07-29T17:30:09.796802Z",
                owned="9.9",
            )
        )
        diff = diff_bootstrap(before, after, detected_at=_NOW)
        availability = [e for e in diff.events if e.kind is EventKind.AVAILABILITY]

        assert len(availability) == 1
        assert availability[0].before == "a"
        assert availability[0].after == "i"
        assert availability[0].is_bad_news

    def test_it_carries_the_sources_own_timestamp(self) -> None:
        """`news_added` is the only timestamp here describing the world.

        A diff knows something moved; it cannot know when the club said it.
        """
        after = _payload(
            _player(1, status="i", news="Foot injury", news_added="2026-07-29T17:30:09.796802Z")
        )
        diff = diff_bootstrap(_payload(_player(1)), after, detected_at=_NOW)
        reported = diff.events[0].source_reported_at

        assert reported is not None
        assert reported.tzinfo is not None
        assert reported.day == 29

    def test_a_squad_member_is_critical(self) -> None:
        diff = diff_bootstrap(
            _payload(_player(1, owned="0.1")),
            _payload(_player(1, status="i", owned="0.1")),
            detected_at=_NOW,
            squad=[1],
        )
        assert diff.events[0].materiality is Materiality.CRITICAL
        assert "squad" in diff.events[0].reason


class TestMateriality:
    def test_a_widely_owned_player_surfaces(self) -> None:
        diff = diff_bootstrap(
            _payload(_player(1, owned="9.9")),
            _payload(_player(1, status="i", owned="9.9")),
            detected_at=_NOW,
        )
        assert diff.events[0].materiality is Materiality.MATERIAL

    def test_a_barely_owned_player_is_minor(self) -> None:
        """Nine of ten real changes were loans for sub-1%-owned players."""
        diff = diff_bootstrap(
            _payload(_player(1, owned="0.1")),
            _payload(_player(1, status="u", owned="0.1")),
            detected_at=_NOW,
        )
        assert diff.events[0].materiality is Materiality.MINOR
        assert diff.worth_surfacing() == ()

    def test_a_highly_rated_unowned_player_still_surfaces(self) -> None:
        """The thing worth knowing before everybody else does."""
        diff = diff_bootstrap(
            _payload(_player(1, owned="0.2")),
            _payload(_player(1, status="i", owned="0.2")),
            detected_at=_NOW,
            expected_points={1: 6.0},
        )
        assert diff.events[0].materiality is Materiality.MATERIAL

    def test_surfaced_events_put_the_squad_first(self) -> None:
        before = _payload(_player(1, owned="30.0"), _player(2, owned="1.0"))
        after = _payload(_player(1, status="i", owned="30.0"), _player(2, status="i", owned="1.0"))
        diff = diff_bootstrap(before, after, detected_at=_NOW, squad=[2])

        assert diff.worth_surfacing()[0].player_code == 2


class TestWhatCountsAsAChange:
    def test_an_identical_payload_produces_nothing(self) -> None:
        same = _payload(_player(1))
        assert diff_bootstrap(same, same, detected_at=_NOW).events == ()

    def test_a_first_run_reports_nothing(self) -> None:
        """There is no change without a baseline; the alternative is reporting
        the entire league as new."""
        diff = diff_bootstrap(None, _payload(_player(1), _player(2)), detected_at=_NOW)
        assert diff.events == ()
        assert diff.players_compared == 2

    def test_clearing_chance_of_playing_is_not_a_drop(self) -> None:
        """FPL clears the field when a player is fit. Reading that as 100 -> null
        would invent an event every time somebody recovered."""
        before = _payload(_player(1, status="d", chance=75))
        after = _payload(_player(1, status="d", chance=None))
        chance_events = [
            e
            for e in diff_bootstrap(before, after, detected_at=_NOW).events
            if e.kind is EventKind.CHANCE_OF_PLAYING
        ]
        assert chance_events == []

    def test_a_probability_drop_without_a_status_change_is_reported(self) -> None:
        """100 to 75 keeps him pickable and halves the case for captaining him."""
        before = _payload(_player(1, status="d", chance=100, owned="9.0"))
        after = _payload(_player(1, status="d", chance=75, owned="9.0"))
        events = diff_bootstrap(before, after, detected_at=_NOW).events

        assert [e.kind for e in events] == [EventKind.CHANCE_OF_PLAYING]
        assert events[0].materiality is Materiality.MATERIAL

    def test_a_small_probability_drop_is_minor(self) -> None:
        before = _payload(_player(1, status="d", chance=100, owned="9.0"))
        after = _payload(_player(1, status="d", chance=90, owned="9.0"))
        assert diff_bootstrap(before, after, detected_at=_NOW).events[0].materiality is (
            Materiality.MINOR
        )

    def test_a_price_move_is_an_event(self) -> None:
        diff = diff_bootstrap(
            _payload(_player(1, cost=75)), _payload(_player(1, cost=76)), detected_at=_NOW
        )
        assert [e.kind for e in diff.events] == [EventKind.PRICE]

    def test_a_new_player_and_a_departure_are_distinguished(self) -> None:
        diff = diff_bootstrap(_payload(_player(1)), _payload(_player(2)), detected_at=_NOW)
        kinds = {e.kind for e in diff.events}
        assert kinds == {EventKind.JOINED, EventKind.DEPARTED}

    def test_identity_is_the_stable_code_not_the_element_id(self) -> None:
        """FPL reissues element ids each season; a diff keyed on them would
        report an entire league of spurious changes every August."""
        before = _payload({**_player(500, name="Same"), "id": 11})
        after = _payload({**_player(500, name="Same"), "id": 99})
        assert diff_bootstrap(before, after, detected_at=_NOW).events == ()


class TestDirection:
    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [("a", "i", -1), ("i", "a", 1), ("a", "d", -1), ("d", "a", 1), ("a", "a", 0)],
    )
    def test_availability_has_a_direction(self, before: str, after: str, expected: int) -> None:
        assert availability_direction(before, after) == expected

    def test_an_unknown_code_is_not_guessed(self) -> None:
        """FPL has added codes before. Treating an unrecognised one as fine
        would suppress exactly the event worth seeing."""
        assert availability_direction("a", "z") == 0

    def test_doubtful_ranks_above_injured(self) -> None:
        """A doubtful player is still selectable and still scores."""
        assert availability_direction("i", "d") == 1


class TestDiffSummary:
    def test_an_unchanged_hash_short_circuits(self) -> None:
        diff = diff_bootstrap(
            _payload(_player(1)),
            _payload(_player(1)),
            detected_at=_NOW,
            previous_hash="abc",
            current_hash="abc",
        )
        assert diff.unchanged

    def test_a_changed_hash_is_not_unchanged(self) -> None:
        diff = diff_bootstrap(
            _payload(_player(1)),
            _payload(_player(1, status="i")),
            detected_at=_NOW,
            previous_hash="abc",
            current_hash="def",
        )
        assert not diff.unchanged

    def test_a_naive_timestamp_is_refused(self) -> None:
        """A naive datetime compares wrongly against the UTC everything else
        produces, and the comparison that breaks is the availability check."""
        with pytest.raises(ValueError, match="timezone-aware"):
            diff_bootstrap(
                _payload(_player(1)),
                _payload(_player(1, status="i")),
                detected_at=datetime(2026, 7, 30, 12, 0),  # noqa: DTZ001
            )
