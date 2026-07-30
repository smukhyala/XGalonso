"""Tests for reading squad requirements out of a manager's own words.

The distinction under test throughout is **start versus own**. "Keep Haaland" is
satisfied by Haaland on the bench and "I want Haaland starting" is not, so a
parser that collapses them silently over-constrains every squad built from the
weaker request. Most of these exist to hold that line.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from xg_alonso.contracts.objective import RequirementKind
from xg_alonso.domain.squad_requests import parse_squad_requirements

_PLAYERS = {
    "Haaland": 1,
    "Saka": 2,
    "Saliba": 3,
    "Salah": 4,
    "Palmer": 5,
}

_TEAMS = {"Arsenal": 1, "Man City": 15, "Liverpool": 14, "Chelsea": 8}


def _kinds(text: str, **kwargs: object) -> dict[str, RequirementKind]:
    """Map each named player to the requirement kind the parser gave him."""
    parse = parse_squad_requirements(text, players=_PLAYERS, teams=_TEAMS, **kwargs)  # type: ignore[arg-type]
    by_player: dict[str, RequirementKind] = {}
    codes = {code: name for name, code in _PLAYERS.items()}
    for requirement in parse.requirements:
        for code in requirement.players:
            by_player[codes[int(code)]] = requirement.kind
    return by_player


def _of_kind(text: str, kind: RequirementKind) -> list[str]:
    parse = parse_squad_requirements(text, players=_PLAYERS, teams=_TEAMS)
    return [r.label for r in parse.requirements if r.kind is kind]


class TestStartVersusOwn:
    def test_starting_binds_the_eleven(self) -> None:
        assert _kinds("I want Haaland starting")["Haaland"] is RequirementKind.MUST_START

    def test_keeping_binds_only_the_squad(self) -> None:
        assert _kinds("keep Haaland")["Haaland"] is RequirementKind.MUST_INCLUDE

    def test_starting_wins_when_both_are_said(self) -> None:
        """ "Keep Haaland starting" is a statement about the eleven."""
        assert _kinds("keep Haaland starting")["Haaland"] is RequirementKind.MUST_START

    @pytest.mark.parametrize(
        "text",
        [
            "start Haaland",
            "Haaland must play",
            "Haaland in the starting xi",
            "I need Haaland starting",
        ],
    )
    def test_start_phrasings(self, text: str) -> None:
        assert _kinds(text)["Haaland"] is RequirementKind.MUST_START

    @pytest.mark.parametrize(
        "text",
        ["keep Haaland", "I want Haaland", "sign Haaland", "Haaland in the squad"],
    )
    def test_own_phrasings(self, text: str) -> None:
        assert _kinds(text)["Haaland"] is RequirementKind.MUST_INCLUDE


class TestGoverningVerb:
    def test_the_nearest_verb_governs(self) -> None:
        """ "avoid Salah and keep Haaland" must not exclude Haaland."""
        kinds = _kinds("avoid Salah and keep Haaland")
        assert kinds["Salah"] is RequirementKind.MUST_EXCLUDE
        assert kinds["Haaland"] is RequirementKind.MUST_INCLUDE

    def test_and_coordinates_rather_than_separates(self) -> None:
        """One verb governs both names, so both must be picked up."""
        kinds = _kinds("start Saka and Haaland")
        assert kinds["Saka"] is RequirementKind.MUST_START
        assert kinds["Haaland"] is RequirementKind.MUST_START

    def test_a_comma_stops_a_verb_leaking(self) -> None:
        kinds = _kinds("keep Saka, avoid Salah")
        assert kinds["Saka"] is RequirementKind.MUST_INCLUDE
        assert kinds["Salah"] is RequirementKind.MUST_EXCLUDE

    def test_a_trailing_phrase_can_override_a_leading_verb(self) -> None:
        """ "Haaland starting and Saliba in the squad" — two different requests."""
        kinds = _kinds("I want Haaland starting and Saliba in the squad")
        assert kinds["Haaland"] is RequirementKind.MUST_START
        assert kinds["Saliba"] is RequirementKind.MUST_INCLUDE

    def test_an_unmentioned_player_gets_no_requirement(self) -> None:
        assert "Palmer" not in _kinds("keep Haaland")

    def test_a_bare_name_with_no_verb_is_not_a_requirement(self) -> None:
        """Naming a player is not asking for him. Inventing one is worse than
        missing one, because the invented one is invisible."""
        assert _kinds("what do you think about Palmer") == {}


class TestCaptaincy:
    @pytest.mark.parametrize(
        "text", ["captain Haaland", "Haaland as captain", "give Haaland the armband"]
    )
    def test_captain_phrasings(self, text: str) -> None:
        assert _kinds(text)["Haaland"] is RequirementKind.MUST_CAPTAIN


class TestFormation:
    def test_a_legal_shape_is_read(self) -> None:
        assert _of_kind("play 3-5-2", RequirementKind.FORMATION) == ["play 3-5-2"]

    @pytest.mark.parametrize("shape", ["3-4-3", "4-4-2", "5-3-2", "3-5-2", "4-5-1"])
    def test_every_legal_shape(self, shape: str) -> None:
        assert _of_kind(f"play {shape}", RequirementKind.FORMATION) == [f"play {shape}"]

    def test_an_illegal_shape_is_ignored_not_invented(self) -> None:
        """3-5-3 is twelve players. Better to miss it than to build to it."""
        assert _of_kind("play 3-5-3", RequirementKind.FORMATION) == []


class TestClubCounts:
    def test_a_floor_is_read(self) -> None:
        labels = _of_kind("at least 3 from Arsenal", RequirementKind.CLUB_FLOOR)
        assert labels
        assert "3" in labels[0]

    def test_a_ceiling_is_read(self) -> None:
        labels = _of_kind("no more than 2 from Liverpool", RequirementKind.CLUB_CEILING)
        assert labels
        assert "2" in labels[0]

    def test_words_count_as_numbers(self) -> None:
        assert _of_kind("at least three from Arsenal", RequirementKind.CLUB_FLOOR)

    def test_an_unknown_club_is_not_guessed(self) -> None:
        assert _of_kind("at least 3 from Barcelona", RequirementKind.CLUB_FLOOR) == []

    def test_an_ambiguous_club_prefix_is_refused(self) -> None:
        """Two clubs sharing a prefix must resolve to neither."""
        parse = parse_squad_requirements(
            "at least 2 from Man",
            players=_PLAYERS,
            teams={"Man City": 15, "Man Utd": 1},
        )
        assert [r for r in parse.requirements if r.kind is RequirementKind.CLUB_FLOOR] == []


class TestBank:
    def test_a_bank_floor_is_read_in_tenths(self) -> None:
        parse = parse_squad_requirements("leave 0.5 in the bank", players=_PLAYERS)
        floors = [r for r in parse.requirements if r.kind is RequirementKind.BANK_FLOOR]
        assert floors
        assert int(floors[0].amount or 0) == 5

    def test_a_zero_bank_floor_is_not_a_requirement(self) -> None:
        parse = parse_squad_requirements("leave 0 in the bank", players=_PLAYERS)
        assert [r for r in parse.requirements if r.kind is RequirementKind.BANK_FLOOR] == []


class TestAmbiguity:
    def test_an_ambiguous_name_is_reported_not_resolved(self) -> None:
        """A wrong lock is invisible; an unresolved one is not."""
        parse = parse_squad_requirements(
            "keep Silva", players={"Silva": 9}, ambiguous_names=["Silva"]
        )
        assert parse.requirements == []
        assert parse.unresolved_names == ["Silva"]


class TestEvidence:
    def test_every_requirement_records_the_phrase_that_produced_it(self) -> None:
        parse = parse_squad_requirements("I want Haaland starting", players=_PLAYERS, teams=_TEAMS)
        assert len(parse.evidence) == len(parse.requirements)
        assert any("Haaland" in evidence for _, _, _, evidence in parse.evidence)

    def test_priorities_put_named_players_above_shapes(self) -> None:
        """When something has to give, a manager means the shape first."""
        parse = parse_squad_requirements(
            "start Haaland, play 3-5-2", players=_PLAYERS, teams=_TEAMS
        )
        by_kind = {r.kind: r.priority for r in parse.requirements}
        assert by_kind[RequirementKind.MUST_START] > by_kind[RequirementKind.FORMATION]


class TestCompiledIntent:
    def test_requirements_reach_the_compiled_intent(self) -> None:
        from xg_alonso.domain.intent import compile_intent

        intent = compile_intent(
            "I want Haaland starting and play 3-5-2", players=_PLAYERS, teams=_TEAMS
        )
        kinds = {r.kind for r in intent.requirements.requirements}
        assert RequirementKind.MUST_START in kinds
        assert RequirementKind.FORMATION in kinds

    def test_an_empty_request_produces_no_requirements(self) -> None:
        from xg_alonso.domain.intent import compile_intent

        assert compile_intent("maximise points", players=_PLAYERS).requirements.requirements == ()


class TestSubstringSafety:
    """Vocabularies are anchored, and this is why.

    Unanchored, `no` matched inside "bru**no**" and — being exactly as close to
    the name as the "starting" that followed it — won the tie-break and
    *excluded* a player the manager had asked to start. A requirement inverted
    by a substring match is the worst failure this parser has, because the squad
    comes back looking deliberate.
    """

    def test_a_name_containing_no_does_not_exclude(self) -> None:
        players = {"fernandes": 1}
        parse = parse_squad_requirements("bruno fernandes starting", players=players)
        assert [r.kind for r in parse.requirements] == [RequirementKind.MUST_START]

    @pytest.mark.parametrize(
        "text",
        [
            "bruno fernandes starting",
            "keep bruno fernandes",
            "brown fernandes starting",
        ],
    )
    def test_substrings_never_flip_a_requirement(self, text: str) -> None:
        parse = parse_squad_requirements(text, players={"fernandes": 1})
        assert parse.requirements
        assert all(r.kind is not RequirementKind.MUST_EXCLUDE for r in parse.requirements)

    def test_a_negated_ownership_verb_still_excludes(self) -> None:
        """ "Never pick X" puts `pick` nearer the name than `never`."""
        assert (
            parse_squad_requirements("never pick Haaland", players=_PLAYERS).requirements[0].kind
            is RequirementKind.MUST_EXCLUDE
        )

    @pytest.mark.parametrize(
        "text", ["never pick Haaland", "don't want Haaland", "do not buy Haaland", "no Haaland"]
    )
    def test_every_negation_excludes(self, text: str) -> None:
        kinds = {r.kind for r in parse_squad_requirements(text, players=_PLAYERS).requirements}
        assert kinds == {RequirementKind.MUST_EXCLUDE}


class TestLongestMatchWins:
    """Two players can share a name, and the longer one is the one written."""

    INDEX: ClassVar[dict[str, int]] = {
        "b.fernandes": 141746,
        "bruno fernandes": 141746,
        "mateus fernandes": 551226,
        "haaland": 223094,
    }

    def test_the_full_name_beats_a_shorter_key(self) -> None:
        parse = parse_squad_requirements("bruno fernandes starting", players=self.INDEX)
        assert [int(c) for r in parse.requirements for c in r.players] == [141746]

    def test_the_other_player_is_still_reachable(self) -> None:
        parse = parse_squad_requirements("mateus fernandes starting", players=self.INDEX)
        assert [int(c) for r in parse.requirements for c in r.players] == [551226]

    def test_one_phrase_produces_one_requirement(self) -> None:
        """Both keys match the same span; only the specific one may count."""
        parse = parse_squad_requirements("bruno fernandes starting", players=self.INDEX)
        assert len(parse.requirements) == 1
