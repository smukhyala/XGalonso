"""Tests for resolving the names a manager actually types.

**A key two players share is not indexed at all.** Two Fernandes is not
hypothetical: Bruno Borges Fernandes and Mateus Fernandes are both in the
2026/27 game, and the second one's *display name* is literally "Fernandes".
The ambiguity rule originally covered derived surnames but exempted display
names, which let exactly that case through the front door — "bruno fernandes"
locked Mateus, silently.
"""

from __future__ import annotations

from typing import ClassVar

from xg_alonso.domain.intent import build_name_index

_DISPLAY = {141746: "B.Fernandes", 551226: "Fernandes", 223094: "Haaland"}
_FULL = {
    141746: "Bruno Borges Fernandes",
    551226: "Mateus Fernandes",
    223094: "Erling Haaland",
}


class TestAmbiguity:
    def test_a_shared_display_name_is_not_indexed(self) -> None:
        assert "fernandes" not in build_name_index(_DISPLAY, full_names=_FULL)

    def test_a_shared_surname_is_not_indexed(self) -> None:
        index = build_name_index({1: "Silva", 2: "Silva"})
        assert "silva" not in index

    def test_refusing_is_preferred_to_guessing(self) -> None:
        """An unmatched name shows up. A wrong one does not."""
        index = build_name_index(_DISPLAY, full_names=_FULL)
        assert index.get("fernandes") is None


class TestReachability:
    def test_full_names_reach_both_players(self) -> None:
        index = build_name_index(_DISPLAY, full_names=_FULL)
        assert index["bruno fernandes"] == 141746
        assert index["mateus fernandes"] == 551226

    def test_the_display_name_still_works_when_unambiguous(self) -> None:
        index = build_name_index(_DISPLAY, full_names=_FULL)
        assert index["b.fernandes"] == 141746

    def test_an_unshared_surname_resolves(self) -> None:
        index = build_name_index(_DISPLAY, full_names=_FULL)
        assert index["haaland"] == 223094
        assert index["erling haaland"] == 223094

    def test_the_index_is_lowercase(self) -> None:
        assert all(key == key.lower() for key in build_name_index(_DISPLAY, full_names=_FULL))

    def test_full_names_are_optional(self) -> None:
        """Every existing caller passes display names only."""
        assert build_name_index({223094: "Haaland"})["haaland"] == 223094


class TestHyphenatedNames:
    """A manager who omits the hyphen must not lock a different footballer.

    Reported from the plan page: "i want morgan gibbs white" produced the
    requirement *White in the squad*, resolved to Benjamin White (code 198869,
    Arsenal, and injured at the time) rather than Morgan Gibbs-White (222531).

    The cause was that every index key kept its hyphen. "Gibbs-White" is one
    token, so it was offered as `gibbs-white` and never as `gibbs white`, and
    nothing in the index matched what the manager wrote. What *did* match was
    the bare `white` belonging to a different player, sitting inside the typed
    phrase. The longest-key-first rule in the requirement parser could not help,
    because the longer key did not exist.

    Hyphens and apostrophes are common enough in this league — Gibbs-White,
    Aït-Nouri, N'Golo, O'Riley — that typing them exactly is not a fair
    expectation of a text box.
    """

    _DISPLAY: ClassVar[dict[int, str]] = {222531: "Gibbs-White", 198869: "White"}
    _FULL: ClassVar[dict[int, str]] = {
        222531: "Morgan Gibbs-White",
        198869: "Benjamin White",
    }

    def test_the_unhyphenated_full_name_resolves(self) -> None:
        index = build_name_index(self._DISPLAY, full_names=self._FULL)
        assert index["morgan gibbs white"] == 222531

    def test_the_unhyphenated_surname_resolves(self) -> None:
        index = build_name_index(self._DISPLAY, full_names=self._FULL)
        assert index["gibbs white"] == 222531

    def test_the_hyphenated_forms_still_resolve(self) -> None:
        index = build_name_index(self._DISPLAY, full_names=self._FULL)
        assert index["gibbs-white"] == 222531
        assert index["morgan gibbs-white"] == 222531

    def test_the_other_player_keeps_his_own_surname(self) -> None:
        """The fix must not cost Benjamin White the key that is rightly his.

        Splitting "Gibbs-White" on the hyphen and offering `white` for him too
        would make the key ambiguous, and the ambiguity rule would then drop it
        for both. A hyphenated surname is one surname, not two.
        """
        index = build_name_index(self._DISPLAY, full_names=self._FULL)
        assert index["white"] == 198869

    def test_an_apostrophe_is_treated_the_same_way(self) -> None:
        index = build_name_index({1: "O'Riley"}, full_names={1: "Matt O'Riley"})
        assert index["o riley"] == 1
        assert index["matt o riley"] == 1
        assert index["o'riley"] == 1
