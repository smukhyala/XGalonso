"""Tests for choosing who is worth searching, and what survives the contract.

No network here. The search itself is the model's job; what is testable — and
what actually keeps this channel safe — is who gets looked up and which
findings are allowed to become signals.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from xg_alonso.contracts.form import FormDirection, FormSignal, FormStrength
from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.interpreter.news import shortlist_from

_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _row(code: int, name: str, *, owned: float = 1.0, status: str = "a") -> dict[str, object]:
    return {
        "player_code": code,
        "web_name": name,
        "selected_by_percent": owned,
        "status": status,
    }


class TestWhoGetsSearched:
    def test_the_squad_comes_first(self) -> None:
        """A search budget spent on somebody you do not own is wasted."""
        rows = [_row(1, "Popular", owned=50.0), _row(2, "Mine", owned=0.1)]
        shortlist = shortlist_from(rows, squad=[2], limit=2)
        assert shortlist[0].name == "Mine"
        assert shortlist[0].in_squad

    def test_then_the_widely_owned(self) -> None:
        rows = [_row(1, "Rare", owned=0.2), _row(2, "Common", owned=40.0)]
        assert shortlist_from(rows, limit=2)[0].name == "Common"

    def test_a_flagged_player_is_skipped(self) -> None:
        """FPL has already said it. An inference restating a fact would
        double-count while looking like corroboration."""
        rows = [_row(1, "Injured", owned=60.0, status="i"), _row(2, "Fit", owned=1.0)]
        assert [e.name for e in shortlist_from(rows, limit=5)] == ["Fit"]

    def test_the_limit_is_the_cost_dial(self) -> None:
        rows = [_row(i, f"P{i}", owned=float(i)) for i in range(1, 60)]
        assert len(shortlist_from(rows, limit=8)) == 8

    def test_projection_breaks_a_tie_on_ownership(self) -> None:
        rows = [_row(1, "Low", owned=1.0), _row(2, "High", owned=1.0)]
        shortlist = shortlist_from(rows, expected_points={2: 7.0}, limit=2)
        assert shortlist[0].name == "High"

    def test_the_reason_is_stated(self) -> None:
        rows = [_row(1, "Mine", owned=0.1)]
        assert shortlist_from(rows, squad=[1], limit=1)[0].why == "in your squad"


class TestWhatBecomesASignal:
    """The contract does the refusing; these confirm it actually refuses."""

    def _signal(self, **overrides: object) -> FormSignal:
        payload: dict[str, object] = {
            "player_code": PlayerCode(1),
            "direction": FormDirection.NEGATIVE,
            "strength": FormStrength.CLEAR,
            "summary": "Left out of the pre-season tour after an extended rest.",
            "sources": ("https://example.com/report",),
            "observed_at": _NOW,
            "expires_at": _NOW + timedelta(days=6),
        }
        payload.update(overrides)
        return FormSignal(**payload)  # type: ignore[arg-type]

    def test_a_sourceless_claim_cannot_be_constructed(self) -> None:
        """The rule that makes this channel safe to have at all."""
        with pytest.raises(ValueError, match="at least 1 item"):
            self._signal(sources=())

    def test_a_source_that_is_not_a_url_is_refused(self) -> None:
        with pytest.raises(ValueError, match="rumours do not move projections"):
            self._signal(sources=("someone on twitter",))

    def test_a_signal_must_expire(self) -> None:
        with pytest.raises(ValueError, match="must expire"):
            self._signal(expires_at=_NOW)

    def test_the_strongest_signal_is_clamped(self) -> None:
        """No amount of narrative overturns a real difference in the numbers."""
        strongest = self._signal(strength=FormStrength.STRONG, direction=FormDirection.POSITIVE)
        assert strongest.multiplier == pytest.approx(1.15)

    def test_a_negative_signal_scales_down(self) -> None:
        assert self._signal(strength=FormStrength.STRONG).multiplier == pytest.approx(0.85)

    def test_an_expired_signal_is_not_live(self) -> None:
        """A stale signal is worse than none, because it looks current."""
        signal = self._signal()
        assert signal.is_live(_NOW)
        assert not signal.is_live(_NOW + timedelta(days=7))
