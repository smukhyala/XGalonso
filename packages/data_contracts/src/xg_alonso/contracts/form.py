"""External form signals — information the FPL API does not carry.

**This deliberately crosses decision D6** ("zero budget, no paid providers"), at
the project owner's explicit request. It is a separate crossing from the
2026-07-29 relaxation, which permits *fetching* free public data from origins
that allow it; this module fetches nothing at all. The
motivating case is real and the API genuinely cannot see it: a player can arrive
at a season off a poor international tournament, having lost his place and his
rhythm, while every statistic the API publishes still describes the player he
was three months ago.

The crossing is bounded rather than open-ended, because an unbounded free-text
channel into a prediction is exactly the arrangement the reason-code module
exists to prevent:

1. **A signal may only scale, never assert.** It multiplies expected points
   within a hard clamp. It cannot introduce a projection, a statistic, or a
   number of its own.
2. **A signal without a source cannot be constructed.** Every one carries at
   least one URL, and the explanation that cites it renders that URL.
3. **A signal expires.** Form information is perishable, and a stale signal is
   worse than none because it looks current. Reading one past its expiry drops
   it rather than applying it.
4. **The magnitude is quantised.** Three strengths, not a free dial, so nobody
   is tempted to express a confidence the evidence does not support.

There is no scraper here and no provider dependency. Signals are data, written
to a file by whatever process a user trusts — a human, a search tool, a
newsroom feed — and this module defines only what a signal must satisfy to be
allowed to move a number.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.identifiers import PlayerCode

__all__ = [
    "FORM_SIGNAL_CLAMP",
    "FormDirection",
    "FormSignal",
    "FormStrength",
    "SignalSet",
]

#: The hardest a signal may move a projection, in either direction.
#:
#: Fifteen percent. Large enough to matter for a marginal pick, small enough
#: that no amount of narrative can overturn a real difference in the underlying
#: numbers — which is the correct hierarchy, since the numbers are measured and
#: the narrative is somebody's judgement.
FORM_SIGNAL_CLAMP: Final[float] = 0.15


class FormDirection(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class FormStrength(StrEnum):
    """How strongly the evidence supports the signal.

    Three levels rather than a continuous confidence. A free dial invites
    precision nobody has: the difference between 0.07 and 0.09 is not something
    a reader of match reports can defend, and offering the choice implies it is.
    """

    SLIGHT = "slight"
    CLEAR = "clear"
    STRONG = "strong"

    @property
    def magnitude(self) -> float:
        """Fraction of :data:`FORM_SIGNAL_CLAMP` this strength applies."""
        return {"slight": 0.33, "clear": 0.66, "strong": 1.0}[self.value]


class FormSignal(BaseModel):
    """One piece of outside information about a player, with its provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_code: PlayerCode
    direction: FormDirection
    strength: FormStrength
    summary: str = Field(
        min_length=10,
        max_length=280,
        description=(
            "What was observed, in one sentence, in the words of the source. "
            "Rendered verbatim; never parsed for numbers."
        ),
    )
    sources: tuple[str, ...] = Field(
        min_length=1,
        description="Where this came from. At least one, and shown to the reader.",
    )
    observed_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def _check_usable(self) -> FormSignal:
        if self.expires_at <= self.observed_at:
            raise ValueError("a signal must expire after it was observed")
        if self.observed_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("signal timestamps must be timezone-aware")
        for source in self.sources:
            if not source.startswith(("http://", "https://")):
                raise ValueError(
                    f"source {source!r} is not a URL; a signal that cannot be checked "
                    "is a rumour, and rumours do not move projections here"
                )
        return self

    @property
    def multiplier(self) -> float:
        """What this signal scales expected points by.

        Clamped by construction: :data:`FORM_SIGNAL_CLAMP` bounds the strongest
        possible signal, and a strength scales within that. There is no path by
        which a signal reaches the optimizer as anything other than a number in
        ``[1 - clamp, 1 + clamp]``.
        """
        shift = FORM_SIGNAL_CLAMP * self.strength.magnitude
        return 1.0 - shift if self.direction is FormDirection.NEGATIVE else 1.0 + shift

    def is_live(self, at: datetime) -> bool:
        return self.observed_at <= at < self.expires_at


class SignalSet(BaseModel):
    """Every signal currently on file, and when they were loaded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signals: tuple[FormSignal, ...] = ()
    loaded_at: datetime | None = None

    def live(self, at: datetime) -> dict[PlayerCode, FormSignal]:
        """Signals in force at a moment, keyed by player.

        When a player carries several, the strongest wins rather than the
        newest. Two independent reports of the same slump are not twice the
        evidence, and compounding their multipliers would let a busy news week
        move a projection further than the clamp allows.
        """
        best: dict[PlayerCode, FormSignal] = {}
        for signal in self.signals:
            if not signal.is_live(at):
                continue
            current = best.get(signal.player_code)
            if current is None or signal.strength.magnitude > current.strength.magnitude:
                best[signal.player_code] = signal
        return best
