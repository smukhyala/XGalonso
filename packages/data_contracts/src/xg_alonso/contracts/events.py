"""Things that changed about a player, and whether they matter.

**Why events rather than a fresher snapshot.** A snapshot answers "what is true
now"; it cannot answer "what changed, and when did we learn it". A manager whose
striker was ruled out overnight does not want a recomputed projection — he wants
to be told. And a system that silently swaps a name in a recommendation, with no
record that anything moved, is one nobody can audit after the fact.

**Detection is a diff, not a scrape.** Everything here is derived by comparing
two snapshots of a source we already fetch. The motivating case makes the point:
a 9.9%-owned striker picked up a foot injury, and FPL published
``status='i'``, ``chance_of_playing_next_round=0`` and a dated ``news`` line
against him. The platform kept recommending him — not because the information
was unavailable, but because nobody re-read it. No amount of scraping fixes
that; one re-poll does.

**Change is not the same as significance.** A single poll turned up ten status
changes, of which one mattered: the rest were sub-1%-owned players joining
lower-league clubs on loan. So an event carries a *materiality* judgement
separate from the fact of the change, and the judgement is explained rather than
asserted — the reason a change is material is itself the useful part.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from xg_alonso.contracts.identifiers import PlayerCode

__all__ = [
    "AVAILABILITY_RANK",
    "EventKind",
    "Materiality",
    "PlayerEvent",
    "SnapshotDiff",
    "availability_direction",
]


class EventKind(StrEnum):
    """What kind of change was observed."""

    AVAILABILITY = "availability"
    """``status`` moved — the field that decides whether a player is pickable."""

    CHANCE_OF_PLAYING = "chance_of_playing"
    """The published probability moved without ``status`` changing.

    Distinct from availability on purpose: 100 to 75 keeps a player selectable
    and still halves the case for captaining him.
    """

    NEWS = "news"
    """A dated announcement appeared or changed. Carries ``news_added``, which is
    the only *source-authoritative* timestamp in the system — a diff can tell you
    something moved but never when the club actually said it."""

    PRICE = "price"
    OWNERSHIP = "ownership"
    JOINED = "joined"
    """A player appeared in the payload who was not there before."""

    DEPARTED = "departed"
    """A player vanished from the payload. Not the same as unavailable: an
    unavailable player still has a price and can be planned around."""


class Materiality(StrEnum):
    """Whether a change can plausibly alter a decision."""

    CRITICAL = "critical"
    """In the current squad or recommendation. Act now."""

    MATERIAL = "material"
    """Widely owned or highly rated. Worth surfacing unprompted."""

    MINOR = "minor"
    """Real, recorded, and not worth interrupting anyone for."""

    @property
    def is_worth_surfacing(self) -> bool:
        return self is not Materiality.MINOR


#: Availability codes ordered worst to best, so a transition has a direction.
#:
#: ``d`` (doubtful) sits above ``i``/``s`` because a doubtful player is still
#: selectable and still scores; the others are not and do not.
AVAILABILITY_RANK: Final[dict[str, int]] = {
    "u": 0,
    "n": 0,
    "i": 1,
    "s": 1,
    "d": 2,
    "a": 3,
}


def availability_direction(before: str | None, after: str | None) -> int:
    """``-1`` worse, ``+1`` better, ``0`` unranked or unchanged.

    Unknown codes return ``0`` rather than guessing. FPL has added codes before,
    and treating an unrecognised one as "fine" would suppress exactly the event
    worth seeing.
    """
    if before is None or after is None:
        return 0
    was = AVAILABILITY_RANK.get(before.lower())
    now = AVAILABILITY_RANK.get(after.lower())
    if was is None or now is None or was == now:
        return 0
    return 1 if now > was else -1


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PlayerEvent(_Frozen):
    """One observed change about one player."""

    player_code: PlayerCode
    web_name: str = ""
    kind: EventKind
    materiality: Materiality = Materiality.MINOR

    before: str | None = None
    after: str | None = None

    detected_at: datetime = Field(
        description="When the diff ran. Ours, not the source's — an observation, not a fact."
    )
    source_reported_at: datetime | None = Field(
        default=None,
        description=(
            "``news_added`` when the source publishes one. The only timestamp "
            "here that describes the world rather than describing us."
        ),
    )

    headline: str = Field(description="What happened, in one line a person would say")
    detail: str = Field(default="", description="The source's own words, when it gave any")
    reason: str = Field(
        default="",
        description=(
            "Why this materiality. Explained rather than asserted, because the "
            "judgement is the part a reader needs to check."
        ),
    )

    ownership: float | None = Field(default=None, ge=0.0, le=100.0)
    expected_points: float | None = None
    in_squad: bool = False

    @field_validator("detected_at", "source_reported_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        """Timestamps carry a timezone or they are refused.

        A naive datetime compares wrongly against the UTC everything else
        produces, and the comparison that goes wrong is the availability check.
        """
        if value is not None and value.tzinfo is None:
            message = "event timestamps must be timezone-aware"
            raise ValueError(message)
        return value

    @property
    def is_bad_news(self) -> bool:
        """Whether this made a player worse, for a reader scanning a list."""
        if self.kind is EventKind.AVAILABILITY:
            return availability_direction(self.before, self.after) < 0
        return self.kind in {EventKind.DEPARTED, EventKind.NEWS}


class SnapshotDiff(_Frozen):
    """Everything one comparison turned up, plus what it cost to find out."""

    events: tuple[PlayerEvent, ...] = ()
    compared_at: datetime
    previous_snapshot: str = Field(default="", description="Content hash of the older payload")
    current_snapshot: str = Field(default="", description="Content hash of the newer payload")
    players_compared: int = 0
    payload_bytes: int = 0

    @property
    def unchanged(self) -> bool:
        """True when the two payloads were byte-identical.

        Worth its own property because it is the common case and the cheap one:
        the bronze store is content-addressed, so an identical payload writes
        nothing and every downstream step can be skipped.
        """
        return bool(self.previous_snapshot) and self.previous_snapshot == self.current_snapshot

    def worth_surfacing(self) -> tuple[PlayerEvent, ...]:
        """Events a person should be shown, worst news first."""
        return tuple(
            sorted(
                (e for e in self.events if e.materiality.is_worth_surfacing),
                key=lambda e: (e.materiality is not Materiality.CRITICAL, -(e.ownership or 0.0)),
            )
        )
