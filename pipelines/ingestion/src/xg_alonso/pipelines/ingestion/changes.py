"""Detect what changed between two bootstrap payloads.

**One request, every player.** ``bootstrap-static`` carries all 564 players in a
single 1.33 MB response, so detection costs exactly one fetch no matter how many
things moved. Nothing here is per-player and nothing is per-request: the diff
runs on a cadence and everything downstream reads what it wrote.

**Polling faster than five minutes buys nothing.** FPL sends no ``ETag`` and no
``Last-Modified``, so a conditional request is not available — but it does send
``cache-control: max-age=300``, and the CDN serves the same bytes inside that
window. Five minutes is the floor at which a poll can return anything new.

**An unchanged payload is free.** The bronze store is content-addressed, so
identical bytes write nothing; comparing content hashes short-circuits the whole
diff before a single player is examined.

**Change is not significance.** One real poll turned up ten status changes, of
which nine were sub-1%-owned players joining lower-league clubs on loan and one
was a 9.9%-owned striker with a foot injury sitting in the recommended eleven.
Reporting those ten as equals would bury the only one that mattered, so
materiality is judged here and the judgement carries its reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final

from xg_alonso.contracts.events import (
    EventKind,
    Materiality,
    PlayerEvent,
    SnapshotDiff,
    availability_direction,
)
from xg_alonso.contracts.identifiers import PlayerCode

__all__ = [
    "MATERIAL_OWNERSHIP",
    "diff_bootstrap",
    "elements_by_code",
]

#: Ownership at or above which a change is worth surfacing unprompted.
#:
#: Two percent is roughly "one manager in fifty holds him". Below that a change
#: is real, recorded, and not worth interrupting anybody for. The threshold is
#: deliberately not tuned to the observed sample: it is a statement about who
#: should be told, not a fit to one poll.
MATERIAL_OWNERSHIP: Final[float] = 2.0

#: Expected points at or above which a change is worth surfacing regardless of
#: ownership. A newly-injured player nobody owns yet is exactly the thing worth
#: knowing before everybody else does.
MATERIAL_EXPECTED_POINTS: Final[float] = 4.5

#: Probability drop that counts as news on its own, when `status` did not move.
#: Twenty-five points is one FPL band — 100 to 75 — and halves the case for a
#: captaincy without making the player unpickable.
MATERIAL_CHANCE_DROP: Final[int] = 25

#: Price movement worth an event, in tenths. FPL moves prices by 0.1m.
_PRICE_STEP: Final[int] = 1


def elements_by_code(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Index a bootstrap payload on the stable player code.

    Keyed on ``code``, never ``id``: FPL reissues element ids every season, so a
    diff keyed on them would report an entire league of spurious changes each
    August.
    """
    out: dict[int, dict[str, Any]] = {}
    for element in payload.get("elements", []):
        code = element.get("code")
        if code is None:
            continue
        out[int(code)] = dict(element)
    return out


def _float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _news_time(element: Mapping[str, Any]) -> datetime | None:
    raw = element.get("news_added")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _materiality(
    *,
    ownership: float | None,
    expected_points: float | None,
    in_squad: bool,
) -> tuple[Materiality, str]:
    """How loudly to report a change, and why.

    The reason is returned with the verdict because a threshold nobody can see
    is a threshold nobody can argue with, and these are judgement calls rather
    than measurements.
    """
    if in_squad:
        return Materiality.CRITICAL, "he is in the current squad"
    if ownership is not None and ownership >= MATERIAL_OWNERSHIP:
        return Materiality.MATERIAL, f"{ownership:.1f}% of managers own him"
    if expected_points is not None and expected_points >= MATERIAL_EXPECTED_POINTS:
        return Materiality.MATERIAL, f"projected {expected_points:.1f} points"
    if ownership is not None:
        return Materiality.MINOR, f"only {ownership:.1f}% own him"
    return Materiality.MINOR, "little exposure"


def diff_bootstrap(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    *,
    detected_at: datetime,
    squad: Sequence[int] = (),
    expected_points: Mapping[int, float] | None = None,
    previous_hash: str = "",
    current_hash: str = "",
    payload_bytes: int = 0,
) -> SnapshotDiff:
    """Compare two bootstrap payloads and describe what moved.

    Args:
        previous: The older payload, or ``None`` on a first run. A first run
            reports nothing rather than reporting the entire league as new —
            there is no change without a baseline.
        squad: Player codes the manager holds. Anything about these is critical,
            because it is the only class of change that demands acting today.
        expected_points: Current projections, so a change to a highly-rated
            player nobody owns yet still surfaces.
    """
    held = {int(code) for code in squad}
    points = {int(k): float(v) for k, v in (expected_points or {}).items()}
    events: list[PlayerEvent] = []

    now = elements_by_code(current)
    before = elements_by_code(previous) if previous is not None else {}

    if previous is None:
        return SnapshotDiff(
            events=(),
            compared_at=detected_at,
            previous_snapshot=previous_hash,
            current_snapshot=current_hash,
            players_compared=len(now),
            payload_bytes=payload_bytes,
        )

    def emit(
        code: int,
        element: Mapping[str, Any],
        kind: EventKind,
        headline: str,
        *,
        was: str | None = None,
        is_now: str | None = None,
        detail: str = "",
        force: Materiality | None = None,
    ) -> None:
        ownership = _float(element.get("selected_by_percent"))
        projected = points.get(code)
        in_squad = code in held
        verdict, reason = _materiality(
            ownership=ownership, expected_points=projected, in_squad=in_squad
        )
        events.append(
            PlayerEvent(
                player_code=PlayerCode(code),
                web_name=str(element.get("web_name") or code),
                kind=kind,
                materiality=force or verdict,
                before=was,
                after=is_now,
                detected_at=detected_at,
                source_reported_at=_news_time(element),
                headline=headline,
                detail=detail,
                reason=reason,
                ownership=ownership,
                expected_points=projected,
                in_squad=in_squad,
            )
        )

    for code, element in now.items():
        was = before.get(code)
        name = str(element.get("web_name") or code)

        if was is None:
            emit(code, element, EventKind.JOINED, f"{name} appeared in the game")
            continue

        old_status = was.get("status")
        new_status = element.get("status")
        if old_status != new_status:
            direction = availability_direction(old_status, new_status)
            verb = "is now" if direction >= 0 else "has been downgraded to"
            emit(
                code,
                element,
                EventKind.AVAILABILITY,
                f"{name} {verb} {new_status}",
                was=old_status,
                is_now=new_status,
                detail=str(element.get("news") or ""),
            )

        old_chance = _int(was.get("chance_of_playing_next_round"))
        new_chance = _int(element.get("chance_of_playing_next_round"))
        # A move to or from "unstated" is not a probability change: FPL clears
        # the field when a player is fully fit, and reading that as a drop from
        # 100 would invent an event every time somebody recovered.
        if (
            old_status == new_status
            and old_chance is not None
            and new_chance is not None
            and old_chance != new_chance
        ):
            drop = old_chance - new_chance
            emit(
                code,
                element,
                EventKind.CHANCE_OF_PLAYING,
                f"{name} is now {new_chance}% to play, was {old_chance}%",
                was=str(old_chance),
                is_now=str(new_chance),
                detail=str(element.get("news") or ""),
                force=None if drop >= MATERIAL_CHANCE_DROP else Materiality.MINOR,
            )

        old_news = (was.get("news") or "").strip()
        new_news = (element.get("news") or "").strip()
        if new_news and new_news != old_news and old_status == new_status:
            emit(
                code,
                element,
                EventKind.NEWS,
                f"{name}: {new_news}",
                was=old_news or None,
                is_now=new_news,
                detail=new_news,
            )

        old_price = _int(was.get("now_cost"))
        new_price = _int(element.get("now_cost"))
        if (
            old_price is not None
            and new_price is not None
            and abs(new_price - old_price) >= _PRICE_STEP
        ):
            way = "rose" if new_price > old_price else "fell"
            emit(
                code,
                element,
                EventKind.PRICE,
                f"{name} {way} to {new_price / 10:.1f}m",
                was=f"{old_price / 10:.1f}",
                is_now=f"{new_price / 10:.1f}",
            )

    for code, element in before.items():
        if code not in now:
            emit(
                code,
                element,
                EventKind.DEPARTED,
                f"{element.get('web_name') or code} is no longer in the game",
            )

    return SnapshotDiff(
        events=tuple(events),
        compared_at=detected_at,
        previous_snapshot=previous_hash,
        current_snapshot=current_hash,
        players_compared=len(now),
        payload_bytes=payload_bytes,
    )
