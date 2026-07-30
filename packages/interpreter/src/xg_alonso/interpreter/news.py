"""Search the news for what FPL has not published yet, and file it as a signal.

**What this is for, and what it is not.** FPL publishes availability — injuries,
suspensions, a percentage chance of playing — and a re-poll catches all of it,
so none of that belongs here. What FPL does *not* publish is everything a
manager reads between the lines: a coach saying somebody "needs a rest", a
returning starter who has taken a place back, a striker who has not looked
himself since a tournament. That is rotation and form risk, it moves decisions,
and no endpoint carries it.

**It can only scale, never assert.** Everything produced here is a
:class:`~xg_alonso.contracts.form.FormSignal`, whose contract already settles
the hard questions: a signal multiplies expected points within a ±15% clamp, it
cannot introduce a projection or a statistic of its own, it must carry a source
URL or it cannot be constructed at all, and it expires. A model reading match
reports gets to nudge a number. It does not get to set one.

**A shortlist, not a league.** Searching 564 players every cycle would be slow,
expensive, and mostly about footballers nobody owns. The shortlist is the squad
plus the highly-owned plus the highly-projected — the players where a nudge can
actually change a decision.

**FPL wins where they overlap.** A player FPL has already flagged is skipped
entirely: the official field is a fact and this is an inference, and letting an
inference restate a fact would double-count it while looking like corroboration.

**Unverifiable is discarded, not softened.** A claim the model cannot attach a
URL to is dropped rather than filed at lower strength. The contract enforces
this — a sourceless signal raises on construction — and the rule is the reason
the channel is safe to have at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

from pydantic import BaseModel, Field

from xg_alonso.contracts.form import FormDirection, FormSignal, FormStrength
from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.interpreter.requests import (
    DEFAULT_MODEL,
    InterpreterUnavailableError,
    load_api_key,
)

__all__ = [
    "DEFAULT_SHORTLIST",
    "NewsSweep",
    "ShortlistEntry",
    "search_player_news",
    "shortlist_from",
]

#: How many players one sweep looks at. Each is a web search plus a judgement,
#: so this is the cost dial — and beyond a few dozen the marginal player is one
#: whose news cannot change anybody's team.
DEFAULT_SHORTLIST: Final[int] = 20

#: How long a signal stays live unless the model says otherwise.
#:
#: Six days: long enough to survive to the next deadline, short enough that it
#: cannot quietly outlive the gameweek it was about. Form information is
#: perishable and a stale signal is worse than none, because it looks current.
DEFAULT_TTL: Final[timedelta] = timedelta(days=6)

#: The web-search tool version for the current model family.
_SEARCH_TOOL: Final[str] = "web_search_20260209"


@dataclass(frozen=True)
class ShortlistEntry:
    """One player worth spending a search on."""

    player_code: int
    name: str
    team: str = ""
    ownership: float = 0.0
    expected_points: float = 0.0
    in_squad: bool = False

    @property
    def why(self) -> str:
        if self.in_squad:
            return "in your squad"
        if self.ownership >= 5.0:
            return f"{self.ownership:.1f}% owned"
        return f"projected {self.expected_points:.1f}"


def shortlist_from(
    players: Sequence[Mapping[str, Any]],
    *,
    squad: Sequence[int] = (),
    expected_points: Mapping[int, float] | None = None,
    limit: int = DEFAULT_SHORTLIST,
    skip_flagged: bool = True,
) -> list[ShortlistEntry]:
    """Choose the players whose news could actually change a decision.

    Squad members first, then by ownership and projection. A player FPL has
    already flagged is skipped by default — the official status is a fact and a
    search would only produce an inference restating it.
    """
    held = {int(code) for code in squad}
    points = {int(k): float(v) for k, v in (expected_points or {}).items()}
    entries: list[ShortlistEntry] = []

    for row in players:
        code = row.get("player_code")
        if code is None:
            continue
        code = int(code)
        status = str(row.get("status") or "a")
        if skip_flagged and status != "a":
            continue

        try:
            ownership = float(row.get("selected_by_percent") or 0.0)
        except (TypeError, ValueError):
            ownership = 0.0

        entries.append(
            ShortlistEntry(
                player_code=code,
                name=str(row.get("web_name") or code),
                team=str(row.get("team_name") or ""),
                ownership=ownership,
                expected_points=points.get(code, 0.0),
                in_squad=code in held,
            )
        )

    entries.sort(key=lambda e: (not e.in_squad, -e.ownership, -e.expected_points))
    return entries[:limit]


class NewsFinding(BaseModel):
    """What the model concluded about one player."""

    player_name: str = Field(description="Exactly as given in the shortlist")
    has_signal: bool = Field(
        description="False when nothing beyond FPL's own data was found. Say so rather than reaching."
    )
    direction: str = Field(default="", description="positive or negative")
    strength: str = Field(default="", description="slight, clear or strong")
    summary: str = Field(
        default="", description="One sentence, in the source's own terms. No numbers you invented."
    )
    sources: list[str] = Field(
        default_factory=list, description="URLs. A claim without one will be discarded."
    )


class NewsBatch(BaseModel):
    findings: list[NewsFinding] = Field(default_factory=list)


@dataclass
class NewsSweep:
    """What one sweep produced, and what it refused."""

    signals: list[FormSignal] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    discarded: list[tuple[str, str]] = field(default_factory=list)
    searched_at: datetime | None = None

    def summary(self) -> str:
        found = len(self.signals)
        return (
            f"checked {len(self.checked)} players, "
            f"filed {found} signal{'' if found == 1 else 's'}, "
            f"discarded {len(self.discarded)}"
        )


_PROMPT = """You research Fantasy Premier League players for information the \
official FPL data does **not** carry.

FPL already publishes injuries, suspensions and a percentage chance of playing, \
and that data is re-read automatically. Do not report any of it — you would be \
restating a fact as an inference.

What is worth reporting is everything between the lines:

- rotation risk: a manager hinting somebody needs a rest, or a congested run
- a place lost or won: a returning player taking a starting spot back
- form a scoreline hides: playing through a knock, or looking off the pace
- role change: moved deeper, off penalties, off set pieces

Rules:

1. **Every finding needs a source URL you actually saw in search results.** A \
claim you cannot link will be discarded, not softened. If you did not find \
anything, set `has_signal` to false — that is a useful and expected answer.
2. **Never invent a number.** The summary is prose in the source's own terms. \
Any statistic you write will be ignored.
3. **Direction** is negative when it makes the player worse for the coming \
gameweek, positive when better.
4. **Strength** is `slight` for a single hint, `clear` for something reported \
plainly, `strong` for something several outlets agree on or a manager said \
outright. Most real findings are `slight` or `clear`.
5. Recency matters. Something from three weeks ago is usually not news.

Report only on the players listed. Use the search tool before answering."""


def search_player_news(
    shortlist: Sequence[ShortlistEntry],
    *,
    now: datetime,
    api_key: str | None = None,
    env_file: Path | None = None,
    model: str = DEFAULT_MODEL,
    ttl: timedelta = DEFAULT_TTL,
    max_searches: int = 8,
) -> NewsSweep:
    """Search for team news on a shortlist and file what can be sourced.

    Args:
        now: Observation time. Signals expire ``ttl`` after it, and a backtest
            passes the deadline rather than the wall clock so it sees what was
            live then.
        max_searches: Cap on web searches for the whole sweep, so one run cannot
            spend without bound.

    Raises:
        InterpreterUnavailableError: no key, or the SDK is not installed.
    """
    key = api_key or load_api_key(env_file=env_file)
    if not key:
        message = "no ANTHROPIC_API_KEY found; team-news search needs one"
        raise InterpreterUnavailableError(message)

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the install
        message = "the anthropic SDK is not installed (uv sync --extra llm)"
        raise InterpreterUnavailableError(message) from exc

    sweep = NewsSweep(checked=[e.name for e in shortlist], searched_at=now)
    if not shortlist:
        return sweep

    roster = "\n".join(
        f"- {e.name}" + (f" ({e.team})" if e.team else "") + f" — {e.why}" for e in shortlist
    )
    by_name = {e.name.strip().lower(): e for e in shortlist}

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.parse(
        model=model,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=_PROMPT,
        # The SDK types tool params as a closed union that does not yet name
        # this search version; the wire format is a plain dict either way.
        tools=cast("Any", [{"type": _SEARCH_TOOL, "name": "web_search", "max_uses": max_searches}]),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Today is {now:%d %B %Y}. Find current team news for these "
                    f"players:\n\n{roster}"
                ),
            }
        ],
        output_format=NewsBatch,
    )

    if getattr(response, "stop_reason", None) == "refusal":
        sweep.discarded.append(("*", "the model declined the request"))
        return sweep

    parsed: NewsBatch | None = getattr(response, "parsed_output", None)
    if parsed is None:
        sweep.discarded.append(("*", "the model returned nothing usable"))
        return sweep

    for finding in parsed.findings:
        entry = by_name.get(finding.player_name.strip().lower())
        if entry is None:
            sweep.discarded.append((finding.player_name, "not on the shortlist"))
            continue
        if not finding.has_signal:
            continue

        # Every rejection below is the contract refusing something rather than
        # this module second-guessing the model. A signal that cannot be
        # constructed is one that should never have reached the optimizer.
        try:
            signal = FormSignal(
                player_code=PlayerCode(entry.player_code),
                direction=FormDirection(finding.direction.strip().lower()),
                strength=FormStrength(finding.strength.strip().lower()),
                summary=finding.summary.strip(),
                sources=tuple(s for s in finding.sources if s.strip()),
                observed_at=now,
                expires_at=now + ttl,
            )
        except ValueError as exc:
            sweep.discarded.append((entry.name, str(exc).split("\n")[0]))
            continue

        sweep.signals.append(signal)

    return sweep
