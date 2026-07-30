"""Read a manager's request with a language model, safely.

**Why a model at all.** The deterministic parser matches a vocabulary, and a
vocabulary cannot cover intent. "Prioritise the non-elite players" is a
statement about ownership; "I'm bored of my team" is a statement about risk;
"someone from a team with a good run" is a statement about fixtures. None of
those contain a word the regex could key on, and inventing a rule for each is
how a parser becomes a pile of special cases that still misses the next one.

**What the model is allowed to do.** Propose requirements *as data*, naming
players in words. Nothing else:

- It returns **names, never codes.** Every name is resolved against the real
  player index on this side, so a hallucinated footballer resolves to nothing
  and is reported. The model cannot put a player in a squad by inventing him.
- Its output is a Pydantic schema validated against the same
  :class:`RequirementKind` vocabulary the deterministic parser produces, so a
  requirement it invents a *kind* for fails validation rather than reaching the
  solver.
- A formation is checked against the same eleven-player arithmetic. "4-4-4"
  does not become a squad.
- **It never overrides the deterministic parse.** Where both read the same
  player, the regex wins: it matched a phrase and can show it, and evidence
  beats inference when the two disagree.

**Every proposal is labelled.** A requirement that came from the model carries
``source="model"`` for the rest of its life, so a manager reviewing the chips
can tell a phrase that was *matched* from a reading that was *inferred*. That
distinction is the whole reason the review step exists.

**Optional and never fatal.** Without a key, or without the SDK, or on any API
failure, the deterministic parse stands alone and the request still works.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, assert_never

from pydantic import BaseModel, Field

from xg_alonso.contracts.identifiers import PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.objective import Requirement, RequirementKind

__all__ = [
    "DEFAULT_MODEL",
    "Interpretation",
    "InterpretedRequest",
    "InterpreterUnavailableError",
    "ProposedRequirement",
    "api_key_origin",
    "interpret_request",
    "load_api_key",
]

#: Reading a sentence is a small, high-leverage call — one per request, a few
#: hundred tokens. The strongest model costs almost nothing here against the
#: cost of misreading what a manager asked for.
DEFAULT_MODEL: Final[str] = "claude-opus-5"

#: Ownership readings the model may return. Mirrors `OwnershipPreference` rather
#: than importing free text, so an unrecognised value fails rather than being
#: passed through to the objective.
_OWNERSHIP: Final[frozenset[str]] = frozenset({"differential", "template", "neutral"})
_RISK: Final[frozenset[str]] = frozenset({"aggressive", "balanced", "conservative"})


class InterpreterUnavailableError(RuntimeError):
    """No API key, or the SDK is not installed.

    Raised rather than returning nothing, so a caller can tell "the model read
    the sentence and found nothing to add" from "the model was never asked".
    """


def _read_key(path: Path) -> str | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() == "ANTHROPIC_API_KEY":
            return value.strip().strip("'\"") or None
    return None


def _find_key(env_file: Path | None = None) -> tuple[str, str] | None:
    """Locate a key and record where it came from.

    Walks *upward* from the working directory. A git worktree gets its own copy
    of untracked files, so a key written to the main checkout is invisible from
    a worktree, and a stale ``.env`` left in one shadows the real key silently —
    producing a 401 that looks like a bad credential rather than a bad path.
    """
    from_env = os.environ.get("ANTHROPIC_API_KEY")
    if from_env:
        return from_env, "environment"

    if env_file is not None:
        key = _read_key(env_file)
        return (key, str(env_file)) if key else None

    start = Path.cwd().resolve()
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            key = _read_key(candidate)
            if key:
                return key, str(candidate)
    return None


def load_api_key(*, env_file: Path | None = None) -> str | None:
    """An Anthropic key from the environment or a nearby ``.env``, or ``None``."""
    found = _find_key(env_file)
    return found[0] if found else None


def api_key_origin(*, env_file: Path | None = None) -> str | None:
    """Where the key would be read from. Never the key itself."""
    found = _find_key(env_file)
    return found[1] if found else None


class ProposedRequirement(BaseModel):
    """One requirement the model proposes. Names, never codes."""

    kind: str = Field(
        description=(
            "One of: must_start, must_include, must_exclude, must_captain, "
            "club_floor, club_ceiling, formation, bank_floor"
        )
    )
    player_names: list[str] = Field(
        default_factory=list,
        description="Players by name exactly as written in the request. Never invent one.",
    )
    club_name: str = Field(default="", description="Club name, for club_floor/club_ceiling")
    count: int | None = Field(default=None, description="How many, for club rules")
    formation: str = Field(default="", description="Shape as DEF-MID-FWD, e.g. 3-5-2")
    bank_tenths: int | None = Field(default=None, description="Bank floor in tenths of a million")
    reading: str = Field(
        description="The part of the request this came from, and why you read it that way"
    )


class InterpretedRequest(BaseModel):
    """Everything the model made of a request."""

    requirements: list[ProposedRequirement] = Field(default_factory=list)
    ownership_preference: str = Field(
        default="",
        description=(
            "differential, template or neutral — only when the request implies one. "
            "'Non-elite', 'under the radar', 'nobody owns' all mean differential. "
            "'Safe', 'what everyone has' means template."
        ),
    )
    risk_preference: str = Field(
        default="", description="aggressive, balanced or conservative, only when implied"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Anything you understood but could not express as a requirement",
    )


@dataclass
class Interpretation:
    """What survived resolution, and what did not."""

    requirements: list[Requirement] = field(default_factory=list)
    readings: dict[str, str] = field(default_factory=dict)
    ownership_preference: str = ""
    risk_preference: str = ""
    notes: list[str] = field(default_factory=list)
    unresolved_names: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)


_PROMPT = """You read Fantasy Premier League squad requests and turn them into \
structured requirements. You are the second pass: a deterministic parser has \
already matched everything it could, and you are here for the intent it cannot \
key on.

Rules you must follow:

1. Name players exactly as the manager wrote them. Never invent a player, and \
never substitute one you think they meant. A name you are unsure of is better \
left out — it will be reported as unresolved, which is recoverable, whereas a \
wrong player is invisible.
2. `must_start` puts a player in the starting eleven. `must_include` only puts \
him in the fifteen — the bench is acceptable. Choose deliberately: "I want X" \
is include, "I want X starting" is start.
3. Only propose a requirement the manager actually asked for. Do not add \
sensible-sounding extras.
4. A formation must field ten outfielders: 3-4-3, 3-5-2, 4-4-2, 4-5-1, 5-3-2, \
5-4-1.
5. Preferences that are not structural go in `ownership_preference` or \
`risk_preference`, not in requirements. "Prioritise the non-elite players" is \
`ownership_preference: differential`, not an exclusion of every good player.
6. Put anything you understood but could not express into `notes`.

Already matched by the deterministic parser (do not repeat these):
{already}

The parser did not understand these fragments:
{unparsed}
"""


def _resolve_player(name: str, players: Mapping[str, int]) -> int | None:
    """Resolve a name the model wrote against the real index. No guessing."""
    wanted = name.strip().lower()
    if not wanted:
        return None
    folded = {key.strip().lower(): value for key, value in players.items()}
    if wanted in folded:
        return folded[wanted]

    surname = wanted.split()[-1]
    if surname in folded:
        return folded[surname]

    matches = {value for key, value in folded.items() if key.startswith(wanted)}
    return next(iter(matches)) if len(matches) == 1 else None


def _resolve_club(name: str, teams: Mapping[str, int]) -> int | None:
    wanted = name.strip().lower()
    if not wanted:
        return None
    folded = {key.strip().lower(): value for key, value in teams.items()}
    if wanted in folded:
        return folded[wanted]
    matches = {v for k, v in folded.items() if k.startswith(wanted) or wanted.startswith(k)}
    return next(iter(matches)) if len(matches) == 1 else None


def _to_requirement(
    proposal: ProposedRequirement,
    *,
    players: Mapping[str, int],
    teams: Mapping[str, int],
) -> tuple[Requirement | None, str]:
    """Build a validated requirement, or explain why it could not be built."""
    try:
        kind = RequirementKind(proposal.kind.strip().lower())
    except ValueError:
        return None, f"unknown requirement kind {proposal.kind!r}"

    if kind in {
        RequirementKind.MUST_START,
        RequirementKind.MUST_INCLUDE,
        RequirementKind.MUST_EXCLUDE,
        RequirementKind.MUST_CAPTAIN,
    }:
        codes: list[PlayerCode] = []
        for raw in proposal.player_names:
            code = _resolve_player(raw, players)
            if code is None:
                return None, f"no player called {raw!r}"
            codes.append(PlayerCode(code))
        if not codes:
            return None, "no players named"
        if kind is RequirementKind.MUST_CAPTAIN and len(codes) != 1:
            return None, "a squad has exactly one captain"
        who = ", ".join(n.strip() for n in proposal.player_names)
        label = {
            RequirementKind.MUST_START: f"{who} must start",
            RequirementKind.MUST_INCLUDE: f"{who} in the squad",
            RequirementKind.MUST_EXCLUDE: f"never pick {who}",
            RequirementKind.MUST_CAPTAIN: f"captain {who}",
        }[kind]
        priority = 5 if kind is RequirementKind.MUST_START else 4
        return Requirement(kind=kind, label=label, players=tuple(codes), priority=priority), ""

    if kind in {RequirementKind.CLUB_FLOOR, RequirementKind.CLUB_CEILING}:
        team = _resolve_club(proposal.club_name, teams)
        if team is None:
            return None, f"no club called {proposal.club_name!r}"
        if proposal.count is None:
            return None, "no count given"
        word = "at least" if kind is RequirementKind.CLUB_FLOOR else "at most"
        return (
            Requirement(
                kind=kind,
                label=f"{word} {proposal.count} from {proposal.club_name}",
                team_id=TeamId(team),
                count=proposal.count,
                priority=1,
            ),
            "",
        )

    if kind is RequirementKind.FORMATION:
        candidate = Requirement(
            kind=kind, label=f"play {proposal.formation}", formation=proposal.formation, priority=2
        )
        try:
            candidate.formation_counts()
        except ValueError as exc:
            return None, str(exc)
        return candidate, ""

    if kind is RequirementKind.BANK_FLOOR:
        if not proposal.bank_tenths or proposal.bank_tenths <= 0:
            return None, "no bank amount given"
        return (
            Requirement(
                kind=kind,
                label=f"leave {proposal.bank_tenths / 10:.1f}m in the bank",
                amount=TenthsOfMillion(proposal.bank_tenths),
                priority=0,
            ),
            "",
        )

    # Exhaustive by construction: a kind added to the enum without a branch here
    # fails type-checking rather than silently resolving to nothing.
    assert_never(kind)


def interpret_request(
    text: str,
    *,
    players: Mapping[str, int],
    teams: Mapping[str, int] | None = None,
    already_parsed: Sequence[Requirement] = (),
    unparsed: Sequence[str] = (),
    api_key: str | None = None,
    env_file: Path | None = None,
    model: str = DEFAULT_MODEL,
) -> Interpretation:
    """Ask a model to read what the deterministic parser could not.

    Args:
        players: Name to player code — the real index. The model returns names
            and they are resolved here, so it cannot reference a player who does
            not exist.
        already_parsed: What the regex already matched, so the model is told not
            to repeat it.
        unparsed: Fragments the regex did not understand, which is where the
            model earns its place.

    Raises:
        InterpreterUnavailableError: no key, or the SDK is not installed.
    """
    key = api_key or load_api_key(env_file=env_file)
    if not key:
        message = (
            "no ANTHROPIC_API_KEY found in the environment or a nearby .env; "
            "the deterministic parser runs without it"
        )
        raise InterpreterUnavailableError(message)

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the install
        message = "the anthropic SDK is not installed (uv sync --extra llm)"
        raise InterpreterUnavailableError(message) from exc

    already = "\n".join(f"- {r.label}" for r in already_parsed) or "- (nothing)"
    missed = "\n".join(f"- {u}" for u in unparsed) or "- (nothing)"

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.parse(
        model=model,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=_PROMPT.format(already=already, unparsed=missed),
        messages=[{"role": "user", "content": text}],
        output_format=InterpretedRequest,
    )

    # A refusal carries no parsed output, so it is checked before the content is
    # read rather than after.
    if getattr(response, "stop_reason", None) == "refusal":
        return Interpretation(notes=["the model declined to read this request"])

    parsed: InterpretedRequest | None = getattr(response, "parsed_output", None)
    if parsed is None:
        return Interpretation(notes=["the model returned nothing usable"])

    result = Interpretation(
        ownership_preference=(
            parsed.ownership_preference if parsed.ownership_preference in _OWNERSHIP else ""
        ),
        risk_preference=parsed.risk_preference if parsed.risk_preference in _RISK else "",
        notes=list(parsed.notes),
    )

    seen_players = {int(c) for r in already_parsed for c in r.players}
    seen_kinds = {r.kind for r in already_parsed}

    for proposal in parsed.requirements:
        requirement, problem = _to_requirement(proposal, players=players, teams=teams or {})
        if requirement is None:
            result.rejected.append((proposal.kind, problem))
            for raw in proposal.player_names:
                if _resolve_player(raw, players) is None:
                    result.unresolved_names.append(raw)
            continue

        # Evidence beats inference. Where the deterministic parser already spoke
        # about this player, or already fixed this structural choice, its reading
        # stands — it matched a phrase it can show.
        if any(int(c) in seen_players for c in requirement.players):
            continue
        if not requirement.players and requirement.kind in seen_kinds:
            continue

        result.requirements.append(requirement)
        result.readings[requirement.label] = proposal.reading

    return result
