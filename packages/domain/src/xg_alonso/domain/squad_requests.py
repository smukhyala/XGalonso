"""Parse a manager's structural demands on a squad, in their own words.

**Why this is separate from the objective compiler.** ``compile_intent`` reads
what a manager is *trying to achieve* — chase a rank, protect one, grow value.
This reads what the resulting squad must *contain*. The two are different kinds
of statement and get compiled into different things: an objective becomes a
scoring function, a requirement becomes a row in the solver.

**Starting and owning are different requests, and the parser must keep them
apart.** "Keep Haaland" is satisfied by Haaland on the bench. "I want Haaland
starting" is not. The optimizer has separate variables for the fifteen and the
eleven, so collapsing the two here would discard a distinction it can express —
and would quietly over-constrain every squad built from a request that only ever
meant "own him".

**Deterministic, and reviewable.** No language model. Every rule is a regex over
a vocabulary listed here, every match records the phrase that produced it, and
anything unmatched is reported rather than dropped. A requirement the parser
invented is worse than one it missed, because the missed one is visible.

**Names are the hard part and the parser refuses to guess.** A surname shared by
two players resolves to neither: locking the wrong player is invisible in a way
that locking nobody is not. The caller supplies the name index, and
``unresolved`` names come back for the UI to disambiguate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from xg_alonso.contracts.identifiers import PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.objective import (
    ParseSource,
    Requirement,
    RequirementKind,
    SquadRequirements,
)

__all__ = [
    "RequirementParse",
    "parse_squad_requirements",
]

#: Words that put a player in the eleven rather than merely in the squad.
#:
#: Matched on either side of the name, because English puts them on both:
#: "start Haaland" and "Haaland has to start" mean the same thing.
_START_WORDS: Final[str] = r"start(?:s|ing|er)?|in the (?:xi|eleven|starting)|must play|has to play"

#: Words that only ask for ownership. Kept distinct from the above on purpose.
_OWN_WORDS: Final[str] = r"keep|own|have|include|want|bring in|sign|buy|pick|hold|retain|lock"


_EXCLUDE_WORDS: Final[str] = r"avoid|exclude|without|never|no|drop|sell|don'?t (?:want|pick|buy)"

_CAPTAIN_WORDS: Final[str] = r"captain|armband|\(c\)|skipper"

#: Vocabularies that may legitimately appear *after* a name.
#:
#: English states the eleven and the armband after the name — "Haaland
#: starting", "Haaland as captain" — and states ownership after it only as a
#: phrase: "Saliba in the squad". The bare verbs never follow ("Haaland keep" is
#: not English), so admitting them would let a verb governing the *next* name
#: reach backwards into this one.
_TRAILING: Final[dict[RequirementKind, str]] = {
    RequirementKind.MUST_START: _START_WORDS,
    RequirementKind.MUST_CAPTAIN: _CAPTAIN_WORDS,
    RequirementKind.MUST_INCLUDE: r"in (?:the|my) (?:squad|fifteen|15)",
}

#: A legal FPL shape. The keeper is implied, so three numbers summing to ten.
_FORMATION = re.compile(r"\b([345])\s*-\s*([2345])\s*-\s*([123])\b")

_CLUB_FLOOR = re.compile(
    r"\b(?:at least|minimum(?: of)?|min|no fewer than)\s+(\w+)\s+"
    r"(?:players?\s+)?(?:from|of)\s+([a-z' ]{3,20}?)\b(?:\s+players?)?"
)

_CLUB_CEILING = re.compile(
    r"\b(?:at most|maximum(?: of)?|max|no more than|only)\s+(\w+)\s+"
    r"(?:players?\s+)?(?:from|of)\s+([a-z' ]{3,20}?)\b(?:\s+players?)?"
)

_WORD_NUMBERS: Final[dict[str, int]] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}

#: How far either side of a name to look for the verb governing it.
#:
#: Wide enough for "I would really like Haaland to be starting", narrow enough
#: that a verb belonging to the next clause does not reach backwards into this
#: one. Sentence punctuation ends the window regardless.
_WINDOW: Final[int] = 40


@dataclass
class RequirementParse:
    """What the parser made of a request, and what it could not."""

    requirements: list[Requirement] = field(default_factory=list)
    evidence: list[tuple[str, float, ParseSource, str]] = field(default_factory=list)
    unresolved_names: list[str] = field(default_factory=list)
    matched_spans: list[tuple[int, int]] = field(default_factory=list)

    def bundle(self) -> SquadRequirements:
        return SquadRequirements(requirements=tuple(self.requirements))


#: Ordered by strength. A name governed by two verbs takes the nearest, and
#: where two are equally near the earlier entry wins — so "starting" beats
#: "keep" in "keep Haaland starting", which is what that sentence means.
_VOCABULARY: Final[tuple[tuple[str, RequirementKind], ...]] = (
    (_EXCLUDE_WORDS, RequirementKind.MUST_EXCLUDE),
    (_CAPTAIN_WORDS, RequirementKind.MUST_CAPTAIN),
    (_START_WORDS, RequirementKind.MUST_START),
    (_OWN_WORDS, RequirementKind.MUST_INCLUDE),
)


def _clause_around(text: str, start: int, end: int) -> tuple[str, str]:
    """The text either side of a match, cut at sentence punctuation.

    ``and`` is deliberately *not* a cut. It coordinates rather than separates:
    "start Saka and Haaland" governs both names with one verb, and cutting there
    would leave the second name with no verb at all. Commas and full stops do
    cut, so "keep Saka, avoid Salah" cannot leak across.
    """
    before = text[max(0, start - _WINDOW) : start]
    after = text[end : end + _WINDOW]
    before = re.split(r"[.;,]", before)[-1]
    after = re.split(r"[.;,]", after)[0]
    return before, after


def _governing_kind(before: str, after: str) -> RequirementKind | None:
    """Which verb governs this name — the nearest one, not merely any one.

    Scanning the whole window for each vocabulary in turn gets "avoid Salah and
    keep Haaland" wrong, because *avoid* appears in Haaland's window too. The
    verb closest to the name is the one that governs it, and ties break toward
    the stronger reading.
    """
    best: tuple[int, int, RequirementKind] | None = None

    for rank, (pattern, kind) in enumerate(_VOCABULARY):
        # Distance is measured from the name outwards: the last match before it,
        # or the first match after it.
        matches = list(re.finditer(pattern, before))
        if matches:
            distance = len(before) - matches[-1].end()
            if best is None or (distance, rank) < (best[0], best[1]):
                best = (distance, rank, kind)

        trailing = _TRAILING.get(kind)
        if trailing is not None:
            found = re.search(trailing, after)
            if found is not None:
                distance = found.start()
                if best is None or (distance, rank) < (best[0], best[1]):
                    best = (distance, rank, kind)

    return best[2] if best else None


def _number(token: str) -> int | None:
    return _WORD_NUMBERS.get(token.strip().lower())


def parse_squad_requirements(
    text: str,
    *,
    players: Mapping[str, int] | None = None,
    teams: Mapping[str, int] | None = None,
    ambiguous_names: Sequence[str] = (),
) -> RequirementParse:
    """Read structural squad requirements out of a manager's request.

    Args:
        text: What the manager typed.
        players: Display name to player code. Whole-word, case-insensitive.
        teams: Club name or short name to team id, for club floors.
        ambiguous_names: Names that match more than one player. Reported as
            unresolved rather than resolved to whichever came first — a wrong
            lock is invisible in a way that a missing one is not.

    Returns:
        The requirements, the evidence for each, and any name it refused to
        resolve. Priorities are assigned by kind: a named player outranks a
        shape, which outranks a club count, because that is the order managers
        actually mean when they cannot have everything.
    """
    lowered = text.lower()
    parse = RequirementParse()
    ambiguous = {name.lower() for name in ambiguous_names}

    def add(requirement: Requirement, field_name: str, confidence: float, evidence: str) -> None:
        parse.requirements.append(requirement)
        parse.evidence.append((field_name, confidence, ParseSource.EXPLICIT, evidence))

    # --- players -----------------------------------------------------------
    seen: set[int] = set()
    for name, code in (players or {}).items():
        if not name or name.lower() in ambiguous:
            continue
        match = re.search(rf"\b{re.escape(name.lower())}\b", lowered)
        if match is None:
            continue
        if code in seen:
            continue

        before, after = _clause_around(lowered, match.start(), match.end())
        phrase = text[max(0, match.start() - 20) : match.end() + 20].strip()

        kind = _governing_kind(before, after)
        if kind is None:
            continue

        seen.add(code)
        detail = {
            RequirementKind.MUST_EXCLUDE: (f"never pick {name}", 4, 0.85, "exclude"),
            RequirementKind.MUST_CAPTAIN: (f"captain {name}", 3, 0.9, "captain"),
            RequirementKind.MUST_START: (f"{name} must start", 5, 0.9, "start"),
            RequirementKind.MUST_INCLUDE: (f"{name} in the squad", 4, 0.85, "include"),
        }[kind]
        label, priority, confidence, field_name = detail
        add(
            Requirement(
                kind=kind,
                label=label,
                players=(PlayerCode(code),),
                priority=priority,
            ),
            f"requirements.{field_name}[{name}]",
            confidence,
            phrase,
        )
        parse.matched_spans.append(match.span())

    for name in ambiguous_names:
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            parse.unresolved_names.append(name)

    # --- formation ---------------------------------------------------------
    shape = _FORMATION.search(lowered)
    if shape is not None:
        defenders, midfielders, forwards = (int(g) for g in shape.groups())
        outfield = 10
        if defenders + midfielders + forwards == outfield:
            add(
                Requirement(
                    kind=RequirementKind.FORMATION,
                    label=f"play {defenders}-{midfielders}-{forwards}",
                    formation=f"{defenders}-{midfielders}-{forwards}",
                    priority=2,
                ),
                "requirements.formation",
                0.95,
                shape.group(0),
            )
            parse.matched_spans.append(shape.span())

    # --- club counts -------------------------------------------------------
    for pattern, kind, priority in (
        (_CLUB_FLOOR, RequirementKind.CLUB_FLOOR, 1),
        (_CLUB_CEILING, RequirementKind.CLUB_CEILING, 1),
    ):
        for match in pattern.finditer(lowered):
            count = _number(match.group(1))
            club_text = match.group(2).strip()
            team_id = _resolve_team(club_text, teams or {})
            if count is None or team_id is None:
                continue
            word = "at least" if kind is RequirementKind.CLUB_FLOOR else "at most"
            add(
                Requirement(
                    kind=kind,
                    label=f"{word} {count} from {club_text}",
                    team_id=TeamId(team_id),
                    count=count,
                    priority=priority,
                ),
                f"requirements.{kind.value}[{club_text}]",
                0.8,
                match.group(0),
            )
            parse.matched_spans.append(match.span())

    # --- bank --------------------------------------------------------------
    bank = re.search(
        r"\b(?:keep|leave|retain|hold|save)\s*(?:at least\s*)?£?\s*([\d.]+)\s*m?\s*(?:in the )?bank",
        lowered,
    )
    if bank is not None:
        tenths = round(float(bank.group(1)) * 10)
        if tenths > 0:
            add(
                Requirement(
                    kind=RequirementKind.BANK_FLOOR,
                    label=f"leave {tenths / 10:.1f}m in the bank",
                    amount=TenthsOfMillion(tenths),
                    priority=0,
                ),
                "requirements.bank_floor",
                0.9,
                bank.group(0),
            )
            parse.matched_spans.append(bank.span())

    return parse


def _resolve_team(text: str, teams: Mapping[str, int]) -> int | None:
    """Resolve a club as written to a team id, refusing to guess.

    Exact match first, then a unique prefix — the same discipline the match-event
    ingester uses. Two clubs sharing a prefix leave the phrase unresolved rather
    than picking one, because a squad silently built around the wrong club is a
    worse outcome than one that says it did not understand.
    """
    wanted = text.strip().lower()
    if not wanted:
        return None

    folded = {key.strip().lower(): value for key, value in teams.items()}
    if wanted in folded:
        return folded[wanted]

    candidates = {
        value for key, value in folded.items() if key.startswith(wanted) or wanted.startswith(key)
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    return None
