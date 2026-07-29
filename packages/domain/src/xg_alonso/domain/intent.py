"""Turn what a manager typed into an objective, constraints and beliefs.

**Deterministic. No language model.** Every rule below is a pattern with a fixed
meaning, and every field it sets carries the substring that set it. That is not a
limitation to be apologised for — it is the correct design for this job, and the
brief asks for exactly it: deterministic parsing for explicit numeric and
categorical fields, an LLM only for genuinely ambiguous semantics.

There is no LLM here for two reasons. The smaller: this repository has no client
library and no key, so a model path would be dead code pretending to be a
feature. The larger: **natural language must never be able to bypass a hard FPL
rule.** A parser that emits a structured object which is then validated by
:class:`~xg_alonso.contracts.objective.ManagerConstraints` and the squad rules
cannot produce an illegal recommendation however it is prompted. That property is
worth more than handling an unusual phrasing.

What is *not* understood is reported rather than dropped. :attr:`unparsed` lists
the clauses no rule matched, so a request that was 80% understood says which 20%
was not — instead of silently answering a narrower question than the one asked.

**Nothing here executes.** The output is a :class:`CompiledIntent` the product
shows to the user for review and editing. Acting on a guess without showing it is
how a system eventually acts on a wrong one silently.

Pure, per the domain layer's contract.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TenthsOfMillion
from xg_alonso.contracts.objective import (
    BeliefEntity,
    BeliefProposition,
    CompiledIntent,
    FeatureDiscoveryRequest,
    FieldConfidence,
    ManagerConstraints,
    ObjectiveBundle,
    OwnershipPreference,
    ParseSource,
    PrimaryMetric,
    RiskPreference,
    SquadArea,
    UserBelief,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.domain.objectives import objective_preset

__all__ = [
    "FEATURE_ALIASES",
    "compile_intent",
]

#: Number words the horizon parser understands, so "next three gameweeks" works
#: as well as "next 3".
_NUMBER_WORDS: Final[dict[str, int]] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

#: Phrases a manager uses for a feature family, mapped to the catalogue columns
#: that actually implement them.
#:
#: Nobody types ``expected_goals_per90_5``. They type "xG". The mapping is
#: explicit and reviewable in one place rather than guessed by substring match,
#: because a required feature that resolves to the wrong column silently anchors
#: the entire search on something the user did not ask for.
FEATURE_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "xg": ("expected_goals_per90_5",),
    "expected goals": ("expected_goals_per90_5",),
    "xa": ("expected_assists_per90_5",),
    "expected assists": ("expected_assists_per90_5",),
    "xgi": ("expected_goal_involvements_per90_10",),
    "goal involvement": ("expected_goal_involvements_per90_10",),
    "minutes": ("minutes_mean_5",),
    "expected minutes": ("minutes_mean_5", "appearance_rate_10"),
    "rest": ("days_since_last_match",),
    "rest days": ("days_since_last_match",),
    "form": ("total_points_mean_5",),
    "fixture difficulty": ("opponent_conceded_xg_mean_5",),
    "fixtures": ("opponent_conceded_xg_mean_5",),
    "ownership": ("selected_mean_5",),
    "threat": ("threat_per90_5",),
    "creativity": ("creativity_per90_5",),
    "bonus": ("bps_per90_5",),
    "clean sheet": ("clean_sheets_mean_5",),
    "clean sheets": ("clean_sheets_mean_5",),
}

_POSITION_WORDS: Final[dict[str, Position]] = {
    "goalkeeper": Position.GKP,
    "goalkeepers": Position.GKP,
    "keeper": Position.GKP,
    "keepers": Position.GKP,
    "defender": Position.DEF,
    "defenders": Position.DEF,
    "defence": Position.DEF,
    "defense": Position.DEF,
    "midfielder": Position.MID,
    "midfielders": Position.MID,
    "midfield": Position.MID,
    "forward": Position.FWD,
    "forwards": Position.FWD,
    "striker": Position.FWD,
    "strikers": Position.FWD,
    "attack": Position.FWD,
    "attackers": Position.FWD,
}

_AREA_WORDS: Final[dict[str, SquadArea]] = {
    "defence": SquadArea.DEFENCE,
    "defense": SquadArea.DEFENCE,
    "backline": SquadArea.DEFENCE,
    "back line": SquadArea.DEFENCE,
    "midfield": SquadArea.MIDFIELD,
    "attack": SquadArea.ATTACK,
    "front line": SquadArea.ATTACK,
    "frontline": SquadArea.ATTACK,
    "bench": SquadArea.BENCH,
}


def _number_at(text: str) -> int | None:
    """A digit or a number word, whichever the text used."""
    stripped = text.strip().lower()
    if stripped.isdigit():
        return int(stripped)
    return _NUMBER_WORDS.get(stripped)


def compile_intent(
    text: str,
    *,
    players: Mapping[str, int] | None = None,
    base_preset: str = "expected_points",
    current_squad: Sequence[int] = (),
    next_gameweek: int | None = None,
) -> CompiledIntent:
    """Parse a manager's request into a reviewable, editable bundle.

    Args:
        text: What the manager typed.
        players: Display name to ``PlayerCode``, for resolving "keep Haaland".
            Matching is case-insensitive on whole words, so "Sonny" will not
            silently match "Son" — a wrong player lock is worse than an unmatched
            one, because it is invisible.
        base_preset: Objective to start from before the text modifies it.
        current_squad: Player codes the manager owns. Needed to turn "don't
            change my defence" into concrete locks rather than a position rule
            that would also block *buying* a defender.
        next_gameweek: Used to scope beliefs stated without a gameweek.

    Returns:
        The bundle, per-field confidence with the evidence that produced it, and
        every clause no rule matched.
    """
    lowered = text.lower()
    confidences: list[FieldConfidence] = []
    matched_spans: list[tuple[int, int]] = []

    def note(field: str, confidence: float, source: ParseSource, evidence: str = "") -> None:
        confidences.append(
            FieldConfidence(
                field=field, confidence=confidence, source=source, evidence=evidence.strip()
            )
        )

    def claim(match: re.Match[str] | None) -> str:
        if match is None:
            return ""
        matched_spans.append(match.span())
        return match.group(0)

    objective = objective_preset(base_preset)
    note("objective.preset", 1.0, ParseSource.PRESET, base_preset)

    updates: dict[str, object] = {}

    # --- rank chasing ------------------------------------------------------
    deficit = re.search(
        r"(\d+)\s*(?:points?|pts?)\s*(?:behind|back|off|adrift)", lowered
    ) or re.search(r"(?:behind|back|down)\s*(?:by\s*)?(\d+)\s*(?:points?|pts?)", lowered)
    chasing = re.search(r"\b(mini[- ]?league|chase|catch up|catch-up|make up ground)\b", lowered)
    if deficit is not None or chasing is not None:
        evidence = claim(deficit) or claim(chasing)
        updates["primary_metric"] = PrimaryMetric.EXPECTED_RANK_GAIN
        updates["ownership_preference"] = OwnershipPreference.DIFFERENTIAL
        note("objective.primary_metric", 0.85, ParseSource.INFERRED, evidence)
        note("objective.ownership_preference", 0.7, ParseSource.INFERRED, evidence)

    protecting = re.search(
        r"\b(protect(?:ing)?\s+(?:my\s+)?rank|hold\s+(?:my\s+)?rank|don'?t\s+(?:want\s+to\s+)?"
        r"(?:drop|fall)|play\s+it\s+safe|safe\s+picks?)\b",
        lowered,
    )
    if protecting is not None:
        evidence = claim(protecting)
        updates["primary_metric"] = PrimaryMetric.DOWNSIDE_PROTECTION
        updates["ownership_preference"] = OwnershipPreference.TEMPLATE
        updates["risk_preference"] = RiskPreference.CONSERVATIVE
        note("objective.primary_metric", 0.8, ParseSource.INFERRED, evidence)

    # --- risk --------------------------------------------------------------
    aggressive = re.search(r"\b(aggressive|risky|high[- ]risk|punt|gamble|bold)\b", lowered)
    conservative = re.search(
        r"\b(conservative|safe|low[- ]risk|cautious|steady|reliable)\b", lowered
    )
    if aggressive is not None:
        updates["risk_preference"] = RiskPreference.AGGRESSIVE
        updates.setdefault("uncertainty_penalty", 0.1)
        note("objective.risk_preference", 0.95, ParseSource.EXPLICIT, claim(aggressive))
    elif conservative is not None:
        updates["risk_preference"] = RiskPreference.CONSERVATIVE
        updates.setdefault("uncertainty_penalty", 0.6)
        note("objective.risk_preference", 0.95, ParseSource.EXPLICIT, claim(conservative))

    # --- ownership ---------------------------------------------------------
    differential = re.search(
        r"\b(differential|low[- ]owned|under[- ]owned|unowned|off[- ]template)\b", lowered
    )
    template = re.search(r"\b(template|highly owned|popular pick|essential)\b", lowered)
    if differential is not None:
        updates["ownership_preference"] = OwnershipPreference.DIFFERENTIAL
        note("objective.ownership_preference", 0.95, ParseSource.EXPLICIT, claim(differential))
    elif template is not None:
        updates["ownership_preference"] = OwnershipPreference.TEMPLATE
        note("objective.ownership_preference", 0.95, ParseSource.EXPLICIT, claim(template))

    # --- horizon -----------------------------------------------------------
    horizon = re.search(
        r"(?:next|over|across|for)\s+(?:the\s+)?(\w+)\s+(?:game ?weeks?|gws?|weeks?)", lowered
    ) or re.search(r"(\w+)[- ](?:game ?week|gw|week)\s+(?:horizon|plan|strategy|view)", lowered)
    if horizon is not None:
        weeks = _number_at(horizon.group(1))
        if weeks is not None and 1 <= weeks <= 10:
            updates["planning_horizon"] = weeks
            note("objective.planning_horizon", 0.95, ParseSource.EXPLICIT, claim(horizon))
    elif re.search(r"\b(this|next)\s+(?:game ?week|gw)\b", lowered):
        updates["planning_horizon"] = 1
        note("objective.planning_horizon", 0.9, ParseSource.EXPLICIT, "next gameweek")

    if re.search(r"\bwild ?card\b", lowered):
        updates.setdefault("planning_horizon", 6)
        note("objective.planning_horizon", 0.7, ParseSource.INFERRED, "wildcard")

    # --- team value --------------------------------------------------------
    value_match = re.search(
        r"\b(team value|squad value|price ris\w+|make money|value growth)\b", lowered
    )
    if value_match is not None:
        updates["team_value_weight"] = 1.0
        note("objective.team_value_weight", 0.85, ParseSource.INFERRED, claim(value_match))

    captain_match = re.search(r"\b(captain\w*|armband|triple captain)\b", lowered)
    if captain_match is not None:
        updates["captaincy_weight"] = max(1.4, float(objective.captaincy_weight))
        note("objective.captaincy_weight", 0.7, ParseSource.INFERRED, claim(captain_match))

    objective = objective.model_copy(update=updates) if updates else objective

    # --- constraints -------------------------------------------------------
    constraint_updates: dict[str, object] = {}

    # "a points hit", "any hits", "take a -4" are all the same instruction. The
    # plural was missed by an earlier version, and because `max_points_hit`
    # defaults to zero the miss was *invisible* — the parse failed and the
    # constraint came out right anyway. Only the `unparsed` report caught it,
    # which is the argument for reporting unmatched clauses at all.
    hits = re.search(
        r"\b(?:no|don'?t|do not|without|avoid|never)\s+(?:tak(?:e|ing)\s+)?(?:a\s+|any\s+)?"
        r"(?:points?\s+)?hits?\b",
        lowered,
    )
    if hits is not None:
        constraint_updates["max_points_hit"] = 0
        note("constraints.max_points_hit", 0.95, ParseSource.EXPLICIT, claim(hits))
    else:
        hit_budget = re.search(
            r"(?:up to|at most|max(?:imum)?)\s*(-?\d+)\s*(?:points?\s*)?hits?", lowered
        )
        willing = re.search(
            r"\b(?:willing|happy|prepared|fine|ok(?:ay)?)\s+to\s+take\s+(?:a\s+)?hits?\b", lowered
        )
        if hit_budget is not None:
            constraint_updates["max_points_hit"] = abs(int(hit_budget.group(1)))
            note("constraints.max_points_hit", 0.9, ParseSource.EXPLICIT, claim(hit_budget))
        elif willing is not None:
            # One hit, not unlimited. "Happy to take a hit" is permission for a
            # single -4, and reading it as an open budget would let the optimizer
            # spend four transfers on the strength of one agreeable sentence.
            constraint_updates["max_points_hit"] = 4
            note("constraints.max_points_hit", 0.7, ParseSource.INFERRED, claim(willing))

    bank = re.search(
        r"(?:keep|leave|retain|hold)\s*(?:at least\s*)?£?\s*([\d.]+)\s*m?\s*(?:in the )?bank",
        lowered,
    )
    if bank is None:
        bank = re.search(r"£\s*([\d.]+)\s*m?\s*(?:left\s*)?in the bank", lowered)
    if bank is not None:
        try:
            constraint_updates["minimum_bank"] = TenthsOfMillion(round(float(bank.group(1)) * 10))
            note("constraints.minimum_bank", 0.9, ParseSource.EXPLICIT, claim(bank))
        except ValueError:  # pragma: no cover - regex guarantees a number
            pass

    transfers = re.search(r"(?:at most|max(?:imum)?|only)\s*(\w+)\s*transfers?", lowered)
    if transfers is not None:
        count = _number_at(transfers.group(1))
        if count is not None:
            constraint_updates["max_transfers"] = count
            note("constraints.max_transfers", 0.9, ParseSource.EXPLICIT, claim(transfers))

    # Locked players. Whole-word, case-insensitive matching only.
    locked: list[PlayerCode] = []
    excluded: list[PlayerCode] = []
    if players:
        for name, code in players.items():
            if not name:
                continue
            pattern = re.compile(rf"\b{re.escape(name.lower())}\b")
            found = pattern.search(lowered)
            if found is None:
                continue
            window = lowered[max(0, found.start() - 40) : found.start()]
            if re.search(
                r"\b(keep|lock|hold|keeping|retain|stick with|must have|starting)\b", window
            ):
                locked.append(PlayerCode(code))
                note(f"constraints.locked_players[{name}]", 0.9, ParseSource.EXPLICIT, name)
                matched_spans.append(found.span())
            elif re.search(r"\b(sell|drop|remove|get rid of|ditch|avoid|exclude|no)\b", window):
                excluded.append(PlayerCode(code))
                note(f"constraints.excluded_players[{name}]", 0.85, ParseSource.EXPLICIT, name)
                matched_spans.append(found.span())
    if locked:
        constraint_updates["locked_players"] = tuple(dict.fromkeys(locked))
    if excluded:
        constraint_updates["excluded_players"] = tuple(dict.fromkeys(excluded))

    # Protected areas and positions. "Do not change my defence" is a statement
    # about the players already owned, not a ban on ever buying a defender.
    protected: list[SquadArea] = []
    locked_positions: list[Position] = []
    for phrase, area in _AREA_WORDS.items():
        pattern = re.compile(
            rf"\b(?:don'?t|do not|no|not|never|keep|leave|retain|preserve|protect)\b[^.;,]{{0,40}}"
            rf"\b(?:chang\w+|touch\w*|transfer\w*|sell\w*|move\w*|alter\w*|keep|as is)?[^.;,]{{0,20}}"
            rf"\b{re.escape(phrase)}\b"
        )
        found = pattern.search(lowered)
        if found is not None:
            protected.append(area)
            matched_spans.append(found.span())
            note(
                f"constraints.protected_squad_areas[{area.value}]",
                0.8,
                ParseSource.INFERRED,
                found.group(0),
            )

    for phrase, position in _POSITION_WORDS.items():
        pattern = re.compile(
            rf"\b(?:don'?t|do not|no|never)\s+(?:transfer|sell|move|change|touch)\s+"
            rf"(?:any\s+|my\s+|the\s+)?{re.escape(phrase)}\b"
        )
        found = pattern.search(lowered)
        if found is not None:
            locked_positions.append(position)
            matched_spans.append(found.span())
            note(
                f"constraints.locked_positions[{position.value}]",
                0.9,
                ParseSource.EXPLICIT,
                found.group(0),
            )

    if protected:
        constraint_updates["protected_squad_areas"] = tuple(dict.fromkeys(protected))
    if locked_positions:
        constraint_updates["locked_positions"] = tuple(dict.fromkeys(locked_positions))

    # --- required features -------------------------------------------------
    required_features: list[str] = []
    complement_targets: list[str] = []
    for alias, columns in FEATURE_ALIASES.items():
        pattern = re.compile(
            rf"\b(?:keep|require\w*|must (?:keep|have|stay|remain)|retain|need)\b[^.;]{{0,50}}"
            rf"\b{re.escape(alias)}\b"
        )
        found = pattern.search(lowered)
        if found is None:
            pattern = re.compile(
                rf"\b{re.escape(alias)}\b[^.;]{{0,40}}\b(?:must (?:stay|remain)|required|stays? in)\b"
            )
            found = pattern.search(lowered)
        if found is not None:
            required_features.extend(columns)
            matched_spans.append(found.span())
            note(
                f"discovery.required_features[{alias}]", 0.85, ParseSource.EXPLICIT, found.group(0)
            )

    complement = re.search(
        r"\b(complement\w*|adds? to|beyond|on top of|in addition to|alongside)\b", lowered
    )
    if complement is not None:
        claim(complement)
        complement_targets = list(required_features)
        note("discovery.complement_targets", 0.8, ParseSource.INFERRED, complement.group(0))

    emphasis: list[str] = []
    for word, tag in (
        ("upside", "upside"),
        ("ceiling", "upside"),
        ("haul", "upside"),
        ("floor", "floor"),
        ("consisten", "floor"),
        ("differential", "differential"),
        ("minutes", "minutes"),
        ("rotation", "minutes"),
        ("fixture", "fixtures"),
    ):
        if word in lowered:
            emphasis.append(tag)

    # The objective implies emphasis the manager did not have to spell out. A
    # request for "aggressive picks" is asking for ceiling whether or not the
    # word "upside" appears, and hypothesis generation should be biased toward
    # mechanisms that produce it.
    if objective.risk_preference is RiskPreference.AGGRESSIVE:
        emphasis.append("upside")
    elif objective.risk_preference is RiskPreference.CONSERVATIVE:
        emphasis.extend(("floor", "minutes"))
    if objective.ownership_preference is OwnershipPreference.DIFFERENTIAL:
        emphasis.append("differential")

    if required_features:
        constraint_updates["required_features"] = tuple(dict.fromkeys(required_features))

    constraints = ManagerConstraints(**constraint_updates)  # type: ignore[arg-type]

    # --- beliefs -----------------------------------------------------------
    beliefs: list[UserBelief] = []
    if players:
        for name, code in players.items():
            if not name:
                continue
            for belief_template, proposition, confidence in _BELIEF_PATTERNS:
                found = re.search(belief_template.format(name=re.escape(name.lower())), lowered)
                if found is None:
                    continue
                strength = confidence
                if re.search(r"\b(strongly|really|definitely|certain|convinced)\b", lowered):
                    strength = min(0.95, strength + 0.1)
                elif re.search(r"\b(might|maybe|could|possibly|think)\b", found.group(0)):
                    strength = max(0.4, strength - 0.15)
                beliefs.append(
                    UserBelief(
                        entity_type=BeliefEntity.PLAYER,
                        entity_id=code,
                        proposition=proposition,
                        confidence=strength,
                        affected_gameweeks=(
                            (GameweekId(next_gameweek),) if next_gameweek is not None else ()
                        ),
                        rationale=found.group(0).strip(),
                    )
                )
                matched_spans.append(found.span())
                note(f"beliefs[{name}]", strength, ParseSource.INFERRED, found.group(0))
                break

    discovery = FeatureDiscoveryRequest(
        required_features=tuple(dict.fromkeys(required_features)),
        complement_targets=tuple(dict.fromkeys(complement_targets)),
        emphasis=tuple(dict.fromkeys(emphasis)),
    )

    bundle = ObjectiveBundle(
        objective=objective,
        constraints=constraints,
        beliefs=tuple(beliefs),
        discovery=discovery,
    )

    return CompiledIntent(
        bundle=bundle,
        confidences=tuple(confidences),
        unparsed=_unparsed(text, matched_spans),
        raw_text=text,
    )


#: Belief phrasings, most specific first. The order matters: "will not score"
#: must be tested before "will score", or the negation is lost and the belief is
#: applied with the wrong sign — a failure that would read as the system
#: disagreeing with the user rather than misreading them.
_BELIEF_PATTERNS: Final[tuple[tuple[str, BeliefProposition, float], ...]] = (
    (
        r"{name}\b[^.;]{{0,30}}\b(?:won'?t|will not|isn'?t going to|not going to)\s+"
        r"(?:score|return|haul|do anything)",
        BeliefProposition.UNDERPERFORM_MODEL,
        0.7,
    ),
    (
        r"{name}\b[^.;]{{0,30}}\b(?:won'?t|will not|might not|doubt.{{0,10}})\s*start",
        BeliefProposition.WILL_NOT_START,
        0.7,
    ),
    (
        r"{name}\b[^.;]{{0,30}}\b(?:will|going to|is going to|to)\s+(?:score|haul|return|deliver)",
        BeliefProposition.WILL_RETURN,
        0.75,
    ),
    (
        r"(?:think|believe|reckon|expect|feel)\b[^.;]{{0,30}}{name}\b[^.;]{{0,40}}"
        r"\b(?:score|haul|return|big|explode|deliver)",
        BeliefProposition.WILL_RETURN,
        0.7,
    ),
    (
        r"{name}\b[^.;]{{0,30}}\b(?:will start|is starting|nailed|guaranteed to start)",
        BeliefProposition.WILL_START,
        0.8,
    ),
    (
        r"(?:think|believe|reckon|expect)\b[^.;]{{0,40}}{name}\b[^.;]{{0,40}}"
        r"\b(?:outperform|beat|exceed|better than)",
        BeliefProposition.OUTPERFORM_MODEL,
        0.7,
    ),
)


def _unparsed(text: str, spans: Sequence[tuple[int, int]]) -> tuple[str, ...]:
    """Clauses no rule matched, so a partial understanding says which part.

    Split on sentence and clause boundaries, then drop anything that overlaps a
    matched span. Short fragments are ignored — "and", "so" — because reporting
    them as misunderstood would bury the real gaps in noise.
    """
    if not text.strip():
        return ()
    covered = sorted(spans)
    clauses: list[str] = []
    position = 0
    for match in re.finditer(r"[.;,]|\band\b|\bbut\b", text.lower()):
        clause = text[position : match.start()]
        if clause.strip():
            clauses.append(clause)
        position = match.end()
    tail = text[position:]
    if tail.strip():
        clauses.append(tail)

    out: list[str] = []
    offset = 0
    for clause in clauses:
        start = text.index(clause, offset)
        end = start + len(clause)
        offset = end
        if len(clause.strip()) < 12:
            continue
        overlaps = any(not (end <= low or start >= high) for low, high in covered)
        if not overlaps:
            out.append(clause.strip())
    return tuple(out)
