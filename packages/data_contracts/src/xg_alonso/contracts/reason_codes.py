"""The single reason-code vocabulary.

Planning produced three competing vocabularies with three naming conventions.
This is the one.

**Grounding rule.** Every reason carries structured numeric evidence, and its
prose is rendered from a template whose placeholders must all resolve from that
evidence. An LLM may rewrite the rendered sentence for readability, but it never
receives free rein to state a cause or a statistic: there is no code path that
lets prose reference a number the evidence does not contain.

**Why the vocabulary grew.** The first version declared seven codes and emitted
three. The recommendation screen therefore justified a transfer with two
unattributed minutes sentences and one derived aggregate, and named no feature
at all — it cited xG nowhere despite xG being in the model. The codes below draw
on the explanatory panel directly, so a reason says *which* statistic moved the
decision, what its value was, and where that value sits among comparable
players.

**Numbers live in evidence; labels live in context.** A template slot that is a
quantity must resolve from ``evidence: dict[str, float]``, which is validated.
Slots that are purely descriptive — a position name, for instance — resolve from
``context: dict[str, str]``. Splitting them keeps the guarantee sharp: no
quantitative claim can enter prose without passing through the validated numeric
channel, and widening ``evidence`` to accept strings would have quietly
dissolved that.

Player names are absent from both. A reason knows its ``subject`` as a code;
attaching a name is the renderer's job, because the contract layer has no
business holding presentation data and because a name map belongs to the surface
that has one.
"""

from __future__ import annotations

from enum import StrEnum
from string import Formatter
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.identifiers import PlayerCode

__all__ = [
    "REASON_TEMPLATES",
    "Reason",
    "ReasonCode",
    "ReasonPolarity",
    "template_placeholders",
]


class ReasonCode(StrEnum):
    """Why a recommendation was made. Each maps to one evidence template."""

    # --- fixtures ---
    FIXTURE_SWING_POSITIVE = "FIXTURE_SWING_POSITIVE"
    FIXTURE_SWING_NEGATIVE = "FIXTURE_SWING_NEGATIVE"
    HOME_FIXTURE = "HOME_FIXTURE"

    # --- minutes and availability ---
    EXPECTED_MINUTES_SECURE = "EXPECTED_MINUTES_SECURE"
    EXPECTED_MINUTES_DECLINE = "EXPECTED_MINUTES_DECLINE"
    AVAILABILITY_RISK_HIGH = "AVAILABILITY_RISK_HIGH"

    # --- underlying attacking numbers ---
    UNDERLYING_STATS_IMPROVING = "UNDERLYING_STATS_IMPROVING"
    UNDERLYING_STATS_DECLINING = "UNDERLYING_STATS_DECLINING"
    XG_RATE_HIGHER = "XG_RATE_HIGHER"
    XG_RATE_LOWER = "XG_RATE_LOWER"
    XA_RATE_HIGHER = "XA_RATE_HIGHER"
    XA_RATE_LOWER = "XA_RATE_LOWER"
    THREAT_HIGHER = "THREAT_HIGHER"

    # --- return shape ---
    CEILING_HIGHER = "CEILING_HIGHER"
    VOLATILITY_LOWER = "VOLATILITY_LOWER"
    BONUS_MAGNET = "BONUS_MAGNET"
    PRICE_EFFICIENCY = "PRICE_EFFICIENCY"
    POINTS_BREAKDOWN = "POINTS_BREAKDOWN"

    # --- information the API cannot see ---
    FORM_SIGNAL_POSITIVE = "FORM_SIGNAL_POSITIVE"
    FORM_SIGNAL_NEGATIVE = "FORM_SIGNAL_NEGATIVE"

    # --- why the choice set was what it was ---
    CONSTRAINT_HELD = "CONSTRAINT_HELD"
    """The manager instructed that this player be kept.

    Distinct from every other code here, and the distinction matters: the rest
    say why the *model* did not want a move. This says the model was never asked.
    Collapsing the two would let a constraint read as a judgement, and a user
    would have no way to tell which of their own instructions was costing them.
    """
    POSITION_LOCKED = "POSITION_LOCKED"
    BUDGET_LOCKED = "BUDGET_LOCKED"
    NO_UPGRADE_AVAILABLE = "NO_UPGRADE_AVAILABLE"


class ReasonPolarity(StrEnum):
    """Whether a reason argues for acquiring or for removing a player."""

    SUPPORTS_IN = "supports_in"
    SUPPORTS_OUT = "supports_out"
    CONTEXT = "context"
    """Neither. Explains the choice set rather than arguing about a player."""


REASON_TEMPLATES: Final[dict[ReasonCode, str]] = {
    ReasonCode.FIXTURE_SWING_POSITIVE: (
        "Kinder fixture: the opponent has conceded {opponent_xg:.2f} expected goals a game "
        "over their last five, against a league average of {league_average:.2f}."
    ),
    ReasonCode.FIXTURE_SWING_NEGATIVE: (
        "Harder fixture: the opponent has conceded {opponent_xg:.2f} expected goals a game "
        "over their last five, against a league average of {league_average:.2f}."
    ),
    ReasonCode.HOME_FIXTURE: "Playing at home.",
    # Expected minutes is a continuous quantity, so it is rendered with a decimal.
    # Integer rounding implied a discreteness the estimate does not have, and it
    # also produced "around 1 minutes expected" whenever the value rounded to one.
    ReasonCode.EXPECTED_MINUTES_SECURE: (
        "Minutes look secure: {p_start:.0%} chance of starting, around "
        "{expected_minutes:.1f} minutes expected."
    ),
    ReasonCode.EXPECTED_MINUTES_DECLINE: (
        "Minutes are a concern: {p_start:.0%} chance of starting, around "
        "{expected_minutes:.1f} minutes expected."
    ),
    ReasonCode.AVAILABILITY_RISK_HIGH: (
        "Availability is in doubt: reported {chance_of_playing:.0%} chance of playing."
    ),
    ReasonCode.UNDERLYING_STATS_IMPROVING: (
        "Higher projected returns: {recent_xgi:.2f} expected goals and assists this "
        "gameweek against {baseline_xgi:.2f}."
    ),
    ReasonCode.UNDERLYING_STATS_DECLINING: (
        "Lower projected returns: {recent_xgi:.2f} expected goals and assists this "
        "gameweek against {baseline_xgi:.2f}."
    ),
    ReasonCode.XG_RATE_HIGHER: (
        "Better shooting numbers: {value:.2f} expected goals per 90 over the last five "
        "appearances against {other:.2f} — {percentile:.0%} among {position}s."
    ),
    ReasonCode.XG_RATE_LOWER: (
        "Weaker shooting numbers: {value:.2f} expected goals per 90 over the last five "
        "appearances against {other:.2f} — {percentile:.0%} among {position}s."
    ),
    ReasonCode.XA_RATE_HIGHER: (
        "Better creative numbers: {value:.2f} expected assists per 90 against "
        "{other:.2f} — {percentile:.0%} among {position}s."
    ),
    ReasonCode.XA_RATE_LOWER: (
        "Weaker creative numbers: {value:.2f} expected assists per 90 against "
        "{other:.2f} — {percentile:.0%} among {position}s."
    ),
    ReasonCode.THREAT_HIGHER: (
        "More dangerous: threat of {value:.0f} per 90 against {other:.0f} — "
        "{percentile:.0%} among {position}s."
    ),
    ReasonCode.CEILING_HIGHER: (
        "Bigger ceiling: best return in the last five was {value:.0f} points against {other:.0f}."
    ),
    ReasonCode.VOLATILITY_LOWER: (
        "Steadier returns: points vary by {value:.2f} week to week against {other:.2f}."
    ),
    ReasonCode.BONUS_MAGNET: (
        "Bonus-point profile: {value:.1f} BPS per 90 — {percentile:.0%} among {position}s."
    ),
    ReasonCode.PRICE_EFFICIENCY: (
        "Better value: {value:.2f} projected points per million against {other:.2f}."
    ),
    ReasonCode.POINTS_BREAKDOWN: (
        "{total:.2f} projected points = {appearance:.2f} for appearing "
        "+ {goals:.2f} goals + {assists:.2f} assists + {clean_sheets:.2f} clean sheet "
        "+ {bonus:.2f} bonus."
    ),
    ReasonCode.FORM_SIGNAL_POSITIVE: (
        "Recent form outside the data: {summary} Projection raised {shift:.0%}. Source: {source}"
    ),
    ReasonCode.FORM_SIGNAL_NEGATIVE: (
        "Recent form outside the data: {summary} Projection cut {shift:.0%}. Source: {source}"
    ),
    ReasonCode.CONSTRAINT_HELD: (
        "Held at your instruction, so no move was considered for him. "
        "The opportunity cost of keeping him is reported separately."
    ),
    ReasonCode.POSITION_LOCKED: (
        "A transfer is like-for-like, so only {candidate_count:.0f} {position}s were "
        "legal replacements — players in other positions were never in contention."
    ),
    ReasonCode.BUDGET_LOCKED: (
        "Budget limits the choice: {budget:.1f}m available, and the cheapest player "
        "projected to beat him costs {shortfall:.1f}m more than that."
    ),
    ReasonCode.NO_UPGRADE_AVAILABLE: (
        "No legal replacement gained enough to be worth a transfer: the best available "
        "move was {best_gain:+.2f} points, below the {threshold:.2f} bar."
    ),
}
"""Prose templates. Every placeholder must resolve from a :class:`Reason`'s evidence or context."""


def template_placeholders(template: str) -> frozenset[str]:
    """Every named placeholder in a template.

    Parsed with the same machinery that renders it, rather than pattern-matched,
    so format specs (``{value:.2f}``) and escaped braces are handled identically
    in validation and in display.
    """
    return frozenset(name for _, name, _, _ in Formatter().parse(template) if name)


class Reason(BaseModel):
    """One grounded reason, with the evidence that justifies it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ReasonCode
    polarity: ReasonPolarity
    subject: PlayerCode = Field(description="The player this reason is about")
    evidence: dict[str, float] = Field(
        default_factory=dict,
        description="Numeric evidence. Must satisfy every numeric placeholder in the template.",
    )
    context: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Non-numeric template slots, such as a position name. Deliberately "
            "separate from evidence so no quantity can bypass numeric validation."
        ),
    )
    weight: float = Field(
        ge=0.0,
        description="Contribution to the decision, in expected points. Used for ranking.",
    )

    @model_validator(mode="after")
    def _evidence_satisfies_template(self) -> Reason:
        """Reject a reason whose evidence cannot fill its own template.

        This is where the no-fabrication guarantee is enforced. A reason that
        cannot be rendered from its evidence never gets constructed, so no
        downstream renderer — LLM or otherwise — is ever handed a gap to fill.

        Unused keys are rejected as well. Evidence a template never renders is
        evidence nobody reads, and in practice it is the signature of a builder
        that was updated while its template was not — which is exactly how a
        reason drifts away from the arithmetic it claims to explain.
        """
        template = REASON_TEMPLATES[self.code]
        required = template_placeholders(template)
        supplied = frozenset(self.evidence) | frozenset(self.context)

        overlap = frozenset(self.evidence) & frozenset(self.context)
        if overlap:
            raise ValueError(
                f"{self.code} defines {sorted(overlap)} in both evidence and context; "
                "a slot filled from two places has no single source of truth"
            )

        missing = required - supplied
        if missing:
            raise ValueError(
                f"{self.code} is missing evidence for {sorted(missing)}; "
                f"the template requires it and prose may not invent it"
            )

        unused = supplied - required
        if unused:
            raise ValueError(
                f"{self.code} carries {sorted(unused)}, which its template never renders; "
                "evidence nobody reads is evidence nobody checks"
            )

        # Prove it renders now rather than at display time, when the failure
        # would surface in front of a user.
        template.format(**self.evidence, **self.context)
        return self

    def render(self) -> str:
        """Render grounded prose. Safe by construction — validation guaranteed it."""
        return REASON_TEMPLATES[self.code].format(**self.evidence, **self.context)
