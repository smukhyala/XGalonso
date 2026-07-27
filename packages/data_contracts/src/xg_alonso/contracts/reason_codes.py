"""The single reason-code vocabulary.

Planning produced three competing vocabularies with three naming conventions.
This is the one. It is deliberately small: slice 1 emits seven codes, each of
which a human can verify against the data. A large vocabulary that nothing emits
is documentation cosplay.

**Grounding rule.** Every reason carries structured numeric evidence, and its
prose is rendered from a template whose placeholders must all resolve from that
evidence. An LLM may rewrite the rendered sentence for readability, but it never
receives free rein to state a cause or a statistic: there is no code path that
lets prose reference a number the evidence does not contain.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.identifiers import PlayerCode

__all__ = [
    "REASON_TEMPLATES",
    "Reason",
    "ReasonCode",
    "ReasonPolarity",
]


class ReasonCode(StrEnum):
    """Why a recommendation was made. Each maps to one evidence template."""

    FIXTURE_SWING_POSITIVE = "FIXTURE_SWING_POSITIVE"
    FIXTURE_SWING_NEGATIVE = "FIXTURE_SWING_NEGATIVE"
    EXPECTED_MINUTES_SECURE = "EXPECTED_MINUTES_SECURE"
    EXPECTED_MINUTES_DECLINE = "EXPECTED_MINUTES_DECLINE"
    UNDERLYING_STATS_IMPROVING = "UNDERLYING_STATS_IMPROVING"
    UNDERLYING_STATS_DECLINING = "UNDERLYING_STATS_DECLINING"
    AVAILABILITY_RISK_HIGH = "AVAILABILITY_RISK_HIGH"


class ReasonPolarity(StrEnum):
    """Whether a reason argues for acquiring or for removing a player."""

    SUPPORTS_IN = "supports_in"
    SUPPORTS_OUT = "supports_out"


REASON_TEMPLATES: Final[dict[ReasonCode, str]] = {
    ReasonCode.FIXTURE_SWING_POSITIVE: (
        "Fixtures ease over the next {horizon:.0f} gameweeks: mean opponent strength "
        "{opponent_strength:.2f} against a league average of {league_average:.2f}."
    ),
    ReasonCode.FIXTURE_SWING_NEGATIVE: (
        "Fixtures harden over the next {horizon:.0f} gameweeks: mean opponent strength "
        "{opponent_strength:.2f} against a league average of {league_average:.2f}."
    ),
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
    ReasonCode.UNDERLYING_STATS_IMPROVING: (
        "Stronger underlying numbers: {recent_xgi:.2f} projected goal involvements "
        "against {baseline_xgi:.2f} for the player leaving."
    ),
    ReasonCode.UNDERLYING_STATS_DECLINING: (
        "Weaker underlying numbers: {recent_xgi:.2f} projected goal involvements "
        "against {baseline_xgi:.2f} for the alternative."
    ),
    ReasonCode.AVAILABILITY_RISK_HIGH: (
        "Availability is in doubt: reported {chance_of_playing:.0%} chance of playing."
    ),
}
"""Prose templates. Every placeholder must resolve from a :class:`Reason`'s evidence."""


class Reason(BaseModel):
    """One grounded reason, with the evidence that justifies it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ReasonCode
    polarity: ReasonPolarity
    subject: PlayerCode = Field(description="The player this reason is about")
    evidence: dict[str, float] = Field(
        description="Numeric evidence. Must satisfy every placeholder in the template."
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
        """
        template = REASON_TEMPLATES[self.code]
        try:
            template.format(**self.evidence)
        except KeyError as exc:
            raise ValueError(
                f"{self.code} is missing evidence key {exc.args[0]!r}; "
                f"template requires it and prose may not invent it"
            ) from exc
        return self

    def render(self) -> str:
        """Render grounded prose. Safe by construction — validation guaranteed it."""
        return REASON_TEMPLATES[self.code].format(**self.evidence)
