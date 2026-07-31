"""FPL scoring rules, loaded from a pinned API snapshot rather than transcribed.

**Why this module refuses to hardcode point values.** A goalkeeper goal is worth
10 points in the live game, not the 6 that nearly every community model assumes.
That error survives review indefinitely because it looks right. The payload
publishes ``game_config.scoring``, so there is no reason to type these numbers
from memory — we read them, pin the snapshot, and fail loudly when they drift.

**What the payload does not give us.** FPL publishes point *values* but not the
*divisors and thresholds* that convert raw counts into those points: how many
saves earn a point, how many goals conceded cost one, and what counts as a
defensive contribution. Those live in :class:`ScoringThresholds`, are marked
``VERIFY``, and carry their provenance in the docstring. Keeping them separate
makes the boundary between "read from the vendor" and "asserted by us" visible
instead of blurred.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from xg_alonso.contracts.prediction import ComponentExpectations, PointsBreakdown, Position

__all__ = [
    "ScoringRules",
    "ScoringThresholds",
    "assemble_points",
]


class ScoringThresholds(BaseModel):
    """Constants FPL does not publish, asserted by us and marked for verification.

    Every field here is a liability. If FPL changes one of these, no drift check
    will catch it, because there is nothing in the payload to compare against —
    so each carries its source and must be re-checked against the official rules
    at the start of every season.

    ``VERIFY``: confirm all four against the official FPL rules page before the
    first gameweek of any new season.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    saves_per_point: int = Field(
        default=3,
        gt=0,
        description="VERIFY: a goalkeeper earns 1 point per 3 saves.",
    )
    goals_conceded_per_deduction: int = Field(
        default=2,
        gt=0,
        description="VERIFY: GK and DEF lose 1 point per 2 goals conceded.",
    )
    defensive_contribution_threshold_def: int = Field(
        default=10,
        gt=0,
        description=(
            "VERIFY: defenders need 10+ clearances, blocks, interceptions and "
            "tackles to earn the defensive-contribution points."
        ),
    )
    defensive_contribution_threshold_outfield: int = Field(
        default=12,
        gt=0,
        description=(
            "VERIFY: midfielders and forwards need 12+ defensive actions "
            "(the defender actions plus ball recoveries)."
        ),
    )
    captain_multiplier: int = Field(
        default=2,
        gt=1,
        description=(
            "VERIFY: FPL doubles the captain. game_config publishes only "
            "chips[].overrides.pick_multiplier, and only for the triple-captain "
            "chip, so the base multiplier is asserted here rather than read."
        ),
    )

    long_play_minutes: int = Field(
        default=60,
        gt=0,
        description="VERIFY: the minute count separating short_play from long_play.",
    )


class ScoringRules(BaseModel):
    """Point values for one season, read from a pinned ``bootstrap-static`` snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(description="Season identifier these rules apply to, e.g. '2026-27'")
    source_sha256: str = Field(
        min_length=64, max_length=64, description="Hash of the snapshot these were read from"
    )
    fetched_at: datetime

    goals_scored: dict[Position, int]
    clean_sheets: dict[Position, int]
    goals_conceded: dict[Position, int]
    defensive_contribution: dict[Position, int]

    assists: int
    saves: int
    bonus: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    short_play: int
    long_play: int

    thresholds: ScoringThresholds = Field(default_factory=ScoringThresholds)

    @classmethod
    def from_bootstrap(
        cls,
        payload: dict[str, Any],
        *,
        version: str,
        source_sha256: str,
        fetched_at: datetime,
    ) -> ScoringRules:
        """Parse scoring rules out of a ``bootstrap-static`` payload.

        Raises:
            KeyError: if the payload has no ``game_config.scoring`` block. That is
                a breaking upstream change, not something to paper over with
                defaults — silently falling back to hardcoded values is exactly
                the failure this module exists to prevent.
        """
        game_config = payload.get("game_config")
        if not isinstance(game_config, dict) or "scoring" not in game_config:
            raise KeyError(
                "bootstrap-static payload has no game_config.scoring block. "
                "Scoring values must never be assumed — fix the ingest rather "
                "than falling back to transcribed constants."
            )
        scoring: dict[str, Any] = game_config["scoring"]

        def by_position(key: str) -> dict[Position, int]:
            raw = scoring[key]
            if not isinstance(raw, dict):
                raise TypeError(f"expected a per-position mapping for {key!r}, got {type(raw)}")
            return {Position(k): int(v) for k, v in raw.items() if k in Position.__members__}

        return cls(
            version=version,
            source_sha256=source_sha256,
            fetched_at=fetched_at,
            goals_scored=by_position("goals_scored"),
            clean_sheets=by_position("clean_sheets"),
            goals_conceded=by_position("goals_conceded"),
            defensive_contribution=by_position("defensive_contribution"),
            assists=int(scoring["assists"]),
            saves=int(scoring["saves"]),
            bonus=int(scoring["bonus"]),
            own_goals=int(scoring["own_goals"]),
            penalties_saved=int(scoring["penalties_saved"]),
            penalties_missed=int(scoring["penalties_missed"]),
            yellow_cards=int(scoring["yellow_cards"]),
            red_cards=int(scoring["red_cards"]),
            short_play=int(scoring["short_play"]),
            long_play=int(scoring["long_play"]),
        )

    def defensive_contribution_threshold(self, position: Position) -> int:
        """Defensive actions required to earn the contribution points."""
        if position is Position.DEF:
            return self.thresholds.defensive_contribution_threshold_def
        return self.thresholds.defensive_contribution_threshold_outfield


#: Fields whose absence from a live payload means the drift check must fail.
DRIFT_CHECKED_FIELDS: Final[tuple[str, ...]] = (
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "short_play",
    "long_play",
    "defensive_contribution",
)


def assemble_points(
    components: ComponentExpectations,
    position: Position,
    rules: ScoringRules,
) -> PointsBreakdown:
    """Convert expected component counts into expected points.

    This is the D8 boundary: models predict components, and only this function
    knows what a component is worth. Swapping in another season's rules re-scores
    the same predictions without retraining anything.

    Because expectation is linear, each term is simply the expected count times
    its point value. Two terms are approximations worth naming:

    - **Goals conceded** deducts a point per *completed pair* conceded, so the
      true expectation involves the distribution of goals against, not just its
      mean. Dividing the mean by two is a first-order approximation that is
      exact only when the count is even. It is close enough to rank transfers
      and biases every defender identically, so it does not distort comparisons.
    - **Saves** has the same structure with a divisor of three.

    Both approximations become unnecessary once the component models emit full
    count distributions rather than means.
    """
    minutes = components.minutes
    thresholds = rules.thresholds

    # An appearance pays short_play; lasting 60 minutes pays long_play instead.
    p_short_only = max(0.0, minutes.p_appearance - minutes.p_60_plus)
    appearance = p_short_only * rules.short_play + minutes.p_60_plus * rules.long_play

    goals = components.goals * rules.goals_scored.get(position, 0)
    assists = components.assists * rules.assists
    clean_sheets = components.clean_sheet_probability * rules.clean_sheets.get(position, 0)

    conceded_rate = rules.goals_conceded.get(position, 0)
    goals_conceded = (
        components.goals_conceded / thresholds.goals_conceded_per_deduction
    ) * conceded_rate

    saves = (components.saves / thresholds.saves_per_point) * rules.saves

    cards = components.yellow_cards * rules.yellow_cards + components.red_cards * rules.red_cards
    own_goals = components.own_goals * rules.own_goals
    penalties = (
        components.penalties_saved * rules.penalties_saved
        + components.penalties_missed * rules.penalties_missed
    )
    defensive_contribution = (
        components.defensive_contribution_probability
        * rules.defensive_contribution.get(position, 0)
    )
    bonus = components.bonus * rules.bonus

    return PointsBreakdown(
        appearance=appearance,
        goals=goals,
        assists=assists,
        clean_sheets=clean_sheets,
        goals_conceded=goals_conceded,
        saves=saves,
        cards=cards,
        own_goals=own_goals,
        penalties=penalties,
        defensive_contribution=defensive_contribution,
        bonus=bonus,
    )
