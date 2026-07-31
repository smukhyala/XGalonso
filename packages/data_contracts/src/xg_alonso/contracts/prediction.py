"""The single prediction output schema.

During planning, four independent designs produced four incompatible prediction
shapes. This module is the one definition; every producer and consumer uses it.

The shape follows decision D8: models predict **components** (minutes, goals,
assists, clean sheets, saves, cards, defensive contributions), and points are
*assembled* from those components through versioned scoring rules held in
``xg_alonso.domain``.

That indirection is deliberate. FPL added defensive-contribution points in
2025/26, so a model trained directly on historical point totals learns a scoring
system that no longer exists. Component labels are rule-version independent: the
whole backfill stays usable, and a rules change becomes a config edit rather
than a full relabel.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.evidence import FeatureEvidence
from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
from xg_alonso.contracts.provenance import PredictionProvenance

__all__ = [
    "MINUTES_STATES",
    "ComponentExpectations",
    "MinutesPrediction",
    "MinutesState",
    "PlayerPrediction",
    "PointsBreakdown",
    "Position",
]


class Position(StrEnum):
    """FPL positions, matching `element_types.singular_name_short`."""

    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class MinutesState(StrEnum):
    """How much of a match a player was on the pitch for.

    Three states, because FPL pays appearance points at exactly two thresholds
    and because almost every other component is driven by which of them applies:
    a player who did not feature scores nothing, a substitute can score but
    cannot earn a clean sheet, and only a 60-minute player can.

    This is the latent state the points distribution is conditioned on. Once you
    condition on it, the dependence that minutes induce *between* components —
    goals and assists and clean sheets all rising together because the player
    stayed on — disappears, and the components can be composed without
    estimating a covariance matrix.

    ``NONE`` is roughly 60% of all player-gameweeks, which is why a points
    distribution without an explicit zero atom cannot be right.
    """

    NONE = "none"
    """Did not play a minute."""

    SHORT = "short"
    """Featured, but for less than the long-play threshold."""

    LONG = "long"
    """Reached the long-play threshold — the clean-sheet gate."""

    @classmethod
    def of(cls, minutes: int, *, long_play_minutes: int) -> MinutesState:
        """Classify realised minutes.

        Args:
            long_play_minutes: The threshold, supplied by the caller rather than
                assumed. ``contracts`` may not import ``domain``, and this is a
                rule FPL does not publish — it lives in ``ScoringThresholds``
                marked ``VERIFY``, and passing it keeps this function honest
                about not knowing it.
        """
        if minutes >= long_play_minutes:
            return cls.LONG
        if minutes > 0:
            return cls.SHORT
        return cls.NONE

    @property
    def class_index(self) -> int:
        """Position in the canonical ordering, for use as a class label.

        Not named ``index``. ``StrEnum`` inherits ``str``, so ``index`` is
        already a method that finds a substring — shadowing it with a property
        would silently break ``MinutesState.LONG.index("o")`` and every other
        string operation, and mypy catches the override as a type error.
        """
        return _STATE_ORDER.index(self)


#: The canonical state ordering. Fixed once, because it is the column order of
#: every ``predict_proba`` output and of every dispersion table keyed by state.
_STATE_ORDER: tuple[MinutesState, ...] = (
    MinutesState.NONE,
    MinutesState.SHORT,
    MinutesState.LONG,
)

MINUTES_STATES: tuple[MinutesState, ...] = _STATE_ORDER


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MinutesPrediction(_Frozen):
    """Expected playing time — the input every other component is conditioned on.

    Minutes dominate FPL scoring: a brilliant forward who does not start is worth
    less than a mediocre one who plays 90. Every component expectation below is
    an unconditional expectation that already accounts for these.

    Appearance and starting are tracked separately because they pay differently.
    A substitute still earns the appearance point, and a substitute introduced
    early can reach 60 minutes — so ``p_60_plus`` is bounded by ``p_appearance``,
    not by ``p_start``.
    """

    p_appearance: float = Field(
        ge=0.0, le=1.0, description="Probability of playing at least one minute"
    )
    p_start: float = Field(ge=0.0, le=1.0, description="Probability of starting")
    expected_minutes: float = Field(ge=0.0, le=120.0)
    p_60_plus: float = Field(ge=0.0, le=1.0, description="Probability of 60+ minutes")
    minutes_sd: float = Field(ge=0.0, description="Standard deviation of expected minutes")

    @model_validator(mode="after")
    def _check_coherent(self) -> MinutesPrediction:
        if self.p_start > self.p_appearance + 1e-9:
            raise ValueError(
                f"p_start ({self.p_start}) cannot exceed p_appearance "
                f"({self.p_appearance}): starting implies appearing"
            )
        if self.p_60_plus > self.p_appearance + 1e-9:
            raise ValueError(
                f"p_60_plus ({self.p_60_plus}) cannot exceed p_appearance "
                f"({self.p_appearance}): a player cannot reach 60 minutes without playing"
            )
        return self


class ComponentExpectations(_Frozen):
    """Unconditional expected counts of scoring events over the horizon.

    These are *counts and probabilities*, never points. Converting them to points
    is the domain layer's job, because only the domain knows the scoring rules
    and only the domain knows which season's rules apply.
    """

    minutes: MinutesPrediction
    goals: float = Field(ge=0.0)
    assists: float = Field(ge=0.0)
    clean_sheet_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Joint probability of a clean sheet AND the player reaching 60 minutes. "
            "FPL pays a clean sheet only to players who last 60 minutes, so folding "
            "the minutes condition in here keeps the assembly a plain expectation."
        ),
    )
    goals_conceded: float = Field(ge=0.0)
    saves: float = Field(ge=0.0)
    yellow_cards: float = Field(ge=0.0)
    red_cards: float = Field(ge=0.0)
    own_goals: float = Field(ge=0.0)
    penalties_saved: float = Field(ge=0.0)
    penalties_missed: float = Field(ge=0.0)
    defensive_contribution_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of reaching the defensive-contribution threshold",
    )
    bonus: float = Field(ge=0.0, description="Expected bonus points")


class PointsBreakdown(_Frozen):
    """How each component contributed to the assembled points total.

    This is the evidence the explanation layer reads. Because it is produced by
    the same domain function that computes the total, an explanation can never
    disagree with the number it explains.
    """

    appearance: float
    goals: float
    assists: float
    clean_sheets: float
    goals_conceded: float
    saves: float
    cards: float
    own_goals: float
    penalties: float
    defensive_contribution: float
    bonus: float

    @property
    def total(self) -> float:
        return (
            self.appearance
            + self.goals
            + self.assists
            + self.clean_sheets
            + self.goals_conceded
            + self.saves
            + self.cards
            + self.own_goals
            + self.penalties
            + self.defensive_contribution
            + self.bonus
        )


class PlayerPrediction(_Frozen):
    """A prediction for one player over one horizon. The unit of model output."""

    player_code: PlayerCode
    position: Position
    from_gameweek: GameweekId = Field(description="First gameweek in the horizon")
    horizon_gameweeks: int = Field(ge=1, le=10)

    components: ComponentExpectations
    breakdown: PointsBreakdown
    expected_points: float
    expected_points_sd: float = Field(
        ge=0.0, description="Uncertainty, consumed by the optimizer's risk penalty"
    )

    scoring_rules_version: str = Field(
        description="Which pinned scoring snapshot assembled these points"
    )
    provenance: PredictionProvenance

    feature_evidence: FeatureEvidence | None = Field(
        default=None,
        description=(
            "The features this prediction was made from, for the explanatory "
            "panel only. Optional because the closed-form baseline does not "
            "build the full catalogue and must not claim to have: absent "
            "evidence is honest, invented evidence is not."
        ),
    )

    @model_validator(mode="after")
    def _breakdown_matches_total(self) -> PlayerPrediction:
        if abs(self.breakdown.total - self.expected_points) > 1e-6:
            raise ValueError(
                f"breakdown sums to {self.breakdown.total} but expected_points is "
                f"{self.expected_points}; an explanation must never disagree with its total"
            )
        return self
