"""Feature evidence carried alongside a prediction.

**Why this exists.** A prediction used to be a set of numbers with no link back
to the features that produced it: ``predict_with_models`` accepted a frame of
171 columns, emitted nine component estimates, and discarded the frame. The
explanation layer therefore had nothing to cite except the model's own output,
which is how a recommendation came to justify itself with "0.53 projected goal
involvements" — a derived quantity — while the underlying xG that drove it sat
in a column nobody could reach.

Attaching evidence closes that gap without weakening the no-fabrication
guarantee: the values here are read from the same frame the model predicted on,
so an explanation citing them is citing the model's actual inputs.

**Why a curated panel rather than every feature.** Exposing all 171 per player
would be a wall of heavily correlated numbers — ``goals_scored_mean_3`` next to
``goals_scored_mean_5`` next to ``goals_scored_mean_10`` — which explains
nothing and invites the reader to pattern-match noise. The panel is small,
interpretable, and cross-checked against measured importance, so what a user is
shown is both legible and load-bearing.

**Why percentiles.** A raw value is not an argument. ``0.53`` expected goals per
90 is elite for a defender and unremarkable for a striker, so every value is
carried with its rank inside the same position. The percentile is what turns a
number into evidence.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EVIDENCE_PANEL_VERSION",
    "EXPLANATORY_PANEL",
    "FeatureEvidence",
    "FeatureValue",
    "PanelEntry",
    "panel_feature_names",
]

EVIDENCE_PANEL_VERSION: Final[str] = "panel_v1"


class PanelEntry(BaseModel):
    """One declared member of the explanatory panel.

    ``label`` is what a reader sees. It is held here rather than derived from the
    feature name because ``expected_goals_per90_5`` is a column name, not a
    phrase anybody would say out loud, and the mapping between the two should be
    reviewable in one place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Catalogue column name")
    label: str = Field(description="Human-readable name, used in prose")
    family: str = Field(description="Catalogue family, used for grouping")
    higher_is_better: bool = Field(
        default=True,
        description=(
            "Whether a larger value argues for the player. False for volatility, "
            "where a percentile must be inverted before it means 'good'."
        ),
    )


#: The declared panel. Membership is deliberate, not generated: each entry earns
#: its place by being something a manager would recognise as a reason.
#:
#: Windows are fixed at the shortest window that is not pure noise — five
#: appearances for rates, three for minutes — because an explanation citing a
#: twenty-match mean is describing a player's past rather than his form, and form
#: is what a transfer decision turns on.
EXPLANATORY_PANEL: Final[tuple[PanelEntry, ...]] = (
    PanelEntry(
        name="expected_goals_per90_5",
        label="expected goals per 90",
        family="player_rate",
    ),
    PanelEntry(
        name="expected_assists_per90_5",
        label="expected assists per 90",
        family="player_rate",
    ),
    PanelEntry(
        name="expected_goal_involvements_per90_10",
        label="expected goal involvements per 90, ten-match",
        family="player_rate",
    ),
    PanelEntry(name="threat_per90_5", label="threat per 90", family="player_rate"),
    PanelEntry(name="creativity_per90_5", label="creativity per 90", family="player_rate"),
    PanelEntry(name="bps_per90_5", label="bonus points system per 90", family="player_rate"),
    PanelEntry(name="minutes_mean_3", label="minutes, last three", family="player_performance"),
    PanelEntry(name="starts_mean_5", label="start rate, last five", family="player_performance"),
    PanelEntry(
        name="total_points_mean_5",
        label="points per match, last five",
        family="player_performance",
    ),
    PanelEntry(name="total_points_max_5", label="best return, last five", family="player_ceiling"),
    PanelEntry(
        name="total_points_std_10",
        label="points volatility",
        family="player_volatility",
        higher_is_better=False,
    ),
    PanelEntry(
        name="opponent_conceded_xg_mean_5",
        label="opponent expected goals conceded",
        family="opponent",
    ),
    PanelEntry(
        name="opponent_clean_sheets_against_mean_5",
        label="opponent clean sheets kept",
        family="opponent",
        higher_is_better=False,
    ),
    PanelEntry(name="is_home", label="playing at home", family="fixture"),
)


def panel_feature_names() -> tuple[str, ...]:
    """Every column the panel reads. Used to check the panel against a catalogue."""
    return tuple(entry.name for entry in EXPLANATORY_PANEL)


class FeatureValue(BaseModel):
    """One feature's value for one player, with the context that makes it legible.

    ``value`` and ``percentile`` are independently nullable. A feature can be
    computed but unrankable — when every player in a position shares the value,
    a percentile would assert a distinction that does not exist.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    label: str
    family: str
    value: float | None = Field(
        default=None,
        description=(
            "The feature's value, or None when it could not be computed. Never "
            "imputed: a player without enough history has no rate, and inventing "
            "one would put a number into an explanation that the data does not support."
        ),
    )
    percentile: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Rank within the same position for the same gameweek, 0-1.",
    )
    higher_is_better: bool = True

    @property
    def is_notable(self) -> bool:
        """Whether this value is far enough from the middle to be worth saying.

        A player at the 51st percentile is not evidence of anything. The
        threshold exists so explanations lead with what distinguishes a player
        rather than with whichever feature happens to be listed first.
        """
        if self.percentile is None:
            return False
        return abs(self.percentile - 0.5) >= 0.2

    @property
    def favourability(self) -> float | None:
        """Percentile oriented so that larger always means better for the player.

        Volatility ranks high when a player is erratic, which is an argument
        *against* him. Callers that rank evidence by how strongly it supports a
        player need that flipped, and doing it here keeps the sign convention in
        one place rather than at each call site.
        """
        if self.percentile is None:
            return None
        return self.percentile if self.higher_is_better else 1.0 - self.percentile


class FeatureEvidence(BaseModel):
    """The panel, materialised for one player.

    Held on a prediction so that an explanation and the number it explains cannot
    come from different places.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    panel_version: str = EVIDENCE_PANEL_VERSION
    values: tuple[FeatureValue, ...] = ()

    def get(self, name: str) -> FeatureValue | None:
        """The panel entry with this name, or ``None`` if it is not carried."""
        for value in self.values:
            if value.name == name:
                return value
        return None

    def value_of(self, name: str) -> float | None:
        """Just the number, when a caller does not need the context."""
        found = self.get(name)
        return None if found is None else found.value

    def notable(self) -> tuple[FeatureValue, ...]:
        """Distinguishing values, most distinguishing first.

        Sorted by distance from the median rather than by raw value, because a
        player's explanation should lead with what makes him different, not with
        whichever of his features happens to be measured on the largest scale.
        """
        candidates = [v for v in self.values if v.is_notable]
        candidates.sort(key=lambda v: -abs((v.percentile or 0.5) - 0.5))
        return tuple(candidates)
