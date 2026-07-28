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
    "PANEL_BY_POSITION",
    "FeatureEvidence",
    "FeatureValue",
    "PanelEntry",
    "panel_feature_names",
    "panel_for",
    "panel_source_columns",
]

EVIDENCE_PANEL_VERSION: Final[str] = "panel_v3"


class PanelEntry(BaseModel):
    """One declared member of the explanatory panel.

    ``label`` is what a reader sees. It is held here rather than derived from the
    feature name because ``expected_goals_per90_5`` is a column name, not a
    phrase anybody would say out loud, and the mapping between the two should be
    reviewable in one place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Canonical identifier for this panel entry")
    label: str = Field(description="Human-readable name, used in prose")
    family: str = Field(description="Catalogue family, used for grouping")
    sources: tuple[str, ...] = Field(
        default=(),
        description=(
            "Column names that supply this entry, in order of preference. "
            "A panel entry names a *concept* — expected goals per 90 — and the "
            "feature sets spell it differently: the catalogue calls it "
            "`expected_goals_per90_5`, the closed-form baseline "
            "`xg_per90_shrunk`. Listing both means an explanation over the "
            "baseline can still cite xG instead of falling silent, which is the "
            "difference between a justification naming a statistic and one "
            "restating the projection in other words. Defaults to `name` alone."
        ),
    )

    higher_is_better: bool = Field(
        default=True,
        description=(
            "Whether a larger value argues for the player. False for volatility, "
            "where a percentile must be inverted before it means 'good'."
        ),
    )

    def columns(self) -> tuple[str, ...]:
        """Source columns to try, in order of preference."""
        return self.sources or (self.name,)


# --- position-specific panels ------------------------------------------------
#
# A single panel for every position produced explanations that read as nonsense
# for half the squad. "Points per 90" is a sensible thing to say about a
# midfielder and a strange thing to say about a goalkeeper, who accumulates
# points by saving shots and keeping clean sheets and cannot meaningfully be
# ranked on an attacking rate at all. Worse, the percentile made it look
# rigorous: a keeper sits in the 2nd percentile for expected goals per 90
# against *keepers*, which is true, uninformative, and reads as a criticism.
#
# So the panel is chosen per position. Each list below is what a manager would
# actually weigh for that position, in the order they would weigh it.

_CORE: Final[tuple[PanelEntry, ...]] = (
    PanelEntry(
        name="minutes_mean_5",
        label="minutes, last five",
        family="player_performance",
    ),
    PanelEntry(
        name="starts_mean_5",
        label="start rate, last five",
        family="player_performance",
        sources=("starts_mean_5", "start_rate_5"),
    ),
    PanelEntry(
        name="total_points_mean_5",
        label="points per match, last five",
        family="player_performance",
    ),
    PanelEntry(
        name="opponent_conceded_xg_mean_5",
        label="opponent expected goals conceded",
        family="opponent",
    ),
    PanelEntry(name="is_home", label="playing at home", family="fixture"),
)
"""Shared by every position: minutes, form and fixture decide everybody."""

_GOALKEEPER: Final[tuple[PanelEntry, ...]] = (
    PanelEntry(name="saves_per90_5", label="saves per 90", family="player_rate"),
    PanelEntry(
        name="clean_sheets_mean_5",
        label="clean-sheet rate, last five",
        family="player_performance",
    ),
    PanelEntry(
        name="expected_goals_conceded_per90_5",
        label="expected goals conceded per 90",
        family="player_rate",
        higher_is_better=False,
    ),
    PanelEntry(
        name="bps_per90_5",
        label="bonus points system per 90",
        family="player_rate",
    ),
)
"""A keeper scores through saves, clean sheets and bonus. Nothing else applies.

Note that ``saves_per90`` and ``expected_goals_conceded_per90`` pull in opposite
directions and both belong here: a keeper on a leaky team faces more shots and
banks more save points, while a keeper behind a good defence banks clean sheets.
Showing one without the other would recommend the wrong keeper for the wrong
reason.
"""

_DEFENDER: Final[tuple[PanelEntry, ...]] = (
    PanelEntry(
        name="clean_sheets_mean_5",
        label="clean-sheet rate, last five",
        family="player_performance",
    ),
    PanelEntry(
        name="expected_goals_conceded_per90_5",
        label="expected goals conceded per 90",
        family="player_rate",
        higher_is_better=False,
    ),
    PanelEntry(
        name="defensive_contribution_per90_5",
        label="defensive actions per 90",
        family="player_rate",
    ),
    PanelEntry(
        name="threat_per90_5",
        label="attacking threat per 90",
        family="player_rate",
    ),
    PanelEntry(
        name="creativity_per90_5",
        label="creativity per 90",
        family="player_rate",
    ),
    PanelEntry(
        name="expected_goals_per90_5",
        label="expected goals per 90",
        family="player_rate",
        sources=("expected_goals_per90_5", "xg_per90_shrunk"),
    ),
    PanelEntry(
        name="bps_per90_5",
        label="bonus points system per 90",
        family="player_rate",
    ),
)
"""Defenders are two jobs in one position.

Clean sheets and defensive actions price the defending; threat and creativity
price the attacking, which is what separates a wing-back from a centre-half and
is most of the reason one defender outscores another at the same price.
"""

_MIDFIELDER: Final[tuple[PanelEntry, ...]] = (
    PanelEntry(
        name="expected_goals_per90_5",
        label="expected goals per 90",
        family="player_rate",
        sources=("expected_goals_per90_5", "xg_per90_shrunk"),
    ),
    PanelEntry(
        name="expected_assists_per90_5",
        label="expected assists per 90",
        family="player_rate",
        sources=("expected_assists_per90_5", "xa_per90_shrunk"),
    ),
    PanelEntry(name="threat_per90_5", label="attacking threat per 90", family="player_rate"),
    PanelEntry(name="creativity_per90_5", label="creativity per 90", family="player_rate"),
    PanelEntry(
        name="defensive_contribution_per90_5",
        label="defensive actions per 90",
        family="player_rate",
    ),
    PanelEntry(
        name="bps_per90_5",
        label="bonus points system per 90",
        family="player_rate",
    ),
    PanelEntry(
        name="clean_sheets_mean_5",
        label="clean-sheet rate, last five",
        family="player_performance",
    ),
)
"""Midfielders can score through any route, so the panel is the widest.

Clean sheets are included because a midfielder is paid one point for them, and
defensive actions because a holding midfielder now banks points a creator never
will.
"""

_FORWARD: Final[tuple[PanelEntry, ...]] = (
    PanelEntry(
        name="expected_goals_per90_5",
        label="expected goals per 90",
        family="player_rate",
        sources=("expected_goals_per90_5", "xg_per90_shrunk"),
    ),
    PanelEntry(name="threat_per90_5", label="attacking threat per 90", family="player_rate"),
    PanelEntry(
        name="expected_assists_per90_5",
        label="expected assists per 90",
        family="player_rate",
        sources=("expected_assists_per90_5", "xa_per90_shrunk"),
    ),
    PanelEntry(
        name="expected_goal_involvements_per90_10",
        label="expected goal involvements per 90",
        family="player_rate",
        sources=("expected_goal_involvements_per90_10", "xgi_per90_shrunk"),
    ),
    PanelEntry(
        name="bps_per90_5",
        label="bonus points system per 90",
        family="player_rate",
    ),
    PanelEntry(
        name="total_points_max_5",
        label="best return, last five",
        family="player_ceiling",
    ),
)
"""A forward earns almost everything through goals, so the panel leads with xG.

Clean sheets are absent: a forward is paid nothing for one, and including it
would put a number on screen that cannot move his score.
"""

_SHAPE: Final[tuple[PanelEntry, ...]] = (
    PanelEntry(
        name="total_points_std_10",
        label="points volatility",
        family="player_volatility",
        higher_is_better=False,
    ),
)
"""Return shape, appended to every position. Two players averaging the same are
not the same player if one alternates hauls and blanks."""

#: Keyed by the position's short name rather than by the `Position` enum, which
#: lives in `contracts.prediction` — and `contracts.prediction` imports
#: `FeatureEvidence` from here. Importing the enum back would close that loop.
#: `Position` is a `StrEnum`, so callers can index this with either form.
PANEL_BY_POSITION: Final[dict[str, tuple[PanelEntry, ...]]] = {
    "GKP": _CORE + _GOALKEEPER + _SHAPE,
    "DEF": _CORE + _DEFENDER + _SHAPE,
    "MID": _CORE + _MIDFIELDER + _SHAPE,
    "FWD": _CORE + _FORWARD + _SHAPE,
}


def panel_for(position: str) -> tuple[PanelEntry, ...]:
    """The panel for one position, by short name (``GKP``, ``DEF``, ...).

    Falls back to the shared core for anything unrecognised rather than raising:
    an unknown position should cost an explanation its detail, not stop a
    prediction being made.
    """
    return PANEL_BY_POSITION.get(str(position), _CORE)


#: Every entry across every position, de-duplicated.
#:
#: Derived rather than declared. It used to be the single hand-written panel and
#: is kept only for callers that need the full vocabulary — a stored prediction
#: reader, say — so leaving it as a second hand-maintained list would guarantee
#: the two drifted.
EXPLANATORY_PANEL: Final[tuple[PanelEntry, ...]] = tuple(
    {entry.name: entry for panel in PANEL_BY_POSITION.values() for entry in panel}.values()
)


def panel_feature_names() -> tuple[str, ...]:
    """Canonical names, one per entry, across every position's panel."""
    seen: list[str] = []
    for panel in PANEL_BY_POSITION.values():
        for entry in panel:
            if entry.name not in seen:
                seen.append(entry.name)
    return tuple(seen)


def panel_source_columns() -> tuple[str, ...]:
    """Every column any panel entry may read, across all feature sets."""
    seen: list[str] = []
    for panel in PANEL_BY_POSITION.values():
        for entry in panel:
            for column in entry.columns():
                if column not in seen:
                    seen.append(column)
    return tuple(seen)


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

    def ranked(self) -> tuple[FeatureValue, ...]:
        """The whole panel, most distinguishing first, unmeasured values last.

        Distinct from :meth:`notable`, which filters to what stands out. The
        panel is chosen per position precisely so a reader can see the metrics
        that matter for *that* position — hiding the middling ones defeats the
        purpose, because "his clean-sheet rate is unremarkable" is itself
        something a manager wants to know before buying a defender.
        """
        measured = [v for v in self.values if v.percentile is not None]
        unmeasured = [v for v in self.values if v.percentile is None]
        measured.sort(key=lambda v: -abs((v.percentile or 0.5) - 0.5))
        return tuple(measured + unmeasured)

    def notable(self) -> tuple[FeatureValue, ...]:
        """Distinguishing values, most distinguishing first.

        Sorted by distance from the median rather than by raw value, because a
        player's explanation should lead with what makes him different, not with
        whichever of his features happens to be measured on the largest scale.
        """
        candidates = [v for v in self.values if v.is_notable]
        candidates.sort(key=lambda v: -abs((v.percentile or 0.5) - 0.5))
        return tuple(candidates)
