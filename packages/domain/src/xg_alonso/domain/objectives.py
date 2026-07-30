"""The objective presets, and how an objective's weights are resolved.

Six presets, covering the situations a manager is actually in. They are starting
points a user edits, not a taxonomy — the point of the objective layer is that
the weights are data, so a preset is one row of that data with a name.

**Why the presets differ in more than one dimension.** A naive version would vary
only ``risk_preference`` and leave everything else alone. That is not how these
situations differ. A manager chasing a mini-league does not merely tolerate more
variance; they want *low ownership* (a template haul moves nobody), they weight
the *captain* far more heavily (it is the largest single swing available), and
they should penalise variance **negatively** — a certain average leaves them
exactly as far behind as they started.

Rank protection inverts all four. That inversion is the thing this package
exists to express, and it is why one global "best features" answer cannot serve
both managers.

This module is pure, per the domain layer's contract: no I/O, no dataframes.
"""

from __future__ import annotations

from typing import Final

from xg_alonso.contracts.objective import (
    ManagerObjective,
    OwnershipPreference,
    PrimaryMetric,
    RiskPreference,
    UtilityWeights,
)

__all__ = [
    "OBJECTIVE_PRESETS",
    "PRESET_IDS",
    "objective_preset",
    "preset_names",
]


_EXPECTED_POINTS: Final[ManagerObjective] = ManagerObjective(
    id="expected_points",
    name="Maximise expected points",
    primary_metric=PrimaryMetric.EXPECTED_POINTS,
    secondary_metrics=(PrimaryMetric.CAPTAINCY_UPSIDE,),
    risk_preference=RiskPreference.BALANCED,
    planning_horizon=5,
    ownership_preference=OwnershipPreference.NEUTRAL,
    team_value_weight=0.0,
    transfer_cost_weight=1.0,
    captaincy_weight=1.0,
    uncertainty_penalty=0.25,
)
"""The default question, and the only one most tools ask.

Moderate risk, five-week horizon. A transfer is permanent and paid for once, so
judging it on the next gameweek alone overvalues chasing a single fixture.
"""


_RANK_PROTECTION: Final[ManagerObjective] = ManagerObjective(
    id="rank_protection",
    name="Protect overall rank",
    primary_metric=PrimaryMetric.DOWNSIDE_PROTECTION,
    secondary_metrics=(PrimaryMetric.EXPECTED_POINTS,),
    risk_preference=RiskPreference.CONSERVATIVE,
    planning_horizon=2,
    ownership_preference=OwnershipPreference.TEMPLATE,
    team_value_weight=0.0,
    transfer_cost_weight=1.5,
    captaincy_weight=0.8,
    uncertainty_penalty=0.6,
    objective_weights=UtilityWeights(stability=1.0, turnover=0.5),
)
"""High ownership, minimal downside, heavy weight on secure minutes.

``transfer_cost_weight`` is above one because a hit is a *guaranteed* four-point
loss against an uncertain gain, and a manager protecting a rank is precisely the
one who should not take that trade. ``stability`` is doubled in the utility
weights for the same reason: a feature that works on average and fails
occasionally is worse here than a weaker feature that never surprises.
"""


_MINI_LEAGUE_CHASE: Final[ManagerObjective] = ManagerObjective(
    id="mini_league_chase",
    name="Chase a mini-league",
    primary_metric=PrimaryMetric.EXPECTED_RANK_GAIN,
    secondary_metrics=(PrimaryMetric.CAPTAINCY_UPSIDE, PrimaryMetric.DIFFERENTIAL_YIELD),
    risk_preference=RiskPreference.AGGRESSIVE,
    planning_horizon=3,
    ownership_preference=OwnershipPreference.DIFFERENTIAL,
    team_value_weight=0.1,
    transfer_cost_weight=0.5,
    captaincy_weight=1.4,
    uncertainty_penalty=0.1,
)
"""Behind, and needing variance to close a gap.

Note ``uncertainty_penalty`` is small **and** the risk preference is aggressive,
which makes the signed penalty negative: variance is actively rewarded. A
certain average is the one outcome that guarantees staying behind.

``transfer_cost_weight`` is halved because a four-point hit is cheap against a
forty-point deficit, and refusing hits is how a chaser runs out of gameweeks.
"""


_TEAM_VALUE: Final[ManagerObjective] = ManagerObjective(
    id="team_value_growth",
    name="Grow team value",
    primary_metric=PrimaryMetric.TEAM_VALUE_GROWTH,
    secondary_metrics=(PrimaryMetric.EXPECTED_POINTS,),
    risk_preference=RiskPreference.BALANCED,
    planning_horizon=3,
    ownership_preference=OwnershipPreference.NEUTRAL,
    team_value_weight=1.0,
    transfer_cost_weight=1.2,
    captaincy_weight=0.8,
    uncertainty_penalty=0.3,
)
"""Transfer momentum, with a floor on points.

**Not a price forecast.** Decision D11 defers the price model and no
current-season price data exists at GW1, so the primary metric scores net
transfer flow relative to ownership — the published leading indicator of a rise.
Points stay a secondary metric rather than being dropped, because a squad
optimised purely for value is a squad that scores nothing.
"""


_WILDCARD: Final[ManagerObjective] = ManagerObjective(
    id="wildcard_prep",
    name="Prepare for a wildcard",
    primary_metric=PrimaryMetric.EXPECTED_POINTS,
    secondary_metrics=(PrimaryMetric.TRANSFER_FLEXIBILITY,),
    risk_preference=RiskPreference.BALANCED,
    planning_horizon=6,
    ownership_preference=OwnershipPreference.NEUTRAL,
    team_value_weight=0.4,
    transfer_cost_weight=2.0,
    captaincy_weight=0.9,
    uncertainty_penalty=0.35,
    objective_weights=UtilityWeights(stability=0.8),
)
"""Six weeks out, optimising structure rather than this week's eleven.

``transfer_cost_weight`` is doubled: a manager about to wildcard should not be
paying hits now for players they are about to replace anyway. Per decision D5 no
chip *logic* runs — this shapes the squad a wildcard would inherit, and does not
decide when to play one.
"""


_HAALAND_AGGRESSIVE: Final[ManagerObjective] = ManagerObjective(
    id="locked_premium_aggressive",
    name="Locked premium, aggressive around him",
    primary_metric=PrimaryMetric.EXPECTED_RANK_GAIN,
    secondary_metrics=(PrimaryMetric.CAPTAINCY_UPSIDE,),
    risk_preference=RiskPreference.AGGRESSIVE,
    planning_horizon=3,
    ownership_preference=OwnershipPreference.DIFFERENTIAL,
    team_value_weight=0.1,
    transfer_cost_weight=0.6,
    captaincy_weight=1.6,
    uncertainty_penalty=0.1,
)
"""A premium asset is locked and the rest of the squad takes the risk.

The constraint — which player is locked — is **not** part of the objective, and
deliberately so. ``ManagerConstraints.locked_players`` carries it, because "keep
Haaland" is a statement about the feasible set and putting it in the objective
would let the optimizer trade it away for enough points.

The captaincy weight is the highest of any preset: when a premium is locked in,
the armband on him is the main lever left.
"""


OBJECTIVE_PRESETS: Final[tuple[ManagerObjective, ...]] = (
    _EXPECTED_POINTS,
    _RANK_PROTECTION,
    _MINI_LEAGUE_CHASE,
    _TEAM_VALUE,
    _WILDCARD,
    _HAALAND_AGGRESSIVE,
)

PRESET_IDS: Final[tuple[str, ...]] = tuple(o.id for o in OBJECTIVE_PRESETS)


def objective_preset(preset_id: str) -> ManagerObjective:
    """Look up a preset by id.

    Raises:
        KeyError: naming every available preset. A typo'd objective id must not
            silently fall back to the default — the whole point of the layer is
            that the objective changes the answer, so quietly answering a
            different question is the worst available failure.
    """
    for preset in OBJECTIVE_PRESETS:
        if preset.id == preset_id:
            return preset
    raise KeyError(f"no objective preset {preset_id!r}; available: {', '.join(PRESET_IDS)}")


def preset_names() -> dict[str, str]:
    """Preset id to human-readable name, for menus and help text."""
    return {preset.id: preset.name for preset in OBJECTIVE_PRESETS}
