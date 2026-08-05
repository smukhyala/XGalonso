"""A manager's situation as a dense numeric vector.

This is the module that makes "context-conditioned" mean something mechanical.
:class:`~xg_alonso.contracts.context.DecisionContext` says what a manager wants
and will not allow; :func:`encode_context` turns that into a fixed-width,
versioned, named vector that a model can consume as an *input* rather than as a
filter applied after the fact.

Two design rules do the load-bearing work, and both are enforced structurally
rather than by convention.

**Player identity is never a feature.**
A locked player enters the vector only through what locking them does to the
*shape* of the remaining problem: how many slots of their position are frozen,
how much of the budget they consume, how concentrated the squad becomes on their
club. "One of three forward slots is frozen" transfers to a manager who never
owned that player. "Player 12345 is frozen" transfers to nobody, and a model fed
the latter would memorise codes and generalise to no one. ``tests/domain/
test_context_features.py`` asserts this by permuting every
:data:`~xg_alonso.contracts.identifiers.PlayerCode` and requiring the vector to
come back bitwise identical.

**Constraints can never be traded against points.**
:func:`encode_context` takes no
:class:`~xg_alonso.contracts.prediction.PlayerPrediction`, no expected-points
mapping, and no scoring rules — look at the signature. There is no argument from
which a points total could be recovered, so the constraint block is *incapable*
of expressing an exchange rate between "keep Haaland" and "score more". That is
the guarantee :mod:`xg_alonso.contracts.objective` spends its module docstring
insisting on, made unfalsifiable-by-construction instead of asserted.

Purity
------

``domain`` may not import a dataframe engine (see the ``domain-purity`` contract
in ``.importlinter``), so this module takes mappings and returns numpy. That is
a constraint worth having: it keeps the encoding a pure function of stated
inputs, which is what makes it cheap to test exhaustively and impossible to make
accidentally depend on row order.

Missing inputs
--------------

Fixture data is optional, and when it is absent the block is zeroed **and
flagged** rather than silently zeroed. A zero that means "no fixture advantage"
and a zero that means "nobody told me about fixtures" are different claims, and
a model given no way to tell them apart will learn the wrong one. The flag is
why the fixture block is four dimensions rather than three.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from xg_alonso.contracts.context import DecisionContext
from xg_alonso.contracts.identifiers import TeamId
from xg_alonso.contracts.objective import PrimaryMetric, RequirementKind
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.domain.constraints import legal_formations
from xg_alonso.domain.rules import SquadRules

__all__ = [
    "CONTEXT_BLOCKS",
    "CONTEXT_FEATURE_NAMES",
    "CONTEXT_FEATURE_VERSION",
    "ContextVector",
    "encode_context",
]

CONTEXT_FEATURE_VERSION: Final[str] = "context_features_v1"
"""Bumped whenever a name is added, removed or redefined.

Carried on every :class:`ContextVector` and recorded in provenance, so a model
fitted under one encoding can never be fed another. A silently reordered vector
is the failure mode this exists to prevent: it produces no error, just quietly
wrong conditioning.
"""

# Normalisation divisors.
#
# Every one of these exists to put a dimension roughly on [0, 1] so no single
# input dominates a distance or a gradient by unit choice alone. They are scale
# hints, not FPL constants — the real game constants (squad size, positional
# quotas, budget, transfer caps) all come from `SquadRules`, which is read from
# the pinned bootstrap snapshot. Nothing here is transcribed from memory.
_HORIZON_SCALE: Final[float] = 10.0
"""Matches `ManagerObjective.planning_horizon`'s validated ceiling of 10."""

_HIT_SCALE: Final[float] = 8.0
"""Two hits. Beyond this the distinction stops mattering behaviourally."""

_BELIEF_SCALE: Final[float] = 5.0
"""Five stated beliefs is already an unusually opinionated request."""

_OUTFIELD_SLOTS: Final[float] = 10.0
"""Outfielders in a starting XI, used only to scale a formation triple."""

_POSITION_ORDER: Final[tuple[Position, ...]] = (
    Position.GKP,
    Position.DEF,
    Position.MID,
    Position.FWD,
)


def _names() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Block name to ordered feature names.

    Declared as data so :data:`CONTEXT_FEATURE_NAMES` and :data:`CONTEXT_BLOCKS`
    cannot disagree — they are both derived from this one definition.
    """
    return (
        (
            "objective",
            (
                *(f"objective_metric_{metric.value}" for metric in PrimaryMetric),
                "objective_signed_uncertainty_penalty",
                "objective_ownership_sign",
                "objective_planning_horizon",
                "objective_team_value_weight",
                "objective_captaincy_weight",
                "objective_transfer_cost_weight",
            ),
        ),
        (
            "transfer_freedom",
            (
                "transfer_free_transfers",
                "transfer_max_bound",
                "transfer_is_bounded",
                "transfer_max_points_hit",
                "transfer_allows_hits",
            ),
        ),
        (
            "slot_pressure",
            tuple(f"slot_pressure_{position.value.lower()}" for position in _POSITION_ORDER),
        ),
        (
            "club_pressure",
            (
                "club_max_concentration",
                "club_locked_concentration",
                "club_share_at_ceiling",
            ),
        ),
        (
            "budget",
            (
                "budget_bank_share",
                "budget_headroom_per_open_slot",
                "budget_squad_value_share",
                "budget_minimum_bank_share",
                "budget_bank_is_binding",
                "budget_sell_on_drag",
            ),
        ),
        (
            "formation",
            (
                "formation_defenders",
                "formation_midfielders",
                "formation_forwards",
                "formation_is_fixed",
                "formation_reachable_share",
            ),
        ),
        (
            "fixture",
            (
                "fixture_locked_load",
                "fixture_sellable_load",
                "fixture_load_gap",
                "fixture_outlook_available",
            ),
        ),
        (
            "belief",
            (
                "belief_count",
                "belief_mean_signed_confidence",
                "belief_squad_share",
            ),
        ),
    )


def _flatten() -> tuple[tuple[str, ...], dict[str, slice]]:
    names: list[str] = []
    blocks: dict[str, slice] = {}
    for block, block_names in _names():
        start = len(names)
        names.extend(block_names)
        blocks[block] = slice(start, len(names))
    return tuple(names), blocks


CONTEXT_FEATURE_NAMES, CONTEXT_BLOCKS = _flatten()
"""Ordered feature names, and the slice each block occupies.

Order is part of the contract. A model stores :data:`CONTEXT_FEATURE_VERSION`
and these names alongside its weights so a reordering is caught at load rather
than absorbed as a silent accuracy regression.
"""


@dataclass(frozen=True)
class ContextVector:
    """One encoded situation.

    Frozen, and the array is made read-only on construction. A representation
    that a caller can mutate in place is one whose recorded fingerprint stops
    describing what was actually used.
    """

    values: np.ndarray
    names: tuple[str, ...]
    version: str

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.names),):
            raise ValueError(
                f"context vector has {self.values.shape} values for "
                f"{len(self.names)} names; the encoding and its declared names "
                "have diverged"
            )
        self.values.setflags(write=False)

    def block(self, name: str) -> np.ndarray:
        """The sub-vector for one named block.

        Named access exists so a caller can reason about — and a test can
        assert on — "the constraint part" without hardcoding offsets that a
        later insertion would shift.
        """
        if name not in CONTEXT_BLOCKS:
            raise KeyError(f"unknown context block {name!r}; have {sorted(CONTEXT_BLOCKS)}")
        return self.values[CONTEXT_BLOCKS[name]]

    def as_mapping(self) -> dict[str, float]:
        """Name to value, for reports and debugging."""
        return dict(zip(self.names, (float(v) for v in self.values), strict=True))

    def fingerprint(self) -> str:
        """Content hash over the rounded values, names and version.

        Rounded to 9 decimals before hashing so a fingerprint survives the last
        bit of floating-point noise between platforms, which would otherwise
        make a reproducibility check fail for a reason that is not a difference.
        """
        payload = "|".join(
            (
                self.version,
                ",".join(self.names),
                ",".join(f"{value:.9f}" for value in self.values),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio that yields 0.0 rather than raising on a zero denominator.

    Used only where a zero denominator genuinely means "this pressure does not
    apply" — an empty squad has no club concentration — never to paper over a
    missing input, which is what the fixture-availability flag is for.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean_load(
    teams: Sequence[TeamId],
    load: Mapping[TeamId, Sequence[float]],
    horizon: int,
) -> float:
    per_team: list[float] = []
    for team in teams:
        window = load.get(team, ())
        clipped = list(window)[:horizon]
        if clipped:
            per_team.append(sum(clipped) / len(clipped))
    if not per_team:
        return 0.0
    return sum(per_team) / len(per_team)


def _objective_block(context: DecisionContext) -> list[float]:
    objective = context.objective
    one_hot = [1.0 if metric is objective.primary_metric else 0.0 for metric in PrimaryMetric]
    return [
        *one_hot,
        objective.signed_uncertainty_penalty,
        objective.ownership_preference.ownership_sign,
        objective.planning_horizon / _HORIZON_SCALE,
        objective.team_value_weight,
        objective.captaincy_weight,
        objective.transfer_cost_weight,
    ]


def _transfer_block(context: DecisionContext, rules: SquadRules) -> list[float]:
    constraints = context.constraints
    ceiling = float(rules.max_free_transfers)
    available = float(context.squad.free_transfers) if context.squad is not None else 1.0
    declared = constraints.max_transfers
    bound = ceiling if declared is None else float(declared)
    return [
        _safe_ratio(available, ceiling),
        _safe_ratio(bound, ceiling),
        0.0 if declared is None else 1.0,
        min(constraints.max_points_hit / _HIT_SCALE, 1.0),
        1.0 if constraints.allows_hits else 0.0,
    ]


def _slot_pressure_block(context: DecisionContext, rules: SquadRules) -> list[float]:
    """Share of each position's squad quota that is frozen.

    The single most important structural encoding of a lock. A manager who has
    frozen two of three forwards is short of forward options regardless of which
    forwards they are.
    """
    if context.squad is None:
        return [0.0 for _ in _POSITION_ORDER]
    frozen = context.locked_codes()
    locked_by_position = Counter(
        pick.position for pick in context.squad.picks if pick.player_code in frozen
    )
    return [
        _safe_ratio(locked_by_position.get(position, 0), rules.rule_for(position).squad_select)
        for position in _POSITION_ORDER
    ]


def _club_pressure_block(context: DecisionContext, rules: SquadRules) -> list[float]:
    if context.squad is None or not context.squad.picks:
        return [0.0, 0.0, 0.0]
    picks = context.squad.picks
    frozen = context.locked_codes()
    ceiling = float(rules.max_per_club)

    all_counts = Counter(pick.team_id for pick in picks)
    locked_counts = Counter(pick.team_id for pick in picks if pick.player_code in frozen)
    at_ceiling = sum(1 for count in all_counts.values() if count >= rules.max_per_club)

    # The most clubs that *can* sit at the ceiling, from the rules rather than
    # from the count of Premier League teams — which SquadRules does not publish
    # and which must therefore not be typed here.
    max_clubs_at_ceiling = rules.squad_size / rules.max_per_club

    return [
        _safe_ratio(float(max(all_counts.values())), ceiling),
        _safe_ratio(float(max(locked_counts.values(), default=0)), ceiling),
        _safe_ratio(float(at_ceiling), max_clubs_at_ceiling),
    ]


def _budget_block(context: DecisionContext, rules: SquadRules) -> list[float]:
    constraints = context.constraints
    total_budget = float(rules.total_budget)
    minimum_bank = float(constraints.minimum_bank)

    if context.squad is None:
        # From scratch: the whole budget is headroom and none of it is committed.
        return [1.0, 1.0, 0.0, _safe_ratio(minimum_bank, total_budget), 0.0, 0.0]

    squad = context.squad
    frozen = context.locked_codes()
    bank = float(squad.bank)
    squad_value = float(squad.squad_value)
    holdings = bank + squad_value

    sellable: list[SquadPick] = [p for p in squad.picks if p.player_code not in frozen]
    open_slots = len(sellable)
    spendable = bank + sum(float(p.selling_price) for p in sellable) - minimum_bank
    headroom = _safe_ratio(spendable, float(open_slots)) if open_slots else 0.0

    drag_terms = [
        _safe_ratio(float(p.current_price) - float(p.selling_price), float(p.current_price))
        for p in sellable
    ]
    drag = sum(drag_terms) / len(drag_terms) if drag_terms else 0.0

    return [
        _safe_ratio(bank, holdings),
        # Normalised against the mean price a single slot commands, so "how much
        # room per remaining slot" reads on the same scale as a player's price.
        #
        # Note the direction: locking players *raises* this, because it removes
        # slots faster than it removes money. That is not a bug and a test pins
        # it. A manager who has protected most of their squad is constrained
        # rather than poor, and the flexibility half of that trade is carried by
        # the `slot_pressure_*` dimensions. Encoding both is what separates
        # "rich and flexible" from "rich but boxed in" — situations that want
        # different recommendations and that one budget number cannot tell apart.
        #
        # Capped at 1.0: past a slot's worth of headroom per slot the difference
        # stops changing what is buyable, and an uncapped outlier would dominate
        # every distance this vector participates in.
        min(_safe_ratio(headroom, _safe_ratio(total_budget, float(rules.squad_size))), 1.0),
        _safe_ratio(squad_value, total_budget),
        _safe_ratio(minimum_bank, total_budget),
        1.0 if bank <= minimum_bank else 0.0,
        drag,
    ]


def _formation_block(context: DecisionContext, rules: SquadRules) -> list[float]:
    shapes = legal_formations(rules)
    requested = context.formation
    if requested is None:
        return [0.0, 0.0, 0.0, 0.0, 1.0]

    shaped = context.requirements.of_kind(RequirementKind.FORMATION)
    defenders, midfielders, forwards = shaped[0].formation_counts()
    reachable = sum(1 for shape in shapes if shape[1:] == (defenders, midfielders, forwards))
    return [
        defenders / _OUTFIELD_SLOTS,
        midfielders / _OUTFIELD_SLOTS,
        forwards / _OUTFIELD_SLOTS,
        1.0,
        _safe_ratio(float(reachable), float(len(shapes))),
    ]


def _fixture_block(
    context: DecisionContext,
    club_fixture_load: Mapping[TeamId, Sequence[float]] | None,
) -> list[float]:
    if club_fixture_load is None or context.squad is None:
        return [0.0, 0.0, 0.0, 0.0]

    frozen = context.locked_codes()
    horizon = context.objective.planning_horizon
    locked_teams = [p.team_id for p in context.squad.picks if p.player_code in frozen]
    sellable_teams = [p.team_id for p in context.squad.picks if p.player_code not in frozen]

    locked_load = _mean_load(locked_teams, club_fixture_load, horizon)
    sellable_load = _mean_load(sellable_teams, club_fixture_load, horizon)
    return [locked_load, sellable_load, locked_load - sellable_load, 1.0]


def _belief_block(context: DecisionContext) -> list[float]:
    beliefs = context.beliefs
    if not beliefs:
        return [0.0, 0.0, 0.0]

    signed = [belief.confidence * belief.proposition.direction for belief in beliefs]
    touched = 0.0
    if context.squad is not None and context.squad.picks:
        held = {int(pick.player_code) for pick in context.squad.picks}
        about_squad = {belief.entity_id for belief in beliefs} & held
        touched = len(about_squad) / len(context.squad.picks)

    return [
        min(len(beliefs) / _BELIEF_SCALE, 1.0),
        sum(signed) / len(signed),
        touched,
    ]


def encode_context(
    context: DecisionContext,
    *,
    rules: SquadRules,
    club_fixture_load: Mapping[TeamId, Sequence[float]] | None = None,
) -> ContextVector:
    """Encode a decision situation as a fixed-width numeric vector.

    Note what is **not** in this signature: no predictions, no expected points,
    no scoring rules, no player values of any kind. The constraint blocks
    therefore cannot express a trade against points even in principle. That is
    the objective/constraint firewall, enforced by the type system rather than
    by a reviewer noticing.

    Args:
        context: The manager's objective, constraints, beliefs and situation.
        rules: Squad quotas, budget and transfer caps, read from the pinned
            bootstrap snapshot. Supplies every normalisation denominator that
            corresponds to a real game constant.
        club_fixture_load: Optional per-club fixture difficulty over the coming
            gameweeks. When omitted the fixture block is zeroed *and* its
            availability flag is set to zero, so a model can tell "no fixture
            edge" from "no fixture data".

    Returns:
        A :class:`ContextVector` whose length always equals
        ``len(CONTEXT_FEATURE_NAMES)``, regardless of how much of the situation
        was actually known. Fixed width is what lets it be a model input.
    """
    values: list[float] = [
        *_objective_block(context),
        *_transfer_block(context, rules),
        *_slot_pressure_block(context, rules),
        *_club_pressure_block(context, rules),
        *_budget_block(context, rules),
        *_formation_block(context, rules),
        *_fixture_block(context, club_fixture_load),
        *_belief_block(context),
    ]
    return ContextVector(
        values=np.asarray(values, dtype=np.float64),
        names=CONTEXT_FEATURE_NAMES,
        version=CONTEXT_FEATURE_VERSION,
    )
