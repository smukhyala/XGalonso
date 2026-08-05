"""The situation a decision is made in, as one addressable thing.

:mod:`~xg_alonso.contracts.objective` models *intent* — what a manager wants
(:class:`~xg_alonso.contracts.objective.ManagerObjective`), what they will not
allow (:class:`~xg_alonso.contracts.objective.ManagerConstraints`), and what they
believe (:class:`~xg_alonso.contracts.objective.UserBelief`). This module adds
the missing half: the *situation* that intent applies to — which squad, which
gameweek, which structural demands.

Why a new umbrella rather than another field on
:class:`~xg_alonso.contracts.objective.ObjectiveBundle`:

- A bundle is reusable across gameweeks and therefore cacheable. A context is
  not: it names a squad and a deadline. Merging them would make every bundle
  cache entry gameweek-specific for no gain.
- :class:`~xg_alonso.contracts.objective.SquadRequirements` deliberately lives on
  ``CompiledIntent`` rather than on the bundle, because the bundle says what the
  optimizer may not *do* while requirements say what the squad must *contain*.
  A ``DecisionContext`` is the first type that legitimately needs both, which is
  precisely why it is a new umbrella and not a field added to either.

**The three-way distinction survives intact and is enforced, not asserted.**
Objective, constraints and beliefs remain separately typed members reachable
through separate properties. Nothing here merges them. The load-bearing
guarantee — that a soft preference can never become a hard filter, nor a hunch
become a fact — is checked two ways:

- :func:`~xg_alonso.domain.context_features.encode_context` takes no
  ``PlayerPrediction`` argument at all, so the constraint block of a context
  vector is *structurally* incapable of trading against expected points.
- Permuting every :data:`~xg_alonso.contracts.identifiers.PlayerCode` in a
  context leaves its encoded vector bitwise identical. Player identity is not a
  feature, and that is a property test rather than a comment.

Three identities, three cardinalities, three consumers
-----------------------------------------------------

Keying anything on the raw constraint set gives every manager a private
registry: ``locked_players`` alone has 2\\ :sup:`15` states. Keying everything on
a coarse bucket throws away the resolution the whole exercise exists to capture.
So there are three identity functions and choosing between them is the design:

===========================  ==================  =====================================
function                     cardinality         keys what
===========================  ==================  =====================================
:meth:`feasibility_digest`   unbounded hash      the reachable-player pool. Covers
                                                 *only* fields that change which
                                                 players can enter the squad.
:meth:`context_key`          bounded (~10\\ :sup:`4`)  cached representations and
                                                 cluster fits. Every component is an
                                                 enum or a small band.
:meth:`context_fingerprint`  unbounded hash      provenance only. **Never** a cache
                                                 key.
===========================  ==================  =====================================

The split is the whole answer to "does conditioning on the manager fragment the
artifact space combinatorially". Bucket for caching; fingerprint for provenance;
digest for the pool. Nothing is written to disk keyed on the raw context.

What :meth:`feasibility_digest` deliberately excludes
----------------------------------------------------

``max_transfers``, ``max_points_hit`` and every belief. None of them changes
*which players are reachable* — they change how many moves are affordable and how
a reachable player is scored, which is a different question answered later. The
exclusion is a claim about the model, so ``tests/contracts/test_context.py``
asserts that varying ``max_points_hit`` alone leaves the digest unchanged.

``required_features`` and ``excluded_features`` are excluded for the same reason
and it is worth stating separately, because it is the less obvious call: they
restrict the *feature* space, not the player pool. A run that changes only its
required features faces an identical set of buyable players.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TenthsOfMillion
from xg_alonso.contracts.objective import (
    ManagerConstraints,
    ManagerObjective,
    ObjectiveBundle,
    RequirementKind,
    SquadRequirements,
    UserBelief,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadState

__all__ = [
    "CONTEXT_VERSION",
    "BeliefLoad",
    "BudgetBand",
    "ClubPressure",
    "ContextBucket",
    "DecisionContext",
    "HitAppetite",
    "LockPressure",
    "LockShape",
    "TransferFreedom",
]

CONTEXT_VERSION = "context_v1"
"""Bumped when the bucket vocabulary or the digest field set changes.

Carried in :meth:`DecisionContext.context_key` so a cache built under one
vocabulary can never be read under another.
"""

_UNCONSTRAINED_FORMATION = "free"
"""Bucket value when the manager named no starting shape."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- bucket vocabulary -------------------------------------------------------
#
# Every band below is deliberately coarse. These values key caches, and a band
# boundary that moves with a 0.1m price change would give every gameweek its own
# cache entry while claiming to describe a situation.


class TransferFreedom(StrEnum):
    """How many moves are available before hits."""

    NONE = "none"
    ONE = "one"
    TWO = "two"
    MANY = "many"


class HitAppetite(StrEnum):
    """How much the manager will pay in points to move."""

    NONE = "none"
    ONE_HIT = "one_hit"
    MULTI = "multi"


class LockPressure(StrEnum):
    """How much of the squad is frozen.

    Banded on the share of the squad that may still be sold, because that is the
    quantity the search actually feels. A manager with fourteen locked players
    and one free slot is solving a nearly different problem from one with none.
    """

    FREE = "free"
    """More than 80% of the squad is sellable."""

    LIGHT = "light"
    HEAVY = "heavy"

    FROZEN = "frozen"
    """Under 20% sellable. Barely a search at all."""

    UNKNOWN = "unknown"
    """No squad attached, so the share is not computable. Not an error: a
    from-scratch build genuinely has no squad to freeze."""


class LockShape(StrEnum):
    """Where the locks sit positionally.

    Position, never identity. "One of three forward slots is frozen" generalises
    to managers who never owned that player; "player 12345 is frozen" does not.
    """

    NONE = "none"
    GK = "gk"
    DEF = "def"
    MID = "mid"
    FWD = "fwd"
    MIXED = "mixed"


class ClubPressure(StrEnum):
    """Whether the three-per-club rule is close to binding."""

    SLACK = "slack"
    TIGHT = "tight"
    UNKNOWN = "unknown"


class BudgetBand(StrEnum):
    """Spending headroom in the bank.

    Banded on absolute money rather than a fraction of squad value: 0.5m in the
    bank means the same thing to every manager, while "5% of squad value" means
    different things at 95.0m and 103.0m.
    """

    BROKE = "broke"
    """Under 0.5m. Effectively a like-for-like market."""

    THIN = "thin"
    COMFORTABLE = "comfortable"
    RICH = "rich"
    UNKNOWN = "unknown"


class BeliefLoad(StrEnum):
    """Whether the manager asserted anything the model cannot see."""

    NONE = "none"
    SOME = "some"


_BANK_BROKE = TenthsOfMillion(5)
_BANK_THIN = TenthsOfMillion(15)
_BANK_COMFORTABLE = TenthsOfMillion(40)

_SELLABLE_FREE = 0.8
_SELLABLE_LIGHT = 0.5
_SELLABLE_HEAVY = 0.2

_CLUB_TIGHT_AT = 3
_MANY_TRANSFERS = 3
_MULTI_HIT = 8


class ContextBucket(_Frozen):
    """A bounded, canonical description of a decision situation.

    Cardinality is bounded by construction — the product of the enum sizes and
    the small formation vocabulary — which is what stops context-conditioning
    from multiplying the artifact space. ``tests/contracts/test_context.py``
    asserts that bound against randomly generated constraint sets rather than
    trusting the arithmetic.

    **This is a cache key, not a model input.** The vector fed to any learned
    model is the raw continuous encoding from
    :func:`~xg_alonso.domain.context_features.encode_context`. Feeding buckets to
    a model would discard exactly the resolution the encoding exists to provide.
    """

    objective_id: str = Field(min_length=1)
    """The existing four-field objective identity, reused unchanged."""

    transfer_freedom: TransferFreedom = TransferFreedom.ONE
    hit_appetite: HitAppetite = HitAppetite.NONE
    lock_pressure: LockPressure = LockPressure.UNKNOWN
    lock_shape: LockShape = LockShape.NONE
    club_pressure: ClubPressure = ClubPressure.UNKNOWN
    budget_band: BudgetBand = BudgetBand.UNKNOWN
    formation: str = _UNCONSTRAINED_FORMATION
    belief_load: BeliefLoad = BeliefLoad.NONE

    def key(self) -> str:
        """A short, stable, human-readable key.

        Readable on purpose. A cache directory of opaque hashes is impossible to
        reason about when a result looks wrong, and the whole claim of this layer
        is that different managers get different treatment — which someone has to
        be able to *see*.
        """
        return "|".join(
            (
                self.objective_id,
                self.transfer_freedom.value,
                self.hit_appetite.value,
                self.lock_pressure.value,
                self.lock_shape.value,
                self.club_pressure.value,
                self.budget_band.value,
                self.formation,
                self.belief_load.value,
            )
        )


_VOLATILE_FIELDS = frozenset({"created_at"})
"""Fields stripped before fingerprinting.

:class:`~xg_alonso.contracts.objective.ManagerObjective` and
:class:`~xg_alonso.contracts.objective.UserBelief` both inherit ``created_at``
from ``_Timestamped``. Hashing it would make
:meth:`DecisionContext.context_fingerprint` differ between two contexts that are
identical in every way that affects a result, purely because they were built a
microsecond apart — which is the opposite of what a provenance fingerprint is
for. A recorded fingerprint has to be reproducible on re-run or it cannot be
used to verify that a run was repeated faithfully.

The timestamps are not lost: they remain on the objects and in any manifest that
records them. They are simply not part of *this* identity.
"""


def _strip_volatile(payload: object) -> object:
    """Recursively drop clock-dependent keys so a fingerprint is reproducible."""
    if isinstance(payload, dict):
        return {
            key: _strip_volatile(value)
            for key, value in payload.items()
            if key not in _VOLATILE_FIELDS
        }
    if isinstance(payload, list):
        return [_strip_volatile(item) for item in payload]
    return payload


def _sha256(payload: object) -> str:
    """Hash a JSON-serialisable payload with sorted keys.

    Mirrors :func:`xg_alonso.contracts.evaluation._sha256` rather than inventing
    a second convention, so two hashes in this repository are never
    incomparable for a reason nobody wrote down.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _lock_shape(positions: frozenset[Position]) -> LockShape:
    if not positions:
        return LockShape.NONE
    if len(positions) > 1:
        return LockShape.MIXED
    only = next(iter(positions))
    return {
        Position.GKP: LockShape.GK,
        Position.DEF: LockShape.DEF,
        Position.MID: LockShape.MID,
        Position.FWD: LockShape.FWD,
    }[only]


def _budget_band(bank: TenthsOfMillion) -> BudgetBand:
    if bank < _BANK_BROKE:
        return BudgetBand.BROKE
    if bank < _BANK_THIN:
        return BudgetBand.THIN
    if bank < _BANK_COMFORTABLE:
        return BudgetBand.COMFORTABLE
    return BudgetBand.RICH


def _lock_pressure(sellable_share: float) -> LockPressure:
    if sellable_share > _SELLABLE_FREE:
        return LockPressure.FREE
    if sellable_share > _SELLABLE_LIGHT:
        return LockPressure.LIGHT
    if sellable_share > _SELLABLE_HEAVY:
        return LockPressure.HEAVY
    return LockPressure.FROZEN


def _transfer_freedom(squad: SquadState | None, constraints: ManagerConstraints) -> TransferFreedom:
    """Effective moves available, taking the tighter of squad state and cap.

    A manager with five free transfers who said "at most one change" has one
    move. Reading only ``free_transfers`` would describe a freedom they
    explicitly gave up.
    """
    available = squad.free_transfers if squad is not None else 1
    if constraints.max_transfers is not None:
        available = min(available, constraints.max_transfers)
    if available <= 0:
        return TransferFreedom.NONE
    if available == 1:
        return TransferFreedom.ONE
    if available < _MANY_TRANSFERS:
        return TransferFreedom.TWO
    return TransferFreedom.MANY


def _hit_appetite(constraints: ManagerConstraints) -> HitAppetite:
    if not constraints.allows_hits:
        return HitAppetite.NONE
    if constraints.max_points_hit < _MULTI_HIT:
        return HitAppetite.ONE_HIT
    return HitAppetite.MULTI


class DecisionContext(_Frozen):
    """Objective, constraints and beliefs, plus the situation they apply to.

    Deliberately **not** timestamped. Every other frozen contract carrying user
    intent inherits ``created_at``, but a fingerprint that changes every time the
    same request is rebuilt is not a fingerprint — and reproducibility of the
    conditioning is the property this whole layer is judged on.
    """

    bundle: ObjectiveBundle
    """Intent: objective (soft), constraints (hard), beliefs (uncertain)."""

    requirements: SquadRequirements = SquadRequirements()
    """Structural demands on the resulting squad — formation, must-start.

    Carried separately from ``bundle.constraints`` for the reason
    :class:`~xg_alonso.contracts.objective.CompiledIntent` already documents: the
    bundle restricts what the optimizer may *do*, these restrict what the squad
    must *contain*, and on a from-scratch build there is nothing to restrict
    doing.
    """

    squad: SquadState | None = None
    """The squad the constraints are relative to. ``None`` for a from-scratch
    build, which is a legitimate situation and not a missing value."""

    as_of_gameweek: GameweekId | None = None
    context_version: str = CONTEXT_VERSION

    # --- read-through accessors ---------------------------------------------
    #
    # So callers never reach past the umbrella into the bundle, and so the three
    # kinds of intent stay visibly distinct at every call site.

    @property
    def objective(self) -> ManagerObjective:
        """What to maximise. **Soft** — trades off against itself."""
        return self.bundle.objective

    @property
    def constraints(self) -> ManagerConstraints:
        """What is not negotiable. **Hard** — never traded against points."""
        return self.bundle.constraints

    @property
    def beliefs(self) -> tuple[UserBelief, ...]:
        """What the manager thinks they know. **Uncertain evidence, never fact.**"""
        return self.bundle.beliefs

    # --- derived situation ---------------------------------------------------

    @property
    def formation(self) -> str | None:
        """The required starting shape, or ``None`` if the manager named none."""
        shapes = self.requirements.of_kind(RequirementKind.FORMATION)
        if not shapes:
            return None
        return shapes[0].formation

    @property
    def sellable_share(self) -> float | None:
        """Fraction of the squad that may still be sold.

        ``None`` without a squad. The single scalar that best answers "how frozen
        is this manager", and the basis for :class:`LockPressure`.
        """
        if self.squad is None:
            return None
        held = len(self.squad.picks)
        if held == 0:
            return None
        frozen = self.locked_codes()
        return (held - len(frozen)) / held

    def locked_codes(self) -> frozenset[PlayerCode]:
        """Every player that may not leave, from all four ways of saying so.

        Explicit locks, locked positions, and protected squad areas all express
        the same thing about a squad member. Collapsing them here means callers
        cannot honour one and silently miss another — which is the failure mode
        that shows up as a "recommendation" to sell a player the manager
        protected.
        """
        locked = set(self.constraints.locked_players)
        if self.squad is None:
            return frozenset(locked)
        protected = self.constraints.locked_position_set()
        if protected:
            locked.update(
                pick.player_code for pick in self.squad.picks if pick.position in protected
            )
        return frozenset(locked)

    def locked_positions(self) -> frozenset[Position]:
        """Positions the locks actually occupy, resolved against the squad."""
        if self.squad is None:
            return self.constraints.locked_position_set()
        frozen = self.locked_codes()
        return frozenset(pick.position for pick in self.squad.picks if pick.player_code in frozen)

    # --- the three identities ------------------------------------------------

    def feasibility_digest(self) -> str:
        """Hash of *only* what changes which players are reachable.

        Two contexts sharing this digest face an identical buyable pool, so a
        pool computation may be reused between them. Two contexts differing only
        in ``max_points_hit`` share it — they can reach the same players, they
        merely differ in how many moves they will pay for.

        The squad is included, summarised: which players are held, what they can
        be sold for, and what is in the bank. The pool genuinely depends on all
        three, so excluding them would let a stale mask be reused against a
        different budget.
        """
        constraints = self.constraints
        payload: dict[str, Any] = {
            "version": self.context_version,
            "locked_players": sorted(int(code) for code in constraints.locked_players),
            "excluded_players": sorted(int(code) for code in constraints.excluded_players),
            "locked_positions": sorted(p.value for p in constraints.locked_position_set()),
            "minimum_bank": int(constraints.minimum_bank),
            "maximum_budget": (
                None if constraints.maximum_budget is None else int(constraints.maximum_budget)
            ),
            "required_teams": sorted(int(t) for t in constraints.required_teams),
            "excluded_teams": sorted(int(t) for t in constraints.excluded_teams),
            "minimum_players_by_team": sorted(
                (int(q.team_id), q.count) for q in constraints.minimum_players_by_team
            ),
            "maximum_players_by_team": sorted(
                (int(q.team_id), q.count) for q in constraints.maximum_players_by_team
            ),
            "formation": self.formation,
        }
        if self.squad is not None:
            payload["squad"] = {
                "bank": int(self.squad.bank),
                "picks": sorted(
                    (int(p.player_code), int(p.selling_price), int(p.team_id), p.position.value)
                    for p in self.squad.picks
                ),
            }
        return _sha256(payload)

    def bucket(self) -> ContextBucket:
        """Canonicalise into the bounded vocabulary used for cache keys."""
        squad = self.squad
        share = self.sellable_share

        club_pressure = ClubPressure.UNKNOWN
        if squad is not None and squad.picks:
            counts = Counter(pick.team_id for pick in squad.picks)
            busiest = max(counts.values())
            club_pressure = ClubPressure.TIGHT if busiest >= _CLUB_TIGHT_AT else ClubPressure.SLACK

        return ContextBucket(
            objective_id=self.objective.id,
            transfer_freedom=_transfer_freedom(squad, self.constraints),
            hit_appetite=_hit_appetite(self.constraints),
            lock_pressure=(LockPressure.UNKNOWN if share is None else _lock_pressure(share)),
            lock_shape=_lock_shape(self.locked_positions()),
            club_pressure=club_pressure,
            budget_band=(BudgetBand.UNKNOWN if squad is None else _budget_band(squad.bank)),
            formation=self.formation or _UNCONSTRAINED_FORMATION,
            belief_load=BeliefLoad.SOME if self.beliefs else BeliefLoad.NONE,
        )

    def context_key(self) -> str:
        """Bounded cache key: objective identity, version, and bucket.

        The objective id leads so a cache directory sorts by objective, which is
        the axis a human browsing it is actually looking for.
        """
        bucket = self.bucket()
        digest = hashlib.sha256(bucket.key().encode("utf-8")).hexdigest()[:10]
        return f"{self.objective.id}|{self.context_version}|{digest}"

    def context_fingerprint(self) -> str:
        """Full hash of the context, for provenance only.

        **Never a cache key.** A run is reproducible because this is recorded,
        not because anything was stored under it — that distinction is what keeps
        per-manager conditioning from becoming a per-manager artifact store.

        Clock-dependent fields are stripped first (see :data:`_VOLATILE_FIELDS`),
        so rebuilding the same request tomorrow reproduces today's fingerprint.
        """
        return _sha256(
            _strip_volatile(
                {
                    "version": self.context_version,
                    "bundle": self.bundle.model_dump(mode="json"),
                    "requirements": self.requirements.model_dump(mode="json"),
                    "squad": (None if self.squad is None else self.squad.model_dump(mode="json")),
                    "as_of_gameweek": (
                        None if self.as_of_gameweek is None else int(self.as_of_gameweek)
                    ),
                }
            )
        )
