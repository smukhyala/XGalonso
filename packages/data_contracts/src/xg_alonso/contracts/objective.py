"""What a manager is actually trying to achieve.

Every other package in this repository answers one question: *which player scores
the most points?* That question has a single right answer, so a single global
accuracy metric is enough to judge it.

This module encodes the observation that the question is wrong. A manager forty
points behind in a mini-league and a manager protecting a top-1% overall rank are
not looking for the same player, are not served by the same features, and are not
well described by the same notion of "similar player". Both are rational. What
differs is the objective.

Three things are modelled, and the separation between them is load-bearing:

- :class:`ManagerObjective` — what to maximise. Soft. Trades off against itself.
- :class:`ManagerConstraints` — what is not negotiable. Hard. Never traded off.
- :class:`UserBelief` — what the manager thinks they know that the model does not.
  **Uncertain evidence, never fact.**

Confusing the first two is how an optimizer comes to sell a player the user said
to keep, having decided the points were worth it. They are not: a constraint is
the user telling the system which question to answer, and an optimizer that
overrides one has answered a different question.

Confusing the second and third is worse. "Keep Haaland" is a constraint. "Haaland
will score this week" is a belief, and a belief that silently became a constraint
would let a hunch quietly rewrite the evidence it was supposed to be weighed
against. :mod:`xg_alonso.prediction.beliefs` therefore returns the raw and the
adjusted prediction side by side, and never overwrites one with the other.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.provenance import utc_now
from xg_alonso.contracts.squad import ChipStatus

__all__ = [
    "BeliefEntity",
    "BeliefProposition",
    "ChipIntent",
    "CompiledIntent",
    "FeatureDiscoveryRequest",
    "FieldConfidence",
    "ManagerConstraints",
    "ManagerObjective",
    "ObjectiveBundle",
    "OwnershipPreference",
    "ParseSource",
    "PrimaryMetric",
    "Requirement",
    "RequirementKind",
    "RiskPreference",
    "SquadArea",
    "SquadRequirements",
    "TeamQuota",
    "UtilityWeights",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Timestamped(_Frozen):
    """Frozen, plus the repo-wide rule that timestamps carry a timezone.

    Naive datetimes compare incorrectly against the UTC data every other layer
    produces, and the comparison that goes wrong is the availability check that
    prevents leakage — so this is enforced rather than trusted.
    """

    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value


# --- objective ---------------------------------------------------------------


class PrimaryMetric(StrEnum):
    """What an objective maximises.

    Each value has exactly one scoring implementation, registered in
    :mod:`xg_alonso.optimization.objective`. Adding a metric means adding a
    scorer, not editing the search engine — which is what keeps a new objective
    from requiring a change to code that already works.
    """

    EXPECTED_POINTS = "expected_points"
    """Total expected points over the horizon. The default question."""

    EXPECTED_RANK_GAIN = "expected_rank_gain"
    """**A proxy, not a rank.**

    True overall rank needs the global distribution of every manager's squad,
    which the public API does not publish and this repository does not ingest.
    What is computed instead is an ownership-weighted points differential against
    the field: points scored by players the field does not own move a manager up,
    points scored by template players do not. That is the mechanism rank-chasing
    actually exploits, but it is a proxy and is labelled one everywhere.
    """

    DOWNSIDE_PROTECTION = "downside_protection"
    """Maximise the lower quantile of the outcome rather than the mean."""

    TEAM_VALUE_GROWTH = "team_value_growth"
    """**Transfer momentum, not a price forecast.**

    Decision D11 defers the price model, and no current-season price data exists
    at GW1. This scores net transfer flow relative to ownership, which is the
    published leading indicator of a price rise, and does not claim to predict
    the rise itself.
    """

    CAPTAINCY_UPSIDE = "captaincy_upside"
    """The ceiling of the armband, conditioned on actually playing."""

    DIFFERENTIAL_YIELD = "differential_yield"
    """Expected points weighted by how few managers own the player."""

    TRANSFER_FLEXIBILITY = "transfer_flexibility"
    """Preserve future optionality: bank, free transfers, squad liquidity."""


class RiskPreference(StrEnum):
    """How variance is traded against expectation."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

    @property
    def variance_sign(self) -> float:
        """Multiplier on the uncertainty penalty.

        Aggressive is **negative**: a manager chasing a deficit wants variance,
        because a certain average keeps them exactly as far behind as they
        started. Treating risk aversion as universally positive is the modelling
        error that makes every tool recommend the same template squad.
        """
        if self is RiskPreference.CONSERVATIVE:
            return 1.0
        if self is RiskPreference.BALANCED:
            return 0.35
        return -0.5


class OwnershipPreference(StrEnum):
    """Where a manager wants to sit relative to the crowd."""

    TEMPLATE = "template"
    """Own what everyone owns. Protects a rank; cannot improve one."""

    NEUTRAL = "neutral"

    DIFFERENTIAL = "differential"
    """Own what few own. The only way to gain rank, and the only way to lose it."""

    @property
    def ownership_sign(self) -> float:
        """Sign applied to a player's ownership share when scoring."""
        if self is OwnershipPreference.TEMPLATE:
            return 1.0
        if self is OwnershipPreference.NEUTRAL:
            return 0.0
        return -1.0


class UtilityWeights(_Frozen):
    """Weights in the feature-utility score.

    Typed rather than a free-form mapping. A dictionary would accept
    ``w_predicton`` without complaint and silently weight it zero, and a
    mis-typed weight that reads as "this feature is worthless" is indis-
    tinguishable from a measurement.

    The score itself::

        utility =  w_prediction      * predictive_gain
                 + w_decision        * decision_quality_gain
                 + w_objective       * objective_specific_gain
                 + w_stability       * temporal_stability
                 + w_complementarity * conditional_incremental_gain
                 - w_complexity      * complexity
                 - w_missingness     * missingness
                 - w_turnover        * turnover
                 - w_leakage         * leakage_risk
    """

    prediction: float = Field(default=1.0, ge=0.0)
    decision: float = Field(default=1.0, ge=0.0)
    objective: float = Field(default=1.0, ge=0.0)
    stability: float = Field(default=0.5, ge=0.0)
    complementarity: float = Field(default=1.0, ge=0.0)
    complexity: float = Field(default=0.1, ge=0.0)
    missingness: float = Field(default=0.5, ge=0.0)
    turnover: float = Field(default=0.2, ge=0.0)
    leakage: float = Field(default=10.0, ge=0.0)
    """Deliberately an order of magnitude above the others.

    A leaking feature is not a slightly worse feature; it is a feature whose
    measured value is fiction. The weight is set so no combination of the other
    terms can carry a flagged program to acceptance.
    """


class ManagerObjective(_Timestamped):
    """A complete statement of what to maximise, and how to trade it off."""

    id: str = Field(min_length=1, description="Stable identifier, e.g. 'mini_league_chase'")
    name: str = Field(min_length=1, description="Human-readable name")
    version: str = Field(default="1", description="Bumped when weights change")

    primary_metric: PrimaryMetric
    secondary_metrics: tuple[PrimaryMetric, ...] = ()

    risk_preference: RiskPreference = RiskPreference.BALANCED
    planning_horizon: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Gameweeks to optimise over. Capped at 10 to match "
            "PlayerPrediction.horizon_gameweeks, so an objective can never ask "
            "for a horizon the prediction contract cannot express."
        ),
    )
    ownership_preference: OwnershipPreference = OwnershipPreference.NEUTRAL

    team_value_weight: float = Field(default=0.0, ge=0.0)
    transfer_cost_weight: float = Field(default=1.0, ge=0.0)
    captaincy_weight: float = Field(default=1.0, ge=0.0)
    uncertainty_penalty: float = Field(
        default=0.25,
        ge=0.0,
        description=(
            "Magnitude of the variance penalty. The *sign* comes from "
            "risk_preference, so an aggressive objective rewards variance rather "
            "than merely penalising it less."
        ),
    )

    objective_weights: UtilityWeights = Field(default_factory=UtilityWeights)

    @model_validator(mode="after")
    def _primary_is_not_repeated(self) -> ManagerObjective:
        if self.primary_metric in self.secondary_metrics:
            raise ValueError(
                f"{self.primary_metric.value} is both the primary metric and a "
                "secondary one; counting it twice silently doubles its weight"
            )
        return self

    @property
    def signed_uncertainty_penalty(self) -> float:
        """The variance term the optimizer actually applies, sign included."""
        return self.uncertainty_penalty * self.risk_preference.variance_sign

    def cache_key(self) -> str:
        """Identity for caching and provenance: id plus version."""
        return f"{self.id}@{self.version}"


# --- constraints -------------------------------------------------------------


class SquadArea(StrEnum):
    """A protected region of the squad.

    Distinct from ``locked_positions``, which names positions. An *area* can be
    a slot range — "do not touch my bench" is not a statement about positions —
    so the two cannot be collapsed without losing one of them.
    """

    GOALKEEPER = "goalkeeper"
    DEFENCE = "defence"
    MIDFIELD = "midfield"
    ATTACK = "attack"
    STARTING_XI = "starting_xi"
    BENCH = "bench"

    def positions(self) -> tuple[Position, ...]:
        """Positions this area covers. Empty for slot-based areas.

        ``DEFENCE`` is defenders only, **not** defenders plus the goalkeeper.
        In football commentary "the defence" usually includes the keeper, and
        clean sheets are certainly a joint property of both — but this mapping
        is used to decide which players a manager may not transfer, and someone
        who says "don't change my defence" is talking about their back line. A
        constraint that silently also froze the goalkeeper would block a
        legitimate transfer for a reason the user never gave, and over-locking
        is invisible in a way under-locking is not: the move simply never
        appears. A manager who means both selects both.
        """
        mapping: dict[str, tuple[Position, ...]] = {
            "goalkeeper": (Position.GKP,),
            "defence": (Position.DEF,),
            "midfield": (Position.MID,),
            "attack": (Position.FWD,),
        }
        return mapping.get(self.value, ())


class TeamQuota(_Frozen):
    """A per-club headcount floor or ceiling."""

    team_id: TeamId
    count: int = Field(ge=0, le=15)


class RequirementKind(StrEnum):
    """What a single squad requirement asks for.

    Named separately from the constraint fields they compile into, because the
    infeasibility report has to say *which requirement* had to give, in the
    words the manager used. A bare "no legal squad exists" is the failure mode
    this vocabulary exists to prevent.
    """

    MUST_START = "must_start"
    """In the starting eleven, not merely in the fifteen."""

    MUST_INCLUDE = "must_include"
    """In the fifteen. The bench is an acceptable answer."""

    MUST_EXCLUDE = "must_exclude"
    MUST_CAPTAIN = "must_captain"
    CLUB_FLOOR = "club_floor"
    CLUB_CEILING = "club_ceiling"
    FORMATION = "formation"
    BANK_FLOOR = "bank_floor"


class Requirement(_Frozen):
    """One thing the manager asked for, in their words and in solver terms.

    **Start and include are different requirements and the difference matters.**
    "I want Haaland" is satisfied by Haaland on the bench, which is not what
    anybody means. "I want Haaland starting" is a constraint on the eleven. The
    optimizer has separate variables for both, so conflating them here would
    throw away a distinction the solver can already express.

    ``priority`` orders *relaxation*, not importance to the solve. Every
    requirement is hard when a legal squad exists; priority only decides what
    gives first when none does, so a manager is told "your three-Arsenal floor
    is what broke this" rather than "infeasible".
    """

    kind: RequirementKind
    label: str = Field(description="What the manager asked for, in their words")

    players: tuple[PlayerCode, ...] = ()
    team_id: TeamId | None = None
    count: int | None = Field(default=None, ge=0, le=15)
    formation: str | None = Field(
        default=None, description="Starting shape as DEF-MID-FWD, e.g. '3-5-2'"
    )
    amount: TenthsOfMillion | None = None

    priority: int = Field(
        default=0,
        description=(
            "Higher survives longer when the set is infeasible. Ties break on "
            "declaration order, so an unranked set relaxes back-to-front — the "
            "most recently added requirement is the first to give."
        ),
    )

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> Requirement:
        """Reject a requirement whose payload cannot express its kind.

        A `CLUB_FLOOR` with no team, or a `MUST_START` with no players, compiles
        to a constraint row that silently binds nothing. That is worse than an
        error: the squad comes back looking like it honoured the request.
        """
        player_kinds = {
            RequirementKind.MUST_START,
            RequirementKind.MUST_INCLUDE,
            RequirementKind.MUST_EXCLUDE,
            RequirementKind.MUST_CAPTAIN,
        }
        club_kinds = {RequirementKind.CLUB_FLOOR, RequirementKind.CLUB_CEILING}

        if self.kind in player_kinds and not self.players:
            raise ValueError(f"{self.kind.value} requires at least one player")
        if self.kind is RequirementKind.MUST_CAPTAIN and len(self.players) != 1:
            raise ValueError("a squad has exactly one captain")
        if self.kind in club_kinds and (self.team_id is None or self.count is None):
            raise ValueError(f"{self.kind.value} requires a team_id and a count")
        if self.kind is RequirementKind.FORMATION and not self.formation:
            raise ValueError("formation requires a shape like '3-5-2'")
        if self.kind is RequirementKind.BANK_FLOOR and self.amount is None:
            raise ValueError("bank_floor requires an amount")
        return self

    def formation_counts(self) -> tuple[int, int, int]:
        """Parse ``'3-5-2'`` into defender, midfielder and forward counts.

        Raises:
            ValueError: on a shape that is not three numbers, or one that does
                not field ten outfielders alongside the compulsory keeper.
        """
        if not self.formation:
            raise ValueError("no formation on this requirement")
        parts = self.formation.split("-")
        expected = 3
        if len(parts) != expected or not all(part.strip().isdigit() for part in parts):
            raise ValueError(f"expected a shape like '3-5-2', got {self.formation!r}")
        counts = tuple(int(part) for part in parts)
        outfield = 10
        if sum(counts) != outfield:
            raise ValueError(
                f"formation {self.formation!r} fields {sum(counts)} outfielders; "
                "eleven players means one keeper and ten outfield"
            )
        return counts[0], counts[1], counts[2]


class ChipIntent(_Frozen):
    """A declared plan for one chip.

    Per decision D5 the MVP models chip *state* but not chip *logic*. This is
    carried and validated so a plan is not lost, and is deliberately not
    optimised.
    """

    chip: str = Field(
        min_length=1, description="wildcard | free_hit | bench_boost | triple_captain"
    )
    gameweek: GameweekId
    status: ChipStatus = ChipStatus.AVAILABLE


class ManagerConstraints(_Frozen):
    """What the optimizer may not do, whatever the points say.

    Every field here is a hard bound. None of them is traded against expected
    points, because a user who says "keep Haaland" has not offered an exchange
    rate — they have restated the problem.
    """

    locked_players: tuple[PlayerCode, ...] = ()
    """Must remain in the squad. Never sold, whatever the alternative scores."""

    excluded_players: tuple[PlayerCode, ...] = ()
    """Must never be bought."""

    locked_positions: tuple[Position, ...] = ()
    """No player in these positions may be transferred out."""

    protected_squad_areas: tuple[SquadArea, ...] = ()
    """Squad regions that may not be disturbed, including slot-based ones."""

    max_transfers: int | None = Field(default=None, ge=0)
    max_points_hit: int = Field(
        default=0,
        ge=0,
        description="Points the manager will pay in hits. Zero forbids hits entirely.",
    )

    minimum_bank: TenthsOfMillion = TenthsOfMillion(0)
    maximum_budget: TenthsOfMillion | None = None

    required_teams: tuple[TeamId, ...] = ()
    excluded_teams: tuple[TeamId, ...] = ()
    minimum_players_by_team: tuple[TeamQuota, ...] = ()
    maximum_players_by_team: tuple[TeamQuota, ...] = ()

    required_features: tuple[str, ...] = ()
    """Features that must stay in the model, whatever their measured value.

    A required feature is never dropped by selection. It is also the anchor for
    complementary discovery: "keep xG and find what adds to it" is expressed
    here.
    """

    excluded_features: tuple[str, ...] = ()

    chip_plan: tuple[ChipIntent, ...] = ()

    @model_validator(mode="after")
    def _check_coherent(self) -> ManagerConstraints:
        overlap = set(self.locked_players) & set(self.excluded_players)
        if overlap:
            raise ValueError(
                f"players {sorted(overlap)} are both locked and excluded; the "
                "constraint set has no solution and would fail silently as an "
                "empty candidate list"
            )
        team_overlap = set(self.required_teams) & set(self.excluded_teams)
        if team_overlap:
            raise ValueError(f"teams {sorted(team_overlap)} are both required and excluded")
        feature_overlap = set(self.required_features) & set(self.excluded_features)
        if feature_overlap:
            raise ValueError(f"features {sorted(feature_overlap)} are both required and excluded")
        for floor in self.minimum_players_by_team:
            for ceiling in self.maximum_players_by_team:
                if floor.team_id == ceiling.team_id and floor.count > ceiling.count:
                    raise ValueError(
                        f"club {floor.team_id}: minimum {floor.count} exceeds maximum "
                        f"{ceiling.count}"
                    )
        return self

    @property
    def allows_hits(self) -> bool:
        return self.max_points_hit > 0

    def locked_position_set(self) -> frozenset[Position]:
        """Positions no player may leave, from both locks and protected areas."""
        positions = set(self.locked_positions)
        for area in self.protected_squad_areas:
            positions.update(area.positions())
        return frozenset(positions)


# --- beliefs -----------------------------------------------------------------


class SquadRequirements(_Frozen):
    """Everything a manager asked of a squad built from scratch.

    Composed from :class:`ManagerConstraints` rather than restating it. The two
    read differently and the difference is deliberate: on the *transfer* path a
    locked player is one that may not be **sold**, while on the *from scratch*
    path there is nothing to sell, so the same field means the player must be
    **bought**. Both compile to "he is in the fifteen"; only the story differs.

    The eleven-level requirements have no transfer-path equivalent at all, which
    is why they live here instead of being pushed down into the shared contract.
    """

    constraints: ManagerConstraints = ManagerConstraints()
    requirements: tuple[Requirement, ...] = ()

    def of_kind(self, kind: RequirementKind) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.kind is kind)

    def relaxation_order(self) -> tuple[Requirement, ...]:
        """Requirements ordered by what should give first when none can hold.

        Lowest priority first, and within a priority the most recently declared
        first — a manager who adds a requirement and immediately sees the set
        break should have that one named, not one they set five minutes ago.
        """
        indexed = list(enumerate(self.requirements))
        indexed.sort(key=lambda pair: (pair[1].priority, -pair[0]))
        return tuple(requirement for _, requirement in indexed)


class BeliefEntity(StrEnum):
    PLAYER = "player"
    TEAM = "team"


class BeliefProposition(StrEnum):
    """What the manager claims. Each maps to one bounded adjustment.

    Deliberately a closed vocabulary. Free text would let a belief mean anything,
    and "anything" cannot be bounded — which is the property that stops a hunch
    becoming an unbounded multiplier on a prediction.
    """

    OUTPERFORM_MODEL = "outperform_model"
    UNDERPERFORM_MODEL = "underperform_model"
    WILL_START = "will_start"
    WILL_NOT_START = "will_not_start"
    WILL_RETURN = "will_return"
    """Will score or assist. Lifts the attacking components specifically."""
    CLEAN_SHEET = "clean_sheet"

    @property
    def direction(self) -> float:
        """+1 when the belief argues for the player, -1 when against."""
        if self in {
            BeliefProposition.UNDERPERFORM_MODEL,
            BeliefProposition.WILL_NOT_START,
        }:
            return -1.0
        return 1.0


class UserBelief(_Timestamped):
    """One piece of manager intuition, held as uncertain evidence.

    ``confidence`` is not a probability of being right in any calibrated sense —
    nobody has measured that — it is a stated strength, and it is used only to
    scale a **bounded** adjustment. That bound is what keeps a belief from
    becoming an assertion.
    """

    entity_type: BeliefEntity
    entity_id: int = Field(description="PlayerCode or TeamId, per entity_type")
    proposition: BeliefProposition
    confidence: float = Field(ge=0.0, le=1.0)
    affected_gameweeks: tuple[GameweekId, ...] = ()
    rationale: str = Field(default="", description="Why the manager believes it")
    decay_rate: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "How fast the belief loses force per gameweek beyond the first "
            "affected one. 1.0 means it applies undiminished to every affected "
            "gameweek and vanishes outside them; larger values decay faster."
        ),
    )

    @model_validator(mode="after")
    def _check_scope(self) -> UserBelief:
        if self.proposition is BeliefProposition.CLEAN_SHEET and (
            self.entity_type is BeliefEntity.PLAYER
        ):
            # Allowed, but it means "this player's team keeps a clean sheet",
            # which is a team claim wearing a player's id. Left legal because a
            # user thinks in players; recorded here so the semantics are stated.
            return self
        return self

    def applies_to(self, gameweek: GameweekId) -> bool:
        """Whether this belief has any force in a gameweek.

        An empty ``affected_gameweeks`` means "the next one, whenever that is",
        so the belief applies everywhere rather than nowhere. The alternative —
        treating empty as no gameweeks — would silently discard every belief a
        user stated without a week attached.
        """
        return not self.affected_gameweeks or gameweek in self.affected_gameweeks

    def weight_at(self, gameweek: GameweekId) -> float:
        """Confidence remaining in this gameweek, after decay.

        Decay is measured from the earliest affected gameweek. A belief about a
        specific fixture should not still be at full strength three weeks later,
        because the reason for it — a matchup, a piece of team news — has by
        then been superseded by evidence the model can see for itself.
        """
        if not self.applies_to(gameweek):
            return 0.0
        if not self.affected_gameweeks:
            return self.confidence
        distance = max(0, int(gameweek) - min(int(g) for g in self.affected_gameweeks))
        if distance == 0:
            return self.confidence
        return self.confidence / (1.0 + self.decay_rate * distance)


# --- the compiled whole ------------------------------------------------------


class FeatureDiscoveryRequest(_Frozen):
    """What the manager wants the discovery loop to look for."""

    required_features: tuple[str, ...] = ()
    """Anchors. Discovery measures value *conditional on* these."""

    excluded_features: tuple[str, ...] = ()

    complement_targets: tuple[str, ...] = ()
    """Features to find complements for. Usually equals ``required_features``."""

    emphasis: tuple[str, ...] = ()
    """Free-form hints — 'upside', 'differential', 'minutes' — used to bias
    hypothesis generation toward some mechanisms. Advisory, never binding."""

    max_hypotheses: int = Field(default=8, ge=1, le=64)
    """Proposals per iteration. Bounded because an unbounded generator produces
    syntactic variations of one idea rather than diverse ideas."""


class ObjectiveBundle(_Frozen):
    """Everything a manager's intent resolves to. The unit passed downstream."""

    objective: ManagerObjective
    constraints: ManagerConstraints = Field(default_factory=ManagerConstraints)
    beliefs: tuple[UserBelief, ...] = ()
    discovery: FeatureDiscoveryRequest = Field(default_factory=FeatureDiscoveryRequest)

    @model_validator(mode="after")
    def _features_agree(self) -> ObjectiveBundle:
        """The two places features can be required must not disagree.

        ``ManagerConstraints.required_features`` is the hard bound; the discovery
        request carries the same list so a search can be run without the full
        constraint set. Letting them drift would mean a feature required by the
        optimizer and ignored by the search, which is exactly the silent
        inconsistency the contracts layer exists to prevent.
        """
        constrained = set(self.constraints.required_features)
        requested = set(self.discovery.required_features)
        if constrained and requested and constrained != requested:
            raise ValueError(
                "constraints.required_features and discovery.required_features "
                f"disagree: {sorted(constrained)} vs {sorted(requested)}"
            )
        return self


class ParseSource(StrEnum):
    """How a field in a compiled intent got its value."""

    EXPLICIT = "explicit"
    """Stated in the input and matched deterministically."""

    INFERRED = "inferred"
    """Derived from a phrase whose meaning had to be interpreted."""

    DEFAULT = "default"
    """Nothing in the input spoke to this field."""

    PRESET = "preset"
    """Taken from a named preset the user selected."""


class FieldConfidence(_Frozen):
    """How sure the compiler is about one field, and what it read.

    ``evidence`` carries the substring that produced the value. Showing it is
    what makes a parse reviewable: a user can see that "aggressive" came from
    their own word and that a three-gameweek horizon came from "next three
    gameweeks", rather than being asked to trust a struct.
    """

    field: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: ParseSource
    evidence: str = ""


class CompiledIntent(_Frozen):
    """A parsed manager request, with everything needed to review it.

    Deliberately *not* executable on its own. The product shows this to the user,
    lets them edit the structured fields, and only then runs the bundle — because
    natural language is ambiguous and a system that acts on its own guess without
    showing it will eventually act on a wrong one silently.
    """

    bundle: ObjectiveBundle
    requirements: SquadRequirements = SquadRequirements()
    """Structural demands on a squad built from scratch.

    Separate from ``bundle.constraints`` because they answer a different
    question: the bundle says what the optimizer may not *do*, while these say
    what the resulting squad must *contain*. A manager asking for Haaland in the
    starting eleven has not restricted a transfer — there is no squad to
    transfer from yet.
    """

    confidences: tuple[FieldConfidence, ...] = ()
    unparsed: tuple[str, ...] = ()
    """Phrases the compiler did not understand. Surfaced rather than dropped: a
    request that was 80% understood should say which 20% was not."""

    raw_text: str = ""

    @property
    def overall_confidence(self) -> float:
        """Mean confidence across fields the input actually spoke to.

        Defaults are excluded. Including them would let a request that said
        almost nothing score highly, because every untouched field is trivially
        "certain" — which inverts the meaning of the number.
        """
        stated = [c.confidence for c in self.confidences if c.source is not ParseSource.DEFAULT]
        if not stated:
            return 0.0
        return sum(stated) / len(stated)

    def field_confidence(self, field: str) -> FieldConfidence | None:
        return next((c for c in self.confidences if c.field == field), None)
