"""HTTP surface over the decision system.

The second surface after the CLI, per decision D4. It is deliberately thin: the
CLI's composition root already wires ingestion, features, prediction, domain
rules and optimization together, so this layer parses requests, calls that, and
shapes responses. Any logic that appears here that is not request handling has
been put in the wrong place.

**Every response carries provenance.** A recommendation is only trustworthy if
you can say which model produced it, over which features, from data cut off
when. Those fields are on the wire, not just in the logs.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from xg_alonso.api.service import DecisionService, ServiceConfig

app = FastAPI(
    title="XG Alonso",
    description="ML-powered Fantasy Premier League decisions.",
    version="0.1.0",
)

# The web app is served from a different origin in development. Locked to
# localhost because D1 keeps everything local — this is not a public API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _service() -> DecisionService:
    """Build the service once.

    Loading four seasons of history and a fitted model takes seconds, so it is
    done at first request and reused. The cache is keyed on nothing because the
    configuration comes from the environment and does not change per request.
    """
    return DecisionService(ServiceConfig())


ServiceDep = Annotated[DecisionService, Depends(_service)]


class Provenance(BaseModel):
    """What produced a response, so a number can be traced back to its inputs."""

    model_name: str
    model_version: str
    feature_set_version: str
    data_cutoff: datetime
    generated_at: datetime
    run_id: str


class HealthResponse(BaseModel):
    status: str
    season: str
    next_gameweek: int
    deadline: datetime
    players_loaded: int
    history_rows: int
    model_loaded: bool
    stale: bool = Field(
        description="True when the deadline being predicted has already passed, "
        "which means the stored snapshot is out of date."
    )


class HistoryNoteOut(BaseModel):
    """A checkable fact about what this player has done in this situation.

    Distinct from a `ReasonOut`: a reason is derived from the model, this is a
    retrieval from matches that were played. It is the one part of the
    explanation a user can verify against a scoreboard.
    """

    kind: str
    text: str
    is_positive: bool


class DerivationLineOut(BaseModel):
    """One scoring term, with the arithmetic that produced it."""

    component: str
    expectation: float
    unit: str
    rate: float
    points: float
    note: str = ""
    sentence: str


class GameweekProjectionOut(BaseModel):
    """What he is projected for in one upcoming gameweek, and who he faces."""

    gameweek: int
    opponent: str | None
    is_home: bool | None
    expected_points: float
    weight: float = Field(
        description=(
            "How much this week counts toward the horizon value. Later weeks "
            "are discounted, because a free transfer arrives every week and a "
            "projection five weeks out is a longer extrapolation."
        )
    )


class PlayerSummary(BaseModel):
    player_code: int
    name: str
    position: str
    team_id: int
    price: int = Field(description="Tenths of a million")
    status: str | None
    expected_points: float
    expected_points_sd: float
    p_start: float
    expected_minutes: float
    history: list[HistoryNoteOut] = Field(
        default_factory=list,
        description="What he has done against this opponent, and in this gameweek.",
    )
    derivation: list[DerivationLineOut] = Field(
        default_factory=list,
        description="How the projection was assembled, largest contribution first.",
    )
    horizon: list[GameweekProjectionOut] = Field(
        default_factory=list,
        description="The next few gameweeks, each with the club he faces.",
    )
    horizon_total: float | None = Field(
        default=None,
        description="Discounted value across the horizon, comparable to a single gameweek.",
    )


class SquadPlayer(PlayerSummary):
    squad_slot: int
    is_captain: bool
    is_vice_captain: bool
    is_starter: bool
    selling_price: int
    purchase_price: int


class SquadResponse(BaseModel):
    entry_id: int
    gameweek: int
    formation: str
    squad_value: int
    bank: int
    free_transfers: int
    projected_points: float
    prices_assumed: bool = Field(
        description="True when purchase prices could not be reconstructed, so "
        "the budget shown is a lower bound."
    )
    players: list[SquadPlayer]
    provenance: Provenance


class ReasonOut(BaseModel):
    code: str
    text: str
    subject: int
    subject_name: str = Field(
        description=(
            "Who the reason is about. Without it the screen rendered "
            "'minutes look secure' and 'minutes are a concern' as an "
            "unattributed list that read as self-contradictory."
        )
    )
    polarity: str
    weight: float


class FeatureValueOut(BaseModel):
    """One panel feature, with the rank that makes the number mean something."""

    name: str
    label: str
    family: str
    value: float | None
    percentile: float | None
    higher_is_better: bool


class BreakdownOut(BaseModel):
    """How a projection was assembled. Sums to the total by construction."""

    appearance: float
    goals: float
    assists: float
    clean_sheets: float
    goals_conceded: float
    saves: float
    cards: float
    defensive_contribution: float
    bonus: float
    total: float


class TransferOptionOut(BaseModel):
    player_out: int
    player_out_name: str
    player_in: int
    player_in_name: str
    position: str
    selling_price: int
    purchase_price: int
    gross_gain: float
    net_gain: float
    hit_cost: int
    risk_penalty: float
    bank_after: int
    reasons: list[ReasonOut]
    history_in: list[HistoryNoteOut] = Field(default_factory=list)
    history_out: list[HistoryNoteOut] = Field(default_factory=list)


class ComparableOut(BaseModel):
    player_code: int
    name: str
    expected_points: float
    price: int | None


class ArchetypeOut(BaseModel):
    """What kind of player this is, and who else is that kind."""

    label: str
    size: int
    rank_within: int = Field(
        description="His place by projected points inside his archetype, 1 being highest."
    )
    comparables: list[ComparableOut]
    caveat: str = Field(
        description=(
            "Why the within-archetype rank is not itself the argument for "
            "picking him. Archetypes are clustered on style and output "
            "together, so a cluster is partly defined by how good its members "
            "are and 'best in his cluster' would be close to circular."
        )
    )


class SwapOut(BaseModel):
    position: str
    player_in: int | None
    player_in_name: str | None
    player_out: int | None
    player_out_name: str | None
    points_in: float
    points_out: float
    delta: float
    is_like_for_like: bool


class LineupComparisonOut(BaseModel):
    """Why the proposed eleven beats the one currently set, term by term.

    The three deltas sum to `total_delta` by construction, and `shape_delta` is
    the named residual rather than a fudge — a decomposition whose last term is
    defined as "everything else" proves nothing unless it is shown.
    """

    yours_points: float
    ours_points: float
    total_delta: float
    swap_delta: float
    captain_delta: float
    shape_delta: float
    yours_formation: str
    ours_formation: str
    yours_captain_name: str | None
    ours_captain_name: str | None
    swaps: list[SwapOut]
    is_identical: bool
    yours_is_better: bool


class PlayerExplanationOut(BaseModel):
    """Everything the system can honestly say about one squad member."""

    player_code: int
    name: str
    position: str
    expected_points: float
    breakdown: BreakdownOut
    evidence: list[FeatureValueOut]
    reasons: list[ReasonOut]
    is_starter: bool
    start_margin: float = Field(
        description=(
            "Points at stake in the start-or-bench call, measured by "
            "re-selecting the XI rather than by comparing raw projections."
        )
    )
    forced_by_quota: bool = Field(
        description="Whether a positional minimum, not his projection, put him in the XI."
    )
    legal_replacements: int
    replacements: list[TransferOptionOut]
    no_replacement_reasons: list[ReasonOut]
    archetype: ArchetypeOut | None = None
    derivation: list[DerivationLineOut] = Field(
        default_factory=list,
        description="How the projection was assembled, largest contribution first.",
    )
    history: list[HistoryNoteOut] = Field(
        default_factory=list,
        description="Checkable facts about this situation, not model output.",
    )
    derivation_reconciles: bool = Field(
        default=True,
        description=(
            "Whether the lines sum to the total they explain. False should be "
            "impossible; it is reported rather than asserted because an "
            "explanation that silently disagrees with its number is worse than "
            "one that admits it."
        ),
    )


class RecommendationResponse(BaseModel):
    entry_id: int
    gameweek: int
    is_hold: bool
    player_out: PlayerSummary | None
    player_in: PlayerSummary | None
    hit_cost: int
    bank_after: int
    projected_hold: float
    projected_after: float
    expected_gain: float
    risk: float
    reasons: list[ReasonOut]
    alternatives: list[TransferOptionOut] = Field(
        default_factory=list, description="Runners-up, best net gain first"
    )
    players: list[PlayerExplanationOut] = Field(
        default_factory=list, description="Per-player justification, in squad order"
    )
    candidates_considered: int = 0
    legal_moves: int = 0
    lineup: LineupComparisonOut | None = Field(
        default=None,
        description="The eleven you have set against the eleven we would field.",
    )
    provenance: Provenance


class SquadBuildResponse(BaseModel):
    """A squad built from scratch — the gameweek-1 answer.

    Separate from `SquadResponse` because it carries the justification for each
    pick. At gameweek 1 there is no squad to transfer from and transfers are
    unlimited, so "which single move is best" is the wrong question entirely;
    the question is which fifteen, and why each of them over the alternatives.
    """

    gameweek: int
    formation: str
    squad_value: int
    bank: int
    projected_points: float
    players: list[SquadPlayer]
    explanations: list[PlayerExplanationOut]
    candidates_considered: int
    provenance: Provenance


class FeatureImportanceOut(BaseModel):
    feature_name: str
    family: str
    importance: float
    rank_stability: float | None = Field(
        default=None,
        description=(
            "Mean standard deviation of this feature's rank across folds. "
            "Null when fewer than two folds were measured, because a zero "
            "there would read as perfect stability rather than as no evidence."
        ),
    )
    per_label: dict[str, float] = Field(default_factory=dict)


class FeatureImportanceResponse(BaseModel):
    features: list[FeatureImportanceOut]
    families: dict[str, float]
    degenerate_labels: list[str]
    labels: list[str]
    label_weights: dict[str, float]
    folds_measured: int
    features_measured: int
    features_with_no_effect: int
    catalogue_version: str
    model_fingerprint: str
    computed_at: datetime
    stale: bool = Field(
        description=(
            "True when the table was computed against a different model than "
            "the one currently loaded. Serving old numbers silently is worse "
            "than serving none."
        )
    )


@app.get("/health", response_model=HealthResponse)
def health(service: ServiceDep) -> HealthResponse:
    """Whether the system can answer, and whether its data is current."""
    return service.health()


@app.get("/players", response_model=list[PlayerSummary])
def players(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
    position: Annotated[str | None, Query()] = None,
    max_price: Annotated[int | None, Query(description="Tenths of a million")] = None,
) -> list[PlayerSummary]:
    """Ranked players for the next gameweek."""
    try:
        return service.players(limit=limit, position=position, max_price=max_price)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/squad/{entry_id}", response_model=SquadResponse)
def squad(
    entry_id: int,
    service: ServiceDep,
    squad_file: Annotated[Path | None, Query()] = None,
) -> SquadResponse:
    """A manager's squad with projected points and the XI that would be fielded."""
    try:
        return service.squad(entry_id, squad_file=squad_file)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/recommend/{entry_id}", response_model=RecommendationResponse)
def recommend(
    entry_id: int,
    service: ServiceDep,
    squad_file: Annotated[Path | None, Query()] = None,
    horizon: Annotated[
        int,
        Query(
            ge=1,
            le=10,
            description=(
                "Gameweeks to judge the transfer over. A transfer is permanent "
                "and paid for once, so scoring it on the next gameweek alone "
                "undervalues buying a better player."
            ),
        ),
    ] = 1,
) -> RecommendationResponse:
    """The best legal single transfer, or an explicit hold."""
    try:
        return service.recommend(entry_id, squad_file=squad_file, horizon=horizon)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/build-squad", response_model=SquadResponse)
def build_squad(service: ServiceDep) -> SquadResponse:
    """A squad built from scratch — the gameweek-1 answer."""
    try:
        return service.build_squad()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/build-squad/explained", response_model=SquadBuildResponse)
def build_squad_explained(service: ServiceDep) -> SquadBuildResponse:
    """The optimal fifteen from scratch, with a justification for every pick.

    This is the gameweek-1 answer. Transfers are unlimited before the first
    deadline, so recommending one swap — which is what `/recommend` does — is
    answering a question nobody is asking.
    """
    try:
        return service.build_squad_explained()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/features/importance", response_model=FeatureImportanceResponse)
def feature_importance(
    service: ServiceDep,
    label: Annotated[str | None, Query(description="Restrict to one component label.")] = None,
    family: Annotated[str | None, Query(description="Restrict to one catalogue family.")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 60,
) -> FeatureImportanceResponse:
    """Which features actually earn their place, measured out of sample.

    Reads the table written by `xg importance`. Returns 404 rather than an empty
    ranking when none has been computed, because an empty list is
    indistinguishable from "no feature matters".
    """
    try:
        return service.feature_importance(label=label, family=family, limit=limit)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Objective-conditioned feature discovery
#
# Read surfaces plus one compile endpoint. Running an experiment is deliberately
# NOT exposed over HTTP: a discovery run fits hundreds of models and takes
# minutes, and there is no job queue in this repository (D1 keeps everything
# local). A request that blocks for that long is not an API, it is a timeout.
# `xg discover` runs them; these endpoints read what it produced.
# ---------------------------------------------------------------------------


class CompileRequest(BaseModel):
    """A manager's request, in their own words."""

    text: str = Field(min_length=1, description="What you want, in plain English")
    preset: str = Field(default="expected_points", description="Objective to start from")


class ParsedField(BaseModel):
    field: str
    confidence: float
    source: str
    evidence: str


class CompiledIntentResponse(BaseModel):
    """The parsed interpretation, for review and editing before anything runs."""

    objective_id: str
    primary_metric: str
    risk_preference: str
    planning_horizon: int
    ownership_preference: str
    signed_uncertainty_penalty: float
    locked_players: list[int]
    excluded_players: list[int]
    locked_positions: list[str]
    max_points_hit: int
    minimum_bank: int
    required_features: list[str]
    complement_targets: list[str]
    emphasis: list[str]
    beliefs: list[dict[str, object]]
    confidences: list[ParsedField]
    unparsed: list[str]
    overall_confidence: float


@app.post("/objectives/compile", response_model=CompiledIntentResponse)
def compile_objective(payload: CompileRequest, service: ServiceDep) -> CompiledIntentResponse:
    """Parse a request into an objective, constraints and beliefs.

    Deterministic — no language model. Returns everything the parser did *not*
    understand alongside what it did, so a partial interpretation is visible
    rather than silently narrowing the question.
    """
    try:
        return CompiledIntentResponse(
            **service.compile_objective(payload.text, preset=payload.preset)
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ObjectivePreset(BaseModel):
    id: str
    name: str
    primary_metric: str
    risk_preference: str
    planning_horizon: int
    ownership_preference: str


@app.get("/objectives", response_model=list[ObjectivePreset])
def objectives(service: ServiceDep) -> list[ObjectivePreset]:
    """The shipped objective presets."""
    return [ObjectivePreset(**row) for row in service.objective_presets()]


class DiscoveredFeature(BaseModel):
    feature: str
    version: str
    hypothesis_id: str
    status: str
    complementarity: str
    utility: float
    incremental_value: float
    folds: int
    folds_improved: int
    stability: float
    missingness: float
    leakage_passed: bool
    reason: str


@app.get("/features/discovered", response_model=list[DiscoveredFeature])
def discovered_features(
    service: ServiceDep,
    objective_id: Annotated[str, Query(description="Which objective the verdicts are under.")],
) -> list[DiscoveredFeature]:
    """Every discovered feature's latest verdict, accepted and rejected alike.

    Rejections are included. A report showing only winners cannot be
    distinguished from one that never looked.
    """
    try:
        return [DiscoveredFeature(**row) for row in service.discovered_features(objective_id)]
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class HypothesisResponse(BaseModel):
    id: str
    title: str
    football_rationale: str
    falsification_condition: str
    expected_relationship: str
    transformation_plan: str
    required_raw_fields: list[str]
    status: str
    generation_source: str
    leakage_risk: str


@app.get("/hypotheses", response_model=list[HypothesisResponse])
def hypotheses(service: ServiceDep) -> list[HypothesisResponse]:
    """Every hypothesis tested, with the condition that would have refuted it."""
    return [HypothesisResponse(**row) for row in service.hypotheses()]


class ClusterResponse(BaseModel):
    cluster_model_version: str
    cluster_id: int
    objective_id: str | None
    size: int
    label: str
    dominant_features: list[list[object]]


@app.get("/clusters", response_model=list[ClusterResponse])
def clusters(
    service: ServiceDep,
    objective_id: Annotated[str | None, Query(description="Objective conditioning.")] = None,
) -> list[ClusterResponse]:
    """Player clusters, with the statistical basis behind each label."""
    return [ClusterResponse(**row) for row in service.clusters(objective_id=objective_id)]


class ClusterHistoryEntry(BaseModel):
    season: str
    gameweek: int
    cluster_id: int
    membership_probability: float
    distance_to_centroid: float
    objective_id: str | None


@app.get("/players/{player_code}/cluster-history", response_model=list[ClusterHistoryEntry])
def cluster_history(player_code: int, service: ServiceDep) -> list[ClusterHistoryEntry]:
    """One player's cluster over time. A cluster is not an identity."""
    return [ClusterHistoryEntry(**row) for row in service.cluster_history(player_code)]


class ExperimentResponse(BaseModel):
    experiment_id: str
    stage: str
    objective_id: str
    objective_version: str
    seasons: list[str]
    hypotheses_proposed: int
    features_compiled: int
    features_accepted: int
    features_rejected: int
    reproducible: bool
    code_version: str
    git_dirty: bool
    started_at: datetime
    completed_at: datetime | None
    metrics: list[list[object]]


@app.get("/experiments", response_model=list[ExperimentResponse])
def experiments(service: ServiceDep) -> list[ExperimentResponse]:
    """Every recorded experiment, newest first."""
    return [ExperimentResponse(**row) for row in service.experiments()]


@app.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
def experiment(experiment_id: str, service: ServiceDep) -> ExperimentResponse:
    """One experiment's manifest."""
    found = service.experiment(experiment_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"no experiment {experiment_id!r}")
    return ExperimentResponse(**found)


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run("xg_alonso.api.main:app", host="127.0.0.1", port=8000, reload=False)


__all__: list[str] = ["app", "main"]
