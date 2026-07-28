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
        )
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
) -> RecommendationResponse:
    """The best legal single transfer, or an explicit hold."""
    try:
        return service.recommend(entry_id, squad_file=squad_file)
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


@app.get("/features/importance", response_model=FeatureImportanceResponse)
def feature_importance(
    service: ServiceDep,
    label: Annotated[
        str | None, Query(description="Restrict to one component label.")
    ] = None,
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


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run("xg_alonso.api.main:app", host="127.0.0.1", port=8000, reload=False)


__all__: list[str] = ["app", "main"]
