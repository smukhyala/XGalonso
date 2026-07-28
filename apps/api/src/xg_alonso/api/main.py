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
    weight: float


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
    provenance: Provenance


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


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run("xg_alonso.api.main:app", host="127.0.0.1", port=8000, reload=False)


__all__: list[str] = ["app", "main"]
