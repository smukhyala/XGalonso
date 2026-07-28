"""The service layer behind the HTTP surface.

Holds the loaded context — four seasons of history, the canonical tables, the
pinned rules and optionally a fitted model — and turns them into the shapes the
API returns. Everything it does is a call into the same packages the CLI uses.

**That shared code is not by itself enough to keep the surfaces agreeing.** This
module used to default `model_path` to `.data/models/late.pkl` while `xg` on the
command line defaulted to the closed-form baseline, so the same team id produced
a different squad, a different projected score and a different bank depending on
which surface you asked — a 4-point, 6.0m disagreement that read as a bug in the
optimizer and cost real time to track down. The default is `None` now, matching
the CLI: a model is opt-in and named explicitly, and whichever is used is
recorded in every response's `Provenance`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from xg_alonso.cli.pipeline import (
    PRICES_WERE_ASSUMED,
    SliceContext,
    build_context,
    build_entities,
    fetch_squad,
    load_squad_file,
    recommend,
    squad_from_payload,
)
from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    Season,
    TeamId,
    TenthsOfMillion,
    parse_season,
)
from xg_alonso.contracts.prediction import PlayerPrediction
from xg_alonso.contracts.provenance import utc_now
from xg_alonso.contracts.squad import SquadState
from xg_alonso.features.catalogue import CATALOGUE_VERSION, build_catalogue
from xg_alonso.features.opponent import build_opponent_features, build_opponent_strength
from xg_alonso.features.slice1 import (
    SLICE1_FEATURE_SET_VERSION,
    build_slice1_features,
    build_team_gameweek_stats,
)
from xg_alonso.optimization import SquadCandidate, best_starting_xi, build_squad
from xg_alonso.pipelines.ingestion import SOURCE_BOOTSTRAP, SOURCE_FIXTURES, FplApiClient
from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA, empty_frame
from xg_alonso.prediction import load_models, predict_with_models
from xg_alonso.prediction.baseline import predict_frame
from xg_alonso.storage import FileSystemBronzeStore

if TYPE_CHECKING:
    from xg_alonso.api.main import (
        HealthResponse,
        PlayerSummary,
        Provenance,
        RecommendationResponse,
        SquadPlayer,
        SquadResponse,
    )

__all__ = ["DecisionService", "ServiceConfig"]


@dataclass(frozen=True)
class ServiceConfig:
    """Where the service reads from. Local paths only, per D1."""

    data_root: Path = field(default_factory=lambda: Path(".data"))
    season: str = "2026-27"

    model_path: Path | None = None
    """A fitted model, or the closed-form baseline when `None`.

    `None` deliberately matches `xg`'s default. Picking a model here silently
    made the API disagree with the CLI about the same team, which is worse than
    either choice on its own — a difference nobody asked for is a difference
    nobody can explain.
    """


class DecisionService:
    """Loads the world once, then answers questions about it."""

    def __init__(self, config: ServiceConfig) -> None:
        self._config = config
        self._season: Season = parse_season(config.season)
        self._context = self._load_context()
        self._models = self._load_models()
        self._predictions: dict[int, list[PlayerPrediction]] = {}

    # -- loading ----------------------------------------------------------

    def _load_context(self) -> SliceContext:
        bronze = FileSystemBronzeStore(self._config.data_root / "bronze")
        bootstrap_ref = bronze.latest(SOURCE_BOOTSTRAP)
        if bootstrap_ref is None:
            raise RuntimeError(
                f"no {SOURCE_BOOTSTRAP} snapshot under {self._config.data_root}. "
                "Run `xg ingest` before starting the API."
            )
        fixtures_ref = bronze.latest(SOURCE_FIXTURES)

        stats_path = self._config.data_root / "silver" / "player_gameweek_stats.parquet"
        player_stats = (
            pl.read_parquet(stats_path)
            if stats_path.exists()
            else empty_frame(PLAYER_GAMEWEEK_STATS_SCHEMA)
        )

        return build_context(
            json.loads(bronze.read(bootstrap_ref)),
            fixtures_payload=json.loads(bronze.read(fixtures_ref)) if fixtures_ref else [],
            player_stats=player_stats,
            season=self._season,
            snapshot_sha256=bootstrap_ref.content_sha256,
            available_time=bootstrap_ref.timestamps.available_time,
        )

    def _load_models(self) -> Any | None:
        path = self._config.model_path
        if path is None or not path.exists():
            return None
        return load_models(path).models

    # -- predictions ------------------------------------------------------

    def _predict(self, gameweek: GameweekId) -> list[PlayerPrediction]:
        """Predict every player for a gameweek, cached per gameweek.

        Cached because the whole catalogue is rebuilt per call and nothing about
        it changes between requests — the cutoff is the deadline, not now.
        """
        key = int(gameweek)
        if key in self._predictions:
            return self._predictions[key]

        cutoff = self._context.deadline_for(gameweek)
        entities = build_entities(self._context, cutoff=cutoff)
        run_id = f"api-{uuid.uuid4().hex[:12]}"

        if self._models is not None:
            features = build_catalogue(entities, player_stats=self._context.player_stats)
            features = build_opponent_features(
                features,
                opponent_strength=build_opponent_strength(self._context.player_stats),
            )
            predictions = predict_with_models(
                features,
                models=self._models,
                rules=self._context.scoring,
                from_gameweek=gameweek,
                data_cutoff=cutoff,
                predicted_at=utc_now(),
                run_id=run_id,
                code_version="api",
                feature_set_version=CATALOGUE_VERSION,
            )
        else:
            features = build_slice1_features(
                entities,
                player_stats=self._context.player_stats,
                team_stats=build_team_gameweek_stats(
                    self._context.player_stats, self._context.players
                ),
            )
            predictions = predict_frame(
                features,
                rules=self._context.scoring,
                from_gameweek=gameweek,
                data_cutoff=cutoff,
                predicted_at=utc_now(),
                run_id=run_id,
                code_version="api",
                feature_set_version=SLICE1_FEATURE_SET_VERSION,
            )

        self._predictions[key] = predictions
        return predictions

    # -- shaping ----------------------------------------------------------

    def _provenance(self, prediction: PlayerPrediction) -> Provenance:
        from xg_alonso.api.main import Provenance

        p = prediction.provenance
        return Provenance(
            model_name=p.model_name,
            model_version=p.model_version,
            feature_set_version=p.feature_set_version,
            data_cutoff=p.data_cutoff,
            generated_at=p.predicted_at,
            run_id=p.run_id,
        )

    def _player_rows(self) -> dict[int, dict[str, Any]]:
        return {int(r["player_code"]): r for r in self._context.players.iter_rows(named=True)}

    def _summary(self, prediction: PlayerPrediction, row: dict[str, Any]) -> PlayerSummary:
        from xg_alonso.api.main import PlayerSummary

        minutes = prediction.components.minutes
        return PlayerSummary(
            player_code=int(prediction.player_code),
            name=str(row["web_name"]),
            position=prediction.position.value,
            team_id=int(row["team_id"]),
            price=int(row["current_price"]),
            status=row.get("status"),
            expected_points=round(prediction.expected_points, 3),
            expected_points_sd=round(prediction.expected_points_sd, 3),
            p_start=round(minutes.p_start, 4),
            expected_minutes=round(minutes.expected_minutes, 1),
        )

    # -- endpoints --------------------------------------------------------

    def health(self) -> HealthResponse:
        from xg_alonso.api.main import HealthResponse

        gameweek = self._context.next_gameweek()
        deadline = self._context.deadline_for(gameweek)
        return HealthResponse(
            status="ok",
            season=str(self._season),
            next_gameweek=int(gameweek),
            deadline=deadline,
            players_loaded=self._context.players.height,
            history_rows=self._context.player_stats.height,
            model_loaded=self._models is not None,
            # A snapshot older than the deadline it is predicting cannot reflect
            # team news, so saying so is more useful than a confident number.
            stale=deadline < utc_now(),
        )

    def players(
        self,
        *,
        limit: int = 50,
        position: str | None = None,
        max_price: int | None = None,
    ) -> list[PlayerSummary]:
        gameweek = self._context.next_gameweek()
        rows = self._player_rows()

        summaries = [
            self._summary(p, rows[int(p.player_code)])
            for p in self._predict(gameweek)
            if int(p.player_code) in rows
        ]
        if position is not None:
            wanted = position.upper()
            summaries = [s for s in summaries if s.position == wanted]
        if max_price is not None:
            summaries = [s for s in summaries if s.price <= max_price]

        summaries.sort(key=lambda s: (-s.expected_points, s.player_code))
        return summaries[:limit]

    def _load_squad(self, entry_id: int, squad_file: Path | None) -> SquadState:
        gameweek = self._context.next_gameweek()
        if squad_file is not None:
            return squad_from_payload(
                load_squad_file(squad_file),
                context=self._context,
                entry_id=EntryId(entry_id),
                gameweek=gameweek,
            )
        try:
            with FplApiClient() as client:
                return fetch_squad(
                    client=client,
                    context=self._context,
                    entry_id=EntryId(entry_id),
                    gameweek=gameweek,
                )
        except Exception as exc:
            raise LookupError(
                f"could not load entry {entry_id} for GW{gameweek}: {exc}. "
                "The picks endpoint returns 404 until that gameweek's deadline."
            ) from exc

    def _squad_response(self, squad: SquadState, *, prices_assumed: bool) -> SquadResponse:
        from xg_alonso.api.main import SquadPlayer, SquadResponse

        gameweek = squad.gameweek
        by_code = {p.player_code: p for p in self._predict(gameweek)}
        rows = self._player_rows()
        selection = best_starting_xi(squad.picks, by_code, self._context.squad_rules)
        starters = {p.player_code for p in selection.starters}

        players: list[SquadPlayer] = []
        for pick in squad.picks:
            prediction = by_code.get(pick.player_code)
            if prediction is None:
                continue
            base = self._summary(prediction, rows[int(pick.player_code)])
            players.append(
                SquadPlayer(
                    **base.model_dump(),
                    squad_slot=pick.squad_slot,
                    is_captain=pick.player_code == selection.captain,
                    is_vice_captain=pick.player_code == selection.vice_captain,
                    is_starter=pick.player_code in starters,
                    selling_price=int(pick.selling_price),
                    purchase_price=int(pick.purchase_price),
                )
            )

        return SquadResponse(
            entry_id=int(squad.entry_id),
            gameweek=int(gameweek),
            formation=selection.formation_label,
            squad_value=int(squad.squad_value),
            bank=int(squad.bank),
            free_transfers=squad.free_transfers,
            projected_points=round(selection.expected_points, 2),
            prices_assumed=prices_assumed,
            players=sorted(players, key=lambda p: (not p.is_starter, p.squad_slot)),
            provenance=self._provenance(next(iter(by_code.values()))),
        )

    def squad(self, entry_id: int, *, squad_file: Path | None = None) -> SquadResponse:
        state = self._load_squad(entry_id, squad_file)
        return self._squad_response(state, prices_assumed=PRICES_WERE_ASSUMED.get(entry_id, False))

    def recommend(self, entry_id: int, *, squad_file: Path | None = None) -> RecommendationResponse:
        from xg_alonso.api.main import ReasonOut, RecommendationResponse

        state = self._load_squad(entry_id, squad_file)
        run_id = f"api-{uuid.uuid4().hex[:12]}"
        recommendation, by_code = recommend(
            context=self._context,
            squad=state,
            entry_id=EntryId(entry_id),
            run_id=run_id,
            code_version="api",
            generated_at=utc_now(),
            models=self._models,
        )
        rows = self._player_rows()

        out_summary = None
        in_summary = None
        if recommendation.package.moves:
            move = recommendation.package.moves[0]
            if move.player_out in by_code:
                out_summary = self._summary(by_code[move.player_out], rows[int(move.player_out)])
            if move.player_in in by_code:
                in_summary = self._summary(by_code[move.player_in], rows[int(move.player_in)])

        return RecommendationResponse(
            entry_id=entry_id,
            gameweek=int(recommendation.gameweek),
            is_hold=recommendation.package.is_hold,
            player_out=out_summary,
            player_in=in_summary,
            hit_cost=recommendation.package.hit_cost,
            bank_after=int(recommendation.package.bank_after),
            projected_hold=round(recommendation.comparison.baseline_expected_points, 2),
            projected_after=round(recommendation.comparison.candidate_expected_points, 2),
            expected_gain=round(recommendation.expected_points_gain, 2),
            risk=round(recommendation.risk_score, 2),
            reasons=[
                ReasonOut(
                    code=r.code.value,
                    text=r.render(),
                    subject=int(r.subject),
                    weight=round(r.weight, 3),
                )
                for r in sorted(recommendation.reasons, key=lambda r: -r.weight)
            ],
            provenance=self._provenance(next(iter(by_code.values()))),
        )

    def build_squad(self) -> SquadResponse:
        gameweek = self._context.next_gameweek()
        predictions = self._predict(gameweek)
        rows = self._player_rows()

        available = {
            code: row for code, row in rows.items() if row.get("status") in (None, "a", "d")
        }
        candidates = [
            SquadCandidate(
                player_code=p.player_code,
                position=p.position,
                team_id=TeamId(int(available[int(p.player_code)]["team_id"])),
                price=TenthsOfMillion(int(available[int(p.player_code)]["current_price"])),
                prediction=p,
            )
            for p in predictions
            if int(p.player_code) in available
        ]
        squad, _ = build_squad(
            candidates,
            rules=self._context.squad_rules,
            entry_id=EntryId(0),
            gameweek=gameweek,
            predictions={p.player_code: p for p in predictions},
        )
        return self._squad_response(squad, prices_assumed=False)
