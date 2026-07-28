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
import os
import uuid
from collections.abc import Sequence
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
    fixtures_by_gameweek,
    horizon_projections,
    load_squad_file,
    recommend,
    squad_from_payload,
)
from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    Season,
    TeamId,
    TenthsOfMillion,
    parse_season,
)
from xg_alonso.contracts.prediction import PlayerPrediction
from xg_alonso.contracts.provenance import utc_now
from xg_alonso.contracts.reason_codes import Reason
from xg_alonso.contracts.recommendation import TransferOption
from xg_alonso.contracts.squad import SquadState
from xg_alonso.evaluation.importance import load_importance
from xg_alonso.explanations.derivation import derive_points
from xg_alonso.explanations.history import build_history_notes
from xg_alonso.explanations.lineup_diff import compare_lineups, selection_from_starters
from xg_alonso.explanations.player import ArchetypeVerdict, Comparable, explain_squad
from xg_alonso.explanations.reasons import PopulationStats
from xg_alonso.features.archetypes import ArchetypeModel, build_archetypes
from xg_alonso.features.assemble import build_model_features
from xg_alonso.features.catalogue import CATALOGUE_VERSION, build_catalogue
from xg_alonso.features.opponent import build_opponent_features, build_opponent_strength
from xg_alonso.features.slice1 import (
    SLICE1_FEATURE_SET_VERSION,
    build_slice1_features,
    build_team_gameweek_stats,
)
from xg_alonso.optimization import SquadCandidate, best_starting_xi, build_squad
from xg_alonso.optimization.horizon import discount_weights, value_over_horizon
from xg_alonso.pipelines.ingestion import SOURCE_BOOTSTRAP, SOURCE_FIXTURES, FplApiClient
from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA, empty_frame
from xg_alonso.prediction import load_models, predict_with_models
from xg_alonso.prediction.baseline import predict_frame
from xg_alonso.prediction.calibration import apply_price_calibration
from xg_alonso.prediction.form import apply_form_signals, form_reason, load_signals
from xg_alonso.storage import FileSystemBronzeStore

if TYPE_CHECKING:
    from xg_alonso.api.main import (
        BreakdownOut,
        FeatureImportanceResponse,
        HealthResponse,
        PlayerExplanationOut,
        PlayerSummary,
        Provenance,
        ReasonOut,
        RecommendationResponse,
        SquadBuildResponse,
        SquadPlayer,
        SquadResponse,
        TransferOptionOut,
    )

__all__ = ["DecisionService", "ServiceConfig"]


def _model_from_environment() -> Path | None:
    """A model path from `XG_MODEL_PATH`, or `None`.

    Opt-in through the environment rather than through a changed default. The
    default stays the closed-form baseline for the reason recorded below, but
    without *some* switch the richer path was unreachable from the web app: the
    baseline builds eight features, so a defender's clean-sheet rate, expected
    goals conceded and defensive actions all read "not measured" on screen —
    honest, and useless.
    """
    raw = os.environ.get("XG_MODEL_PATH")
    return Path(raw) if raw else None


@dataclass(frozen=True)
class ServiceConfig:
    """Where the service reads from. Local paths only, per D1."""

    data_root: Path = field(default_factory=lambda: Path(".data"))
    season: str = "2026-27"

    model_path: Path | None = field(default_factory=_model_from_environment)
    """A fitted model, or the closed-form baseline when `None`.

    Defaults to `None` unless `XG_MODEL_PATH` is set, which matches `xg`'s own
    default. Picking a model here unconditionally silently made the API disagree
    with the CLI about the same team, which is worse than either choice on its
    own — a difference nobody asked for is a difference nobody can explain.
    """


class DecisionService:
    """Loads the world once, then answers questions about it."""

    def __init__(self, config: ServiceConfig) -> None:
        self._config = config
        self._season: Season = parse_season(config.season)
        self._context = self._load_context()
        self._models = self._load_models()
        self._predictions: dict[int, list[PlayerPrediction]] = {}
        self._archetypes: dict[int, ArchetypeModel] = {}
        self._history: dict[int, dict[int, list[Any]]] = {}
        self._horizon: dict[tuple[int, int], dict[PlayerCode, list[float]]] = {}

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
            features = build_model_features(entities, player_stats=self._context.player_stats)
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

        # Correct the model's measured under-projection of expensive players
        # before anything downstream reads a number. Left uncorrected it makes
        # the optimizer bank money rather than upgrade, because the players a
        # budget would be spent on are exactly the ones scored too low.
        rows = self._player_rows()
        prices = {
            PlayerCode(code): TenthsOfMillion(int(row["current_price"]))
            for code, row in rows.items()
        }
        predictions = apply_price_calibration(predictions, prices)

        # Outside information, applied here rather than in each endpoint so that
        # `/players`, `/squad` and `/build-squad` cannot disagree with each
        # other about the same player — the failure mode this module's header
        # already records once.
        predictions = apply_form_signals(
            predictions,
            load_signals(self._config.data_root / "signals" / "form_signals.json"),
            at=cutoff,
        )

        self._predictions[key] = predictions
        return predictions

    def _archetype_model(self, gameweek: GameweekId) -> ArchetypeModel:
        """Cluster the whole player population into archetypes, cached per gameweek.

        Built from the same feature frame the predictions came from, so a
        player's archetype and his projection describe the same player at the
        same cutoff.
        """
        key = int(gameweek)
        if key in self._archetypes:
            return self._archetypes[key]

        cutoff = self._context.deadline_for(gameweek)
        entities = build_entities(self._context, cutoff=cutoff)
        features = build_catalogue(entities, player_stats=self._context.player_stats)
        features = build_opponent_features(
            features,
            opponent_strength=build_opponent_strength(self._context.player_stats),
        )
        model = build_archetypes(
            features,
            positions=[str(p) for p in features["position"].to_list()],
            player_codes=[int(c) for c in features["player_code"].to_list()],
        )
        self._archetypes[key] = model
        return model

    _ARCHETYPE_CAVEAT = (
        "Archetypes are clustered on playing style and output together, so a "
        "cluster is partly defined by how good its members are. Read this rank "
        "as a fact about the group, not as the argument for picking him — that "
        "argument is in the evidence above, which compares him against his "
        "position and his price."
    )

    def _archetype_verdicts(
        self,
        gameweek: GameweekId,
        by_code: dict[PlayerCode, PlayerPrediction],
    ) -> dict[PlayerCode, ArchetypeVerdict]:
        """Each player's archetype, his rank inside it, and his nearest comps."""
        model = self._archetype_model(gameweek)
        rows = self._player_rows()

        verdicts: dict[PlayerCode, ArchetypeVerdict] = {}
        for placement in model.players:
            code = PlayerCode(placement.player_code)
            archetype = model.archetype_of(placement.player_code)
            if archetype is None or code not in by_code:
                continue

            members = [
                member
                for member in model.members(placement.position, placement.archetype)
                if PlayerCode(member) in by_code
            ]
            ordered = sorted(
                members,
                key=lambda member: -by_code[PlayerCode(member)].expected_points,
            )
            rank = (
                ordered.index(placement.player_code) + 1 if placement.player_code in ordered else 0
            )

            verdicts[code] = ArchetypeVerdict(
                label=archetype.label,
                size=archetype.size,
                rank_within=rank,
                comparables=tuple(
                    Comparable(
                        player_code=PlayerCode(other),
                        expected_points=round(by_code[PlayerCode(other)].expected_points, 2),
                        price=(
                            TenthsOfMillion(int(rows[other]["current_price"]))
                            if other in rows
                            else None
                        ),
                    )
                    for other in placement.comparables
                    if PlayerCode(other) in by_code
                ),
            )
        return verdicts

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

    def _summary(
        self,
        prediction: PlayerPrediction,
        row: dict[str, Any],
        *,
        with_depth: bool = False,
    ) -> PlayerSummary:
        """One player, optionally with the arithmetic and the weeks ahead.

        ``with_depth`` is opt-in because the horizon costs a feature build and
        several model passes. The squad and recommendation views carry their own
        deeper explanation already; the board is where a reader has nothing else
        to go on, so that is where it is turned on.
        """
        from xg_alonso.api.main import PlayerSummary

        minutes = prediction.components.minutes
        derivation: list[Any] = []
        horizon: list[Any] = []
        horizon_total: float | None = None
        if with_depth:
            derivation, _ = self._derivation_out(prediction)
            horizon, horizon_total = self._horizon_out(
                prediction.from_gameweek, prediction.player_code, self._HORIZON_WEEKS
            )
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
            history=self._history_out(prediction.from_gameweek, int(prediction.player_code)),
            derivation=derivation,
            horizon=horizon,
            horizon_total=horizon_total,
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
        top = summaries[:limit]

        # Depth is computed only for the rows actually returned. Building the
        # horizon for six hundred players to display twelve would be most of a
        # second spent on numbers nobody sees.
        rows_by_code = {int(code): row for code, row in rows.items()}
        predictions = {int(p.player_code): p for p in self._predict(gameweek)}
        return [
            self._summary(predictions[s.player_code], rows_by_code[s.player_code], with_depth=True)
            if s.player_code in predictions and s.player_code in rows_by_code
            else s
            for s in top
        ]

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

    def _form_reasons(self, gameweek: GameweekId) -> dict[PlayerCode, Reason]:
        """Sourced outside information, as grounded reasons.

        Evaluated against the deadline rather than the wall clock so the
        explanation matches the prediction, which was adjusted at the same
        moment.
        """
        signals = load_signals(self._config.data_root / "signals" / "form_signals.json")
        at = self._context.deadline_for(gameweek)
        return {
            code: form_reason(signal, code, weight=2.0) for code, signal in signals.live(at).items()
        }

    #: Gameweeks ahead shown alongside a projection. Five, matching the horizon
    #: the optimizer values transfers over.
    _HORIZON_WEEKS: int = 5

    def _horizon_projections(
        self, gameweek: GameweekId, weeks: int
    ) -> dict[PlayerCode, list[float]]:
        """Per-gameweek projections for the whole league, cached.

        Costly enough to be worth caching and cheap enough to be worth having:
        the catalogue is built once and only the fixture varies per week, so
        this is one feature build plus a handful of model passes rather than
        five full rebuilds.
        """
        key = (int(gameweek), weeks)
        if key in self._horizon:
            return self._horizon[key]

        projections = horizon_projections(
            self._context,
            from_gameweek=gameweek,
            horizon=weeks,
            generated_at=utc_now(),
            run_id=f"api-{uuid.uuid4().hex[:8]}",
            code_version="api",
            models=self._models,
        )
        self._horizon[key] = projections
        return projections

    def _horizon_out(
        self, gameweek: GameweekId, player: PlayerCode, weeks: int
    ) -> tuple[list[Any], float | None]:
        """One player's next few gameweeks, each with the club he faces."""
        from xg_alonso.api.main import GameweekProjectionOut

        weekly = self._horizon_projections(gameweek, weeks).get(player)
        if not weekly:
            return [], None

        schedule = fixtures_by_gameweek(
            self._context, [GameweekId(int(gameweek) + i) for i in range(weeks)]
        )
        teams = {
            int(row["team_id"]): str(row["name"])
            for row in self._context.teams.iter_rows(named=True)
        }
        club = self._player_rows().get(int(player), {}).get("team_id")
        weights = discount_weights(len(weekly))

        out: list[Any] = []
        for index, points in enumerate(weekly):
            week = int(gameweek) + index
            opponent: str | None = None
            is_home: bool | None = None
            fixtures = schedule.get(week)
            if fixtures is not None and club is not None:
                match = fixtures.filter(pl.col("team_id") == int(club))
                if not match.is_empty():
                    opponent = teams.get(int(match["opponent_team_id"][0]))
                    is_home = bool(match["was_home"][0])
            out.append(
                GameweekProjectionOut(
                    gameweek=week,
                    opponent=opponent,
                    is_home=is_home,
                    expected_points=round(points, 2),
                    weight=round(weights[index], 3),
                )
            )

        total = value_over_horizon(weekly).total
        return out, round(total, 2)

    def _history_notes(self, gameweek: GameweekId) -> dict[int, list[Any]]:
        """Situational history for every player, cached per gameweek.

        One scan of the stats table serves the whole league. Asking per player
        would repeat the same hundred-thousand-row filter six hundred times to
        answer six hundred versions of the same question.
        """
        key = int(gameweek)
        if key in self._history:
            return self._history[key]

        cutoff = self._context.deadline_for(gameweek)
        entities = build_entities(self._context, cutoff=cutoff)
        fixtures = {
            int(row["player_code"]): (int(row["opponent_team_id"]), bool(row["was_home"]))
            for row in entities.iter_rows(named=True)
            if row.get("opponent_team_id") is not None
        }
        teams = {
            int(row["team_id"]): str(row["name"])
            for row in self._context.teams.iter_rows(named=True)
        }
        notes = build_history_notes(
            self._context.player_stats,
            fixtures=fixtures,
            gameweek=int(gameweek),
            team_names=teams,
            cutoff=cutoff,
        )
        self._history[key] = notes
        return notes

    def _history_out(self, gameweek: GameweekId, player: int, limit: int = 3) -> list[Any]:
        from xg_alonso.api.main import HistoryNoteOut

        return [
            HistoryNoteOut(kind=n.kind, text=n.text, is_positive=n.is_positive)
            for n in self._history_notes(gameweek).get(player, [])[:limit]
        ]

    def _names(self) -> dict[int, str]:
        return {code: str(row["web_name"]) for code, row in self._player_rows().items()}

    def _reasons_out(self, reasons: Sequence[Reason]) -> list[ReasonOut]:
        """Render reasons with their subject named, strongest first.

        Naming the subject is the whole fix for the screen that showed
        "minutes look secure" directly above "minutes are a concern" with no
        indication that they were about different players.
        """
        from xg_alonso.api.main import ReasonOut

        names = self._names()
        return [
            ReasonOut(
                code=reason.code.value,
                text=reason.render(),
                subject=int(reason.subject),
                subject_name=names.get(int(reason.subject), f"player {int(reason.subject)}"),
                polarity=reason.polarity.value,
                weight=round(reason.weight, 3),
            )
            for reason in sorted(reasons, key=lambda r: -r.weight)
        ]

    def _option_out(self, option: TransferOption, gameweek: GameweekId) -> TransferOptionOut:
        from xg_alonso.api.main import TransferOptionOut

        names = self._names()
        rows = self._player_rows()
        out_row = rows.get(int(option.move.player_out), {})
        return TransferOptionOut(
            player_out=int(option.move.player_out),
            player_out_name=names.get(int(option.move.player_out), "unknown"),
            player_in=int(option.move.player_in),
            player_in_name=names.get(int(option.move.player_in), "unknown"),
            position=str(out_row.get("position", "")),
            selling_price=int(option.move.selling_price),
            purchase_price=int(option.move.purchase_price),
            gross_gain=round(option.gross_gain, 2),
            net_gain=round(option.net_gain, 2),
            hit_cost=option.hit_cost,
            risk_penalty=round(option.risk_penalty, 2),
            bank_after=int(option.bank_after),
            reasons=self._reasons_out(option.reasons),
            history_in=self._history_out(gameweek, int(option.move.player_in), limit=2),
            history_out=self._history_out(gameweek, int(option.move.player_out), limit=2),
        )

    def _derivation_out(self, prediction: PlayerPrediction) -> tuple[list[Any], bool]:
        """The arithmetic behind one projection, biggest term first."""
        from xg_alonso.api.main import DerivationLineOut

        derivation = derive_points(prediction, self._context.scoring)
        return (
            [
                DerivationLineOut(
                    component=line.component,
                    expectation=round(line.expectation, 4),
                    unit=line.unit,
                    rate=line.rate,
                    points=round(line.points, 3),
                    note=line.note,
                    sentence=line.explain(),
                )
                for line in derivation.material()
            ],
            derivation.reconciles,
        )

    def _lineup_comparison(
        self,
        squad: SquadState,
        by_code: dict[PlayerCode, PlayerPrediction],
    ) -> Any:
        """The eleven the manager has set, against the one we would field.

        A manager's own starters come from the picks payload, where slots 1-11
        are the XI. That is a *given*, so it is priced rather than optimised —
        the comparison is only meaningful if one side is genuinely his.
        """
        from xg_alonso.api.main import LineupComparisonOut, SwapOut

        names = self._names()
        theirs = [pick.player_code for pick in squad.picks if pick.squad_slot <= 11]
        if len(theirs) != 11:
            return None

        yours = selection_from_starters(theirs, squad.picks, by_code)
        ours = best_starting_xi(squad.picks, by_code, self._context.squad_rules)
        comparison = compare_lineups(yours, ours, by_code)

        def named(code: PlayerCode | None) -> str | None:
            return None if code is None else names.get(int(code), "unknown")

        return LineupComparisonOut(
            yours_points=round(comparison.yours_points, 2),
            ours_points=round(comparison.ours_points, 2),
            total_delta=round(comparison.total_delta, 2),
            swap_delta=round(comparison.swap_delta, 2),
            captain_delta=round(comparison.captain_delta, 2),
            shape_delta=round(comparison.shape_delta, 2),
            yours_formation=comparison.yours_formation,
            ours_formation=comparison.ours_formation,
            yours_captain_name=named(comparison.yours_captain),
            ours_captain_name=named(comparison.ours_captain),
            swaps=[
                SwapOut(
                    position=swap.position.value,
                    player_in=None if swap.player_in is None else int(swap.player_in),
                    player_in_name=named(swap.player_in),
                    player_out=None if swap.player_out is None else int(swap.player_out),
                    player_out_name=named(swap.player_out),
                    points_in=round(swap.points_in, 2),
                    points_out=round(swap.points_out, 2),
                    delta=round(swap.delta, 2),
                    is_like_for_like=swap.is_like_for_like,
                )
                for swap in comparison.swaps
            ],
            is_identical=comparison.is_identical,
            yours_is_better=comparison.yours_is_better,
        )

    def _breakdown_out(self, prediction: PlayerPrediction) -> BreakdownOut:
        from xg_alonso.api.main import BreakdownOut

        b = prediction.breakdown
        return BreakdownOut(
            appearance=round(b.appearance, 3),
            goals=round(b.goals, 3),
            assists=round(b.assists, 3),
            clean_sheets=round(b.clean_sheets, 3),
            goals_conceded=round(b.goals_conceded, 3),
            saves=round(b.saves, 3),
            cards=round(b.cards, 3),
            defensive_contribution=round(b.defensive_contribution, 3),
            bonus=round(b.bonus, 3),
            total=round(b.total, 3),
        )

    def _explanations_out(
        self,
        recommendation: Any,
        squad: SquadState,
        by_code: dict[PlayerCode, PlayerPrediction],
    ) -> list[PlayerExplanationOut]:
        """Per-player justification for the whole squad.

        Built from the board the optimizer already produced, so a player's row
        and the headline recommendation are the same computation rather than two
        that happen to agree today.
        """
        from xg_alonso.api.main import FeatureValueOut, PlayerExplanationOut

        board = recommendation.board
        if board is None:
            return []

        gameweek = squad.gameweek
        rows = self._player_rows()
        names = self._names()
        selection = best_starting_xi(squad.picks, by_code, self._context.squad_rules)
        starters = frozenset(p.player_code for p in selection.starters)
        by_player = {entry.player_out: entry for entry in board.by_player}
        prices = {
            pick.player_code: TenthsOfMillion(int(pick.selling_price)) for pick in squad.picks
        }
        chances = {
            PlayerCode(code): float(row["chance_of_playing_next_round"]) / 100.0
            for code, row in rows.items()
            if row.get("chance_of_playing_next_round") is not None
        }

        explanations = explain_squad(
            picks=squad.picks,
            predictions=by_code,
            rules=self._context.squad_rules,
            starters=starters,
            baseline_points=selection.expected_points,
            board_by_player=by_player,
            alternatives=board.top,
            population=PopulationStats.from_predictions(
                list(by_code.values()),
                prices={
                    PlayerCode(code): TenthsOfMillion(int(row["current_price"]))
                    for code, row in rows.items()
                },
            ),
            prices=prices,
            chances_of_playing=chances,
            archetypes=self._archetype_verdicts(gameweek, by_code),
            form_reasons=self._form_reasons(gameweek),
        )

        out: list[PlayerExplanationOut] = []
        for explanation in explanations:
            code = int(explanation.player_code)
            entry = by_player.get(explanation.player_code)
            derivation_lines, derivation_ok = self._derivation_out(by_code[explanation.player_code])
            out.append(
                PlayerExplanationOut(
                    player_code=code,
                    name=names.get(code, "unknown"),
                    position=str(rows[code]["position"]) if code in rows else "",
                    expected_points=round(explanation.expected_points, 2),
                    breakdown=self._breakdown_out(by_code[explanation.player_code]),
                    evidence=[
                        FeatureValueOut(
                            name=value.name,
                            label=value.label,
                            family=value.family,
                            value=None if value.value is None else round(value.value, 3),
                            percentile=(
                                None if value.percentile is None else round(value.percentile, 3)
                            ),
                            higher_is_better=value.higher_is_better,
                        )
                        for value in explanation.evidence
                    ],
                    reasons=self._reasons_out(explanation.reasons),
                    is_starter=explanation.start_verdict.is_starter,
                    start_margin=round(explanation.start_verdict.margin, 2),
                    forced_by_quota=explanation.start_verdict.forced_by_quota,
                    legal_replacements=0 if entry is None else entry.legal_replacements,
                    replacements=[
                        self._option_out(option, gameweek) for option in explanation.replacements
                    ],
                    no_replacement_reasons=self._reasons_out(explanation.no_replacement_reasons),
                    archetype=self._archetype_out(explanation.archetype),
                    history=self._history_out(gameweek, code),
                    derivation=derivation_lines,
                    derivation_reconciles=derivation_ok,
                )
            )
        return out

    def _archetype_out(self, verdict: Any) -> Any:
        from xg_alonso.api.main import ArchetypeOut, ComparableOut

        if verdict is None:
            return None
        names = self._names()
        return ArchetypeOut(
            label=verdict.label,
            size=verdict.size,
            rank_within=verdict.rank_within,
            comparables=[
                ComparableOut(
                    player_code=int(c.player_code),
                    name=names.get(int(c.player_code), "unknown"),
                    expected_points=c.expected_points,
                    price=None if c.price is None else int(c.price),
                )
                for c in verdict.comparables
            ],
            caveat=self._ARCHETYPE_CAVEAT,
        )

    def recommend(
        self, entry_id: int, *, squad_file: Path | None = None, horizon: int = 1
    ) -> RecommendationResponse:
        from xg_alonso.api.main import RecommendationResponse

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
            horizon=horizon,
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

        gameweek = recommendation.gameweek
        board = recommendation.board
        chosen = None
        if recommendation.package.moves:
            chosen = recommendation.package.moves[0]

        alternatives = []
        if board is not None:
            alternatives = [
                self._option_out(option, gameweek)
                for option in board.top
                # The headline move is already stated above; repeating it as its
                # own runner-up would suggest the board contains one fewer real
                # alternative than it does.
                if chosen is None
                or (
                    option.move.player_in != chosen.player_in
                    or option.move.player_out != chosen.player_out
                )
            ]

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
            reasons=self._reasons_out(recommendation.reasons),
            alternatives=alternatives,
            players=self._explanations_out(recommendation, state, by_code),
            candidates_considered=0 if board is None else board.candidates_considered,
            legal_moves=0 if board is None else board.legal_moves,
            lineup=self._lineup_comparison(state, by_code),
            provenance=self._provenance(next(iter(by_code.values()))),
        )

    def build_squad_explained(self) -> SquadBuildResponse:
        """The optimal fifteen from scratch, with a case for every pick.

        The gameweek-1 answer. Before the first deadline there is no squad to
        transfer from and transfers are unlimited, so the single-transfer
        recommendation is answering a question nobody is asking.

        No transfer board is built here, for the same reason: "who would replace
        him" has no meaning when every player was chosen freely and could be
        swapped at no cost. What each pick carries instead is why *he* projects
        as he does, what kind of player he is, and who else is that kind.
        """
        from xg_alonso.api.main import SquadBuildResponse

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
        by_code = {p.player_code: p for p in predictions}

        squad, selection = build_squad(
            candidates,
            rules=self._context.squad_rules,
            entry_id=EntryId(0),
            gameweek=gameweek,
            predictions=by_code,
        )

        base = self._squad_response(squad, prices_assumed=False)
        starters = frozenset(p.player_code for p in selection.starters)
        prices = {
            pick.player_code: TenthsOfMillion(int(pick.current_price)) for pick in squad.picks
        }
        chances = {
            PlayerCode(code): float(row["chance_of_playing_next_round"]) / 100.0
            for code, row in rows.items()
            if row.get("chance_of_playing_next_round") is not None
        }

        explanations = explain_squad(
            picks=squad.picks,
            predictions=by_code,
            rules=self._context.squad_rules,
            starters=starters,
            baseline_points=selection.expected_points,
            population=PopulationStats.from_predictions(
                predictions,
                prices={
                    PlayerCode(code): TenthsOfMillion(int(row["current_price"]))
                    for code, row in rows.items()
                },
            ),
            prices=prices,
            chances_of_playing=chances,
            archetypes=self._archetype_verdicts(gameweek, by_code),
            form_reasons=self._form_reasons(gameweek),
        )

        from xg_alonso.api.main import FeatureValueOut, PlayerExplanationOut

        names = self._names()
        shaped: list[PlayerExplanationOut] = []
        for explanation in explanations:
            code = int(explanation.player_code)
            derivation_lines, derivation_ok = self._derivation_out(by_code[explanation.player_code])
            shaped.append(
                PlayerExplanationOut(
                    player_code=code,
                    name=names.get(code, "unknown"),
                    position=str(rows[code]["position"]) if code in rows else "",
                    expected_points=round(explanation.expected_points, 2),
                    breakdown=self._breakdown_out(by_code[explanation.player_code]),
                    evidence=[
                        FeatureValueOut(
                            name=value.name,
                            label=value.label,
                            family=value.family,
                            value=None if value.value is None else round(value.value, 3),
                            percentile=(
                                None if value.percentile is None else round(value.percentile, 3)
                            ),
                            higher_is_better=value.higher_is_better,
                        )
                        for value in explanation.evidence
                    ],
                    reasons=self._reasons_out(explanation.reasons),
                    is_starter=explanation.start_verdict.is_starter,
                    start_margin=round(explanation.start_verdict.margin, 2),
                    forced_by_quota=explanation.start_verdict.forced_by_quota,
                    legal_replacements=0,
                    replacements=[],
                    no_replacement_reasons=[],
                    archetype=self._archetype_out(explanation.archetype),
                    history=self._history_out(gameweek, code),
                    derivation=derivation_lines,
                    derivation_reconciles=derivation_ok,
                )
            )

        return SquadBuildResponse(
            gameweek=int(gameweek),
            formation=base.formation,
            squad_value=base.squad_value,
            bank=base.bank,
            projected_points=base.projected_points,
            players=base.players,
            explanations=shaped,
            candidates_considered=len(candidates),
            provenance=base.provenance,
        )

    def feature_importance(
        self,
        *,
        label: str | None = None,
        family: str | None = None,
        limit: int = 60,
    ) -> FeatureImportanceResponse:
        """Serve the measured importance table.

        Reads the parquet `xg importance` writes rather than computing on
        demand: a permutation run is minutes of work over 184 features and nine
        labels, and doing it inside a request would make the page look broken.
        """
        from xg_alonso.api.main import FeatureImportanceOut, FeatureImportanceResponse

        path = self._config.data_root / "gold" / "feature_importance.parquet"
        if not path.exists():
            raise LookupError(f"no importance table at {path}. Run `xg importance` to measure it.")

        frame = load_importance(path)
        if frame.is_empty():
            raise LookupError("the importance table is empty; re-run `xg importance`.")

        labels = sorted(frame["label"].unique().to_list())
        degenerate = sorted(frame.filter(pl.col("degenerate_label"))["label"].unique().to_list())
        weights = {
            str(row["label"]): float(row["label_weight"])
            for row in frame.select(["label", "label_weight"]).unique().iter_rows(named=True)
        }

        scoped = frame.filter(~pl.col("degenerate_label"))
        if label is not None:
            scoped = scoped.filter(pl.col("label") == label)
        if family is not None:
            scoped = scoped.filter(pl.col("family") == family)

        if scoped.is_empty():
            raise LookupError(f"no importance rows for label={label!r} family={family!r}")

        # A single label is already a like-for-like comparison, so weighting it
        # would only rescale every row by the same constant and make the numbers
        # harder to read against the CLI's output.
        weighted = (
            pl.col("relative_delta")
            if label is not None
            else pl.col("relative_delta") * pl.col("label_weight")
        )
        ranked = (
            scoped.with_columns(weighted.alias("__weighted"))
            .group_by(["feature_name", "family"])
            .agg(pl.col("__weighted").mean().alias("importance"))
            .sort("importance", descending=True)
        )

        per_label: dict[str, dict[str, float]] = {}
        for row in (
            scoped.group_by(["feature_name", "label"])
            .agg(pl.col("relative_delta").mean().alias("value"))
            .iter_rows(named=True)
        ):
            per_label.setdefault(str(row["feature_name"]), {})[str(row["label"])] = round(
                float(row["value"]), 6
            )

        stability = self._rank_stability(scoped)
        families = {
            str(row["family"]): round(float(row["importance"]), 6)
            for row in ranked.group_by("family")
            .agg(pl.col("importance").sum().alias("importance"))
            .iter_rows(named=True)
        }

        fingerprint = str(frame["model_fingerprint"][0])
        return FeatureImportanceResponse(
            features=[
                FeatureImportanceOut(
                    feature_name=str(row["feature_name"]),
                    family=str(row["family"]),
                    importance=round(float(row["importance"]), 6),
                    rank_stability=stability.get(str(row["feature_name"])),
                    per_label=per_label.get(str(row["feature_name"]), {}),
                )
                for row in ranked.head(limit).iter_rows(named=True)
            ],
            families=families,
            degenerate_labels=degenerate,
            labels=labels,
            label_weights=weights,
            folds_measured=int(frame["fold_index"].n_unique()),
            features_measured=int(frame["feature_name"].n_unique()),
            features_with_no_effect=int(ranked.filter(pl.col("importance") <= 0.0).height),
            catalogue_version=str(frame["catalogue_version"][0]),
            model_fingerprint=fingerprint,
            computed_at=frame["computed_at"][0],
            stale=(self._models is not None and self._models.fingerprint() != fingerprint),
        )

    @staticmethod
    def _rank_stability(scoped: pl.DataFrame) -> dict[str, float]:
        """Mean spread of a feature's rank across folds, within each label.

        Returns nothing when only one fold was measured. A zero there would read
        as "perfectly stable" when it actually means "never checked".
        """
        if scoped["fold_index"].n_unique() < 2:
            return {}

        ranked = scoped.with_columns(
            pl.col("relative_delta")
            .rank(method="ordinal", descending=True)
            .over(["fold_index", "label"])
            .alias("__rank")
        )
        spread = (
            ranked.group_by(["feature_name", "label"])
            .agg(pl.col("__rank").std().alias("__sd"))
            .group_by("feature_name")
            .agg(pl.col("__sd").mean().alias("stability"))
        )
        return {
            str(row["feature_name"]): round(float(row["stability"]), 2)
            for row in spread.iter_rows(named=True)
            if row["stability"] is not None
        }

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
