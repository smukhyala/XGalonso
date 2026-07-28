"""The slice-1 workflow, wired end to end.

This module is the composition root: the one place that knows about every
package. Keeping the wiring here is what lets each package below stay unaware of
the others — ``domain`` never learns there is a database, ``optimization`` never
learns there is an HTTP client.

The workflow proves every contract boundary the vertical slice exists to prove::

    bronze snapshot -> canonical tables -> point-in-time features
      -> component predictions -> domain scoring -> legal transfer search
      -> grounded explanation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    PlayerElementId,
    Season,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.recommendation import TransferRecommendation
from xg_alonso.contracts.squad import ChipState, ChipStatus, SquadPick, SquadState
from xg_alonso.domain.constraints import check_squad, check_starting_xi
from xg_alonso.domain.pricing import selling_price
from xg_alonso.domain.purchase_prices import parse_transfer_log, reconstruct_purchase_prices
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringRules
from xg_alonso.explanations.reasons import PopulationStats
from xg_alonso.features.career import build_career_features
from xg_alonso.features.catalogue import CATALOGUE_VERSION, build_catalogue
from xg_alonso.features.opponent import build_opponent_features, build_opponent_strength
from xg_alonso.features.slice1 import (
    SLICE1_FEATURE_SET_VERSION,
    build_slice1_features,
    build_team_gameweek_stats,
)
from xg_alonso.optimization.transfer import Candidate, best_single_transfer
from xg_alonso.pipelines.ingestion.fpl_client import FplApiClient
from xg_alonso.pipelines.normalization import (
    normalize_fixtures,
    normalize_gameweeks,
    normalize_players,
    normalize_teams,
)
from xg_alonso.prediction.baseline import predict_frame
from xg_alonso.prediction.form import apply_form_signals, load_signals
from xg_alonso.prediction.inference import predict_with_models
from xg_alonso.prediction.trained import ComponentModels

#: Records, per entry, whether purchase prices had to be assumed. Read by the
#: CLI so an approximate budget is never presented as an exact one. Decision
#: D3 reconstructs real prices from entry/{id}/transfers/, but that endpoint
#: returns [] until a season produces transfers.
PRICES_WERE_ASSUMED: dict[int, bool] = {}

#: Where sourced form signals are read from when no path is given. A missing
#: file means no outside information, which is the correct default: the system
#: must answer exactly as it always did when nobody has written anything down.
_DEFAULT_SIGNALS = Path(".data/signals/form_signals.json")

__all__ = [
    "PRICES_WERE_ASSUMED",
    "SliceContext",
    "build_context",
    "build_entities",
    "fetch_squad",
    "load_squad_file",
    "recommend",
]


@dataclass(frozen=True)
class SliceContext:
    """Everything the slice needs, loaded once from a bronze snapshot."""

    season: Season
    scoring: ScoringRules
    squad_rules: SquadRules
    players: pl.DataFrame
    teams: pl.DataFrame
    gameweeks: pl.DataFrame
    fixtures: pl.DataFrame
    player_stats: pl.DataFrame
    snapshot_sha256: str
    available_time: datetime

    def next_gameweek(self) -> GameweekId:
        """The gameweek being predicted.

        Prefers the API's own ``is_next`` flag, then the first unfinished
        gameweek. Falling back to "the first one" is correct in preseason, when
        nothing is current and nothing has finished.
        """
        if self.gameweeks.is_empty():
            return GameweekId(1)
        flagged = self.gameweeks.filter(pl.col("is_next"))
        if not flagged.is_empty():
            return GameweekId(int(flagged["gameweek_id"][0]))
        unfinished = self.gameweeks.filter(~pl.col("finished").fill_null(False))
        if not unfinished.is_empty():
            return GameweekId(int(unfinished.sort("gameweek_id")["gameweek_id"][0]))
        highest = self.gameweeks["gameweek_id"].max()
        return GameweekId(int(highest) if isinstance(highest, int) else 1)

    def deadline_for(self, gameweek: GameweekId) -> datetime:
        """The gameweek deadline — the natural prediction cutoff.

        Features are built as of this instant, so a recommendation reflects
        exactly what a manager could have known when the decision was due.
        """
        row = self.gameweeks.filter(pl.col("gameweek_id") == int(gameweek))
        if row.is_empty() or row["deadline_time"][0] is None:
            return self.available_time
        deadline: datetime = row["deadline_time"][0]
        return deadline

    def player_names(self) -> dict[PlayerCode, str]:
        return {
            PlayerCode(int(r["player_code"])): str(r["web_name"])
            for r in self.players.iter_rows(named=True)
        }


def build_context(
    bootstrap_payload: dict[str, Any],
    *,
    fixtures_payload: list[dict[str, Any]] | None,
    player_stats: pl.DataFrame,
    season: Season,
    snapshot_sha256: str,
    available_time: datetime,
) -> SliceContext:
    """Normalize a bronze snapshot into everything downstream needs."""
    return SliceContext(
        season=season,
        scoring=ScoringRules.from_bootstrap(
            bootstrap_payload,
            version=str(season),
            source_sha256=snapshot_sha256,
            fetched_at=available_time,
        ),
        squad_rules=SquadRules.from_bootstrap(
            bootstrap_payload, version=str(season), source_sha256=snapshot_sha256
        ),
        players=normalize_players(bootstrap_payload, season=season, available_time=available_time),
        teams=normalize_teams(bootstrap_payload, season=season, available_time=available_time),
        gameweeks=normalize_gameweeks(
            bootstrap_payload, season=season, available_time=available_time
        ),
        fixtures=normalize_fixtures(
            fixtures_payload or [], season=season, available_time=available_time
        ),
        player_stats=player_stats,
        snapshot_sha256=snapshot_sha256,
        available_time=available_time,
    )


def build_entities(context: SliceContext, *, cutoff: datetime) -> pl.DataFrame:
    """One prediction row per player, all sharing the gameweek deadline as cutoff.

    Carries the upcoming fixture when it is known. Fixtures are published well
    in advance, so ``opponent_team_id`` and ``was_home`` are inputs available at
    the deadline — only the result of the match is withheld.
    """
    entities = context.players.select(
        "player_code", "position", "team_id", "current_price", "web_name", "status"
    ).with_columns(pl.lit(cutoff).alias("prediction_timestamp"))

    fixtures = context.fixtures
    if fixtures.is_empty():
        return entities.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("opponent_team_id"),
            pl.lit(None, dtype=pl.Boolean).alias("was_home"),
        )

    upcoming = fixtures.filter(~pl.col("finished").fill_null(False))
    home = upcoming.select(
        pl.col("home_team_id").alias("team_id"),
        pl.col("away_team_id").alias("opponent_team_id"),
        pl.lit(True).alias("was_home"),
        pl.col("kickoff_time"),
    )
    away = upcoming.select(
        pl.col("away_team_id").alias("team_id"),
        pl.col("home_team_id").alias("opponent_team_id"),
        pl.lit(False).alias("was_home"),
        pl.col("kickoff_time"),
    )
    # Sort AFTER concatenating. Sorting each side separately and then taking the
    # first row per team always matches the home row, because every home row
    # precedes every away row — which made `is_home` identically true.
    next_fixture = (
        pl.concat([home, away])
        .sort(["kickoff_time", "team_id"], nulls_last=True)
        .unique(subset=["team_id"], keep="first", maintain_order=True)
        .drop("kickoff_time")
    )
    return entities.join(next_fixture, on="team_id", how="left")


def load_squad_file(path: Path) -> dict[str, Any]:
    """Load a squad from disk.

    Necessary, not merely convenient: ``entry/{id}/event/{gw}/picks/`` returns
    404 before a gameweek's deadline, so before the season opens there is no way
    to fetch a squad at all. Without this flag the whole squad path would be
    untestable until the first deadline passes.
    """
    data: dict[str, Any] = json.loads(path.read_text())
    if "picks" not in data:
        raise ValueError(
            f"{path} has no 'picks' key. Expected the shape of "
            "entry/{id}/event/{gw}/picks/, or {'picks': [...], 'bank': N, 'free_transfers': N}."
        )
    return data


def squad_from_payload(
    payload: dict[str, Any],
    *,
    context: SliceContext,
    entry_id: EntryId,
    gameweek: GameweekId,
) -> SquadState:
    """Build a squad from a picks payload, pricing each player for sale.

    Purchase price is taken from the payload when present and otherwise assumed
    equal to the current price. That assumption is explicit and conservative: it
    understates realisable value for players who have risen, so the optimizer
    proposes only transfers that would still be affordable if it is wrong.
    """
    by_element = {int(r["element_id"]): r for r in context.players.iter_rows(named=True)}

    # Tracks whether any price was assumed rather than evidenced. Surfacing this
    # matters: an assumed purchase price understates realisable value for a
    # player who has risen, so the budget is a lower bound, not a fact.
    assumed_prices = False
    picks: list[SquadPick] = []
    for index, pick in enumerate(payload["picks"], start=1):
        element_id = int(pick["element"])
        player = by_element.get(element_id)
        if player is None:
            raise KeyError(
                f"element {element_id} is not in this season's player list; "
                "the squad file and the snapshot describe different seasons"
            )

        current = TenthsOfMillion(int(player["current_price"]))
        declared = pick.get("purchase_price")
        if declared is None:
            assumed_prices = True
        purchase = TenthsOfMillion(int(declared if declared is not None else current))
        picks.append(
            SquadPick(
                player_code=PlayerCode(int(player["player_code"])),
                position=Position(str(player["position"])),
                team_id=TeamId(int(player["team_id"])),
                purchase_price=purchase,
                current_price=current,
                selling_price=selling_price(
                    purchase_price=purchase,
                    current_price=current,
                    rules=context.squad_rules,
                ),
                squad_slot=int(pick.get("position", index)),
                is_captain=bool(pick.get("is_captain", False)),
                is_vice_captain=bool(pick.get("is_vice_captain", False)),
            )
        )

    state = SquadState(
        entry_id=entry_id,
        gameweek=gameweek,
        picks=tuple(picks),
        bank=TenthsOfMillion(int(payload.get("bank", 0))),
        free_transfers=int(payload.get("free_transfers", 1)),
    )

    # Validate before returning. A squad file with an illegal starting XI would
    # otherwise produce a confident recommendation built on an impossible
    # position, and the user would only discover it in the game.
    violations = check_squad(
        state.picks, rules=context.squad_rules, bank=state.bank
    ) + check_starting_xi(state.starters, rules=context.squad_rules)
    if violations:
        detail = "; ".join(f"{v.rule}: {v.detail}" for v in violations)
        raise ValueError(f"squad is not legal — {detail}")

    PRICES_WERE_ASSUMED[int(entry_id)] = assumed_prices
    return state


def recommend(
    *,
    context: SliceContext,
    squad: SquadState,
    entry_id: EntryId,
    run_id: str,
    code_version: str,
    generated_at: datetime,
    horizon_gameweeks: int = 1,
    models: ComponentModels | None = None,
    form_signals_path: Path | None = None,
) -> tuple[TransferRecommendation, dict[PlayerCode, PlayerPrediction]]:
    """Run the full slice: features, predictions, and the best legal transfer.

    Args:
        models: Trained component models. When supplied, predictions come from
            the full catalogue and the fitted models; otherwise from the
            closed-form baseline over the eight legacy features, where clean
            sheets, goals conceded and saves are league-average constants.
    """
    gameweek = squad.gameweek
    cutoff = context.deadline_for(gameweek)

    entities = build_entities(context, cutoff=cutoff)

    if models is not None:
        # The trained path needs the full catalogue plus opponent context, and
        # the fixture — who each player faces — which is published before the
        # deadline and is therefore an input rather than a leak.
        features = build_catalogue(entities, player_stats=context.player_stats)
        features = build_opponent_features(
            features, opponent_strength=build_opponent_strength(context.player_stats)
        )
        # Career-length evidence. Every catalogue window is measured in
        # appearances and tops out at twenty, so a player with four elite
        # seasons and one with a single hot streak look identical inside it.
        features = build_career_features(features, player_stats=context.player_stats)
        predictions = predict_with_models(
            features,
            models=models,
            rules=context.scoring,
            from_gameweek=gameweek,
            data_cutoff=cutoff,
            predicted_at=generated_at,
            run_id=run_id,
            code_version=code_version,
            feature_set_version=CATALOGUE_VERSION,
            horizon_gameweeks=horizon_gameweeks,
        )
    else:
        team_stats = build_team_gameweek_stats(context.player_stats, context.players)
        features = build_slice1_features(
            entities, player_stats=context.player_stats, team_stats=team_stats
        )
        predictions = predict_frame(
            features,
            rules=context.scoring,
            from_gameweek=gameweek,
            data_cutoff=cutoff,
            predicted_at=generated_at,
            run_id=run_id,
            code_version=code_version,
            feature_set_version=SLICE1_FEATURE_SET_VERSION,
            horizon_gameweeks=horizon_gameweeks,
        )
    # Outside information the API cannot see — a poor international tournament,
    # a public fallout — scaled within a hard clamp and only where a sourced
    # signal exists. Evaluated against the deadline rather than the wall clock,
    # so a backtest sees what was live then.
    predictions = apply_form_signals(
        predictions, load_signals(form_signals_path or _DEFAULT_SIGNALS), at=cutoff
    )
    by_code = {p.player_code: p for p in predictions}

    # Unavailable players are excluded from the buy list rather than penalised.
    # Recommending an injured player is a category error, not a scoring nuance.
    available = {
        int(r["player_code"])
        for r in context.players.iter_rows(named=True)
        if r.get("status") in (None, "a", "d")
    }

    price_by_code = {
        PlayerCode(int(r["player_code"])): TenthsOfMillion(int(r["current_price"]))
        for r in context.players.iter_rows(named=True)
    }
    team_by_code = {
        PlayerCode(int(r["player_code"])): TeamId(int(r["team_id"]))
        for r in context.players.iter_rows(named=True)
    }

    candidates = [
        Candidate(
            player_code=prediction.player_code,
            position=prediction.position,
            team_id=team_by_code[prediction.player_code],
            price=price_by_code[prediction.player_code],
            prediction=prediction,
        )
        for prediction in predictions
        if int(prediction.player_code) in available
    ]

    # League reference values, so a reason can say where a number sits rather
    # than only what it is. Built from the whole predicted population, which is
    # why it is assembled here rather than inside the optimizer.
    population = PopulationStats.from_predictions(predictions, prices=price_by_code)

    recommendation = best_single_transfer(
        squad,
        candidates=candidates,
        predictions=by_code,
        rules=context.squad_rules,
        entry_id=entry_id,
        gameweek=gameweek,
        generated_at=generated_at,
        run_id=run_id,
        optimizer_config_hash=SLICE1_FEATURE_SET_VERSION,
        horizon_gameweeks=horizon_gameweeks,
        population=population,
    )
    return recommendation, by_code


def fetch_squad(
    *,
    client: FplApiClient,
    context: SliceContext,
    entry_id: EntryId,
    gameweek: GameweekId,
) -> SquadState:
    """Load a manager's squad from the public API.

    This is the product's front door. It reads three public endpoints and needs
    no authentication (D3):

    - ``entry/{id}/event/{gw}/picks/`` — the fifteen players and their roles
    - ``entry/{id}/transfers/`` — replayed to recover what was paid for each
    - ``entry/{id}/history/`` — bank and chips played

    Purchase prices come from the transfer log rather than being assumed equal
    to the current price, which is the difference between a correct budget and
    one that silently understates every rise.

    Raises:
        httpx.HTTPStatusError: notably 404 from ``picks`` before a gameweek's
            deadline, when the squad genuinely does not exist yet. Callers fall
            back to ``--squad-file``.
    """
    picks_payload = json.loads(client.entry_picks(int(entry_id), int(gameweek)).payload)
    transfers_payload = json.loads(client.entry_transfers(int(entry_id)).payload)
    history_payload = json.loads(client.entry_history(int(entry_id)).payload)

    element_ids = [PlayerElementId(int(p["element"])) for p in picks_payload.get("picks", [])]
    opening = {
        PlayerElementId(int(r["element_id"])): TenthsOfMillion(int(r["current_price"]))
        for r in context.players.iter_rows(named=True)
    }
    reconstruction = reconstruct_purchase_prices(
        current_squad=element_ids,
        transfers=parse_transfer_log(transfers_payload),
        opening_prices=opening,
    )

    # entry_history carries per-gameweek state; the row for this gameweek holds
    # the bank as it stood at the deadline.
    bank = 0
    for row in history_payload.get("current", []):
        if int(row.get("event", -1)) == int(gameweek):
            bank = int(row.get("bank", 0))
            break

    played = {str(c.get("name")) for c in history_payload.get("chips", [])}
    chips = ChipState(
        wildcard=ChipStatus.PLAYED if "wildcard" in played else ChipStatus.AVAILABLE,
        free_hit=ChipStatus.PLAYED if "freehit" in played else ChipStatus.AVAILABLE,
        bench_boost=ChipStatus.PLAYED if "bboost" in played else ChipStatus.AVAILABLE,
        triple_captain=ChipStatus.PLAYED if "3xc" in played else ChipStatus.AVAILABLE,
    )

    enriched = dict(picks_payload)
    enriched["bank"] = bank
    enriched["picks"] = [
        {
            **pick,
            "purchase_price": int(
                reconstruction.purchase_prices.get(
                    PlayerElementId(int(pick["element"])),
                    opening.get(PlayerElementId(int(pick["element"])), TenthsOfMillion(0)),
                )
            ),
        }
        for pick in picks_payload.get("picks", [])
    ]

    state = squad_from_payload(enriched, context=context, entry_id=entry_id, gameweek=gameweek)
    # Prices came from evidence, not assumption — record that so the CLI does
    # not warn about a budget it can actually stand behind.
    PRICES_WERE_ASSUMED[int(entry_id)] = not reconstruction.complete
    return state.model_copy(update={"chips": chips})
