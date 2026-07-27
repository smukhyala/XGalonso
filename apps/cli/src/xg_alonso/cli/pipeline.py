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
    Season,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.recommendation import TransferRecommendation
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.constraints import check_squad, check_starting_xi
from xg_alonso.domain.pricing import selling_price
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringRules
from xg_alonso.features.slice1 import (
    SLICE1_FEATURE_SET_VERSION,
    build_slice1_features,
    build_team_gameweek_stats,
)
from xg_alonso.optimization.transfer import Candidate, best_single_transfer
from xg_alonso.pipelines.normalization import (
    normalize_fixtures,
    normalize_gameweeks,
    normalize_players,
    normalize_teams,
)
from xg_alonso.prediction.baseline import predict_frame

#: Records, per entry, whether purchase prices had to be assumed. Read by the
#: CLI so an approximate budget is never presented as an exact one. Decision
#: D3 reconstructs real prices from entry/{id}/transfers/, but that endpoint
#: returns [] until a season produces transfers.
PRICES_WERE_ASSUMED: dict[int, bool] = {}

__all__ = [
    "PRICES_WERE_ASSUMED",
    "SliceContext",
    "build_context",
    "build_entities",
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
    """One prediction row per player, all sharing the gameweek deadline as cutoff."""
    return context.players.select(
        "player_code", "position", "team_id", "current_price", "web_name", "status"
    ).with_columns(pl.lit(cutoff).alias("prediction_timestamp"))


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
) -> tuple[TransferRecommendation, dict[PlayerCode, PlayerPrediction]]:
    """Run the full slice: features, predictions, and the best legal transfer."""
    gameweek = squad.gameweek
    cutoff = context.deadline_for(gameweek)

    entities = build_entities(context, cutoff=cutoff)
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
    )
    return recommendation, by_code
