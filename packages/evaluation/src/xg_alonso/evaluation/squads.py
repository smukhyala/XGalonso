"""Where an evaluation starts from, and why more than one starting squad.

The published headline walked one season from one starting squad. The squad was
the most expensive legal fifteen — a reasonable choice, and completely
unexamined: nothing said whether it was a favourable place to start, and a
policy comparison run from a single unusually good or unusually poor squad
measures the squad as much as the policies.

So this module builds *cohorts*, and it reports where each squad sits. Every
squad carries an ``ex_ante_percentile`` — how strong its best eleven looked
under the predictions available at its own deadline. That single number answers
"was the headline evaluated from a lucky start" as a measurement rather than an
opinion, and it goes in the report.

**On real historical squads.** The honest answer is that they are unobtainable.
``entry/{id}/event/{gw}/picks/`` is season-scoped and no entry snapshots were
ever stored, so nobody's actual 2022-25 squad can be recovered. What *is*
stored is ``selected`` — real ownership, per player, per gameweek, non-null in
every season — so :func:`template_ownership_squad` reconstructs what the
population actually held. It is a substitute, it is named as one, and the
report says so.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from xg_alonso.contracts.evaluation import SquadCohort, SquadSource
from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.constraints import check_squad
from xg_alonso.domain.rules import SquadRules
from xg_alonso.optimization.lineup import starting_xi_points

__all__ = [
    "StartingSquad",
    "load_squad_artifact",
    "most_expensive_legal_squad",
    "random_legal_squads",
    "squad_identity",
    "template_ownership_squad",
    "write_squad_artifact",
]


def squad_identity(squad: SquadState) -> str:
    """A stable id from the squad's contents, not from how it was built.

    Reordering the picks must not change it, or the same squad generated twice
    would occupy two run directories and the pairing in the statistics would
    silently break.
    """
    import hashlib

    payload = json.dumps(
        {
            "players": sorted([int(p.player_code), int(p.purchase_price)] for p in squad.picks),
            "bank": int(squad.bank),
            "free_transfers": squad.free_transfers,
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class StartingSquad:
    """One starting position, with where it came from and how strong it looked."""

    squad_id: str
    squad: SquadState
    source: SquadSource
    cohort: SquadCohort = SquadCohort.UNBANDED
    seed: int | None = None
    ex_ante_xi_points: float = 0.0
    """Best legal XI under the predictions available at this squad's deadline."""
    ex_ante_percentile: float | None = None
    """Where that sits among a comparison pool. ``None`` when none was drawn."""
    # RUF009 guards against mutable or expensive dataclass defaults. `TenthsOfMillion`
    # is a NewType, so this call is the identity at runtime and is neither.
    spend: TenthsOfMillion = TenthsOfMillion(0)  # noqa: RUF009
    provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "squad_id": self.squad_id,
            "source": str(self.source),
            "cohort": str(self.cohort),
            "seed": self.seed,
            "ex_ante_xi_points": self.ex_ante_xi_points,
            "ex_ante_percentile": self.ex_ante_percentile,
            "spend": int(self.spend),
            "provenance": self.provenance,
            "squad": self.squad.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StartingSquad:
        return cls(
            squad_id=str(payload["squad_id"]),
            squad=SquadState.model_validate(payload["squad"]),
            source=SquadSource(payload["source"]),
            cohort=SquadCohort(payload["cohort"]),
            seed=payload.get("seed"),
            ex_ante_xi_points=float(payload.get("ex_ante_xi_points", 0.0)),
            ex_ante_percentile=payload.get("ex_ante_percentile"),
            spend=TenthsOfMillion(int(payload.get("spend", 0))),
            provenance=dict(payload.get("provenance", {})),
        )


def write_squad_artifact(squad: StartingSquad, path: Path) -> None:
    """Materialise a squad so a later run loads it rather than rebuilding it.

    Necessary for anything solver-built: HiGHS can break ties differently on a
    degenerate optimum across platforms, so "reproducible from a seed" is not
    available and "reproducible from an artifact" is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(squad.to_dict(), indent=2, sort_keys=True))
    tmp.replace(path)


def load_squad_artifact(path: Path) -> StartingSquad:
    return StartingSquad.from_dict(json.loads(path.read_text()))


def _pick(row: Mapping[str, Any], slot: int) -> SquadPick:
    price = TenthsOfMillion(int(row["current_price"]))
    return SquadPick(
        player_code=PlayerCode(int(row["player_code"])),
        position=Position(str(row["position"])),
        team_id=TeamId(int(row["team_id"])),
        purchase_price=price,
        current_price=price,
        selling_price=price,
        squad_slot=slot,
        # No captain is set. The XI and the armband are chosen from predictions
        # at decision time; baking one in at construction is how earlier runs
        # ended up captaining whoever sat in slot one.
    )


def _assemble(
    chosen: Mapping[str, list[Mapping[str, Any]]],
    *,
    rules: SquadRules,
    entry_id: EntryId,
    gameweek: GameweekId,
) -> SquadState:
    """Order the picks into a legal starting eleven plus a bench."""
    starting = chosen["GKP"][:1] + chosen["DEF"][:4] + chosen["MID"][:4] + chosen["FWD"][:2]
    bench = chosen["GKP"][1:] + chosen["DEF"][4:] + chosen["MID"][4:] + chosen["FWD"][2:]

    picks = [_pick(row, slot) for slot, row in enumerate(starting + bench, start=1)]
    spend = sum(int(p.current_price) for p in picks)
    return SquadState(
        entry_id=entry_id,
        gameweek=gameweek,
        picks=tuple(picks),
        bank=TenthsOfMillion(int(rules.total_budget) - spend),
        free_transfers=1,
    )


def _greedy_fill(
    players: pl.DataFrame,
    *,
    rules: SquadRules,
    order_by: str,
    descending: bool,
) -> dict[str, list[Mapping[str, Any]]]:
    """Fill every position by a ranking, respecting budget and club limits.

    Reserves a floor for the slots not yet filled, so spending early cannot
    leave the squad unable to complete itself legally.
    """
    budget = int(rules.total_budget)
    club_count: dict[int, int] = {}
    chosen: dict[str, list[Mapping[str, Any]]] = {}
    remaining_slots = sum(r.squad_select for r in rules.positions)
    spend = 0

    for rule in sorted(rules.positions, key=lambda r: -r.squad_select):
        pool = list(
            players.filter(pl.col("position") == rule.position.value)
            .sort(order_by, descending=descending)
            .iter_rows(named=True)
        )
        cheapest = min((int(r["current_price"]) for r in pool), default=40)
        picked: list[Mapping[str, Any]] = []
        for row in pool:
            if len(picked) == rule.squad_select:
                break
            team = int(row["team_id"])
            if club_count.get(team, 0) >= rules.max_per_club:
                continue
            price = int(row["current_price"])
            slots_after = remaining_slots - len(picked) - 1
            if spend + price + slots_after * cheapest > budget:
                continue
            club_count[team] = club_count.get(team, 0) + 1
            picked.append(row)
            spend += price
        if len(picked) < rule.squad_select:
            raise ValueError(
                f"could not fill {rule.position.value} within budget; only "
                f"{len(picked)} of {rule.squad_select} affordable"
            )
        remaining_slots -= rule.squad_select
        chosen[rule.position.value] = picked
    return chosen


def most_expensive_legal_squad(
    players: pl.DataFrame, *, rules: SquadRules, entry_id: EntryId, gameweek: GameweekId
) -> StartingSquad:
    """The most expensive legal fifteen that fits, from opening prices only.

    Moved verbatim from ``cli/main.py::_opening_squad`` so the legacy headline
    can be reproduced exactly, and so a function that is load-bearing for the
    metric stops living in a CLI command.

    Its own history is the argument for the cohorts around it: an earlier
    version picked the *cheapest* legal fifteen, left 36m unspent, and produced
    a hold baseline of players who barely appear. Everything beat that, so the
    backtest reported a 100% beat-hold rate — a number that measured the
    starting squad rather than the recommendations.
    """
    chosen = _greedy_fill(players, rules=rules, order_by="current_price", descending=True)
    squad = _assemble(chosen, rules=rules, entry_id=entry_id, gameweek=gameweek)
    return StartingSquad(
        squad_id=squad_identity(squad),
        squad=squad,
        source=SquadSource.MOST_EXPENSIVE_LEGAL,
        spend=TenthsOfMillion(sum(int(p.current_price) for p in squad.picks)),
        provenance={"rule": "most expensive legal fifteen at opening prices"},
    )


def template_ownership_squad(
    players: pl.DataFrame,
    player_stats: pl.DataFrame,
    *,
    rules: SquadRules,
    season: str,
    gameweek: GameweekId,
    entry_id: EntryId,
) -> StartingSquad:
    """The most-owned legal fifteen at this gameweek's deadline.

    **The substitute for real historical squads, and deliberately named as one.**
    Actual entry picks are unobtainable — the endpoint is season-scoped and no
    snapshots were stored — but ``selected`` records what the population really
    held, so this is what managers collectively owned rather than what a
    heuristic would have chosen.

    Ownership is read from gameweek ``g - 1`` only. Gameweek ``g``'s ownership
    is recorded alongside its result, which is after its own deadline; using it
    would let the squad know which players the crowd piled into *because* of
    what happened.
    """
    previous = int(gameweek) - 1
    prior_rows = player_stats.filter(
        (pl.col("season") == season) & (pl.col("gameweek_id") == previous)
    )
    if prior_rows.is_empty():
        # No prior gameweek — `gameweek` is the season opener, or the window
        # starts before any recorded data. Every player would fill-null to zero
        # ownership, and `_greedy_fill` would then rank a 600-player field by a
        # column that is constant, returning whichever fifteen the frame
        # happened to order first. That is not a template squad; it is an
        # arbitrary one wearing the label, and it would enter a report as the
        # "what the crowd owned" baseline.
        raise ValueError(
            f"no gameweek {previous} ownership for {season}, so a template squad for "
            f"gameweek {gameweek} cannot be built. Ownership must be read from the "
            "gameweek before the one being started, and there is none."
        )
    owned = prior_rows.group_by("player_code").agg(pl.col("selected").max().alias("selected"))
    ranked = players.join(owned, on="player_code", how="left").with_columns(
        pl.col("selected").fill_null(0)
    )
    chosen = _greedy_fill(ranked, rules=rules, order_by="selected", descending=True)
    squad = _assemble(chosen, rules=rules, entry_id=entry_id, gameweek=gameweek)
    return StartingSquad(
        squad_id=squad_identity(squad),
        squad=squad,
        source=SquadSource.TEMPLATE_OWNERSHIP,
        spend=TenthsOfMillion(sum(int(p.current_price) for p in squad.picks)),
        provenance={
            "rule": "most-owned legal fifteen",
            "ownership_from_gameweek": str(previous),
            "caveat": "a substitute for real entry picks, which are unobtainable",
        },
    )


def random_legal_squads(
    players: pl.DataFrame,
    *,
    rules: SquadRules,
    seed: int,
    count: int,
    entry_id: EntryId,
    gameweek: GameweekId,
) -> tuple[StartingSquad, ...]:
    """Legal squads drawn by seeded shuffle, for the comparison pool.

    Each draw reshuffles the player pool and fills greedily, so the result is
    reproducible from the seed alone and genuinely varies between seeds — both
    asserted, because a generator that quietly ignores its seed produces a
    "cohort" of identical squads and nothing about the output would say so.
    """
    squads: list[StartingSquad] = []
    for index in range(count):
        rng = random.Random(seed + index)
        shuffled = players.with_columns(
            pl.Series("__order", [rng.random() for _ in range(players.height)])
        )
        chosen = _greedy_fill(shuffled, rules=rules, order_by="__order", descending=True)
        squad = _assemble(chosen, rules=rules, entry_id=entry_id, gameweek=gameweek)
        squads.append(
            StartingSquad(
                squad_id=squad_identity(squad),
                squad=squad,
                source=SquadSource.RANDOM_LEGAL,
                seed=seed + index,
                spend=TenthsOfMillion(sum(int(p.current_price) for p in squad.picks)),
                provenance={"rule": "seeded shuffle, greedy legal fill"},
            )
        )
    return tuple(squads)


def score_ex_ante(
    squad: StartingSquad,
    *,
    predictions: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    pool: Sequence[float] = (),
) -> StartingSquad:
    """Attach how strong this squad looked before the season was walked.

    Args:
        pool: XI scores of a comparison population. Supplying it turns the
            score into a percentile, which is the number that answers whether a
            headline was evaluated from a favourable start.

    Uses only the predictions available at the squad's own deadline, so the
    measurement is *ex ante* — a squad ranked by how it turned out would be a
    different and useless quantity.
    """
    points = starting_xi_points(squad.squad.picks, dict(predictions), rules)
    percentile: float | None = None
    if pool:
        below = sum(1 for value in pool if value <= points)
        percentile = below / len(pool)
    return StartingSquad(
        squad_id=squad.squad_id,
        squad=squad.squad,
        source=squad.source,
        cohort=squad.cohort,
        seed=squad.seed,
        ex_ante_xi_points=round(points, 6),
        ex_ante_percentile=percentile,
        spend=squad.spend,
        provenance=squad.provenance,
    )


def is_legal(squad: StartingSquad, *, rules: SquadRules) -> bool:
    """Whether the game would accept this as a starting position."""
    return not check_squad(squad.squad.picks, rules=rules, bank=squad.squad.bank)
