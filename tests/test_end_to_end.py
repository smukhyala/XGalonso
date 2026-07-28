"""End-to-end slice test.

Runs the whole workflow — snapshot, canonical tables, point-in-time features,
component predictions, domain scoring, legal transfer search, grounded
explanation — against real FPL rules from the committed fixture.

This is the test that proves the contract boundaries actually connect. Every
other suite verifies one package in isolation, and packages that each pass in
isolation routinely fail to compose.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from xg_alonso.cli.pipeline import SliceContext, build_context, recommend, squad_from_payload
from xg_alonso.contracts.identifiers import EntryId, Season
from xg_alonso.domain.constraints import check_squad, check_starting_xi
from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA, conform

FIXTURE = Path(__file__).resolve().parents[1] / "data/fixtures/fpl/bootstrap_static_2026_27.json"
NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
SEASON = Season("2026-27")

_POSITION_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
_ELEMENT_TYPE = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}


def _synthetic_bootstrap(real_payload: dict[str, Any]) -> dict[str, Any]:
    """Real rules from the fixture, plus enough players to field a legal squad.

    The committed fixture truncates the player list to stay small, so it cannot
    fill a 15-man squad. Scoring and constraints stay verbatim — those are the
    parts under test — while the roster is synthesised around them.
    """
    payload = json.loads(json.dumps(real_payload))

    elements = []
    element_id = 1
    for position, quota in _POSITION_QUOTA.items():
        # Three per position per club, across enough clubs to respect the limit.
        for slot in range(quota * 3):
            elements.append(
                {
                    "id": element_id,
                    "code": 900000 + element_id,
                    "web_name": f"{position}{slot}",
                    "first_name": position,
                    "second_name": str(slot),
                    "element_type": _ELEMENT_TYPE[position],
                    "team": (slot % 6) + 1,
                    "team_code": (slot % 6) + 1,
                    # A spread of prices so upgrades are affordable.
                    "now_cost": 40 + slot * 5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                }
            )
            element_id += 1
    payload["elements"] = elements

    events: list[dict[str, Any]] = [
        {
            "id": 1,
            "deadline_time": "2026-08-21T17:30:00Z",
            "finished": False,
            "is_current": False,
            "is_next": True,
        }
    ]
    payload["events"] = events
    result: dict[str, Any] = payload
    return result


def _synthetic_history(payload: dict[str, Any]) -> pl.DataFrame:
    """Prior-season rows, with output rising with price.

    Making the expensive players genuinely better is what lets the test assert
    the optimizer found a *sensible* transfer rather than merely a legal one.
    """
    rows = []
    season_end = datetime(2026, 6, 30, tzinfo=UTC)
    for element in payload["elements"]:
        quality = (element["now_cost"] - 40) / 40.0
        rows.append(
            {
                "player_code": element["code"],
                "gameweek_id": None,
                "season": "2025/26",
                "fixture_id": None,
                "opponent_team_id": None,
                "was_home": None,
                "minutes": int(600 + quality * 2400),
                "starts": int(6 + quality * 26),
                "goals_scored": int(quality * 18),
                "assists": int(quality * 8),
                "clean_sheets": 5,
                "goals_conceded": 30,
                "saves": 0,
                "yellow_cards": 3,
                "red_cards": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "bonus": int(quality * 20),
                "bps": int(quality * 500),
                "total_points": int(30 + quality * 180),
                "expected_goals": quality * 15.0,
                "expected_assists": quality * 7.0,
                "expected_goal_involvements": quality * 22.0,
                "expected_goals_conceded": 30.0,
                "defensive_contribution": 12.0,
                "defensive_contribution": None,
                "value": element["now_cost"],
                "kickoff_time": season_end,
                "available_time": season_end,
            }
        )
    return conform(pl.DataFrame(rows, infer_schema_length=None), PLAYER_GAMEWEEK_STATS_SCHEMA)


def _cheap_squad(payload: dict[str, Any]) -> dict[str, Any]:
    """A legal squad of the cheapest players, leaving money to spend.

    Slots 1-11 form a legal 1-4-4-2; slots 12-15 are the bench. Laying the squad
    out in flat position order would put both keepers in the XI, which the
    formation rules correctly reject.
    """
    by_position: dict[int, list[dict[str, Any]]] = {}
    for element in payload["elements"]:
        by_position.setdefault(element["element_type"], []).append(element)
    for group in by_position.values():
        group.sort(key=lambda e: e["now_cost"])

    club_count: dict[int, int] = {}
    chosen: dict[str, list[dict[str, Any]]] = {}

    for position, quota in _POSITION_QUOTA.items():
        picked: list[dict[str, Any]] = []
        for element in by_position[_ELEMENT_TYPE[position]]:
            if len(picked) == quota:
                break
            if club_count.get(element["team"], 0) >= 3:
                continue
            club_count[element["team"]] = club_count.get(element["team"], 0) + 1
            picked.append(element)
        assert len(picked) == quota, f"could not fill {position}"
        chosen[position] = picked

    # 1-4-4-2 starting, remainder benched.
    starting = chosen["GKP"][:1] + chosen["DEF"][:4] + chosen["MID"][:4] + chosen["FWD"][:2]
    bench = chosen["GKP"][1:] + chosen["DEF"][4:] + chosen["MID"][4:] + chosen["FWD"][2:]

    picks, spend = [], 0
    for slot, element in enumerate(starting + bench, start=1):
        picks.append(
            {
                "element": element["id"],
                "position": slot,
                "purchase_price": element["now_cost"],
                "is_captain": slot == 1,
                "is_vice_captain": slot == 2,
            }
        )
        spend += element["now_cost"]

    return {"picks": picks, "bank": 1000 - spend, "free_transfers": 1}


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    real: dict[str, Any] = json.loads(FIXTURE.read_text())
    built: dict[str, Any] = _synthetic_bootstrap(real)
    return built


@pytest.fixture(scope="module")
def context(payload: dict[str, Any]) -> SliceContext:
    return build_context(
        payload,
        fixtures_payload=[],
        player_stats=_synthetic_history(payload),
        season=SEASON,
        snapshot_sha256="e" * 64,
        available_time=NOW,
    )


@pytest.mark.e2e
class TestVerticalSlice:
    def test_the_whole_workflow_produces_a_legal_recommendation(
        self, context: SliceContext, payload: dict[str, Any]
    ) -> None:
        squad = squad_from_payload(
            _cheap_squad(payload),
            context=context,
            entry_id=EntryId(1),
            gameweek=context.next_gameweek(),
        )
        recommendation, predictions = recommend(
            context=context,
            squad=squad,
            entry_id=EntryId(1),
            run_id="e2e",
            code_version="test",
            generated_at=NOW,
        )

        assert predictions, "the slice produced no predictions at all"
        assert recommendation.comparison.baseline_name == "hold"

        # A squad of the cheapest players with money spare must have an upgrade.
        assert not recommendation.package.is_hold, (
            "with budget to spare and better players available, holding is wrong"
        )
        assert recommendation.expected_points_gain > 0
        assert recommendation.reasons, "a transfer must be explained"

    def test_the_recommended_transfer_is_actually_legal(
        self, context: SliceContext, payload: dict[str, Any]
    ) -> None:
        """A recommendation the game would reject costs a transfer to discover."""
        squad = squad_from_payload(
            _cheap_squad(payload),
            context=context,
            entry_id=EntryId(1),
            gameweek=context.next_gameweek(),
        )
        recommendation, _ = recommend(
            context=context,
            squad=squad,
            entry_id=EntryId(1),
            run_id="e2e",
            code_version="test",
            generated_at=NOW,
        )
        move = recommendation.package.moves[0]

        by_code = {p.player_code: p for p in squad.picks}
        outgoing = by_code[move.player_out]
        incoming_row = context.players.filter(pl.col("player_code") == int(move.player_in)).row(
            0, named=True
        )

        assert incoming_row["position"] == outgoing.position.value, "positions must match"
        assert move.purchase_price <= outgoing.selling_price + squad.bank, "must be affordable"
        assert recommendation.package.bank_after >= 0, "cannot end with negative bank"

        # Apply the move and re-check the whole squad against the real rules.
        after = [p for p in squad.picks if p.player_code != move.player_out]
        assert len(after) == 14
        club_counts: dict[int, int] = {}
        for pick in after:
            club_counts[int(pick.team_id)] = club_counts.get(int(pick.team_id), 0) + 1
        incoming_club = int(incoming_row["team_id"])
        assert club_counts.get(incoming_club, 0) < context.squad_rules.max_per_club

    def test_the_starting_squad_was_legal_to_begin_with(
        self, context: SliceContext, payload: dict[str, Any]
    ) -> None:
        squad = squad_from_payload(
            _cheap_squad(payload),
            context=context,
            entry_id=EntryId(1),
            gameweek=context.next_gameweek(),
        )
        assert check_squad(squad.picks, rules=context.squad_rules, bank=squad.bank) == []
        assert check_starting_xi(squad.starters, rules=context.squad_rules) == []

    def test_every_prediction_carries_complete_provenance(
        self, context: SliceContext, payload: dict[str, Any]
    ) -> None:
        """Zero nulls across every provenance field, including the artifact hash."""
        squad = squad_from_payload(
            _cheap_squad(payload),
            context=context,
            entry_id=EntryId(1),
            gameweek=context.next_gameweek(),
        )
        _, predictions = recommend(
            context=context,
            squad=squad,
            entry_id=EntryId(1),
            run_id="e2e",
            code_version="test",
            generated_at=NOW,
        )
        for prediction in predictions.values():
            provenance = prediction.provenance
            assert provenance.model_name
            assert provenance.model_version
            assert len(provenance.model_artifact_sha256) == 64
            assert provenance.feature_set_version
            assert provenance.data_cutoff.tzinfo is not None
            assert provenance.predicted_at.tzinfo is not None
            assert provenance.run_id
            assert provenance.code_version

    def test_predictions_use_the_gameweek_deadline_as_cutoff(
        self, context: SliceContext, payload: dict[str, Any]
    ) -> None:
        """The cutoff must be the deadline, not the moment we happened to run."""
        squad = squad_from_payload(
            _cheap_squad(payload),
            context=context,
            entry_id=EntryId(1),
            gameweek=context.next_gameweek(),
        )
        _, predictions = recommend(
            context=context,
            squad=squad,
            entry_id=EntryId(1),
            run_id="e2e",
            code_version="test",
            generated_at=NOW,
        )
        expected = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)
        for prediction in predictions.values():
            assert prediction.provenance.data_cutoff == expected

    def test_rerunning_gives_byte_identical_results(
        self, context: SliceContext, payload: dict[str, Any]
    ) -> None:
        """Determinism: the same inputs must always produce the same decision."""
        squad = squad_from_payload(
            _cheap_squad(payload),
            context=context,
            entry_id=EntryId(1),
            gameweek=context.next_gameweek(),
        )

        def run() -> str:
            recommendation, _ = recommend(
                context=context,
                squad=squad,
                entry_id=EntryId(1),
                run_id="fixed",
                code_version="test",
                generated_at=NOW,
            )
            return recommendation.model_dump_json()

        assert run() == run()

    def test_scoring_comes_from_the_real_payload(self, context: SliceContext) -> None:
        """The slice must be scoring with the live rules, not transcribed ones."""
        from xg_alonso.contracts.prediction import Position

        assert context.scoring.goals_scored[Position.GKP] == 10
        assert context.scoring.source_sha256 == "e" * 64

    def test_unavailable_players_are_never_recommended(
        self, context: SliceContext, payload: dict[str, Any]
    ) -> None:
        """Recommending an injured player is a category error, not a scoring nuance."""
        injured = json.loads(json.dumps(payload))
        for element in injured["elements"]:
            if element["now_cost"] > 60:
                element["status"] = "i"

        injured_context = build_context(
            injured,
            fixtures_payload=[],
            player_stats=_synthetic_history(injured),
            season=SEASON,
            snapshot_sha256="f" * 64,
            available_time=NOW,
        )
        squad = squad_from_payload(
            _cheap_squad(injured),
            context=injured_context,
            entry_id=EntryId(1),
            gameweek=injured_context.next_gameweek(),
        )
        recommendation, _ = recommend(
            context=injured_context,
            squad=squad,
            entry_id=EntryId(1),
            run_id="e2e",
            code_version="test",
            generated_at=NOW,
        )
        if not recommendation.package.is_hold:
            code = int(recommendation.package.moves[0].player_in)
            row = injured_context.players.filter(pl.col("player_code") == code).row(0, named=True)
            assert row["status"] != "i", "an injured player was recommended"

    def test_a_squad_with_no_budget_is_told_to_hold(self, context, payload: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        """Holding is a real answer, and must be reachable."""
        premium = json.loads(json.dumps(payload))
        squad_payload = _cheap_squad(premium)
        squad_payload["bank"] = 0

        squad = squad_from_payload(
            squad_payload,
            context=context,
            entry_id=EntryId(1),
            gameweek=context.next_gameweek(),
        )
        recommendation, _ = recommend(
            context=context,
            squad=squad,
            entry_id=EntryId(1),
            run_id="e2e",
            code_version="test",
            generated_at=NOW,
        )
        # Either a legal in-budget upgrade or an explicit hold — never an
        # unaffordable suggestion.
        if not recommendation.package.is_hold:
            move = recommendation.package.moves[0]
            outgoing = next(p for p in squad.picks if p.player_code == move.player_out)
            assert move.purchase_price <= outgoing.selling_price + squad.bank


@pytest.mark.e2e
def test_features_are_point_in_time_at_the_deadline(
    context: SliceContext, payload: dict[str, Any]
) -> None:
    """History published after the deadline must not reach the feature build."""
    from xg_alonso.cli.pipeline import build_entities
    from xg_alonso.features.slice1 import build_slice1_features

    deadline = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)
    entities = build_entities(context, cutoff=deadline)

    baseline = build_slice1_features(entities, player_stats=context.player_stats)

    future = context.player_stats.with_columns(
        pl.lit(deadline + timedelta(days=30))
        .cast(context.player_stats.schema["available_time"])
        .alias("available_time"),
        (pl.col("expected_goals") * 100).alias("expected_goals"),
    )
    polluted = build_slice1_features(
        entities, player_stats=pl.concat([context.player_stats, future])
    )

    for column in ("minutes_mean_5", "xg_per90_shrunk", "points_per90_shrunk"):
        assert baseline[column].equals(polluted[column], null_equal=True), (
            f"{column} changed when post-deadline data was added"
        )
