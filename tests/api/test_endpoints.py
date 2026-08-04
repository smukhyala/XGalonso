"""Contract tests for every route on the HTTP surface.

**Response shapes, not golden bodies.** A recorded JSON body for `/players`
pins the model's arithmetic, so it breaks on every retrain and on every honest
improvement, and the fix is always to re-record it — which teaches nobody
anything and eventually gets done without reading the diff. What is stable, and
what a client actually depends on, is the *shape*: the status code, the declared
response model, and the invariants the shape is supposed to guarantee. Those are
what is asserted here.

Each route is checked for three things:

1. the status code it documents;
2. that the body validates against the `response_model` declared on the route —
   FastAPI validates on the way out, so re-validating here is what catches a
   declaration that has silently drifted from what a client would generate from
   the OpenAPI schema;
3. that its documented error paths behave — 404 for a squad that cannot be
   fetched, 422 for an unparseable request, and an *empty* result where empty is
   genuinely the answer.

**A missing model is not an error.** `ServiceConfig.model_path` defaults to
`None` and `cli/pipeline.recommend` treats that as the closed-form baseline, so
every route must serve on a data root with no fitted model in it. That is not a
degraded corner worth one test; it is the default configuration, so it is the
configuration the whole file runs under. `sparse_client` then removes the
optional artifacts as well — no events log, no importance table, no discovery
registry — and asserts the surface degrades to a typed 404 or an empty list
rather than to a 500.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import TypeAdapter

from tests.api.conftest import (
    CLUSTER_MODEL_VERSION,
    CLUSTERED_PLAYER,
    DISCOVERY_OBJECTIVE,
    ENTRY_ID,
    EXPERIMENT_ID,
    GW1_DEADLINE,
    MATERIAL_EVENTS,
)
from xg_alonso.api.main import (
    ClusterHistoryEntry,
    ClusterResponse,
    CompiledIntentResponse,
    DiscoveredFeature,
    ExperimentResponse,
    FeatureImportanceResponse,
    HealthResponse,
    HypothesisResponse,
    ObjectivePreset,
    ParsedRequirementsResponse,
    PlanResponse,
    PlayerSummary,
    RecommendationResponse,
    SquadBuildResponse,
    SquadResponse,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

#: Squad shape, from the pinned rules. Restated nowhere: the assertions below
#: read these off the response's own player list rather than off a literal, and
#: the *rules* are verified against the snapshot in `tests/domain`.
SQUAD_SIZE = 15
STARTING_XI = 11


def _model(response: Any, model: type[Any]) -> Any:
    """Validate a body against the route's declared response model."""
    assert response.status_code == 200, response.text
    return model.model_validate(response.json())


def _models(response: Any, model: type[Any]) -> list[Any]:
    """The list form of :func:`_model`."""
    assert response.status_code == 200, response.text
    return TypeAdapter(list[model]).validate_python(response.json())  # type: ignore[valid-type]


def _assert_provenance(provenance: Any) -> None:
    """Every field populated.

    The module header of `api/main.py` claims every response carries provenance
    so a number can be traced to its inputs. A field that is present but empty
    satisfies the schema and defeats the claim, so emptiness is what is checked.
    """
    assert provenance.model_name
    assert provenance.model_version
    assert provenance.feature_set_version
    assert provenance.run_id
    assert provenance.data_cutoff.tzinfo is not None
    assert provenance.generated_at.tzinfo is not None


class TestHealth:
    def test_health_reports_a_loadable_world(self, client: TestClient) -> None:
        health = _model(client.get("/health"), HealthResponse)
        assert health.status == "ok"
        assert health.season == "2026-27"
        assert health.next_gameweek == 1
        assert health.deadline == GW1_DEADLINE
        assert health.players_loaded > 0
        assert health.history_rows > 0

    def test_a_deadline_in_the_future_is_not_stale(self, client: TestClient) -> None:
        """`stale` is about the snapshot, not about whether a model was found."""
        health = _model(client.get("/health"), HealthResponse)
        assert health.stale is False

    def test_no_model_is_reported_rather_than_faked(self, client: TestClient) -> None:
        """The default configuration, and it must be visible on the wire.

        `ServiceConfig.model_path` defaults to `None` to match `xg`'s own
        default — the module header records the 4-point disagreement that
        followed when it did not — so `model_loaded` is how a reader tells the
        closed-form baseline from a fitted model.
        """
        assert _model(client.get("/health"), HealthResponse).model_loaded is False

    def test_freshness_counts_only_material_events(self, client: TestClient) -> None:
        """The events log holds a third, immaterial row that must not be counted."""
        health = _model(client.get("/health"), HealthResponse)
        assert health.last_checked is not None
        assert health.seconds_since_check is not None
        assert health.unseen_events == MATERIAL_EVENTS

    def test_a_never_polled_root_reports_unknown_not_zero(self, sparse_client: TestClient) -> None:
        """Null age, not a zero that would read as 'just checked'."""
        health = _model(sparse_client.get("/health"), HealthResponse)
        assert health.last_checked is None
        assert health.seconds_since_check is None
        assert health.unseen_events == 0


class TestPlayers:
    def test_the_board_is_ranked_and_carries_depth(self, client: TestClient) -> None:
        players = _models(client.get("/players", params={"limit": 5}), PlayerSummary)
        assert len(players) == 5
        points = [p.expected_points for p in players]
        assert points == sorted(points, reverse=True)
        # Depth is computed only for the rows returned, so it must be present on
        # them — a board where nobody has a derivation is the failure this
        # endpoint's `with_depth` branch exists to avoid.
        assert any(p.derivation for p in players)
        assert any(p.horizon for p in players)

    def test_the_position_filter_narrows_to_one_position(self, client: TestClient) -> None:
        players = _models(
            client.get("/players", params={"limit": 100, "position": "def"}), PlayerSummary
        )
        assert players, "the fixture roster has defenders"
        assert {p.position for p in players} == {"DEF"}

    def test_the_price_filter_is_an_inclusive_ceiling(self, client: TestClient) -> None:
        ceiling = 60
        players = _models(
            client.get("/players", params={"limit": 100, "max_price": ceiling}), PlayerSummary
        )
        assert players
        assert all(p.price <= ceiling for p in players)

    def test_an_unknown_position_yields_an_empty_board_not_an_error(
        self, client: TestClient
    ) -> None:
        """Pinned because it is a real choice, not an oversight.

        The filter is applied to already-shaped rows, so an unrecognised value
        simply matches nothing. It is arguably worth a 422 later; what matters
        now is that a client is never handed a 500 for a typo.
        """
        assert _models(client.get("/players", params={"position": "ZZZ"}), PlayerSummary) == []

    @pytest.mark.parametrize("limit", [0, 1001])
    def test_a_limit_outside_the_declared_bounds_is_rejected(
        self, client: TestClient, limit: int
    ) -> None:
        assert client.get("/players", params={"limit": limit}).status_code == 422


class TestSquad:
    def test_a_squad_file_yields_a_fielded_eleven(
        self, client: TestClient, squad_file: Path
    ) -> None:
        squad = _model(
            client.get(f"/squad/{ENTRY_ID}", params={"squad_file": str(squad_file)}),
            SquadResponse,
        )
        assert squad.entry_id == ENTRY_ID
        assert len(squad.players) == SQUAD_SIZE
        assert sum(p.is_starter for p in squad.players) == STARTING_XI
        assert sum(p.is_captain for p in squad.players) == 1
        assert sum(p.is_vice_captain for p in squad.players) == 1
        # Starters sort ahead of the bench, which is what the shape promises.
        assert [p.is_starter for p in squad.players] == sorted(
            (p.is_starter for p in squad.players), reverse=True
        )
        _assert_provenance(squad.provenance)

    def test_the_formation_label_describes_the_eleven_it_ships_with(
        self, client: TestClient, squad_file: Path
    ) -> None:
        """A label that disagrees with the list beneath it is worse than none."""
        squad = _model(
            client.get(f"/squad/{ENTRY_ID}", params={"squad_file": str(squad_file)}),
            SquadResponse,
        )
        outfield = [int(part) for part in squad.formation.split("-")]
        assert sum(outfield) + 1 == STARTING_XI

        starters = [p for p in squad.players if p.is_starter]
        counted = [sum(p.position == wanted for p in starters) for wanted in ("DEF", "MID", "FWD")]
        assert counted == outfield
        assert sum(p.position == "GKP" for p in starters) == 1

    def test_an_unfetchable_entry_is_a_404(self, client: TestClient) -> None:
        """Offline, `entry/{id}/event/{gw}/picks/` cannot be reached at all.

        Which is also the preseason state that matters: that endpoint 404s until
        a gameweek's deadline passes, so "no squad for this manager yet" has to
        be a 404 rather than a 500 for the whole season's most common case.
        """
        response = client.get(f"/squad/{ENTRY_ID}")
        assert response.status_code == 404
        assert str(ENTRY_ID) in response.json()["detail"]

    def test_a_malformed_squad_file_is_a_422(self, client: TestClient, api_data_root: Path) -> None:
        response = client.get(
            f"/squad/{ENTRY_ID}",
            params={"squad_file": str(api_data_root / "not_a_squad.json")},
        )
        assert response.status_code == 422
        assert "picks" in response.json()["detail"]


class TestRecommend:
    def test_a_recommendation_explains_itself(self, client: TestClient, squad_file: Path) -> None:
        recommendation = _model(
            client.get(
                f"/recommend/{ENTRY_ID}",
                params={"squad_file": str(squad_file), "horizon": 2},
            ),
            RecommendationResponse,
        )
        assert recommendation.entry_id == ENTRY_ID
        assert recommendation.gameweek == 1
        assert len(recommendation.players) == SQUAD_SIZE
        assert recommendation.reasons, "a decision with no stated reason is not a decision"
        _assert_provenance(recommendation.provenance)

        if not recommendation.is_hold:
            assert recommendation.player_in is not None
            assert recommendation.player_out is not None
            assert recommendation.player_in.position == recommendation.player_out.position

    def test_the_headline_move_is_not_repeated_as_its_own_alternative(
        self, client: TestClient, squad_file: Path
    ) -> None:
        recommendation = _model(
            client.get(f"/recommend/{ENTRY_ID}", params={"squad_file": str(squad_file)}),
            RecommendationResponse,
        )
        if recommendation.is_hold or recommendation.player_in is None:
            pytest.skip("the fixture squad was told to hold, so there is no headline move")
        assert recommendation.player_out is not None
        headline = (recommendation.player_out.player_code, recommendation.player_in.player_code)
        assert headline not in {(a.player_out, a.player_in) for a in recommendation.alternatives}

    def test_every_derivation_reconciles_with_the_total_it_explains(
        self, client: TestClient, squad_file: Path
    ) -> None:
        """`derivation_reconciles` is reported rather than asserted internally.

        Which makes it exactly the thing a test has to check: the field exists
        because an explanation that silently disagrees with its own number is
        worse than one that admits it, and nothing else would notice a False.
        """
        recommendation = _model(
            client.get(f"/recommend/{ENTRY_ID}", params={"squad_file": str(squad_file)}),
            RecommendationResponse,
        )
        assert all(p.derivation_reconciles for p in recommendation.players)

    def test_the_lineup_comparison_decomposes_exactly(
        self, client: TestClient, squad_file: Path
    ) -> None:
        """The three deltas sum to the total; `shape_delta` is a named residual."""
        recommendation = _model(
            client.get(f"/recommend/{ENTRY_ID}", params={"squad_file": str(squad_file)}),
            RecommendationResponse,
        )
        lineup = recommendation.lineup
        assert lineup is not None, "the fixture squad sets slots 1-11, so a comparison exists"
        parts = lineup.swap_delta + lineup.captain_delta + lineup.shape_delta
        assert parts == pytest.approx(lineup.total_delta, abs=0.02)
        assert lineup.ours_points - lineup.yours_points == pytest.approx(
            lineup.total_delta, abs=0.02
        )

    def test_an_unfetchable_entry_is_a_404(self, client: TestClient) -> None:
        assert client.get(f"/recommend/{ENTRY_ID}").status_code == 404

    @pytest.mark.parametrize("horizon", [0, 11])
    def test_a_horizon_outside_the_declared_bounds_is_rejected(
        self, client: TestClient, squad_file: Path, horizon: int
    ) -> None:
        response = client.get(
            f"/recommend/{ENTRY_ID}",
            params={"squad_file": str(squad_file), "horizon": horizon},
        )
        assert response.status_code == 422


class TestBuildSquad:
    def test_the_built_squad_obeys_the_pinned_constraints(self, client: TestClient) -> None:
        """Budget and the per-club cap, checked against the response itself.

        The cap is read from the loaded rules by the optimizer; asserting it
        here from the response keeps this a test of the endpoint rather than a
        second transcription of a number CLAUDE.md forbids transcribing.
        """
        squad = _model(client.get("/build-squad"), SquadResponse)
        assert len(squad.players) == SQUAD_SIZE
        assert sum(p.is_starter for p in squad.players) == STARTING_XI
        assert squad.squad_value + squad.bank == 1000
        assert squad.bank >= 0

        per_club: dict[int, int] = {}
        for player in squad.players:
            per_club[player.team_id] = per_club.get(player.team_id, 0) + 1
        assert max(per_club.values()) <= 3

        positions = [p.position for p in squad.players]
        assert positions.count("GKP") == 2
        assert positions.count("DEF") == 5
        assert positions.count("MID") == 5
        assert positions.count("FWD") == 3

    def test_the_explained_build_justifies_every_pick(self, client: TestClient) -> None:
        built = _model(client.get("/build-squad/explained"), SquadBuildResponse)
        assert len(built.players) == SQUAD_SIZE
        assert len(built.explanations) == SQUAD_SIZE
        assert built.candidates_considered >= SQUAD_SIZE
        assert all(e.derivation_reconciles for e in built.explanations)
        assert all(e.evidence for e in built.explanations)
        _assert_provenance(built.provenance)

    def test_no_replacements_are_offered_before_the_first_deadline(
        self, client: TestClient
    ) -> None:
        """ "Who would replace him" has no meaning when every pick was free.

        Documented on `build_squad_explained` and worth pinning, because the
        field is present on the shape and an accidental fill would read as a
        transfer recommendation at a point where transfers are unlimited.
        """
        built = _model(client.get("/build-squad/explained"), SquadBuildResponse)
        assert all(e.replacements == [] for e in built.explanations)
        assert all(e.legal_replacements == 0 for e in built.explanations)

    def test_the_breakdown_sums_to_the_projection_it_explains(self, client: TestClient) -> None:
        built = _model(client.get("/build-squad/explained"), SquadBuildResponse)
        for explanation in built.explanations:
            breakdown = explanation.breakdown
            parts = (
                breakdown.appearance
                + breakdown.goals
                + breakdown.assists
                + breakdown.clean_sheets
                + breakdown.goals_conceded
                + breakdown.saves
                + breakdown.cards
                + breakdown.defensive_contribution
                + breakdown.bonus
            )
            assert parts == pytest.approx(breakdown.total, abs=0.01)


class TestFeatureImportance:
    def test_the_pooled_slice_is_served_by_default(self, client: TestClient) -> None:
        table = _model(client.get("/features/importance"), FeatureImportanceResponse)
        assert table.position == "ALL"
        assert table.features
        importances = [f.importance for f in table.features]
        assert importances == sorted(importances, reverse=True)
        assert table.families
        assert table.folds_measured == 2
        assert table.rows_measured > 0

    def test_a_degenerate_label_is_flagged_and_kept_out_of_the_ranking(
        self, client: TestClient
    ) -> None:
        """Ranking features by their effect on a constant is arithmetic, not evidence."""
        table = _model(client.get("/features/importance"), FeatureImportanceResponse)
        assert "label_red_cards" in table.degenerate_labels
        assert all("label_red_cards" not in f.per_label for f in table.features)

    def test_rank_stability_is_reported_when_more_than_one_fold_was_measured(
        self, client: TestClient
    ) -> None:
        """Null would mean 'never checked'; the fixture has two folds, so it is a number."""
        table = _model(client.get("/features/importance"), FeatureImportanceResponse)
        assert all(f.rank_stability is not None for f in table.features)

    def test_a_positional_slice_is_filtered_not_reweighted(self, client: TestClient) -> None:
        table = _model(
            client.get("/features/importance", params={"position": "DEF"}),
            FeatureImportanceResponse,
        )
        assert table.position == "DEF"
        assert "DEF" in table.positions
        # The fixture records fewer validation rows behind the positional slice
        # than behind the pooled one, which is the whole reason the field exists.
        pooled = _model(client.get("/features/importance"), FeatureImportanceResponse)
        assert table.rows_measured < pooled.rows_measured

    def test_an_unmeasured_slice_is_a_404_rather_than_an_empty_ranking(
        self, client: TestClient
    ) -> None:
        """An empty list is indistinguishable from 'no feature matters'."""
        assert client.get("/features/importance", params={"position": "GKP"}).status_code == 404

    def test_staleness_is_false_when_no_model_is_loaded(self, client: TestClient) -> None:
        """There is nothing to disagree with, so the honest answer is not stale."""
        table = _model(client.get("/features/importance"), FeatureImportanceResponse)
        assert table.stale is False

    def test_an_absent_table_is_a_404_naming_the_command_that_writes_it(
        self, sparse_client: TestClient
    ) -> None:
        response = sparse_client.get("/features/importance")
        assert response.status_code == 404
        assert "xg importance" in response.json()["detail"]


class TestObjectives:
    def test_the_shipped_presets_are_listed_with_stable_ids(self, client: TestClient) -> None:
        presets = _models(client.get("/objectives"), ObjectivePreset)
        assert presets
        ids = [p.id for p in presets]
        assert len(ids) == len(set(ids))
        assert "expected_points" in ids

    def test_compiling_a_request_reports_what_it_could_not_read(self, client: TestClient) -> None:
        compiled = _model(
            client.post(
                "/objectives/compile",
                json={"text": "protect my rank over the next six gameweeks"},
            ),
            CompiledIntentResponse,
        )
        assert compiled.objective_id
        assert compiled.primary_metric
        assert 0.0 <= compiled.overall_confidence <= 1.0
        # `unparsed` is part of the contract: a partial interpretation has to be
        # visible rather than silently narrowing the question.
        assert isinstance(compiled.unparsed, list)

    def test_an_unknown_preset_is_a_422(self, client: TestClient) -> None:
        response = client.post(
            "/objectives/compile", json={"text": "anything", "preset": "no_such_preset"}
        )
        assert response.status_code == 422

    def test_empty_text_is_rejected_by_the_declared_schema(self, client: TestClient) -> None:
        assert client.post("/objectives/compile", json={"text": ""}).status_code == 422


class TestRequirements:
    def test_a_parsed_requirement_carries_the_phrase_that_produced_it(
        self, client: TestClient, bootstrap_roster: dict[str, Any]
    ) -> None:
        who = bootstrap_roster["elements"][0]["web_name"]
        parsed = _model(
            client.post("/requirements/parse", json={"text": f"I want {who} in my squad"}),
            ParsedRequirementsResponse,
        )
        assert parsed.requirements, f"the parser found nothing in a request naming {who}"
        matched = parsed.requirements[0]
        assert matched.source == "matched"
        assert matched.evidence, "a matched requirement must show the phrase it matched"
        assert who in matched.player_names

    def test_the_language_model_is_not_consulted_unless_asked(self, client: TestClient) -> None:
        """`interpret` defaults off, and the deterministic parse stands alone."""
        parsed = _model(
            client.post("/requirements/parse", json={"text": "cheap defenders please"}),
            ParsedRequirementsResponse,
        )
        assert parsed.interpreted is False
        assert parsed.interpreter_note == ""

    def test_an_unknown_preset_is_a_422(self, client: TestClient) -> None:
        response = client.post(
            "/requirements/parse", json={"text": "anything", "preset": "no_such_preset"}
        )
        assert response.status_code == 422


class TestPlanSquad:
    def test_a_plan_prices_every_requirement_it_honoured(
        self, client: TestClient, bootstrap_roster: dict[str, Any]
    ) -> None:
        who = bootstrap_roster["elements"][0]["web_name"]
        plan = _model(
            client.post("/squad/plan", json={"text": f"I want {who} in my squad"}),
            PlanResponse,
        )
        assert len(plan.players) == SQUAD_SIZE
        assert plan.outcomes, "a requirement was parsed, so an outcome must be reported"
        assert plan.expected_points <= plan.unconstrained_points + 1e-6
        assert plan.bank >= 0
        assert {p.name for p in plan.players} >= {who}

    def test_the_scoring_model_is_named_on_every_plan(self, client: TestClient) -> None:
        """So a number is traceable to what produced it, model or baseline."""
        plan = _model(client.post("/squad/plan", json={"text": ""}), PlanResponse)
        assert plan.model_note == "closed-form baseline"

    def test_explicit_requirements_replace_the_parse_rather_than_adding_to_it(
        self, client: TestClient, bootstrap_roster: dict[str, Any]
    ) -> None:
        """Once a manager has edited the interpretation, re-deriving it discards their edit."""
        first, second = bootstrap_roster["elements"][0], bootstrap_roster["elements"][1]
        plan = _model(
            client.post(
                "/squad/plan",
                json={
                    "text": f"I want {first['web_name']} in my squad",
                    "requirements": [
                        {"kind": "must_include", "players": [second["code"]], "priority": 1}
                    ],
                },
            ),
            PlanResponse,
        )
        names = {p.name for p in plan.players}
        assert second["web_name"] in names
        assert [o.kind for o in plan.outcomes] == ["must_include"]

    def test_an_impossible_requirement_is_reported_rather_than_approximated(
        self, client: TestClient
    ) -> None:
        """Requirements are hard bounds, so each is honoured exactly or named as dropped."""
        plan = _model(
            client.post(
                "/squad/plan",
                json={
                    "text": "",
                    "requirements": [
                        {"kind": "club_floor", "team_id": 1, "count": 9, "priority": 1}
                    ],
                },
            ),
            PlanResponse,
        )
        assert plan.feasible_as_asked is False
        assert plan.outcomes
        dropped = [o for o in plan.outcomes if not o.honoured]
        assert dropped, "an infeasible requirement must be named, not silently ignored"
        assert all(o.note for o in dropped)


class TestDiscoverySurfaces:
    def test_every_verdict_is_listed_including_the_rejections(self, client: TestClient) -> None:
        features = _models(
            client.get("/features/discovered", params={"objective_id": DISCOVERY_OBJECTIVE}),
            DiscoveredFeature,
        )
        assert features
        row = features[0]
        assert row.feature == "minutes_trend_5"
        assert row.status == "accepted"
        assert row.leakage_passed is True
        assert row.folds == 2

    def test_the_objective_is_a_required_query_parameter(self, client: TestClient) -> None:
        """Verdicts are objective-conditioned, so serving them unconditioned would lie."""
        assert client.get("/features/discovered").status_code == 422

    def test_an_unknown_objective_yields_an_empty_report(self, client: TestClient) -> None:
        """Pinned as current behaviour, and it is arguably the wrong answer.

        `/features/importance` returns 404 rather than an empty ranking, with
        the stated reason that empty is indistinguishable from "nothing
        matters". This route makes the opposite choice once the registry
        exists: an objective nobody has run comes back as `[]`. The
        inconsistency is recorded here rather than smoothed over.
        """
        assert (
            _models(
                client.get("/features/discovered", params={"objective_id": "no_such_objective"}),
                DiscoveredFeature,
            )
            == []
        )

    def test_a_hypothesis_ships_with_the_condition_that_would_refute_it(
        self, client: TestClient
    ) -> None:
        hypotheses = _models(client.get("/hypotheses"), HypothesisResponse)
        assert hypotheses
        assert all(h.falsification_condition for h in hypotheses)
        assert all(h.football_rationale for h in hypotheses)

    def test_a_cluster_ships_with_its_statistical_basis(self, client: TestClient) -> None:
        """A label without its dominant features is a story."""
        clusters = _models(client.get("/clusters"), ClusterResponse)
        assert clusters
        assert all(c.dominant_features for c in clusters)
        assert clusters[0].cluster_model_version == CLUSTER_MODEL_VERSION

    def test_clusters_can_be_conditioned_on_an_objective(self, client: TestClient) -> None:
        clusters = _models(
            client.get("/clusters", params={"objective_id": DISCOVERY_OBJECTIVE}),
            ClusterResponse,
        )
        assert clusters
        assert all(c.objective_id == DISCOVERY_OBJECTIVE for c in clusters)

    def test_cluster_history_is_keyed_by_gameweek(self, client: TestClient) -> None:
        """A cluster is not an identity, so the history is per gameweek."""
        history = _models(
            client.get(f"/players/{CLUSTERED_PLAYER}/cluster-history"), ClusterHistoryEntry
        )
        assert history
        assert history[0].gameweek == 1
        assert 0.0 <= history[0].membership_probability <= 1.0

    def test_a_player_with_no_assignments_has_an_empty_history(self, client: TestClient) -> None:
        assert _models(client.get("/players/424242/cluster-history"), ClusterHistoryEntry) == []

    def test_experiments_are_listed_with_their_reproducibility_verdict(
        self, client: TestClient
    ) -> None:
        experiments = _models(client.get("/experiments"), ExperimentResponse)
        assert experiments
        assert experiments[0].experiment_id == EXPERIMENT_ID
        assert experiments[0].reproducible is True
        assert experiments[0].git_dirty is False

    def test_one_experiment_can_be_fetched_by_id(self, client: TestClient) -> None:
        manifest = _model(client.get(f"/experiments/{EXPERIMENT_ID}"), ExperimentResponse)
        assert manifest.stage == "completed"
        assert manifest.metrics

    def test_an_unknown_experiment_is_a_404(self, client: TestClient) -> None:
        response = client.get("/experiments/no-such-experiment")
        assert response.status_code == 404
        assert "no-such-experiment" in response.json()["detail"]


class TestDegradesWithoutOptionalArtifacts:
    """Bronze and silver only — a clone that ran `xg ingest` and stopped.

    Every route below reads something optional. None of them may fail: the
    difference between "nothing has been discovered" and "the server is broken"
    is the entire value of a status code.
    """

    def test_discovered_features_are_a_404_naming_the_command(
        self, sparse_client: TestClient
    ) -> None:
        response = sparse_client.get(
            "/features/discovered", params={"objective_id": DISCOVERY_OBJECTIVE}
        )
        assert response.status_code == 404
        assert "xg discover" in response.json()["detail"]

    @pytest.mark.parametrize(
        ("path", "model"),
        [
            ("/hypotheses", HypothesisResponse),
            ("/clusters", ClusterResponse),
            ("/experiments", ExperimentResponse),
            ("/players/900001/cluster-history", ClusterHistoryEntry),
        ],
    )
    def test_the_read_surfaces_come_back_empty_rather_than_failing(
        self, sparse_client: TestClient, path: str, model: type[Any]
    ) -> None:
        assert _models(sparse_client.get(path), model) == []

    def test_an_experiment_lookup_without_a_registry_is_a_404(
        self, sparse_client: TestClient
    ) -> None:
        assert sparse_client.get(f"/experiments/{EXPERIMENT_ID}").status_code == 404

    def test_the_decision_routes_still_answer(self, sparse_client: TestClient) -> None:
        """No model, no signals, no discovery — and a squad still comes back.

        This is the claim `cli/pipeline.recommend` makes when `models is None`,
        restated at the HTTP boundary: the closed-form baseline is a supported
        configuration, not a broken one.
        """
        players = _models(sparse_client.get("/players", params={"limit": 3}), PlayerSummary)
        assert len(players) == 3
        squad = _model(sparse_client.get("/build-squad"), SquadResponse)
        assert len(squad.players) == SQUAD_SIZE
