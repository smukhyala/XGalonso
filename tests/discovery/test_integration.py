"""The properties the whole system exists to have.

Each test here asserts something that would be true of *no* single-objective
system: that switching the objective changes the answer, that a constraint is
never traded away, and that a belief and the evidence stay separable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
import pytest

from xg_alonso.contracts.discovery import ExperimentStage
from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.objective import (
    BeliefEntity,
    BeliefProposition,
    ManagerConstraints,
    ObjectiveBundle,
    UserBelief,
)
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.discovery.clusters import fit_clusters, objective_weights, summarise
from xg_alonso.discovery.embeddings import fit_embedding
from xg_alonso.discovery.experiment import ExperimentConfig, run_discovery
from xg_alonso.discovery.harness import HarnessConfig, make_scorer
from xg_alonso.discovery.hypotheses import SEEDED_HYPOTHESES, generate_from_residuals
from xg_alonso.discovery.registry import DiscoveryRegistry
from xg_alonso.discovery.search import greedy_forward
from xg_alonso.domain.intent import compile_intent
from xg_alonso.domain.objectives import OBJECTIVE_PRESETS, objective_preset
from xg_alonso.domain.scoring import ScoringRules
from xg_alonso.features.generators import stage_window
from xg_alonso.optimization.objective import (
    ObjectiveContext,
    constraint_filter,
    objective_value,
    opportunity_cost,
    validate_constraints,
)
from xg_alonso.prediction.beliefs import BELIEF_CLAMP, apply_beliefs, belief_sensitivity
from xg_alonso.storage.duckdb_store import DuckDBTableStore

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


# --- helpers -----------------------------------------------------------------


def _prediction(
    code: int, *, points: float, sd: float, position: Position = Position.FWD, p60: float = 0.9
) -> PlayerPrediction:
    minutes = MinutesPrediction(
        p_appearance=0.95, p_start=0.9, expected_minutes=80.0, p_60_plus=p60, minutes_sd=10.0
    )
    components = ComponentExpectations(
        minutes=minutes,
        goals=points / 8.0,
        assists=0.1,
        clean_sheet_probability=0.2,
        goals_conceded=1.0,
        saves=0.0,
        yellow_cards=0.1,
        red_cards=0.0,
        own_goals=0.0,
        penalties_saved=0.0,
        penalties_missed=0.0,
        defensive_contribution_probability=0.0,
        bonus=0.3,
    )
    from xg_alonso.contracts.prediction import PointsBreakdown

    breakdown = PointsBreakdown(
        appearance=2.0,
        goals=points - 2.0,
        assists=0.0,
        clean_sheets=0.0,
        goals_conceded=0.0,
        saves=0.0,
        cards=0.0,
        own_goals=0.0,
        penalties=0.0,
        defensive_contribution=0.0,
        bonus=0.0,
    )
    return PlayerPrediction(
        player_code=PlayerCode(code),
        position=position,
        from_gameweek=GameweekId(5),
        horizon_gameweeks=1,
        components=components,
        breakdown=breakdown,
        expected_points=breakdown.total,
        expected_points_sd=sd,
        scoring_rules_version="2026-27",
        provenance=PredictionProvenance(
            model_name="test",
            model_version="1",
            model_artifact_sha256="a" * 64,
            feature_set_name="test",
            feature_set_version="1",
            data_cutoff=NOW,
            predicted_at=NOW,
            run_id="run",
            code_version="deadbeef",
        ),
    )


def _squad() -> SquadState:
    layout = [
        (Position.GKP, 2),
        (Position.DEF, 5),
        (Position.MID, 5),
        (Position.FWD, 3),
    ]
    picks: list[SquadPick] = []
    slot = 1
    code = 100
    for position, count in layout:
        for _ in range(count):
            picks.append(
                SquadPick(
                    player_code=PlayerCode(code),
                    position=position,
                    team_id=TeamId(1 + code % 12),
                    purchase_price=TenthsOfMillion(50),
                    current_price=TenthsOfMillion(50),
                    selling_price=TenthsOfMillion(50),
                    squad_slot=slot,
                )
            )
            slot += 1
            code += 1
    # Reorder so the starting XI is a legal 1-4-4-2.
    order = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 1, 6, 11, 14]
    picks = [
        picks[original].model_copy(update={"squad_slot": index + 1})
        for index, original in enumerate(order)
    ]
    return SquadState(
        entry_id=1,  # type: ignore[arg-type]
        gameweek=GameweekId(5),
        picks=tuple(picks),
        bank=TenthsOfMillion(10),
        free_transfers=1,
    )


@pytest.fixture(scope="module")
def scoring_rules() -> ScoringRules:
    import json
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"
    )
    return ScoringRules.from_bootstrap(
        json.loads(fixture.read_text()),
        version="2026-27",
        source_sha256="a" * 64,
        fetched_at=NOW,
    )


# --- the properties ----------------------------------------------------------


class TestObjectiveChangesTheAnswer:
    """If switching the objective changed nothing, the layer would be decoration."""

    def test_aggressive_and_conservative_rank_volatility_oppositely(self) -> None:
        steady = _prediction(1, points=6.0, sd=1.0)
        volatile = _prediction(2, points=6.0, sd=4.0)

        chase = objective_preset("mini_league_chase")
        protect = objective_preset("rank_protection")

        assert objective_value(volatile, objective=chase) > objective_value(steady, objective=chase)
        assert objective_value(steady, objective=protect) > objective_value(
            volatile, objective=protect
        )

    def test_the_variance_penalty_is_genuinely_signed(self) -> None:
        """Not merely smaller for an aggressive objective — negative."""
        assert objective_preset("mini_league_chase").signed_uncertainty_penalty < 0
        assert objective_preset("rank_protection").signed_uncertainty_penalty > 0

    def test_ownership_flips_sign_with_preference(self) -> None:
        owned = _prediction(1, points=6.0, sd=2.0)
        rare = _prediction(2, points=6.0, sd=2.0)
        context = ObjectiveContext(ownership={PlayerCode(1): 0.7, PlayerCode(2): 0.02})

        chase = objective_preset("mini_league_chase")
        protect = objective_preset("rank_protection")

        assert objective_value(rare, objective=chase, context=context) > objective_value(
            owned, objective=chase, context=context
        )
        assert objective_value(owned, objective=protect, context=context) > objective_value(
            rare, objective=protect, context=context
        )

    def test_objectives_emphasise_different_embedding_axes(self) -> None:
        columns = ("total_points_max_5", "minutes_mean_5", "selected_mean_5", "starts_mean_5")
        chase = objective_weights(objective_preset("mini_league_chase"), columns)
        protect = objective_weights(objective_preset("rank_protection"), columns)
        # Chase leans on the ceiling; protection leans on minutes security.
        assert chase[0] > protect[0]
        assert protect[1] > chase[1]

    def test_objective_conditioned_clusters_differ_from_the_control(self) -> None:
        generator = np.random.default_rng(3)
        rows = 300
        frame = pl.DataFrame(
            {
                "player_code": list(range(rows)),
                "position": ["MID"] * rows,
                "minutes_mean_5": generator.normal(60, 20, rows),
                "minutes_mean_20": generator.normal(60, 20, rows),
                "starts_mean_5": generator.uniform(0, 1, rows),
                "total_points_mean_5": generator.normal(4, 2, rows),
                "total_points_max_5": generator.normal(9, 5, rows),
                "total_points_std_10": generator.uniform(0, 6, rows),
                "selected_mean_5": generator.lognormal(6, 1.5, rows),
                "threat_per90_5": generator.normal(30, 15, rows),
                "expected_goals_per90_5": generator.normal(0.3, 0.2, rows),
                "appearance_rate_10": generator.uniform(0, 1, rows),
            }
        )
        embedding = fit_embedding(frame, n_components=4)
        plain, _, _ = fit_clusters(frame, embedding=embedding, k=4, seed=1).assign(frame)
        chased, _, _ = fit_clusters(
            frame, objective=objective_preset("mini_league_chase"), embedding=embedding, k=4, seed=1
        ).assign(frame)
        assert not np.array_equal(plain, chased)

    def test_a_cluster_model_version_names_its_objective(self) -> None:
        generator = np.random.default_rng(4)
        frame = pl.DataFrame(
            {
                "player_code": list(range(200)),
                "position": ["MID"] * 200,
                "minutes_mean_5": generator.normal(60, 20, 200),
                "total_points_mean_5": generator.normal(4, 2, 200),
                "selected_mean_5": generator.lognormal(6, 1.5, 200),
                "threat_per90_5": generator.normal(30, 15, 200),
            }
        )
        model = fit_clusters(frame, objective=objective_preset("rank_protection"), k=3)
        assert "rank_protection" in model.model_version()


class TestConstraintsAreNeverTradedAway:
    def test_a_locked_player_is_removed_from_the_market_not_penalised(self) -> None:
        squad = _squad()
        locked = squad.picks[0].player_code
        sellable, _, report = constraint_filter(
            squad,
            constraints=ManagerConstraints(locked_players=(locked,)),
            candidate_codes=[PlayerCode(900)],
            candidate_teams={PlayerCode(900): 19},
        )
        assert locked not in sellable
        assert report.locked_players == (locked,)
        assert report.is_binding

    def test_a_locked_position_freezes_every_player_in_it(self) -> None:
        squad = _squad()
        sellable, _, _ = constraint_filter(
            squad,
            constraints=ManagerConstraints(locked_positions=(Position.DEF,)),
            candidate_codes=[],
            candidate_teams={},
        )
        assert not any(p.position is Position.DEF for p in squad.picks if p.player_code in sellable)

    def test_excluded_players_never_enter_the_buy_list(self) -> None:
        squad = _squad()
        _, buyable, report = constraint_filter(
            squad,
            constraints=ManagerConstraints(excluded_players=(PlayerCode(900),)),
            candidate_codes=[PlayerCode(900), PlayerCode(901)],
            candidate_teams={PlayerCode(900): 19, PlayerCode(901): 18},
        )
        assert PlayerCode(900) not in buyable
        assert PlayerCode(901) in buyable
        assert report.removed_candidates == 1

    def test_an_unsatisfiable_constraint_set_says_so_up_front(self) -> None:
        squad = _squad()
        problems = validate_constraints(
            ManagerConstraints(locked_players=(PlayerCode(9999),)), squad
        )
        assert any("not in the squad" in p for p in problems)

    def test_locking_everything_is_reported_rather_than_returning_nothing(self) -> None:
        squad = _squad()
        problems = validate_constraints(
            ManagerConstraints(
                locked_positions=(Position.GKP, Position.DEF, Position.MID, Position.FWD)
            ),
            squad,
        )
        assert any("no transfer is possible" in p for p in problems)

    def test_opportunity_cost_of_a_lock_is_measured_and_surfaced(self) -> None:
        squad = _squad()
        locked = squad.picks[0]
        predictions = {
            locked.player_code: _prediction(
                int(locked.player_code), points=4.0, sd=1.0, position=locked.position
            ),
            PlayerCode(900): _prediction(900, points=9.0, sd=1.0, position=locked.position),
        }
        best, forgone = opportunity_cost(
            locked,
            predictions=predictions,
            candidates=[PlayerCode(900)],
            prices={PlayerCode(900): TenthsOfMillion(55)},
            budget=TenthsOfMillion(60),
            objective=objective_preset("expected_points"),
        )
        assert best == PlayerCode(900)
        assert forgone > 0

    def test_no_opportunity_cost_when_nothing_affordable_is_better(self) -> None:
        squad = _squad()
        locked = squad.picks[0]
        predictions = {
            locked.player_code: _prediction(
                int(locked.player_code), points=9.0, sd=1.0, position=locked.position
            ),
            PlayerCode(900): _prediction(900, points=4.0, sd=1.0, position=locked.position),
        }
        best, forgone = opportunity_cost(
            locked,
            predictions=predictions,
            candidates=[PlayerCode(900)],
            prices={PlayerCode(900): TenthsOfMillion(55)},
            budget=TenthsOfMillion(60),
            objective=objective_preset("expected_points"),
        )
        assert best is None
        assert forgone == 0.0


class TestBeliefsStaySeparableFromEvidence:
    def test_both_predictions_are_retained(self, scoring_rules: ScoringRules) -> None:
        prediction = _prediction(1, points=6.0, sd=2.0)
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=1,
            proposition=BeliefProposition.WILL_RETURN,
            confidence=0.9,
            affected_gameweeks=(GameweekId(5),),
        )
        [adjustment] = apply_beliefs(
            [prediction], [belief], gameweek=GameweekId(5), rules=scoring_rules
        )
        assert adjustment.raw is prediction
        assert adjustment.adjusted is not prediction
        assert adjustment.adjusted.expected_points > adjustment.raw.expected_points
        assert adjustment.moved

    def test_the_raw_prediction_is_never_mutated(self, scoring_rules: ScoringRules) -> None:
        prediction = _prediction(1, points=6.0, sd=2.0)
        before = prediction.expected_points
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=1,
            proposition=BeliefProposition.OUTPERFORM_MODEL,
            confidence=1.0,
        )
        apply_beliefs([prediction], [belief], gameweek=GameweekId(5), rules=scoring_rules)
        assert prediction.expected_points == before

    def test_the_adjustment_is_clamped(self, scoring_rules: ScoringRules) -> None:
        """A hunch cannot overrule the model however sure the manager is."""
        prediction = _prediction(1, points=6.0, sd=2.0)
        beliefs = [
            UserBelief(
                entity_type=BeliefEntity.PLAYER,
                entity_id=1,
                proposition=BeliefProposition.OUTPERFORM_MODEL,
                confidence=1.0,
            )
            for _ in range(5)
        ]
        [adjustment] = apply_beliefs(
            [prediction], beliefs, gameweek=GameweekId(5), rules=scoring_rules
        )
        assert adjustment.multiplier <= 1.0 + BELIEF_CLAMP + 1e-9

    def test_a_negative_belief_lowers_the_projection(self, scoring_rules: ScoringRules) -> None:
        prediction = _prediction(1, points=6.0, sd=2.0)
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=1,
            proposition=BeliefProposition.WILL_NOT_START,
            confidence=0.8,
        )
        [adjustment] = apply_beliefs(
            [prediction], [belief], gameweek=GameweekId(5), rules=scoring_rules
        )
        assert adjustment.adjusted.expected_points < adjustment.raw.expected_points

    def test_players_without_a_belief_are_still_returned_unchanged(
        self, scoring_rules: ScoringRules
    ) -> None:
        predictions = [_prediction(1, points=6.0, sd=2.0), _prediction(2, points=5.0, sd=1.0)]
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=1,
            proposition=BeliefProposition.WILL_RETURN,
            confidence=0.5,
        )
        adjustments = apply_beliefs(
            predictions, [belief], gameweek=GameweekId(5), rules=scoring_rules
        )
        assert len(adjustments) == 2
        assert adjustments[1].multiplier == 1.0
        assert not adjustments[1].moved

    def test_uncertainty_is_not_reduced_by_a_belief(self, scoring_rules: ScoringRules) -> None:
        """Acting on a hunch must not make the projection look more certain."""
        prediction = _prediction(1, points=6.0, sd=2.0)
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=1,
            proposition=BeliefProposition.WILL_RETURN,
            confidence=1.0,
        )
        [adjustment] = apply_beliefs(
            [prediction], [belief], gameweek=GameweekId(5), rules=scoring_rules
        )
        assert adjustment.adjusted.expected_points_sd == prediction.expected_points_sd

    def test_a_belief_decays_across_the_gameweeks_it_covers(self) -> None:
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=1,
            proposition=BeliefProposition.WILL_RETURN,
            confidence=0.8,
            affected_gameweeks=(GameweekId(5), GameweekId(6), GameweekId(7)),
            decay_rate=1.0,
        )
        assert belief.weight_at(GameweekId(5)) == pytest.approx(0.8)
        assert belief.weight_at(GameweekId(6)) == pytest.approx(0.4)
        assert belief.weight_at(GameweekId(7)) == pytest.approx(0.8 / 3)
        assert belief.weight_at(GameweekId(9)) == 0.0

    def test_sensitivity_shows_where_a_recommendation_would_flip(
        self, scoring_rules: ScoringRules
    ) -> None:
        prediction = _prediction(1, points=6.0, sd=2.0)
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=1,
            proposition=BeliefProposition.WILL_RETURN,
            confidence=0.5,
        )
        curve = belief_sensitivity(prediction, belief, gameweek=GameweekId(5), rules=scoring_rules)
        points = [value for _, value in curve]
        assert points == sorted(points)
        assert points[0] < points[-1]


class TestHypothesisLibrary:
    def test_every_seeded_hypothesis_states_a_falsification_condition(self) -> None:
        for seed in SEEDED_HYPOTHESES:
            assert seed.hypothesis.falsification_condition.strip()
            assert seed.hypothesis.football_rationale.strip()

    def test_every_seeded_program_is_structurally_valid(self) -> None:
        for seed in SEEDED_HYPOTHESES:
            assert seed.program.depth() >= 1
            assert seed.program.columns()

    def test_seeded_programs_have_distinct_semantics(self) -> None:
        versions = [seed.program.version() for seed in SEEDED_HYPOTHESES]
        assert len(versions) == len(set(versions))

    def test_generation_is_aimed_at_measured_weakness(self) -> None:
        proposals = generate_from_residuals(
            weak_segments=[("position", "FWD", 1.3)],
            base_metrics=("expected_goals", "threat"),
            limit=4,
        )
        assert proposals
        assert all("FWD" in p.hypothesis.football_rationale for p in proposals)
        assert all(p.hypothesis.falsification_condition for p in proposals)

    def test_already_rejected_families_are_not_re_proposed(self) -> None:
        first = generate_from_residuals(
            weak_segments=[("position", "FWD", 1.3)],
            base_metrics=("expected_goals",),
            limit=1,
        )
        family = first[0].hypothesis.family
        again = generate_from_residuals(
            weak_segments=[("position", "FWD", 1.3)],
            base_metrics=("expected_goals",),
            rejected_families=(family,),
            limit=1,
        )
        assert all(p.hypothesis.family != family for p in again)

    def test_duplicate_semantics_are_suppressed(self) -> None:
        """A known program is skipped — and the generator moves on rather than stopping.

        Suppression must not end generation. If it did, one already-tested idea
        would block every untried mechanism behind it, and the loop would go
        quiet exactly as it accumulated knowledge. The correct behaviour is to
        propose something *different*.
        """
        first = generate_from_residuals(
            weak_segments=[("position", "FWD", 1.3)],
            base_metrics=("expected_goals",),
            limit=1,
        )
        known = {p.program.version() for p in first}

        again = generate_from_residuals(
            weak_segments=[("position", "FWD", 1.3)],
            base_metrics=("expected_goals",),
            known_versions=list(known),
            limit=1,
        )
        assert all(p.program.version() not in known for p in again)
        assert again, "suppressing one duplicate must not silence the generator"

    def test_one_call_proposes_at_most_one_mechanism_per_segment_and_metric(self) -> None:
        """The diversity guard: a batch must not fill up with variations of one idea."""
        proposals = generate_from_residuals(
            weak_segments=[("position", "FWD", 1.3)],
            base_metrics=("expected_goals",),
            limit=64,
        )
        assert len(proposals) == 1

    def test_repeated_calls_explore_and_eventually_go_quiet(self) -> None:
        """The loop terminates rather than proposing forever.

        Because one call proposes at most one mechanism per (segment, metric),
        exploration happens *across* iterations — which is the right shape for a
        discovery loop that runs weekly. This asserts the accumulation actually
        converges: eventually there is nothing new to say.
        """
        seen_versions: set[str] = set()
        seen_names: set[str] = set()

        for _ in range(20):
            batch = generate_from_residuals(
                weak_segments=[("position", "FWD", 1.3)],
                base_metrics=("expected_goals",),
                known_versions=list(seen_versions),
                existing_names=list(seen_names),
                limit=64,
            )
            if not batch:
                break
            seen_versions.update(p.program.version() for p in batch)
            seen_names.update(p.program.name for p in batch)
        else:  # pragma: no cover - only reached if the generator never stops
            pytest.fail("the generator never ran out of ideas, so the loop would not converge")

        assert len(seen_versions) > 1, "it should have explored several mechanisms first"


class TestComplementarySearch:
    def test_the_search_finds_the_planted_signal_and_skips_the_noise(
        self, training_frame: pl.DataFrame
    ) -> None:
        """``label_total_points`` depends on ``signal`` and not on ``noise``."""
        scorer, _ = make_scorer(
            training_frame, config=HarnessConfig(min_train_gameweeks=10, max_folds=3)
        )
        result = greedy_forward(
            required=("anchor",),
            candidates=("signal", "noise"),
            scorer=scorer,
            max_features=2,
            min_gain=0.005,
        )
        assert "signal" in result.discovered
        assert "noise" not in result.discovered

    def test_a_required_feature_is_never_dropped(self, training_frame: pl.DataFrame) -> None:
        scorer, _ = make_scorer(
            training_frame, config=HarnessConfig(min_train_gameweeks=10, max_folds=3)
        )
        result = greedy_forward(
            required=("anchor", "noise"),
            candidates=("signal",),
            scorer=scorer,
            max_features=1,
        )
        assert "anchor" in result.selected
        assert "noise" in result.selected

    def test_rejected_candidates_are_reported_not_hidden(
        self, training_frame: pl.DataFrame
    ) -> None:
        scorer, _ = make_scorer(
            training_frame, config=HarnessConfig(min_train_gameweeks=10, max_folds=3)
        )
        result = greedy_forward(
            required=("anchor",), candidates=("signal", "noise"), scorer=scorer, max_features=1
        )
        assert any(name == "noise" for name, _ in result.rejected)


class TestEndToEnd:
    """Compile intent, discover, register, report — the whole path."""

    @pytest.mark.e2e
    def test_the_loop_runs_and_produces_a_reproducible_manifest(
        self, training_frame: pl.DataFrame
    ) -> None:
        history = pl.DataFrame(
            {
                "player_code": [1, 1, 2, 2],
                "minutes": [90, 80, 20, 30],
                "expected_goals": [0.4, 0.2, 0.05, 0.1],
                "total_points": [6, 3, 1, 2],
                "threat": [40.0, 20.0, 5.0, 8.0],
                "kickoff_time": [NOW] * 4,
                "available_time": [NOW] * 4,
            }
        )
        bundle = ObjectiveBundle(
            objective=objective_preset("expected_points"),
            constraints=ManagerConstraints(required_features=("anchor",)),
        )
        registry = DiscoveryRegistry(DuckDBTableStore(":memory:"))
        stages: list[ExperimentStage] = []

        result = run_discovery(
            bundle=bundle,
            training=training_frame,
            player_stats=history,
            stage_window=stage_window,
            registry=registry,
            config=ExperimentConfig(
                harness=HarnessConfig(min_train_gameweeks=10, max_folds=3),
                max_hypotheses=2,
                run_controls=False,
                run_search=False,
                fit_clusters_for_objective=False,
            ),
            on_stage=lambda stage, _: stages.append(stage),
        )

        assert result.manifest.stage is ExperimentStage.COMPLETED
        assert ExperimentStage.COMPLETED in stages
        assert result.manifest.fold_definitions
        assert result.manifest.seeds
        # Every seeded program reads real match columns the tiny history lacks,
        # so they are refused *statically* rather than computing nonsense.
        assert result.rejected_programs

    def test_a_missing_required_feature_stops_the_run(self, training_frame: pl.DataFrame) -> None:
        """Never silently drop a required feature."""
        bundle = ObjectiveBundle(
            objective=objective_preset("expected_points"),
            constraints=ManagerConstraints(required_features=("not_a_column",)),
        )
        with pytest.raises(ValueError, match="hard constraint"):
            run_discovery(
                bundle=bundle,
                training=training_frame,
                player_stats=training_frame,
                stage_window=stage_window,
                config=ExperimentConfig(fit_clusters_for_objective=False),
            )

    def test_compiled_intent_flows_into_a_bundle(self) -> None:
        intent = compile_intent(
            "I am 40 points behind in my mini-league. Keep Haaland. "
            "Aggressive three-gameweek strategy. Recent xG must remain in the model.",
            players={"Haaland": 223094},
            next_gameweek=5,
        )
        bundle = intent.bundle
        assert bundle.constraints.locked_players == (PlayerCode(223094),)
        assert "expected_goals_per90_5" in bundle.constraints.required_features
        assert bundle.objective.planning_horizon == 3
        assert bundle.objective.signed_uncertainty_penalty < 0
        assert intent.overall_confidence > 0.5


class TestPresets:
    def test_every_preset_is_distinct_and_addressable(self) -> None:
        ids = [preset.id for preset in OBJECTIVE_PRESETS]
        assert len(ids) == len(set(ids)) == 6
        for preset_id in ids:
            assert objective_preset(preset_id).id == preset_id

    def test_an_unknown_preset_names_the_available_ones(self) -> None:
        with pytest.raises(KeyError, match="available"):
            objective_preset("does_not_exist")

    def test_cluster_summaries_carry_their_statistical_basis(self) -> None:
        generator = np.random.default_rng(6)
        frame = pl.DataFrame(
            {
                "player_code": list(range(200)),
                "position": ["MID"] * 200,
                "minutes_mean_5": generator.normal(60, 25, 200),
                "total_points_mean_5": generator.normal(4, 3, 200),
                "selected_mean_5": generator.lognormal(6, 1.5, 200),
                "threat_per90_5": generator.normal(30, 20, 200),
            }
        )
        model = fit_clusters(frame, k=3, seed=2)
        summaries = summarise(model, frame)
        assert summaries
        # A label is decoration; the dominant features are the evidence.
        assert any(summary.dominant_features for summary in summaries)
        assert sum(summary.size for summary in summaries) == frame.height


class TestObjectiveSubstitution:
    """`objective_valued` is the single point the objective enters the search."""

    def test_the_breakdown_still_sums_to_the_total(self) -> None:
        """The prediction contract enforces this; a scaled total would be rejected."""
        from xg_alonso.optimization.objective import objective_valued

        predictions = {PlayerCode(1): _prediction(1, points=6.0, sd=2.0)}
        priced = objective_valued(predictions, objective=objective_preset("mini_league_chase"))
        result = priced[PlayerCode(1)]
        assert result.breakdown.total == pytest.approx(result.expected_points)

    def test_an_aggressive_objective_raises_a_volatile_player(self) -> None:
        from xg_alonso.optimization.objective import objective_valued

        volatile = {PlayerCode(1): _prediction(1, points=6.0, sd=4.0)}
        chase = objective_valued(volatile, objective=objective_preset("mini_league_chase"))
        protect = objective_valued(volatile, objective=objective_preset("rank_protection"))
        assert chase[PlayerCode(1)].expected_points > 6.0
        assert protect[PlayerCode(1)].expected_points < 6.0

    def test_uncertainty_is_not_rescaled(self) -> None:
        """The objective changes how variance is priced, not how much there is."""
        from xg_alonso.optimization.objective import objective_valued

        predictions = {PlayerCode(1): _prediction(1, points=6.0, sd=2.0)}
        priced = objective_valued(predictions, objective=objective_preset("rank_protection"))
        assert priced[PlayerCode(1)].expected_points_sd == 2.0


class TestLocksAreUnbreakable:
    """A lock is a filter, not a penalty a good enough alternative overcomes."""

    def _rules(self) -> Any:
        import json
        from pathlib import Path

        from xg_alonso.domain.rules import SquadRules

        fixture = (
            Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"
        )
        return SquadRules.from_bootstrap(
            json.loads(fixture.read_text()), version="2026-27", source_sha256="a" * 64
        )

    def test_a_held_player_is_never_ranked_however_good_the_alternative(self) -> None:
        from xg_alonso.optimization.transfer import Candidate, rank_single_transfers

        squad = _squad()
        rules = self._rules()
        held = squad.picks[0]

        predictions = {
            p.player_code: _prediction(int(p.player_code), points=1.0, sd=1.0, position=p.position)
            for p in squad.picks
        }
        # A vastly better replacement in the same position, affordable.
        replacement = _prediction(999, points=50.0, sd=1.0, position=held.position)
        predictions[PlayerCode(999)] = replacement
        candidates = [
            Candidate(
                player_code=PlayerCode(999),
                position=held.position,
                team_id=TeamId(19),
                price=TenthsOfMillion(40),
                prediction=replacement,
            )
        ]

        unconstrained = rank_single_transfers(
            squad, candidates=candidates, predictions=predictions, rules=rules
        )
        assert any(c.out_pick.player_code == held.player_code for c in unconstrained), (
            "without the lock, selling him should be on the table"
        )

        sellable = frozenset(
            p.player_code for p in squad.picks if p.player_code != held.player_code
        )
        constrained = rank_single_transfers(
            squad,
            candidates=candidates,
            predictions=predictions,
            rules=rules,
            sellable=sellable,
        )
        assert not any(c.out_pick.player_code == held.player_code for c in constrained)

    def test_a_held_player_still_appears_on_the_board_with_a_reason(self) -> None:
        """ "Why not him?" must have an answer for every squad member."""
        from xg_alonso.contracts.reason_codes import ReasonCode
        from xg_alonso.optimization.transfer import build_transfer_board

        squad = _squad()
        rules = self._rules()
        held = squad.picks[0]
        predictions = {
            p.player_code: _prediction(int(p.player_code), points=3.0, sd=1.0, position=p.position)
            for p in squad.picks
        }
        sellable = frozenset(
            p.player_code for p in squad.picks if p.player_code != held.player_code
        )
        board = build_transfer_board(
            squad, candidates=[], predictions=predictions, rules=rules, sellable=sellable
        )
        entry = next(e for e in board.by_player if e.player_out == held.player_code)
        assert entry.option is None
        assert any(r.code is ReasonCode.CONSTRAINT_HELD for r in entry.reasons)
        assert "your instruction" in entry.reasons[0].render()

    def test_every_squad_member_still_appears(self) -> None:
        from xg_alonso.optimization.transfer import build_transfer_board

        squad = _squad()
        rules = self._rules()
        predictions = {
            p.player_code: _prediction(int(p.player_code), points=3.0, sd=1.0, position=p.position)
            for p in squad.picks
        }
        board = build_transfer_board(
            squad,
            candidates=[],
            predictions=predictions,
            rules=rules,
            sellable=frozenset({squad.picks[0].player_code}),
        )
        assert len(board.by_player) == 15
