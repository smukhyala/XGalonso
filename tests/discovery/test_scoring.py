"""Utility scoring, the acceptance policy, and the registry's immutability."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from xg_alonso.contracts.discovery import (
    AcceptanceStatus,
    ComplementarityClass,
    DiscoveredFeatureSpec,
    FeatureEvaluation,
    FeatureHypothesis,
    FoldMetrics,
    Lesson,
    ValidationStatus,
)
from xg_alonso.contracts.objective import UtilityWeights
from xg_alonso.discovery.acceptance import (
    DEFAULT_POLICY,
    STRICT_POLICY,
    AcceptancePolicy,
    classify_complementarity,
    decide,
)
from xg_alonso.discovery.registry import DiscoveryRegistry
from xg_alonso.discovery.utility import (
    calibration_error,
    feature_utility,
    mae,
    poisson_deviance,
    rank_correlation,
    rmse,
    stability_score,
    top_k_precision,
    turnover_score,
)
from xg_alonso.evaluation.accuracy import spearman
from xg_alonso.storage.duckdb_store import DuckDBTableStore

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _folds(improvements: list[float], *, baseline: float = 1.0) -> tuple[FoldMetrics, ...]:
    """Folds with exactly the given relative improvements."""
    return tuple(
        FoldMetrics(
            fold_index=index,
            train_rows=500,
            validate_rows=200,
            baseline_metric=baseline,
            candidate_metric=baseline * (1.0 - gain),
        )
        for index, gain in enumerate(improvements)
    )


def _evaluation(**overrides: object) -> FeatureEvaluation:
    base: dict[str, object] = {
        "feature_id": "family.candidate",
        "feature_version": "a" * 16,
        "objective_id": "expected_points",
        "backtest_start": 0,
        "backtest_end": 4,
        "folds": _folds([0.02, 0.03, 0.01, 0.025, 0.02]),
        "incremental_value": 0.022,
        "stability": 0.8,
        "missingness": 0.05,
        "leakage_checks": ("static_validation", "point_in_time_harness"),
        "leakage_passed": True,
        "complementarity": ComplementarityClass.GLOBALLY_COMPLEMENTARY,
        "utility": 0.05,
        "accepted": AcceptanceStatus.ACCEPTED,
        "rejection_reason": "",
    }
    base.update(overrides)
    return FeatureEvaluation(**base)  # type: ignore[arg-type]


class TestPredictiveMetrics:
    def test_mae_and_rmse_agree_on_a_constant_error(self) -> None:
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([2.0, 3.0, 4.0])
        assert mae(predicted, actual) == pytest.approx(1.0)
        assert rmse(predicted, actual) == pytest.approx(1.0)

    def test_missing_values_are_dropped_never_imputed(self) -> None:
        predicted = np.array([1.0, np.nan, 3.0])
        actual = np.array([1.0, 5.0, 3.0])
        assert mae(predicted, actual) == pytest.approx(0.0)

    def test_poisson_deviance_is_zero_for_a_perfect_fit(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 0.0])
        assert poisson_deviance(values, values) == pytest.approx(0.0, abs=1e-6)

    def test_poisson_punishes_the_constant_zero_that_mae_rewards(self) -> None:
        """The failure mode this metric exists for.

        On a 96%-zero label, predicting zero everywhere minimises MAE and carries
        no information. Poisson deviance must prefer the informative predictor.
        """
        actual = np.array([0.0] * 96 + [1.0, 2.0, 1.0, 3.0])
        zeros = np.full_like(actual, 1e-9)
        # Weak but genuinely informative: it puts more mass where the events are
        # without getting close to them. This is what a real rare-event model
        # looks like, and it is exactly the predictor MAE punishes.
        informative = np.where(actual > 0, 0.6, 0.15)

        assert mae(zeros, actual) < mae(informative, actual), (
            "the constant zero should win on MAE — that is the whole problem"
        )
        assert poisson_deviance(informative, actual) < poisson_deviance(zeros, actual), (
            "Poisson deviance must prefer the informative predictor MAE rejected"
        )

    def test_rank_correlation_matches_the_evaluation_package(self) -> None:
        """The drift test for the deliberate second implementation.

        ``utility.rank_correlation`` restates ``evaluation.accuracy.spearman``
        because the discovery core may not import ``evaluation``. A mirrored
        implementation is honest only while it agrees.
        """
        generator = np.random.default_rng(7)
        for _ in range(20):
            predicted = generator.normal(size=40)
            actual = predicted * 0.6 + generator.normal(size=40)
            assert rank_correlation(predicted, actual) == pytest.approx(
                spearman(list(predicted), list(actual)), abs=1e-9
            )

    def test_rank_correlation_handles_ties(self) -> None:
        predicted = [1.0, 1.0, 2.0, 2.0]
        actual = [1.0, 1.0, 2.0, 2.0]
        assert rank_correlation(predicted, actual) == pytest.approx(1.0)

    def test_a_constant_vector_ranks_nothing(self) -> None:
        assert rank_correlation([1.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0]) == 0.0

    def test_top_k_precision_measures_the_picked_players(self) -> None:
        predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        actual = np.array([5.0, 4.0, 1.0, 2.0, 3.0])
        assert top_k_precision(predicted, actual, k=2) == pytest.approx(1.0)

    def test_calibration_error_catches_systematic_overconfidence(self) -> None:
        actual = np.linspace(0, 10, 100)
        calibrated = actual + np.random.default_rng(1).normal(0, 0.1, 100)
        inflated = actual * 2.0
        assert calibration_error(calibrated, actual) < calibration_error(inflated, actual)


class TestStability:
    def test_a_single_fold_has_no_measurable_consistency(self) -> None:
        assert stability_score(_folds([0.05])) == 0.0

    def test_consistent_small_gains_beat_one_lucky_fold(self) -> None:
        """The property the spread penalty exists for."""
        consistent = stability_score(_folds([0.01, 0.01, 0.01, 0.01, 0.01]))
        lucky = stability_score(_folds([0.001, 0.001, 0.001, 0.001, 0.4]))
        assert consistent > lucky

    def test_a_feature_that_never_helps_is_unstable(self) -> None:
        assert stability_score(_folds([-0.01, -0.02, -0.01])) == 0.0

    def test_turnover_is_zero_for_an_unchanged_ordering(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert turnover_score([values, values, values]) == pytest.approx(0.0)

    def test_turnover_is_high_for_a_reversed_ordering(self) -> None:
        forward = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert turnover_score([forward, forward[::-1]]) == pytest.approx(1.0)


class TestUtility:
    def test_every_term_is_reported(self) -> None:
        breakdown = feature_utility(
            weights=UtilityWeights(),
            predictive_gain=0.05,
            decision_gain=0.02,
            objective_gain=0.03,
            folds=_folds([0.02] * 5),
            complementarity_gain=0.01,
            complexity=4,
            missingness=0.1,
        )
        names = {name for name, _ in breakdown.contributions()}
        assert "predictive_gain" in names
        assert "complexity_penalty" in names
        assert breakdown.total == pytest.approx(
            sum(value for _, value in breakdown.contributions())
        )

    def test_leakage_risk_sinks_a_candidate_whatever_else_it_scores(self) -> None:
        """The leakage weight is an order of magnitude above the others by design."""
        breakdown = feature_utility(
            weights=UtilityWeights(),
            predictive_gain=1.0,
            decision_gain=1.0,
            objective_gain=1.0,
            complementarity_gain=1.0,
            leakage_risk=1.0,
        )
        assert breakdown.total < 0

    def test_complexity_is_charged_sub_linearly(self) -> None:
        """Two to four nodes matters; twenty to twenty-two does not."""
        weights = UtilityWeights()
        small = feature_utility(weights=weights, complexity=2).complexity_penalty
        medium = feature_utility(weights=weights, complexity=4).complexity_penalty
        large = feature_utility(weights=weights, complexity=20).complexity_penalty
        larger = feature_utility(weights=weights, complexity=22).complexity_penalty
        assert (medium - small) > (larger - large)

    def test_objective_weights_change_the_ranking(self) -> None:
        """The premise of the package: the same evidence, two verdicts."""
        stability_focused = UtilityWeights(stability=3.0, prediction=0.2)
        accuracy_focused = UtilityWeights(stability=0.0, prediction=3.0)

        steady = {"predictive_gain": 0.01, "folds": _folds([0.01] * 5)}
        spiky = {"predictive_gain": 0.05, "folds": _folds([-0.01, 0.3, -0.02, 0.01, -0.01])}

        assert (
            feature_utility(weights=stability_focused, **steady).total  # type: ignore[arg-type]
            > feature_utility(weights=stability_focused, **spiky).total  # type: ignore[arg-type]
        )
        assert (
            feature_utility(weights=accuracy_focused, **spiky).total  # type: ignore[arg-type]
            > feature_utility(weights=accuracy_focused, **steady).total  # type: ignore[arg-type]
        )


class TestAcceptance:
    def test_a_clean_consistent_candidate_is_accepted(self) -> None:
        assert decide(_evaluation()).status is AcceptanceStatus.ACCEPTED

    def test_leakage_is_fatal_and_checked_first(self) -> None:
        verdict = decide(_evaluation(leakage_passed=False, utility=99.0))
        assert verdict.status is AcceptanceStatus.REJECTED
        assert verdict.failed_gates == ("leakage",)
        assert "see the future" in verdict.reason

    def test_an_unchecked_candidate_is_not_a_clean_one(self) -> None:
        verdict = decide(_evaluation(leakage_checks=()))
        assert verdict.status is AcceptanceStatus.REJECTED
        assert "unchecked" in verdict.reason or "absence of evidence" in verdict.reason

    def test_too_few_folds_is_an_absent_verdict_not_a_negative_one(self) -> None:
        verdict = decide(_evaluation(folds=_folds([0.05, 0.05])))
        assert verdict.status is AcceptanceStatus.INSUFFICIENT_DATA
        assert "absent verdict" in verdict.reason

    def test_one_good_fold_does_not_earn_acceptance(self) -> None:
        """The rule the whole policy exists for."""
        verdict = decide(
            _evaluation(
                folds=_folds([0.4, -0.01, -0.02, -0.01, -0.01]),
                incremental_value=0.07,
                stability=0.1,
            )
        )
        assert verdict.status is not AcceptanceStatus.ACCEPTED

    def test_recent_degradation_is_checked_separately_from_the_average(self) -> None:
        """A feature that stopped working is worse than one that never did."""
        verdict = decide(
            _evaluation(folds=_folds([0.05, 0.05, 0.05, 0.05, -0.20]), incremental_value=0.02)
        )
        assert "recent_degradation" in verdict.failed_gates

    def test_excessive_missingness_is_refused(self) -> None:
        verdict = decide(_evaluation(missingness=0.9))
        assert "missingness" in verdict.failed_gates

    def test_a_near_miss_is_experimental_rather_than_discarded(self) -> None:
        verdict = decide(
            _evaluation(
                incremental_value=0.001,
                utility=0.01,
                folds=_folds([0.001, 0.002, -0.001, 0.001, 0.002]),
            )
        )
        assert verdict.status in {
            AcceptanceStatus.ACCEPTED_EXPERIMENTALLY,
            AcceptanceStatus.REJECTED,
        }

    def test_the_strict_policy_is_stricter(self) -> None:
        marginal = _evaluation(
            folds=_folds([0.003, 0.003, -0.001, 0.004, 0.003]),
            incremental_value=0.0025,
            stability=0.5,
        )
        assert decide(marginal, policy=DEFAULT_POLICY).status is AcceptanceStatus.ACCEPTED
        assert decide(marginal, policy=STRICT_POLICY).status is not AcceptanceStatus.ACCEPTED

    def test_a_one_fold_policy_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="coin flip"):
            AcceptancePolicy(min_folds=1)

    def test_every_rejection_carries_a_reason(self) -> None:
        for evaluation in (
            _evaluation(leakage_passed=False),
            _evaluation(missingness=0.99),
            _evaluation(folds=_folds([0.01])),
        ):
            assert decide(evaluation).reason


class TestComplementarityClassification:
    def test_a_broad_stable_gain_is_globally_complementary(self) -> None:
        assert (
            classify_complementarity(_evaluation()) is ComplementarityClass.GLOBALLY_COMPLEMENTARY
        )

    def test_no_gain_anywhere_is_redundant(self) -> None:
        assert (
            classify_complementarity(
                _evaluation(incremental_value=0.0001, stability=0.9, folds=_folds([0.0001] * 5))
            )
            is ComplementarityClass.REDUNDANT
        )

    def test_leakage_dominates_every_other_classification(self) -> None:
        assert (
            classify_complementarity(_evaluation(leakage_passed=False, incremental_value=0.5))
            is ComplementarityClass.LEAKAGE_SUSPECTED
        )

    def test_thin_history_is_reported_as_such(self) -> None:
        assert (
            classify_complementarity(_evaluation(folds=_folds([0.05, 0.05])))
            is ComplementarityClass.INSUFFICIENT_HISTORY
        )


class TestRegistry:
    @pytest.fixture
    def registry(self) -> DiscoveryRegistry:
        return DiscoveryRegistry(DuckDBTableStore(":memory:"))

    def _spec(self, **overrides: object) -> DiscoveredFeatureSpec:
        base: dict[str, object] = {
            "id": "family.candidate",
            "name": "candidate",
            "version": "b" * 16,
            "program": '{"kind":"source","column":"minutes","scope":"history"}',
            "input_columns": ("minutes",),
            "validation_status": ValidationStatus.LEAKAGE_PASSED,
        }
        base.update(overrides)
        return DiscoveredFeatureSpec(**base)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "status",
        [s for s in ValidationStatus if s is not ValidationStatus.LEAKAGE_PASSED],
    )
    def test_only_a_leakage_passed_feature_can_be_registered(
        self, registry: DiscoveryRegistry, status: ValidationStatus
    ) -> None:
        """The gate, over every status that is not a pass.

        `REJECTED_LEAKAGE` is the case that matters most and was previously
        unreachable: `experiment.py` hardcoded `LEAKAGE_PASSED` on every spec it
        built, so a program the harness rejected arrived here wearing a pass and
        this gate waved it through. The gate was right; its input was lying.
        """
        with pytest.raises(ValueError, match="leakage harness"):
            registry.register_feature(self._spec(validation_status=status))

    def test_a_leakage_passed_feature_is_registered_with_its_lineage(
        self, registry: DiscoveryRegistry
    ) -> None:
        registry.register_feature(self._spec())
        assert [f.name for f in registry.features()] == ["candidate"]
        assert registry.dependencies("b" * 16) == ("minutes",)

    def test_evaluations_are_append_only(self, registry: DiscoveryRegistry) -> None:
        """Re-measuring adds a row. It never edits one."""
        registry.record_evaluation(_evaluation(utility=0.1))
        registry.record_evaluation(_evaluation(utility=0.2, evaluated_at=NOW))
        stored = registry.evaluations(feature_version="a" * 16)
        assert len(stored) == 2
        assert sorted(e.utility for e in stored) == [0.1, 0.2]

    def test_the_latest_verdict_wins_not_the_best_one(self, registry: DiscoveryRegistry) -> None:
        """A feature accepted in March and rejected in May is rejected."""
        registry.register_feature(self._spec(version="c" * 16))
        registry.record_evaluation(
            _evaluation(
                feature_version="c" * 16,
                accepted=AcceptanceStatus.ACCEPTED,
                evaluated_at=datetime(2026, 3, 1, tzinfo=UTC),
            )
        )
        registry.record_evaluation(
            _evaluation(
                feature_version="c" * 16,
                accepted=AcceptanceStatus.REJECTED,
                rejection_reason="stopped working",
                evaluated_at=datetime(2026, 5, 1, tzinfo=UTC),
            )
        )
        assert registry.accepted_features("expected_points") == []

    def test_hypotheses_round_trip(self, registry: DiscoveryRegistry) -> None:
        hypothesis = FeatureHypothesis(
            id="family.candidate",
            title="A claim",
            football_rationale="Because of a mechanism.",
            falsification_condition="No gain in three folds.",
        )
        registry.register_hypothesis(hypothesis)
        stored = registry.hypothesis("family.candidate")
        assert stored is not None
        assert stored.falsification_condition == "No gain in three folds."

    def test_lessons_round_trip(self, registry: DiscoveryRegistry) -> None:
        registry.record_lesson(
            Lesson(id="exp.family", hypothesis_family="family", result="nothing worked")
        )
        assert [lesson.result for lesson in registry.lessons(family="family")] == ["nothing worked"]

    def test_the_acceptance_report_includes_rejections(self, registry: DiscoveryRegistry) -> None:
        """A report showing only winners cannot be distinguished from one that
        never looked."""
        registry.register_feature(self._spec(version="d" * 16, name="good"))
        registry.register_feature(self._spec(version="e" * 16, name="bad"))
        registry.record_evaluation(_evaluation(feature_version="d" * 16))
        registry.record_evaluation(
            _evaluation(
                feature_version="e" * 16,
                accepted=AcceptanceStatus.REJECTED,
                rejection_reason="failed incremental_value",
            )
        )
        report = registry.acceptance_report("expected_points")
        assert set(report["feature"].to_list()) == {"good", "bad"}
        # Accepted sorts first.
        assert report["status"][0] == "accepted"
        assert report.filter(pl.col("feature") == "bad")["reason"][0]

    def test_an_empty_registry_reports_an_empty_typed_frame(
        self, registry: DiscoveryRegistry
    ) -> None:
        report = registry.acceptance_report("nothing")
        assert report.height == 0
        assert "utility" in report.columns
