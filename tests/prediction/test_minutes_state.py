"""The three-class minutes-state head, and the incumbent it has to beat.

The head exists to replace an algebraic reconciliation, not a model. Today
``inference.py::_minutes_from`` derives ``p_appearance`` and ``p_60_plus`` from
an expected-minutes regression and a start probability, because those two are
fitted independently and can contradict each other. That reconciliation feeds
the appearance term of ``assemble_points`` — the largest single term for most
players — so the question worth asking is not "does the head beat a base rate"
but "does estimating the states directly beat reconciling them".

The comparison is therefore against the incumbent, on log loss and multiclass
Brier, out of fold. The head is deliberately *not* wired into inference here:
measure first.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from xg_alonso.contracts.prediction import MINUTES_STATES, MinutesState
from xg_alonso.domain.scoring import ScoringRules
from xg_alonso.features.schema import model_feature_names
from xg_alonso.prediction.inference import _minutes_from
from xg_alonso.prediction.trained import (
    StateFoldReport,
    incumbent_state_probabilities,
    train_component_models,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data/fixtures/fpl/bootstrap_static_2026_27.json"
SILVER = ROOT / ".data/silver/player_gameweek_stats.parquet"
TRAINING = ROOT / ".data/gold/training_frame.parquet"


@pytest.fixture(scope="module")
def rules() -> ScoringRules:
    payload: dict[str, Any] = json.loads(FIXTURE.read_text())
    return ScoringRules.from_bootstrap(
        payload, version="2026-27", source_sha256="a" * 64, fetched_at=datetime.now(UTC)
    )


class TestTheStateEnum:
    def test_the_ordering_is_fixed(self) -> None:
        """It is the column order of every ``predict_proba`` output and of every
        dispersion table keyed by state. A reordering would silently transpose
        probabilities onto the wrong states."""
        assert MINUTES_STATES == (MinutesState.NONE, MinutesState.SHORT, MinutesState.LONG)
        assert [s.class_index for s in MINUTES_STATES] == [0, 1, 2]

    def test_the_threshold_is_supplied_not_assumed(self, rules: ScoringRules) -> None:
        threshold = rules.thresholds.long_play_minutes
        assert MinutesState.of(0, long_play_minutes=threshold) is MinutesState.NONE
        assert MinutesState.of(1, long_play_minutes=threshold) is MinutesState.SHORT
        assert MinutesState.of(threshold - 1, long_play_minutes=threshold) is MinutesState.SHORT
        assert MinutesState.of(threshold, long_play_minutes=threshold) is MinutesState.LONG

    def test_a_different_threshold_moves_the_boundary(self) -> None:
        """Guards against the threshold being ignored and 60 hardcoded."""
        assert MinutesState.of(45, long_play_minutes=30) is MinutesState.LONG
        assert MinutesState.of(45, long_play_minutes=60) is MinutesState.SHORT


class TestTheIncumbentReplica:
    """``incumbent_state_probabilities`` duplicates ``_minutes_from``'s
    arithmetic because ``inference`` cannot be imported from ``trained``. The
    duplication is only safe while it is pinned, so it is pinned here."""

    @pytest.mark.parametrize(
        ("minutes", "start"),
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (30.0, 0.1),
            (45.0, 0.5),
            (60.0, 0.5),
            (69.9, 0.2),
            (70.0, 0.2),
            (89.0, 0.95),
            (90.0, 1.0),
            (120.0, 1.0),
            (-5.0, -0.5),
            (50.0, 1.5),
        ],
    )
    def test_it_matches_minutes_from_elementwise(self, minutes: float, start: float) -> None:
        expected = _minutes_from(minutes, start)
        got = incumbent_state_probabilities(
            np.array([minutes], dtype=np.float64), np.array([start], dtype=np.float64)
        )
        assert got[0, MinutesState.NONE.class_index] == pytest.approx(
            1.0 - expected.p_appearance, abs=1e-6
        )
        assert got[0, MinutesState.SHORT.class_index] == pytest.approx(
            expected.p_appearance - expected.p_60_plus, abs=1e-6
        )
        assert got[0, MinutesState.LONG.class_index] == pytest.approx(expected.p_60_plus, abs=1e-6)

    def test_every_row_is_a_distribution(self) -> None:
        rng = np.random.default_rng(20260727)
        minutes = rng.uniform(-10.0, 130.0, 500)
        start = rng.uniform(-0.2, 1.2, 500)
        got = incumbent_state_probabilities(minutes, start)
        assert np.all(got >= -1e-12), "a negative mass is not a probability"
        assert np.allclose(got.sum(axis=1), 1.0, atol=1e-9)


def _synthetic_training(n_gameweeks: int = 24, n_players: int = 60) -> pl.DataFrame:
    """A small frame with a learnable state, for testing the *plumbing*.

    Explicitly not the place to decide whether the head beats the incumbent.
    ``label_starts`` is derived from the same 60-minute threshold as the state,
    which hands the reconciliation a perfect signal for ``P(long)`` and produces
    a near-tie that says nothing about the real data. :class:`TestTheGate`
    settles that question where it can be settled.
    """
    rng = np.random.default_rng(20260727)
    rows: list[dict[str, Any]] = []
    for gameweek in range(1, n_gameweeks + 1):
        for player in range(n_players):
            # A stable per-player propensity, visible to the model as a feature.
            propensity = (player % 10) / 10.0
            draw = rng.random()
            if draw < 0.25 * (1.0 - propensity):
                minutes = 0
            elif draw < 0.25 * (1.0 - propensity) + 0.2:
                minutes = int(rng.integers(1, 59))
            else:
                minutes = int(rng.integers(60, 91))
            rows.append(
                {
                    "player_code": player,
                    "label_season": "2024-25",
                    "label_gameweek": gameweek,
                    "feature_propensity": propensity,
                    "feature_noise": float(rng.normal()),
                    "label_minutes": float(minutes),
                    "label_starts": float(minutes >= 60),
                }
            )
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.when(pl.col("label_minutes") >= 60)
        .then(MinutesState.LONG.class_index)
        .when(pl.col("label_minutes") > 0)
        .then(MinutesState.SHORT.class_index)
        .otherwise(MinutesState.NONE.class_index)
        .cast(pl.Int64)
        .alias("label_minutes_state")
    )


FEATURES = ("feature_propensity", "feature_noise")
LABELS = ("label_minutes", "label_starts", "label_minutes_state")
SMALL = {"max_iter": 30, "max_depth": 3}


@pytest.fixture(scope="module")
def fitted() -> Any:
    return train_component_models(
        _synthetic_training(),
        feature_columns=FEATURES,
        label_columns=LABELS,
        min_train_gameweeks=8,
        validate_gameweeks=4,
        model_kwargs=SMALL,
    )


class TestTheHead:
    def test_it_is_fitted_as_a_multiclass_model(self, fitted: Any) -> None:
        assert "label_minutes_state" in fitted.models
        assert list(fitted.models["label_minutes_state"].classes_) == [0, 1, 2]

    def test_it_stays_out_of_the_scalar_prediction_mapping(self, fitted: Any) -> None:
        """``predict`` promises one scalar per row per label. A three-column
        array placed in it would type-check and break every consumer."""
        frame = _synthetic_training(n_gameweeks=2, n_players=10)
        predicted = fitted.predict(frame)
        assert "label_minutes_state" not in predicted
        for value in predicted.values():
            assert value.ndim == 1

    def test_predict_minutes_state_returns_a_distribution(self, fitted: Any) -> None:
        frame = _synthetic_training(n_gameweeks=2, n_players=10)
        probabilities = fitted.predict_minutes_state(frame)
        assert probabilities is not None
        assert probabilities.shape == (frame.height, len(MINUTES_STATES))
        assert np.all(probabilities >= 0.0)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)

    def test_an_artifact_without_the_head_says_so(self) -> None:
        """``None``, not a uniform prior. A silently invented state distribution
        would be the single most load-bearing fabrication in the system."""
        without = train_component_models(
            _synthetic_training(),
            feature_columns=FEATURES,
            label_columns=("label_minutes", "label_starts"),
            min_train_gameweeks=8,
            validate_gameweeks=4,
            model_kwargs=SMALL,
        )
        assert without.predict_minutes_state(_synthetic_training(n_gameweeks=2)) is None
        assert without.state_reports == []

    def test_a_single_observed_state_is_refused(self) -> None:
        """One class cannot be a distribution over three, and forcing it would
        produce a confidently constant model."""
        frame = _synthetic_training(n_gameweeks=24, n_players=20).with_columns(
            pl.lit(MinutesState.LONG.class_index, dtype=pl.Int64).alias("label_minutes_state")
        )
        fitted = train_component_models(
            frame,
            feature_columns=FEATURES,
            label_columns=LABELS,
            min_train_gameweeks=8,
            validate_gameweeks=4,
            model_kwargs=SMALL,
        )
        assert "label_minutes_state" not in fitted.models


class TestTheComparisonAgainstTheIncumbent:
    def test_every_fold_is_scored_against_the_incumbent(self, fitted: Any) -> None:
        assert fitted.state_reports, "no fold produced a comparison"
        for report in fitted.state_reports:
            assert report.validate_rows > 0
            assert report.train_rows > 0
            assert report.baseline_log_loss > 0.0
            assert sum(report.state_frequencies) == pytest.approx(1.0, abs=1e-9)

    def test_the_comparison_can_fail(self) -> None:
        """The negative control. Scored against a forecaster that is *better*
        than the head, both gains must go negative — a comparison that reports
        an improvement whatever it is handed is not a comparison.

        The rival is near-perfect rather than perfect on purpose. A perfect
        forecaster has log loss of exactly zero, which the ``<= 0`` guard reads
        as "no baseline" and reports as a flat 0.0 — correct behaviour, but it
        would make this control vacuous.
        """
        from xg_alonso.prediction.trained import StateFoldReport, _brier, _log_loss

        truth = np.array([0, 1, 2, 2, 0], dtype=np.int64)
        confident = np.full((5, 3), 0.05)
        confident[np.arange(5), truth] = 0.90
        head = np.full((5, 3), 1.0 / 3.0)

        report = StateFoldReport(
            fold_index=0,
            train_rows=10,
            validate_rows=5,
            log_loss=_log_loss(head, truth),
            brier=_brier(head, truth),
            baseline_log_loss=_log_loss(confident, truth),
            baseline_brier=_brier(confident, truth),
        )
        assert report.log_loss_gain < 0.0
        assert report.brier_gain < 0.0

    def test_a_degenerate_baseline_reports_no_gain_rather_than_dividing_by_zero(
        self,
    ) -> None:
        report = StateFoldReport(
            fold_index=0,
            train_rows=10,
            validate_rows=5,
            log_loss=0.5,
            brier=0.3,
            baseline_log_loss=0.0,
            baseline_brier=0.0,
        )
        assert report.log_loss_gain == 0.0
        assert report.brier_gain == 0.0


@pytest.fixture(scope="module")
def measured(rules: ScoringRules) -> Any:
    """The head and the incumbent, fitted on the real frame with production
    hyperparameters. Module-scoped: the fit is the expensive part and all three
    gate assertions read the same fitted object."""
    threshold = rules.thresholds.long_play_minutes
    frame = pl.read_parquet(TRAINING).with_columns(
        pl.when(pl.col("label_minutes") >= threshold)
        .then(MinutesState.LONG.class_index)
        .when(pl.col("label_minutes") > 0)
        .then(MinutesState.SHORT.class_index)
        .otherwise(MinutesState.NONE.class_index)
        .cast(pl.Int64)
        .alias("label_minutes_state")
    )
    features = tuple(c for c in model_feature_names() if c in frame.columns)
    return train_component_models(
        frame,
        feature_columns=features,
        label_columns=("label_minutes", "label_starts", "label_minutes_state"),
        min_train_gameweeks=8,
        validate_gameweeks=4,
        embargo_gameweeks=1,
    )


@pytest.mark.dataset
@pytest.mark.skipif(not TRAINING.exists(), reason="requires the local .data gold training frame")
class TestTheGate:
    """The comparison that decides whether the head is worth wiring in.

    Run on the real training frame with production hyperparameters, because the
    synthetic fixture above cannot settle this. That fixture derives
    ``label_starts`` from the same threshold as the state itself, which hands
    the incumbent a perfect signal for ``P(long)`` — in the real data starts and
    the 60-minute state agree 97.8% of the time, close enough that a fixture
    built on the identity flatters the reconciliation into a tie.

    Where the incumbent is actually weak is the ``none``/``short`` split, which
    it derives as ``min(1, expected_minutes / 70)`` — a shape, not an estimate.
    """

    def test_it_beats_the_incumbent_on_log_loss(self, measured: Any) -> None:
        reports = measured.state_reports
        assert len(reports) >= 10, "too few folds to conclude anything"
        gain = sum(r.log_loss_gain for r in reports) / len(reports)
        assert gain > 0.20, f"log loss improved by only {gain:+.1%} over the incumbent"

    def test_it_beats_the_incumbent_on_brier(self, measured: Any) -> None:
        reports = measured.state_reports
        gain = sum(r.brier_gain for r in reports) / len(reports)
        assert gain > 0.05, f"Brier improved by only {gain:+.1%} over the incumbent"

    def test_it_wins_on_every_fold_not_just_on_average(self, measured: Any) -> None:
        """A mean gain can hide a policy that is much better on a few folds and
        worse on the rest. It is not: the head wins all fifteen."""
        losses = [r.fold_index for r in measured.state_reports if r.log_loss_gain <= 0.0]
        assert losses == [], f"the incumbent won folds {losses}"


@pytest.mark.dataset
@pytest.mark.skipif(not SILVER.exists(), reason="requires the local .data silver tables")
class TestTheRealBaseRates:
    def test_the_three_states_match_the_measured_shares(self, rules: ScoringRules) -> None:
        """Pins the population the composition engine is built around. If these
        move materially, the zero atom the distribution models has moved too."""
        frame = pl.read_parquet(SILVER, columns=["minutes"])
        threshold = rules.thresholds.long_play_minutes
        minutes = frame["minutes"].to_numpy()
        none = float(np.mean(minutes == 0))
        long_play = float(np.mean(minutes >= threshold))
        short = 1.0 - none - long_play

        assert none == pytest.approx(0.596, abs=0.02)
        assert short == pytest.approx(0.128, abs=0.02)
        assert long_play == pytest.approx(0.277, abs=0.02)
