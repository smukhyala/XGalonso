"""One seed, derived by name.

Eight occurrences of the literal ``20260727`` sat across four packages. The
scheme is replaced here; the numbers deliberately are not. Applying
``derive_seed`` to the estimator's ``random_state`` would change every fitted
artifact and move the headline before the evaluation framework can reproduce
it — naming a constant is a refactor, renumbering it is an experiment.
"""

from __future__ import annotations

import subprocess
import sys

from xg_alonso.contracts.seeds import ROOT_SEED, SeedLedger, derive_seed


class TestTheRootIsPinned:
    def test_the_value_has_not_moved(self) -> None:
        """Pinned explicitly, so a refactor cannot silently refit every model."""
        assert ROOT_SEED == 20260727

    def test_the_estimator_still_uses_it(self) -> None:
        from xg_alonso.prediction.trained import _REGRESSOR_KWARGS

        assert _REGRESSOR_KWARGS["random_state"] == ROOT_SEED


class TestDerivedSeeds:
    def test_the_same_path_gives_the_same_seed(self) -> None:
        a = derive_seed(ROOT_SEED, "policy", "random", "2024-25", 6)
        b = derive_seed(ROOT_SEED, "policy", "random", "2024-25", 6)
        assert a == b

    def test_seeds_are_stable_across_processes(self) -> None:
        """`hash()` is salted per process; a seed that moves is not a seed."""
        here = derive_seed(ROOT_SEED, "squad", "random_legal", 3)
        code = (
            "from xg_alonso.contracts.seeds import ROOT_SEED, derive_seed;"
            "print(derive_seed(ROOT_SEED, 'squad', 'random_legal', 3))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert int(out.stdout.strip()) == here

    def test_every_part_changes_the_seed(self) -> None:
        base = derive_seed(ROOT_SEED, "policy", "random", "2024-25", 6)
        assert derive_seed(ROOT_SEED, "policy", "random", "2024-25", 7) != base
        assert derive_seed(ROOT_SEED, "policy", "random", "2025-26", 6) != base
        assert derive_seed(ROOT_SEED, "policy", "model", "2024-25", 6) != base
        assert derive_seed(ROOT_SEED + 1, "policy", "random", "2024-25", 6) != base

    def test_order_is_part_of_the_identity(self) -> None:
        assert derive_seed(ROOT_SEED, "a", "b") != derive_seed(ROOT_SEED, "b", "a")

    def test_seeds_fit_the_thirty_two_bit_range(self) -> None:
        """numpy and scikit-learn both refuse anything wider."""
        for i in range(500):
            seed = derive_seed(ROOT_SEED, "x", i)
            assert 0 <= seed < 2**32

    def test_distinct_labels_do_not_collide(self) -> None:
        seeds = {derive_seed(ROOT_SEED, "label", i) for i in range(10_000)}
        assert len(seeds) == 10_000


class TestTheLedgerRecordsWhatWasDrawn:
    def test_recording_is_append_only(self) -> None:
        ledger = SeedLedger().record("estimator", 1).record("policy", 2)
        assert ledger.entries == (("estimator", 1), ("policy", 2))

    def test_a_label_can_be_looked_up(self) -> None:
        ledger = SeedLedger().record("estimator", 99)
        assert ledger.for_label("estimator") == 99
        assert ledger.for_label("absent") is None


class TestRulesResolution:
    def test_a_past_season_reports_that_its_rules_are_not_exact(self) -> None:
        """`.data/pinned` holds only the current season, and now says so."""
        from pathlib import Path

        from xg_alonso.cli.main import _resolve_rules
        from xg_alonso.contracts.identifiers import parse_season

        resolved = _resolve_rules(Path(".data"), parse_season("2024-25"))

        assert not resolved.exact
        assert resolved.source_season != "2024-25"
        assert "no snapshot exists" in resolved.caveat

    def test_the_current_season_resolves_exactly_and_carries_no_caveat(self) -> None:
        from pathlib import Path

        from xg_alonso.cli.main import _resolve_rules
        from xg_alonso.contracts.identifiers import parse_season

        resolved = _resolve_rules(Path(".data"), parse_season("2026-27"))

        assert resolved.exact
        assert resolved.caveat == ""


class TestFreezeProvenanceIsRecorded:
    def test_the_calibration_declares_where_it_was_measured(self) -> None:
        """A freeze check has to read this; a docstring is not readable."""
        from xg_alonso.prediction.calibration import CALIBRATION_MEASURED_ON

        assert CALIBRATION_MEASURED_ON == ("2025-26",)

    def test_a_fitted_model_records_its_hyperparameters(self) -> None:
        """Applied at fit time and never stored, so nothing could check them."""
        import polars as pl
        from conftest import FAST, synthetic_stats

        from xg_alonso.prediction.dataset import build_training_frame
        from xg_alonso.prediction.trained import train_component_models

        stats: pl.DataFrame = synthetic_stats(players=12, gameweeks=14)
        data = build_training_frame(stats, min_gameweek=2)
        models = train_component_models(
            data.frame,
            feature_columns=data.feature_columns,
            label_columns=data.label_columns[:1],
            min_train_gameweeks=3,
            validate_gameweeks=2,
            model_kwargs=FAST,
        )

        assert models.estimator_kwargs["max_iter"] == FAST["max_iter"]
        assert models.estimator_kwargs["random_state"] == ROOT_SEED
