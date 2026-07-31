"""One seed, derived by name.

Eight occurrences of the literal ``20260727`` sat across four packages. The
scheme is replaced here; the numbers deliberately are not. Applying
``derive_seed`` to the estimator's ``random_state`` would change every fitted
artifact and move the headline before the evaluation framework can reproduce
it — naming a constant is a refactor, renumbering it is an experiment.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer

from xg_alonso.cli.main import _resolve_rules
from xg_alonso.contracts.identifiers import parse_season
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.provenance import SourceTimestamps, TimeSource
from xg_alonso.contracts.seeds import ROOT_SEED, SeedLedger, derive_seed
from xg_alonso.pipelines.ingestion.bootstrap import SOURCE_BOOTSTRAP
from xg_alonso.storage.bronze import FileSystemBronzeStore


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


#: A value that differs between the pinned seasons the tests construct, so an
#: assertion can tell *which snapshot's rules were actually loaded* rather than
#: only which one was named. Six is the wrong answer for a goalkeeper goal —
#: the exact transcription error `domain/scoring.py` exists to prevent — which
#: makes it unmistakable in a failure message.
_WRONG_GKP_GOAL = 6


def _data_root_pinned_at(tmp_path: Path, *seasons: str) -> Path:
    """A data root holding pinned rules for exactly ``seasons``.

    Built rather than borrowed. These tests previously read the developer's
    own ``.data`` directory, which made them depend on unversioned local state
    and on the process's working directory — so they passed on a machine that
    had run ``xg ingest`` and failed everywhere else, including CI. Constructing
    the input is also the only way to test the *fallback ordering*, which needs
    more than one pinned season and the real directory has only ever had one.
    """
    fixture = (
        Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"
    )
    payload = json.loads(fixture.read_text())
    raw = json.dumps(payload).encode("utf-8")

    pinned = tmp_path / "pinned"
    pinned.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        # Every season but the newest gets a deliberately distinguishable
        # goalkeeper-goal value, so a test can prove which file was read.
        stored = json.loads(json.dumps(payload))
        if season != "2026-27":
            stored["game_config"]["scoring"]["goals_scored"]["GKP"] = _WRONG_GKP_GOAL
        body = json.dumps(stored).encode("utf-8")
        (pinned / f"rules_{season}.json").write_text(
            json.dumps(
                {
                    "fetched_at": "2026-07-27T00:00:00+00:00",
                    "payload": stored,
                    "source_sha256": hashlib.sha256(body).hexdigest(),
                }
            )
        )

    # `_resolve_rules` builds a full slice context, which reads the bootstrap
    # payload back out of bronze rather than off the network.
    moment = datetime(2026, 7, 27, tzinfo=UTC)
    FileSystemBronzeStore(tmp_path / "bronze").write(
        source=SOURCE_BOOTSTRAP,
        payload=raw,
        timestamps=SourceTimestamps(
            event_time=moment,
            observed_time=moment,
            available_time=moment,
            processed_time=moment,
            time_source=TimeSource.HTTP_DATE_MINUS_AGE,
        ),
        run_id="test-rules-resolution",
    )
    return tmp_path


class TestRulesResolution:
    """Three call sites reached for `DEFAULT_SEASON` and none of them said so,
    scoring a 2024-25 backtest under 2026-27 rules. The fix was not to invent a
    historical snapshot but to record which one was actually used."""

    def test_an_exact_match_carries_no_caveat(self, tmp_path: Path) -> None:
        resolved = _resolve_rules(
            _data_root_pinned_at(tmp_path, "2026-27"), parse_season("2026-27")
        )

        assert resolved.exact
        assert resolved.source_season == "2026-27"
        assert resolved.caveat == ""

    def test_a_season_with_no_snapshot_reports_that_it_is_not_exact(self, tmp_path: Path) -> None:
        resolved = _resolve_rules(
            _data_root_pinned_at(tmp_path, "2026-27"), parse_season("2024-25")
        )

        assert not resolved.exact
        assert resolved.source_season == "2026-27"
        assert "no snapshot exists" in resolved.caveat

    def test_it_prefers_the_newest_season_not_later_than_the_one_asked_for(
        self, tmp_path: Path
    ) -> None:
        """The ordering that matters, and that a single-season directory could
        never exercise: given 2022-23 and 2026-27, a 2024-25 evaluation must
        fall *back* to 2022-23 rather than forward to rules that did not exist
        yet."""
        root = _data_root_pinned_at(tmp_path, "2022-23", "2026-27")

        resolved = _resolve_rules(root, parse_season("2024-25"))

        assert not resolved.exact
        assert resolved.source_season == "2022-23"
        assert resolved.scoring.goals_scored[Position.GKP] == _WRONG_GKP_GOAL

    def test_it_loads_the_named_snapshots_values_not_just_its_name(self, tmp_path: Path) -> None:
        """The bug this pair of assertions exists for.

        `_resolve_rules` used to call `_load_context`, which parses whatever
        bootstrap payload is newest in bronze. So a 2022-23 resolution returned
        `exact=True`, `source_season="2022-23"` and a `ScoringRules` whose
        `version` field also said "2022-23" — while its *values* came from the
        2026-27 snapshot. Every label agreed and the numbers were another
        season's.

        Asserting `source_season` alone cannot catch that, because
        `source_season` was the one thing that was right.
        """
        root = _data_root_pinned_at(tmp_path, "2022-23", "2026-27")

        old = _resolve_rules(root, parse_season("2022-23"))
        new = _resolve_rules(root, parse_season("2026-27"))

        assert old.exact
        assert new.exact
        assert old.scoring.goals_scored[Position.GKP] == _WRONG_GKP_GOAL
        assert new.scoring.goals_scored[Position.GKP] == 10
        assert old.scoring.goals_scored != new.scoring.goals_scored

    def test_it_falls_forward_only_when_nothing_earlier_exists(self, tmp_path: Path) -> None:
        root = _data_root_pinned_at(tmp_path, "2026-27")

        resolved = _resolve_rules(root, parse_season("2022-23"))

        assert not resolved.exact
        assert resolved.source_season == "2026-27"

    def test_nothing_pinned_says_so_instead_of_naming_a_file(self, tmp_path: Path) -> None:
        """The same mislabelling, one branch further in.

        With nothing pinned the rules can only come from the live bronze
        snapshot. Reporting `source_season=DEFAULT_SEASON` and a caveat reading
        "resolved from the 2026-27 pinned snapshot" would name a file that does
        not exist — which is the defect this function was just fixed for.
        """
        root = _data_root_pinned_at(tmp_path)  # bronze only, nothing pinned
        assert list((root / "pinned").glob("rules_*.json")) == []

        resolved = _resolve_rules(root, parse_season("2024-25"))

        assert not resolved.exact
        assert not resolved.pinned
        assert resolved.source_season == "2024-25"
        assert "live bronze snapshot" in resolved.caveat
        assert "pinned snapshot" not in resolved.caveat

    def test_a_pinned_fallback_still_names_its_snapshot(self, tmp_path: Path) -> None:
        resolved = _resolve_rules(
            _data_root_pinned_at(tmp_path, "2026-27"), parse_season("2024-25")
        )

        assert resolved.pinned
        assert "2026-27 pinned snapshot" in resolved.caveat

    def test_require_exact_refuses_the_fallback(self, tmp_path: Path) -> None:
        """Otherwise the caveat is the only thing standing between a reader and
        a number scored under the wrong season's rules."""
        root = _data_root_pinned_at(tmp_path, "2026-27")

        with pytest.raises(typer.BadParameter, match="no pinned rules for 2024-25"):
            _resolve_rules(root, parse_season("2024-25"), require_exact=True)


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
