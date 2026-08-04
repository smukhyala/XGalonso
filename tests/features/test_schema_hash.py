"""The catalogue and rules hashes, and what each has to be sensitive to.

A hash over feature *names* would miss the change that matters most: altering
``prior_strength`` from 3.0 to 4.0 changes every shrunk rate's values while
changing no name at all. And a rules hash taken over the bootstrap payload
would move every day, because that payload carries prices.

Both cases are pinned below, along with the two things each hash must ignore.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import pytest

from xg_alonso.contracts.prediction import Position
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringRules, rules_snapshot_hash
from xg_alonso.features.catalogue import catalogue_specs
from xg_alonso.features.schema import catalogue_hash, model_feature_names


@pytest.fixture(scope="module")
def payload(bootstrap_payload: dict[str, Any]) -> dict[str, Any]:
    return bootstrap_payload


@pytest.fixture(scope="module")
def scoring(payload: dict[str, Any]) -> ScoringRules:
    return ScoringRules.from_bootstrap(
        payload, version="2026-27", source_sha256="a" * 64, fetched_at=datetime.now(UTC)
    )


@pytest.fixture(scope="module")
def squad(payload: dict[str, Any]) -> SquadRules:
    return SquadRules.from_bootstrap(payload, version="2026-27", source_sha256="a" * 64)


class TestTheCatalogueHash:
    def test_it_is_stable_across_processes(self) -> None:
        """A digest that drifts between interpreters is worse than none."""
        here = catalogue_hash()
        code = "from xg_alonso.features.schema import catalogue_hash;print(catalogue_hash())"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == here

    def test_adding_a_feature_changes_it(self) -> None:
        specs = catalogue_specs()
        assert catalogue_hash(specs[:-1]) != catalogue_hash(specs)

    def test_reordering_changes_it(self) -> None:
        """Order is the column order every artifact will be selected by."""
        specs = catalogue_specs()
        assert catalogue_hash([specs[1], specs[0], *specs[2:]]) != catalogue_hash(specs)

    def test_changing_prior_strength_changes_it(self) -> None:
        """The case a name-only hash misses: same names, different values."""
        specs = catalogue_specs()
        shrunk = next(s for s in specs if s.generator == "shrunk_rate")
        bumped = shrunk.__class__(
            **{**shrunk.__dict__, "prior_strength": shrunk.prior_strength + 1.0}
        )
        altered = [bumped if s is shrunk else s for s in specs]
        assert catalogue_hash(altered) != catalogue_hash(specs)

    def test_an_extension_feature_changes_it(self) -> None:
        assert catalogue_hash(extension=("discovered_x",)) != catalogue_hash()

    def test_the_feature_set_is_defined_once(self) -> None:
        """`dataset.py` composed this inline; two definitions drift apart."""
        from xg_alonso.features.career import CAREER_FEATURES
        from xg_alonso.features.catalogue import feature_names
        from xg_alonso.features.opponent import OPPONENT_FEATURES
        from xg_alonso.features.recency import RECENCY_FEATURES

        expected = tuple(feature_names()) + OPPONENT_FEATURES + CAREER_FEATURES + RECENCY_FEATURES
        assert model_feature_names() == expected

    def test_the_names_are_unique(self) -> None:
        names = model_feature_names()
        assert len(set(names)) == len(names)


class TestTheRulesHash:
    def test_it_ignores_the_snapshot_identity(
        self, scoring: ScoringRules, squad: SquadRules
    ) -> None:
        """`source_sha256` covers the whole price-bearing payload.

        Gating on it would mark every model rules-drifted within a day of
        training, because prices change daily and the hash covers them.
        """
        base = rules_snapshot_hash(scoring, squad)
        moved = scoring.model_copy(update={"source_sha256": "f" * 64})
        assert rules_snapshot_hash(moved, squad) == base

    def test_it_ignores_a_refetch(self, scoring: ScoringRules, squad: SquadRules) -> None:
        """Re-pinning identical rules must not invalidate a model."""
        base = rules_snapshot_hash(scoring, squad)
        refetched = scoring.model_copy(update={"fetched_at": datetime(2020, 1, 1, tzinfo=UTC)})
        assert rules_snapshot_hash(refetched, squad) == base

    def test_it_catches_the_goalkeeper_goal_error(
        self, scoring: ScoringRules, squad: SquadRules
    ) -> None:
        """The exact transcription error `domain/scoring.py` exists to prevent."""
        base = rules_snapshot_hash(scoring, squad)
        wrong = dict(scoring.goals_scored)
        wrong[Position.GKP] = 6
        assert (
            rules_snapshot_hash(scoring.model_copy(update={"goals_scored": wrong}), squad) != base
        )

    def test_it_catches_a_threshold_change(self, scoring: ScoringRules, squad: SquadRules) -> None:
        """No drift check can catch this — FPL does not publish it."""
        base = rules_snapshot_hash(scoring, squad)
        altered = scoring.thresholds.model_copy(update={"saves_per_point": 4})
        assert (
            rules_snapshot_hash(scoring.model_copy(update={"thresholds": altered}), squad) != base
        )

    def test_it_catches_a_squad_rule_change(self, scoring: ScoringRules, squad: SquadRules) -> None:
        base = rules_snapshot_hash(scoring, squad)
        altered = squad.model_copy(update={"max_per_club": 4})
        assert rules_snapshot_hash(scoring, altered) != base

    def test_it_is_stable_across_calls(self, scoring: ScoringRules, squad: SquadRules) -> None:
        assert rules_snapshot_hash(scoring, squad) == rules_snapshot_hash(scoring, squad)
