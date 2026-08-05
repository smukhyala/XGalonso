"""Pool-scoped verdicts: what transfers between managers, and what must not.

The rule under test is *the program is universal; the verdict is not.* A DSL
feature program is a statement about football — nothing in a manager's bank
balance changes whether xG times expected minutes predicts points — so one
program has one spec row however many managers discover it. What is
pool-specific is the **evidence**.

:meth:`TestTransferRule.test_a_narrow_verdict_does_not_transfer_to_another_narrow_pool`
is the important one. Evidence transfers to a *wider* population and never to a
narrower one: "this helped across the whole league" is weaker but real evidence
for any subset of it, while "this helped among premium forwards" says nothing
whatever about budget defenders. Reusing the latter would be the registry
asserting a result it never measured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xg_alonso.contracts.discovery import (
    AcceptanceStatus,
    DiscoveredFeatureSpec,
    FeatureEvaluation,
    ValidationStatus,
)
from xg_alonso.discovery.registry import DiscoveryRegistry
from xg_alonso.storage import ParquetTableStore

_OBJECTIVE = "expected_points_balanced_h1_neutral"
_PREMIUM = "F|over_11.0|15_35pc|free"
_BUDGET = "GD|under_4.0|60_85pc|free"


@pytest.fixture
def registry(tmp_path: Path) -> DiscoveryRegistry:
    return DiscoveryRegistry(ParquetTableStore(tmp_path / "discovery"))


def _spec(name: str = "xg_x_minutes") -> DiscoveredFeatureSpec:
    return DiscoveredFeatureSpec(
        id=f"family.{name}",
        name=name,
        version=f"{name}_version".ljust(16, "0"),
        program='{"kind":"source","column":"minutes","scope":"history"}',
        input_columns=("minutes",),
        validation_status=ValidationStatus.LEAKAGE_PASSED,
    )


def _evaluation(
    version: str,
    *,
    pool: str,
    accepted: AcceptanceStatus = AcceptanceStatus.ACCEPTED,
) -> FeatureEvaluation:
    return FeatureEvaluation(
        feature_id="family.xg_x_minutes",
        feature_version=version,
        objective_id=_OBJECTIVE,
        pool_signature=pool,
        backtest_start=1,
        backtest_end=10,
        accepted=accepted,
        # The contract refuses an unexplained rejection: it would be
        # indistinguishable from a crash.
        rejection_reason="" if accepted.is_usable else "did not beat the controls",
        utility=0.2,
    )


class TestTheProgramIsUniversal:
    def test_one_program_is_one_spec_row(self, registry: DiscoveryRegistry) -> None:
        """Two managers discovering the same feature must not duplicate it."""
        spec = _spec()
        registry.register_feature(spec)
        registry.register_feature(spec)
        assert len([f for f in registry.features() if f.version == spec.version]) == 1

    def test_evaluations_are_append_only_per_pool(self, registry: DiscoveryRegistry) -> None:
        spec = _spec()
        registry.register_feature(spec)
        registry.record_evaluation(_evaluation(spec.version, pool="global"))
        registry.record_evaluation(_evaluation(spec.version, pool=_PREMIUM))
        found = registry.evaluations(feature_version=spec.version, objective_id=_OBJECTIVE)
        assert {e.pool_signature for e in found} == {"global", _PREMIUM}


class TestTransferRule:
    def test_an_exact_pool_match_carries_no_note(self, registry: DiscoveryRegistry) -> None:
        spec = _spec()
        registry.register_feature(spec)
        registry.record_evaluation(_evaluation(spec.version, pool=_PREMIUM))

        found = registry.accepted_features(_OBJECTIVE, pool_signature=_PREMIUM)
        assert [(s.name, note) for s, note in found] == [(spec.name, "")]

    def test_a_global_verdict_transfers_with_a_note(self, registry: DiscoveryRegistry) -> None:
        """Weaker but real evidence for any subset — and labelled as borrowed."""
        spec = _spec()
        registry.register_feature(spec)
        registry.record_evaluation(_evaluation(spec.version, pool="global"))

        found = registry.accepted_features(_OBJECTIVE, pool_signature=_PREMIUM)
        assert len(found) == 1
        _, note = found[0]
        assert "not measured under your constraints" in note

    def test_a_narrow_verdict_does_not_transfer_to_another_narrow_pool(
        self, registry: DiscoveryRegistry
    ) -> None:
        """The rule that stops the registry asserting what it never measured.

        A feature shown to help among premium forwards says nothing about budget
        defenders. Silently reusing it would attach real-looking evidence to a
        population it was never tested on.
        """
        spec = _spec()
        registry.register_feature(spec)
        registry.record_evaluation(_evaluation(spec.version, pool=_PREMIUM))

        assert registry.accepted_features(_OBJECTIVE, pool_signature=_BUDGET) == []

    def test_an_exact_match_is_preferred_over_a_global_one(
        self, registry: DiscoveryRegistry
    ) -> None:
        spec = _spec()
        registry.register_feature(spec)
        registry.record_evaluation(_evaluation(spec.version, pool="global"))
        registry.record_evaluation(_evaluation(spec.version, pool=_PREMIUM))

        found = registry.accepted_features(_OBJECTIVE, pool_signature=_PREMIUM)
        assert len(found) == 1
        assert found[0][1] == "", "an exact measurement must not be reported as borrowed"

    def test_a_rejected_verdict_never_transfers(self, registry: DiscoveryRegistry) -> None:
        spec = _spec()
        registry.register_feature(spec)
        registry.record_evaluation(
            _evaluation(spec.version, pool="global", accepted=AcceptanceStatus.REJECTED)
        )
        assert registry.accepted_features(_OBJECTIVE, pool_signature=_PREMIUM) == []

    def test_a_different_objective_never_transfers(self, registry: DiscoveryRegistry) -> None:
        spec = _spec()
        registry.register_feature(spec)
        registry.record_evaluation(_evaluation(spec.version, pool="global"))
        assert registry.accepted_features("some_other_objective", pool_signature="global") == []


class TestUnscopedReadIsUnchanged:
    """Callers that do not ask about pools keep the previous behaviour."""

    def test_without_a_pool_the_latest_verdict_wins(self, registry: DiscoveryRegistry) -> None:
        spec = _spec()
        registry.register_feature(spec)
        registry.record_evaluation(_evaluation(spec.version, pool="global"))

        found = registry.accepted_features(_OBJECTIVE)
        assert [s.name for s, _ in found] == [spec.name]

    def test_an_empty_registry_returns_nothing(self, registry: DiscoveryRegistry) -> None:
        assert registry.accepted_features(_OBJECTIVE) == []
        assert registry.accepted_features(_OBJECTIVE, pool_signature=_PREMIUM) == []


class TestNoArtifactExplosion:
    """Per-manager conditioning must not become a per-manager registry."""

    def test_many_pools_share_one_spec_row(self, registry: DiscoveryRegistry) -> None:
        spec = _spec()
        registry.register_feature(spec)
        for index in range(50):
            registry.record_evaluation(
                _evaluation(spec.version, pool=f"GD|band_{index}|60_85pc|free")
            )

        # Fifty distinct pools, fifty evaluation rows — and exactly one feature.
        assert len([f for f in registry.features() if f.version == spec.version]) == 1
        assert (
            len(registry.evaluations(feature_version=spec.version, objective_id=_OBJECTIVE)) == 50
        )
