"""The experiment configuration, and the properties its hash has to have.

The hash is the experiment's identity: the run directory is named for it, so a
resumed run can never mix results from two configurations. That only holds if
the hash is stable across processes and sensitive to every field that changes
what runs — and insensitive to the two that do not.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from xg_alonso.contracts.evaluation import (
    FULL_GRID,
    LEGACY_HEADLINE,
    PRESETS,
    SMOKE,
    EvaluationWindow,
    ExperimentConfig,
    ModelSpec,
    PolicyKind,
    PolicyParameters,
    PolicySpec,
    RngScope,
    SquadCohortSpec,
    SquadSource,
)


def _config(**overrides: object) -> ExperimentConfig:
    base = {
        "name": "t",
        "windows": (EvaluationWindow(season="2024-25", start_gameweeks=(6,), end_gameweek=12),),
        "squads": (SquadCohortSpec(source=SquadSource.MOST_EXPENSIVE_LEGAL),),
        "models": (ModelSpec(name="closed_form"),),
        "policies": (PolicySpec(name="hold", selector=PolicyKind.HOLD, model="closed_form"),),
    }
    return ExperimentConfig(**{**base, **overrides})  # type: ignore[arg-type]


class TestTheHashIsTheIdentity:
    def test_it_is_stable_across_processes(self) -> None:
        """A run directory named for a per-process hash would be unresumable."""
        here = SMOKE.config_hash()
        code = "from xg_alonso.contracts.evaluation import SMOKE;print(SMOKE.config_hash())"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == here

    def test_recorded_only_fields_do_not_change_it(self) -> None:
        """Otherwise any commit would orphan a half-finished experiment."""
        base = _config()
        assert base.config_hash() == _config(code_version="abc123").config_hash()
        assert base.config_hash() == _config(git_dirty=True).config_hash()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("replicates", 9),
            ("root_seed", 1),
            ("rng_scope", RngScope.RUN),
            ("bootstrap_resamples", 500),
            ("confidence", 0.9),
            ("max_workers", 4),
            ("require_exact_rules", True),
            ("data_manifest_sha256", "d" * 64),
            ("parameters", PolicyParameters(min_net_gain=0.4)),
        ],
    )
    def test_every_field_that_changes_the_run_changes_the_hash(
        self, field: str, value: object
    ) -> None:
        assert _config(**{field: value}).config_hash() != _config().config_hash()

    def test_the_experiment_id_carries_the_name_and_the_hash(self) -> None:
        assert SMOKE.experiment_id.startswith("smoke-")
        assert SMOKE.config_hash().startswith(SMOKE.experiment_id.split("-")[-1])


class TestPolicyParametersCannotHideAKnob:
    def test_an_undeclared_knob_is_refused(self) -> None:
        """`extra="forbid"` is what makes the tuned-on-test check complete."""
        with pytest.raises(ValidationError):
            PolicyParameters(some_new_threshold=0.5)  # type: ignore[call-arg]

    def test_tuning_provenance_defaults_to_never_tuned(self) -> None:
        assert PolicyParameters().tuned_on_seasons == ()


class TestTheOracleIsGated:
    def test_an_oracle_without_diagnostic_is_refused(self) -> None:
        """Negative control: hindsight must not be presentable as a policy."""
        with pytest.raises(ValidationError, match="reads outcomes"):
            _config(
                policies=(
                    PolicySpec(name="oracle", selector=PolicyKind.ORACLE, model="closed_form"),
                ),
                allow_oracle=True,
            )

    def test_an_oracle_without_allow_oracle_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="reads outcomes"):
            _config(
                policies=(
                    PolicySpec(
                        name="oracle",
                        selector=PolicyKind.ORACLE,
                        model="closed_form",
                        diagnostic=True,
                    ),
                ),
                allow_oracle=False,
            )

    def test_a_properly_declared_oracle_is_permitted(self) -> None:
        config = _config(
            policies=(
                PolicySpec(
                    name="oracle",
                    selector=PolicyKind.ORACLE,
                    model="closed_form",
                    diagnostic=True,
                ),
            ),
            allow_oracle=True,
        )
        assert config.policies[0].diagnostic

    def test_no_shipped_preset_enables_the_oracle(self) -> None:
        for preset in PRESETS.values():
            assert not preset.allow_oracle


class TestStructuralValidation:
    def test_a_policy_naming_an_undeclared_model_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="undeclared models"):
            _config(
                policies=(PolicySpec(name="p", selector=PolicyKind.MODEL, model="does_not_exist"),),
            )

    def test_a_start_gameweek_after_the_end_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not before the end"):
            EvaluationWindow(season="2024-25", start_gameweeks=(30,), end_gameweek=25)

    def test_an_empty_window_list_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _config(windows=())


class TestThePresetsSayWhatTheyAre:
    def test_only_random_is_stochastic(self) -> None:
        """The seed axis expands for these and no others."""
        stochastic = {p.name for p in FULL_GRID.policies if p.stochastic}
        assert stochastic == {"random"}

    def test_hold_does_not_need_the_candidate_search(self) -> None:
        hold = next(p for p in FULL_GRID.policies if p.name == "hold")
        assert not hold.needs_candidates

    def test_the_legacy_preset_uses_the_run_scoped_rng(self) -> None:
        """It reproduces the published report's conditions, coupling included."""
        assert LEGACY_HEADLINE.rng_scope is RngScope.RUN
        assert LEGACY_HEADLINE.replicates == 1
        assert len(LEGACY_HEADLINE.squads) == 1

    def test_the_full_grid_spans_three_seasons_and_twelve_squads(self) -> None:
        assert FULL_GRID.evaluation_seasons == ("2023-24", "2024-25", "2025-26")
        assert sum(s.count for s in FULL_GRID.squads) == 12

    def test_the_earliest_backfilled_season_is_training_only(self) -> None:
        """2022-23 has no prior season to train from, so it cannot be evaluated."""
        assert "2022-23" not in FULL_GRID.evaluation_seasons
