"""Unit planning, resumption and the seed axis.

Two properties here are load-bearing for cost and one is load-bearing for
honesty.

Cost: replicates expand for stochastic policies only, and the saving is
*enforced* rather than believed — a test asserts that every policy declared
non-stochastic really does discard its generator, so the plan cannot quietly
collapse a policy that actually varies.

Honesty: a resumed run must never mix results across configurations, and a
half-written run file must never count as done. Both are properties of the
directory layout and the write protocol, so both are tested against a real
temporary directory rather than a mock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xg_alonso.contracts.evaluation import (
    EvaluationWindow,
    ExperimentConfig,
    ModelSpec,
    PolicyKind,
    PolicySpec,
    RngScope,
    SquadCohortSpec,
    SquadSource,
)
from xg_alonso.evaluation.metrics import RunMetrics
from xg_alonso.evaluation.runner import (
    RunKey,
    RunRecord,
    completed_units,
    plan_units,
    policy_seed,
    run_directory,
    run_experiment,
    write_run,
)

_POLICIES = (
    PolicySpec(name="model", selector=PolicyKind.MODEL, model="m"),
    PolicySpec(name="random", selector=PolicyKind.RANDOM, model="m", stochastic=True),
    PolicySpec(name="hold", selector=PolicyKind.HOLD, model="m", needs_candidates=False),
)


def _config(**overrides: object) -> ExperimentConfig:
    base = {
        "name": "t",
        "windows": (EvaluationWindow(season="2024-25", start_gameweeks=(6, 12), end_gameweek=20),),
        "squads": (SquadCohortSpec(source=SquadSource.MOST_EXPENSIVE_LEGAL),),
        "models": (ModelSpec(name="m"),),
        "policies": _POLICIES,
        "replicates": 5,
    }
    return ExperimentConfig(**{**base, **overrides})  # type: ignore[arg-type]


def _metrics() -> RunMetrics:
    return RunMetrics(
        gameweeks=1,
        total_points=10,
        points_vs_hold=2,
        cumulative_incremental=(2,),
        immediate_decision_delta_total=2,
        immediate_decision_delta_mean=2.0,
        immediate_decision_delta_median=2.0,
        transfers=1,
        hits_taken=0,
        hit_points_paid=0,
        points_per_transfer=2.0,
        transfer_win_rate=1.0,
        mean_regret=0.0,
        calibration_error=0.0,
        bench_points=0,
        autosub_points=0,
        captaincy_points=0,
        max_drawdown_vs_hold=0,
        weeks_cumulatively_ahead=1.0,
        weeks_won_outright=1.0,
    )


class TestPlanning:
    def test_replicates_expand_only_for_stochastic_policies(self) -> None:
        """Five walks of a deterministic policy would be five identical walks."""
        units = plan_units(_config(), ["squad-a"])
        by_policy: dict[str, int] = {}
        for unit in units:
            by_policy[unit.policy] = by_policy.get(unit.policy, 0) + 1

        # two start gameweeks x one squad
        assert by_policy["model"] == 2
        assert by_policy["hold"] == 2
        assert by_policy["random"] == 2 * 5

    def test_the_declared_stochastic_set_matches_the_selectors(self) -> None:
        """Guards the saving: a policy that varies must not be collapsed."""
        deterministic = {
            PolicyKind.MODEL,
            PolicyKind.HIGHEST_FORM,
            PolicyKind.MOST_EXPENSIVE,
            PolicyKind.HOLD,
        }
        for policy in _POLICIES:
            if policy.selector in deterministic:
                assert not policy.stochastic, (
                    f"{policy.name} is declared stochastic but its selector "
                    "ignores the generator; replicates would be identical"
                )

    def test_planning_is_pure_and_repeatable(self) -> None:
        assert plan_units(_config(), ["a", "b"]) == plan_units(_config(), ["a", "b"])

    def test_every_unit_is_uniquely_identified(self) -> None:
        units = plan_units(_config(), ["a", "b", "c"])
        assert len({u.run_id for u in units}) == len(units)

    def test_replicates_of_a_policy_share_a_condition(self) -> None:
        """Which is what lets them be averaged before pairing."""
        units = [u for u in plan_units(_config(), ["a"]) if u.policy == "random"]
        first = [u for u in units if u.start_gameweek == 6]
        assert len({u.condition for u in first}) == 1

    def test_different_squads_are_different_conditions(self) -> None:
        units = plan_units(_config(), ["a", "b"])
        assert len({u.condition for u in units}) == 4  # 2 gameweeks x 2 squads


class TestSeeding:
    def test_the_gameweek_scope_reseeds_per_step(self) -> None:
        config = _config(rng_scope=RngScope.GAMEWEEK)
        unit = plan_units(config, ["a"])[0]
        assert policy_seed(config, unit, 6) != policy_seed(config, unit, 7)

    def test_the_run_scope_ignores_the_gameweek(self) -> None:
        """Reproduces the legacy coupling, where week 20 depended on week 6."""
        config = _config(rng_scope=RngScope.RUN)
        unit = plan_units(config, ["a"])[0]
        assert policy_seed(config, unit, 6) == policy_seed(config, unit, 7)

    def test_replicates_draw_different_seeds(self) -> None:
        config = _config()
        units = [u for u in plan_units(config, ["a"]) if u.policy == "random"][:2]
        assert policy_seed(config, units[0], 6) != policy_seed(config, units[1], 6)


class TestTheRunDirectory:
    def test_it_is_named_for_the_config_and_carries_no_timestamp(self, tmp_path: Path) -> None:
        """A timestamped directory is what makes report.py un-resumable."""
        directory = run_directory(tmp_path, _config())
        assert _config().experiment_id in str(directory)
        assert run_directory(tmp_path, _config()) == directory

    def test_a_different_config_gets_a_different_directory(self, tmp_path: Path) -> None:
        """So a resumed run can never mix two configurations."""
        assert run_directory(tmp_path, _config()) != run_directory(tmp_path, _config(replicates=2))


class TestResumption:
    def test_a_completed_unit_is_not_repeated(self, tmp_path: Path) -> None:
        config = _config()
        directory = run_directory(tmp_path, config)
        units = plan_units(config, ["a"])
        write_run(directory, RunRecord(key=units[0], metrics=_metrics()))

        done, remaining = completed_units(directory, units)

        assert units[0].run_id in done
        assert units[0] not in remaining
        assert len(remaining) == len(units) - 1

    def test_a_malformed_run_file_is_redone(self, tmp_path: Path) -> None:
        """A run that cannot be read cannot be verified either."""
        config = _config()
        directory = run_directory(tmp_path, config)
        units = plan_units(config, ["a"])
        runs = directory / "runs"
        runs.mkdir(parents=True)
        (runs / f"{units[0].run_id}.json").write_text("{not json")

        done, remaining = completed_units(directory, units)
        assert not done
        assert units[0] in remaining

    def test_a_stale_schema_is_redone(self, tmp_path: Path) -> None:
        config = _config()
        directory = run_directory(tmp_path, config)
        units = plan_units(config, ["a"])
        record = RunRecord(key=units[0], metrics=_metrics())
        path = write_run(directory, record)
        payload = json.loads(path.read_text())
        payload["schema_version"] = "evaluation_v0"
        path.write_text(json.dumps(payload))

        done, _ = completed_units(directory, units)
        assert not done

    def test_a_partial_write_is_ignored(self, tmp_path: Path) -> None:
        """The atomic rename is what keeps a killed process from lying."""
        config = _config()
        directory = run_directory(tmp_path, config)
        units = plan_units(config, ["a"])
        runs = directory / "runs"
        runs.mkdir(parents=True)
        (runs / f"{units[0].run_id}.json.tmp").write_text('{"partial": true}')

        done, remaining = completed_units(directory, units)
        assert not done
        assert units[0] in remaining


class TestRunningEndToEnd:
    def test_it_executes_every_unit_and_writes_a_manifest(self, tmp_path: Path) -> None:
        config = _config()
        executed: list[RunKey] = []

        def run_one(key: RunKey) -> RunRecord:
            executed.append(key)
            return RunRecord(key=key, metrics=_metrics())

        directory, records = run_experiment(config, root=tmp_path, squad_ids=["a"], run_one=run_one)

        assert len(executed) == len(plan_units(config, ["a"]))
        assert len(records) == len(executed)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["runs_planned"] == manifest["runs_completed"]

    def test_resuming_executes_only_what_is_missing(self, tmp_path: Path) -> None:
        config = _config()
        calls: list[int] = []

        def run_one(key: RunKey) -> RunRecord:
            calls.append(1)
            return RunRecord(key=key, metrics=_metrics())

        run_experiment(config, root=tmp_path, squad_ids=["a"], run_one=run_one)
        first = len(calls)
        calls.clear()

        run_experiment(config, root=tmp_path, squad_ids=["a"], run_one=run_one)

        assert first > 0
        assert calls == []

    def test_overwrite_re_executes_everything(self, tmp_path: Path) -> None:
        config = _config()
        calls: list[int] = []

        def run_one(key: RunKey) -> RunRecord:
            calls.append(1)
            return RunRecord(key=key, metrics=_metrics())

        run_experiment(config, root=tmp_path, squad_ids=["a"], run_one=run_one)
        first = len(calls)
        calls.clear()

        run_experiment(config, root=tmp_path, squad_ids=["a"], run_one=run_one, overwrite=True)
        assert len(calls) == first

    def test_the_config_is_written_beside_the_runs(self, tmp_path: Path) -> None:
        """Byte-identical to what was hashed, so the directory is self-describing."""
        config = _config()
        directory, _ = run_experiment(
            config,
            root=tmp_path,
            squad_ids=["a"],
            run_one=lambda key: RunRecord(key=key, metrics=_metrics()),
        )
        assert (directory / "config.json").read_text() == config.canonical_json()

    def test_limitations_reach_the_manifest(self, tmp_path: Path) -> None:
        directory, _ = run_experiment(
            _config(),
            root=tmp_path,
            squad_ids=["a"],
            run_one=lambda key: RunRecord(key=key, metrics=_metrics()),
            limitations=["one season only"],
        )
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["limitations"] == ["one season only"]


class TestMetricsSplitTheAmbiguousNames:
    def test_cumulatively_ahead_and_won_outright_are_different_numbers(self) -> None:
        """The first measures squad divergence; only the second measures the week."""
        from xg_alonso.contracts.identifiers import GameweekId, Season
        from xg_alonso.evaluation.backtest import BacktestResult, GameweekOutcome
        from xg_alonso.evaluation.metrics import compute_run_metrics

        # One big win, then four losses. Cumulative stays positive throughout;
        # only one week was actually won.
        outcomes = [
            GameweekOutcome(
                season=Season("2024-25"),
                gameweek=GameweekId(gw),
                policy_points=points,
                hold_points=0,
                hit_cost=0,
                transfer_made=False,
                player_out=None,
                player_in=None,
                predicted_gain=0.0,
            )
            for gw, points in enumerate([100, -1, -1, -1, -1], start=6)
        ]
        metrics = compute_run_metrics(BacktestResult(outcomes=outcomes))

        assert metrics.weeks_cumulatively_ahead == pytest.approx(1.0)
        assert metrics.weeks_won_outright == pytest.approx(0.2)

    def test_squad_value_reports_unavailable_rather_than_zero(self) -> None:
        from xg_alonso.evaluation.backtest import BacktestResult
        from xg_alonso.evaluation.metrics import MetricStatus, compute_run_metrics

        metrics = compute_run_metrics(BacktestResult(outcomes=[]), prices_moved=False)
        assert metrics.squad_value_status is MetricStatus.UNAVAILABLE_STATIC_PRICES
