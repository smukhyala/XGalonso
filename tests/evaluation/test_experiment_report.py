"""Reports, and the caveats they cannot be written without.

The property that matters most here is a refusal: `summary.md` will not render
without a limitations section. A report that omits its sample size is not a
weaker report, it is a misleading one, and the writer declines rather than
trusting a reviewer to notice what is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xg_alonso.contracts.evaluation import (
    EvaluationWindow,
    ExperimentConfig,
    ModelSpec,
    PolicyKind,
    PolicySpec,
    SquadCohortSpec,
    SquadSource,
)
from xg_alonso.evaluation.experiment_report import (
    build_report,
    generate_limitations,
    load_runs,
    render_summary,
    write_report,
)


def _config(**overrides: object) -> ExperimentConfig:
    base = {
        "name": "t",
        "windows": (EvaluationWindow(season="2024-25", start_gameweeks=(6,), end_gameweek=20),),
        "squads": (SquadCohortSpec(source=SquadSource.MOST_EXPENSIVE_LEGAL),),
        "models": (ModelSpec(name="m"),),
        "policies": (
            PolicySpec(name="model", selector=PolicyKind.MODEL, model="m"),
            PolicySpec(name="hold", selector=PolicyKind.HOLD, model="m"),
        ),
        "bootstrap_resamples": 200,
        "code_version": "abc123",
    }
    return ExperimentConfig(**{**base, **overrides})  # type: ignore[arg-type]


def _run(policy: str, squad: str, value: float, *, season: str = "2024-25") -> dict[str, Any]:
    return {
        "key": {
            "experiment_id": "e",
            "season": season,
            "start_gameweek": 6,
            "end_gameweek": 20,
            "squad_id": squad,
            "policy": policy,
            "replicate": 0,
        },
        "schema_version": "evaluation_v1",
        "metrics": {
            "points_vs_hold": value,
            "transfers": 5,
            "transfer_win_rate": 0.6,
            "points_per_transfer": 1.2,
        },
    }


class TestLimitationsAreGenerated:
    def test_the_sample_size_is_always_stated(self) -> None:
        runs = [_run("model", "a", 10.0), _run("hold", "a", 0.0)]
        limitations = generate_limitations(_config(), runs)
        assert any("paired starting condition" in item for item in limitations)

    def test_a_thin_season_sample_says_the_interval_is_narrow(self) -> None:
        limitations = generate_limitations(_config(), [_run("model", "a", 1.0)])
        assert any("Fewer than three seasons" in item for item in limitations)

    def test_a_dirty_tree_is_recorded_as_unreproducible(self) -> None:
        limitations = generate_limitations(_config(git_dirty=True), [_run("model", "a", 1.0)])
        assert any("dirty working tree" in item for item in limitations)

    def test_inexact_rules_are_recorded(self) -> None:
        limitations = generate_limitations(
            _config(),
            [_run("model", "a", 1.0)],
            rules_exact=False,
            rules_source_season="2026-27",
        )
        assert any("2026-27 pinned snapshot" in item for item in limitations)

    def test_unsimulated_autosubs_are_recorded(self) -> None:
        limitations = generate_limitations(
            _config(), [_run("model", "a", 1.0)], autosubs_simulated=False
        )
        assert any("Autosubs were not simulated" in item for item in limitations)

    def test_a_squad_percentile_reaches_the_report(self) -> None:
        """Answers "was this a favourable start" without anyone having to ask."""
        limitations = generate_limitations(
            _config(), [_run("model", "a", 1.0)], squad_percentiles={"squad-a": 0.92}
        )
        assert any("92% percentile" in item for item in limitations)


class TestTheSummaryRefusesToMislead:
    def test_rendering_without_limitations_is_refused(self) -> None:
        report = build_report(_config(), [_run("model", "a", 5.0), _run("hold", "a", 0.0)])
        with pytest.raises(ValueError, match="no limitations recorded"):
            render_summary(report)

    def test_with_limitations_it_renders(self) -> None:
        runs = [_run("model", "a", 5.0), _run("hold", "a", 0.0)]
        report = build_report(_config(), runs, limitations=generate_limitations(_config(), runs))
        text = render_summary(report)
        assert "## Limitations" in text
        assert "paired starting condition" in text


class TestComparisons:
    def test_a_policy_is_compared_against_hold_per_condition(self) -> None:
        runs = [
            _run("model", "a", 10.0),
            _run("hold", "a", 0.0),
            _run("model", "b", 6.0),
            _run("hold", "b", 0.0),
        ]
        report = build_report(_config(), runs, limitations=["x"])
        model = next(c for c in report.comparisons if c.policy == "model")

        assert model.n_conditions == 2
        assert model.differences.mean == pytest.approx(8.0)
        assert model.conditions_won == 2

    def test_a_diagnostic_policy_is_labelled(self) -> None:
        config = _config(
            policies=(
                PolicySpec(name="oracle", selector=PolicyKind.ORACLE, model="m", diagnostic=True),
                PolicySpec(name="hold", selector=PolicyKind.HOLD, model="m"),
            ),
            allow_oracle=True,
        )
        runs = [_run("oracle", "a", 50.0), _run("hold", "a", 0.0)]
        report = build_report(config, runs, limitations=["x"])

        assert next(c for c in report.comparisons if c.policy == "oracle").diagnostic
        text = render_summary(report)
        assert "upper bounds, not baselines" in text
        assert "No manager had them" in text

    def test_policies_sharing_no_condition_are_omitted_not_faked(self) -> None:
        """A comparison across different squads would measure the squads."""
        runs = [_run("model", "a", 5.0), _run("hold", "b", 0.0)]
        report = build_report(_config(), runs, limitations=["x"])
        assert report.comparisons == ()


class TestArtifacts:
    def test_everything_is_written_and_reloadable(self, tmp_path: Path) -> None:
        runs = [_run("model", "a", 5.0), _run("hold", "a", 0.0)]
        report = build_report(_config(), runs, limitations=["x"])
        written = write_report(tmp_path, report)

        assert set(written) == {"aggregate", "comparisons", "summary"}
        aggregate = json.loads(written["aggregate"].read_text())
        assert "points_vs_hold" in aggregate
        comparisons = json.loads(written["comparisons"].read_text())
        assert comparisons[0]["policy"] == "model"

    def test_the_aggregate_carries_a_distribution_not_a_mean(self, tmp_path: Path) -> None:
        """A naked mean has no path to a report, by construction."""
        runs = [_run("model", s, float(i)) for i, s in enumerate("abcd")]
        runs += [_run("hold", s, 0.0) for s in "abcd"]
        report = build_report(_config(), runs, limitations=["x"])
        written = write_report(tmp_path, report)

        entry = json.loads(written["aggregate"].read_text())["points_vs_hold"]["model"]
        assert {"p5", "p25", "p75", "p95", "worst", "best", "sd"} <= set(entry)

    def test_loading_runs_skips_unreadable_files(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "good.json").write_text(json.dumps(_run("model", "a", 1.0)))
        (runs_dir / "bad.json").write_text("{not json")

        assert len(load_runs(tmp_path)) == 1
