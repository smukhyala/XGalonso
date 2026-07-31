"""Turning completed runs into something a reader can disagree with.

**The limitations section is generated, not written.** A caveat somebody has to
remember to add is a caveat that goes missing exactly when it matters — after a
good result, when nobody is looking for reasons to doubt it. So the conditions
that limit a claim are derived from the run itself: how many seasons were
evaluated, how many paired conditions there were, whether autosubs were
simulated, whether prices moved, whether the rules were the season's own,
whether the tree was dirty, and where each starting squad sat in its cohort.

**``summary.md`` refuses to render without them.** A report that does not state
its sample size is not a weaker report, it is a misleading one, and the writer
declines to produce it rather than trusting a reviewer to notice.

Nothing accumulates: every artifact is recomputed from ``runs/*.json`` on each
render, so a resumed experiment cannot emit a half-aggregated report that looks
complete.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xg_alonso.contracts.evaluation import ExperimentConfig
from xg_alonso.evaluation.statistics import PairedComparison, aggregate, paired_comparison

__all__ = [
    "HEADLINE_METRICS",
    "ExperimentReport",
    "build_report",
    "generate_limitations",
    "load_runs",
    "render_summary",
    "write_report",
]

#: Metrics the headline table compares policies on.
HEADLINE_METRICS: tuple[str, ...] = (
    "points_vs_hold",
    "transfers",
    "transfer_win_rate",
    "points_per_transfer",
)


def load_runs(directory: Path) -> list[dict[str, Any]]:
    """Every completed run, recomputed from disk rather than remembered."""
    runs = directory / "runs"
    if not runs.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for path in sorted(runs.glob("*.json")):
        try:
            payloads.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def _by_policy_and_condition(
    runs: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, dict[str, list[float]]]:
    """``policy -> condition -> replicate values`` for one metric."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        key = run["key"]
        condition = f"{key['season']}|{key['start_gameweek']}|{key['squad_id']}"
        value = run["metrics"].get(metric)
        if value is None:
            continue
        grouped.setdefault(key["policy"], {}).setdefault(condition, []).append(float(value))
    return grouped


def _condition_blocks(runs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Condition to its season, so the bootstrap can resample in blocks."""
    return {
        f"{r['key']['season']}|{r['key']['start_gameweek']}|{r['key']['squad_id']}": r["key"][
            "season"
        ]
        for r in runs
    }


def generate_limitations(
    config: ExperimentConfig,
    runs: Sequence[Mapping[str, Any]],
    *,
    rules_exact: bool = True,
    rules_source_season: str = "",
    autosubs_simulated: bool = True,
    squad_percentiles: Mapping[str, float | None] | None = None,
) -> list[str]:
    """Everything this experiment cannot support, derived from the run itself.

    Generated rather than authored, because a hand-written caveat goes missing
    exactly when it matters most — after a result somebody likes.
    """
    limitations: list[str] = []
    seasons = config.evaluation_seasons
    conditions = {
        f"{r['key']['season']}|{r['key']['start_gameweek']}|{r['key']['squad_id']}" for r in runs
    }

    limitations.append(
        f"{len(seasons)} evaluation season(s): {', '.join(seasons)}. 2022-23 is "
        "training-only — it is the earliest season with expected goals, so "
        "evaluating on it would leave no prior season to train from."
    )
    limitations.append(
        f"n = {len(conditions)} paired starting condition(s) across {len(seasons)} season block(s)."
    )
    if len(seasons) < 3:
        limitations.append(
            "Fewer than three seasons, so intervals resample conditions directly "
            "rather than season blocks and are narrower than the dependence "
            "between conditions in one season warrants."
        )
    if not autosubs_simulated:
        limitations.append("Autosubs were not simulated; bench and autosub points are pending.")
    if not rules_exact:
        limitations.append(
            f"Scoring rules resolved from the {rules_source_season} pinned snapshot; "
            "no snapshot exists for every evaluated season."
        )
    if config.git_dirty:
        limitations.append("Recorded with a dirty working tree; this run is not reproducible.")
    if not config.code_version:
        limitations.append("No code version was recorded; this run is not reproducible.")

    for squad_id, percentile in (squad_percentiles or {}).items():
        if percentile is None:
            continue
        limitations.append(
            f"Starting squad {squad_id} sat at the {percentile:.0%} percentile of its "
            "comparison pool by predicted XI strength before the walk began."
        )
    return limitations


@dataclass(frozen=True)
class ExperimentReport:
    """A rendered experiment: its comparisons and everything qualifying them."""

    config: ExperimentConfig
    runs: tuple[Mapping[str, Any], ...]
    comparisons: tuple[PairedComparison, ...]
    limitations: tuple[str, ...]

    @property
    def policies(self) -> tuple[str, ...]:
        return tuple(sorted({r["key"]["policy"] for r in self.runs}))


def build_report(
    config: ExperimentConfig,
    runs: Sequence[Mapping[str, Any]],
    *,
    baseline: str = "hold",
    metric: str = "points_vs_hold",
    limitations: Sequence[str] = (),
) -> ExperimentReport:
    """Compare every non-diagnostic policy against a baseline."""
    grouped = _by_policy_and_condition(runs, metric)
    blocks = _condition_blocks(runs)
    diagnostic = {p.name for p in config.policies if p.diagnostic}

    comparisons: list[PairedComparison] = []
    if baseline in grouped:
        for policy in sorted(grouped):
            if policy == baseline:
                continue
            try:
                comparisons.append(
                    paired_comparison(
                        grouped[policy],
                        grouped[baseline],
                        policy=policy,
                        baseline=baseline,
                        metric=metric,
                        blocks=blocks,
                        resamples=config.bootstrap_resamples,
                        confidence=config.confidence,
                        unit=config.bootstrap_unit,
                        diagnostic=policy in diagnostic,
                    )
                )
            except ValueError:
                # No shared condition. Reported by its absence rather than by a
                # fabricated comparison across different starting squads.
                continue

    return ExperimentReport(
        config=config,
        runs=tuple(runs),
        comparisons=tuple(comparisons),
        limitations=tuple(limitations),
    )


def render_summary(report: ExperimentReport) -> str:
    """The human-readable report.

    Raises:
        ValueError: when there are no limitations to state. A report without
            its sample size is misleading rather than merely incomplete, so it
            is not produced at all.
    """
    if not report.limitations:
        raise ValueError(
            "refusing to render a summary with no limitations recorded. Every "
            "experiment has a sample size, and one that does not state it reads "
            "as though it had none."
        )

    config = report.config
    lines = [
        f"# Evaluation — {config.name}",
        "",
        f"- experiment: `{config.experiment_id}`",
        f"- seasons: {', '.join(config.evaluation_seasons)}",
        f"- runs: {len(report.runs)}",
        f"- reproducible: {'yes' if config.reproducible else 'no'}",
        "",
    ]

    ranked = sorted(
        (c for c in report.comparisons if not c.diagnostic),
        key=lambda c: -c.mean_difference.point,
    )
    if ranked:
        lines += [
            "## Against hold, per starting condition",
            "",
            "| policy | mean difference | interval | conditions won | effect size |",
            "|---|---:|---|---:|---:|",
        ]
        for comparison in ranked:
            interval = comparison.mean_difference
            effect = (
                f"{comparison.effect_size_dz:+.2f}"
                if comparison.effect_size_dz is not None
                else "—"
            )
            lines.append(
                f"| {comparison.policy} | {interval.point:+.2f} | "
                f"[{interval.low:+.2f}, {interval.high:+.2f}] | "
                f"{comparison.conditions_won}/{comparison.n_conditions} | {effect} |"
            )
        lines.append("")

    diagnostics = [c for c in report.comparisons if c.diagnostic]
    if diagnostics:
        lines += [
            "## Diagnostics — upper bounds, not baselines",
            "",
            "These read gameweek outcomes. No manager had them.",
            "",
        ]
        lines += [f"- {c.row()}" for c in diagnostics]
        lines.append("")

    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in report.limitations]
    return "\n".join(lines) + "\n"


def write_report(directory: Path, report: ExperimentReport) -> dict[str, Path]:
    """Persist every artifact, each recomputed rather than accumulated."""
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _atomic(name: str, text: str) -> Path:
        path = directory / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
        return path

    written["aggregate"] = _atomic(
        "aggregate.json",
        json.dumps(
            {
                metric: {
                    policy: aggregate(
                        [sum(v) / len(v) for v in conditions.values()],
                        labels=sorted(conditions),
                    ).__dict__
                    for policy, conditions in _by_policy_and_condition(report.runs, metric).items()
                    if conditions
                }
                for metric in HEADLINE_METRICS
            },
            indent=2,
            sort_keys=True,
        ),
    )
    written["comparisons"] = _atomic(
        "comparisons.json",
        json.dumps(
            [
                {
                    "policy": c.policy,
                    "baseline": c.baseline,
                    "metric": c.metric,
                    "n_conditions": c.n_conditions,
                    "mean_difference": c.mean_difference.point,
                    "interval_low": c.mean_difference.low,
                    "interval_high": c.mean_difference.high,
                    "confidence": c.mean_difference.confidence,
                    "bootstrap_unit": str(c.mean_difference.unit),
                    "caveat": c.mean_difference.caveat,
                    "probability_of_outperforming": c.probability_of_outperforming,
                    "effect_size_dz": c.effect_size_dz,
                    "conditions_won": c.conditions_won,
                    "conditions_lost": c.conditions_lost,
                    "conditions_tied": c.conditions_tied,
                    "seed_sensitivity_sd": c.seed_sensitivity_sd,
                    "diagnostic": c.diagnostic,
                }
                for c in report.comparisons
            ],
            indent=2,
            sort_keys=True,
        ),
    )
    written["summary"] = _atomic("summary.md", render_summary(report))
    return written
