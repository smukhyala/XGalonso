"""Persist backtest results so they survive the process that produced them.

Until now every measured number was echoed to a terminal and lost. That made it
impossible to answer the only question that matters when changing the model or
the objective: *did this help?* A comparison needs a before, and a before has to
have been written down.

Each run is a JSON record plus a Markdown summary, written under a timestamped
directory. The JSON carries enough provenance to reproduce the run — code
version, model fingerprint, the seasons the model was trained on, and every
per-gameweek outcome — so a later run can be diffed against it rather than
remembered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from xg_alonso.evaluation.backtest import BacktestResult

__all__ = [
    "BacktestReport",
    "compare_to_previous",
    "load_reports",
    "write_report",
]


@dataclass(frozen=True)
class BacktestReport:
    """One backtest run, with everything needed to compare it against another."""

    run_id: str
    created_at: datetime
    season: str
    first_gameweek: int
    last_gameweek: int
    code_version: str
    model_fingerprint: str | None
    model_trained_on: tuple[str, ...]
    results: dict[str, BacktestResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "season": self.season,
            "first_gameweek": self.first_gameweek,
            "last_gameweek": self.last_gameweek,
            "code_version": self.code_version,
            "model_fingerprint": self.model_fingerprint,
            "model_trained_on": list(self.model_trained_on),
            "policies": {
                name: {
                    "total_incremental": result.total_incremental,
                    "transfers_made": result.transfers_made,
                    "total_hits": result.total_hits,
                    "decision_win_rate": result.decision_win_rate,
                    "mean_decision_delta": result.mean_decision_delta,
                    "mean_regret": result.mean_regret,
                    "calibration_error": result.calibration_error,
                    "gameweeks": result.gameweeks,
                    "outcomes": [
                        {
                            "gameweek": int(o.gameweek),
                            "policy_points": o.policy_points,
                            "hold_points": o.hold_points,
                            "hit_cost": o.hit_cost,
                            "transfer_made": o.transfer_made,
                            "player_out": int(o.player_out) if o.player_out else None,
                            "player_in": int(o.player_in) if o.player_in else None,
                            "predicted_gain": o.predicted_gain,
                            "decision_delta": o.decision_delta,
                        }
                        for o in result.outcomes
                    ],
                }
                for name, result in self.results.items()
            },
        }

    def to_markdown(self) -> str:
        """A human-readable summary for skimming a run's history."""
        lines = [
            f"# Backtest — {self.season} GW{self.first_gameweek}-{self.last_gameweek}",
            "",
            f"- run: `{self.run_id}`",
            f"- created: {self.created_at.isoformat()}",
            f"- code: `{self.code_version[:12]}`",
        ]
        if self.model_fingerprint:
            lines.append(
                f"- model: `{self.model_fingerprint[:12]}` "
                f"trained on {', '.join(self.model_trained_on)}"
            )
        lines += [
            "",
            "| policy | vs hold | transfers | win rate | pts/xfer | pred err |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, r in sorted(self.results.items(), key=lambda kv: -kv[1].total_incremental):
            lines.append(
                f"| {name} | {r.total_incremental:+d} | {r.transfers_made} | "
                f"{r.decision_win_rate:.0%} | {r.mean_decision_delta:+.2f} | "
                f"{r.calibration_error:.2f} |"
            )
        lines += [
            "",
            "A single season and one starting squad is a small sample. Treat the",
            "ordering as a signal, not the margin.",
        ]
        return "\n".join(lines)


def write_report(report: BacktestReport, root: Path) -> Path:
    """Write a run to ``root`` as JSON plus Markdown, returning the directory.

    The directory name is sortable and unique, so runs accumulate in
    chronological order without a database.
    """
    stamp = report.created_at.strftime("%Y%m%dT%H%M%SZ")
    directory = root / f"{stamp}-{report.season}-gw{report.first_gameweek}-{report.last_gameweek}"
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "result.json").write_text(json.dumps(report.to_dict(), indent=1, sort_keys=True))
    (directory / "summary.md").write_text(report.to_markdown())
    return directory


def load_reports(root: Path, *, season: str | None = None) -> list[dict[str, Any]]:
    """Load previous runs, newest first, so a change can be diffed against them."""
    if not root.exists():
        return []

    loaded: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/result.json"), reverse=True):
        try:
            record: dict[str, Any] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # A malformed report should not prevent reading the rest — history
            # is more useful partially than not at all.
            continue
        if season is None or record.get("season") == season:
            loaded.append(record)
    return loaded


def compare_to_previous(current: BacktestReport, root: Path) -> dict[str, tuple[int, int]] | None:
    """Per-policy ``(previous, current)`` totals against the last matching run.

    Returns ``None`` when there is no prior run for this season and window —
    the first measurement of anything has nothing to be compared against, and
    saying so is better than inventing a baseline.
    """
    history = [
        r
        for r in load_reports(root, season=current.season)
        if r.get("first_gameweek") == current.first_gameweek
        and r.get("last_gameweek") == current.last_gameweek
        and r.get("run_id") != current.run_id
    ]
    if not history:
        return None

    previous = history[0]["policies"]
    return {
        name: (previous[name]["total_incremental"], result.total_incremental)
        for name, result in current.results.items()
        if name in previous
    }
