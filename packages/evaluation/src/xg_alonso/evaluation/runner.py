"""Planning, running and resuming an experiment.

**The unit list is derivable without touching disk.** :func:`plan_units` is
pure and total, so "how far along am I" is ``completed / len(units)`` rather
than a guess, and a resumed run knows what it still owes before it opens a
single file.

**The seed axis expands only for stochastic policies.** ``model``,
``highest_form``, ``most_expensive`` and ``hold`` all discard their generator,
so five replicates would be five identical walks. Only ``random`` gets them.
That is a 3.6x saving on the full grid and it is *enforced* — a test asserts
identity across seeds for every policy declared non-stochastic, rather than the
saving resting on a belief about code nobody re-reads.

**Aggregates are always recomputed from the run files.** Nothing accumulates
incrementally, because a partially-aggregated report is the single most likely
way a framework of this shape ends up lying: it looks complete, it is
internally consistent, and it describes a subset nobody chose.

**The run directory is named for the config hash and carries no timestamp.**
``report.py`` timestamps its directories, which is exactly what makes it
un-resumable. Identity here is what was run, not when.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from xg_alonso.contracts.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    ExperimentConfig,
    RngScope,
)
from xg_alonso.contracts.provenance import utc_now
from xg_alonso.contracts.seeds import derive_seed
from xg_alonso.evaluation.metrics import RunMetrics

__all__ = [
    "RunKey",
    "RunRecord",
    "completed_units",
    "plan_units",
    "policy_seed",
    "read_run",
    "run_directory",
    "write_run",
]


@dataclass(frozen=True)
class RunKey:
    """One unit of work: a policy walking one condition once."""

    experiment_id: str
    season: str
    start_gameweek: int
    end_gameweek: int
    squad_id: str
    policy: str
    replicate: int = 0

    @property
    def condition(self) -> str:
        """The starting condition this unit belongs to.

        The unit of analysis for every paired comparison — see
        :mod:`xg_alonso.evaluation.statistics`. Replicates of one policy share
        a condition, which is what lets them be averaged before pairing.

        ``end_gameweek`` is part of it. Two windows may name the same season and
        the same start with different ends — a GW6-25 walk beside a GW6-38 walk
        — and without it those two share a condition, so a twenty-week result
        and a thirty-three-week result are averaged together as replicates of
        one starting position and then paired against another policy as a
        single number.

        ``policy_seed`` deliberately does *not* include it: two walks from the
        same start should make the same draws for as long as they overlap, so
        the shorter is a prefix of the longer rather than an unrelated sample.
        """
        return f"{self.season}|{self.start_gameweek}|{self.end_gameweek}|{self.squad_id}"

    @property
    def run_id(self) -> str:
        import hashlib

        payload = json.dumps(
            {
                "experiment_id": self.experiment_id,
                "season": self.season,
                "start_gameweek": self.start_gameweek,
                "end_gameweek": self.end_gameweek,
                "squad_id": self.squad_id,
                "policy": self.policy,
                "replicate": self.replicate,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "season": self.season,
            "start_gameweek": self.start_gameweek,
            "end_gameweek": self.end_gameweek,
            "squad_id": self.squad_id,
            "policy": self.policy,
            "replicate": self.replicate,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunKey:
        return cls(
            experiment_id=str(payload["experiment_id"]),
            season=str(payload["season"]),
            start_gameweek=int(payload["start_gameweek"]),
            end_gameweek=int(payload["end_gameweek"]),
            squad_id=str(payload["squad_id"]),
            policy=str(payload["policy"]),
            replicate=int(payload.get("replicate", 0)),
        )


@dataclass(frozen=True)
class RunRecord:
    """A completed unit, with everything needed to compare or reproduce it."""

    key: RunKey
    metrics: RunMetrics
    schema_version: str = EVALUATION_SCHEMA_VERSION
    seeds: tuple[tuple[str, int], ...] = ()
    prediction_inputs_sha256: tuple[tuple[int, str], ...] = ()
    """(gameweek, content hash) per step, so "every policy saw byte-identical
    inputs" is checkable across processes rather than asserted in a comment."""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    code_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "schema_version": self.schema_version,
            "metrics": self.metrics.to_dict(),
            "seeds": [list(pair) for pair in self.seeds],
            "prediction_inputs_sha256": [list(pair) for pair in self.prediction_inputs_sha256],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "code_version": self.code_version,
        }


def plan_units(config: ExperimentConfig, squad_ids: Sequence[str]) -> tuple[RunKey, ...]:
    """Every unit this experiment owes, in a deterministic order.

    Pure and total: no disk, no clock, no randomness. That is what makes
    progress reportable and a resume decidable.

    Replicates expand for stochastic policies only. A deterministic policy
    discards its generator, so a second replicate would be a byte-identical
    walk — and running it would not merely waste time, it would put five copies
    of one number into a sample and misrepresent the spread.
    """
    units: list[RunKey] = []
    for window in sorted(config.windows, key=lambda w: w.season):
        for start in sorted(window.start_gameweeks):
            for squad_id in squad_ids:
                for policy in config.policies:
                    replicates = config.replicates if policy.stochastic else 1
                    for replicate in range(replicates):
                        units.append(
                            RunKey(
                                experiment_id=config.experiment_id,
                                season=window.season,
                                start_gameweek=start,
                                end_gameweek=window.end_gameweek,
                                squad_id=squad_id,
                                policy=policy.name,
                                replicate=replicate,
                            )
                        )
    return tuple(units)


def policy_seed(config: ExperimentConfig, key: RunKey, gameweek: int | None = None) -> int:
    """The seed a stochastic policy draws with, derived by name.

    Under ``RngScope.GAMEWEEK`` the generator is fresh per step, which decouples
    gameweeks and is what allows a walk to be checkpointed. ``RngScope.RUN``
    reproduces the legacy coupling, where a draw in gameweek 20 depended on how
    many candidates existed in every week before it.
    """
    parts: list[str | int] = [
        "policy",
        key.policy,
        key.season,
        key.start_gameweek,
        key.squad_id,
        key.replicate,
    ]
    if config.rng_scope is RngScope.GAMEWEEK and gameweek is not None:
        parts.append(gameweek)
    return derive_seed(config.root_seed, *parts)


def run_directory(root: Path, config: ExperimentConfig) -> Path:
    """Where this experiment's results live.

    Named for the config hash and carrying no timestamp, so the same settings
    always resolve to the same directory and different settings never share
    one. ``report.py::write_report`` timestamps its directories, which is
    precisely what makes those un-resumable.
    """
    return root / "experiments" / config.experiment_id


def write_run(directory: Path, record: RunRecord) -> Path:
    """Persist one unit atomically.

    Written to a temporary name and renamed, so a process killed mid-write
    leaves a ``.tmp`` the glob ignores rather than a half-parsed run that a
    resume would count as done.
    """
    runs = directory / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{record.key.run_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    tmp.replace(path)
    return path


def read_run(path: Path) -> dict[str, Any] | None:
    """Read one run file, or ``None`` when it is unusable.

    Unusable means missing, malformed, or written under a different schema.
    All three are re-executed rather than trusted, because a run that cannot be
    read cannot be verified either.
    """
    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        return None
    return payload


def completed_units(directory: Path, units: Sequence[RunKey]) -> tuple[set[str], list[RunKey]]:
    """Split the plan into what is already done and what remains.

    A unit counts as done only when its file parses, its schema matches, *and*
    its recorded key equals the unit asked for. Anything else is redone —
    there is deliberately no partial-run state to reconcile, because
    reconciling one is where this class of framework quietly loses runs.
    """
    done: set[str] = set()
    for unit in units:
        payload = read_run(directory / "runs" / f"{unit.run_id}.json")
        if payload is None:
            continue
        if RunKey.from_dict(payload["key"]) == unit:
            done.add(unit.run_id)
    return done, [u for u in units if u.run_id not in done]


def write_manifest(
    directory: Path,
    config: ExperimentConfig,
    *,
    planned: int,
    completed: int,
    limitations: Sequence[str],
    started_at: datetime,
    resumed_across_commits: Sequence[str] = (),
) -> Path:
    """Record what ran, under what conditions, and what it cannot support."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash(),
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "started_at": started_at.isoformat(),
        "completed_at": utc_now().isoformat(),
        "code_version": config.code_version,
        "git_dirty": config.git_dirty,
        "resumed_across_commits": list(resumed_across_commits),
        "runs_planned": planned,
        "runs_completed": completed,
        "reproducible": config.reproducible,
        "limitations": list(limitations),
    }
    path = directory / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)
    return path


RunOne = Callable[[RunKey], RunRecord]
"""What the runner needs from its caller: execute one unit, return its record."""


def run_experiment(
    config: ExperimentConfig,
    *,
    root: Path,
    squad_ids: Sequence[str],
    run_one: RunOne,
    limitations: Sequence[str] = (),
    overwrite: bool = False,
    on_progress: Callable[[int, int, RunKey], None] | None = None,
) -> tuple[Path, list[RunRecord]]:
    """Execute or resume an experiment, returning its directory and records.

    Args:
        run_one: Executes a single unit. Injected rather than imported so the
            runner can be tested without a model, a parquet file or a solver.
        overwrite: Discard completed runs and start again. Off by default,
            because the expensive mistake is destroying six hours of work, not
            resuming into it.
    """
    started = utc_now()
    directory = run_directory(root, config)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(config.canonical_json())

    units = plan_units(config, squad_ids)
    if overwrite:
        for stale in (directory / "runs").glob("*.json"):
            stale.unlink()
    _, remaining = completed_units(directory, units)

    records: list[RunRecord] = []
    for index, unit in enumerate(remaining, start=1):
        record = run_one(unit)
        write_run(directory, record)
        records.append(record)
        if on_progress is not None:
            on_progress(index, len(remaining), unit)

    done, _ = completed_units(directory, units)
    write_manifest(
        directory,
        config,
        planned=len(units),
        completed=len(done),
        limitations=limitations,
        started_at=started,
    )
    return directory, records
