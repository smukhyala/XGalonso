"""Ingest official FPL data into immutable bronze snapshots.

The pipeline does the I/O and hands already-parsed objects to the domain, which
keeps rule interpretation pure and testable without a network or a database.

Two guards run here rather than downstream, because both fail silently if left
until later:

1. **Rule drift.** Live scoring and squad constants are compared against the
   pinned snapshot on every ingest.
2. **Preseason stat carry-over.** Before the season opens, ``bootstrap-static``
   still carries last season's totals on some fields while zeroing others.
   Treating those as current-season data poisons every rolling feature, and the
   symptom appears weeks later as inexplicable model behaviour.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from xg_alonso.contracts.identifiers import Season
from xg_alonso.contracts.provenance import RunManifest, utc_now
from xg_alonso.contracts.storage import BronzeSnapshotStore, SnapshotRef
from xg_alonso.domain.drift import RuleChange, diff_scoring_rules, diff_squad_rules
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringRules
from xg_alonso.pipelines.ingestion.fpl_client import FplApiClient

__all__ = [
    "SOURCE_BOOTSTRAP",
    "SOURCE_FIXTURES",
    "IngestResult",
    "PreseasonWarning",
    "detect_preseason_hazards",
    "git_manifest",
    "ingest_bootstrap",
    "load_rules_from_snapshot",
]

SOURCE_BOOTSTRAP = "fpl.bootstrap_static"
SOURCE_FIXTURES = "fpl.fixtures"


@dataclass(frozen=True)
class PreseasonWarning:
    """A field that cannot be trusted in the current payload."""

    field_name: str
    detail: str


@dataclass
class IngestResult:
    """What one ingest run produced."""

    manifest: RunManifest
    snapshots: list[SnapshotRef] = field(default_factory=list)
    rule_changes: list[RuleChange] = field(default_factory=list)
    preseason_warnings: list[PreseasonWarning] = field(default_factory=list)

    @property
    def rules_drifted(self) -> bool:
        return bool(self.rule_changes)


def git_manifest(pipeline: str, *, run_id: str, config_hash: str = "default") -> RunManifest:
    """Build a run manifest, recording the commit and whether the tree was dirty."""

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return ""

    commit = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain"))

    return RunManifest(
        run_id=run_id,
        pipeline=pipeline,
        started_at=utc_now(),
        git_commit=commit,
        git_dirty=dirty,
        config_hash=config_hash,
    )


def load_rules_from_snapshot(
    payload: dict[str, Any],
    *,
    season: Season,
    source_sha256: str,
    fetched_at: datetime,
) -> tuple[ScoringRules, SquadRules]:
    """Parse both rule sets out of one bootstrap payload."""
    scoring = ScoringRules.from_bootstrap(
        payload, version=str(season), source_sha256=source_sha256, fetched_at=fetched_at
    )
    squad = SquadRules.from_bootstrap(payload, version=str(season), source_sha256=source_sha256)
    return scoring, squad


def detect_preseason_hazards(payload: dict[str, Any]) -> list[PreseasonWarning]:
    """Flag fields that are unusable in the current payload.

    Verified live on 2026-07-27: every team's ``strength_attack_*`` and
    ``strength_defence_*`` reads 0 and ``strength`` is null, while
    ``strength_overall_*`` sits on a 1-5 scale that becomes roughly 1000-1400
    once the season starts. Any fixture-difficulty term built on those fields is
    therefore either identically 1.0 or discontinuous across the season boundary
    — which is why team strength is derived from expected goals instead.
    """
    warnings: list[PreseasonWarning] = []
    teams = payload.get("teams") or []
    if not teams:
        return warnings

    for name in (
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ):
        values = [t.get(name) for t in teams]
        if all(v in (0, None) for v in values):
            warnings.append(
                PreseasonWarning(
                    field_name=name,
                    detail=(
                        f"{name} is 0 or null for all {len(teams)} teams; unusable as a "
                        "fixture-difficulty input. Derive team strength from expected goals."
                    ),
                )
            )

    overall = [t.get("strength_overall_home") for t in teams if t.get("strength_overall_home")]
    if overall and max(overall) <= 5:
        warnings.append(
            PreseasonWarning(
                field_name="strength_overall_home",
                detail=(
                    f"strength_overall_home maxes at {max(overall)}, the preseason 1-5 scale. "
                    "In-season it runs to roughly 1400, so the two are not comparable."
                ),
            )
        )

    elements = payload.get("elements") or []
    if elements:
        unknown = sum(1 for e in elements if e.get("chance_of_playing_next_round") is None)
        if unknown > len(elements) * 0.5:
            warnings.append(
                PreseasonWarning(
                    field_name="chance_of_playing_next_round",
                    detail=(
                        f"null for {unknown}/{len(elements)} players; availability signals "
                        "are cold until team news begins."
                    ),
                )
            )

    return warnings


def ingest_bootstrap(
    *,
    client: FplApiClient,
    bronze: BronzeSnapshotStore,
    season: Season,
    run_id: str,
    pinned_scoring: ScoringRules | None = None,
    pinned_squad: SquadRules | None = None,
) -> IngestResult:
    """Fetch ``bootstrap-static`` and ``fixtures`` into bronze, checking for drift.

    Args:
        client: Live FPL API client.
        bronze: Immutable snapshot store.
        season: Season these rules apply to.
        run_id: Identifier tying every snapshot to this run.
        pinned_scoring: Previously pinned scoring rules to compare against.
        pinned_squad: Previously pinned squad rules to compare against.

    Returns:
        The snapshots written, any rule drift, and any preseason hazards.
    """
    manifest = git_manifest("ingest.bootstrap", run_id=run_id)
    result = IngestResult(manifest=manifest)

    response = client.bootstrap_static()
    ref = bronze.write(
        source=SOURCE_BOOTSTRAP,
        payload=response.payload,
        timestamps=response.timestamps,
        run_id=run_id,
        http_status=response.status_code,
    )
    result.snapshots.append(ref)

    payload: dict[str, Any] = json.loads(response.payload)
    live_scoring, live_squad = load_rules_from_snapshot(
        payload,
        season=season,
        source_sha256=ref.content_sha256,
        fetched_at=ref.timestamps.available_time,
    )

    if pinned_scoring is not None:
        result.rule_changes.extend(diff_scoring_rules(pinned_scoring, live_scoring))
    if pinned_squad is not None:
        result.rule_changes.extend(diff_squad_rules(pinned_squad, live_squad))

    result.preseason_warnings.extend(detect_preseason_hazards(payload))

    fixtures = client.fixtures()
    result.snapshots.append(
        bronze.write(
            source=SOURCE_FIXTURES,
            payload=fixtures.payload,
            timestamps=fixtures.timestamps,
            run_id=run_id,
            http_status=fixtures.status_code,
        )
    )

    return result


def read_snapshot_payload(bronze: BronzeSnapshotStore, ref: SnapshotRef) -> dict[str, Any]:
    """Read and parse a stored JSON snapshot."""
    data: dict[str, Any] = json.loads(bronze.read(ref))
    return data


def load_local_payload(path: Path) -> dict[str, Any]:
    """Load a payload from disk — used by tests and by ``--squad-file``."""
    data: dict[str, Any] = json.loads(path.read_text())
    return data
