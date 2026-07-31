"""Deciding whether a saved model may be used, before it is opened.

**The check order is the guarantee.** Every blocking condition is evaluated
against the sidecar manifest — plain JSON — so a refusal happens *before*
``pickle.load`` is ever called. That is what makes "a raw ``KeyError`` must
never be the first failure" a structural property rather than a matter of
which line happens to run first.

**Satisfiability, not equality.** ``trained.py::_matrix`` selects the
artifact's own column tuple out of the frame it is given, so extra columns in
the frame are harmless. Measured on the eight artifacts on disk, seven need
only columns the active build supplies and one needs five it does not. A gate
on ordered equality would refuse six working models the day it shipped.

**The dangerous case has no natural exception.** A missing column raises. A
mismatched arity raises, eventually, from inside scikit-learn. A *reordered*
tuple of the right length raises nothing: it feeds every column to the wrong
tree splits and returns plausible, wrong numbers. It is checked explicitly
because nothing else will ever notice it.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xg_alonso.contracts.artifacts import (
    ARTIFACT_MANIFEST_VERSION,
    ArtifactCompatibility,
    ArtifactIncompatibility,
    ArtifactManifest,
    ArtifactStatus,
    CompatibilityReason,
    FeatureSchemaDiff,
    Severity,
)
from xg_alonso.contracts.provenance import utc_now

__all__ = [
    "ActiveSchema",
    "ArtifactCompatibilityError",
    "check_compatibility",
    "manifest_path_for",
    "read_manifest",
    "write_manifest",
]


class ArtifactCompatibilityError(ValueError):
    """A saved model cannot be used with the active catalogue and rules.

    Subclasses ``ValueError`` because ``load_models`` already documents that
    for a stale artifact and callers already handle it. Carries the structured
    verdict so a caller can render more than the message.
    """

    def __init__(self, compatibility: ArtifactCompatibility) -> None:
        self.compatibility = compatibility
        super().__init__(compatibility.explain())


@dataclass(frozen=True)
class ActiveSchema:
    """What the current build can supply, and under what provenance."""

    feature_names: tuple[str, ...]
    catalogue_version: str = ""
    catalogue_hash: str = ""
    rules_snapshot_hash: str = ""
    extension_features: tuple[str, ...] = ()
    """Discovered features available now, beyond the declarative catalogue."""

    @classmethod
    def from_catalogue(
        cls,
        *,
        rules_snapshot_hash: str = "",
        extension_features: Sequence[str] = (),
    ) -> ActiveSchema:
        """Read the active schema from ``features``, one layer up from contracts."""
        from xg_alonso.features.catalogue import CATALOGUE_VERSION
        from xg_alonso.features.schema import catalogue_hash, model_feature_names

        return cls(
            feature_names=model_feature_names(),
            catalogue_version=CATALOGUE_VERSION,
            catalogue_hash=catalogue_hash(extension=extension_features),
            rules_snapshot_hash=rules_snapshot_hash,
            extension_features=tuple(extension_features),
        )

    @property
    def available(self) -> frozenset[str]:
        return frozenset(self.feature_names) | frozenset(self.extension_features)


def manifest_path_for(path: Path) -> Path:
    """The sidecar beside an artifact."""
    return path.with_suffix(path.suffix + ".manifest.json")


def read_manifest(path: Path) -> ArtifactManifest | None:
    """Read an artifact's sidecar. Never unpickles anything.

    Reading provenance must not execute code — unpickling an unknown file to
    find out whether it is safe to unpickle is not a strategy.

    Returns ``None`` when absent. An *unreadable* sidecar is a different
    situation and raises, because a manifest that cannot be parsed is a
    stronger signal than one that was never written.
    """
    sidecar = manifest_path_for(path)
    if not sidecar.exists():
        return None

    raw = json.loads(sidecar.read_text())
    # Read the version out of the raw payload first. `extra="forbid"` means a
    # newer manifest would otherwise surface as an unreadable pydantic dump
    # rather than as "this file is from a later version".
    version = raw.get("artifact_version")
    if version != ARTIFACT_MANIFEST_VERSION:
        raise ArtifactCompatibilityError(
            ArtifactCompatibility(
                artifact_path=path,
                status=ArtifactStatus.CORRUPT,
                checked_at=utc_now(),
                findings=(
                    ArtifactIncompatibility(
                        reason=CompatibilityReason.ARTIFACT_VERSION_UNSUPPORTED,
                        severity=Severity.BLOCKING,
                        detail=(
                            f"manifest is {version!r}; this code reads "
                            f"{ARTIFACT_MANIFEST_VERSION!r}. Retrain with `xg train`."
                        ),
                    ),
                ),
            )
        )
    return ArtifactManifest.model_validate(raw)


def write_manifest(path: Path, manifest: ArtifactManifest) -> Path:
    """Write the sidecar atomically, beside its artifact."""
    sidecar = manifest_path_for(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar.with_suffix(".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2))
    tmp.replace(sidecar)
    return sidecar


def _schema_diff(needed: Sequence[str], active: ActiveSchema) -> FeatureSchemaDiff:
    """Compare what an artifact needs against what the build can supply."""
    available = active.available
    missing = tuple(c for c in needed if c not in available)
    unexpected = tuple(c for c in active.feature_names if c not in set(needed))

    # Relative order of the shared names. Selection is by name so this cannot
    # break inference, but it means the catalogue definition moved under the
    # artifact, which is worth saying.
    shared = [c for c in active.feature_names if c in set(needed)]
    artifact_order = [c for c in needed if c in available]
    reordered = tuple(a for a, b in zip(shared, artifact_order, strict=False) if a != b)

    return FeatureSchemaDiff(
        expected_order=tuple(needed),
        active_order=tuple(active.feature_names),
        missing=missing,
        unexpected=unexpected,
        reordered=reordered,
    )


def check_compatibility(
    manifest: ArtifactManifest | None,
    *,
    active: ActiveSchema,
    artifact_path: Path,
    artifact_feature_columns: Sequence[str] | None = None,
    estimator_arities: Mapping[str, int] | None = None,
    payload_sha256: str | None = None,
) -> ArtifactCompatibility:
    """Decide whether this artifact may be used, and say exactly why not.

    Args:
        artifact_feature_columns: The fitted columns, when the payload is open.
            Supplying them enables the reordered-schema check, which is the one
            failure with no natural exception.
        estimator_arities: ``label -> n_features_in_``, when available.
    """
    findings: list[ArtifactIncompatibility] = []
    needed = tuple(manifest.feature_names) if manifest else tuple(artifact_feature_columns or ())
    diff = _schema_diff(needed, active) if needed else None

    if manifest is None:
        findings.append(
            ArtifactIncompatibility(
                reason=CompatibilityReason.MANIFEST_ABSENT,
                severity=Severity.WARNING,
                detail=(
                    "no manifest sidecar; schema checks ran but provenance could not "
                    "be verified. Re-save with `xg models backfill-manifest`."
                ),
            )
        )

    # 1. Arity. Pre-empts scikit-learn's "X has N features, expecting M",
    #    which surfaces from inside a prediction loop far from the cause.
    if manifest is not None and artifact_feature_columns is not None:
        if len(manifest.feature_names) != len(artifact_feature_columns):
            findings.append(
                ArtifactIncompatibility(
                    reason=CompatibilityReason.FEATURE_COUNT_MISMATCH,
                    severity=Severity.BLOCKING,
                    detail=(
                        f"manifest declares {len(manifest.feature_names)} features but the "
                        f"artifact was fitted on {len(artifact_feature_columns)}"
                    ),
                )
            )
        # 2. Ordered identity. THE dangerous case: right length, wrong order,
        #    no exception anywhere — every column reaches the wrong splits.
        elif tuple(manifest.feature_names) != tuple(artifact_feature_columns):
            findings.append(
                ArtifactIncompatibility(
                    reason=CompatibilityReason.REORDERED_FEATURES,
                    severity=Severity.BLOCKING,
                    detail=(
                        "the manifest's feature order does not match the fitted order. "
                        "Selection is positional inside the estimator, so this would "
                        "return plausible, wrong numbers rather than raising."
                    ),
                )
            )

    for label, arity in (estimator_arities or {}).items():
        if needed and arity != len(needed):
            findings.append(
                ArtifactIncompatibility(
                    reason=CompatibilityReason.ESTIMATOR_ARITY_MISMATCH,
                    severity=Severity.BLOCKING,
                    detail=f"estimator {label!r} expects {arity} features, schema declares {len(needed)}",
                )
            )

    # 3. Satisfiability — the question that actually decides usability.
    if diff is not None and diff.missing:
        findings.append(
            ArtifactIncompatibility(
                reason=CompatibilityReason.MISSING_FEATURES,
                severity=Severity.BLOCKING,
                detail=(
                    f"{len(diff.missing)} feature(s) the active build cannot supply: "
                    f"{', '.join(diff.missing[:8])}"
                    + (" ..." if len(diff.missing) > 8 else "")
                    + ". Retrain with `xg train`, or supply them as extension features."
                ),
                features=diff.missing[:8],
            )
        )

    # 4. Staleness. Non-blocking by measurement: six of the eight artifacts on
    #    disk are proper subsets of the active schema and predict correctly.
    if diff is not None and diff.unexpected:
        findings.append(
            ArtifactIncompatibility(
                reason=CompatibilityReason.UNEXPECTED_FEATURES,
                severity=Severity.WARNING,
                detail=(
                    f"the active build supplies {len(diff.unexpected)} feature(s) this "
                    "artifact never saw; it is stale, not broken"
                ),
                features=diff.unexpected[:8],
            )
        )

    if manifest is not None:
        satisfiable = diff is None or diff.satisfiable
        if (
            active.catalogue_version
            and manifest.feature_catalogue_version != active.catalogue_version
        ):
            findings.append(
                ArtifactIncompatibility(
                    reason=CompatibilityReason.CATALOGUE_VERSION_MISMATCH,
                    severity=Severity.WARNING if satisfiable else Severity.BLOCKING,
                    detail=(
                        f"fitted against {manifest.feature_catalogue_version!r}, active is "
                        f"{active.catalogue_version!r}"
                    ),
                )
            )
        if (
            active.catalogue_hash
            and manifest.feature_catalogue_hash
            and manifest.feature_catalogue_hash != active.catalogue_hash
        ):
            findings.append(
                ArtifactIncompatibility(
                    reason=CompatibilityReason.CATALOGUE_HASH_MISMATCH,
                    severity=Severity.WARNING if satisfiable else Severity.BLOCKING,
                    detail="the catalogue definition changed since this model was fitted",
                )
            )
        # 5. Rules. Blocking, and for a different reason from the catalogue:
        #    components are priced with the *active* rules, so a model fitted
        #    when a goalkeeper goal was worth 6 and priced when it is worth 10
        #    produces a number nobody can reconcile.
        if (
            active.rules_snapshot_hash
            and manifest.rules_snapshot_hash
            and manifest.rules_snapshot_hash != active.rules_snapshot_hash
        ):
            findings.append(
                ArtifactIncompatibility(
                    reason=CompatibilityReason.RULES_HASH_MISMATCH,
                    severity=Severity.BLOCKING,
                    detail=(
                        "the scoring rules changed since this model was fitted. Its "
                        "components would be priced under rules it never saw."
                    ),
                )
            )
        if payload_sha256 and manifest.payload_sha256 and manifest.payload_sha256 != payload_sha256:
            findings.append(
                ArtifactIncompatibility(
                    reason=CompatibilityReason.PAYLOAD_DIGEST_MISMATCH,
                    severity=Severity.BLOCKING,
                    detail=(
                        "the artifact's bytes do not match its manifest; one of the two "
                        "was modified or half-written"
                    ),
                )
            )

    blocking = [f for f in findings if f.severity is Severity.BLOCKING]
    if blocking:
        status = (
            ArtifactStatus.MIGRATABLE
            if any(f.reason is CompatibilityReason.MISSING_FEATURES for f in blocking)
            else ArtifactStatus.CORRUPT
        )
    elif manifest is None:
        status = ArtifactStatus.UNVERIFIED
    else:
        status = ArtifactStatus.COMPATIBLE

    return ArtifactCompatibility(
        artifact_path=artifact_path,
        status=status,
        checked_at=utc_now(),
        manifest=manifest,
        schema_diff=diff,
        findings=tuple(findings),
    )


def payload_digest(path: Path) -> str:
    """SHA-256 of an artifact's bytes. The true identity of the file.

    ``ComponentModels.fingerprint()`` hashes name, version, sorted model keys,
    columns and row count — so two models fitted on *different data* with the
    same shape collide. This does not.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_quietly(path: Path) -> tuple[Any, list[str]]:
    """Unpickle, capturing version warnings rather than letting them explode.

    ``pyproject.toml`` sets ``filterwarnings = ["error"]``, so the day a
    ``uv sync`` moves scikit-learn off the pinned version, every artifact load
    in the suite becomes a hard error with no diagnosis attached. Captured here
    and reported as drift instead.
    """
    import pickle

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
    return loaded, [str(w.message) for w in caught]
