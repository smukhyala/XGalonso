"""Turn trained component models into predictions, and persist them.

The mapping from model outputs to :class:`ComponentExpectations` is where the
trained path rejoins the closed-form one: both produce components, and both
hand them to :func:`~xg_alonso.domain.assemble_points`. Nothing here knows what
a goal is worth, so swapping estimators never touches scoring.

**Persistence carries provenance.** A saved model records the seasons and
gameweeks it was fitted on, so a later run can check it was not trained on the
period it is about to be evaluated over — the mistake that makes a backtest
meaningless while looking entirely normal.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

import polars as pl

from xg_alonso.contracts.artifacts import (
    ArtifactCompatibility,
    ArtifactIncompatibility,
    ArtifactManifest,
    ArtifactStatus,
    CompatibilityReason,
    Severity,
)
from xg_alonso.contracts.evidence import FeatureEvidence
from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
from xg_alonso.contracts.prediction import (
    ComponentExpectations,
    MinutesPrediction,
    PlayerPrediction,
    Position,
)
from xg_alonso.contracts.provenance import PredictionProvenance, utc_now
from xg_alonso.domain.scoring import ScoringRules, assemble_points
from xg_alonso.prediction.artifacts import (
    ActiveSchema,
    ArtifactCompatibilityError,
    check_compatibility,
    load_quietly,
    manifest_path_for,
    payload_digest,
    read_manifest,
    write_manifest,
)
from xg_alonso.prediction.evidence import build_feature_evidence
from xg_alonso.prediction.trained import (
    TRAINED_MODEL_NAME,
    TRAINED_MODEL_VERSION,
    ComponentModels,
)

__all__ = [
    "SavedModel",
    "load_models",
    "predict_with_models",
    "save_models",
]

_FULL_MATCH = 90.0


@dataclass(frozen=True)
class SavedModel:
    """A persisted model plus the provenance needed to use it safely."""

    models: ComponentModels
    trained_seasons: tuple[str, ...]
    trained_gameweeks: tuple[int, ...]
    saved_at: datetime
    manifest: ArtifactManifest | None = None
    """Provenance, also written beside the file as a sidecar.

    Appended with a default so a pickle written before this field existed
    restores into the new class and simply reports ``None`` rather than raising
    an AttributeError from inside a prediction loop.
    """
    compatibility: ArtifactCompatibility | None = None
    """The verdict from *this* load. Describes the check, not the artifact, so
    it is excluded from persistence."""

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["compatibility"] = None
        return state

    def overlaps(self, season: str, gameweeks: tuple[int, ...]) -> bool:
        """Whether this model was fitted on any of the given gameweeks.

        A backtest over a period the model trained on measures memorisation,
        not skill, and the result looks perfectly reasonable — so the check is
        mechanical rather than left to whoever runs it.
        """
        if season not in self.trained_seasons:
            return False
        return bool(set(gameweeks) & set(self.trained_gameweeks))


def save_models(saved: SavedModel, path: Path, *, manifest: ArtifactManifest | None = None) -> Path:
    """Persist a fitted model, and write its manifest beside it.

    The manifest is written **twice**: as a sidecar and embedded in the pickle.
    The sidecar is the point — reading provenance must not execute code, and
    unpickling an unknown file to find out whether it is safe to unpickle is
    not a strategy. The embedded copy exists because a sidecar can be lost when
    somebody copies a ``.pkl`` around, and an artifact that cannot describe
    itself is exactly the failure being fixed.

    They are tied together by ``payload_sha256``, so a disagreement means one
    of the two was modified or half-written.

    Returns:
        The sidecar path, or the artifact path when no manifest was supplied.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    embedded = replace(saved, manifest=manifest, compatibility=None)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(embedded, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if manifest is None:
        tmp.replace(path)
        return path

    # Hash the written bytes, then record that hash in the sidecar. The
    # embedded copy cannot contain its own digest, so the sidecar is the one
    # that carries it and the one the gate compares against.
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.replace(path)
    write_manifest(
        path,
        replace_manifest(manifest, payload_sha256=digest, payload_bytes=path.stat().st_size),
    )
    return manifest_path_for(path)


def replace_manifest(manifest: ArtifactManifest, **updates: object) -> ArtifactManifest:
    """A manifest with fields replaced. Frozen models need a copy, not a set."""
    return manifest.model_copy(update=updates)


#: Attributes a usable artifact must carry. Checked on load so an artifact
#: saved by older code fails here, with a message that says what to do, rather
#: than raising an AttributeError from inside a prediction loop.
_REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "models",
    "feature_columns",
    "reports",
    "label_means",
)


def load_models(
    path: Path,
    *,
    active: ActiveSchema | None = None,
    require_manifest: bool = False,
    on_incompatible: Literal["raise", "warn", "ignore"] = "raise",
) -> SavedModel:
    """Load a persisted model, refusing anything unusable.

    The compatibility gate runs against the sidecar manifest — plain JSON —
    **before** ``pickle.load`` is called, so a schema mismatch can never first
    surface as a ``KeyError`` from inside a prediction loop, or as a
    scikit-learn arity error from three frames deeper.

    Args:
        active: What the current build can supply. Without it only the
            structural checks run, which is the behaviour every existing caller
            had.
        require_manifest: Treat an absent sidecar as blocking. Off by default:
            the artifacts already on disk predate manifests and still work.
        on_incompatible: ``raise`` refuses, ``warn`` reports and continues,
            ``ignore`` says nothing. Refusing is the default because a silently
            wrong prediction is worse than a loud failure.

    Raises:
        TypeError: if the file does not hold a ``SavedModel``. Unpickling
            arbitrary objects and hoping is not a loading strategy.
        ValueError: if the artifact predates a field the current code needs.
        ArtifactCompatibilityError: if it cannot be used with the active
            catalogue and rules.
    """
    manifest = read_manifest(path)

    if manifest is None and require_manifest:
        _refuse(
            path,
            CompatibilityReason.MANIFEST_ABSENT,
            f"{path.name} has no manifest. Re-save it with "
            "`xg models backfill-manifest`, or retrain with `xg train`.",
        )

    # Gate on the manifest alone, before any code from the file is executed.
    if manifest is not None and active is not None:
        verdict = check_compatibility(
            manifest,
            active=active,
            artifact_path=path,
            payload_sha256=payload_digest(path),
        )
        if not verdict.compatible and on_incompatible == "raise":
            raise ArtifactCompatibilityError(verdict)

    loaded, version_warnings = load_quietly(path)
    if not isinstance(loaded, SavedModel):
        raise TypeError(f"{path} does not contain a SavedModel (got {type(loaded).__name__})")

    missing = [a for a in _REQUIRED_ATTRIBUTES if not hasattr(loaded.models, a)]
    if missing:
        raise ValueError(
            f"{path} was saved by an older version and is missing {missing}. "
            "Retrain with `xg train` rather than using a stale artifact."
        )

    if active is not None:
        verdict = check_compatibility(
            manifest,
            active=active,
            artifact_path=path,
            artifact_feature_columns=loaded.models.feature_columns,
            payload_sha256=payload_digest(path) if manifest else None,
        )
        if not verdict.compatible and on_incompatible == "raise":
            raise ArtifactCompatibilityError(verdict)
        if not verdict.compatible and on_incompatible == "warn":
            print(verdict.explain())  # noqa: T201 - the caller asked to be told
        loaded = replace(loaded, compatibility=verdict)

    if version_warnings:
        # Captured rather than raised: `filterwarnings = ["error"]` would
        # otherwise turn a scikit-learn bump into an undiagnosable suite-wide
        # failure at every artifact load.
        print(  # noqa: T201
            f"  note: {path.name} loaded with {len(version_warnings)} version warning(s)"
        )

    return loaded


def _refuse(path: Path, reason: CompatibilityReason, detail: str) -> None:
    raise ArtifactCompatibilityError(
        ArtifactCompatibility(
            artifact_path=path,
            status=ArtifactStatus.UNVERIFIED,
            checked_at=utc_now(),
            findings=(
                ArtifactIncompatibility(reason=reason, severity=Severity.BLOCKING, detail=detail),
            ),
        )
    )


#: Days of inactivity beyond which the last match stops describing the next one.
#: Matches the staleness threshold the recency features use.
_STALE_DAYS: Final[float] = 42.0


def _established_minutes_floor(
    minutes_when_played: float | None, appearance_rate: float | None
) -> float | None:
    """What a player's own record says he plays, ignoring his last match.

    ``appearance_rate x minutes_when_played`` is the expectation implied by how
    often he features and how long he lasts when he does — the same two facts a
    fixture-based rolling mean multiplies together and then loses.
    """
    if minutes_when_played is None or appearance_rate is None:
        return None
    if minutes_when_played <= 0.0 or appearance_rate <= 0.0:
        return None
    return float(appearance_rate) * float(minutes_when_played)


def _minutes_from(expected_minutes: float, p_start: float) -> MinutesPrediction:
    """Assemble a coherent minutes distribution from two model outputs.

    The models predict expected minutes and start probability independently, so
    they can disagree — a player can come back with 80 expected minutes and a
    0.2 start probability. The contract's invariants would reject that, so the
    quantities are reconciled here rather than at validation time.
    """
    mean = max(0.0, min(expected_minutes, _FULL_MATCH))
    start = max(0.0, min(p_start, 1.0))

    # Appearing is implied by the minutes actually expected, and by starting.
    p_appearance = max(start, min(1.0, mean / 70.0) if mean < 70.0 else 1.0)
    p_60 = min(p_appearance, start * 0.9 + max(0.0, (mean - 60.0) / 30.0) * 0.1)
    sd = 30.0 * (1.0 - abs(mean - 45.0) / 45.0) + 6.0

    return MinutesPrediction(
        p_appearance=round(p_appearance, 6),
        p_start=round(min(start, p_appearance), 6),
        expected_minutes=round(mean, 6),
        p_60_plus=round(max(0.0, p_60), 6),
        minutes_sd=round(sd, 6),
    )


def predict_with_models(
    features: pl.DataFrame,
    *,
    models: ComponentModels,
    rules: ScoringRules,
    from_gameweek: GameweekId,
    data_cutoff: datetime,
    predicted_at: datetime,
    run_id: str,
    code_version: str,
    feature_set_version: str,
    horizon_gameweeks: int = 1,
    shrink_by_skill: bool = False,
    with_evidence: bool = True,
) -> list[PlayerPrediction]:
    """Predict components with the trained models and assemble them into points.

    Requires ``player_code`` and ``position`` on the feature frame. Rows whose
    position is unrecognised are skipped rather than guessed.

    Args:
        shrink_by_skill: Blend each component toward its population mean in
            proportion to measured out-of-sample skill, so a weakly-predicted
            component cannot drive a recommendation as hard as a strong one.
        with_evidence: Attach the explanatory feature panel, ranked within
            position across this batch. On by default because a prediction that
            cannot be traced to its inputs cannot be explained — the whole
            reason the recommendation screen could only cite model output was
            that this frame used to be discarded here.
    """
    for required in ("player_code", "position"):
        if required not in features.columns:
            raise KeyError(f"feature frame must carry {required!r}")
    if features.is_empty():
        return []

    predicted = models.predict(features, shrink_by_skill=shrink_by_skill)

    evidence: list[FeatureEvidence] | None = None
    if with_evidence:
        evidence = build_feature_evidence(
            features,
            positions=[str(p) for p in features["position"].to_list()],
        )

    def _optional_column(frame: pl.DataFrame, name: str) -> list[float | None]:
        """A feature column when the frame carries it, all-null when it does not.

        The floor is a refinement, so a feature set without the recency columns
        must still predict rather than fail.
        """
        if name not in frame.columns:
            return [None] * frame.height
        return [
            None if value is None else float(value)
            for value in frame[name].cast(pl.Float64, strict=False).to_list()
        ]

    def column(label: str, default: float = 0.0) -> list[float]:
        values = predicted.get(label)
        if values is None:
            return [default] * features.height
        return [float(v) for v in values]

    minutes = column("label_minutes")
    starts = column("label_starts")

    # --- stale-form floor -------------------------------------------------
    #
    # **Why a rule and not a feature.** Three attempts to fix this with features
    # failed and are recorded because the failures are informative: adding
    # `days_since_last_match` did nothing, removing the single-match window cost
    # skill (minutes +60.6% to +57.8%) and changed no projection, and adding
    # appearance-conditioned minutes did not move it either. The model had the
    # right inputs each time and kept its answer.
    #
    # It kept it because the relationship it learned is correct on average. A
    # player who did not feature in his club's last fixture usually is not
    # starting the next one, and that holds across almost all of the training
    # data. It fails for one subpopulation — an established starter rested in a
    # dead rubber before a long break — and four seasons contain three season
    # openers, which is not enough examples to learn the exception from.
    #
    # So the exception is stated rather than learned, and bounded so it cannot
    # do anything else: only when the recent window is genuinely stale, and only
    # ever raising a projection to what the player's own appearance record
    # already implies. It cannot invent minutes for someone who does not play.
    stale = _optional_column(features, "form_is_stale")
    when_played = _optional_column(features, "minutes_when_played_mean_5")
    appearance_rate = _optional_column(features, "appearance_rate_10")
    days_since = _optional_column(features, "days_since_last_match")
    goals = column("label_goals_scored")
    assists = column("label_assists")
    clean_sheets = column("label_clean_sheets")
    conceded = column("label_goals_conceded")
    saves = column("label_saves")
    bonus = column("label_bonus")
    yellows = column("label_yellow_cards")

    provenance = PredictionProvenance(
        model_name=TRAINED_MODEL_NAME,
        model_version=TRAINED_MODEL_VERSION,
        model_artifact_sha256=models.fingerprint(),
        feature_set_name="catalogue",
        feature_set_version=feature_set_version,
        data_cutoff=data_cutoff,
        predicted_at=predicted_at,
        run_id=run_id,
        code_version=code_version,
    )

    out: list[PlayerPrediction] = []
    for index, row in enumerate(features.iter_rows(named=True)):
        position = row.get("position")
        if not isinstance(position, str) or position not in Position.__members__:
            continue

        expected = minutes[index]
        started = starts[index]

        is_stale = (stale[index] or 0.0) >= 0.5 or (days_since[index] or 0.0) >= _STALE_DAYS
        if is_stale:
            floor = _established_minutes_floor(when_played[index], appearance_rate[index])
            if floor is not None and floor > expected:
                expected = floor
                # A start probability of 0.3 alongside 72 expected minutes is
                # incoherent, so the two are raised together — the contract's
                # own invariants would reject the pair otherwise.
                started = max(started, min(1.0, floor / _FULL_MATCH))

        minute_prediction = _minutes_from(expected, started)
        scale = horizon_gameweeks

        components = ComponentExpectations(
            minutes=minute_prediction,
            goals=max(0.0, goals[index]) * scale,
            assists=max(0.0, assists[index]) * scale,
            # The clean-sheet model predicts the event; FPL pays it only to
            # players who last 60 minutes, so the two are combined here to keep
            # the assembly a plain expectation.
            clean_sheet_probability=min(
                1.0, max(0.0, clean_sheets[index]) * minute_prediction.p_60_plus
            ),
            goals_conceded=max(0.0, conceded[index]) * scale,
            saves=max(0.0, saves[index]) * scale,
            yellow_cards=max(0.0, yellows[index]) * scale,
            red_cards=0.0,
            own_goals=0.0,
            penalties_saved=0.0,
            penalties_missed=0.0,
            # Not modelled: the statistic exists for one season only, which is
            # not enough to fit. Left visibly at zero rather than guessed.
            defensive_contribution_probability=0.0,
            bonus=max(0.0, bonus[index]) * scale,
        )

        breakdown = assemble_points(components, Position(position), rules)
        sd = max(
            0.5,
            abs(breakdown.total) * (minute_prediction.minutes_sd / _FULL_MATCH) + 0.8,
        )

        out.append(
            PlayerPrediction(
                player_code=PlayerCode(int(row["player_code"])),
                position=Position(position),
                from_gameweek=from_gameweek,
                horizon_gameweeks=horizon_gameweeks,
                components=components,
                breakdown=breakdown,
                expected_points=breakdown.total,
                expected_points_sd=round(sd, 6),
                scoring_rules_version=rules.version,
                provenance=provenance,
                feature_evidence=None if evidence is None else evidence[index],
            )
        )
    return out


def model_summary(saved: SavedModel) -> dict[str, Any]:
    """A compact description of a saved model, for reporting."""
    return {
        "name": TRAINED_MODEL_NAME,
        "version": TRAINED_MODEL_VERSION,
        "fingerprint": saved.models.fingerprint()[:12],
        "rows": saved.models.trained_on_rows,
        "folds": len(saved.models.folds),
        "seasons": list(saved.trained_seasons),
        "labels": sorted(saved.models.models),
        "skill": saved.models.skill_by_label(),
    }
