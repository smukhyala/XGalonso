"""The checks that make an evaluation's result mean what it says.

Every one of these guards a failure that produces a *better-looking* number, so
none of them will ever be caught by a result that seems wrong. A model trained
on the season it is evaluated over does not crash; it wins. A calibration
fitted on the test period does not crash; it improves the error. That
asymmetry is the reason these run before anything else does, and the reason
each ships with a negative control asserting it actually fires.

They return the full record even when everything passes. "We checked and it was
fine" is only auditable if the check itself is written down, so
``assertions.json`` carries every outcome and not merely the failures.

Eight checks live here. A ninth — that a prediction's ``data_cutoff`` really is
the gameweek's deadline and precedes its first kickoff — needs the built feature
frames and therefore lives with the prediction cache, not in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from xg_alonso.contracts.evaluation import ExperimentConfig
from xg_alonso.contracts.folds import WalkForwardFold
from xg_alonso.prediction.calibration import CALIBRATION_MEASURED_ON
from xg_alonso.prediction.inference import SavedModel

__all__ = ["FreezeCheck", "FreezeViolation", "assert_frozen"]


class FreezeViolation(AssertionError):  # noqa: N818 - reads as an assertion, not an error
    """Raised when an evaluation is not frozen in a way that makes it honest."""


@dataclass(frozen=True)
class FreezeCheck:
    """One check, its outcome, and enough detail to act on a failure."""

    name: str
    passed: bool
    detail: str = ""
    fatal: bool = True

    def row(self) -> str:
        mark = "ok  " if self.passed else ("FAIL" if self.fatal else "warn")
        return f"[{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


def assert_frozen(
    config: ExperimentConfig,
    *,
    models: Mapping[str, SavedModel | None],
    data_manifest_sha256: str = "",
    scoring_rules_hash: str = "",
    squad_rules_hash: str = "",
) -> tuple[FreezeCheck, ...]:
    """Run every freeze check. Raise on the first fatal failure.

    Args:
        config: The experiment about to run.
        models: Loaded artifacts by ``ModelSpec.name``. ``None`` for the
            closed-form baseline, which has no artifact and nothing to freeze.
        data_manifest_sha256: The training data's identity as it stands now.
        scoring_rules_hash: The active scoring rules' identity.
        squad_rules_hash: The active squad rules' identity.

    Returns:
        Every check, in order, passed or not.

    Raises:
        FreezeViolation: on the first fatal failure, naming the fix.
    """
    seasons = set(config.evaluation_seasons)
    checks: list[FreezeCheck] = [
        _train_seasons_precede_test(config, models, seasons),
        _no_model_overlap_with_evaluation_period(config, models),
        _architecture_and_hyperparameters_frozen(config, models),
        _feature_selection_frozen(config, models),
        _calibration_not_fitted_on_test(config, seasons),
        _thresholds_not_selected_on_test(config, seasons),
        _folds_are_walk_forward(models),
        _inputs_are_pinned(config, data_manifest_sha256, scoring_rules_hash, squad_rules_hash),
    ]

    for check in checks:
        if not check.passed and check.fatal:
            raise FreezeViolation(f"{check.name}: {check.detail}")
    return tuple(checks)


def _train_seasons_precede_test(
    config: ExperimentConfig,
    models: Mapping[str, SavedModel | None],
    seasons: set[str],
) -> FreezeCheck:
    """Every training season sorts strictly before every evaluation season.

    Strictly stronger than ``SavedModel.overlaps``, which only compares
    gameweeks *within* one season and therefore passes a model trained on
    2025-26 and evaluated on 2024-25 — a model that has seen the future of the
    period it is being judged on.

    Lexicographic comparison is correct for ``YYYY-YY``.
    """
    offences: list[str] = []
    for name, saved in models.items():
        if saved is None:
            continue
        for trained in saved.trained_seasons:
            later = sorted(s for s in seasons if trained >= s)
            if later:
                offences.append(f"{name} trained on {trained}, evaluated on {later}")
    return FreezeCheck(
        name="train_seasons_precede_test",
        passed=not offences,
        detail="; ".join(offences),
    )


def _no_model_overlap_with_evaluation_period(
    config: ExperimentConfig, models: Mapping[str, SavedModel | None]
) -> FreezeCheck:
    """No artifact was fitted on a gameweek it is about to be scored over.

    The CLI has always checked this. It lives here too so the library enforces
    it, rather than the guarantee depending on which entry point was used.
    """
    offences: list[str] = []
    for name, saved in models.items():
        if saved is None:
            continue
        for window in config.windows:
            span = tuple(range(min(window.start_gameweeks), window.end_gameweek + 1))
            if saved.overlaps(window.season, span):
                offences.append(f"{name} overlaps {window.season} GW{span[0]}-{span[-1]}")
    return FreezeCheck(
        name="no_model_overlap_with_evaluation_period",
        passed=not offences,
        detail="; ".join(offences),
    )


def _architecture_and_hyperparameters_frozen(
    config: ExperimentConfig, models: Mapping[str, SavedModel | None]
) -> FreezeCheck:
    """The artifact is the one the config pinned.

    Only checked where a fingerprint was declared. A config that pins nothing
    is not lying about being frozen; it simply is not, and the report says so
    through the empty ``expected_fingerprint``.
    """
    offences: list[str] = []
    for spec in config.models:
        saved = models.get(spec.name)
        if saved is None or spec.expected_fingerprint is None:
            continue
        actual = saved.models.fingerprint()
        if actual != spec.expected_fingerprint:
            offences.append(
                f"{spec.name} is {actual[:12]}, config pinned {spec.expected_fingerprint[:12]}"
            )
    return FreezeCheck(
        name="architecture_and_hyperparameters_frozen",
        passed=not offences,
        detail="; ".join(offences),
    )


def _feature_selection_frozen(
    config: ExperimentConfig, models: Mapping[str, SavedModel | None]
) -> FreezeCheck:
    """The feature set has not widened since the config was written.

    Catches the discovery path silently promoting a new column between pinning
    and running, which would make the evaluated model a different model.
    """
    import hashlib

    offences: list[str] = []
    for spec in config.models:
        saved = models.get(spec.name)
        if saved is None or spec.expected_feature_columns_hash is None:
            continue
        columns = "|".join(saved.models.feature_columns)
        actual = hashlib.sha256(columns.encode("utf-8")).hexdigest()
        if actual != spec.expected_feature_columns_hash:
            offences.append(
                f"{spec.name} has {len(saved.models.feature_columns)} features hashing to "
                f"{actual[:12]}, config pinned {spec.expected_feature_columns_hash[:12]}"
            )
    return FreezeCheck(
        name="feature_selection_frozen", passed=not offences, detail="; ".join(offences)
    )


def _calibration_not_fitted_on_test(config: ExperimentConfig, seasons: set[str]) -> FreezeCheck:
    """No model applies a calibration measured on a season being evaluated.

    This one fires today for any 2025-26 evaluation with calibration enabled.
    ``PRICE_BAND_BIAS`` was measured on 2025-26 and the API applies it
    unconditionally, which is train-on-test wearing a different name.
    """
    overlap = sorted(set(CALIBRATION_MEASURED_ON) & seasons)
    using = [m.name for m in config.models if m.apply_price_calibration]
    failed = bool(overlap and using)
    return FreezeCheck(
        name="calibration_not_fitted_on_test",
        passed=not failed,
        detail=(
            f"models {using} apply a price calibration measured on {overlap}, which is "
            "among the evaluation seasons"
            if failed
            else ""
        ),
    )


def _thresholds_not_selected_on_test(config: ExperimentConfig, seasons: set[str]) -> FreezeCheck:
    """No policy threshold was tuned on a season being evaluated.

    Complete rather than a whitelist, because ``PolicyParameters`` forbids
    extra fields: a new knob cannot appear without declaring its provenance.
    """
    overlap = sorted(set(config.parameters.tuned_on_seasons) & seasons)
    return FreezeCheck(
        name="thresholds_not_selected_on_test",
        passed=not overlap,
        detail=(
            f"policy parameters were tuned on {overlap}, which is among the evaluation seasons"
            if overlap
            else ""
        ),
    )


def _folds_are_walk_forward(models: Mapping[str, SavedModel | None]) -> FreezeCheck:
    """Every stored fold still trains strictly before it validates.

    ``WalkForwardFold`` validates this at construction, so this re-asserts it on
    the *loaded* object rather than trusting what a pickle claims to be.
    """
    offences: list[str] = []
    for name, saved in models.items():
        if saved is None:
            continue
        for fold in saved.models.folds:
            if not _is_forward(fold):
                offences.append(
                    f"{name} fold {fold.fold_index} validates from "
                    f"{fold.validate_start} but trains to {fold.train_end}"
                )
    return FreezeCheck(
        name="folds_are_walk_forward", passed=not offences, detail="; ".join(offences)
    )


def _is_forward(fold: WalkForwardFold) -> bool:
    return fold.validate_start >= fold.train_end + 1 + fold.embargo_gameweeks


def _inputs_are_pinned(
    config: ExperimentConfig,
    data_manifest_sha256: str,
    scoring_rules_hash: str,
    squad_rules_hash: str,
) -> FreezeCheck:
    """The data and rules are the ones the config was written against.

    Non-fatal. A config that pinned nothing is the common case on a first run,
    and refusing it would make the framework unusable before it has ever
    produced the hashes it wants pinned. A *mismatch* is what matters, and that
    is reported.
    """
    mismatches: list[str] = []
    for label, pinned, actual in (
        ("data", config.data_manifest_sha256, data_manifest_sha256),
        ("scoring rules", config.scoring_rules_hash, scoring_rules_hash),
        ("squad rules", config.squad_rules_hash, squad_rules_hash),
    ):
        if pinned and actual and pinned != actual:
            mismatches.append(f"{label} pinned {pinned[:12]}, found {actual[:12]}")
    return FreezeCheck(
        name="inputs_are_pinned",
        passed=not mismatches,
        detail="; ".join(mismatches),
        fatal=True,
    )
