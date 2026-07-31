"""How one policy evaluation is specified — and why the config is a contract.

There is no configuration framework here, and adding Hydra or OmegaConf to run
a backtest would be infrastructure nobody asked for. The alternative is not
"scatter the knobs across CLI flags", which is where they live today: the
config *is* a contract, built from a preset, a JSON file or flags, and hashing
to the experiment's identity.

That hash is what makes a run findable. ``RunManifest.config_hash`` has always
existed and has always been the literal ``"default"``, because nothing computed
one. An experiment directory named for its config hash cannot be confused with
a directory for different settings, which is the whole of the resume story.

**``extra="forbid"`` on :class:`PolicyParameters` is load-bearing.** A new
tuning knob cannot appear without also declaring the seasons it was tuned on,
which is what makes "no threshold was selected on the test season" a complete
check rather than a whitelist somebody has to remember to extend.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.seeds import ROOT_SEED

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "FULL_GRID",
    "LEGACY_HEADLINE",
    "PRESETS",
    "SMOKE",
    "BootstrapUnit",
    "EvaluationWindow",
    "ExperimentConfig",
    "ModelSpec",
    "PolicyKind",
    "PolicyParameters",
    "PolicySpec",
    "RngScope",
    "SquadCohort",
    "SquadCohortSpec",
    "SquadSource",
]

EVALUATION_SCHEMA_VERSION: Final[str] = "evaluation_v1"


class PolicyKind(StrEnum):
    """How a policy chooses among the legal transfers it is offered."""

    MODEL = "model"
    HIGHEST_FORM = "highest_form"
    MOST_EXPENSIVE = "most_expensive"
    RANDOM = "random"
    HOLD = "hold"
    ORACLE = "oracle"
    """Reads outcomes. An upper bound, never a baseline — see the validator."""


class SquadSource(StrEnum):
    MOST_EXPENSIVE_LEGAL = "most_expensive_legal"
    TEMPLATE_OWNERSHIP = "template_ownership"
    MILP_PRE_DEADLINE = "milp_pre_deadline"
    RANDOM_LEGAL = "random_legal"
    EX_ANTE_COHORT = "ex_ante_cohort"
    IMPORTED_ARTIFACT = "imported_artifact"


class SquadCohort(StrEnum):
    WEAK = "weak"
    MEDIAN = "median"
    STRONG = "strong"
    UNBANDED = "unbanded"


class RngScope(StrEnum):
    """How often a stochastic policy's generator is reseeded."""

    GAMEWEEK = "gameweek"
    """Fresh per step. Decouples gameweeks, which is what allows checkpointing."""

    RUN = "run"
    """One generator for the whole walk. Reproduces the legacy coupling."""


class BootstrapUnit(StrEnum):
    SEASON_BLOCK = "season_block"
    """Resample seasons, then conditions within them. Conditions inside one
    season share the same actual outcomes, so they are not independent."""

    CONDITION = "condition"
    """Resample conditions directly. Understates the interval; used only when
    there are too few seasons to block on."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PolicyParameters(_Frozen):
    """Every knob a policy reads, with the provenance that makes it freezable.

    ``tuned_on_seasons`` is the field that matters. An empty tuple asserts
    "never tuned"; anything else is checked against the evaluation windows, and
    an overlap is train-on-test under a different name.
    """

    min_net_gain: float = 0.15
    risk_weight: float = 0.25
    horizon_gameweeks: int = Field(default=1, ge=1, le=10)
    horizon_discount: float = Field(default=0.85, gt=0.0, le=1.0)
    tuned_on_seasons: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        return _sha256(self.model_dump(mode="json"))


class ModelSpec(_Frozen):
    """A model artifact, pinned by fingerprint rather than by path.

    A path says where a file was; a fingerprint says what it contained. Only
    the second survives somebody retraining into the same filename, which is
    exactly the accident a frozen evaluation has to notice.
    """

    name: str = Field(min_length=1)
    artifact_path: Path | None = None
    """``None`` selects the closed-form baseline, which has no artifact."""
    expected_fingerprint: str | None = None
    expected_feature_columns_hash: str | None = None
    catalogue_version: str = ""
    shrink_by_skill: bool = False
    apply_price_calibration: bool = False


class PolicySpec(_Frozen):
    name: str = Field(min_length=1)
    selector: PolicyKind
    model: str = Field(min_length=1, description="Which ModelSpec this policy predicts with")
    stochastic: bool = False
    """Only a stochastic policy needs replicates. Asserted by a test, not assumed."""
    needs_candidates: bool = True
    """`hold` provably ignores its candidate list, so it can skip the search."""
    diagnostic: bool = False
    """Excluded from every baseline comparison. Required for the oracle."""


class SquadCohortSpec(_Frozen):
    source: SquadSource
    count: int = Field(default=1, ge=1)
    cohort: SquadCohort = SquadCohort.UNBANDED
    pool_size: int = Field(default=200, ge=1)
    artifact_paths: tuple[Path, ...] = ()


class EvaluationWindow(_Frozen):
    season: str = Field(min_length=1)
    start_gameweeks: tuple[int, ...] = Field(min_length=1)
    end_gameweek: int = Field(default=38, ge=1)

    @model_validator(mode="after")
    def _starts_precede_the_end(self) -> EvaluationWindow:
        late = [gw for gw in self.start_gameweeks if gw >= self.end_gameweek]
        if late:
            raise ValueError(
                f"start gameweeks {late} are not before the end gameweek {self.end_gameweek}"
            )
        return self


class ExperimentConfig(_Frozen):
    """One evaluation, completely specified.

    Everything that changes what runs is in the hash. ``code_version`` and
    ``git_dirty`` deliberately are not: including them would orphan a run
    directory after any commit, which would make a long experiment
    un-resumable. They live in the manifest, where a mismatch is reported.
    """

    name: str = Field(min_length=1)
    schema_version: str = EVALUATION_SCHEMA_VERSION

    windows: tuple[EvaluationWindow, ...] = Field(min_length=1)
    squads: tuple[SquadCohortSpec, ...] = Field(min_length=1)
    models: tuple[ModelSpec, ...] = Field(min_length=1)
    policies: tuple[PolicySpec, ...] = Field(min_length=1)
    parameters: PolicyParameters = PolicyParameters()

    replicates: int = Field(default=5, ge=1)
    root_seed: int = ROOT_SEED
    rng_scope: RngScope = RngScope.GAMEWEEK

    bootstrap_resamples: int = Field(default=10_000, ge=100)
    bootstrap_unit: BootstrapUnit = BootstrapUnit.SEASON_BLOCK
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)

    allow_oracle: bool = False
    require_exact_rules: bool = False
    max_workers: int = Field(default=1, ge=1)

    # Pinned inputs, checked before anything runs.
    data_manifest_sha256: str = ""
    scoring_rules_hash: str = ""
    squad_rules_hash: str = ""

    # Recorded, never hashed.
    code_version: str = ""
    git_dirty: bool = False

    @model_validator(mode="after")
    def _policies_reference_declared_models(self) -> ExperimentConfig:
        known = {m.name for m in self.models}
        unknown = sorted({p.model for p in self.policies} - known)
        if unknown:
            raise ValueError(f"policies reference undeclared models: {unknown}")
        return self

    @model_validator(mode="after")
    def _oracle_is_gated(self) -> ExperimentConfig:
        """An oracle that is not declared diagnostic is a fabricated baseline.

        It reads gameweek outcomes, which no manager had. Reporting it beside
        `hold` and `random` without a flag would make hindsight look like a
        policy somebody could have run.
        """
        for policy in self.policies:
            if policy.selector is PolicyKind.ORACLE and not (
                policy.diagnostic and self.allow_oracle
            ):
                raise ValueError(
                    f"policy {policy.name!r} reads outcomes. It may only run with "
                    "diagnostic=True and allow_oracle=True."
                )
        return self

    @property
    def evaluation_seasons(self) -> tuple[str, ...]:
        return tuple(sorted({w.season for w in self.windows}))

    def canonical_json(self) -> str:
        """The bytes the hash is taken over. Excludes recorded-only fields."""
        payload = self.model_dump(mode="json", exclude={"code_version", "git_dirty"})
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def experiment_id(self) -> str:
        return f"{self.name}-{self.config_hash()[:12]}"

    @property
    def reproducible(self) -> bool:
        return bool(self.code_version) and not self.git_dirty


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --- presets ----------------------------------------------------------------
#
# Module-level constants rather than a factory, following the idiom
# `discovery.acceptance.DEFAULT_POLICY` and `discovery.hypotheses.SEEDED_HYPOTHESES`
# already set. A preset is a value that can be diffed, not a function call whose
# result depends on when it ran.

_CLOSED_FORM = ModelSpec(name="closed_form")
_TRAINED = ModelSpec(name="trained")
_TRAINED_SHRUNK = ModelSpec(name="trained_shrunk", shrink_by_skill=True)

_STANDARD_POLICIES: Final[tuple[PolicySpec, ...]] = (
    PolicySpec(name="model", selector=PolicyKind.MODEL, model="closed_form"),
    PolicySpec(name="trained", selector=PolicyKind.MODEL, model="trained"),
    PolicySpec(name="trained_shrunk", selector=PolicyKind.MODEL, model="trained_shrunk"),
    PolicySpec(name="highest_form", selector=PolicyKind.HIGHEST_FORM, model="closed_form"),
    PolicySpec(name="most_expensive", selector=PolicyKind.MOST_EXPENSIVE, model="closed_form"),
    PolicySpec(name="random", selector=PolicyKind.RANDOM, model="closed_form", stochastic=True),
    PolicySpec(name="hold", selector=PolicyKind.HOLD, model="closed_form", needs_candidates=False),
)

SMOKE: Final[ExperimentConfig] = ExperimentConfig(
    name="smoke",
    windows=(EvaluationWindow(season="2024-25", start_gameweeks=(6,), end_gameweek=12),),
    squads=(SquadCohortSpec(source=SquadSource.MOST_EXPENSIVE_LEGAL),),
    models=(_CLOSED_FORM, _TRAINED, _TRAINED_SHRUNK),
    policies=_STANDARD_POLICIES,
    replicates=1,
)
"""Minutes, not hours. For proving the wiring, never for a claim."""

LEGACY_HEADLINE: Final[ExperimentConfig] = ExperimentConfig(
    name="legacy-headline",
    windows=(EvaluationWindow(season="2024-25", start_gameweeks=(6,), end_gameweek=25),),
    squads=(SquadCohortSpec(source=SquadSource.MOST_EXPENSIVE_LEGAL),),
    models=(_CLOSED_FORM, _TRAINED, _TRAINED_SHRUNK),
    policies=_STANDARD_POLICIES,
    replicates=1,
    rng_scope=RngScope.RUN,
)
"""Reproduces the existing published report's conditions exactly.

One season, one starting squad, one seed, and the legacy run-scoped RNG. It
exists to be *compared against*, not believed: the whole point of the framework
around it is that a single condition cannot support the claim the report makes.
"""

FULL_GRID: Final[ExperimentConfig] = ExperimentConfig(
    name="full-grid",
    windows=(
        EvaluationWindow(season="2023-24", start_gameweeks=(6, 12, 20)),
        EvaluationWindow(season="2024-25", start_gameweeks=(6, 12, 20)),
        EvaluationWindow(season="2025-26", start_gameweeks=(6, 12, 20)),
    ),
    squads=(
        SquadCohortSpec(source=SquadSource.MOST_EXPENSIVE_LEGAL),
        SquadCohortSpec(source=SquadSource.TEMPLATE_OWNERSHIP),
        SquadCohortSpec(source=SquadSource.MILP_PRE_DEADLINE),
        SquadCohortSpec(source=SquadSource.RANDOM_LEGAL, count=3),
        SquadCohortSpec(source=SquadSource.EX_ANTE_COHORT, cohort=SquadCohort.WEAK, count=2),
        SquadCohortSpec(source=SquadSource.EX_ANTE_COHORT, cohort=SquadCohort.MEDIAN, count=2),
        SquadCohortSpec(source=SquadSource.EX_ANTE_COHORT, cohort=SquadCohort.STRONG, count=2),
    ),
    models=(_CLOSED_FORM, _TRAINED, _TRAINED_SHRUNK),
    policies=_STANDARD_POLICIES,
    replicates=5,
)
"""Three seasons x three start gameweeks x twelve squads. 2022-23 is
training-only: it is the earliest season with xG, so evaluating on it would
leave no prior season to train from."""

PRESETS: Final[dict[str, ExperimentConfig]] = {
    "smoke": SMOKE,
    "legacy": LEGACY_HEADLINE,
    "full": FULL_GRID,
}
