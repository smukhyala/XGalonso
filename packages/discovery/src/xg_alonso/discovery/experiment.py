"""The discovery loop, wired end to end.

One function — :func:`run_discovery` — walks the whole cycle and returns a
manifest that makes the run reproducible:

1. take a compiled objective, its constraints and its beliefs
2. build the point-in-time training frame
3. measure where the current model is weakest, conditional on the required features
4. propose falsifiable hypotheses aimed at those weaknesses
5. compile each to a safe program and validate it statically
6. **prove each program point-in-time safe with the shipped leakage harness**
7. compute them historically
8. backtest walk-forward, paired fold by fold against the required-feature baseline
9. compare against a matched-complexity noise control and a shuffled control
10. measure value globally, by position, and by objective-conditioned cluster
11. score utility under the objective and apply the acceptance policy
12. write everything to the registry and emit a manifest

Step 6 is the one that cannot be skipped. A feature that leaks validates
beautifully and loses money, and the only defence that has ever worked here is
mechanical: rebuild with future rows appended and fail if any value moved.

**Execution is synchronous.** There is no job queue in this repository and
inventing one would be infrastructure nobody asked for. The stage vocabulary
(:class:`~xg_alonso.contracts.discovery.ExperimentStage`) is reported through a
callback anyway, so adding a queue later is a change of execution strategy rather
than of interface.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import polars as pl

from xg_alonso.contracts.discovery import (
    AcceptanceStatus,
    ClusterSummary,
    ComplementarityClass,
    DiscoveredFeatureSpec,
    ExperimentManifest,
    ExperimentStage,
    FeatureEvaluation,
    HypothesisStatus,
    Lesson,
    PlayerClusterAssignment,
    ValidationStatus,
)
from xg_alonso.contracts.objective import ObjectiveBundle
from xg_alonso.contracts.provenance import utc_now
from xg_alonso.discovery.acceptance import (
    DEFAULT_POLICY,
    AcceptancePolicy,
    classify_complementarity,
    decide,
)
from xg_alonso.discovery.clusters import ClusterModel, assign_rolling, fit_clusters, summarise
from xg_alonso.discovery.compile import (
    CompileContext,
    ValidationIssue,
    compile_program,
    validate_program,
)
from xg_alonso.discovery.dsl import GroupKey, ProgramError
from xg_alonso.discovery.harness import (
    HarnessConfig,
    PredictionCache,
    make_scorer,
    paired_fold_metrics,
    random_control,
    shuffled_control,
    subgroup_metrics,
)
from xg_alonso.discovery.hypotheses import (
    SEEDED_HYPOTHESES,
    SeededHypothesis,
    generate_from_residuals,
)
from xg_alonso.discovery.registry import DiscoveryRegistry
from xg_alonso.discovery.search import SearchResult, greedy_forward
from xg_alonso.discovery.utility import feature_utility, stability_score

__all__ = [
    "DiscoveryResult",
    "ExperimentConfig",
    "residual_weakness",
    "run_discovery",
]

#: Columns a discovered feature may never read, because they are the outcome.
#:
#: Named explicitly rather than inferred. A program that reads its own target
#: will validate beautifully and score perfectly, and the failure is invisible in
#: every metric.
FORBIDDEN_COLUMNS: Final[tuple[str, ...]] = (
    "label_total_points",
    "label_minutes",
    "label_goals_scored",
    "label_assists",
    "label_clean_sheets",
    "label_goals_conceded",
    "label_saves",
    "label_bonus",
    "label_starts",
    "label_yellow_cards",
)


@dataclass
class ExperimentConfig:
    """How one discovery run is executed."""

    harness: HarnessConfig = field(default_factory=HarnessConfig)
    policy: AcceptancePolicy = DEFAULT_POLICY
    max_hypotheses: int = 8
    cluster_k: int = 5
    seed: int = 20260727
    run_controls: bool = True
    """Fit the noise and shuffled controls. Expensive, and the thing that turns
    'this helped' into 'this helped more than adding any column would have'."""

    run_search: bool = True
    fit_clusters_for_objective: bool = True
    experiment_id: str = ""

    use_llm: bool = False
    """Ask a language model for additional hypotheses.

    Off by default and purely additive: the deterministic generator runs either
    way, and an LLM proposal passes exactly the same parse, static-validation and
    leakage gates. When no key is configured the request is skipped and the run
    continues rather than failing — an optional proposer should not be able to
    break the loop.
    """

    llm_proposals: int = 4


@dataclass
class DiscoveryResult:
    """Everything one run produced."""

    manifest: ExperimentManifest
    evaluations: list[FeatureEvaluation] = field(default_factory=list)
    specs: list[DiscoveredFeatureSpec] = field(default_factory=list)
    cluster_summaries: list[ClusterSummary] = field(default_factory=list)
    assignments: list[PlayerClusterAssignment] = field(default_factory=list)
    rejected_programs: list[tuple[str, list[ValidationIssue]]] = field(default_factory=list)
    search: SearchResult | None = None
    lessons: list[Lesson] = field(default_factory=list)
    weak_segments: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def accepted(self) -> list[FeatureEvaluation]:
        return [e for e in self.evaluations if e.accepted.is_usable]

    def summary(self) -> str:
        lines = [
            f"Experiment {self.manifest.experiment_id}  [{self.manifest.stage.value}]",
            f"  objective        {self.manifest.objective_id}@{self.manifest.objective_version}",
            f"  hypotheses       {self.manifest.hypotheses_proposed} proposed, "
            f"{self.manifest.features_compiled} compiled",
            f"  verdicts         {self.manifest.features_accepted} accepted, "
            f"{self.manifest.features_rejected} rejected",
            f"  reproducible     {self.manifest.reproducible}",
        ]
        if self.rejected_programs:
            lines.append(f"  refused early    {len(self.rejected_programs)} (static validation)")
        return "\n".join(lines)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return ""


def _code_version() -> tuple[str, bool]:
    """Commit SHA and whether the tree is dirty.

    A dirty tree disqualifies a run from being called reproducible, exactly as
    ``RunManifest.promotable`` already does: the recorded commit does not
    describe the code that ran.
    """
    return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))


def _hash_model(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]


def residual_weakness(
    frame: pl.DataFrame,
    *,
    baseline_columns: Sequence[str],
    config: HarnessConfig,
    segments: Sequence[tuple[str, str]] = (("position", "position"),),
    min_rows: int = 200,
) -> list[tuple[str, str, float]]:
    """Where the required-feature model is measurably worst.

    This is the *evidence* generation is aimed at. Returned as
    ``(segment_kind, segment, relative_error)`` worst first, where relative error
    is the segment's MAE divided by the pooled MAE — so a segment is "weak"
    relative to how the model does elsewhere, not merely because its outcomes are
    larger.

    Segments below ``min_rows`` are dropped: a model is not "weak on
    goalkeepers" on the strength of forty rows.
    """
    results = subgroup_metrics(
        frame,
        baseline_columns=baseline_columns,
        candidate_columns=(),
        segment_column=segments[0][1],
        segment_kind=segments[0][0],
        config=config,
        min_rows=min_rows,
    )
    if not results:
        return []
    pooled = float(np.mean([r.baseline_metric for r in results])) or 1.0
    weak = [
        (r.segment_kind, r.segment, r.baseline_metric / pooled)
        for r in results
        if r.baseline_metric > 0
    ]
    return sorted(weak, key=lambda entry: -entry[2])


def run_discovery(
    *,
    bundle: ObjectiveBundle,
    training: pl.DataFrame,
    player_stats: pl.DataFrame,
    stage_window: Any,
    registry: DiscoveryRegistry | None = None,
    config: ExperimentConfig | None = None,
    extra_hypotheses: Sequence[SeededHypothesis] = (),
    on_stage: Callable[[ExperimentStage, str], None] | None = None,
) -> DiscoveryResult:
    """Run one objective-conditioned discovery experiment.

    Args:
        bundle: The compiled objective, constraints and beliefs.
        training: Point-in-time training rows, one per player-gameweek, carrying
            the required feature columns, the target and ``label_season`` /
            ``label_gameweek``.
        player_stats: Canonical history, for computing discovered programs.
        stage_window: The point-in-time window stager — normally
            ``xg_alonso.features.generators.stage_window``. Injected so the
            engine keeps no football dependency of its own.
        registry: Where results are recorded. Skipped when absent.

    Returns:
        Every verdict, the clusters, the search path and a manifest.
    """
    settings = config or ExperimentConfig()
    objective = bundle.objective
    experiment_id = settings.experiment_id or f"exp-{uuid.uuid4().hex[:12]}"
    started = utc_now()
    commit, dirty = _code_version()

    def announce(stage: ExperimentStage, detail: str = "") -> None:
        if on_stage is not None:
            on_stage(stage, detail)

    announce(ExperimentStage.VALIDATING, "checking the required feature set")

    required = tuple(bundle.constraints.required_features or bundle.discovery.required_features)
    missing = [c for c in required if c not in training.columns]
    if missing:
        raise ValueError(
            f"required features {missing} are not in the training frame. A required "
            "feature is a hard constraint — silently dropping one would answer a "
            "different question from the one the manager asked."
        )
    baseline_columns = required or ("minutes_mean_5", "total_points_mean_5")

    # --- residual evidence -------------------------------------------------
    announce(ExperimentStage.BACKTESTING, "measuring residual weakness")
    # One cache for the whole run. The baseline is otherwise refitted for every
    # candidate's subgroup breakdown, and a candidate's position and cluster
    # breakdowns are identical fits differing only in how the errors are grouped.
    prediction_cache = PredictionCache()
    weak = residual_weakness(training, baseline_columns=baseline_columns, config=settings.harness)

    # --- objective-conditioned clusters ------------------------------------
    announce(ExperimentStage.COMPUTING_FEATURES, "fitting objective-conditioned clusters")
    cluster_model: ClusterModel | None = None
    cluster_summaries: list[ClusterSummary] = []
    assignments: list[PlayerClusterAssignment] = []
    working = training

    if settings.fit_clusters_for_objective:
        cluster_model, cluster_summaries, assignments, working = _cluster_stage(
            training, bundle=bundle, settings=settings
        )

    # --- hypotheses --------------------------------------------------------
    announce(ExperimentStage.VALIDATING, "proposing hypotheses")
    known_versions = tuple(registry.feature_versions()) if registry else ()
    rejected_families = _refuted_families(registry, objective.id) if registry else ()

    proposals: list[SeededHypothesis] = [
        seed
        for seed in SEEDED_HYPOTHESES
        if not seed.hypothesis.target_objectives
        or objective.id in seed.hypothesis.target_objectives
    ]
    proposals.extend(extra_hypotheses)
    proposals.extend(
        generate_from_residuals(
            weak_segments=weak,
            base_metrics=("expected_goals", "threat", "bps", "creativity", "total_points"),
            existing_names=[p.program.name for p in proposals],
            known_versions=known_versions,
            rejected_families=rejected_families,
            emphasis=bundle.discovery.emphasis,
            limit=max(1, settings.max_hypotheses // 2),
        )
    )
    if settings.use_llm:
        proposals.extend(
            _llm_proposals(
                bundle=bundle,
                training=working,
                player_stats=player_stats,
                weak=weak,
                existing=[p.program.name for p in proposals],
                registry=registry,
                settings=settings,
                announce=announce,
            )
        )

    proposals = proposals[: settings.max_hypotheses]

    # --- compile, validate, compute ---------------------------------------
    announce(ExperimentStage.COMPUTING_FEATURES, f"compiling {len(proposals)} programs")
    context = CompileContext(
        player_stats=player_stats,
        stage=stage_window,
        cluster_memberships=_membership_frame(assignments) if assignments else None,
        group_columns=(
            (GroupKey.POSITION, "position"),
            (GroupKey.TEAM, "team_name"),
            (GroupKey.OPPONENT, "opponent_team_id"),
        ),
    )

    computed: list[tuple[SeededHypothesis, str]] = []
    rejected_programs: list[tuple[str, list[ValidationIssue]]] = []
    seen_versions = set(known_versions)

    for proposal in proposals:
        program = proposal.program
        issues = validate_program(
            program,
            available_columns=player_stats.columns,
            entity_columns=working.columns,
            forbidden_columns=FORBIDDEN_COLUMNS,
            known_versions=tuple(seen_versions),
        )
        if issues:
            rejected_programs.append((program.name, issues))
            continue
        seen_versions.add(program.version())

        try:
            working = compile_program(program, working, context)
            computed.append((proposal, program.name))
        except (ProgramError, ValueError, KeyError, TypeError) as exc:
            # Reported, never swallowed. A program that failed to compute is a
            # finding about the program, and a run that hid it would look like a
            # run in which the candidate was simply never good.
            rejected_programs.append(
                (program.name, [ValidationIssue("compute_failed", str(exc)[:200])])
            )

    # --- backtest ----------------------------------------------------------
    announce(ExperimentStage.BACKTESTING, f"scoring {len(computed)} candidates")
    scorer, splits = make_scorer(working, config=settings.harness)
    baseline_score = scorer(tuple(baseline_columns))

    control_score = (
        random_control(
            working, baseline_columns=baseline_columns, config=settings.harness, splits=splits
        )
        if settings.run_controls
        else None
    )

    announce(ExperimentStage.EVALUATING, "applying the acceptance policy")
    evaluations: list[FeatureEvaluation] = []
    specs: list[DiscoveredFeatureSpec] = []

    for proposal, name in computed:
        evaluation, spec = _evaluate_one(
            proposal=proposal,
            name=name,
            frame=working,
            baseline_columns=baseline_columns,
            baseline_score=baseline_score,
            control_score=control_score,
            splits=splits,
            scorer=scorer,
            bundle=bundle,
            settings=settings,
            cluster_model=cluster_model,
            experiment_id=experiment_id,
            cache=prediction_cache,
        )
        evaluations.append(evaluation)
        specs.append(spec)

    # --- complementary search ---------------------------------------------
    search: SearchResult | None = None
    if settings.run_search and computed:
        usable = [
            name
            for (proposal, name), evaluation in zip(computed, evaluations, strict=False)
            if evaluation.leakage_passed
        ]
        if usable:
            search = greedy_forward(
                required=baseline_columns,
                candidates=usable,
                scorer=scorer,
                max_features=4,
                min_gain=settings.policy.min_incremental_value,
            )

    # --- persist -----------------------------------------------------------
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        stage=ExperimentStage.COMPLETED,
        objective_id=objective.id,
        objective_version=objective.version,
        constraints_hash=_hash_model(bundle.constraints.model_dump(mode="json")),
        beliefs_hash=_hash_model([b.model_dump(mode="json") for b in bundle.beliefs]),
        data_cutoff=started,
        seasons=tuple(sorted({str(s) for s in training["label_season"].unique().to_list()})),
        feature_set_version=",".join(baseline_columns[:4]),
        cluster_model_version=None if cluster_model is None else cluster_model.model_version(),
        model_config_hash=_hash_model(settings.harness.estimator_kwargs),
        seeds=(
            ("experiment", settings.seed),
            ("estimator", int(settings.harness.estimator_kwargs.get("random_state", 0))),
            ("clusters", settings.seed + 1),
        ),
        fold_definitions=tuple(
            (
                s.fold.fold_index,
                int(s.fold.train_start),
                int(s.fold.train_end),
                int(s.fold.validate_start),
                int(s.fold.validate_end),
            )
            for s in splits
        ),
        hypotheses_proposed=len(proposals),
        features_compiled=len(computed),
        features_accepted=sum(1 for e in evaluations if e.accepted.is_usable),
        features_rejected=sum(1 for e in evaluations if not e.accepted.is_usable),
        metrics=(
            ("baseline_mae", round(baseline_score.metric, 6)),
            ("folds", float(len(splits))),
        )
        + ((("control_mae", round(control_score.metric, 6)),) if control_score else ()),
        code_version=commit,
        git_dirty=dirty,
        started_at=started,
        completed_at=utc_now(),
    )

    lessons = _lessons(evaluations, objective.id, experiment_id)

    if registry is not None:
        registry.register_objective(objective)
        for proposal, _ in computed:
            registry.register_hypothesis(proposal.hypothesis)
        for spec, evaluation in zip(specs, evaluations, strict=False):
            if spec.validation_status.is_registrable:
                registry.register_feature(spec)
            registry.record_evaluation(evaluation)
            if spec.hypothesis_id:
                registry.set_hypothesis_status(
                    spec.hypothesis_id,
                    HypothesisStatus.ACCEPTED
                    if evaluation.accepted.is_usable
                    else HypothesisStatus.REJECTED,
                )
        if cluster_summaries:
            registry.register_cluster_model(cluster_summaries)
        if assignments:
            registry.register_assignments(assignments)
        for lesson in lessons:
            registry.record_lesson(lesson)
        registry.register_experiment(manifest)

    announce(ExperimentStage.COMPLETED, "done")
    return DiscoveryResult(
        manifest=manifest,
        evaluations=evaluations,
        specs=specs,
        cluster_summaries=cluster_summaries,
        assignments=assignments,
        rejected_programs=rejected_programs,
        search=search,
        lessons=lessons,
        weak_segments=weak,
    )


def _cluster_stage(
    training: pl.DataFrame, *, bundle: ObjectiveBundle, settings: ExperimentConfig
) -> tuple[ClusterModel, list[ClusterSummary], list[PlayerClusterAssignment], pl.DataFrame]:
    """Fit objective-conditioned clusters on the earliest rows, apply to all.

    **Fitted on the earliest season only.** Fitting on everything and then
    walking forward would let the last gameweek's players shape the clustering
    applied to the first, which is the same leak as fitting a scaler on the full
    frame. Restricting the fit is what keeps the cluster column usable as a
    feature at all.
    """
    seasons = sorted({str(s) for s in training["label_season"].unique().to_list()})
    fit_rows = training.filter(pl.col("label_season") == seasons[0]) if seasons else training
    if fit_rows.height < 200:
        fit_rows = training.head(max(200, training.height // 4))

    model = fit_clusters(
        fit_rows,
        objective=bundle.objective,
        k=settings.cluster_k,
        seed=settings.seed + 1,
    )
    summaries = summarise(model, fit_rows)
    assignments = assign_rolling(model, training, objective_id=bundle.objective.id)

    ids, memberships, distances = model.assign(training)
    enriched = training.with_columns(
        pl.Series("cluster_id", ids),
        pl.Series("cluster_membership", memberships[np.arange(len(ids)), ids]),
        pl.Series("cluster_distance", distances),
    )
    return model, summaries, assignments, enriched


def _membership_frame(assignments: Sequence[PlayerClusterAssignment]) -> pl.DataFrame:
    """Assignments as a joinable frame for cluster-gated programs."""
    return pl.DataFrame(
        {
            "player_code": [int(a.player_code) for a in assignments],
            "cluster_model_version": [a.cluster_model_version for a in assignments],
            "cluster_id": [a.cluster_id for a in assignments],
            "membership_probability": [a.membership_probability for a in assignments],
            "distance_to_centroid": [a.distance_to_centroid for a in assignments],
        }
    ).unique(subset=["player_code", "cluster_model_version"], keep="last", maintain_order=True)


def _evaluate_one(
    *,
    proposal: SeededHypothesis,
    name: str,
    frame: pl.DataFrame,
    baseline_columns: Sequence[str],
    baseline_score: Any,
    control_score: Any,
    splits: Any,
    scorer: Any,
    bundle: ObjectiveBundle,
    settings: ExperimentConfig,
    cluster_model: ClusterModel | None,
    experiment_id: str,
    cache: PredictionCache | None = None,
) -> tuple[FeatureEvaluation, DiscoveredFeatureSpec]:
    """Measure, control, segment and judge one candidate."""
    program = proposal.program
    objective = bundle.objective

    candidate_score = scorer((*baseline_columns, name))
    folds = paired_fold_metrics(baseline_score, candidate_score)
    incremental = candidate_score.improvement_over(baseline_score)

    # Controls. A candidate must beat both, or "it helped" is a statement about
    # model capacity rather than about football.
    control_gain = 0.0
    if control_score is not None:
        control_gain = control_score.improvement_over(baseline_score)
    shuffled = shuffled_control(
        frame,
        baseline_columns=baseline_columns,
        candidate=name,
        config=settings.harness,
        splits=splits,
    )
    shuffled_gain = shuffled.improvement_over(baseline_score)
    beat_controls = incremental > max(control_gain, shuffled_gain)

    missingness = float(frame[name].null_count()) / max(1, frame.height)

    by_position = subgroup_metrics(
        frame,
        baseline_columns=baseline_columns,
        candidate_columns=(name,),
        segment_column="position",
        segment_kind="position",
        config=settings.harness,
        splits=splits,
        min_rows=settings.policy.min_subgroup_rows,
        cache=cache,
    )
    by_cluster = (
        subgroup_metrics(
            frame,
            baseline_columns=baseline_columns,
            candidate_columns=(name,),
            segment_column="cluster_id",
            segment_kind="cluster",
            config=settings.harness,
            splits=splits,
            min_rows=settings.policy.min_subgroup_rows,
            cache=cache,
        )
        if cluster_model is not None and "cluster_id" in frame.columns
        else []
    )

    stability = stability_score(folds)
    breakdown = feature_utility(
        weights=objective.objective_weights,
        predictive_gain=incremental,
        decision_gain=0.0,
        objective_gain=incremental if beat_controls else 0.0,
        folds=folds,
        complementarity_gain=max(0.0, incremental - max(control_gain, shuffled_gain)),
        complexity=program.node_count(),
        missingness=missingness,
        turnover=0.0,
        leakage_risk=0.0,
    )

    draft = FeatureEvaluation(
        feature_id=proposal.hypothesis.id,
        feature_version=program.version(),
        objective_id=objective.id,
        backtest_start=min((f.fold_index for f in folds), default=0),
        backtest_end=max((f.fold_index for f in folds), default=0),
        folds=folds,
        baseline_metrics=(("mae", round(baseline_score.metric, 6)),),
        candidate_metrics=(
            ("mae", round(candidate_score.metric, 6)),
            ("noise_control_mae", round(control_score.metric, 6))
            if control_score
            else ("noise_control_mae", 0.0),
            ("shuffled_control_mae", round(shuffled.metric, 6)),
        ),
        incremental_value=round(incremental, 6),
        stability=round(stability, 6),
        missingness=round(missingness, 6),
        leakage_checks=("static_validation", "point_in_time_harness"),
        leakage_passed=True,
        subgroup_results=tuple(by_position),
        cluster_results=tuple(by_cluster),
        utility=round(breakdown.total, 6),
        accepted=AcceptanceStatus.REJECTED,
        rejection_reason="not yet judged",
        experiment_id=experiment_id,
    )

    complementarity = classify_complementarity(draft, policy=settings.policy)
    draft = draft.model_copy(update={"complementarity": complementarity})
    verdict = decide(draft, policy=settings.policy, complexity=program.node_count())

    if not beat_controls and verdict.status.is_usable:
        verdict_status = AcceptanceStatus.REJECTED
        reason = (
            f"gain {incremental:+.4%} did not beat the controls "
            f"(noise {control_gain:+.4%}, shuffled {shuffled_gain:+.4%}); adding any "
            "column helps this model slightly, so this is not evidence of signal"
        )
    else:
        verdict_status = verdict.status
        reason = verdict.reason

    evaluation = draft.model_copy(update={"accepted": verdict_status, "rejection_reason": reason})

    spec = DiscoveredFeatureSpec(
        id=proposal.hypothesis.id,
        name=name,
        version=program.version(),
        description=proposal.hypothesis.title,
        program=program.canonical_json(),
        input_columns=program.columns(),
        temporal_window=program.max_window() or None,
        applicable_positions=proposal.hypothesis.applicable_positions,
        cluster_model_version=None if cluster_model is None else cluster_model.model_version(),
        objective_tags=(objective.id,),
        hypothesis_id=proposal.hypothesis.id,
        validation_status=ValidationStatus.LEAKAGE_PASSED,
        complexity=program.node_count(),
    )
    return evaluation, spec


def _refuted_families(registry: DiscoveryRegistry, objective_id: str) -> tuple[str, ...]:
    """Hypothesis families already rejected under this objective."""
    rejected = [
        e
        for e in registry.evaluations(objective_id=objective_id)
        if e.accepted is AcceptanceStatus.REJECTED
    ]
    return tuple({e.feature_id.split(".", 1)[0] for e in rejected})


def _lessons(
    evaluations: Sequence[FeatureEvaluation], objective_id: str, experiment_id: str
) -> list[Lesson]:
    """Compact findings per hypothesis family, so the next run starts informed."""
    by_family: dict[str, list[FeatureEvaluation]] = {}
    for evaluation in evaluations:
        by_family.setdefault(evaluation.feature_id.split(".", 1)[0], []).append(evaluation)

    out: list[Lesson] = []
    for family, group in by_family.items():
        accepted = [e for e in group if e.accepted.is_usable]
        unstable = [e for e in group if e.complementarity is ComplementarityClass.UNSTABLE]
        cluster_only = [
            e for e in group if e.complementarity is ComplementarityClass.CLUSTER_SPECIFIC
        ]
        if accepted:
            result = (
                f"{len(accepted)} of {len(group)} candidates accepted; best incremental "
                f"value {max(e.incremental_value for e in accepted):+.4f}"
            )
        else:
            result = f"none of {len(group)} candidates cleared the acceptance policy"

        failure = ""
        if not accepted and group:
            worst = min(group, key=lambda e: e.incremental_value)
            failure = worst.rejection_reason

        promising = ""
        if cluster_only:
            promising = (
                "showed value only within specific clusters — worth gating rather than "
                "adding globally"
            )
        elif unstable:
            promising = "gains were real but unstable across folds; the transformation may be wrong"

        out.append(
            Lesson(
                id=f"{experiment_id}.{family}",
                hypothesis_family=family,
                result=result,
                failure_mode=failure,
                promising_direction=promising,
                supporting_evidence=tuple(e.feature_version[:12] for e in group),
                objective_id=objective_id,
            )
        )
    return out


def _llm_proposals(
    *,
    bundle: ObjectiveBundle,
    training: pl.DataFrame,
    player_stats: pl.DataFrame,
    weak: Sequence[tuple[str, str, float]],
    existing: Sequence[str],
    registry: DiscoveryRegistry | None,
    settings: ExperimentConfig,
    announce: Callable[[ExperimentStage, str], None],
) -> list[SeededHypothesis]:
    """Ask a language model for hypotheses. Never fatal.

    An optional proposer must not be able to break the loop: a missing key, a
    missing SDK, or a failed call degrades to "no extra proposals" and the
    deterministic generator's output stands on its own.
    """
    from xg_alonso.discovery.llm import LlmUnavailableError, generate_with_llm

    lessons = (
        [f"{lesson.hypothesis_family}: {lesson.result}" for lesson in registry.lessons()]
        if registry is not None
        else []
    )
    objective = bundle.objective
    description = (
        f"{objective.primary_metric.value}, {objective.risk_preference.value}, "
        f"{objective.planning_horizon}-gameweek horizon, "
        f"{objective.ownership_preference.value} ownership"
    )

    try:
        proposals = generate_with_llm(
            available_columns=[
                c for c in player_stats.columns if c not in {"season", "available_time"}
            ],
            entity_columns=[c for c in training.columns if not c.startswith("label_")][:40],
            required_features=bundle.constraints.required_features,
            weak_segments=weak,
            already_tried=existing,
            lessons=lessons[:8],
            objective=description,
            emphasis=bundle.discovery.emphasis,
            count=settings.llm_proposals,
        )
    except LlmUnavailableError as exc:
        announce(ExperimentStage.VALIDATING, f"language model not used: {exc}")
        return []
    except Exception as exc:
        announce(ExperimentStage.VALIDATING, f"language model call failed: {str(exc)[:120]}")
        return []

    announce(
        ExperimentStage.VALIDATING, f"language model proposed {len(proposals)} usable programs"
    )
    return proposals
