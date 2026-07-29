"""Player clusters that move over time, and that mean different things per objective.

Three properties the existing :mod:`xg_alonso.features.archetypes` does not have,
each of which turned out to be load-bearing:

**Clusters move.** ``archetypes`` fits one static clustering. But a midfielder
pushed forward after an injury genuinely becomes a different kind of asset, and a
permanent label cannot notice. Here a cluster is keyed by gameweek, and the
transition history is itself a signal — a player *leaving* a rotation cluster is
often the earliest evidence a manager has.

**Membership is soft.** A hard ``argmin`` says a player sitting exactly between
two archetypes belongs wholly to one. Soft membership lets a gated feature apply
in proportion, which is what a boundary actually means.

**Similarity depends on the objective.** This is the central idea. Two forwards
with identical expected points are interchangeable to a manager maximising
points, and *completely different* to one chasing a mini-league if one is owned
by 60% of the field and the other by 3%. The distance function has to know which
question is being asked.

Two approaches, both implemented, and compared against three controls:

``A — objective-weighted distance`` (:func:`objective_weights`)
    Reweight the embedding axes by how much the objective cares about each.
    Deterministic, cheap, interpretable, and correct at this data volume.

``B — supervised projection`` (:func:`fit_supervised_projection`)
    Learn a ridge-regularised linear map onto the objective's own target, then
    cluster in that space. Only fitted when there are enough rows to validate it;
    below the floor it **refuses** rather than degrading, because a projection
    fitted on a thin fold is noise with a matrix behind it.

**Fit and apply are separate, structurally.** :meth:`ClusterModel.assign` uses
stored centroids. There is no way to cluster a frame using its own statistics, so
a validation fold cannot influence the clustering applied to it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

import numpy as np
import polars as pl

from xg_alonso.contracts.discovery import ClusterSummary, PlayerClusterAssignment
from xg_alonso.contracts.identifiers import GameweekId, PlayerCode
from xg_alonso.contracts.objective import ManagerObjective, OwnershipPreference, PrimaryMetric
from xg_alonso.discovery.embeddings import EmbeddingModel, fit_embedding

__all__ = [
    "CLUSTER_VERSION",
    "ClusterModel",
    "ClusterSelection",
    "assign_rolling",
    "fit_clusters",
    "fit_supervised_projection",
    "objective_weights",
    "select_k",
    "summarise",
    "transition_history",
]

CLUSTER_VERSION: Final[str] = "cluster_v1"

#: Minimum rows before clustering is attempted. Below this the result describes
#: the sample rather than the population.
_MIN_ROWS: Final[int] = 60

#: Minimum share of the population a cluster must hold to count as real.
_MIN_CLUSTER_SHARE: Final[float] = 0.04

#: Rows needed before the supervised projection is fitted at all.
_MIN_SUPERVISED_ROWS: Final[int] = 800

#: Softmax temperature for k-means membership. Chosen so a player at equal
#: distance from two centroids splits roughly evenly, and one clearly inside a
#: cluster reads above 0.8 — a gate that is never confident is not a gate.
_MEMBERSHIP_TEMPERATURE: Final[float] = 0.5


def objective_weights(objective: ManagerObjective, columns: Sequence[str]) -> np.ndarray:
    """Per-column emphasis implied by an objective. **Approach A.**

    Returns a non-negative weight per embedding column. Applied to a
    *standardised* space, so a weight of 2.0 genuinely doubles that axis's
    influence on distance rather than reflecting its original units.

    The weights encode what each objective means by "similar":

    - **differential / rank gain** — ceiling, volatility, ownership and fixture
      explosiveness. Two players with the same mean are not alike if one is owned
      by 60% of the field.
    - **downside protection** — expected minutes, start rate, floor, consistency.
      The tail that matters here is the left one.
    - **team value growth** — transfer flow, ownership momentum, price.
    - **transfer flexibility / wildcard** — minutes stability and price structure
      over a longer window.

    Columns not named by the objective keep weight 1.0 rather than 0.0. Zeroing
    them would say the objective considers them irrelevant, which is stronger
    than the evidence: it considers them *less* relevant.
    """
    emphasis: dict[str, float] = {}

    def bump(names: Sequence[str], weight: float) -> None:
        for name in names:
            emphasis[name] = max(emphasis.get(name, 1.0), weight)

    metric = objective.primary_metric
    secondary = set(objective.secondary_metrics)

    if metric is PrimaryMetric.EXPECTED_RANK_GAIN or PrimaryMetric.DIFFERENTIAL_YIELD in secondary:
        bump(("total_points_max_5", "total_points_std_10", "threat_per90_5"), 2.2)
        bump(("selected_mean_5", "transfers_balance_mean_5"), 2.0)
        bump(("expected_goals_per90_5", "opponent_conceded_xg_mean_5"), 1.6)

    if metric is PrimaryMetric.DIFFERENTIAL_YIELD:
        bump(("selected_mean_5",), 2.5)

    if metric is PrimaryMetric.DOWNSIDE_PROTECTION:
        bump(
            (
                "minutes_mean_5",
                "minutes_mean_20",
                "starts_mean_5",
                "appearance_rate_10",
                "minutes_when_played_mean_5",
            ),
            2.2,
        )
        bump(("total_points_std_10",), 1.8)
        bump(("selected_mean_5",), 1.5)

    if metric is PrimaryMetric.TEAM_VALUE_GROWTH:
        bump(("transfers_balance_mean_5", "selected_mean_5", "value_mean_1"), 2.4)
        bump(("total_points_mean_5",), 1.4)

    if metric is PrimaryMetric.CAPTAINCY_UPSIDE or PrimaryMetric.CAPTAINCY_UPSIDE in secondary:
        bump(("total_points_max_5", "expected_goals_per90_5", "threat_per90_5"), 2.0)
        bump(("minutes_mean_5", "starts_mean_5"), 1.6)

    if metric is PrimaryMetric.TRANSFER_FLEXIBILITY:
        bump(("value_mean_1", "minutes_mean_20", "appearance_rate_10"), 2.0)

    # Ownership preference sharpens the ownership axis in either direction: a
    # template-seeker and a differential-hunter both need ownership to dominate
    # similarity, they just want opposite ends of it. Distance is symmetric, so
    # one weight serves both.
    if objective.ownership_preference is not OwnershipPreference.NEUTRAL:
        bump(("selected_mean_5",), 2.2)

    if objective.planning_horizon >= 5:
        bump(("minutes_mean_20", "total_points_mean_5"), 1.5)

    return np.array([emphasis.get(name, 1.0) for name in columns], dtype=np.float64)


@dataclass(frozen=True)
class ClusterSelection:
    """Why a particular ``k`` was chosen. Kept so the choice is reviewable."""

    k: int
    silhouette: float
    min_share: float
    stability: float
    utility: float
    score: float
    considered: tuple[tuple[int, float], ...] = ()

    def explain(self) -> str:
        return (
            f"k={self.k} (silhouette {self.silhouette:.3f}, smallest cluster "
            f"{self.min_share:.1%}, stability {self.stability:.2f}, score {self.score:.3f})"
        )


@dataclass(frozen=True)
class ClusterModel:
    """A fitted clustering. Immutable; :meth:`assign` is the only way to use it."""

    version: str
    embedding: EmbeddingModel
    centroids: np.ndarray
    weights: np.ndarray
    """Per-dimension weights applied before distance. Ones for an unconditioned model."""

    k: int
    method: str
    objective_id: str | None = None
    projection: np.ndarray | None = None
    """Approach B's supervised map, applied after the embedding. ``None`` for A."""

    selection: ClusterSelection | None = None
    position: str | None = None
    fitted_rows: int = 0
    seed: int = 20260728

    def model_version(self) -> str:
        """Identity that includes the objective, so two conditionings never collide."""
        payload = "|".join(
            [
                self.version,
                self.embedding.fingerprint(),
                str(self.k),
                self.method,
                self.objective_id or "none",
                self.position or "all",
                f"{float(np.sum(self.centroids)):.6f}",
            ]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        suffix = f"-{self.objective_id}" if self.objective_id else ""
        return f"{self.version}{suffix}-{digest}"

    def _space(self, frame: pl.DataFrame) -> np.ndarray:
        vectors = self.embedding.transform(frame)
        if self.projection is not None:
            vectors = vectors @ self.projection
        return np.asarray(vectors * self.weights, dtype=np.float64)

    def assign(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign rows using the **stored** centroids.

        Returns:
            ``(cluster_ids, membership_probabilities, distances)``. Memberships
            are a softmax over negative squared distance, so a player between two
            centroids splits between them rather than being forced into one.
        """
        vectors = self._space(frame)
        if not vectors.size:
            empty = np.zeros((0,), dtype=np.int64)
            return empty, np.zeros((0, self.k)), np.zeros((0,))

        squared = np.stack(
            [np.sum((vectors - centroid) ** 2, axis=1) for centroid in self.centroids], axis=1
        )
        ids = np.argmin(squared, axis=1).astype(np.int64)
        distances = np.sqrt(squared[np.arange(vectors.shape[0]), ids])

        # Softmax over negative scaled distance. Subtracting the row max first
        # keeps the exponential from overflowing on a far-out player, which is
        # exactly the row a gate would otherwise silently turn into NaN.
        scaled = -squared / max(_MEMBERSHIP_TEMPERATURE, 1e-9)
        scaled -= scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled)
        memberships = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
        return ids, memberships, distances


def _kmeans(
    matrix: np.ndarray, k: int, *, seed: int, iterations: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with k-means++ seeding.

    Mirrors :func:`xg_alonso.features.archetypes._kmeans` deliberately. That one
    is private to ``feature_factory``, which keeps a numpy-only dependency chain
    on purpose; importing it here would either break that boundary or force
    scikit-learn down a layer. The algorithm is twenty lines and the seeding is
    the part that matters — labels reach user-facing prose, so a run-to-run
    reshuffle would rename every player's cluster between page loads.
    """
    generator = np.random.default_rng(seed)
    rows = matrix.shape[0]
    centres = [matrix[generator.integers(rows)]]
    for _ in range(k - 1):
        gaps = np.min(
            np.stack([np.sum((matrix - centre) ** 2, axis=1) for centre in centres]), axis=0
        )
        total = float(gaps.sum())
        if total <= 0:
            centres.append(matrix[generator.integers(rows)])
            continue
        centres.append(matrix[generator.choice(rows, p=gaps / total)])

    centroids = np.stack(centres)
    assignments = np.zeros(rows, dtype=np.int64)
    for _ in range(iterations):
        squared = np.stack(
            [np.sum((matrix - centroid) ** 2, axis=1) for centroid in centroids], axis=1
        )
        updated = np.argmin(squared, axis=1)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        for index in range(k):
            members = matrix[assignments == index]
            if members.size:
                centroids[index] = members.mean(axis=0)
    return assignments, centroids


def _silhouette(matrix: np.ndarray, labels: np.ndarray, *, sample: int, seed: int) -> float:
    """Mean silhouette, on a sample when the population is large.

    Silhouette is O(n²). At 700 players per gameweek the exact value is
    affordable; across a season it is not, and a sampled estimate of the right
    quantity beats an exact value nobody waits for.
    """
    unique = np.unique(labels)
    if unique.size < 2 or matrix.shape[0] < 3:
        return 0.0
    from sklearn.metrics import silhouette_score

    rows = matrix.shape[0]
    if rows > sample:
        generator = np.random.default_rng(seed)
        index = generator.choice(rows, size=sample, replace=False)
        matrix, labels = matrix[index], labels[index]
        if np.unique(labels).size < 2:
            return 0.0
    return float(silhouette_score(matrix, labels))


def select_k(
    matrix: np.ndarray,
    *,
    candidates: Sequence[int] = (3, 4, 5, 6, 7, 8),
    seed: int = 20260728,
    utility: Mapping[int, float] | None = None,
    silhouette_sample: int = 1500,
) -> ClusterSelection:
    """Choose a cluster count from several criteria, never from one.

    The brief is explicit that a single unsupervised metric must not decide this,
    and it is right: silhouette rewards a small number of fat clusters, so
    following it alone reliably produces three clusters that mean "good",
    "average" and "bad" — a ranking with extra steps, which is exactly what
    ``archetypes`` warns against.

    Combined here:

    - **silhouette** — are the clusters separated at all
    - **minimum share** — a cluster holding 1% of the pool describes nobody, and
      is penalised hard rather than merely noted
    - **stability** — agreement between two seeds. A partition that moves when
      the seed does is a partition of noise.
    - **downstream utility** — supplied by the caller when it has measured it.
      This is the only criterion that knows what the clusters are *for*.
    """
    scores: list[tuple[int, float]] = []
    best: ClusterSelection | None = None

    for k in candidates:
        if k >= matrix.shape[0]:
            continue
        labels, _ = _kmeans(matrix, k, seed=seed)
        counts = np.bincount(labels, minlength=k)
        min_share = float(counts.min()) / float(matrix.shape[0])

        silhouette = _silhouette(matrix, labels, sample=silhouette_sample, seed=seed)

        # Stability: refit with a different seed and measure agreement through a
        # pair-counting index, which is invariant to how the labels are numbered.
        other, _ = _kmeans(matrix, k, seed=seed + 977)
        stability = _label_agreement(labels, other)

        downstream = float(utility.get(k, 0.0)) if utility else 0.0

        # A cluster below the minimum share is a defect, not a trade-off, so the
        # penalty is steep enough to disqualify rather than to discourage.
        share_penalty = 0.0 if min_share >= _MIN_CLUSTER_SHARE else 1.0
        score = 0.40 * silhouette + 0.30 * stability + 0.30 * downstream - share_penalty
        scores.append((k, score))
        if best is None or score > best.score:
            best = ClusterSelection(
                k=k,
                silhouette=silhouette,
                min_share=min_share,
                stability=stability,
                utility=downstream,
                score=score,
            )

    if best is None:
        raise ValueError(
            f"no candidate k fits {matrix.shape[0]} rows; clustering fewer rows than "
            "clusters would give every player their own archetype"
        )
    return ClusterSelection(
        k=best.k,
        silhouette=best.silhouette,
        min_share=best.min_share,
        stability=best.stability,
        utility=best.utility,
        score=best.score,
        considered=tuple(scores),
    )


def _label_agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Rand index between two labelings. Invariant to label permutation."""
    if left.size != right.size or left.size < 2:
        return 0.0
    # Compare pair co-membership rather than labels, so cluster 0 becoming
    # cluster 2 between seeds is correctly read as no change at all.
    same_left = left[:, None] == left[None, :]
    same_right = right[:, None] == right[None, :]
    upper = np.triu_indices(left.size, k=1)
    agree = same_left[upper] == same_right[upper]
    return float(np.mean(agree))


def fit_supervised_projection(
    vectors: np.ndarray,
    target: np.ndarray,
    *,
    n_components: int = 4,
    ridge: float = 1.0,
    min_rows: int = _MIN_SUPERVISED_ROWS,
) -> np.ndarray | None:
    """**Approach B.** A ridge-regularised map toward an objective's own target.

    The idea: if two players are "similar" for a rank-chaser, they should be
    similar *in the directions that predict the thing a rank-chaser cares about*.
    Regressing the objective's target on the embedding and keeping the leading
    directions of the fitted map produces exactly that space.

    Closed form rather than iterative, seeded nowhere, so it is deterministic.

    Returns ``None`` — and the caller falls back to Approach A — when there are
    too few rows or the target does not vary. **Refusing is the correct
    behaviour**, not a degradation: a projection fitted on a thin fold is noise
    with a matrix behind it, and it would be indistinguishable from a real one in
    every downstream report.
    """
    if vectors.shape[0] < min_rows:
        return None
    finite = np.isfinite(target) & np.all(np.isfinite(vectors), axis=1)
    if int(finite.sum()) < min_rows:
        return None
    x, y = vectors[finite], target[finite]
    if float(np.std(y)) < 1e-9:
        return None

    width = min(n_components, x.shape[1])
    if width < 1:
        return None

    # Ridge solution of the normal equations, then the leading right-singular
    # directions of the resulting coefficient structure.
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    try:
        beta = np.linalg.solve(gram, x.T @ y)
    except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this near-impossible
        return None

    # Build a subspace around the target direction: the regression direction
    # first, then the highest-variance directions orthogonal to it. Clustering in
    # that space groups players who are alike *in ways that matter here*.
    direction = beta / max(float(np.linalg.norm(beta)), 1e-12)
    residual = x - np.outer(x @ direction, direction)
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    basis = [direction]
    for row in vt:
        if len(basis) >= width:
            break
        basis.append(row)
    return np.asarray(np.stack(basis[:width], axis=1), dtype=np.float64)


def fit_clusters(
    frame: pl.DataFrame,
    *,
    objective: ManagerObjective | None = None,
    target: np.ndarray | None = None,
    embedding: EmbeddingModel | None = None,
    k: int | None = None,
    approach: str = "weighted",
    position: str | None = None,
    seed: int = 20260728,
    version: str = CLUSTER_VERSION,
    utility: Mapping[int, float] | None = None,
) -> ClusterModel:
    """Fit clusters on **training rows only**.

    Args:
        frame: Training rows. Never a validation fold.
        objective: When given, the space is conditioned on it. When ``None`` the
            result is the ordinary unsupervised control.
        target: Objective-aligned target, required for ``approach="projection"``.
        embedding: A pre-fitted embedding. Fitted here when omitted — on this
            frame, which is why this frame must be training rows.
        approach: ``"weighted"`` (A) or ``"projection"`` (B). Projection falls
            back to weighted when the data cannot support it.
        k: Cluster count. Chosen by :func:`select_k` when omitted.

    Raises:
        ValueError: when there are too few rows to cluster meaningfully.
    """
    if frame.height < _MIN_ROWS:
        raise ValueError(
            f"{frame.height} rows is too few to cluster; a model fitted on this many "
            "describes these players and not the population"
        )

    space = embedding if embedding is not None else fit_embedding(frame, seed=seed)
    vectors = space.transform(frame)

    projection: np.ndarray | None = None
    weights = np.ones(vectors.shape[1], dtype=np.float64)

    if objective is not None:
        if approach == "projection" and target is not None:
            projection = fit_supervised_projection(vectors, target)
        if projection is not None:
            vectors = vectors @ projection
            weights = np.ones(vectors.shape[1], dtype=np.float64)
        else:
            # Approach A. Weights are declared per *source column*, so they are
            # mapped through the PCA loadings when the space is projected —
            # otherwise a weight meant for `selected_mean_5` would be applied to
            # whichever component happens to sit in that slot.
            column_weights = objective_weights(objective, space.columns)
            weights = _project_weights(column_weights, space)

    selection: ClusterSelection | None = None
    if k is None:
        selection = select_k(vectors * weights, seed=seed, utility=utility)
        k = selection.k

    assignments, centroids = _kmeans(vectors * weights, k, seed=seed)
    del assignments  # fitted centroids are what the model keeps; rows are re-assigned on use

    return ClusterModel(
        version=version,
        embedding=space,
        centroids=centroids,
        weights=weights,
        k=k,
        method="kmeans" if projection is None else "kmeans_projected",
        objective_id=None if objective is None else objective.id,
        projection=projection,
        selection=selection,
        position=position,
        fitted_rows=frame.height,
        seed=seed,
    )


def _project_weights(column_weights: np.ndarray, space: EmbeddingModel) -> np.ndarray:
    """Carry per-column emphasis through the PCA loadings onto the components.

    A component's weight is the loading-magnitude-weighted average of the column
    weights it is built from. Without this step an objective's emphasis on
    ownership would be applied to an arbitrary component and would emphasise
    whatever that component happened to encode.
    """
    if space.components is None:
        return column_weights
    magnitude = np.abs(space.components)
    totals = np.clip(magnitude.sum(axis=1), 1e-12, None)
    return np.asarray((magnitude @ column_weights) / totals, dtype=np.float64)


def assign_rolling(
    model: ClusterModel,
    frame: pl.DataFrame,
    *,
    season_col: str = "label_season",
    gameweek_col: str = "label_gameweek",
    objective_id: str | None = None,
) -> list[PlayerClusterAssignment]:
    """Assign every row, keyed by gameweek, using stored centroids.

    This is what makes clusters dynamic: the same player is assigned separately
    in each gameweek from that gameweek's features, so movement between clusters
    is recorded rather than averaged away.
    """
    if frame.is_empty():
        return []
    ids, memberships, distances = model.assign(frame)
    version = model.model_version()

    codes = frame["player_code"].to_list()
    seasons = frame[season_col].to_list() if season_col in frame.columns else [""] * frame.height
    weeks = frame[gameweek_col].to_list() if gameweek_col in frame.columns else [0] * frame.height

    out: list[PlayerClusterAssignment] = []
    for index in range(frame.height):
        cluster_id = int(ids[index])
        out.append(
            PlayerClusterAssignment(
                player_code=PlayerCode(int(codes[index])),
                gameweek=GameweekId(int(weeks[index] or 0)),
                season=str(seasons[index] or ""),
                cluster_model_version=version,
                cluster_id=cluster_id,
                membership_probability=round(float(memberships[index, cluster_id]), 6),
                distance_to_centroid=round(float(distances[index]), 6),
                objective_id=objective_id or model.objective_id,
            )
        )
    return out


def summarise(
    model: ClusterModel,
    frame: pl.DataFrame,
    *,
    top: int = 5,
) -> list[ClusterSummary]:
    """Describe each cluster by the columns it is extreme on.

    **The statistical description is the output; a label is decoration.** The
    summary carries ``dominant_features`` as (column, standardised centroid)
    pairs computed from the original columns, so a reader can see *why* a cluster
    is what it is. ``label`` is generated from those same numbers and never
    invented — an LLM could later produce a prettier name, but the basis stays
    visible underneath it, which is the difference between a description and a
    story.
    """
    if frame.is_empty():
        return []
    ids, memberships, _ = model.assign(frame)
    standardised = model.embedding.transform(frame)
    if model.embedding.components is not None:
        # Describe in original-column terms by projecting back, so the summary
        # says "high shot threat" rather than "high component 2".
        standardised = standardised @ model.embedding.components
    columns = model.embedding.columns
    version = model.model_version()

    out: list[ClusterSummary] = []
    for cluster_id in range(model.k):
        mask = ids == cluster_id
        size = int(mask.sum())
        if size == 0:
            continue
        centre = standardised[mask].mean(axis=0)
        order = np.argsort(-np.abs(centre))[:top]
        dominant = tuple(
            (columns[int(i)], round(float(centre[int(i)]), 4))
            for i in order
            if abs(float(centre[int(i)])) >= 0.25
        )
        out.append(
            ClusterSummary(
                cluster_model_version=version,
                cluster_id=cluster_id,
                position=model.position,
                objective_id=model.objective_id,
                size=size,
                label=_label(dominant),
                dominant_features=dominant,
                stability=round(float(memberships[mask, cluster_id].mean()), 4),
            )
        )
    return out


#: Readable phrasing for a column at a high or low centroid. Only columns a
#: person would recognise are named; the rest fall back to the column itself,
#: which is ugly and honest.
_PHRASES: Final[dict[str, tuple[str, str]]] = {
    "minutes_mean_5": ("high minutes", "low minutes"),
    "minutes_mean_20": ("consistently played", "rarely played"),
    "starts_mean_5": ("nailed starter", "rotation risk"),
    "appearance_rate_10": ("always involved", "seldom involved"),
    "expected_goals_per90_5": ("high shot quality", "low shot quality"),
    "expected_assists_per90_5": ("high chance creation", "low chance creation"),
    "threat_per90_5": ("high shot volume", "low shot volume"),
    "creativity_per90_5": ("creative", "direct"),
    "total_points_max_5": ("high ceiling", "low ceiling"),
    "total_points_std_10": ("volatile", "consistent"),
    "total_points_mean_5": ("high returning", "low returning"),
    "selected_mean_5": ("highly owned", "low owned"),
    "transfers_balance_mean_5": ("being bought", "being sold"),
    "value_mean_1": ("premium priced", "budget priced"),
    "clean_sheets_mean_5": ("clean-sheet source", "leaky"),
    "saves_per90_5": ("busy keeper", "protected keeper"),
    "expected_goals_conceded_per90_5": ("exposed defence", "solid defence"),
    "defensive_contribution_per90_5": ("high defensive volume", "low defensive volume"),
    "bps_per90_5": ("bonus magnet", "bonus-light"),
    "days_since_last_match": ("well rested", "recently played"),
}


def _label(dominant: Sequence[tuple[str, float]]) -> str:
    """A readable name derived only from the dominant dimensions."""
    if not dominant:
        return "unremarkable"
    words: list[str] = []
    for column, value in dominant[:3]:
        phrases = _PHRASES.get(column)
        if phrases is None:
            words.append(f"{'high' if value > 0 else 'low'} {column}")
        else:
            words.append(phrases[0] if value > 0 else phrases[1])
    return ", ".join(words)


def transition_history(
    assignments: Sequence[PlayerClusterAssignment], player_code: int
) -> tuple[tuple[int, int, int], ...]:
    """One player's cluster over time as ``(gameweek, from_cluster, to_cluster)``.

    Only the moves are returned, not every week. A player who sat in the same
    cluster for twenty weeks generates no rows, which is correct: the transition
    is the event, and a manager scanning for role changes wants the changes.
    """
    ordered = sorted(
        (a for a in assignments if int(a.player_code) == player_code),
        key=lambda a: (a.season, int(a.gameweek)),
    )
    out: list[tuple[int, int, int]] = []
    for previous, current in pairwise(ordered):
        if previous.cluster_id != current.cluster_id:
            out.append((int(current.gameweek), previous.cluster_id, current.cluster_id))
    return tuple(out)
