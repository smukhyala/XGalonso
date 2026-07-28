"""Player archetypes: who a player resembles, and what kind of player he is.

**The question this answers.** A percentile says where a player sits among
everyone in his position. That is the right baseline for "is he good", and the
wrong one for "is he the right pick". Every centre-back is measured against
every other centre-back, including ball-playing defenders in possession-heavy
sides who are barely doing the same job. What a manager wants to know is *which
kind* of defender this is, who else is that kind, and why this one is the pick
among them.

**What the clusters are built on.** Style and quality together, by decision.
That makes a cluster a tier-within-a-type — "high-output attacking full-back"
rather than merely "attacking full-back" — and it has one consequence worth
stating plainly, because it constrains how the output may honestly be used:

    A cluster is partly defined by how good its members are. So "he is the best
    in his cluster" is close to circular and is never claimed anywhere in this
    codebase.

What the clusters *are* used for is what they support without circularity:

1. **Comps.** Nearest neighbours inside the space — "most similar to X, Y, Z".
   This is a similarity claim, and similarity is exactly what the space encodes.
2. **Archetype labels.** A readable name for the region a player occupies,
   derived from which dimensions the cluster is extreme on.
3. **Cross-archetype comparison at a price point.** "For £5.5m you can have this
   archetype or that one." The comparison runs *between* clusters, so it never
   asks a cluster to justify its own membership.

Percentile baselines elsewhere in the system stay scoped to position, not to
cluster, for the same reason.

**Team playstyle is part of the space.** A defender's return depends heavily on
whether his side dominates matches or defends deep, so team attacking and
defensive strength are dimensions here — derived by inverting the stats table,
the same trick :mod:`~xg_alonso.features.opponent` uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

__all__ = [
    "ARCHETYPE_VERSION",
    "STYLE_AXES",
    "Archetype",
    "ArchetypeModel",
    "PlayerArchetype",
    "build_archetypes",
]

ARCHETYPE_VERSION: Final[str] = "archetype_v1"

#: How many archetypes per position.
#:
#: Four rather than a tuned number. Each position genuinely has a handful of
#: recognisable roles — a keeper is busy or protected, a defender attacks or
#: does not — and a k chosen by silhouette score would move between runs as the
#: season progresses, so the labels would not be stable enough to build prose
#: on. Stability is worth more here than a marginally tighter fit.
_CLUSTERS_PER_POSITION: Final[int] = 4

#: Minimum players in a position before clustering is attempted at all.
_MIN_POPULATION: Final[int] = 24

#: Minimum relative spread before an axis is allowed to define an archetype.
#:
#: **Standardising destroys the evidence that a population varies at all.** After
#: z-scoring, a group where every player sits within a hair of the mean looks
#: exactly like one with genuine spread — same unit variance, same centroids —
#: so the labeller happily called one cluster "attacking" and another
#: "stay-at-home" over differences that were pure noise. Measured on the raw
#: values *before* scaling, this is the check that a difference exists to name.
_MIN_RELATIVE_SPREAD: Final[float] = 0.05


@dataclass(frozen=True)
class StyleAxis:
    """One dimension of the archetype space."""

    name: str
    label: str
    sources: tuple[str, ...]
    high: str
    """What a player at the top of this axis is, in words."""
    low: str
    """What a player at the bottom of this axis is, in words."""


#: The axes, per position. Deliberately few: with four clusters and a few hundred
#: players, a dozen dimensions would put every player equidistant from every
#: other and the labels would mean nothing.
STYLE_AXES: Final[dict[str, tuple[StyleAxis, ...]]] = {
    "GKP": (
        StyleAxis(
            name="saves",
            label="save volume",
            sources=("saves_per90_5",),
            high="busy",
            low="protected",
        ),
        StyleAxis(
            name="defence",
            label="defensive solidity",
            sources=("expected_goals_conceded_per90_5",),
            high="exposed",
            low="solid",
        ),
        StyleAxis(
            name="bonus",
            label="bonus profile",
            sources=("bps_per90_5",),
            high="bonus-magnet",
            low="quiet",
        ),
        StyleAxis(
            name="output",
            label="points output",
            sources=("total_points_mean_5",),
            high="high-returning",
            low="low-returning",
        ),
    ),
    "DEF": (
        StyleAxis(
            name="attacking",
            label="attacking involvement",
            sources=("threat_per90_5",),
            high="attacking",
            low="stay-at-home",
        ),
        StyleAxis(
            name="creating",
            label="chance creation",
            sources=("creativity_per90_5",),
            high="creative",
            low="direct",
        ),
        StyleAxis(
            name="defending",
            label="defensive workload",
            sources=("defensive_contribution_per90_5",),
            high="high-volume",
            low="low-volume",
        ),
        StyleAxis(
            name="solidity",
            label="defensive solidity",
            sources=("expected_goals_conceded_per90_5",),
            high="exposed",
            low="solid",
        ),
        StyleAxis(
            name="output",
            label="points output",
            sources=("total_points_mean_5",),
            high="high-returning",
            low="low-returning",
        ),
    ),
    "MID": (
        StyleAxis(
            name="scoring",
            label="goal threat",
            sources=("expected_goals_per90_5", "xg_per90_shrunk"),
            high="goalscoring",
            low="non-scoring",
        ),
        StyleAxis(
            name="creating",
            label="chance creation",
            sources=("expected_assists_per90_5", "xa_per90_shrunk"),
            high="creative",
            low="non-creative",
        ),
        StyleAxis(
            name="defending",
            label="defensive workload",
            sources=("defensive_contribution_per90_5",),
            high="defensive",
            low="advanced",
        ),
        StyleAxis(
            name="output",
            label="points output",
            sources=("total_points_mean_5",),
            high="high-returning",
            low="low-returning",
        ),
    ),
    "FWD": (
        StyleAxis(
            name="scoring",
            label="goal threat",
            sources=("expected_goals_per90_5", "xg_per90_shrunk"),
            high="poacher",
            low="non-scoring",
        ),
        StyleAxis(
            name="creating",
            label="chance creation",
            sources=("expected_assists_per90_5", "xa_per90_shrunk"),
            high="link-up",
            low="pure-finisher",
        ),
        StyleAxis(
            name="volume",
            label="shot volume",
            sources=("threat_per90_5",),
            high="high-volume",
            low="low-volume",
        ),
        StyleAxis(
            name="output",
            label="points output",
            sources=("total_points_mean_5",),
            high="high-returning",
            low="low-returning",
        ),
    ),
}


@dataclass(frozen=True)
class Archetype:
    """One cluster: a kind of player, named from what makes it extreme."""

    position: str
    index: int
    label: str
    """A readable name, e.g. "attacking, high-returning full-back"."""

    size: int
    centroid: dict[str, float]
    """Mean of each axis, in the standardised space. Zero is positional average."""

    defining: tuple[str, ...]
    """Axis names this cluster is unusual on, strongest first."""

    def describe(self) -> str:
        """One sentence a person can read."""
        if not self.defining:
            return f"A typical {self.position}, unremarkable on every axis measured."
        return self.label


@dataclass(frozen=True)
class PlayerArchetype:
    """One player's placement, plus who he most resembles."""

    player_code: int
    position: str
    archetype: int
    distance: float
    """Distance to his own centroid. Large means he sits between archetypes."""

    comparables: tuple[int, ...]
    """Nearest players in the space, closest first. Excludes himself."""


@dataclass(frozen=True)
class ArchetypeModel:
    """Fitted archetypes and every player's placement."""

    version: str
    archetypes: tuple[Archetype, ...]
    players: tuple[PlayerArchetype, ...]

    def for_player(self, player_code: int) -> PlayerArchetype | None:
        for entry in self.players:
            if entry.player_code == player_code:
                return entry
        return None

    def archetype_of(self, player_code: int) -> Archetype | None:
        placement = self.for_player(player_code)
        if placement is None:
            return None
        for archetype in self.archetypes:
            if archetype.position == placement.position and archetype.index == placement.archetype:
                return archetype
        return None

    def members(self, position: str, index: int) -> tuple[int, ...]:
        return tuple(
            entry.player_code
            for entry in self.players
            if entry.position == position and entry.archetype == index
        )


def _resolve(frame: pl.DataFrame, axis: StyleAxis) -> np.ndarray | None:
    """An axis's values, from the first source column present."""
    for name in axis.sources:
        if name in frame.columns:
            series = frame[name].cast(pl.Float64, strict=False).fill_null(0.0)
            return np.asarray(series.to_list(), dtype=np.float64)
    return None


def _standardise(matrix: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per column.

    Without this the axis measured in the largest units decides every cluster:
    ``threat`` runs to the hundreds while ``expected_goals_per90`` rarely passes
    one, so an unscaled distance would be threat and nothing else.
    """
    mean = matrix.mean(axis=0)
    spread = matrix.std(axis=0)
    spread[spread == 0.0] = 1.0
    return (matrix - mean) / spread


def _kmeans(
    matrix: np.ndarray, clusters: int, *, seed: int, iterations: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with k-means++ seeding.

    Written out rather than imported so the whole pipeline keeps one numeric
    dependency chain and, more usefully, so the seeding is explicit: archetype
    *labels* end up in user-facing prose, and a run-to-run reshuffle would
    rename every player's archetype between page loads.
    """
    generator = np.random.default_rng(seed)
    rows = matrix.shape[0]

    # k-means++: first centre uniform, the rest weighted by squared distance.
    centres = [matrix[generator.integers(rows)]]
    for _ in range(clusters - 1):
        gaps = np.min(
            np.stack([np.sum((matrix - centre) ** 2, axis=1) for centre in centres]), axis=0
        )
        total = gaps.sum()
        if total <= 0:
            centres.append(matrix[generator.integers(rows)])
            continue
        centres.append(matrix[generator.choice(rows, p=gaps / total)])

    centroids = np.stack(centres)
    assignments = np.zeros(rows, dtype=np.int64)

    for _ in range(iterations):
        distances = np.stack(
            [np.sum((matrix - centroid) ** 2, axis=1) for centroid in centroids], axis=1
        )
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        for index in range(clusters):
            members = matrix[assignments == index]
            if members.size:
                centroids[index] = members.mean(axis=0)

    return assignments, centroids


#: How far from positional average a centroid must sit before the axis is named.
#: Below this the cluster is not actually distinguished on that axis, and saying
#: so would put a label on a difference that is not there.
_DEFINING_THRESHOLD: Final[float] = 0.45


def _label(
    position: str,
    centroid: np.ndarray,
    axes: tuple[StyleAxis, ...],
    *,
    words_used: int = 2,
) -> tuple[str, tuple[str, ...]]:
    """Name a cluster from the axes it is extreme on."""
    ranked = sorted(range(len(axes)), key=lambda index: -abs(float(centroid[index])))
    defining = [index for index in ranked if abs(float(centroid[index])) >= _DEFINING_THRESHOLD]

    if not defining:
        return f"typical {position}", ()

    words = [
        axes[index].high if centroid[index] > 0 else axes[index].low
        for index in defining[:words_used]
    ]
    return f"{', '.join(words)} {position}", tuple(axes[index].name for index in defining)


def _unique_labels(
    position: str, centroids: np.ndarray, axes: tuple[StyleAxis, ...]
) -> list[tuple[str, tuple[str, ...]]]:
    """Label every cluster in a position, keeping the names distinct.

    Two clusters can be extreme on the same top two axes and still be different
    places in the space — the first run produced two goalkeeper archetypes both
    called "high-returning, bonus-magnet GKP", which is useless as a label
    because it cannot tell a reader which one a player is in. Where that
    happens, the name widens to include the next defining axis until the
    collision clears.
    """
    for width in (2, 3, 4):
        labelled = [_label(position, centroid, axes, words_used=width) for centroid in centroids]
        names = [name for name, _ in labelled]
        if len(set(names)) == len(names):
            return labelled

    # Still colliding after every axis has been named: the clusters genuinely
    # sit in the same region, so number them rather than shipping duplicates.
    labelled = [_label(position, centroid, axes, words_used=4) for centroid in centroids]
    seen: dict[str, int] = {}
    disambiguated: list[tuple[str, tuple[str, ...]]] = []
    for name, defining in labelled:
        seen[name] = seen.get(name, 0) + 1
        suffix = "" if seen[name] == 1 else f" ({seen[name]})"
        disambiguated.append((f"{name}{suffix}", defining))
    return disambiguated


def build_archetypes(
    features: pl.DataFrame,
    *,
    positions: list[str],
    player_codes: list[int],
    clusters: int = _CLUSTERS_PER_POSITION,
    comparables: int = 4,
    seed: int = 20260728,
) -> ArchetypeModel:
    """Cluster players into archetypes, one clustering per position.

    Args:
        features: A feature frame, one row per player.
        positions: Each row's position.
        player_codes: Each row's player code.
        clusters: Archetypes per position.
        comparables: How many nearest neighbours to record per player.
        seed: Fixed, so archetype labels are stable between runs.

    Returns:
        Fitted archetypes and every player's placement. Positions with too few
        players are skipped entirely rather than clustered into noise — an
        archetype fitted on nine players describes those nine and nothing else.
    """
    if not (len(positions) == len(player_codes) == features.height):
        raise ValueError(
            "positions, player_codes and the feature frame must be the same length; "
            "a mismatch would attribute one player's style to another"
        )

    all_archetypes: list[Archetype] = []
    all_players: list[PlayerArchetype] = []

    for position, axes in STYLE_AXES.items():
        indices = [index for index, value in enumerate(positions) if value == position]
        if len(indices) < _MIN_POPULATION:
            continue

        subset = features[indices]
        columns: list[np.ndarray] = []
        used: list[StyleAxis] = []
        for axis in axes:
            values = _resolve(subset, axis)
            if values is None:
                continue
            spread = float(np.std(values))
            scale = abs(float(np.mean(values))) + 1e-9
            if spread / scale < _MIN_RELATIVE_SPREAD:
                # Everyone is the same on this axis, so it distinguishes nobody.
                continue
            columns.append(values)
            used.append(axis)

        # One usable axis is a ranking, not a clustering.
        if len(used) < 2:
            continue

        matrix = _standardise(np.stack(columns, axis=1))
        k = min(clusters, len(indices) // 6) or 1
        assignments, centroids = _kmeans(matrix, k, seed=seed)

        labels = _unique_labels(position, centroids, tuple(used))
        for index in range(k):
            label, defining = labels[index]
            all_archetypes.append(
                Archetype(
                    position=position,
                    index=index,
                    label=label,
                    size=int(np.sum(assignments == index)),
                    centroid={
                        axis.name: round(float(centroids[index][slot]), 4)
                        for slot, axis in enumerate(used)
                    },
                    defining=defining,
                )
            )

        # Comps are nearest neighbours in the same space, regardless of cluster.
        # A player near a boundary is most like the players actually next to him,
        # and forcing comps to share his cluster label would hide exactly that.
        for slot, row_index in enumerate(indices):
            gaps = np.sum((matrix - matrix[slot]) ** 2, axis=1)
            gaps[slot] = np.inf
            nearest = np.argsort(gaps)[:comparables]
            all_players.append(
                PlayerArchetype(
                    player_code=player_codes[row_index],
                    position=position,
                    archetype=int(assignments[slot]),
                    distance=round(
                        float(np.sqrt(np.sum((matrix[slot] - centroids[assignments[slot]]) ** 2))),
                        4,
                    ),
                    comparables=tuple(
                        player_codes[indices[int(other)]]
                        for other in nearest
                        if np.isfinite(gaps[int(other)])
                    ),
                )
            )

    return ArchetypeModel(
        version=ARCHETYPE_VERSION,
        archetypes=tuple(all_archetypes),
        players=tuple(all_players),
    )
