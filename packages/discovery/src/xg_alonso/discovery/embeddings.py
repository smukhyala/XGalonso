"""Player representations, with a fit/apply split that makes leakage structural.

Two levels, because "who is this player" and "what is he doing right now" are
different questions and a single vector answers neither well:

``player_gameweek``
    Time-varying role and context, as of one deadline. A midfielder pushed
    forward after an injury moves in this space within a week.

``player_static``
    A longer-window profile of the same player, from the same columns over a
    wider window. Stable enough to describe what kind of footballer he is.

**The fit/apply split is the leakage defence, and it is structural rather than
disciplined.** :func:`fit_embedding` returns an immutable
:class:`EmbeddingModel`; the only way to produce vectors for new rows is
:meth:`EmbeddingModel.transform`, which uses the *stored* mean, scale and
loadings. There is no function that scales a frame using its own statistics, so
a validation fold cannot contribute to the standardisation that is applied to it
— not because the caller remembered, but because the API offers no way to do it.

That matters more than it sounds. Fitting a scaler on the full frame and then
splitting is the second most common leak in this kind of work after the rolling
window, and it is invisible in every metric except live performance.

**PCA before clustering, and why the components are kept.** Raw feature vectors
here are heavily correlated — six windows of the same metric — so a Euclidean
distance over them is dominated by whichever metric happens to have the most
windows declared. Projecting first gives each direction of genuine variation one
axis. The loadings are retained so a cluster can be described in terms of the
original columns rather than in terms of "component 3", which explains nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

__all__ = [
    "EMBEDDING_VERSION",
    "EmbeddingModel",
    "EmbeddingSpace",
    "fit_embedding",
    "loading_report",
]

EMBEDDING_VERSION: Final[str] = "embedding_v1"

#: Columns the embedding is built from, in declaration order.
#:
#: **Only columns this repository actually has.** The brief's wish list includes
#: shots in the box, touches in the box, key passes, passes into the penalty area
#: and opponent pressing intensity. None of those is published by the official
#: FPL API, and decision D6 forbids any other source, so they are absent rather
#: than approximated by something that would silently read as zero.
#:
#: What stands in for them: `threat` proxies shot volume and `creativity` chance
#: creation — both Opta-derived and officially published — and `bps` carries the
#: match-event detail that feeds the bonus system.
DEFAULT_EMBEDDING_COLUMNS: Final[tuple[str, ...]] = (
    # Attacking output
    "expected_goals_per90_5",
    "expected_assists_per90_5",
    "expected_goal_involvements_per90_10",
    "threat_per90_5",
    "creativity_per90_5",
    "influence_per90_5",
    # Playing time — the component every other one is conditioned on
    "minutes_mean_5",
    "minutes_mean_20",
    "starts_mean_5",
    "appearance_rate_10",
    "minutes_when_played_mean_5",
    # Defensive and goalkeeping
    "expected_goals_conceded_per90_5",
    "clean_sheets_mean_5",
    "saves_per90_5",
    "defensive_contribution_per90_5",
    # Bonus and overall return
    "bps_per90_5",
    "total_points_mean_5",
    "total_points_max_5",
    "total_points_std_10",
    # Market state — what the crowd believed, which no rate contains
    "selected_mean_5",
    "transfers_balance_mean_5",
    "value_mean_1",
    # Context
    "opponent_conceded_xg_mean_5",
    "is_home",
    "days_since_last_match",
)


@dataclass(frozen=True)
class EmbeddingSpace:
    """Vectors plus the identities they belong to."""

    vectors: np.ndarray
    player_codes: tuple[int, ...]
    positions: tuple[str, ...]
    columns: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.player_codes)

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0


@dataclass(frozen=True)
class EmbeddingModel:
    """A fitted standardisation and projection. Immutable, and the only way to transform.

    Holds the statistics of the rows it was fitted on. Applying it to other rows
    uses those statistics, never theirs — which is what keeps a validation fold
    out of its own scaler.
    """

    version: str
    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray | None
    """PCA loadings, shape ``(n_components, n_features)``. ``None`` means the
    standardised space is used directly."""

    explained_variance_ratio: tuple[float, ...] = ()
    fitted_rows: int = 0
    seed: int = 20260728

    @property
    def dimensions(self) -> int:
        if self.components is not None:
            return int(self.components.shape[0])
        return len(self.columns)

    def fingerprint(self) -> str:
        """Stable identity, so an assignment cannot be read against another model."""
        payload = "|".join(
            [
                self.version,
                ",".join(self.columns),
                str(self.dimensions),
                str(self.fitted_rows),
                f"{float(np.sum(self.mean)):.6f}",
                f"{float(np.sum(self.scale)):.6f}",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        """Project rows into the fitted space using the **stored** statistics.

        Missing columns are treated as entirely absent — filled with the fitted
        mean, which becomes zero after standardisation. That is the honest
        neutral value: it says "no information", not "low value". Filling with a
        literal zero before scaling would place the player at whatever the
        unscaled zero happens to mean, which for ``days_since_last_match`` is
        "played today" and for ``selected`` is "owned by nobody" — two confident
        and wrong assertions.
        """
        raw = _matrix(frame, self.columns)
        centred = np.where(np.isfinite(raw), raw, self.mean)
        standardised = (centred - self.mean) / self.scale
        if self.components is None:
            return np.asarray(standardised, dtype=np.float64)
        return np.asarray(standardised @ self.components.T, dtype=np.float64)

    def embed(self, frame: pl.DataFrame) -> EmbeddingSpace:
        """Transform, carrying identities alongside the vectors."""
        return EmbeddingSpace(
            vectors=self.transform(frame),
            player_codes=tuple(int(v) for v in frame["player_code"].to_list()),
            positions=tuple(
                str(v) for v in (frame["position"].to_list() if "position" in frame.columns else [])
            )
            or ("",) * frame.height,
            columns=self.columns,
        )


def _matrix(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Feature matrix with missing columns and nulls alike as NaN."""
    height = frame.height
    out = np.full((height, len(columns)), np.nan, dtype=np.float64)
    for index, name in enumerate(columns):
        if name not in frame.columns:
            continue
        values = frame[name].cast(pl.Float64, strict=False).to_numpy()
        out[:, index] = values
    return out


def fit_embedding(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str] = DEFAULT_EMBEDDING_COLUMNS,
    n_components: int | None = 8,
    version: str = EMBEDDING_VERSION,
    seed: int = 20260728,
    min_rows: int = 50,
) -> EmbeddingModel:
    """Fit standardisation and an optional PCA projection on **training rows only**.

    Args:
        frame: Training rows. Nothing from a validation fold may appear here.
        columns: Feature columns. Ones the frame lacks are dropped and recorded,
            rather than silently zero-filled — a feature set quietly smaller
            than its version string claims is the failure the catalogue's own
            ``KeyError`` exists to prevent.
        n_components: PCA dimensions. ``None`` keeps the standardised space.
        min_rows: Below this the projection is skipped; a PCA fitted on forty
            rows describes those forty rows.

    Raises:
        ValueError: if no declared column is present at all.
    """
    present = tuple(name for name in columns if name in frame.columns)
    if not present:
        raise ValueError(
            "none of the declared embedding columns exist in this frame; an "
            "embedding over nothing would produce vectors that are all identical "
            f"and cluster into one group. Wanted any of: {list(columns)[:6]}..."
        )

    raw = _matrix(frame, present)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = np.where(np.isfinite(raw), raw, mean)

    scale = filled.std(axis=0)
    # A constant column distinguishes nobody. Setting its scale to one leaves it
    # at exactly zero after centring, so it contributes nothing to any distance
    # rather than producing a division by zero.
    scale = np.where(scale < 1e-9, 1.0, scale)
    standardised = (filled - mean) / scale

    components: np.ndarray | None = None
    explained: tuple[float, ...] = ()
    if n_components is not None and frame.height >= min_rows and len(present) > 1:
        width = min(n_components, len(present), frame.height - 1)
        if width >= 2:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=width, random_state=seed)
            pca.fit(standardised)
            components = np.asarray(pca.components_, dtype=np.float64)
            explained = tuple(float(v) for v in pca.explained_variance_ratio_)

    return EmbeddingModel(
        version=version,
        columns=present,
        mean=mean,
        scale=scale,
        components=components,
        explained_variance_ratio=explained,
        fitted_rows=frame.height,
        seed=seed,
    )


def loading_report(model: EmbeddingModel, *, top: int = 4) -> tuple[tuple[str, ...], ...]:
    """Which original columns dominate each projected axis.

    Returned so a cluster summary can say "high shot threat, low minutes" rather
    than "high component 2". A description in terms of latent axes is not a
    description; it is a restatement of the coordinates.
    """
    if model.components is None:
        return tuple((name,) for name in model.columns)
    out: list[tuple[str, ...]] = []
    for axis in model.components:
        order = np.argsort(-np.abs(axis))[:top]
        out.append(tuple(model.columns[int(i)] for i in order))
    return tuple(out)
