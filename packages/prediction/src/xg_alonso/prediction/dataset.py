"""Build the training dataset: features as of each historical deadline, plus labels.

The subtlety is that a training row must be built from the *manager's* vantage
point, not from a dataset-wide view. For gameweek *N* the features may use only
data available before gameweek *N*'s deadline, and the label is what happened
in gameweek *N* — information that is by construction not yet visible.

Doing this correctly is slow, because the feature window has to be recomputed
per gameweek rather than once over the whole frame. That cost is the price of a
dataset that is not silently poisoned, and the alternative — computing features
once over all history and slicing — is the single most common way fantasy-sports
models come to validate beautifully and lose money.

Labels are **components**, not points, per decision D8. A model trained on point
totals learns a scoring system that changes; a model trained on goals and
minutes does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import polars as pl

from xg_alonso.features.catalogue import build_catalogue, feature_names

__all__ = [
    "COMPONENT_LABELS",
    "TrainingData",
    "build_training_frame",
]

#: FPL deadlines fall roughly 90 minutes before a gameweek's first kickoff.
_DEADLINE_MARGIN = timedelta(minutes=90)

#: What the component models predict. Each is a raw count or indicator that the
#: domain layer knows how to price — never a points total.
COMPONENT_LABELS: tuple[str, ...] = (
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "yellow_cards",
)


@dataclass(frozen=True)
class TrainingData:
    """A point-in-time-safe training frame, with its provenance."""

    frame: pl.DataFrame
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    gameweeks: tuple[int, ...]
    seasons: tuple[str, ...]

    @property
    def rows(self) -> int:
        return self.frame.height


def _deadlines(player_stats: pl.DataFrame) -> pl.DataFrame:
    return (
        player_stats.filter(pl.col("kickoff_time").is_not_null())
        .group_by(["season", "gameweek_id"])
        .agg((pl.col("kickoff_time").min() - _DEADLINE_MARGIN).alias("deadline"))
        .sort(["season", "gameweek_id"])
    )


def build_training_frame(
    player_stats: pl.DataFrame,
    *,
    seasons: Sequence[str] | None = None,
    min_gameweek: int = 4,
    progress: bool = False,
) -> TrainingData:
    """Assemble features-as-of-deadline plus same-gameweek labels.

    Args:
        player_stats: Canonical ``player_gameweek_stats`` across all seasons.
        seasons: Restrict to these seasons. All of them when omitted.
        min_gameweek: Skip the opening gameweeks of each season, where rolling
            windows are nearly empty and rows would mostly encode the prior.
        progress: Print a line per gameweek processed.

    Returns:
        One row per player-gameweek, with catalogue features computed strictly
        before that gameweek's deadline and labels taken from it.
    """
    stats = player_stats
    if seasons is not None:
        stats = stats.filter(pl.col("season").is_in(list(seasons)))
    if stats.is_empty():
        raise ValueError("no rows for the requested seasons")

    deadline_rows = _deadlines(stats)
    labels = [c for c in COMPONENT_LABELS if c in stats.columns]

    built: list[pl.DataFrame] = []
    for row in deadline_rows.iter_rows(named=True):
        gameweek = int(row["gameweek_id"])
        season = str(row["season"])
        if gameweek < min_gameweek:
            continue

        outcomes = stats.filter((pl.col("season") == season) & (pl.col("gameweek_id") == gameweek))
        if outcomes.is_empty():
            continue

        # One entity row per player who actually featured in this gameweek. The
        # cutoff is the deadline, so the features cannot see this week's result.
        entities = (
            outcomes.select("player_code")
            .unique(keep="first", maintain_order=True)
            .with_columns(
                pl.lit(row["deadline"]).alias("prediction_timestamp"),
                pl.lit(season).alias("label_season"),
                pl.lit(gameweek).alias("label_gameweek"),
            )
        )

        features = build_catalogue(entities, player_stats=stats)

        label_frame = outcomes.group_by("player_code").agg(
            [pl.col(c).sum().alias(f"label_{c}") for c in labels]
        )
        # Sort explicitly. A join does not guarantee row order, and an
        # unstable training frame can change how a model bins ties — which
        # would quietly break the reproducibility guarantee everything else
        # here is built on.
        built.append(features.join(label_frame, on="player_code", how="inner").sort("player_code"))

        if progress:
            print(f"  {season} GW{gameweek:<3} {built[-1].height:>4} rows")  # noqa: T201

    if not built:
        raise ValueError("no gameweeks produced training rows")

    frame = pl.concat(built, how="vertical").sort(["label_season", "label_gameweek", "player_code"])
    return TrainingData(
        frame=frame,
        feature_columns=tuple(feature_names()),
        label_columns=tuple(f"label_{c}" for c in labels),
        gameweeks=tuple(sorted({int(g) for g in frame["label_gameweek"].unique()})),
        seasons=tuple(sorted({str(s) for s in frame["label_season"].unique()})),
    )
