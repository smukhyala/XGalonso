"""The declarative feature catalogue.

This is the Feature Factory proper: features are *declared*, not hand-written.
Adding one is a line of configuration plus an existing generator, which is the
property the design documents call for — "adding a feature requires
configuration and a reusable generator rather than bespoke pipeline code".

**Why generation is constrained.** Crossing every metric with every window and
every aggregation would produce thousands of columns, most of them noise, and
D12 caps the target at 300-700 *quality* candidates precisely to avoid that.
So the expansion here is deliberate rather than exhaustive:

- Rate metrics get per-90 shrinkage, because raw per-90 is meaningless at low
  minutes. Count metrics do not.
- Volatility (``std``) is generated only for metrics where variance is
  interpretable — points and minutes — not for, say, own goals.
- Trend is generated only over windows long enough for a slope to mean
  something.
- Nothing is crossed with anything: interactions are a separate, gated concern
  (see `docs/ml/04_interaction_discovery.md`), and generating them here would
  be exactly the combinatorial explosion the specification warns against.

Every generated feature is point-in-time safe by construction, because every
one goes through the same as-of generators the leakage harness verifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import polars as pl

from xg_alonso.features.generators import rolling_as_of, shrunk_rate_as_of

__all__ = [
    "CATALOGUE_VERSION",
    "FeatureSpec",
    "build_catalogue",
    "catalogue_specs",
    "feature_names",
]

CATALOGUE_VERSION: Final[str] = "catalogue_v1"

GeneratorKind = Literal["rolling", "shrunk_rate"]


@dataclass(frozen=True)
class FeatureSpec:
    """One declared feature. The unit the catalogue is built from."""

    name: str
    generator: GeneratorKind
    source_column: str
    window: int
    aggregation: str = "mean"
    denominator: str | None = None
    prior_strength: float = 3.0
    min_periods: int = 1
    family: str = "player_performance"

    def __post_init__(self) -> None:
        if self.generator == "shrunk_rate" and self.denominator is None:
            raise ValueError(f"{self.name}: a shrunk rate needs a denominator")


# --- what the catalogue is built from ---------------------------------------

#: Windows in appearances. Short windows capture form, long ones capture level;
#: both matter and the evaluation layer decides which earns its place.
_WINDOWS: Final[tuple[int, ...]] = (1, 3, 5, 10, 20)

#: Counting metrics. Rolling means and sums are meaningful; per-90 is not,
#: because a count already scales with the minutes played.
_COUNT_METRICS: Final[tuple[str, ...]] = (
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "yellow_cards",
    "bonus",
    "bps",
    "total_points",
)

#: Rate metrics: quantities worth expressing per 90 minutes, with shrinkage so
#: a 12-minute cameo cannot masquerade as an elite rate.
_RATE_METRICS: Final[tuple[str, ...]] = (
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "bonus",
    "bps",
    "total_points",
    "saves",
)

#: Metrics whose variance is interpretable. A volatile points return is a real
#: signal about a player's risk profile; volatile own goals is noise.
_VOLATILITY_METRICS: Final[tuple[str, ...]] = ("total_points", "minutes", "bps")

#: Windows long enough for a rolling standard deviation to mean anything.
_VOLATILITY_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20)

#: Rate windows. Shorter than three appearances leaves shrinkage doing all the
#: work, which produces a column that is nearly constant across players.
_RATE_WINDOWS: Final[tuple[int, ...]] = (3, 5, 10, 20)


def catalogue_specs() -> list[FeatureSpec]:
    """Every declared feature, generated from the metric and window grids."""
    specs: list[FeatureSpec] = []

    # Rolling means and sums over counting metrics.
    for metric in _COUNT_METRICS:
        for window in _WINDOWS:
            specs.append(
                FeatureSpec(
                    name=f"{metric}_mean_{window}",
                    generator="rolling",
                    source_column=metric,
                    window=window,
                    aggregation="mean",
                    family="player_performance",
                )
            )
            # Sums duplicate means at a fixed window, so only generate them
            # where the total itself is the quantity of interest.
            if metric in {"minutes", "total_points", "bonus"} and window >= 3:
                specs.append(
                    FeatureSpec(
                        name=f"{metric}_sum_{window}",
                        generator="rolling",
                        source_column=metric,
                        window=window,
                        aggregation="sum",
                        family="player_volume",
                    )
                )

    # Shrunk per-90 rates.
    for metric in _RATE_METRICS:
        for window in _RATE_WINDOWS:
            specs.append(
                FeatureSpec(
                    name=f"{metric}_per90_{window}",
                    generator="shrunk_rate",
                    source_column=metric,
                    denominator="minutes",
                    window=window,
                    family="player_rate",
                )
            )

    # Volatility, where variance is interpretable.
    for metric in _VOLATILITY_METRICS:
        for window in _VOLATILITY_WINDOWS:
            specs.append(
                FeatureSpec(
                    name=f"{metric}_std_{window}",
                    generator="rolling",
                    source_column=metric,
                    window=window,
                    aggregation="std",
                    min_periods=3,  # a standard deviation over two points is noise
                    family="player_volatility",
                )
            )

    # Ceiling and floor: the best and worst recent returns. Captures the
    # explosiveness a mean hides — two players averaging 5 are different if one
    # alternates 2 and 8 and the other returns 5 every week.
    for metric in ("total_points", "bps", "minutes"):
        for window in (5, 10):
            specs.append(
                FeatureSpec(
                    name=f"{metric}_max_{window}",
                    generator="rolling",
                    source_column=metric,
                    window=window,
                    aggregation="max",
                    family="player_ceiling",
                )
            )
            specs.append(
                FeatureSpec(
                    name=f"{metric}_min_{window}",
                    generator="rolling",
                    source_column=metric,
                    window=window,
                    aggregation="min",
                    family="player_floor",
                )
            )

    names = [s.name for s in specs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"catalogue generated duplicate feature names: {sorted(duplicates)}")
    return specs


def feature_names() -> list[str]:
    """Every feature name the catalogue produces."""
    return [spec.name for spec in catalogue_specs()]


def build_catalogue(
    entities: pl.DataFrame,
    *,
    player_stats: pl.DataFrame,
    specs: list[FeatureSpec] | None = None,
    prediction_time_col: str = "prediction_timestamp",
) -> pl.DataFrame:
    """Materialise the catalogue for a frame of prediction rows.

    Args:
        entities: One row per prediction, carrying ``player_code`` and a cutoff.
        player_stats: Canonical ``player_gameweek_stats``.
        specs: Which features to build. Defaults to the whole catalogue.
        prediction_time_col: Cutoff column on ``entities``.

    Returns:
        ``entities`` plus one column per spec, in the original row order.

    Raises:
        KeyError: if a spec names a column the source does not have. Silently
            skipping would produce a feature set that is quietly smaller than
            the version string claims.
    """
    chosen = specs if specs is not None else catalogue_specs()
    available = set(player_stats.columns)

    missing = sorted(
        {s.source_column for s in chosen if s.source_column not in available}
        | {s.denominator for s in chosen if s.denominator and s.denominator not in available}
    )
    if missing:
        raise KeyError(
            f"player_stats is missing columns required by the catalogue: {missing}. "
            "A feature set that silently drops features is not the version it claims to be."
        )

    frame = entities
    for spec in chosen:
        if spec.generator == "rolling":
            frame = rolling_as_of(
                frame,
                player_stats,
                entity_keys=["player_code"],
                value_column=spec.source_column,
                window=spec.window,
                aggregation=spec.aggregation,
                output_name=spec.name,
                prediction_time_col=prediction_time_col,
                order_col="kickoff_time",
                min_periods=spec.min_periods,
            )
        else:
            assert spec.denominator is not None  # guaranteed by __post_init__
            frame = shrunk_rate_as_of(
                frame,
                player_stats,
                entity_keys=["player_code"],
                numerator=spec.source_column,
                denominator=spec.denominator,
                window=spec.window,
                prior_strength=spec.prior_strength,
                output_name=spec.name,
                # A fixed prior of zero keeps the feature independent of the
                # rest of the batch. An empirical prior pooled across entities
                # with different cutoffs would leak, which the generator
                # refuses outright.
                prior_value=0.0,
                prediction_time_col=prediction_time_col,
                order_col="kickoff_time",
            )
    return frame
