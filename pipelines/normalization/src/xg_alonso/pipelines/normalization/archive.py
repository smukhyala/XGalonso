"""Normalize archive CSVs into the canonical per-gameweek stats table.

Two things happen here that matter more than the parsing:

**Identity.** The archive keys rows on ``element``, the per-season id. Joining
those across seasons would silently merge different players. Every row is
therefore translated to the stable ``code`` from that season's
``players_raw.csv``, and rows whose element cannot be resolved are dropped with
a count rather than passed through on a guess.

**Availability.** ``available_time`` is each fixture's kickoff plus a settlement
delay — not the moment the file was downloaded. Stamping archive rows with the
fetch time would make a 2022 match look as though it had been knowable since
2026, which is leakage wearing the costume of bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from xg_alonso.contracts.identifiers import Season
from xg_alonso.pipelines.normalization.schema import (
    PLAYER_GAMEWEEK_STATS_SCHEMA,
    conform,
    empty_frame,
)

__all__ = ["ArchiveNormalizationResult", "normalize_archive_season"]

#: Results are provisional until a match is finalised — bonus points in
#: particular are recomputed after full time.
_SETTLEMENT_HOURS = 3

#: Columns the archive publishes only from 2025/26. Absent seasons get null.
_DEFENSIVE_COLUMNS = ("defensive_contribution",)


@dataclass(frozen=True)
class ArchiveNormalizationResult:
    """Normalized rows, plus what had to be discarded and why."""

    stats: pl.DataFrame
    rows_in: int
    rows_dropped_unresolved_element: int
    rows_dropped_bad_kickoff: int
    has_defensive_contributions: bool

    @property
    def rows_out(self) -> int:
        return self.stats.height


def _element_to_code(players_raw: pl.DataFrame) -> pl.DataFrame:
    """Map this season's element ids to stable player codes."""
    missing = [c for c in ("id", "code") for _ in [0] if c not in players_raw.columns]
    if missing:
        raise KeyError(
            f"players_raw is missing {missing}; without a stable code, rows cannot be "
            "joined across seasons and must not be normalized on a guess"
        )
    return (
        players_raw.select(
            pl.col("id").cast(pl.Int64).alias("element"),
            pl.col("code").cast(pl.Int64).alias("player_code"),
        )
        .unique(subset=["element"], keep="first", maintain_order=True)
        .sort("element")
    )


def normalize_archive_season(
    merged_gw: pl.DataFrame,
    players_raw: pl.DataFrame,
    *,
    season: Season,
) -> ArchiveNormalizationResult:
    """Turn one season's archive files into canonical per-gameweek rows.

    Args:
        merged_gw: Parsed ``merged_gw.csv``.
        players_raw: Parsed ``players_raw.csv``, supplying the id-to-code map.
        season: The season these rows belong to.

    Returns:
        The normalized rows and an account of everything dropped.
    """
    rows_in = merged_gw.height
    if rows_in == 0:
        return ArchiveNormalizationResult(
            stats=empty_frame(PLAYER_GAMEWEEK_STATS_SCHEMA),
            rows_in=0,
            rows_dropped_unresolved_element=0,
            rows_dropped_bad_kickoff=0,
            has_defensive_contributions=False,
        )

    mapping = _element_to_code(players_raw)

    frame = merged_gw.with_columns(pl.col("element").cast(pl.Int64))
    joined = frame.join(mapping, on="element", how="left")
    unresolved = int(joined["player_code"].null_count())
    joined = joined.filter(pl.col("player_code").is_not_null())

    # The archive stores kickoff as an ISO string; a row without one cannot be
    # placed in time and therefore cannot be used point-in-time safely.
    joined = joined.with_columns(
        pl.col("kickoff_time").cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)
    )
    bad_kickoff = int(joined["kickoff_time"].null_count())
    joined = joined.filter(pl.col("kickoff_time").is_not_null())

    has_defensive = all(c in merged_gw.columns for c in _DEFENSIVE_COLUMNS)

    joined = joined.with_columns(
        pl.lit(str(season)).alias("season"),
        pl.col("round").cast(pl.Int64).alias("gameweek_id"),
        pl.col("fixture").cast(pl.Int64).alias("fixture_id"),
        pl.col("opponent_team").cast(pl.Int64).alias("opponent_team_id"),
        (pl.col("kickoff_time") + pl.duration(hours=_SETTLEMENT_HOURS)).alias("available_time"),
    )

    if not has_defensive:
        # Null, never zero. Zero would assert the player made no defensive
        # contributions in a season where the statistic did not exist.
        joined = joined.with_columns(pl.lit(None, dtype=pl.Int64).alias("defensive_contribution"))

    stats = conform(joined, PLAYER_GAMEWEEK_STATS_SCHEMA)

    return ArchiveNormalizationResult(
        stats=stats,
        rows_in=rows_in,
        rows_dropped_unresolved_element=unresolved,
        rows_dropped_bad_kickoff=bad_kickoff,
        has_defensive_contributions=has_defensive,
    )
