"""Career-feature tests.

The motivating case is concrete: a player with four elite seasons and a player
with one must not look alike. Everything below is a way of checking that the
sample size is actually doing work, and that a single season is never reported
as if it were proof.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest

from xg_alonso.features.career import CAREER_FEATURES, build_career_features

T0 = datetime(2022, 8, 1, tzinfo=UTC)
SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")


def _stats(profiles: dict[int, list[tuple[str, int, int]]]) -> pl.DataFrame:
    """Season totals per player as ``(season, minutes, goals)``."""
    rows = []
    for code, seasons in profiles.items():
        for season, minutes, goals in seasons:
            index = SEASONS.index(season)
            moment = T0 + timedelta(days=365 * index + 300)
            rows.append(
                {
                    "player_code": code,
                    "season": season,
                    "minutes": minutes,
                    "goals_scored": goals,
                    "assists": goals // 2,
                    "expected_goal_involvements": float(goals) * 1.1,
                    "total_points": 40 + goals * 5,
                    "bps": 200 + goals * 20,
                    "available_time": moment,
                }
            )
    return pl.DataFrame(rows)


def _entities(codes: tuple[int, ...], at: datetime | None = None) -> pl.DataFrame:
    when = at or (T0 + timedelta(days=365 * 5))
    return pl.DataFrame({"player_code": list(codes), "prediction_timestamp": [when] * len(codes)})


#: A proven elite scorer against a one-season breakout at the same rate.
_PROVEN = 1
_BREAKOUT = 2
_JOURNEYMAN = 3

_PROFILES = {
    _PROVEN: [(season, 2700, 30) for season in SEASONS],
    _BREAKOUT: [("2025-26", 2700, 30)],
    _JOURNEYMAN: [(season, 2700, 4) for season in SEASONS],
}


def _built(at: datetime | None = None) -> pl.DataFrame:
    return build_career_features(
        _entities((_PROVEN, _BREAKOUT, _JOURNEYMAN), at), player_stats=_stats(_PROFILES)
    )


def _row(frame: pl.DataFrame, code: int) -> dict[str, Any]:
    return frame.filter(pl.col("player_code") == code).row(0, named=True)


class TestShape:
    def test_every_declared_feature_is_built(self) -> None:
        built = _built()
        for name in CAREER_FEATURES:
            assert name in built.columns, f"{name} was declared but not built"

    def test_row_order_and_count_are_preserved(self) -> None:
        entities = _entities((_PROVEN, _BREAKOUT, _JOURNEYMAN))
        built = build_career_features(entities, player_stats=_stats(_PROFILES))
        assert built["player_code"].to_list() == entities["player_code"].to_list()

    def test_missing_source_columns_fail_loudly(self) -> None:
        with pytest.raises(KeyError, match="missing columns"):
            build_career_features(
                _entities((_PROVEN,)), player_stats=_stats(_PROFILES).drop("season")
            )


class TestSampleSize:
    def test_the_same_rate_over_more_seasons_survives_shrinkage_better(self) -> None:
        """The whole point. Identical rates, different amounts of evidence."""
        built = _built()
        proven = _row(built, _PROVEN)
        breakout = _row(built, _BREAKOUT)

        assert proven["career_goals_per90"] == pytest.approx(
            breakout["career_goals_per90"], rel=1e-6
        ), "fixture should give both the same raw rate"
        assert proven["career_goals_per90_shrunk"] > breakout["career_goals_per90_shrunk"]

    def test_shrinkage_pulls_toward_the_population_not_to_zero(self) -> None:
        """A breakout is uncertain, not disproven."""
        built = _built()
        breakout = _row(built, _BREAKOUT)
        journeyman = _row(built, _JOURNEYMAN)
        assert breakout["career_goals_per90_shrunk"] > journeyman["career_goals_per90_shrunk"]

    def test_seasons_are_counted(self) -> None:
        built = _built()
        assert _row(built, _PROVEN)["career_seasons"] == 4
        assert _row(built, _BREAKOUT)["career_seasons"] == 1

    def test_only_a_multi_season_player_is_marked_proven(self) -> None:
        built = _built()
        assert _row(built, _PROVEN)["career_is_proven"] == 1.0
        assert _row(built, _BREAKOUT)["career_is_proven"] == 0.0


class TestHonesty:
    def test_consistency_is_unknown_with_one_season(self) -> None:
        """Filling the missing variance with zero scored a single season as
        perfectly consistent — the most flattering reading of the least
        evidence, and exactly the failure these features remove."""
        built = _built()
        assert _row(built, _BREAKOUT)["career_points_consistency"] is None
        assert _row(built, _PROVEN)["career_points_consistency"] is not None

    def test_a_player_with_no_history_has_no_rate(self) -> None:
        """No career is different from a career of zero."""
        built = build_career_features(_entities((99,)), player_stats=_stats(_PROFILES))
        row = _row(built, 99)
        assert row["career_seasons"] == 0
        assert row["career_goals_per90"] is None

    def test_a_thin_season_does_not_count_as_evidence(self) -> None:
        """Two hundred minutes is not a season of proof."""
        thin = {_PROVEN: [("2025-26", 200, 5)]}
        built = build_career_features(_entities((_PROVEN,)), player_stats=_stats(thin))
        assert _row(built, _PROVEN)["career_seasons"] == 0


class TestPointInTime:
    def test_seasons_after_the_cutoff_are_invisible(self) -> None:
        """A career figure must never contain a season not yet played."""
        early = build_career_features(
            _entities((_PROVEN,), at=T0 + timedelta(days=400)),
            player_stats=_stats(_PROFILES),
        )
        late = _built()
        assert _row(early, _PROVEN)["career_seasons"] < _row(late, _PROVEN)["career_seasons"]

    def test_the_rate_is_minutes_weighted_not_a_mean_of_seasons(self) -> None:
        """A 200-minute season should not weigh as much as a 3000-minute one."""
        uneven = {
            _PROVEN: [("2022-23", 2700, 30), ("2023-24", 950, 1)],
        }
        built = build_career_features(_entities((_PROVEN,)), player_stats=_stats(uneven))
        rate = _row(built, _PROVEN)["career_goals_per90"]
        pooled = (30 + 1) / (2700 + 950) * 90.0
        naive_mean = ((30 / 2700) + (1 / 950)) / 2 * 90.0
        assert rate == pytest.approx(pooled, rel=1e-6)
        assert rate != pytest.approx(naive_mean, rel=1e-3)
