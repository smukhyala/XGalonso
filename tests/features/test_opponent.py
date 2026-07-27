"""Opponent-feature tests.

Opponent strength is derived by *inverting* the stats table — what a team
concedes is the sum of what the players facing them produced. That inversion is
easy to get subtly wrong, so it is tested directly rather than inferred from the
fact that the numbers look plausible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from xg_alonso.features.leakage import (
    FeatureBuilder,
    assert_detects_leakage,
    assert_no_leakage,
)
from xg_alonso.features.opponent import (
    OPPONENT_FEATURES,
    build_opponent_features,
    build_opponent_strength,
)

T0 = datetime(2025, 8, 1, 12, 0, tzinfo=UTC)


def _stats() -> pl.DataFrame:
    """Two teams over six gameweeks. Team 1 leaks goals; team 2 does not."""
    rows = []
    for week in range(1, 7):
        moment = T0 + timedelta(days=7 * week)
        # Players facing team 1 do well.
        for player in (10, 11):
            rows.append(
                {
                    "player_code": player,
                    "season": "2025-26",
                    "gameweek_id": week,
                    "opponent_team_id": 1,
                    "was_home": True,
                    "expected_goals": 0.5,
                    "goals_scored": 1.0,
                    "total_points": 8.0,
                    "clean_sheets": 0.0,
                    "kickoff_time": moment,
                    "available_time": moment + timedelta(hours=3),
                }
            )
        # Players facing team 2 do badly.
        for player in (20, 21):
            rows.append(
                {
                    "player_code": player,
                    "season": "2025-26",
                    "gameweek_id": week,
                    "opponent_team_id": 2,
                    "was_home": False,
                    "expected_goals": 0.05,
                    "goals_scored": 0.0,
                    "total_points": 2.0,
                    "clean_sheets": 1.0,
                    "kickoff_time": moment,
                    "available_time": moment + timedelta(hours=3),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("kickoff_time").dt.replace_time_zone("UTC"),
        pl.col("available_time").dt.replace_time_zone("UTC"),
    )


def _entities(opponent: int = 1, days: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_code": [99],
            "opponent_team_id": [opponent],
            "was_home": [True],
            "prediction_timestamp": [T0 + timedelta(days=days)],
        }
    ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))


class TestOpponentStrength:
    def test_conceded_is_the_sum_of_what_opponents_produced(self) -> None:
        strength = build_opponent_strength(_stats())
        team_one = strength.filter(pl.col("opponent_team_id") == 1)
        # Two players at 0.5 xG each facing team 1.
        assert team_one["conceded_xg"].to_list() == [1.0] * 6

    def test_a_leaky_team_and_a_solid_team_are_distinguished(self) -> None:
        strength = build_opponent_strength(_stats())
        leaky = strength.filter(pl.col("opponent_team_id") == 1)["conceded_xg"].to_list()
        solid = strength.filter(pl.col("opponent_team_id") == 2)["conceded_xg"].to_list()
        assert min(leaky) > max(solid), "the feature must separate weak defences from strong ones"

    def test_one_row_per_team_gameweek(self) -> None:
        strength = build_opponent_strength(_stats())
        assert strength.height == 12  # 2 teams x 6 gameweeks

    def test_missing_columns_fail_loudly(self) -> None:
        with pytest.raises(KeyError, match="opponent strength"):
            build_opponent_strength(pl.DataFrame({"player_code": [1]}))

    def test_empty_input_yields_the_right_schema(self) -> None:
        empty = build_opponent_strength(_stats().head(0))
        assert empty.is_empty()
        assert "conceded_xg" in empty.columns


class TestOpponentFeatures:
    def test_features_reflect_the_opponent_faced(self) -> None:
        strength = build_opponent_strength(_stats())
        against_leaky = build_opponent_features(_entities(1), opponent_strength=strength)
        against_solid = build_opponent_features(_entities(2), opponent_strength=strength)
        assert (
            against_leaky["opponent_conceded_xg_mean_5"].item()
            > against_solid["opponent_conceded_xg_mean_5"].item()
        )

    def test_every_declared_feature_is_produced(self) -> None:
        strength = build_opponent_strength(_stats())
        built = build_opponent_features(_entities(), opponent_strength=strength)
        for name in OPPONENT_FEATURES:
            assert name in built.columns

    def test_home_flag_is_carried_through(self) -> None:
        strength = build_opponent_strength(_stats())
        built = build_opponent_features(_entities(), opponent_strength=strength)
        assert built["is_home"].item() == 1.0

    def test_a_missing_fixture_yields_null_not_an_average(self) -> None:
        """'No fixture' and 'an average fixture' are different situations."""
        strength = build_opponent_strength(_stats())
        blank = _entities().with_columns(pl.lit(None, dtype=pl.Int64).alias("opponent_team_id"))
        built = build_opponent_features(blank, opponent_strength=strength)
        assert built["opponent_conceded_xg_mean_5"].item() is None

    def test_entities_without_an_opponent_column_are_rejected(self) -> None:
        strength = build_opponent_strength(_stats())
        with pytest.raises(KeyError, match="opponent_team_id"):
            build_opponent_features(
                _entities().drop("opponent_team_id"), opponent_strength=strength
            )

    def test_row_order_is_preserved(self) -> None:
        strength = build_opponent_strength(_stats())
        entities = pl.DataFrame(
            {
                "player_code": [1, 2, 3],
                "opponent_team_id": [2, 1, 2],
                "was_home": [True, False, True],
                "prediction_timestamp": [T0 + timedelta(days=100)] * 3,
            }
        ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))
        built = build_opponent_features(entities, opponent_strength=strength)
        assert built["player_code"].to_list() == [1, 2, 3]


@pytest.mark.leakage
class TestOpponentFeaturesAreLeakFree:
    def _builder(self) -> FeatureBuilder:
        def builder(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            return build_opponent_features(
                entities, opponent_strength=build_opponent_strength(source)
            )

        return builder

    def _future(self) -> pl.DataFrame:
        """Later gameweeks where the leaky team collapses completely.

        The values must be *large*, not zero. An earlier version used zeros,
        which cannot change a sum — so the negative control passed trivially and
        would have certified a genuinely leaky builder as clean.
        """
        rows = []
        for week in range(20, 24):
            moment = T0 + timedelta(days=7 * week)
            for player in (10, 11):
                rows.append(
                    {
                        "player_code": player,
                        "season": "2025-26",
                        "gameweek_id": week,
                        "opponent_team_id": 1,
                        "was_home": True,
                        "expected_goals": 9.0,
                        "goals_scored": 8.0,
                        "total_points": 40.0,
                        "clean_sheets": 0.0,
                        "kickoff_time": moment,
                        "available_time": moment + timedelta(hours=3),
                    }
                )
        return pl.DataFrame(rows, infer_schema_length=None).with_columns(
            pl.col("kickoff_time").dt.replace_time_zone("UTC"),
            pl.col("available_time").dt.replace_time_zone("UTC"),
        )

    def test_future_form_does_not_change_past_opponent_features(self) -> None:
        assert_no_leakage(
            self._builder(),
            entities=_entities(days=60),
            source=_stats(),
            future_records=self._future(),
            compare_columns=[n for n in OPPONENT_FEATURES if n != "is_home"],
        )

    def test_the_harness_catches_a_season_wide_opponent_average(self) -> None:
        """Negative control, and the exact mistake this feature invites.

        Averaging a team's whole season is the natural way to write this and is
        wrong: it uses matches that had not been played at prediction time.
        """

        def leaky(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            season_average = source.group_by("opponent_team_id").agg(
                pl.col("expected_goals").sum().alias("opponent_conceded_xg_mean_5")
            )
            return entities.join(season_average, on="opponent_team_id", how="left")

        assert_detects_leakage(
            leaky,
            entities=_entities(days=60),
            source=_stats(),
            future_records=self._future(),
        )
