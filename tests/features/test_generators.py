"""Generator tests, including the leakage harness applied to the real generators.

The harness tests in ``test_point_in_time.py`` prove the *harness* works. These
prove the generators that ship actually pass it — which is the claim that
matters.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from xg_alonso.features.generators import rolling_as_of, shrunk_rate_as_of
from xg_alonso.features.leakage import (
    assert_detects_leakage,
    assert_no_leakage,
    make_future_records,
)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _history(rows: int = 8) -> pl.DataFrame:
    """A player's match history, one row per week, oldest first."""
    return pl.DataFrame(
        {
            "player_code": [1] * rows + [2] * rows,
            "available_time": [T0 + timedelta(days=7 * i) for i in range(rows)] * 2,
            "kickoff_time": [T0 + timedelta(days=7 * i) for i in range(rows)] * 2,
            "minutes": [90] * rows + [20] * rows,
            "goals": list(range(rows)) + [0] * rows,
            "starts": [1] * rows + [0] * rows,
        }
    ).with_columns(
        pl.col("available_time").dt.replace_time_zone("UTC"),
        pl.col("kickoff_time").dt.replace_time_zone("UTC"),
    )


def _entities(cutoff_days: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {"player_code": [1, 2], "prediction_timestamp": [T0 + timedelta(days=cutoff_days)] * 2}
    ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))


class TestRollingAsOf:
    def test_window_limits_the_records_used(self) -> None:
        """A 3-window over goals 0..7 must use only the last three: 5, 6, 7."""
        result = rolling_as_of(
            _entities(),
            _history(),
            entity_keys=["player_code"],
            value_column="goals",
            window=3,
            aggregation="mean",
            output_name="g",
        )
        assert result.filter(pl.col("player_code") == 1)["g"].item() == pytest.approx(6.0)

    def test_cutoff_excludes_later_records(self) -> None:
        """Same data, earlier cutoff, must give an earlier answer."""
        early = rolling_as_of(
            _entities(cutoff_days=21),  # only weeks 0,1,2,3 visible
            _history(),
            entity_keys=["player_code"],
            value_column="goals",
            window=3,
            aggregation="mean",
            output_name="g",
        )
        assert early.filter(pl.col("player_code") == 1)["g"].item() == pytest.approx(2.0)

    def test_min_periods_yields_null_not_a_fake_number(self) -> None:
        """A mean over one observation is noise, and should say so."""
        result = rolling_as_of(
            _entities(cutoff_days=1),  # only the first record is visible
            _history(),
            entity_keys=["player_code"],
            value_column="goals",
            window=5,
            aggregation="mean",
            output_name="g",
            min_periods=3,
        )
        assert result["g"].to_list() == [None, None]

    def test_row_order_is_preserved(self) -> None:
        entities = _entities()
        result = rolling_as_of(
            entities,
            _history(),
            entity_keys=["player_code"],
            value_column="minutes",
            window=5,
            output_name="m",
        )
        assert result["player_code"].to_list() == entities["player_code"].to_list()

    def test_unknown_aggregation_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported aggregation"):
            rolling_as_of(
                _entities(),
                _history(),
                entity_keys=["player_code"],
                value_column="goals",
                window=3,
                aggregation="kurtosis",
                output_name="g",
            )

    def test_entity_with_no_history_gets_null(self) -> None:
        entities = pl.DataFrame(
            {"player_code": [999], "prediction_timestamp": [T0 + timedelta(days=100)]}
        ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))
        result = rolling_as_of(
            entities,
            _history(),
            entity_keys=["player_code"],
            value_column="goals",
            window=3,
            output_name="g",
        )
        assert result["g"].to_list() == [None]


class TestShrunkRate:
    def test_no_observations_returns_exactly_the_prior(self) -> None:
        """The defining property of shrinkage: zero data means pure prior."""
        entities = pl.DataFrame(
            {"player_code": [999], "prediction_timestamp": [T0 + timedelta(days=100)]}
        ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))
        result = shrunk_rate_as_of(
            entities,
            _history(),
            entity_keys=["player_code"],
            numerator="goals",
            denominator="minutes",
            window=10,
            prior_strength=3.0,
            prior_value=0.5,
            output_name="rate",
        )
        assert result["rate"].item() == pytest.approx(0.5)

    def test_more_data_moves_the_estimate_toward_the_observed_rate(self) -> None:
        """A rich sample must overcome the prior; a thin one must not."""
        history = _history(rows=8)

        def rate_at(days: int) -> float:
            result = shrunk_rate_as_of(
                _entities(cutoff_days=days),
                history,
                entity_keys=["player_code"],
                numerator="goals",
                denominator="minutes",
                window=20,
                prior_strength=3.0,
                prior_value=0.0,
                output_name="rate",
            )
            return float(result.filter(pl.col("player_code") == 1)["rate"].item())

        thin, rich = rate_at(8), rate_at(100)
        assert rich > thin, "accumulating goals must raise the shrunk rate"

    def test_a_cameo_does_not_produce_an_absurd_rate(self) -> None:
        """One goal in 12 minutes is 7.5 per 90 raw. Shrinkage must tame it."""
        cameo = pl.DataFrame(
            {
                "player_code": [7],
                "available_time": [T0],
                "kickoff_time": [T0],
                "minutes": [12],
                "goals": [1],
                "starts": [0],
            }
        ).with_columns(
            pl.col("available_time").dt.replace_time_zone("UTC"),
            pl.col("kickoff_time").dt.replace_time_zone("UTC"),
        )
        entities = pl.DataFrame(
            {"player_code": [7], "prediction_timestamp": [T0 + timedelta(days=30)]}
        ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))

        result = shrunk_rate_as_of(
            entities,
            cameo,
            entity_keys=["player_code"],
            numerator="goals",
            denominator="minutes",
            window=10,
            prior_strength=3.0,
            prior_value=0.3,
            output_name="rate",
        )
        rate = float(result["rate"].item())
        assert rate < 1.0, f"unshrunk rate would be 7.5/90; got {rate}"

    def test_negative_prior_strength_rejected(self) -> None:
        with pytest.raises(ValueError, match="prior_strength"):
            shrunk_rate_as_of(
                _entities(),
                _history(),
                entity_keys=["player_code"],
                numerator="goals",
                denominator="minutes",
                window=5,
                prior_strength=-1.0,
                output_name="rate",
            )


@pytest.mark.leakage
class TestShippedGeneratorsAreLeakFree:
    """The claim that matters: the generators we actually ship pass the harness."""

    def _future(self) -> pl.DataFrame:
        return make_future_records(
            _history(), after=T0 + timedelta(days=500), entity_keys=["player_code"]
        )

    def test_rolling_generator_is_leak_free(self) -> None:
        def builder(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            return rolling_as_of(
                entities,
                source,
                entity_keys=["player_code"],
                value_column="goals",
                window=5,
                output_name="g",
            )

        assert_no_leakage(
            builder,
            entities=_entities(),
            source=_history(),
            future_records=self._future(),
            compare_columns=["g"],
        )

    def test_shrinkage_generator_is_leak_free(self) -> None:
        def builder(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            return shrunk_rate_as_of(
                entities,
                source,
                entity_keys=["player_code"],
                numerator="goals",
                denominator="minutes",
                window=10,
                prior_strength=3.0,
                prior_value=0.3,  # fixed, so the prior itself cannot leak
                output_name="rate",
            )

        assert_no_leakage(
            builder,
            entities=_entities(),
            source=_history(),
            future_records=self._future(),
            compare_columns=["rate"],
        )

    def test_negative_control_still_has_teeth_on_this_data(self) -> None:
        """Guards against a harness that passes because the fixture is inert."""

        def leaky(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            totals = source.group_by("player_code").agg(pl.col("goals").mean().alias("g"))
            return entities.join(totals, on="player_code", how="left")

        assert_detects_leakage(
            leaky,
            entities=_entities(),
            source=_history(),
            future_records=self._future(),
        )

    def test_an_empirical_prior_is_safe_at_a_single_cutoff(self) -> None:
        """A pooled prior computed from already-filtered rows cannot see the future.

        Worth asserting rather than assuming: the prior looks like it ranges over
        the whole dataset, and only the point-in-time filter upstream makes it
        safe.
        """

        def empirical_prior(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            return shrunk_rate_as_of(
                entities,
                source,
                entity_keys=["player_code"],
                numerator="goals",
                denominator="minutes",
                window=10,
                prior_strength=3.0,
                prior_value=None,
                output_name="rate",
            )

        assert_no_leakage(
            empirical_prior,
            entities=_entities(),
            source=_history(),
            future_records=self._future(),
            compare_columns=["rate"],
        )

    def test_an_empirical_prior_across_mixed_cutoffs_is_refused(self) -> None:
        """The subtle case: pooling across cutoffs leaks between rows.

        An entity predicting gameweek 3 would borrow strength from an entity
        predicting gameweek 30. A harness whose entities share one cutoff can
        never surface this, so the generator refuses the input instead.
        """
        mixed = pl.DataFrame(
            {
                "player_code": [1, 2],
                "prediction_timestamp": [T0 + timedelta(days=10), T0 + timedelta(days=400)],
            }
        ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))

        with pytest.raises(ValueError, match="would leak later rows"):
            shrunk_rate_as_of(
                mixed,
                _history(),
                entity_keys=["player_code"],
                numerator="goals",
                denominator="minutes",
                window=10,
                prior_strength=3.0,
                prior_value=None,
                output_name="rate",
            )

    def test_an_explicit_prior_allows_mixed_cutoffs(self) -> None:
        """The escape hatch: a fixed prior cannot leak, so mixing is fine."""
        mixed = pl.DataFrame(
            {
                "player_code": [1, 2],
                "prediction_timestamp": [T0 + timedelta(days=10), T0 + timedelta(days=400)],
            }
        ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))

        result = shrunk_rate_as_of(
            mixed,
            _history(),
            entity_keys=["player_code"],
            numerator="goals",
            denominator="minutes",
            window=10,
            prior_strength=3.0,
            prior_value=0.25,
            output_name="rate",
        )
        assert result.height == 2


class TestTeamGameweekStats:
    """Guards the season-grouping bug: gameweek numbers repeat every year."""

    def _stats(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA, conform

        rows = []
        for season, year in (("2024-25", 2024), ("2025-26", 2025)):
            for player in (1, 2):
                rows.append(
                    {
                        "player_code": player,
                        "gameweek_id": 1,  # same gameweek number in both seasons
                        "season": season,
                        "minutes": 90,
                        "expected_goals": 0.5,
                        "expected_goals_conceded": 1.2,
                        "kickoff_time": datetime(year, 8, 15, tzinfo=UTC),
                        "available_time": datetime(year, 8, 15, tzinfo=UTC),
                    }
                )
        stats = conform(pl.DataFrame(rows, infer_schema_length=None), PLAYER_GAMEWEEK_STATS_SCHEMA)
        players = pl.DataFrame({"player_code": [1, 2], "team_id": [7, 7]})
        return stats, players

    def test_seasons_are_not_merged(self) -> None:
        """Grouping without the season would fold both years into one row."""
        from xg_alonso.features.slice1 import build_team_gameweek_stats

        stats, players = self._stats()
        team = build_team_gameweek_stats(stats, players)

        assert team.height == 2, "one row per (team, season, gameweek), not one per gameweek"
        # Two players at 0.5 xG each is 1.0 for the team — not 2.0 across seasons.
        assert team["team_xg"].to_list() == [1.0, 1.0]

    def test_team_xg_stays_in_a_plausible_range(self) -> None:
        """A merged-season bug shows up as a number several times too large."""
        from xg_alonso.features.slice1 import build_team_gameweek_stats

        stats, players = self._stats()
        team = build_team_gameweek_stats(stats, players)
        assert max(team["team_xg"].to_list()) < 4.0, "team xG per match should not approach 4"

    def test_conceded_is_a_team_value_not_a_squad_sum(self) -> None:
        from xg_alonso.features.slice1 import build_team_gameweek_stats

        stats, players = self._stats()
        team = build_team_gameweek_stats(stats, players)
        assert team["team_xgc"].to_list() == [1.2, 1.2], "conceded must not be summed per player"

    def test_empty_input_yields_the_right_schema(self) -> None:
        from xg_alonso.features.slice1 import build_team_gameweek_stats
        from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA, empty_frame

        team = build_team_gameweek_stats(
            empty_frame(PLAYER_GAMEWEEK_STATS_SCHEMA),
            pl.DataFrame({"player_code": [], "team_id": []}),
        )
        assert team.is_empty()
        assert "season" in team.columns
