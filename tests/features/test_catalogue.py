"""Catalogue tests.

The important one is :class:`TestWholeCatalogueIsLeakFree`. Declaring features
in configuration means a new one can be added without anyone writing generator
code — which is the point, and also the risk. So the harness runs over the
*entire* catalogue rather than a sample: a leak introduced by a future
declaration is caught by a test nobody has to remember to write.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from xg_alonso.features.catalogue import (
    CATALOGUE_VERSION,
    FeatureSpec,
    build_catalogue,
    catalogue_specs,
    feature_names,
)
from xg_alonso.features.leakage import (
    assert_detects_leakage,
    assert_no_leakage,
    make_future_records,
)

T0 = datetime(2025, 8, 1, 12, 0, tzinfo=UTC)

#: Every column the catalogue's specs can reference.
_SOURCE_COLUMNS = (
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
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "influence",
    "creativity",
    "threat",
    "ict_index",
)


def _stats(rows: int = 12, players: tuple[int, ...] = (1, 2, 3)) -> pl.DataFrame:
    data: dict[str, list[object]] = {
        "player_code": [],
        "available_time": [],
        "kickoff_time": [],
    }
    for column in _SOURCE_COLUMNS:
        data[column] = []

    for player in players:
        for week in range(rows):
            moment = T0 + timedelta(days=7 * week)
            data["player_code"].append(player)
            data["available_time"].append(moment)
            data["kickoff_time"].append(moment)
            for index, column in enumerate(_SOURCE_COLUMNS):
                # Deterministic and strictly varied. An earlier version used a
                # modulus, which was periodic within the longer windows and made
                # short and long windows coincide — a fixture artefact that
                # looked like a catalogue bug.
                #
                # The per-week slope is scaled by the player rather than added
                # to it. With a shared slope, two players' series differ by a
                # constant, and standard deviation is translation-invariant — so
                # every `*_std_*` column was mathematically identical across
                # players. `test_no_feature_is_constant_across_players` then
                # passed only because catastrophic cancellation left different
                # last bits, which made it architecture-dependent: it passed on
                # arm64 and failed on x86_64 for the same commit. Scaling makes
                # the spread genuinely differ, so the test measures the
                # catalogue instead of the floating-point unit.
                data[column].append(player * 2.0 + week * 0.37 * player + index * 0.11)
    return pl.DataFrame(data).with_columns(
        pl.col("available_time").dt.replace_time_zone("UTC"),
        pl.col("kickoff_time").dt.replace_time_zone("UTC"),
        pl.col("minutes").cast(pl.Float64) * 9,  # realistic minute scale
    )


def _entities(players: tuple[int, ...] = (1, 2, 3), days: int = 200) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_code": list(players),
            "prediction_timestamp": [T0 + timedelta(days=days)] * len(players),
        }
    ).with_columns(pl.col("prediction_timestamp").dt.replace_time_zone("UTC"))


class TestCatalogueShape:
    def test_it_declares_a_substantial_number_of_features(self) -> None:
        assert len(catalogue_specs()) >= 100

    def test_names_are_unique(self) -> None:
        """A duplicate name would silently overwrite an earlier column."""
        names = feature_names()
        assert len(names) == len(set(names))

    def test_every_spec_has_a_family(self) -> None:
        assert all(spec.family for spec in catalogue_specs())

    def test_a_rate_without_a_denominator_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs a denominator"):
            FeatureSpec(name="bad", generator="shrunk_rate", source_column="goals_scored", window=5)

    def test_version_is_declared(self) -> None:
        assert CATALOGUE_VERSION


class TestBuild:
    def test_every_declared_feature_is_materialised(self) -> None:
        built = build_catalogue(_entities(), player_stats=_stats())
        for name in feature_names():
            assert name in built.columns, f"{name} was declared but not built"

    def test_row_order_and_count_are_preserved(self) -> None:
        entities = _entities()
        built = build_catalogue(entities, player_stats=_stats())
        assert built.height == entities.height
        assert built["player_code"].to_list() == entities["player_code"].to_list()

    def test_a_missing_source_column_fails_loudly(self) -> None:
        """Silently dropping features would misrepresent the feature-set version."""
        thin = _stats().drop("bps")
        with pytest.raises(KeyError, match="missing columns required"):
            build_catalogue(_entities(), player_stats=thin)

    def test_a_subset_can_be_built(self) -> None:
        subset = catalogue_specs()[:5]
        built = build_catalogue(_entities(), player_stats=_stats(), specs=subset)
        assert all(s.name in built.columns for s in subset)

    def test_no_feature_is_constant_across_players(self) -> None:
        """A constant column carries no information and should not be shipped."""
        built = build_catalogue(_entities(), player_stats=_stats())
        constant = [
            name
            for name in feature_names()
            if built[name].drop_nulls().len() > 1 and built[name].drop_nulls().n_unique() == 1
        ]
        assert not constant, f"constant features: {constant[:10]}"

    def test_shorter_windows_react_faster_than_longer_ones(self) -> None:
        """A sanity check that the window parameter is actually doing something."""
        stats = _stats(rows=20, players=(1,))
        entities = _entities(players=(1,), days=300)
        built = build_catalogue(entities, player_stats=stats)
        assert built["total_points_mean_1"].item() != built["total_points_mean_20"].item()


@pytest.mark.leakage
class TestWholeCatalogueIsLeakFree:
    """Runs every declared feature through the harness.

    A per-feature test would have to be remembered for each new declaration.
    Sweeping the catalogue means a leaky future feature fails automatically.
    """

    def _builder(self) -> object:
        def builder(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            return build_catalogue(entities, player_stats=source)

        return builder

    def test_no_feature_sees_the_future(self) -> None:
        stats = _stats()
        future = make_future_records(
            stats, after=T0 + timedelta(days=900), entity_keys=["player_code"], rows=6
        )
        assert_no_leakage(
            self._builder(),  # type: ignore[arg-type]
            entities=_entities(),
            source=stats,
            future_records=future,
            compare_columns=feature_names(),
        )

    def test_the_harness_still_has_teeth_on_this_fixture(self) -> None:
        """Negative control: a clean sweep is only meaningful if a leak would fail."""

        def leaky(entities: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
            totals = source.group_by("player_code").agg(
                pl.col("total_points").mean().alias("total_points_mean_5")
            )
            return entities.join(totals, on="player_code", how="left")

        stats = _stats()
        assert_detects_leakage(
            leaky,
            entities=_entities(),
            source=stats,
            future_records=make_future_records(
                stats, after=T0 + timedelta(days=900), entity_keys=["player_code"], rows=6
            ),
        )

    def test_an_earlier_cutoff_yields_different_values(self) -> None:
        """Proves the cutoff is honoured rather than ignored.

        If features were identical across cutoffs, the leakage sweep would pass
        trivially — it would be comparing two computations that ignore time.
        """
        stats = _stats(rows=20)
        early = build_catalogue(_entities(days=40), player_stats=stats)
        late = build_catalogue(_entities(days=300), player_stats=stats)

        differing = [
            name
            for name in feature_names()
            if not early[name].equals(late[name], null_equal=True, check_dtypes=False)
        ]
        assert len(differing) > len(feature_names()) // 2, (
            "most features should change when the cutoff moves; "
            "if they do not, the cutoff is not being applied"
        )
