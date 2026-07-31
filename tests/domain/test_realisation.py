"""The realisation kernel, and the reconstruction that proves it.

Two kinds of test live here.

The unit tests pin the cases where a plausible-looking implementation is wrong:
floor division rather than a divided mean, the 60-minute boundary, the two
different defensive-contribution thresholds, and the goalkeeper goal that is
worth 10 and not 6.

The reconstruction test is the real gate. It scores every row of the silver
stats table and requires an exact match against ``total_points`` — 113,270 rows
across four seasons. Nothing short of exactness is acceptable, because the
composition engine convolves component distributions through this map and a
one-point systematic error would move every distribution it produces.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xg_alonso.contracts.prediction import Position
from xg_alonso.domain.realisation import (
    RealisedCounts,
    realised_points,
    realised_points_matrix,
)
from xg_alonso.domain.scoring import ScoringRules

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data/fixtures/fpl/bootstrap_static_2026_27.json"
SILVER = ROOT / ".data/silver/player_gameweek_stats.parquet"
HISTORY = ROOT / ".data/silver/players_history.parquet"

#: Every count field, so the tests below construct counts by name rather than
#: by position and a renamed field fails loudly.
COUNT_FIELDS = (
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "bonus",
)


@pytest.fixture(scope="module")
def rules() -> ScoringRules:
    payload: dict[str, Any] = json.loads(FIXTURE.read_text())
    return ScoringRules.from_bootstrap(
        payload, version="2026-27", source_sha256="a" * 64, fetched_at=datetime.now(UTC)
    )


class TestTheAppearanceTerm:
    def test_no_minutes_pays_nothing(self, rules: ScoringRules) -> None:
        assert realised_points(RealisedCounts(minutes=0), Position.MID, rules) == 0

    def test_one_minute_pays_short_play(self, rules: ScoringRules) -> None:
        got = realised_points(RealisedCounts(minutes=1), Position.MID, rules)
        assert got == rules.short_play

    def test_fifty_nine_minutes_is_still_short(self, rules: ScoringRules) -> None:
        """The boundary is ``>= 60``, and off-by-one here is worth a point on
        every substitute in the game."""
        got = realised_points(RealisedCounts(minutes=59), Position.MID, rules)
        assert got == rules.short_play

    def test_sixty_minutes_pays_long_play(self, rules: ScoringRules) -> None:
        got = realised_points(RealisedCounts(minutes=60), Position.MID, rules)
        assert got == rules.long_play


class TestTheFlooredTerms:
    """The two places ``assemble_points`` is a linear approximation."""

    def test_three_conceded_costs_one_deduction_not_one_and_a_half(
        self, rules: ScoringRules
    ) -> None:
        base = realised_points(RealisedCounts(minutes=90), Position.DEF, rules)
        got = realised_points(RealisedCounts(minutes=90, goals_conceded=3), Position.DEF, rules)
        assert got - base == rules.goals_conceded[Position.DEF]

    def test_four_conceded_costs_two_deductions(self, rules: ScoringRules) -> None:
        base = realised_points(RealisedCounts(minutes=90), Position.DEF, rules)
        got = realised_points(RealisedCounts(minutes=90, goals_conceded=4), Position.DEF, rules)
        assert got - base == 2 * rules.goals_conceded[Position.DEF]

    def test_a_midfielder_is_not_charged_for_concessions(self, rules: ScoringRules) -> None:
        base = realised_points(RealisedCounts(minutes=90), Position.MID, rules)
        got = realised_points(RealisedCounts(minutes=90, goals_conceded=4), Position.MID, rules)
        assert got == base

    def test_five_saves_pay_one_point(self, rules: ScoringRules) -> None:
        base = realised_points(RealisedCounts(minutes=90), Position.GKP, rules)
        got = realised_points(RealisedCounts(minutes=90, saves=5), Position.GKP, rules)
        assert got - base == rules.saves

    def test_six_saves_pay_two_points(self, rules: ScoringRules) -> None:
        base = realised_points(RealisedCounts(minutes=90), Position.GKP, rules)
        got = realised_points(RealisedCounts(minutes=90, saves=6), Position.GKP, rules)
        assert got - base == 2 * rules.saves


class TestPositionPricing:
    def test_a_goalkeeper_goal_is_worth_ten(self, rules: ScoringRules) -> None:
        """The exact transcription error ``domain/scoring.py`` exists to prevent.

        Asserted against the pinned payload rather than the literal 10, so this
        tracks the rules rather than restating a number from memory.
        """
        base = realised_points(RealisedCounts(minutes=90), Position.GKP, rules)
        got = realised_points(RealisedCounts(minutes=90, goals_scored=1), Position.GKP, rules)
        assert got - base == rules.goals_scored[Position.GKP]
        assert rules.goals_scored[Position.GKP] == 10

    @pytest.mark.parametrize("position", list(Position))
    def test_a_goal_is_priced_by_position(self, position: Position, rules: ScoringRules) -> None:
        base = realised_points(RealisedCounts(minutes=90), position, rules)
        got = realised_points(RealisedCounts(minutes=90, goals_scored=1), position, rules)
        assert got - base == rules.goals_scored[position]

    def test_a_forward_earns_nothing_for_a_clean_sheet(self, rules: ScoringRules) -> None:
        base = realised_points(RealisedCounts(minutes=90), Position.FWD, rules)
        got = realised_points(RealisedCounts(minutes=90, clean_sheets=1), Position.FWD, rules)
        assert got == base


class TestDefensiveContribution:
    def test_absent_and_zero_are_different_things(self, rules: ScoringRules) -> None:
        """``None`` means the rule did not exist for this row; ``0`` means the
        player recorded no defensive actions. Both score zero here, but they
        must stay distinguishable — collapsing them is how a missing column
        silently becomes a measurement."""
        absent = RealisedCounts(minutes=90, defensive_contribution=None)
        recorded = RealisedCounts(minutes=90, defensive_contribution=0)
        assert absent.defensive_contribution is None
        assert recorded.defensive_contribution == 0
        assert realised_points(absent, Position.DEF, rules) == realised_points(
            recorded, Position.DEF, rules
        )

    def test_a_defender_needs_the_defender_threshold(self, rules: ScoringRules) -> None:
        threshold = rules.defensive_contribution_threshold(Position.DEF)
        base = realised_points(RealisedCounts(minutes=90), Position.DEF, rules)
        short = realised_points(
            RealisedCounts(minutes=90, defensive_contribution=threshold - 1),
            Position.DEF,
            rules,
        )
        met = realised_points(
            RealisedCounts(minutes=90, defensive_contribution=threshold),
            Position.DEF,
            rules,
        )
        assert short == base
        assert met - base == rules.defensive_contribution[Position.DEF]

    def test_a_midfielder_needs_the_higher_threshold(self, rules: ScoringRules) -> None:
        """The defender threshold is lower, so a midfielder on exactly the
        defender's count must earn nothing — the case a single shared threshold
        gets wrong."""
        defender_threshold = rules.defensive_contribution_threshold(Position.DEF)
        outfield_threshold = rules.defensive_contribution_threshold(Position.MID)
        assert outfield_threshold > defender_threshold

        base = realised_points(RealisedCounts(minutes=90), Position.MID, rules)
        at_defender_level = realised_points(
            RealisedCounts(minutes=90, defensive_contribution=defender_threshold),
            Position.MID,
            rules,
        )
        at_own_level = realised_points(
            RealisedCounts(minutes=90, defensive_contribution=outfield_threshold),
            Position.MID,
            rules,
        )
        assert at_defender_level == base
        assert at_own_level - base == rules.defensive_contribution[Position.MID]


class TestNegativeTotals:
    def test_a_total_can_be_negative(self, rules: ScoringRules) -> None:
        """A substitute who is sent off after an own goal. The support of the
        points distribution starts below zero, and an implementation that
        clamps at zero would truncate the left tail."""
        got = realised_points(
            RealisedCounts(minutes=10, red_cards=1, own_goals=1), Position.FWD, rules
        )
        assert got == rules.short_play + rules.red_cards + rules.own_goals
        assert got < 0


class TestTheVectorisedForm:
    def test_it_agrees_with_the_scalar_form_elementwise(self, rules: ScoringRules) -> None:
        """A vectorised reimplementation that drifts from the scalar one would
        be invisible in every metric it feeds, so the agreement is asserted
        rather than assumed."""
        rng = np.random.default_rng(20260727)
        n = 400
        positions = [list(Position)[i % 4] for i in range(n)]
        arrays = {
            "minutes": rng.integers(0, 91, n, dtype=np.int64),
            "goals_scored": rng.integers(0, 4, n, dtype=np.int64),
            "assists": rng.integers(0, 4, n, dtype=np.int64),
            "clean_sheets": rng.integers(0, 2, n, dtype=np.int64),
            "goals_conceded": rng.integers(0, 7, n, dtype=np.int64),
            "saves": rng.integers(0, 11, n, dtype=np.int64),
            "yellow_cards": rng.integers(0, 2, n, dtype=np.int64),
            "red_cards": rng.integers(0, 2, n, dtype=np.int64),
            "own_goals": rng.integers(0, 2, n, dtype=np.int64),
            "penalties_saved": rng.integers(0, 2, n, dtype=np.int64),
            "penalties_missed": rng.integers(0, 2, n, dtype=np.int64),
            "bonus": rng.integers(0, 4, n, dtype=np.int64),
        }
        contribution = rng.integers(0, 20, n, dtype=np.int64)

        vectorised = realised_points_matrix(
            defensive_contribution=contribution,
            positions=positions,
            rules=rules,
            **arrays,
        )

        for i in range(n):
            counts = RealisedCounts(
                minutes=int(arrays["minutes"][i]),
                defensive_contribution=int(contribution[i]),
                **{name: int(arrays[name][i]) for name in COUNT_FIELDS},
            )
            assert int(vectorised[i]) == realised_points(counts, positions[i], rules)

    def test_an_absent_contribution_column_is_honoured(self, rules: ScoringRules) -> None:
        positions = [Position.DEF, Position.MID]
        zeros = np.zeros(2, dtype=np.int64)
        arrays = {name: zeros.copy() for name in COUNT_FIELDS}
        got = realised_points_matrix(
            minutes=np.array([90, 90], dtype=np.int64),
            defensive_contribution=None,
            positions=positions,
            rules=rules,
            **arrays,
        )
        assert list(got) == [rules.long_play, rules.long_play]

    def test_a_length_mismatch_is_rejected(self, rules: ScoringRules) -> None:
        zeros = np.zeros(3, dtype=np.int64)
        with pytest.raises(ValueError, match="positions has 2 entries for 3 rows"):
            realised_points_matrix(
                minutes=zeros,
                defensive_contribution=None,
                positions=[Position.DEF, Position.MID],
                rules=rules,
                **{name: zeros.copy() for name in COUNT_FIELDS},
            )


@pytest.mark.dataset
@pytest.mark.skipif(
    not (SILVER.exists() and HISTORY.exists()),
    reason="requires the local .data silver tables",
)
class TestTheReconstruction:
    """The gate: score reality and require an exact match.

    This also validates every ``VERIFY``-marked field in
    :class:`ScoringThresholds`. FPL does not publish ``saves_per_point``,
    ``goals_conceded_per_deduction``, ``long_play_minutes`` or either
    defensive-contribution threshold, so no drift check can catch a change to
    them — but a wrong value would produce mismatches somewhere in 113,270
    rows, which is the only verification available.
    """

    def test_it_reproduces_total_points_exactly(self, rules: ScoringRules) -> None:
        import polars as pl

        stats = pl.read_parquet(SILVER)
        history = pl.read_parquet(HISTORY).select(
            "player_code",
            "season",
            # The archive labels goalkeepers "GK"; the rest of the system uses "GKP".
            pl.col("position").replace({"GK": Position.GKP.value}).alias("position"),
        )
        frame = stats.join(history, on=["player_code", "season"], how="left")
        assert frame["position"].null_count() == 0, "every row must resolve a position"

        # Scored one season at a time, which is how the kernel documents the
        # absent-column case: the batch is split rather than sentinel-filled.
        scored = 0
        seasons = sorted(set(frame["season"]))
        for season in seasons:
            block = frame.filter(pl.col("season") == season)
            contribution = block["defensive_contribution"]
            positions = [Position(p) for p in block["position"]]
            computed = realised_points_matrix(
                minutes=block["minutes"].to_numpy(),
                defensive_contribution=(
                    None if contribution.null_count() == block.height else contribution.to_numpy()
                ),
                positions=positions,
                rules=rules,
                **{name: block[name].to_numpy() for name in COUNT_FIELDS},
            )
            mismatches = int(np.count_nonzero(computed != block["total_points"].to_numpy()))
            assert mismatches == 0, (
                f"{season}: {mismatches} of {block.height} rows did not reconstruct; "
                "a VERIFY threshold in ScoringThresholds is likely wrong"
            )
            scored += block.height

        assert scored == 113_270
        assert len(seasons) == 4

    def test_the_null_contribution_column_is_exactly_the_pre_rule_seasons(self) -> None:
        """``None`` must mean "the rule did not exist", not "the value is
        missing for this row". If a 2025-26 row were null, filling it with a
        sentinel would silently award or withhold points."""
        import polars as pl

        frame = pl.read_parquet(SILVER, columns=["season", "defensive_contribution"])
        by_season = frame.group_by("season").agg(
            pl.col("defensive_contribution").null_count().alias("nulls"),
            pl.len().alias("rows"),
        )
        for row in by_season.iter_rows(named=True):
            assert row["nulls"] in (0, row["rows"]), (
                f"season {row['season']} is partially null "
                f"({row['nulls']} of {row['rows']}), so absence is not a season-level fact"
            )
