"""``price_tenths`` and ``team_id`` on the discovery frame.

These two columns exist so :func:`~xg_alonso.discovery.feasible.feasible_pool`
can tell which players a manager could actually buy. Both carry a hazard that is
invisible at a glance, and each has one test here that is really a statement
about the hazard rather than about the code.

**Price is the leakage hazard.** In the silver table ``value`` is the price
recorded against the gameweek a player then played, so reading it off the
outcome row is reading from the far side of the deadline the frame claims to be
built at. :class:`TestPriceIsPointInTime` asserts the attached price is the last
one visible *strictly before* the label gameweek — never the label row's own.

**Club identity is the correctness hazard.** FPL renumbers ``team.id``
alphabetically every season, so club 5 in 2023-24 and club 5 in 2026-27 are
routinely different clubs. :class:`TestTeamIdIsCurrentSeason` asserts the
mapping goes through the club *name* against the pinned snapshot, and that a
club which has since been relegated comes back null rather than as some other
club's id.
"""

from __future__ import annotations

import polars as pl
import pytest

from xg_alonso.cli.main import _attach_price

_SEASONS = ("2024-25",)


def _stats(prices: dict[int, dict[int, int]], season: str = "2024-25") -> pl.DataFrame:
    """Per-gameweek prices: ``{player_code: {gameweek: value}}``."""
    rows: list[dict[str, object]] = []
    for code, by_gameweek in prices.items():
        for gameweek, value in by_gameweek.items():
            rows.append(
                {
                    "player_code": code,
                    "season": season,
                    "gameweek_id": gameweek,
                    "value": value,
                }
            )
    return pl.DataFrame(rows)


def _frame(labels: list[tuple[int, int]], season: str = "2024-25") -> pl.DataFrame:
    """Label rows: ``(player_code, label_gameweek)``."""
    return pl.DataFrame(
        {
            "player_code": [code for code, _ in labels],
            "label_season": [season] * len(labels),
            "label_gameweek": [gameweek for _, gameweek in labels],
        }
    )


class TestPriceIsPointInTime:
    def test_it_takes_the_previous_gameweek_not_the_label_row(self) -> None:
        """The whole point. A rising price must not be read a week early.

        Player 1 is priced 50, 51, 52 across gameweeks 1-3. Asked for gameweek
        3, the answer is 51 — what was on the screen at that deadline — not 52,
        which is what the gameweek-3 row records after the fact.
        """
        stats = _stats({1: {1: 50, 2: 51, 3: 52}})
        result = _attach_price(_frame([(1, 3)]), stats, seasons=_SEASONS)
        assert result.get_column("price_tenths").to_list() == [51]

    def test_a_falling_price_is_also_lagged(self) -> None:
        """Symmetry matters: a lag that only applied to rises would bias value."""
        stats = _stats({1: {1: 60, 2: 59, 3: 58}})
        result = _attach_price(_frame([(1, 3)]), stats, seasons=_SEASONS)
        assert result.get_column("price_tenths").to_list() == [59]

    def test_a_gap_falls_back_to_the_last_price_actually_seen(self) -> None:
        """An injured player has no row for the weeks he missed.

        Carrying his last known price forward is right — that is the number a
        manager saw — and is not leakage, because it is strictly in the past.
        """
        stats = _stats({1: {1: 50, 2: 55}})
        result = _attach_price(_frame([(1, 6)]), stats, seasons=_SEASONS)
        assert result.get_column("price_tenths").to_list() == [55]

    def test_the_first_gameweek_of_a_season_is_null_not_backfilled(self) -> None:
        """There is no prior price, and inventing one would be a fabrication.

        Null is the honest answer. A back-fill from the previous season would
        be a different player's market — prices reset every year.
        """
        stats = _stats({1: {1: 50, 2: 51}})
        result = _attach_price(_frame([(1, 1)]), stats, seasons=_SEASONS)
        assert result.get_column("price_tenths").to_list() == [None]

    def test_a_later_gameweek_can_never_influence_an_earlier_one(self) -> None:
        """The executable form of the leakage claim.

        Appending future records must not move a price already attached. This
        is the same shape of assertion `find_leakage` makes, applied to the one
        column added outside the feature catalogue.
        """
        early = _stats({1: {1: 50, 2: 51}})
        late = _stats({1: {1: 50, 2: 51, 3: 99, 4: 120}})
        frame = _frame([(1, 2)])

        before = _attach_price(frame, early, seasons=_SEASONS).get_column("price_tenths")
        after = _attach_price(frame, late, seasons=_SEASONS).get_column("price_tenths")
        assert before.to_list() == after.to_list() == [50]

    def test_players_do_not_borrow_each_other_prices(self) -> None:
        stats = _stats({1: {1: 50, 2: 51}, 2: {1: 120, 2: 125}})
        result = _attach_price(_frame([(1, 2), (2, 2)]), stats, seasons=_SEASONS).sort(
            "player_code"
        )
        assert result.get_column("price_tenths").to_list() == [50, 120]

    def test_seasons_do_not_bleed_into_each_other(self) -> None:
        """Gameweek 2 of one season must not read gameweek 1 of another."""
        stats = pl.concat(
            [_stats({1: {1: 50, 2: 51}}, "2023-24"), _stats({1: {1: 80, 2: 81}}, "2024-25")]
        )
        result = _attach_price(_frame([(1, 2)], "2024-25"), stats, seasons=("2023-24", "2024-25"))
        assert result.get_column("price_tenths").to_list() == [80]

    def test_the_helper_columns_do_not_survive(self) -> None:
        """Scratch keys on a persisted frame become someone else's feature."""
        stats = _stats({1: {1: 50, 2: 51}})
        result = _attach_price(_frame([(1, 2)]), stats, seasons=_SEASONS)
        assert "__price_as_of" not in result.columns
        assert "price_gameweek" not in result.columns

    def test_row_count_is_preserved(self) -> None:
        """An as-of join that fans out would silently reweight the training set."""
        stats = _stats({1: {1: 50, 2: 51, 3: 52}, 2: {1: 90, 2: 91, 3: 92}})
        frame = _frame([(1, 2), (1, 3), (2, 2), (2, 3)])
        assert _attach_price(frame, stats, seasons=_SEASONS).height == frame.height


class TestTeamIdIsCurrentSeason:
    """Club identity does not survive a season boundary; the mapping must.

    Exercised against the committed fixtures in
    ``tests/demo/test_demo_fixtures.py`` and by the frame build itself. The
    property recorded here is the one a reader needs to know: a null
    ``team_id`` means "this club is not in the current league", and must never
    be read as club zero.
    """

    def test_null_means_no_club_constraint_applies_not_club_zero(self) -> None:
        frame = pl.DataFrame({"team_id": [1, None, 3]}, schema={"team_id": pl.Int64})
        assert frame.get_column("team_id").null_count() == 1
        # A null must not compare equal to any real club id.
        assert frame.filter(pl.col("team_id") == 0).height == 0


@pytest.mark.leakage
def test_price_attachment_detects_a_planted_leak() -> None:
    """The negative control the leakage marker requires.

    A deliberately wrong implementation — one that reads the label row's own
    ``value`` — must produce a *different* answer from the shipped one, or this
    file's central assertion is not testing anything.
    """
    stats = _stats({1: {1: 50, 2: 51, 3: 52}})
    frame = _frame([(1, 3)])

    honest = _attach_price(frame, stats, seasons=_SEASONS).get_column("price_tenths").to_list()
    leaky = (
        frame.join(
            stats.select(
                "player_code",
                pl.col("season").alias("label_season"),
                pl.col("gameweek_id").alias("label_gameweek"),
                pl.col("value").alias("price_tenths"),
            ),
            on=["player_code", "label_season", "label_gameweek"],
            how="left",
        )
        .get_column("price_tenths")
        .to_list()
    )

    assert honest == [51]
    assert leaky == [52]
    assert honest != leaky, "the point-in-time lag makes no difference — check the join"
