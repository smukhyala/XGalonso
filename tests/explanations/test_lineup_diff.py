"""Lineup comparison tests.

The property that carries the feature is that the decomposition **accounts for
the whole gap**. A comparison that explains most of a difference and quietly
loses the rest is worse than no comparison, because the missing part is
invisible — and captaincy, which is usually the largest single term, is exactly
the part a per-player diff drops.
"""

from __future__ import annotations

import pytest

from tests.explanations.conftest import make_prediction
from xg_alonso.contracts.identifiers import PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.domain.rules import PositionRule, SquadRules
from xg_alonso.explanations.lineup_diff import compare_lineups, selection_from_starters
from xg_alonso.optimization.lineup import best_starting_xi

_LAYOUT = (
    [(Position.GKP, code) for code in (1, 2)]
    + [(Position.DEF, code) for code in (3, 4, 5, 6, 7)]
    + [(Position.MID, code) for code in (8, 9, 10, 11, 12)]
    + [(Position.FWD, code) for code in (13, 14, 15)]
)


def _rules() -> SquadRules:
    return SquadRules(
        version="test",
        source_sha256="a" * 64,
        squad_size=15,
        starting_size=11,
        max_per_club=3,
        total_budget=TenthsOfMillion(1000),
        sell_on_fee=0.5,
        sell_at_purchase_price=False,
        max_extra_free_transfers=4,
        transfers_cap=20,
        positions=(
            PositionRule(position=Position.GKP, squad_select=2, min_play=1, max_play=1),
            PositionRule(position=Position.DEF, squad_select=5, min_play=3, max_play=5),
            PositionRule(position=Position.MID, squad_select=5, min_play=2, max_play=5),
            PositionRule(position=Position.FWD, squad_select=3, min_play=1, max_play=3),
        ),
    )


def _swap_within_position(ours, position: Position):  # type: ignore[no-untyped-def]
    """A legal one-for-one change: a starter out, a bench player of the same
    position in.

    Swapping across positions produces an *illegal* eleven — two goalkeepers, or
    two defenders — and an illegal eleven can outscore the optimum, because the
    optimum is only optimal among legal shapes. An earlier version of these
    tests did exactly that and read as the solver being beaten.
    """
    dropped = next((p for p in ours.starters if p.position is position), None)
    incoming = next((p for p in ours.bench if p.position is position), None)
    return dropped, incoming


def _position_with_a_bench_option(ours) -> Position:  # type: ignore[no-untyped-def]
    """A position that has both a starter and a substitute, so a legal swap exists."""
    for position in Position:
        dropped, incoming = _swap_within_position(ours, position)
        if dropped is not None and incoming is not None:
            return position
    raise AssertionError("fixture has no legal like-for-like swap")


def _squad():  # type: ignore[no-untyped-def]
    picks, predictions = [], {}
    for index, (position, code) in enumerate(_LAYOUT):
        picks.append(
            SquadPick(
                player_code=PlayerCode(code),
                position=position,
                team_id=TeamId(1 + code % 8),
                purchase_price=TenthsOfMillion(50),
                current_price=TenthsOfMillion(50),
                selling_price=TenthsOfMillion(50),
                squad_slot=index + 1,
            )
        )
        # Output falls with code, so the optimal XI is knowable by hand.
        predictions[PlayerCode(code)] = make_prediction(
            code=code, position=position, goals=0.5 - code * 0.02
        )
    return picks, predictions


class TestDecomposition:
    def test_the_parts_account_for_the_whole_gap(self) -> None:
        """Swaps plus captaincy plus shape must equal the real difference."""
        picks, predictions = _squad()
        ours = best_starting_xi(picks, predictions, _rules())
        # A deliberately worse eleven: bench the best forward, start a spare keeper's
        # outfield alternative instead.
        position = _position_with_a_bench_option(ours)
        dropped, incoming = _swap_within_position(ours, position)
        yours_codes = [p.player_code for p in ours.starters]
        yours_codes[yours_codes.index(dropped.player_code)] = incoming.player_code
        yours = selection_from_starters(yours_codes, picks, predictions)

        comparison = compare_lineups(yours, ours, predictions)
        parts = comparison.swap_delta + comparison.captain_delta + comparison.shape_delta
        assert parts == pytest.approx(comparison.total_delta, abs=1e-9)

    def test_an_identical_eleven_shows_no_difference(self) -> None:
        picks, predictions = _squad()
        ours = best_starting_xi(picks, predictions, _rules())
        mine = selection_from_starters(
            [p.player_code for p in ours.starters], picks, predictions, captain=ours.captain
        )
        comparison = compare_lineups(mine, ours, predictions)
        assert comparison.is_identical
        assert comparison.total_delta == pytest.approx(0.0, abs=1e-9)

    def test_the_optimal_eleven_is_never_beaten(self) -> None:
        """If a hand-picked XI outscored the solver's, the solver is broken."""
        picks, predictions = _squad()
        ours = best_starting_xi(picks, predictions, _rules())
        position = _position_with_a_bench_option(ours)
        dropped, incoming = _swap_within_position(ours, position)
        yours_codes = [p.player_code for p in ours.starters]
        yours_codes[yours_codes.index(dropped.player_code)] = incoming.player_code
        yours = selection_from_starters(yours_codes, picks, predictions)
        assert compare_lineups(yours, ours, predictions).total_delta >= -1e-9


class TestSwaps:
    def test_swaps_pair_within_position_where_possible(self) -> None:
        """Pairing a keeper against a striker would read as nonsense."""
        picks, predictions = _squad()
        ours = best_starting_xi(picks, predictions, _rules())
        position = _position_with_a_bench_option(ours)
        dropped, incoming = _swap_within_position(ours, position)
        yours_codes = [p.player_code for p in ours.starters]
        yours_codes[yours_codes.index(dropped.player_code)] = incoming.player_code
        yours = selection_from_starters(yours_codes, picks, predictions)

        comparison = compare_lineups(yours, ours, predictions)
        assert comparison.swaps
        for swap in comparison.swaps:
            assert swap.is_like_for_like
            assert swap.position is position

    def test_a_one_sided_swap_is_flagged_as_a_shape_change(self) -> None:
        picks, predictions = _squad()
        ours = best_starting_xi(picks, predictions, _rules())
        # Drop a midfielder, start a forward: the shape changes rather than a
        # like-for-like substitution occurring.
        yours_codes = [p.player_code for p in ours.starters]
        mid = next(p for p in ours.starters if p.position is Position.MID)
        spare = next((p for p in ours.bench if p.position is not Position.MID), ours.bench[0])
        yours_codes[yours_codes.index(mid.player_code)] = spare.player_code
        yours = selection_from_starters(yours_codes, picks, predictions)

        comparison = compare_lineups(yours, ours, predictions)
        assert any(not swap.is_like_for_like for swap in comparison.swaps)


class TestCaptaincy:
    def test_moving_the_armband_is_priced_separately(self) -> None:
        """It doubles a return, so it can outweigh every substitution combined."""
        picks, predictions = _squad()
        ours = best_starting_xi(picks, predictions, _rules())
        worst = min(ours.starters, key=lambda p: predictions[p.player_code].expected_points)
        yours = selection_from_starters(
            [p.player_code for p in ours.starters], picks, predictions, captain=worst.player_code
        )
        comparison = compare_lineups(yours, ours, predictions)
        assert not comparison.swaps
        assert comparison.captain_delta > 0
        assert comparison.captain_delta == pytest.approx(comparison.total_delta, abs=1e-9)
