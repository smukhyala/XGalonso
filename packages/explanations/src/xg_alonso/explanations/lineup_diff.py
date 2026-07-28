"""Compare two starting elevens and account for the whole gap between them.

**Why a plain per-player diff is not enough.** Two elevens picked from the same
fifteen differ in three ways at once: who plays, what shape they play, and who
wears the armband. A list of swaps explains only the first, and captaincy is
usually the largest single term on the page — it doubles one player's return, so
moving the armband can outweigh every substitution combined.

So the difference is decomposed into terms that **sum to the actual gap**, and
the residual is reported rather than hidden. If the parts do not account for the
whole, the comparison says so instead of presenting a tidy story that is missing
something.

The decomposition:

- **Swaps** — players in one eleven and not the other, at their base value.
- **Captaincy** — the doubled term, which changes independently of the eleven.
- **Shape** — what is left once swaps and captaincy are accounted for. Non-zero
  when the two elevens play different formations, because a formation change
  re-prices a bench slot rather than substituting like for like.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.optimization.lineup import CAPTAIN_MULTIPLIER, XiSelection

__all__ = ["LineupComparison", "PlayerSwap", "compare_lineups"]


@dataclass(frozen=True)
class PlayerSwap:
    """One player entering the proposed eleven, and one leaving it."""

    position: Position
    player_in: PlayerCode | None
    player_out: PlayerCode | None
    points_in: float
    points_out: float

    @property
    def delta(self) -> float:
        return self.points_in - self.points_out

    @property
    def is_like_for_like(self) -> bool:
        """Whether this is a straight swap rather than a shape change.

        A swap with only one side is not a substitution: it is one half of a
        formation change, where an outfield slot moved between positions.
        """
        return self.player_in is not None and self.player_out is not None


@dataclass(frozen=True)
class LineupComparison:
    """Why one eleven outscores another, decomposed so the parts sum to the gap."""

    yours_points: float
    ours_points: float
    swaps: tuple[PlayerSwap, ...]
    captain_delta: float
    yours_captain: PlayerCode | None
    ours_captain: PlayerCode | None
    yours_formation: str
    ours_formation: str

    @property
    def total_delta(self) -> float:
        """The real gap, straight from the two selections."""
        return self.ours_points - self.yours_points

    @property
    def swap_delta(self) -> float:
        return sum(swap.delta for swap in self.swaps)

    @property
    def shape_delta(self) -> float:
        """Whatever the swaps and the armband do not account for.

        Reported rather than absorbed. A decomposition that always adds up
        because its last term is defined as the remainder is not evidence of
        anything; naming it lets a reader see when the story is incomplete.
        """
        return self.total_delta - self.swap_delta - self.captain_delta

    @property
    def is_identical(self) -> bool:
        return not self.swaps and abs(self.captain_delta) < 1e-9

    @property
    def yours_is_better(self) -> bool:
        return self.total_delta < 0


def _base_points(
    code: PlayerCode | None, predictions: Mapping[PlayerCode, PlayerPrediction]
) -> float:
    if code is None:
        return 0.0
    prediction = predictions.get(code)
    return 0.0 if prediction is None else prediction.expected_points


def compare_lineups(
    yours: XiSelection,
    ours: XiSelection,
    predictions: Mapping[PlayerCode, PlayerPrediction],
) -> LineupComparison:
    """Account for the difference between two elevens from the same squad.

    Args:
        yours: The eleven as it stands — a manager's own picks.
        ours: The eleven the optimizer would field.
        predictions: Expected points per player, for pricing each term.

    Returns:
        The comparison, with swaps, captaincy and shape separated.
    """
    yours_codes = [pick.player_code for pick in yours.starters]
    ours_codes = [pick.player_code for pick in ours.starters]

    position_of = {pick.player_code: pick.position for pick in yours.starters + ours.starters}

    entering = [code for code in ours_codes if code not in set(yours_codes)]
    leaving = [code for code in yours_codes if code not in set(ours_codes)]

    # Pair within position where possible, so a swap reads as "this defender for
    # that one" rather than pairing a keeper against a striker by list order.
    swaps: list[PlayerSwap] = []
    remaining_out = list(leaving)
    for code in entering:
        position = position_of.get(code)
        match = next((other for other in remaining_out if position_of.get(other) is position), None)
        if match is not None:
            remaining_out.remove(match)
        swaps.append(
            PlayerSwap(
                position=position or Position.MID,
                player_in=code,
                player_out=match,
                points_in=_base_points(code, predictions),
                points_out=_base_points(match, predictions),
            )
        )

    # Anyone left dropping had no counterpart entering in his position — the
    # other half of a formation change.
    for code in remaining_out:
        swaps.append(
            PlayerSwap(
                position=position_of.get(code) or Position.MID,
                player_in=None,
                player_out=code,
                points_in=0.0,
                points_out=_base_points(code, predictions),
            )
        )

    captain_delta = (
        _base_points(ours.captain, predictions) - _base_points(yours.captain, predictions)
    ) * (CAPTAIN_MULTIPLIER - 1)

    return LineupComparison(
        yours_points=yours.expected_points,
        ours_points=ours.expected_points,
        swaps=tuple(swaps),
        captain_delta=captain_delta,
        yours_captain=yours.captain,
        ours_captain=ours.captain,
        yours_formation=yours.formation_label,
        ours_formation=ours.formation_label,
    )


def selection_from_starters(
    starters: Sequence[PlayerCode],
    picks: Sequence[object],
    predictions: Mapping[PlayerCode, PlayerPrediction],
    captain: PlayerCode | None = None,
) -> XiSelection:
    """Build a selection from an explicit list of starters.

    Used to price a manager's own eleven, which is a *given* rather than
    something to optimise. The captain defaults to the highest scorer among
    them, matching how the optimizer picks one, so the comparison isolates the
    eleven rather than confounding it with a captaincy the manager never made.
    """
    chosen = [pick for pick in picks if pick.player_code in set(starters)]  # type: ignore[attr-defined]
    bench = [pick for pick in picks if pick.player_code not in set(starters)]  # type: ignore[attr-defined]

    counts = dict.fromkeys(Position, 0)
    for pick in chosen:
        counts[pick.position] += 1  # type: ignore[attr-defined]

    ranked = sorted(chosen, key=lambda p: -_base_points(p.player_code, predictions))  # type: ignore[attr-defined]
    leader = captain or (ranked[0].player_code if ranked else None)  # type: ignore[attr-defined]

    base = sum(_base_points(p.player_code, predictions) for p in chosen)  # type: ignore[attr-defined]
    total = base + _base_points(leader, predictions) * (CAPTAIN_MULTIPLIER - 1)

    return XiSelection(
        starters=tuple(chosen),  # type: ignore[arg-type]
        bench=tuple(bench),  # type: ignore[arg-type]
        captain=leader,
        vice_captain=ranked[1].player_code if len(ranked) > 1 else None,  # type: ignore[attr-defined]
        formation=(
            counts[Position.GKP],
            counts[Position.DEF],
            counts[Position.MID],
            counts[Position.FWD],
        ),
        expected_points=total,
    )
