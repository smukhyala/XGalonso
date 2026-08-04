"""Starting XI, captaincy and bench order.

**This is the largest scoring lever in the game and it was missing.** Until now
the system validated an XI it was handed but never chose one, and the captain
was whatever sat in squad slot 1 — which, in every backtest run to date, was the
goalkeeper. Captaincy doubles a player's return, so a system that never picks a
captain has never used the single biggest term available to it.

Three things are decided together here, because they are one decision:

- **Formation.** Any legal shape, not a fixed 4-4-2. A squad with three
  outstanding forwards should play them.
- **The XI.** The highest-scoring legal eleven, which is *not* simply the top
  eleven by expected points — positional minima force a goalkeeper and at least
  three defenders into the side regardless of how they rank.
- **Captain and vice.** The best and second-best expected scorers in that XI.

Enumeration is exhaustive because the space is tiny: with one keeper and 3-5
defenders, 2-5 midfielders and 1-3 forwards summing to eleven, there are only a
handful of legal shapes. Within a shape the best choice is the top *n* by
expected points per position, so the whole search is a sort and a few sums.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.domain.constraints import legal_formations
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringThresholds

# `legal_formations` moved down to `domain.constraints`: it derives shapes from
# `SquadRules` alone and is needed by `domain.context_features`, which sits below
# this layer. Re-exported here so every existing import keeps resolving to the
# single definition rather than acquiring a second one.

__all__ = [
    "CAPTAIN_MULTIPLIER",
    "XiSelection",
    "best_starting_xi",
    "legal_formations",
    "starting_xi_points",
]

#: FPL doubles the captain's score.
#:
#: Sourced from `ScoringThresholds` rather than typed here, so it obeys the same
#: no-literals rule as every other scoring constant. It is a `VERIFY` field
#: rather than a read one: `game_config` publishes only
#: `chips[].overrides.pick_multiplier`, and only for the triple-captain chip.
#:
#: The name is kept so the ten call sites and `tests/optimization/test_lineup.py`
#: are untouched. Vice-captaincy is now modelled — see `domain.autosubs` — but
#: `best_starting_xi` still selects against predictions, where nobody has yet
#: failed to play, so the armband it assigns is the captain's.
CAPTAIN_MULTIPLIER: Final[int] = ScoringThresholds().captain_multiplier


@dataclass(frozen=True)
class XiSelection:
    """A chosen eleven, its captain, and the bench in substitution order."""

    starters: tuple[SquadPick, ...]
    bench: tuple[SquadPick, ...]
    captain: PlayerCode | None
    vice_captain: PlayerCode | None
    formation: tuple[int, int, int, int]
    """Players per position in ``(GKP, DEF, MID, FWD)`` order."""

    expected_points: float
    """Total expected points including the captain's doubling."""

    @property
    def formation_label(self) -> str:
        """Conventional shorthand, e.g. ``3-4-3`` (the keeper is implied)."""
        _, defenders, midfielders, forwards = self.formation
        return f"{defenders}-{midfielders}-{forwards}"


def _expected(pick: SquadPick, predictions: Mapping[PlayerCode, PlayerPrediction]) -> float:
    prediction = predictions.get(pick.player_code)
    return prediction.expected_points if prediction is not None else 0.0


def best_starting_xi(
    picks: Sequence[SquadPick],
    predictions: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    *,
    required: PlayerCode | None = None,
) -> XiSelection:
    """Choose the highest-scoring legal eleven, plus captain, vice and bench.

    Args:
        picks: The full squad. Need not be exactly 15 — a mid-search squad may
            be short a player — but must contain enough of each position to
            fill some legal shape.
        predictions: Expected points per player. Missing players score zero,
            which correctly makes an unpredicted player a last resort rather
            than an error.
        rules: Supplies the positional bounds and the starting-XI size.
        required: A player who must start. Used to price what benching him
            actually costs: the difference between the best XI containing him
            and the best XI overall is exact, where comparing his expected
            points against the lowest starter's is not — it ignores that the
            swap may force a different, legal formation.

    Returns:
        The best selection found. When no legal shape can be filled, returns an
        empty selection scoring zero rather than raising — the caller is usually
        ranking thousands of candidate squads and an illegal one should simply
        lose, not abort the search.
    """
    by_position: dict[Position, list[SquadPick]] = {p: [] for p in Position}
    for pick in picks:
        by_position[pick.position].append(pick)

    required_pick: SquadPick | None = None
    if required is not None:
        required_pick = next((p for p in picks if p.player_code == required), None)
        if required_pick is None:
            raise ValueError(f"player {required} is not in this squad and cannot be required")

    # Sorting once per position is what makes the formation loop cheap: within
    # a shape, the best choice is always the top n.
    for position in by_position:
        by_position[position].sort(key=lambda p: (-_expected(p, predictions), int(p.player_code)))

    best: XiSelection | None = None

    for shape in legal_formations(rules):
        counts = dict(
            zip((Position.GKP, Position.DEF, Position.MID, Position.FWD), shape, strict=True)
        )
        if any(len(by_position[pos]) < n for pos, n in counts.items()):
            continue  # squad cannot field this shape

        # A required player consumes one slot in his own position, and the rest
        # of that position is filled from the next best. A shape with no slot
        # for him is simply not a shape he can start in.
        if required_pick is not None and counts[required_pick.position] < 1:
            continue

        starters: list[SquadPick] = []
        for position, count in counts.items():
            pool = by_position[position]
            if required_pick is not None and position is required_pick.position:
                others = [p for p in pool if p.player_code != required_pick.player_code]
                starters.append(required_pick)
                starters.extend(others[: count - 1])
            else:
                starters.extend(pool[:count])

        base = sum(_expected(p, predictions) for p in starters)
        captain_pick = max(starters, key=lambda p: (_expected(p, predictions), -int(p.player_code)))
        total = base + _expected(captain_pick, predictions) * (CAPTAIN_MULTIPLIER - 1)

        if best is None or total > best.expected_points:
            remaining = [p for p in picks if p not in starters]
            # Bench order is substitution priority: best first, but the reserve
            # keeper is fixed in place because he can only replace the keeper.
            outfield = sorted(
                (p for p in remaining if p.position is not Position.GKP),
                key=lambda p: (-_expected(p, predictions), int(p.player_code)),
            )
            keepers = [p for p in remaining if p.position is Position.GKP]
            ordered_starters = sorted(
                starters,
                key=lambda p: (
                    list(Position).index(p.position),
                    -_expected(p, predictions),
                ),
            )
            ranked = sorted(
                starters, key=lambda p: (-_expected(p, predictions), int(p.player_code))
            )
            best = XiSelection(
                starters=tuple(ordered_starters),
                bench=tuple(outfield + keepers),
                captain=ranked[0].player_code if ranked else None,
                vice_captain=ranked[1].player_code if len(ranked) > 1 else None,
                formation=shape,
                expected_points=total,
            )

    if best is None:
        return XiSelection(
            starters=(),
            bench=tuple(picks),
            captain=None,
            vice_captain=None,
            formation=(0, 0, 0, 0),
            expected_points=0.0,
        )
    return best


def starting_xi_points(
    picks: Sequence[SquadPick],
    predictions: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
) -> float:
    """Expected points from the best legal XI, captain doubled.

    This is the quantity every transfer must be judged against. Scoring a squad
    by a *fixed* eleven credits a bench-out transfer with its full expected
    gain, when in reality replacing a benched player changes nothing until he
    is good enough to start.
    """
    return best_starting_xi(picks, predictions, rules).expected_points
