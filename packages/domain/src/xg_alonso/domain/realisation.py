"""Realised points from realised counts — the exact inverse of :func:`assemble_points`.

``assemble_points`` maps *expected* counts to *expected* points, and it is linear
because expectation is linear. That linearity is an approximation in two places,
and its own docstring says so: goals conceded deducts a point per completed
*pair*, and saves pay a point per completed *triple*, so dividing a mean by two
or three is exact only when the count happens to be even or a multiple of three.

This module is the other half. It takes counts that actually happened and
applies the rules exactly — ``floor(conceded / 2)``, ``floor(saves / 3)``, a
threshold test on defensive contribution — producing the integer FPL actually
awarded.

**Why it is needed.** Nothing downstream can score a *distribution* over
outcomes without a function that prices one realisation. The composition engine
convolves component PMFs through this map; the calibration report scores
forecasts against it; and the difference between its mean and
``assemble_points``' total is precisely the linearisation gap, which becomes a
measurable quantity rather than a caveat in a docstring.

**Why it belongs in `domain`.** It is the same knowledge ``assemble_points``
holds — what a component is worth — and it must load that knowledge from the
same pinned :class:`ScoringRules`. Splitting the two across packages would
create two places where FPL's rules live.

**Verified against reality.** ``tests/domain/test_realisation.py`` reconstructs
``total_points`` for every row of the silver stats table: 113,270 of 113,270
exact across all four seasons. That check also validates every ``VERIFY``-marked
field in :class:`ScoringThresholds`, because a wrong divisor or threshold would
show up as a mismatch somewhere in 113,270 rows.

**On the missing defensive-contribution column.** FPL introduced defensive
contribution in 2025/26, so the column is null for all 83,513 rows of the three
earlier seasons. ``None`` is therefore modelled explicitly and means *the rule
did not exist for this row*, which is the one case where contributing zero is
semantically correct rather than a coerced missing value. A row that genuinely
recorded zero defensive actions carries ``0``, not ``None``, and the two are
kept distinguishable on purpose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from xg_alonso.contracts.prediction import Position
from xg_alonso.domain.scoring import ScoringRules

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "RealisedCounts",
    "realised_points",
    "realised_points_matrix",
]


@dataclass(frozen=True, slots=True)
class RealisedCounts:
    """What one player actually recorded in one fixture.

    Field names match the silver ``player_gameweek_stats`` columns exactly, so
    the mapping from stored data to this object needs no translation table and
    a renamed column fails loudly at the call site.
    """

    minutes: int
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    saves: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    bonus: int = 0
    defensive_contribution: int | None = None
    """Raw defensive actions, or ``None`` for seasons before the rule existed."""


def realised_points(
    counts: RealisedCounts,
    position: Position,
    rules: ScoringRules,
) -> int:
    """Score one realised fixture exactly, under the given season's rules.

    Args:
        counts: What actually happened.
        position: The player's position, which prices goals, clean sheets,
            concessions and defensive contribution differently.
        rules: A pinned :class:`ScoringRules`. Never transcribed constants.

    Returns:
        The integer FPL awarded. Negative totals are possible and legal — a red
        card and an own goal on a substitute appearance is ``-4``.
    """
    thresholds = rules.thresholds

    if counts.minutes >= thresholds.long_play_minutes:
        total = rules.long_play
    elif counts.minutes > 0:
        total = rules.short_play
    else:
        total = 0

    total += counts.goals_scored * rules.goals_scored.get(position, 0)
    total += counts.assists * rules.assists
    total += counts.clean_sheets * rules.clean_sheets.get(position, 0)

    # Floor division, not a mean divided: FPL deducts per *completed* pair, so a
    # player who concedes three loses one point, not one and a half.
    total += (
        counts.goals_conceded // thresholds.goals_conceded_per_deduction
    ) * rules.goals_conceded.get(position, 0)
    total += (counts.saves // thresholds.saves_per_point) * rules.saves

    total += counts.yellow_cards * rules.yellow_cards
    total += counts.red_cards * rules.red_cards
    total += counts.own_goals * rules.own_goals
    total += counts.penalties_saved * rules.penalties_saved
    total += counts.penalties_missed * rules.penalties_missed
    total += counts.bonus * rules.bonus

    if counts.defensive_contribution is not None:
        threshold = rules.defensive_contribution_threshold(position)
        if counts.defensive_contribution >= threshold:
            total += rules.defensive_contribution.get(position, 0)

    return total


def realised_points_matrix(
    *,
    minutes: npt.NDArray[np.int64],
    goals_scored: npt.NDArray[np.int64],
    assists: npt.NDArray[np.int64],
    clean_sheets: npt.NDArray[np.int64],
    goals_conceded: npt.NDArray[np.int64],
    saves: npt.NDArray[np.int64],
    yellow_cards: npt.NDArray[np.int64],
    red_cards: npt.NDArray[np.int64],
    own_goals: npt.NDArray[np.int64],
    penalties_saved: npt.NDArray[np.int64],
    penalties_missed: npt.NDArray[np.int64],
    bonus: npt.NDArray[np.int64],
    defensive_contribution: npt.NDArray[np.int64] | None,
    positions: Sequence[Position],
    rules: ScoringRules,
) -> npt.NDArray[np.int64]:
    """Score many realised fixtures at once, identically to :func:`realised_points`.

    The elementwise agreement between the two is asserted by test rather than
    assumed: a vectorised reimplementation that drifts from the scalar one would
    be invisible in every metric it feeds.

    Args:
        defensive_contribution: ``None`` when the column is absent for every row
            (a pre-2025/26 batch). Per-row absence via a sentinel value is *not*
            supported — split the batch by season instead, which is how the
            stats table is naturally partitioned anyway.
        positions: One per row, same length as every array.

    Returns:
        An ``int64`` array of realised points, one per row.
    """
    n = minutes.shape[0]
    if len(positions) != n:
        raise ValueError(f"positions has {len(positions)} entries for {n} rows")

    thresholds = rules.thresholds

    def per_row(table: dict[Position, int]) -> npt.NDArray[np.int64]:
        return np.array([table.get(p, 0) for p in positions], dtype=np.int64)

    total = np.where(
        minutes >= thresholds.long_play_minutes,
        rules.long_play,
        np.where(minutes > 0, rules.short_play, 0),
    ).astype(np.int64)

    total = total + goals_scored * per_row(rules.goals_scored)
    total = total + assists * rules.assists
    total = total + clean_sheets * per_row(rules.clean_sheets)
    total = total + (goals_conceded // thresholds.goals_conceded_per_deduction) * per_row(
        rules.goals_conceded
    )
    total = total + (saves // thresholds.saves_per_point) * rules.saves
    total = total + yellow_cards * rules.yellow_cards
    total = total + red_cards * rules.red_cards
    total = total + own_goals * rules.own_goals
    total = total + penalties_saved * rules.penalties_saved
    total = total + penalties_missed * rules.penalties_missed
    total = total + bonus * rules.bonus

    if defensive_contribution is not None:
        needed = np.array(
            [rules.defensive_contribution_threshold(p) for p in positions], dtype=np.int64
        )
        earned = (defensive_contribution >= needed).astype(np.int64)
        total = total + earned * per_row(rules.defensive_contribution)

    return total.astype(np.int64)
