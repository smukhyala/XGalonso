"""Value a transfer over a horizon, not over one gameweek.

**What was wrong with a one-week objective.** A transfer is permanent and its
cost is paid once, but its benefit accrues every week the player stays in the
squad. Judging it on the next gameweek alone systematically undervalues buying
a better player and overvalues chasing a favourable fixture — the classic
mistake of taking a hit for a one-week bump, made by the optimizer rather than
by the manager.

**Why not simply sum five gameweeks.** Two reasons, and both change the answer:

1. **A free transfer arrives every week.** A move made now can be unmade in a
   week at no cost, so holding a player for five gameweeks is a choice, not a
   consequence. Value that decays with the option to revisit it is worth less
   than value locked in.
2. **Predictions decay.** A projection five gameweeks out is built from the same
   features as the one for next week, with five more weeks of unmodelled news
   between. Weighting them equally asserts a confidence in week five that
   nothing in the data supports.

So future gameweeks are discounted geometrically. The discount is not a
time-preference — points in week five are worth exactly as many points — it
prices *uncertainty and reversibility together*, which is why it is a single
factor rather than two.

**On hits.** The four-point cost of an extra transfer is paid once and compared
against a horizon of benefit, which is the comparison a manager actually faces
and the one a single-gameweek objective could never express.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DEFAULT_DISCOUNT",
    "DEFAULT_HORIZON",
    "HorizonValue",
    "discount_weights",
    "value_over_horizon",
]

#: Gameweeks a transfer is judged over.
#:
#: Five. Long enough that a permanent improvement outweighs a one-week fixture
#: swing, short enough that the fixture list and the squad are still recognisable
#: — beyond about six weeks the transfers already planned change the squad more
#: than the decision being made now.
DEFAULT_HORIZON: Final[int] = 5

#: Per-gameweek discount.
#:
#: 0.85, so week five carries about half the weight of week one. Set by what it
#: has to express rather than fitted: a free transfer each week means roughly a
#: 1-in-N chance the player is moved on before any given future week, and the
#: prediction for that week is materially less certain than the one for the
#: next. A value near 1.0 would treat a five-week-old forecast as fact; a value
#: near 0.5 would collapse back to the single-gameweek objective this replaces.
DEFAULT_DISCOUNT: Final[float] = 0.85


def discount_weights(
    horizon: int = DEFAULT_HORIZON, discount: float = DEFAULT_DISCOUNT
) -> tuple[float, ...]:
    """Weight per gameweek, normalised so the first is exactly 1.0.

    Normalising to the first week rather than to a sum of one keeps the units
    interpretable: a horizon value is still "expected points in a gameweek",
    just one that counts the weeks after it at a discount. A sum-normalised
    weighting would silently rescale every figure on the screen.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least one gameweek, got {horizon}")
    if not 0.0 < discount <= 1.0:
        raise ValueError(f"discount must be in (0, 1], got {discount}")
    return tuple(discount**index for index in range(horizon))


@dataclass(frozen=True)
class HorizonValue:
    """What a player is worth over the horizon, and how that was reached."""

    per_gameweek: tuple[float, ...]
    weights: tuple[float, ...]
    total: float
    """Discounted sum. Comparable to a single-gameweek projection in units."""

    immediate: float
    """The next gameweek alone — what the old objective would have used."""

    @property
    def horizon(self) -> int:
        return len(self.per_gameweek)

    @property
    def lift_over_immediate(self) -> float:
        """How much the horizon view adds over judging by next week alone.

        Positive for a player whose fixtures improve, negative for one being
        bought on a single good week. This is the number that says whether the
        horizon changed the decision at all.
        """
        return self.total - self.immediate

    def explain(self) -> str:
        weeks = ", ".join(f"{points:.2f}" for points in self.per_gameweek)
        return (
            f"{self.total:.2f} over {self.horizon} gameweeks ({weeks}), "
            f"against {self.immediate:.2f} for the next one alone"
        )


def value_over_horizon(
    per_gameweek: Sequence[float],
    *,
    discount: float = DEFAULT_DISCOUNT,
) -> HorizonValue:
    """Collapse a run of per-gameweek projections into one comparable number.

    Args:
        per_gameweek: Expected points for each gameweek, nearest first.
        discount: Per-gameweek weight decay.

    Returns:
        The discounted value, with the arithmetic retained so an explanation can
        show which weeks carried it.
    """
    if not per_gameweek:
        raise ValueError("a horizon needs at least one gameweek of projections")

    weights = discount_weights(len(per_gameweek), discount)
    total = sum(points * weight for points, weight in zip(per_gameweek, weights, strict=True))

    return HorizonValue(
        per_gameweek=tuple(float(points) for points in per_gameweek),
        weights=weights,
        total=round(total, 6),
        immediate=float(per_gameweek[0]),
    )


def transfer_is_worth_a_hit(
    gain: HorizonValue,
    *,
    hit_cost: int,
    threshold: float = 0.0,
) -> bool:
    """Whether a horizon gain justifies paying for the transfer.

    The comparison a single-gameweek objective could not make: a hit is paid
    once, in full, and set against benefit accruing over several weeks. Judged
    on next week alone a four-point hit almost never clears; judged over five it
    sometimes does, which is why managers take them and the old objective never
    would have.
    """
    return gain.total - hit_cost > threshold
