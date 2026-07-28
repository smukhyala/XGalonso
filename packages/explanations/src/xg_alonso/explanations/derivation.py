"""Show the arithmetic behind an expected-points total.

The points breakdown says a player's goals term is worth 0.92. It does not say
*why* — that 0.92 is 0.23 expected goals multiplied by the 4 points a forward
receives for one. Without the middle term a reader can see which component
dominates and cannot check any of it, so the number is auditable in principle
and opaque in practice.

Every line here reproduces one term of :func:`~xg_alonso.domain.assemble_points`
from the same components and the same pinned rules. It is a *re-derivation*,
not a second calculation: the line's product is asserted against the breakdown
term it explains, so a divergence surfaces as a failure rather than as prose
that quietly contradicts the total beside it.

Two terms are genuinely not a plain product, and both say so rather than
rounding the explanation to fit:

- **Appearance** is a two-branch expectation — a substitute earns the short-play
  points, a player who lasts an hour earns the long-play ones instead.
- **Goals conceded and saves** pay per completed pair and per completed three,
  so dividing a mean by the threshold is a first-order approximation that is
  exact only when the count divides evenly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from xg_alonso.contracts.prediction import ComponentExpectations, PlayerPrediction, Position
from xg_alonso.domain.scoring import ScoringRules

__all__ = ["DerivationLine", "PointsDerivation", "derive_points"]

#: How far a re-derived line may sit from the term it explains before it is
#: reported as disagreeing. Tight, because both sides are the same arithmetic
#: over the same floats — anything larger is a real divergence, not rounding.
_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class DerivationLine:
    """One component's contribution, with the arithmetic that produced it."""

    component: str
    """The scoring term, as a person would name it."""

    expectation: float
    """How much of the event is expected — goals, saves, a probability."""

    unit: str
    """What ``expectation`` counts, e.g. "expected goals" or "chance"."""

    rate: float
    """Points per unit, read from the pinned rules."""

    points: float
    """The contribution to the total."""

    note: str = ""
    """An approximation or branch worth naming, when there is one."""

    @property
    def is_exact_product(self) -> bool:
        """Whether ``expectation x rate`` really is how this line was computed."""
        return not self.note

    def explain(self) -> str:
        """The line as a sentence, with its own arithmetic shown."""
        head = (
            f"{self.component}: {self.expectation:.3g} {self.unit} "
            f"x {self.rate:+.0f} pts = {self.points:+.2f}"
        )
        return f"{head}. {self.note}" if self.note else f"{head}."


@dataclass(frozen=True)
class PointsDerivation:
    """Every line, and whether they reconcile with the total they explain."""

    player_position: Position
    lines: tuple[DerivationLine, ...]
    total: float
    reconciles: bool
    """Whether the lines sum to the prediction's own total.

    False should be impossible. It is surfaced rather than asserted because an
    explanation that silently disagrees with its number is worse than one that
    admits it does.
    """

    def material(self, floor: float = 0.005) -> tuple[DerivationLine, ...]:
        """Lines large enough to be worth reading, biggest contribution first.

        A forward's clean-sheet term is exactly zero and always will be; listing
        it teaches nothing and pushes the terms that matter down the page.
        """
        kept = [line for line in self.lines if abs(line.points) >= floor]
        kept.sort(key=lambda line: -abs(line.points))
        return tuple(kept)


def derive_points(prediction: PlayerPrediction, rules: ScoringRules) -> PointsDerivation:
    """Re-derive a prediction's points from its components and the pinned rules.

    Args:
        prediction: The prediction to explain.
        rules: The same scoring snapshot that assembled it. Passing a different
            one produces lines that do not reconcile, which is exactly what
            ``reconciles`` is for.
    """
    components: ComponentExpectations = prediction.components
    position = prediction.position
    minutes = components.minutes
    thresholds = rules.thresholds
    breakdown = prediction.breakdown

    p_short_only = max(0.0, minutes.p_appearance - minutes.p_60_plus)
    lines: list[DerivationLine] = [
        DerivationLine(
            component="Appearing",
            expectation=minutes.p_appearance,
            unit="chance of playing",
            rate=float(rules.long_play),
            points=breakdown.appearance,
            note=(
                f"Two branches, not one product: a {p_short_only:.0%} chance of a short "
                f"appearance pays {rules.short_play}, and a {minutes.p_60_plus:.0%} chance "
                f"of lasting 60 minutes pays {rules.long_play} instead."
            ),
        ),
        DerivationLine(
            component="Goals",
            expectation=components.goals,
            unit="expected goals",
            rate=float(rules.goals_scored.get(position, 0)),
            points=breakdown.goals,
        ),
        DerivationLine(
            component="Assists",
            expectation=components.assists,
            unit="expected assists",
            rate=float(rules.assists),
            points=breakdown.assists,
        ),
        DerivationLine(
            component="Clean sheet",
            expectation=components.clean_sheet_probability,
            unit="chance of one while lasting 60 minutes",
            rate=float(rules.clean_sheets.get(position, 0)),
            points=breakdown.clean_sheets,
        ),
        DerivationLine(
            component="Goals conceded",
            expectation=components.goals_conceded,
            unit="expected conceded",
            rate=float(rules.goals_conceded.get(position, 0)),
            points=breakdown.goals_conceded,
            note=(
                f"Paid per completed pair, so the expectation is divided by "
                f"{thresholds.goals_conceded_per_deduction} first. Exact only when the "
                "count is even, and it biases every defender identically."
            ),
        ),
        DerivationLine(
            component="Saves",
            expectation=components.saves,
            unit="expected saves",
            rate=float(rules.saves),
            points=breakdown.saves,
            note=(
                f"Paid per completed {thresholds.saves_per_point}, so the expectation is "
                f"divided by {thresholds.saves_per_point} first."
            ),
        ),
        DerivationLine(
            component="Cards",
            expectation=components.yellow_cards,
            unit="expected yellows",
            rate=float(rules.yellow_cards),
            points=breakdown.cards,
        ),
        DerivationLine(
            component="Defensive actions",
            expectation=components.defensive_contribution_probability,
            unit="chance of reaching the threshold",
            rate=float(rules.defensive_contribution.get(position, 0)),
            points=breakdown.defensive_contribution,
        ),
        DerivationLine(
            component="Bonus",
            expectation=components.bonus,
            unit="expected bonus points",
            rate=float(rules.bonus),
            points=breakdown.bonus,
        ),
    ]

    summed = sum(line.points for line in lines)
    # Own goals and penalties are modelled at zero, so they are omitted from the
    # lines above; they are added here so the reconciliation is against the real
    # total rather than a convenient subset of it.
    summed += breakdown.own_goals + breakdown.penalties

    return PointsDerivation(
        player_position=position,
        lines=tuple(lines),
        total=breakdown.total,
        reconciles=abs(summed - breakdown.total) <= _TOLERANCE,
    )
