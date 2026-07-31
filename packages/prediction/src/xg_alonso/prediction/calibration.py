"""Correct the model's measured bias by price band.

**The finding this exists for.** Scored on a held-out season, the model
under-predicts expensive players more than cheap ones, and the gap is not small:

    band                    bias     skill    rank corr
    budget  (<£5.5m)      -0.141    +44.5%       0.67
    mid     (£5.5-8.0m)   -0.241    +26.3%       0.69
    premium (£8.0-11.0m)  -0.351    +20.4%       0.60
    elite   (£11.0m+)     +0.098     +3.3%       0.31

Two separate problems live in that table and only one of them is fixable here.

**The bias is fixable.** A systematic under-projection of a fifth of a point per
gameweek, concentrated in the players a squad spends its budget on, is exactly
the error that makes an optimizer bank money rather than upgrade. It compounds:
over five gameweeks a premium is under-valued by more than a point against a
budget alternative, which is more than most transfer decisions turn on. It is
measured out of sample, so adding it back is a correction rather than a guess.

**The collapsing skill is not.** At £11m and above the model has a rank
correlation of 0.31 and 3.3% skill — it can barely order elite players at all.
No additive correction fixes that, and pretending otherwise would be worse than
leaving it: a confident number that cannot discriminate is more dangerous than
an uncertain one. So the elite band's *uncertainty* is widened instead, which is
what the optimizer's risk penalty already knows how to consume.

**Why the elite band is not bias-corrected.** Its measured bias is +0.098 on 64
observations. That is both the wrong sign and far too small a sample to act on;
correcting it would be fitting noise, and the honest reading of 64 rows is that
we do not know.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from xg_alonso.contracts.identifiers import PlayerCode, TenthsOfMillion
from xg_alonso.contracts.prediction import PlayerPrediction

__all__ = [
    "CALIBRATION_MEASURED_ON",
    "CALIBRATION_VERSION",
    "PRICE_BAND_BIAS",
    "PriceBandCorrection",
    "apply_price_calibration",
    "correction_for",
]

CALIBRATION_VERSION: Final[str] = "price_bias_v1"

CALIBRATION_MEASURED_ON: Final[tuple[str, ...]] = ("2025-26",)
"""Seasons `PRICE_BAND_BIAS` was measured on.

In code rather than in a docstring because a freeze check has to read it.
A calibration fitted on the season it is later evaluated over is
train-on-test wearing a different name, and the only defence is a machine
comparing this tuple against the evaluation windows.
"""


@dataclass(frozen=True)
class PriceBandCorrection:
    """What to do about one band's measured error."""

    label: str
    lower: int
    upper: int
    bias: float
    """Measured mean of predicted minus actual. Negative means under-projected."""

    sample: int
    rank_correlation: float

    @property
    def is_trustworthy(self) -> bool:
        """Whether the measurement is worth acting on.

        A hundred observations is the floor. Below it the band's mean is mostly
        the sampling error of the band, and correcting by it moves every player
        in the tier on the strength of a handful of matches.
        """
        return self.sample >= 100

    @property
    def adjustment(self) -> float:
        """Points to add back, or zero where the evidence does not support it."""
        if not self.is_trustworthy or self.bias >= 0.0:
            return 0.0
        return -self.bias

    @property
    def uncertainty_multiplier(self) -> float:
        """How much to widen this band's uncertainty.

        Driven by rank correlation rather than by error size, because the
        optimizer's risk penalty exists to price *not knowing which player is
        better* — and that is what a low correlation says. A band the model can
        barely order should not be selected from as confidently as one it orders
        well.
        """
        if self.rank_correlation >= 0.6:
            return 1.0
        # Linear from 1.0 at rho=0.6 up to 2.0 at rho=0.0.
        return 1.0 + (0.6 - max(self.rank_correlation, 0.0)) / 0.6


#: Measured on 2025/26 with a model trained on 2022/23 through 2024/25 — a
#: genuine holdout, produced by `xg score`. Re-measure and update after a
#: retrain; the version string exists so a stale table is visible.
PRICE_BAND_BIAS: Final[tuple[PriceBandCorrection, ...]] = (
    PriceBandCorrection("budget", 0, 55, bias=-0.141, sample=18066, rank_correlation=0.67),
    PriceBandCorrection("mid", 55, 80, bias=-0.241, sample=7100, rank_correlation=0.69),
    PriceBandCorrection("premium", 80, 110, bias=-0.351, sample=520, rank_correlation=0.60),
    PriceBandCorrection("elite", 110, 10_000, bias=+0.098, sample=64, rank_correlation=0.31),
)


def correction_for(price: int) -> PriceBandCorrection | None:
    """The band a price falls into."""
    for band in PRICE_BAND_BIAS:
        if band.lower <= price < band.upper:
            return band
    return None


def apply_price_calibration(
    predictions: Sequence[PlayerPrediction],
    prices: Mapping[PlayerCode, TenthsOfMillion],
) -> list[PlayerPrediction]:
    """Add back each band's measured under-projection, and widen where blind.

    Applied to the assembled total rather than to a component, because the bias
    was measured on the total and attributing it to goals rather than to minutes
    would be inventing a mechanism the measurement does not identify.

    Args:
        predictions: Assembled predictions.
        prices: Current price per player, in tenths of a million.

    Returns:
        Calibrated predictions. Players with no price are returned untouched.
    """
    calibrated: list[PlayerPrediction] = []

    for prediction in predictions:
        price = prices.get(prediction.player_code)
        band = None if price is None else correction_for(int(price))
        if band is None or (band.adjustment == 0.0 and band.uncertainty_multiplier == 1.0):
            calibrated.append(prediction)
            continue

        adjustment = band.adjustment
        total = prediction.expected_points
        breakdown = prediction.breakdown

        if adjustment and abs(total) > 1e-9:
            factor = (total + adjustment) / total
            breakdown = breakdown.model_copy(
                update={
                    field: getattr(breakdown, field) * factor
                    for field in (
                        "appearance",
                        "goals",
                        "assists",
                        "clean_sheets",
                        "goals_conceded",
                        "saves",
                        "cards",
                        "own_goals",
                        "penalties",
                        "defensive_contribution",
                        "bonus",
                    )
                }
            )
            total = breakdown.total

        calibrated.append(
            prediction.model_copy(
                update={
                    "breakdown": breakdown,
                    "expected_points": total,
                    "expected_points_sd": prediction.expected_points_sd
                    * band.uncertainty_multiplier,
                }
            )
        )

    return calibrated
