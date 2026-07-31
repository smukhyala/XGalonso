"""The one place raw model output becomes a number a surface may show.

**Why this module exists.** Three surfaces predicted the same player for the
same gameweek and applied three different sets of adjustments:

===========================================  ============  ===========  ====
surface                                      availability  calibration  form
===========================================  ============  ===========  ====
``cli.main._predict_all``                    yes           no           yes
``cli.pipeline.recommend``                   no            no           yes
``api.service._predictions``                 no            yes          yes
===========================================  ============  ===========  ====

So ``xg plan`` tempered a doubtful player and ``xg recommend`` did not; the API
corrected the premium-price bias and neither CLI path did. Same team id, same
deadline, three different projections — and nothing in the code said which was
intended. ``api/service.py`` already carries a header recording an earlier
instance of exactly this, a four-point disagreement that read as an optimizer
bug and cost real time to find.

Adding the missing calls at each site would have fixed today's divergence and
left the mechanism that produced it. So the sequence lives here once, every
surface calls :func:`adjust_predictions`, and a new adjustment is added in one
place or not at all.

**The order is deliberate and the reasons differ per step.**

1. **Price calibration** first, because it corrects the *model*. It is a
   measured property of the estimator, so it belongs against the estimator's
   own output before anything else reads it.
2. **Availability** second. FPL's published chance of playing is the game's own
   statement of fact, applied in full and unclamped, scaling the minutes
   distribution so every component moves with it.
3. **Form signals** last, and clamped. A signal is somebody's reading of a match
   report — the weakest evidence here, so it adjusts a number the stronger
   sources have already settled rather than one they then overwrite.

Every step is optional and skipping one is explicit at the call site, so
"this surface does less" is visible in the caller rather than inferred from
which imports it happens to have.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from xg_alonso.contracts.form import SignalSet
from xg_alonso.contracts.identifiers import PlayerCode, TenthsOfMillion
from xg_alonso.contracts.prediction import PlayerPrediction
from xg_alonso.prediction.availability import apply_availability
from xg_alonso.prediction.calibration import apply_price_calibration
from xg_alonso.prediction.form import apply_form_signals

__all__ = ["ADJUSTMENT_ORDER", "adjust_predictions"]

#: The sequence, named so a report can state what was applied rather than
#: leaving a reader to infer it from the numbers.
ADJUSTMENT_ORDER: tuple[str, ...] = ("price_calibration", "availability", "form_signals")


def adjust_predictions(
    predictions: list[PlayerPrediction],
    *,
    prices: Mapping[PlayerCode, TenthsOfMillion] | None = None,
    chances: Mapping[PlayerCode, int | None] | None = None,
    signals: SignalSet | None = None,
    at: datetime | None = None,
) -> list[PlayerPrediction]:
    """Apply every adjustment a surface is entitled to, in the settled order.

    Args:
        predictions: Raw estimator output, already assembled into points.
        prices: Current price per player. Enables the measured price-band bias
            correction. Omitted means the correction is skipped, not that it is
            zero.
        chances: FPL's ``chance_of_playing_next_round`` per player, on a 0-100
            scale, with ``None`` meaning fully fit — which is what the payload
            itself means by a null.
        signals: External form signals. Omitted or empty means no signal exists,
            which is different from a signal that says nothing.
        at: The instant signals are evaluated against. Pass the **deadline**,
            never the wall clock, or a backtest sees signals that had not been
            published when the decision was made.

    Returns:
        A new list. Nothing is mutated, so a caller holding the raw predictions
        still has them — which matters when one surface wants both the adjusted
        number and the model's unadjusted opinion.
    """
    adjusted = predictions

    if prices:
        adjusted = apply_price_calibration(adjusted, prices)

    if chances:
        adjusted = apply_availability(adjusted, chances)

    if signals is not None and at is not None:
        adjusted = apply_form_signals(adjusted, signals, at=at)

    return adjusted
