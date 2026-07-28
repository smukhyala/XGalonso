"""Load external form signals and apply them to predictions.

Signals are read from a JSON file rather than fetched. There is no scraper and
no provider client anywhere in this package, which keeps the runtime dependency
surface exactly where decision D6 put it: whatever process produces the file —
a person, a search tool, a newsroom feed — lives outside the prediction path,
and this module only decides what a signal is allowed to do once it arrives.

**Where the adjustment is applied, and why there.** After the components are
assembled into points, not before. A signal is a statement about a player, not
about a rate of shot creation, so pushing it into a component would require
choosing *which* component it moved — a decision the evidence does not support.
Scaling the assembled total says what the source actually said: this player is
worth somewhat less than his numbers suggest.

The breakdown is rescaled with the total so the two continue to agree. The
prediction contract enforces that they sum, so a signal that moved one without
the other would not construct.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from xg_alonso.contracts.form import FormDirection, FormSignal, SignalSet
from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.prediction import PlayerPrediction, PointsBreakdown
from xg_alonso.contracts.reason_codes import Reason, ReasonCode, ReasonPolarity

__all__ = ["apply_form_signals", "form_reason", "load_signals"]


def load_signals(path: Path) -> SignalSet:
    """Read signals from disk, or return an empty set when there are none.

    A missing file is not an error. Outside information is optional by design —
    the system must produce the same answer it always did when nobody has
    written anything down.
    """
    if not path.exists():
        return SignalSet()

    payload = json.loads(path.read_text())
    entries = payload.get("signals", payload) if isinstance(payload, dict) else payload
    return SignalSet(
        signals=tuple(FormSignal.model_validate(entry) for entry in entries),
        loaded_at=None,
    )


def _rescale(breakdown: PointsBreakdown, factor: float) -> PointsBreakdown:
    """Scale every term by the same factor.

    Uniform because the signal says nothing about *which* part of a player's
    game fell away. Guessing that a confidence problem costs goals rather than
    assists would be inventing a mechanism the source never described.
    """
    return PointsBreakdown(
        appearance=breakdown.appearance * factor,
        goals=breakdown.goals * factor,
        assists=breakdown.assists * factor,
        clean_sheets=breakdown.clean_sheets * factor,
        goals_conceded=breakdown.goals_conceded * factor,
        saves=breakdown.saves * factor,
        cards=breakdown.cards * factor,
        own_goals=breakdown.own_goals * factor,
        penalties=breakdown.penalties * factor,
        defensive_contribution=breakdown.defensive_contribution * factor,
        bonus=breakdown.bonus * factor,
    )


def apply_form_signals(
    predictions: Sequence[PlayerPrediction],
    signals: SignalSet,
    *,
    at: datetime,
) -> list[PlayerPrediction]:
    """Scale predictions by any live signal for that player.

    Args:
        predictions: Model output, already assembled into points.
        signals: Everything on file.
        at: The moment to evaluate expiry against — the gameweek deadline, not
            the wall clock, so a backtest sees the signals that were live then
            rather than the ones live now.

    Returns:
        Predictions, adjusted where a signal applies and untouched elsewhere.
    """
    live = signals.live(at)
    if not live:
        return list(predictions)

    adjusted: list[PlayerPrediction] = []
    for prediction in predictions:
        signal = live.get(prediction.player_code)
        if signal is None:
            adjusted.append(prediction)
            continue

        factor = signal.multiplier
        breakdown = _rescale(prediction.breakdown, factor)
        adjusted.append(
            prediction.model_copy(
                update={
                    "breakdown": breakdown,
                    "expected_points": breakdown.total,
                    # Outside information is softer than measurement, so acting
                    # on it should widen the uncertainty the optimizer prices,
                    # not narrow it. A signal that made a projection *more*
                    # confident would be the wrong shape entirely.
                    "expected_points_sd": prediction.expected_points_sd * (1.0 + abs(1.0 - factor)),
                }
            )
        )
    return adjusted


def form_reason(signal: FormSignal, player: PlayerCode, weight: float) -> Reason:
    """A grounded reason citing the signal and its source.

    The summary and the URL travel through ``context``, which is the non-numeric
    channel, so neither can smuggle a statistic into prose. The only number the
    template renders is the shift, and that comes from the clamped multiplier.
    """
    code = (
        ReasonCode.FORM_SIGNAL_NEGATIVE
        if signal.direction is FormDirection.NEGATIVE
        else ReasonCode.FORM_SIGNAL_POSITIVE
    )
    return Reason(
        code=code,
        polarity=(
            ReasonPolarity.SUPPORTS_OUT
            if signal.direction is FormDirection.NEGATIVE
            else ReasonPolarity.SUPPORTS_IN
        ),
        subject=player,
        evidence={"shift": abs(1.0 - signal.multiplier)},
        context={"summary": signal.summary, "source": signal.sources[0]},
        weight=weight,
    )
