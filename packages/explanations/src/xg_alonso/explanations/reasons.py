"""Build grounded reasons from predictions and their feature evidence.

**Why this is one module rather than two.** Reasons used to be built inside
``optimization.transfer``, which meant the only thing that could be explained
was a transfer. A player's own projection had no explanation at all, and when
one was needed the obvious move — writing a second builder next to the first —
would have produced two sets of prose that could disagree about the same player
while both being technically correct.

So construction lives here, once, and both callers pass through it: a transfer
option explains itself by comparing two players, and a squad member explains
itself by describing one. The evidence is the same evidence in both cases.

**What is and is not asserted.** Every reason is built from values already
present in a :class:`~xg_alonso.contracts.prediction.PlayerPrediction` — either
its assembled points breakdown or the explanatory panel captured at prediction
time. Nothing is recomputed here, so an explanation cannot drift from the
arithmetic it explains. Where a value is missing, no reason is emitted; saying
less is the correct behaviour, and it is why every builder below returns a
variable number of reasons rather than a fixed set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from xg_alonso.contracts.evidence import FeatureValue
from xg_alonso.contracts.identifiers import PlayerCode, TenthsOfMillion
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.reason_codes import Reason, ReasonCode, ReasonPolarity

__all__ = [
    "PopulationStats",
    "build_no_move_reasons",
    "build_player_reasons",
    "build_transfer_reasons",
]

#: Panel feature names this module reads. Named as constants because a typo in a
#: string literal would silently stop a reason being emitted, and a reason that
#: quietly never fires is exactly the failure this work exists to correct.
_XG = "expected_goals_per90_5"
_XA = "expected_assists_per90_5"
_THREAT = "threat_per90_5"
_BPS = "bps_per90_5"
_CEILING = "total_points_max_5"
_VOLATILITY = "total_points_std_10"
_OPPONENT_XG = "opponent_conceded_xg_mean_5"
_HOME = "is_home"

#: A percentile at or above this is worth calling out on its own.
_STRONG_PERCENTILE = 0.75

#: Minimum relative gap before two players' rates are described as different.
#: Without it, 0.31 against 0.30 would be reported as "better shooting numbers",
#: which is true of the arithmetic and false of the football.
_MATERIAL_GAP = 0.15


@dataclass(frozen=True)
class PopulationStats:
    """League-wide reference values for the gameweek being explained.

    Percentiles say where a player sits; a league mean says what the middle
    *is*. Both are needed — "84th percentile" without a reference number is a
    rank with no units, and a raw value without a rank is a number with no
    scale.

    Held as a separate object rather than folded into each prediction because it
    is a property of the population, and duplicating it across six hundred
    players would invite the copies to diverge.
    """

    means: dict[str, float]
    mean_points_per_million: float | None = None
    """League-average projected points per million, when prices were supplied.

    Separate from ``means`` because it is not a panel feature: it is derived
    from a prediction and a price, and prices live outside the prediction.
    """

    def mean_of(self, name: str) -> float | None:
        return self.means.get(name)

    @classmethod
    def from_predictions(
        cls,
        predictions: Sequence[PlayerPrediction],
        *,
        prices: Mapping[PlayerCode, TenthsOfMillion] | None = None,
    ) -> PopulationStats:
        """Mean of each panel feature across every player with a value for it."""
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for prediction in predictions:
            if prediction.feature_evidence is None:
                continue
            for value in prediction.feature_evidence.values:
                if value.value is None:
                    continue
                totals[value.name] = totals.get(value.name, 0.0) + value.value
                counts[value.name] = counts.get(value.name, 0) + 1

        efficiency: float | None = None
        if prices:
            ratios = [
                prediction.expected_points / (prices[prediction.player_code] / 10.0)
                for prediction in predictions
                if prices.get(prediction.player_code, TenthsOfMillion(0)) > 0
            ]
            if ratios:
                efficiency = sum(ratios) / len(ratios)

        return cls(
            means={k: totals[k] / counts[k] for k in totals if counts[k] > 0},
            mean_points_per_million=efficiency,
        )


def _panel(prediction: PlayerPrediction, name: str) -> FeatureValue | None:
    if prediction.feature_evidence is None:
        return None
    return prediction.feature_evidence.get(name)


def _value(prediction: PlayerPrediction, name: str) -> float | None:
    found = _panel(prediction, name)
    return None if found is None else found.value


def _materially_greater(left: float | None, right: float | None) -> bool:
    """Whether ``left`` exceeds ``right`` by enough to be worth a sentence.

    Compared in relative terms against the larger of the two so the threshold
    means the same thing for expected goals (order 0.1) and for threat (order
    100). An absolute epsilon would be far too strict for one and meaningless
    for the other.
    """
    if left is None or right is None:
        return False
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return False
    return (left - right) / scale >= _MATERIAL_GAP


def _rate_pair(
    *,
    prediction: PlayerPrediction,
    other: PlayerPrediction,
    feature: str,
    higher_code: ReasonCode,
    lower_code: ReasonCode | None,
    subject_is_incoming: bool,
    weight: float,
) -> Reason | None:
    """One comparison reason for a single panel rate, or ``None``.

    Both directions come through here because the argument is symmetric: the
    incoming player having the better rate and the outgoing player having the
    worse one are the same fact, and emitting both as independently-built
    reasons is how a screen ends up repeating itself.
    """
    mine = _panel(prediction, feature)
    theirs = _panel(other, feature)
    if mine is None or theirs is None or mine.value is None or theirs.value is None:
        return None
    if mine.percentile is None:
        return None

    if subject_is_incoming:
        if not _materially_greater(mine.value, theirs.value):
            return None
        code = higher_code
        polarity = ReasonPolarity.SUPPORTS_IN
    else:
        if lower_code is None or not _materially_greater(theirs.value, mine.value):
            return None
        code = lower_code
        polarity = ReasonPolarity.SUPPORTS_OUT

    return Reason(
        code=code,
        polarity=polarity,
        subject=prediction.player_code,
        evidence={
            "value": mine.value,
            "other": theirs.value,
            "percentile": mine.percentile,
        },
        context={"position": prediction.position.value},
        weight=weight,
    )


def build_transfer_reasons(
    *,
    incoming: PlayerPrediction,
    outgoing: PlayerPrediction,
    gross_gain: float,
    incoming_price: TenthsOfMillion,
    outgoing_price: TenthsOfMillion,
    population: PopulationStats | None = None,
    candidate_count: int | None = None,
) -> tuple[Reason, ...]:
    """Why this swap, grounded in both players' evidence.

    Reasons are ordered by weight downstream, so the order they are built in
    does not matter. What matters is that each one is only built when the
    evidence supports it — an explanation with three sentences and an
    explanation with seven are both correct outputs, and forcing a fixed number
    would mean inventing the difference.
    """
    reasons: list[Reason] = []
    magnitude = abs(gross_gain)

    # --- minutes, the term that dominates everything else ---
    in_minutes = incoming.components.minutes
    out_minutes = outgoing.components.minutes

    if in_minutes.p_start >= 0.6:
        reasons.append(
            Reason(
                code=ReasonCode.EXPECTED_MINUTES_SECURE,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=incoming.player_code,
                evidence={
                    "p_start": in_minutes.p_start,
                    "expected_minutes": in_minutes.expected_minutes,
                },
                weight=magnitude * 0.4,
            )
        )

    if out_minutes.p_start < in_minutes.p_start:
        reasons.append(
            Reason(
                code=ReasonCode.EXPECTED_MINUTES_DECLINE,
                polarity=ReasonPolarity.SUPPORTS_OUT,
                subject=outgoing.player_code,
                evidence={
                    "p_start": out_minutes.p_start,
                    "expected_minutes": out_minutes.expected_minutes,
                },
                weight=magnitude * 0.3,
            )
        )

    # --- projected returns, which is what the optimizer actually compared ---
    in_xgi = incoming.components.goals + incoming.components.assists
    out_xgi = outgoing.components.goals + outgoing.components.assists
    if in_xgi > out_xgi:
        reasons.append(
            Reason(
                code=ReasonCode.UNDERLYING_STATS_IMPROVING,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=incoming.player_code,
                evidence={"recent_xgi": in_xgi, "baseline_xgi": out_xgi},
                weight=magnitude * 0.5,
            )
        )

    # --- the underlying rates behind those projections ---
    for feature, higher, lower, weight in (
        (_XG, ReasonCode.XG_RATE_HIGHER, ReasonCode.XG_RATE_LOWER, 0.55),
        (_XA, ReasonCode.XA_RATE_HIGHER, ReasonCode.XA_RATE_LOWER, 0.45),
        (_THREAT, ReasonCode.THREAT_HIGHER, None, 0.3),
    ):
        for prediction, other, is_incoming in (
            (incoming, outgoing, True),
            (outgoing, incoming, False),
        ):
            reason = _rate_pair(
                prediction=prediction,
                other=other,
                feature=feature,
                higher_code=higher,
                lower_code=lower,
                subject_is_incoming=is_incoming,
                weight=magnitude * weight,
            )
            if reason is not None:
                reasons.append(reason)

    # --- return shape: two players with the same mean are not the same player ---
    in_ceiling = _value(incoming, _CEILING)
    out_ceiling = _value(outgoing, _CEILING)
    if _materially_greater(in_ceiling, out_ceiling):
        # `_materially_greater` returns False for either side being None, so both
        # are known here. Narrowed with locals rather than an assert so the type
        # holds without a runtime check that only ever documents the guard above.
        ceiling_in = float(in_ceiling or 0.0)
        ceiling_out = float(out_ceiling or 0.0)
        reasons.append(
            Reason(
                code=ReasonCode.CEILING_HIGHER,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=incoming.player_code,
                evidence={"value": ceiling_in, "other": ceiling_out},
                weight=magnitude * 0.25,
            )
        )

    in_volatility = _value(incoming, _VOLATILITY)
    out_volatility = _value(outgoing, _VOLATILITY)
    if _materially_greater(out_volatility, in_volatility):
        volatility_in = float(in_volatility or 0.0)
        volatility_out = float(out_volatility or 0.0)
        reasons.append(
            Reason(
                code=ReasonCode.VOLATILITY_LOWER,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=incoming.player_code,
                evidence={"value": volatility_in, "other": volatility_out},
                weight=magnitude * 0.2,
            )
        )

    bps = _panel(incoming, _BPS)
    if bps is not None and bps.value is not None and (bps.percentile or 0.0) >= _STRONG_PERCENTILE:
        reasons.append(
            Reason(
                code=ReasonCode.BONUS_MAGNET,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=incoming.player_code,
                evidence={"value": bps.value, "percentile": bps.percentile or 0.0},
                context={"position": incoming.position.value},
                weight=magnitude * 0.2,
            )
        )

    # --- fixture, which the vocabulary has always been able to say and never did ---
    reasons.extend(_fixture_reasons(incoming, population, ReasonPolarity.SUPPORTS_IN, magnitude))

    # --- money ---
    if incoming_price > 0 and outgoing_price > 0:
        in_efficiency = incoming.expected_points / (incoming_price / 10.0)
        out_efficiency = outgoing.expected_points / (outgoing_price / 10.0)
        if _materially_greater(in_efficiency, out_efficiency):
            reasons.append(
                Reason(
                    code=ReasonCode.PRICE_EFFICIENCY,
                    polarity=ReasonPolarity.SUPPORTS_IN,
                    subject=incoming.player_code,
                    evidence={"value": in_efficiency, "other": out_efficiency},
                    weight=magnitude * 0.15,
                )
            )

    # --- why the alternatives you were expecting were not considered ---
    if candidate_count is not None:
        reasons.append(
            Reason(
                code=ReasonCode.POSITION_LOCKED,
                polarity=ReasonPolarity.CONTEXT,
                subject=outgoing.player_code,
                evidence={"candidate_count": float(candidate_count)},
                context={"position": outgoing.position.value},
                weight=0.0,
            )
        )

    return tuple(reasons)


def _fixture_reasons(
    prediction: PlayerPrediction,
    population: PopulationStats | None,
    polarity: ReasonPolarity,
    magnitude: float,
) -> list[Reason]:
    """Fixture difficulty, stated against the league rather than in the abstract.

    ``FIXTURE_SWING_*`` has been in the vocabulary since the first version and
    was never emitted, which is why fixtures — the factor most managers weigh
    first — appeared in no explanation the system produced.
    """
    reasons: list[Reason] = []

    opponent = _value(prediction, _OPPONENT_XG)
    league = None if population is None else population.mean_of(_OPPONENT_XG)
    if opponent is not None and league is not None and league > 0:
        if opponent >= league * (1.0 + _MATERIAL_GAP):
            code = ReasonCode.FIXTURE_SWING_POSITIVE
        elif opponent <= league * (1.0 - _MATERIAL_GAP):
            code = ReasonCode.FIXTURE_SWING_NEGATIVE
        else:
            code = None
        if code is not None:
            reasons.append(
                Reason(
                    code=code,
                    polarity=polarity,
                    subject=prediction.player_code,
                    evidence={"opponent_xg": opponent, "league_average": league},
                    weight=magnitude * 0.35,
                )
            )

    home = _value(prediction, _HOME)
    if home is not None and home >= 0.5:
        reasons.append(
            Reason(
                code=ReasonCode.HOME_FIXTURE,
                polarity=polarity,
                subject=prediction.player_code,
                weight=magnitude * 0.1,
            )
        )

    return reasons


def build_player_reasons(
    prediction: PlayerPrediction,
    *,
    population: PopulationStats | None = None,
    price: TenthsOfMillion | None = None,
    chance_of_playing: float | None = None,
) -> tuple[Reason, ...]:
    """Why this player projects the way he does, independent of any transfer.

    This is the answer to "why is he 2.6" — a question the product could not
    previously answer for any player, because reasons only existed for the two
    players involved in the single recommended move.
    """
    reasons: list[Reason] = []
    breakdown = prediction.breakdown

    reasons.append(
        Reason(
            code=ReasonCode.POINTS_BREAKDOWN,
            polarity=ReasonPolarity.CONTEXT,
            subject=prediction.player_code,
            evidence={
                "total": breakdown.total,
                "appearance": breakdown.appearance,
                "goals": breakdown.goals,
                "assists": breakdown.assists,
                "clean_sheets": breakdown.clean_sheets,
                "bonus": breakdown.bonus,
            },
            weight=abs(breakdown.total),
        )
    )

    minutes = prediction.components.minutes
    code = (
        ReasonCode.EXPECTED_MINUTES_SECURE
        if minutes.p_start >= 0.6
        else ReasonCode.EXPECTED_MINUTES_DECLINE
    )
    reasons.append(
        Reason(
            code=code,
            polarity=ReasonPolarity.CONTEXT,
            subject=prediction.player_code,
            evidence={
                "p_start": minutes.p_start,
                "expected_minutes": minutes.expected_minutes,
            },
            weight=abs(prediction.expected_points) * 0.5,
        )
    )

    if chance_of_playing is not None and chance_of_playing < 1.0:
        reasons.append(
            Reason(
                code=ReasonCode.AVAILABILITY_RISK_HIGH,
                polarity=ReasonPolarity.SUPPORTS_OUT,
                subject=prediction.player_code,
                evidence={"chance_of_playing": chance_of_playing},
                weight=abs(prediction.expected_points),
            )
        )

    # Standout panel values, so a player's own page names the statistics that
    # distinguish him rather than restating his projection in other words.
    for feature, reason_code in (
        (_XG, ReasonCode.XG_RATE_HIGHER),
        (_XA, ReasonCode.XA_RATE_HIGHER),
        (_BPS, ReasonCode.BONUS_MAGNET),
    ):
        panel = _panel(prediction, feature)
        if panel is None or panel.value is None or panel.percentile is None:
            continue
        if panel.percentile < _STRONG_PERCENTILE:
            continue
        league = None if population is None else population.mean_of(feature)
        if reason_code is ReasonCode.BONUS_MAGNET:
            reasons.append(
                Reason(
                    code=reason_code,
                    polarity=ReasonPolarity.CONTEXT,
                    subject=prediction.player_code,
                    evidence={"value": panel.value, "percentile": panel.percentile},
                    context={"position": prediction.position.value},
                    weight=panel.percentile,
                )
            )
        elif league is not None:
            # Compared against the league mean, because with no second player in
            # frame there is nothing else for "against" to mean.
            reasons.append(
                Reason(
                    code=reason_code,
                    polarity=ReasonPolarity.CONTEXT,
                    subject=prediction.player_code,
                    evidence={
                        "value": panel.value,
                        "other": league,
                        "percentile": panel.percentile,
                    },
                    context={"position": prediction.position.value},
                    weight=panel.percentile,
                )
            )

    reasons.extend(
        _fixture_reasons(
            prediction,
            population,
            ReasonPolarity.CONTEXT,
            abs(prediction.expected_points),
        )
    )

    if price is not None and price > 0:
        efficiency = prediction.expected_points / (price / 10.0)
        league_points = None if population is None else population.mean_points_per_million
        if league_points is not None and _materially_greater(efficiency, league_points):
            reasons.append(
                Reason(
                    code=ReasonCode.PRICE_EFFICIENCY,
                    polarity=ReasonPolarity.CONTEXT,
                    subject=prediction.player_code,
                    evidence={"value": efficiency, "other": league_points},
                    weight=0.1,
                )
            )

    return tuple(reasons)


def build_no_move_reasons(
    player: PlayerCode,
    position: Position,
    *,
    candidate_count: int,
    best_gain: float | None,
    threshold: float,
    budget: TenthsOfMillion | None = None,
    cheapest_upgrade_shortfall: TenthsOfMillion | None = None,
) -> tuple[Reason, ...]:
    """Why no transfer was proposed for this player.

    A player who simply does not appear in the output reads as one the system
    overlooked. Saying *why* — that his position offered N legal replacements,
    that the best of them gained less than the bar, that the budget stopped
    short of the next upgrade — turns an absence into an answer.
    """
    reasons: list[Reason] = [
        Reason(
            code=ReasonCode.POSITION_LOCKED,
            polarity=ReasonPolarity.CONTEXT,
            subject=player,
            evidence={"candidate_count": float(candidate_count)},
            context={"position": position.value},
            weight=0.0,
        )
    ]

    if best_gain is not None:
        reasons.append(
            Reason(
                code=ReasonCode.NO_UPGRADE_AVAILABLE,
                polarity=ReasonPolarity.CONTEXT,
                subject=player,
                evidence={"best_gain": best_gain, "threshold": threshold},
                weight=0.0,
            )
        )

    if budget is not None and cheapest_upgrade_shortfall is not None:
        reasons.append(
            Reason(
                code=ReasonCode.BUDGET_LOCKED,
                polarity=ReasonPolarity.CONTEXT,
                subject=player,
                evidence={
                    "budget": budget / 10.0,
                    "shortfall": cheapest_upgrade_shortfall / 10.0,
                },
                weight=0.0,
            )
        )

    return tuple(reasons)
