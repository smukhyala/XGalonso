"""Per-player justification.

**The question this answers.** A manager looking at a projected 2.2 next to a
projected 2.6 has two questions the product could not previously answer: *why is
he 2.2*, and *who would you put there instead*. Both were computable — the
points breakdown is validated to sum to the total, and the optimizer scores
every legal replacement — and neither reached the screen. Only the two players
in the single recommended move had any explanation attached, so thirteen of
fifteen squad members were unexplained by construction.

**Why the start verdict is computed rather than inferred.** The tempting
shortcut is to compare a player's expected points against the lowest-scoring
starter. That is wrong whenever positional minima bind: a defender projected at
2.2 can start ahead of a midfielder projected at 3.0 because the shape requires
at least three defenders, and a comparison of the two numbers alone would report
the opposite. So the margin is measured by re-selecting the XI with the player
forced in or held out, which prices the constraint instead of ignoring it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from xg_alonso.contracts.evidence import FeatureValue
from xg_alonso.contracts.identifiers import PlayerCode, TenthsOfMillion
from xg_alonso.contracts.prediction import PlayerPrediction, PointsBreakdown
from xg_alonso.contracts.reason_codes import Reason
from xg_alonso.contracts.recommendation import PlayerBestMove, TransferOption
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.domain.rules import SquadRules
from xg_alonso.explanations.reasons import PopulationStats, build_player_reasons

__all__ = [
    "ArchetypeVerdict",
    "Comparable",
    "PlayerExplanation",
    "StartVerdict",
    "explain_player",
    "explain_squad",
]


@dataclass(frozen=True)
class StartVerdict:
    """Whether a player makes the XI, and what that decision is worth."""

    is_starter: bool

    margin: float
    """Points at stake, always as a gain relative to the alternative.

    For a starter, what the XI would lose without him. For a bench player, what
    the XI would lose by playing him — reported as a negative number, so the
    sign always means the same thing: positive is the case for the current
    decision.
    """

    forced_by_quota: bool
    """Whether a positional minimum, rather than his projection, put him in.

    Worth stating plainly. A defender who starts only because the shape demands
    three is a different kind of starter from one who earned the place, and a
    manager deciding whether to transfer him needs to know which.
    """


@dataclass(frozen=True)
class Comparable:
    """One player this player resembles, with the gap between them."""

    player_code: PlayerCode
    expected_points: float
    price: TenthsOfMillion | None


@dataclass(frozen=True)
class ArchetypeVerdict:
    """What kind of player this is, and how he ranks among that kind.

    **The rank here is against his own archetype, and that is a weaker claim
    than it looks.** Archetypes are clustered on style *and* output, so a
    cluster is partly defined by how good its members are — "best in his
    cluster" would be close to circular. The rank is reported because it is a
    fact a reader can check, not as an argument that he is the right pick. The
    argument for picking him lives in `PlayerExplanation.reasons`, which
    compares him against his position and his price, not against his cluster.
    """

    label: str
    size: int
    rank_within: int
    """His position by expected points inside his archetype, 1 being highest."""

    comparables: tuple[Comparable, ...]


@dataclass(frozen=True)
class PlayerExplanation:
    """Everything the product can honestly say about one squad member."""

    player_code: PlayerCode
    expected_points: float
    breakdown: PointsBreakdown
    evidence: tuple[FeatureValue, ...]
    """Panel values that distinguish him, most distinguishing first.

    Filtered to the notable ones. A player at the 51st percentile for everything
    has nothing to say about himself, and listing fourteen middling numbers to
    prove it would bury the players who do.
    """

    reasons: tuple[Reason, ...]
    start_verdict: StartVerdict
    replacements: tuple[TransferOption, ...]
    no_replacement_reasons: tuple[Reason, ...]
    archetype: ArchetypeVerdict | None = None

    @property
    def has_upgrade(self) -> bool:
        return bool(self.replacements)


def _start_verdict(
    player: PlayerCode,
    *,
    picks: Sequence[SquadPick],
    predictions: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    is_starter: bool,
    baseline_points: float,
) -> StartVerdict:
    """Price the start-or-bench decision by re-selecting the eleven.

    Imported locally because ``explanations`` does not otherwise depend on the
    optimizer, and a module-level import would make the dependency permanent for
    every caller of this package including those that only render prose.
    """
    from xg_alonso.optimization.lineup import best_starting_xi

    if is_starter:
        without = [p for p in picks if p.player_code != player]
        alternative = best_starting_xi(without, predictions, rules).expected_points
        margin = baseline_points - alternative

        # If the XI without him scores no less, his place is the quota's doing
        # rather than his own — the squad simply has no other player of his
        # position to put there.
        prediction = predictions.get(player)
        own_points = 0.0 if prediction is None else prediction.expected_points
        forced = margin < own_points * 0.5
        return StartVerdict(is_starter=True, margin=margin, forced_by_quota=forced)

    forced_in = best_starting_xi(picks, predictions, rules, required=player).expected_points
    return StartVerdict(
        is_starter=False,
        margin=forced_in - baseline_points,
        forced_by_quota=False,
    )


def explain_player(
    prediction: PlayerPrediction,
    *,
    picks: Sequence[SquadPick],
    predictions: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    starters: frozenset[PlayerCode],
    baseline_points: float,
    best_move: PlayerBestMove | None = None,
    alternatives: Sequence[TransferOption] = (),
    population: PopulationStats | None = None,
    price: TenthsOfMillion | None = None,
    chance_of_playing: float | None = None,
    archetype: ArchetypeVerdict | None = None,
) -> PlayerExplanation:
    """Assemble one player's justification.

    Ranking is not performed here. ``best_move`` and ``alternatives`` come from
    the board the optimizer already produced, so what a player's row says about
    his replacements is the same computation that produced the headline
    recommendation — there is no second opinion to drift.
    """
    verdict = _start_verdict(
        prediction.player_code,
        picks=picks,
        predictions=predictions,
        rules=rules,
        is_starter=prediction.player_code in starters,
        baseline_points=baseline_points,
    )

    replacements: list[TransferOption] = []
    if best_move is not None and best_move.option is not None:
        replacements.append(best_move.option)
    for option in alternatives:
        if option.move.player_out != prediction.player_code:
            continue
        if any(option.move.player_in == kept.move.player_in for kept in replacements):
            continue
        replacements.append(option)

    return PlayerExplanation(
        player_code=prediction.player_code,
        expected_points=prediction.expected_points,
        breakdown=prediction.breakdown,
        evidence=(
            () if prediction.feature_evidence is None else prediction.feature_evidence.notable()
        ),
        reasons=build_player_reasons(
            prediction,
            population=population,
            price=price,
            chance_of_playing=chance_of_playing,
        ),
        start_verdict=verdict,
        replacements=tuple(replacements[:3]),
        no_replacement_reasons=(
            () if best_move is None or best_move.option is not None else best_move.reasons
        ),
        archetype=archetype,
    )


def explain_squad(
    *,
    picks: Sequence[SquadPick],
    predictions: Mapping[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    starters: frozenset[PlayerCode],
    baseline_points: float,
    board_by_player: Mapping[PlayerCode, PlayerBestMove] | None = None,
    alternatives: Sequence[TransferOption] = (),
    population: PopulationStats | None = None,
    prices: Mapping[PlayerCode, TenthsOfMillion] | None = None,
    chances_of_playing: Mapping[PlayerCode, float] | None = None,
    archetypes: Mapping[PlayerCode, ArchetypeVerdict] | None = None,
) -> list[PlayerExplanation]:
    """Explain every squad member, in squad order.

    Every pick with a prediction is explained. A pick without one is skipped
    rather than explained with zeros, because "we have no projection for him" is
    a different statement from "we project him at nothing", and the second is
    what a zero would say.
    """
    board_by_player = board_by_player or {}
    prices = prices or {}
    chances_of_playing = chances_of_playing or {}
    archetypes = archetypes or {}

    explanations: list[PlayerExplanation] = []
    for pick in picks:
        code = pick.player_code
        prediction = predictions.get(code)
        if prediction is None:
            continue
        explanations.append(
            explain_player(
                prediction,
                picks=picks,
                predictions=predictions,
                rules=rules,
                starters=starters,
                baseline_points=baseline_points,
                best_move=board_by_player.get(code),
                alternatives=alternatives,
                population=population,
                price=prices.get(code),
                chance_of_playing=chances_of_playing.get(code),
                archetype=archetypes.get(code),
            )
        )
    return explanations
