"""Single-transfer optimization.

Per `CLAUDE.md`, the optimizer is the product and predictions merely feed it. So
this module owns the decision, and its output is judged against one question:
**does it beat holding?**

**Why exhaustive search rather than MILP.** A single transfer is one of roughly
15 outgoing players times a few hundred affordable replacements — tens of
thousands of candidates, each evaluated with integer arithmetic and a few
comparisons. That runs in well under a second. A solver would add a dependency,
a modelling layer and a class of "infeasible" failures that are hard to explain
to a user, in exchange for nothing at this size. MILP earns its place when
multi-transfer packages arrive and the combinatorics stop being trivial.

Every candidate is checked against the real constraints before it is scored, so
an illegal transfer can never be ranked, let alone recommended. A recommendation
the game would reject costs the user a transfer to discover.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.prediction import PlayerPrediction, Position
from xg_alonso.contracts.reason_codes import Reason, ReasonCode, ReasonPolarity
from xg_alonso.contracts.recommendation import (
    BaselineComparison,
    PlayerBestMove,
    TransferBoard,
    TransferMove,
    TransferOption,
    TransferPackage,
    TransferRecommendation,
)
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.rules import SquadRules
from xg_alonso.explanations.reasons import (
    PopulationStats,
    build_no_move_reasons,
    build_transfer_reasons,
)
from xg_alonso.optimization.horizon import (
    DEFAULT_DISCOUNT,
    HorizonValue,
    value_over_horizon,
)
from xg_alonso.optimization.lineup import starting_xi_points

__all__ = [
    "DEFAULT_BOARD_SIZE",
    "HOLD_BASELINE",
    "Candidate",
    "TransferCandidate",
    "best_single_transfer",
    "build_transfer_board",
    "hold_expected_points",
    "horizon_valued",
    "rank_single_transfers",
]

#: How many globally-best moves the board carries. Eight rather than three
#: because a manager comparing options needs enough of the market to see the
#: shape of it, and rather than fifty because a list nobody reads to the end is
#: a list that hides its own top.
DEFAULT_BOARD_SIZE: Final[int] = 8

HOLD_BASELINE: Final[str] = "hold"

#: How much of a point of predicted gain must survive the uncertainty penalty
#: before a transfer is worth recommending. Set above zero because a marginal
#: gain inside the noise is not a reason to spend a transfer.
_MIN_NET_GAIN: Final[float] = 0.15

#: Weight on prediction uncertainty in the objective. A transfer that gains 1.0
#: expected points with wide variance is worse than one gaining 0.9 with narrow.
_RISK_WEIGHT: Final[float] = 0.25


@dataclass(frozen=True)
class Candidate:
    """A player who could be bought, with everything needed to price the move."""

    player_code: PlayerCode
    position: Position
    team_id: TeamId
    price: TenthsOfMillion
    prediction: PlayerPrediction


@dataclass(frozen=True)
class TransferCandidate:
    """One evaluated out-for-in move."""

    out_pick: SquadPick
    incoming: Candidate
    out_prediction: PlayerPrediction
    gross_gain: float
    hit_cost: int
    risk_penalty: float

    @property
    def net_gain(self) -> float:
        """Expected points gained after the hit and the uncertainty penalty."""
        return self.gross_gain - self.hit_cost - self.risk_penalty

    @property
    def bank_after(self) -> TenthsOfMillion:
        return TenthsOfMillion(self.out_pick.selling_price - self.incoming.price)


def horizon_valued(
    predictions: dict[PlayerCode, PlayerPrediction],
    projections: Mapping[PlayerCode, Sequence[float]],
    *,
    discount: float = DEFAULT_DISCOUNT,
) -> tuple[dict[PlayerCode, PlayerPrediction], dict[PlayerCode, HorizonValue]]:
    """Re-price every prediction over the horizon rather than the next gameweek.

    **Why substitute the predictions rather than change the search.** A transfer
    is chosen by scoring the eleven a squad would field, and that scoring runs
    through `starting_xi_points`, captaincy and the bench in several places. If
    the horizon were threaded through each of them, some paths would use it and
    others would not, and the ones that did not would be exactly the subtle
    cases — a captain chosen on next week while the squad was chosen on five.

    Swapping the number every one of those paths already reads is a single
    substitution point that cannot be partially applied. The breakdown is scaled
    with the total so the two still agree, which the contract enforces.

    Returns:
        Re-priced predictions, and the horizon detail for explanation.
    """
    repriced: dict[PlayerCode, PlayerPrediction] = {}
    values: dict[PlayerCode, HorizonValue] = {}

    for code, prediction in predictions.items():
        weekly = projections.get(code)
        if not weekly:
            repriced[code] = prediction
            continue

        value = value_over_horizon(weekly, discount=discount)
        values[code] = value

        base = prediction.expected_points
        # Scale rather than replace, so the component breakdown keeps its shape
        # and continues to sum to the total it explains.
        factor = (value.total / base) if abs(base) > 1e-9 else 0.0
        breakdown = prediction.breakdown
        repriced[code] = prediction.model_copy(
            update={
                "breakdown": breakdown.model_copy(
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
                ),
                "expected_points": value.total,
                # Uncertainty grows with the horizon: a five-week projection is
                # a longer extrapolation from the same features, and pricing it
                # as confidently as next week's would be the whole reason a
                # horizon objective goes wrong.
                "expected_points_sd": prediction.expected_points_sd
                * (1.0 + 0.1 * (value.horizon - 1)),
            }
        )

    return repriced, values


def hold_expected_points(
    squad: SquadState,
    predictions: dict[PlayerCode, PlayerPrediction],
    rules: SquadRules,
) -> float:
    """Expected points from doing nothing — the baseline everything is measured against.

    Scored as the **best legal XI with the captain doubled**, not as the squad's
    stored slot order. Two earlier versions of this got it wrong in ways that
    compounded:

    - Summing the stored starters credited whatever eleven the payload happened
      to name, so a transfer that improved a bench player looked like a full
      gain even though it changed nothing that scores.
    - No captain was ever chosen, so the largest single term in FPL scoring —
      a doubled return — was absent from every measurement.
    """
    return starting_xi_points(squad.picks, predictions, rules)


def _club_counts(picks: Sequence[SquadPick]) -> dict[TeamId, int]:
    counts: dict[TeamId, int] = {}
    for pick in picks:
        counts[pick.team_id] = counts.get(pick.team_id, 0) + 1
    return counts


def rank_single_transfers(
    squad: SquadState,
    *,
    candidates: Sequence[Candidate],
    predictions: dict[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    sellable: frozenset[PlayerCode] | None = None,
) -> list[TransferCandidate]:
    """Every legal single transfer, best net gain first.

    Legality is enforced before scoring, so an illegal move never enters the
    ranking. Checks applied: position match, affordability against the selling
    price plus bank, the three-per-club limit, and no buying a player already
    owned.

    Args:
        sellable: Squad members the manager is willing to sell. ``None`` means
            all of them, which is the unconstrained behaviour every existing
            caller gets.

            **A lock is a filter, not a penalty.** Expressing "keep Haaland" as
            a large negative score still lets a good enough alternative buy him
            out, which is precisely what the manager said not to do — and it
            would do so silently, since the recommendation would look like any
            other. Removing him from the search makes the constraint
            unbreakable rather than merely expensive.
    """
    owned = {pick.player_code for pick in squad.picks}
    counts = _club_counts(squad.picks)
    hit_cost = 0 if squad.free_transfers >= 1 else rules.hit_cost_per_transfer

    # The bar every candidate must clear: what this squad scores untouched.
    hold_points = starting_xi_points(squad.picks, predictions, rules)

    evaluated: list[TransferCandidate] = []

    for pick in squad.picks:
        if sellable is not None and pick.player_code not in sellable:
            continue
        out_prediction = predictions.get(pick.player_code)
        if out_prediction is None:
            continue

        budget = TenthsOfMillion(pick.selling_price + squad.bank)

        for candidate in candidates:
            if candidate.player_code in owned:
                continue
            if candidate.position is not pick.position:
                continue
            if candidate.price > budget:
                continue

            # Selling one player from a club frees a slot in that club.
            club_after = counts.get(candidate.team_id, 0) + (
                -1 if candidate.team_id == pick.team_id else 0
            )
            if club_after >= rules.max_per_club:
                continue

            # Score the squad this move would actually produce, then re-pick
            # the XI. A straight difference between the two players is wrong
            # whenever either sits on the bench: replacing a benched player with
            # a slightly better benched player gains nothing at all.
            after = [p for p in squad.picks if p.player_code != pick.player_code]
            after.append(
                pick.model_copy(
                    update={
                        "player_code": candidate.player_code,
                        "position": candidate.position,
                        "team_id": candidate.team_id,
                        "purchase_price": candidate.price,
                        "current_price": candidate.price,
                        "selling_price": candidate.price,
                    }
                )
            )
            gross = starting_xi_points(after, predictions, rules) - hold_points

            # Combine the two uncertainties; a swap inherits both.
            risk = _RISK_WEIGHT * (
                candidate.prediction.expected_points_sd + out_prediction.expected_points_sd
            )

            evaluated.append(
                TransferCandidate(
                    out_pick=pick,
                    incoming=candidate,
                    out_prediction=out_prediction,
                    gross_gain=gross,
                    hit_cost=hit_cost,
                    risk_penalty=risk,
                )
            )

    # Sort by net gain, breaking ties on player code so the ranking is stable
    # across runs rather than dependent on candidate iteration order.
    evaluated.sort(key=lambda c: (-c.net_gain, c.incoming.player_code, c.out_pick.player_code))
    return evaluated


def _build_reasons(
    candidate: TransferCandidate,
    *,
    population: PopulationStats | None = None,
    candidate_count: int | None = None,
) -> tuple[Reason, ...]:
    """Ground a move in the evidence that actually drove it.

    Delegates to the shared builder so a move explained on the transfer board
    and the same move explained on a player's row cannot disagree.
    """
    return build_transfer_reasons(
        incoming=candidate.incoming.prediction,
        outgoing=candidate.out_prediction,
        gross_gain=candidate.gross_gain,
        incoming_price=candidate.incoming.price,
        outgoing_price=candidate.out_pick.selling_price,
        population=population,
        candidate_count=candidate_count,
    )


def _as_option(
    candidate: TransferCandidate,
    *,
    population: PopulationStats | None,
    candidate_count: int | None,
) -> TransferOption:
    return TransferOption(
        move=TransferMove(
            player_out=candidate.out_pick.player_code,
            player_in=candidate.incoming.player_code,
            selling_price=candidate.out_pick.selling_price,
            purchase_price=candidate.incoming.price,
        ),
        gross_gain=round(candidate.gross_gain, 6),
        net_gain=round(candidate.net_gain, 6),
        hit_cost=candidate.hit_cost,
        risk_penalty=round(candidate.risk_penalty, 6),
        bank_after=candidate.bank_after,
        reasons=_build_reasons(candidate, population=population, candidate_count=candidate_count),
    )


def build_transfer_board(
    squad: SquadState,
    *,
    candidates: Sequence[Candidate],
    predictions: dict[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    population: PopulationStats | None = None,
    size: int = DEFAULT_BOARD_SIZE,
    sellable: frozenset[PlayerCode] | None = None,
) -> TransferBoard:
    """Every move worth showing, plus the best move for each squad member.

    The ranking already exists — :func:`rank_single_transfers` has always scored
    the whole market and returned it sorted, and the product then displayed
    exactly one row of it. This assembles what was being thrown away.

    ``by_player`` is the part that answers "why not him?". It covers all fifteen
    picks, so a player with nothing worth doing carries a grounded reason rather
    than simply being absent, and a user can see the move that *was* available
    and how far short it fell.
    """
    ranked = rank_single_transfers(
        squad, candidates=candidates, predictions=predictions, rules=rules, sellable=sellable
    )

    # How many players could legally have taken each slot. Counted per position
    # because that is the constraint a user is most likely to be surprised by:
    # a forward cannot be swapped for a defender however much better the
    # defender looks.
    legal_by_player: dict[PlayerCode, int] = {}
    best_by_player: dict[PlayerCode, TransferCandidate] = {}
    for evaluated in ranked:
        code = evaluated.out_pick.player_code
        legal_by_player[code] = legal_by_player.get(code, 0) + 1
        if code not in best_by_player:
            # `ranked` is sorted by net gain, so the first sighting is the best.
            best_by_player[code] = evaluated

    top = tuple(
        _as_option(evaluated, population=population, candidate_count=None)
        for evaluated in ranked[:size]
    )

    entries: list[PlayerBestMove] = []
    for pick in squad.picks:
        code = pick.player_code
        legal = legal_by_player.get(code, 0)
        best = best_by_player.get(code)

        if sellable is not None and code not in sellable:
            # Held by the manager's own instruction. He still appears on the
            # board — "why not him?" must have an answer for every squad member,
            # and "you told me not to" is a better answer than silence.
            entries.append(
                PlayerBestMove(
                    player_out=code,
                    position=pick.position,
                    legal_replacements=0,
                    reasons=(
                        Reason(
                            code=ReasonCode.CONSTRAINT_HELD,
                            polarity=ReasonPolarity.CONTEXT,
                            subject=code,
                            # No evidence: the template renders no placeholders,
                            # and the contract refuses a Reason carrying values
                            # nobody reads.
                            evidence={},
                            weight=0.0,
                        ),
                    ),
                )
            )
            continue

        if best is not None and best.net_gain >= _MIN_NET_GAIN:
            entries.append(
                PlayerBestMove(
                    player_out=code,
                    position=pick.position,
                    legal_replacements=legal,
                    option=_as_option(best, population=population, candidate_count=legal),
                )
            )
            continue

        entries.append(
            PlayerBestMove(
                player_out=code,
                position=pick.position,
                legal_replacements=legal,
                reasons=build_no_move_reasons(
                    code,
                    pick.position,
                    candidate_count=legal,
                    best_gain=None if best is None else round(best.net_gain, 6),
                    threshold=_MIN_NET_GAIN,
                    budget=TenthsOfMillion(pick.selling_price + squad.bank),
                    cheapest_upgrade_shortfall=_shortfall_to_upgrade(
                        pick=pick,
                        squad=squad,
                        candidates=candidates,
                        predictions=predictions,
                    ),
                ),
            )
        )

    return TransferBoard(
        top=top,
        by_player=tuple(entries),
        candidates_considered=len(candidates),
        legal_moves=len(ranked),
    )


def _shortfall_to_upgrade(
    *,
    pick: SquadPick,
    squad: SquadState,
    candidates: Sequence[Candidate],
    predictions: dict[PlayerCode, PlayerPrediction],
) -> TenthsOfMillion | None:
    """How much more money the cheapest better player in this position costs.

    Returns ``None`` when budget is not the binding constraint — either an
    affordable upgrade exists, or no better player exists at any price. Saying
    "budget stopped you" when it did not would be a plausible, wrong cause, and
    those are the explanations that do the most damage to trust.
    """
    current = predictions.get(pick.player_code)
    if current is None:
        return None

    budget = pick.selling_price + squad.bank
    owned = {p.player_code for p in squad.picks}

    cheapest_better: int | None = None
    for candidate in candidates:
        if candidate.player_code in owned or candidate.position is not pick.position:
            continue
        if candidate.prediction.expected_points <= current.expected_points:
            continue
        if candidate.price <= budget:
            return None  # An upgrade was affordable, so money was not the limit.
        if cheapest_better is None or candidate.price < cheapest_better:
            cheapest_better = int(candidate.price)

    if cheapest_better is None:
        return None
    return TenthsOfMillion(cheapest_better - budget)


def best_single_transfer(
    squad: SquadState,
    *,
    candidates: Sequence[Candidate],
    predictions: dict[PlayerCode, PlayerPrediction],
    rules: SquadRules,
    entry_id: EntryId,
    gameweek: GameweekId,
    generated_at: datetime,
    run_id: str,
    optimizer_config_hash: str,
    horizon_gameweeks: int = 1,
    population: PopulationStats | None = None,
    with_board: bool = True,
    sellable: frozenset[PlayerCode] | None = None,
) -> TransferRecommendation:
    """The best single transfer, or an explicit hold when none clears the bar.

    Returning a hold is a real answer, not a failure. Most gameweeks the correct
    move is to keep the transfer, and a tool that always finds something to do
    is a tool that costs its user points.

    Args:
        population: League reference values, so reasons can say where a player
            sits rather than only what his numbers are.
        with_board: Attach the alternatives this was chosen from. A hold in
            particular is much more convincing when the moves it beat are
            visible, since "nothing was good enough" and "nothing was
            considered" look identical otherwise.
    """
    baseline = hold_expected_points(squad, predictions, rules)
    ranked = rank_single_transfers(
        squad, candidates=candidates, predictions=predictions, rules=rules, sellable=sellable
    )

    board = (
        build_transfer_board(
            squad,
            candidates=candidates,
            predictions=predictions,
            rules=rules,
            population=population,
            sellable=sellable,
        )
        if with_board
        else None
    )

    best = ranked[0] if ranked else None

    if best is None or best.net_gain < _MIN_NET_GAIN:
        package = TransferPackage(
            moves=(),
            transfers_used=0,
            free_transfers_available=squad.free_transfers,
            hit_cost=0,
            bank_before=squad.bank,
            bank_after=squad.bank,
        )
        return TransferRecommendation(
            entry_id=entry_id,
            gameweek=gameweek,
            package=package,
            comparison=BaselineComparison(
                baseline_name=HOLD_BASELINE,
                baseline_expected_points=baseline,
                candidate_expected_points=baseline,
                horizon_gameweeks=horizon_gameweeks,
            ),
            reasons=(),
            expected_points_gain=0.0,
            risk_score=0.0,
            generated_at=generated_at,
            run_id=run_id,
            optimizer_config_hash=optimizer_config_hash,
            board=board,
        )

    move = TransferMove(
        player_out=best.out_pick.player_code,
        player_in=best.incoming.player_code,
        selling_price=best.out_pick.selling_price,
        purchase_price=best.incoming.price,
    )
    package = TransferPackage(
        moves=(move,),
        transfers_used=1,
        free_transfers_available=squad.free_transfers,
        hit_cost=best.hit_cost,
        bank_before=squad.bank,
        bank_after=TenthsOfMillion(squad.bank - move.net_cost),
    )

    return TransferRecommendation(
        entry_id=entry_id,
        gameweek=gameweek,
        package=package,
        comparison=BaselineComparison(
            baseline_name=HOLD_BASELINE,
            baseline_expected_points=baseline,
            candidate_expected_points=baseline + best.gross_gain - best.hit_cost,
            horizon_gameweeks=horizon_gameweeks,
        ),
        reasons=_build_reasons(
            best,
            population=population,
            candidate_count=(
                None
                if board is None
                else next(
                    (
                        entry.legal_replacements
                        for entry in board.by_player
                        if entry.player_out == best.out_pick.player_code
                    ),
                    None,
                )
            ),
        ),
        expected_points_gain=round(best.gross_gain - best.hit_cost, 6),
        expected_value_gain=TenthsOfMillion(0),
        risk_score=round(best.risk_penalty, 6),
        generated_at=generated_at,
        run_id=run_id,
        optimizer_config_hash=optimizer_config_hash,
        board=board,
    )
