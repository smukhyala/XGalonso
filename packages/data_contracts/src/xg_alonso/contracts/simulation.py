"""What actually happened to a squad in a gameweek, decomposed.

**Why the decomposition is a contract rather than a convention.** A season's
result is the sum of several very different things — the eleven that were
picked, the substitutes the game brought on afterwards, the doubled armband,
and the points paid for transfers. Reporting only the total makes a policy that
wins on captaincy indistinguishable from one that wins on selection, and a
harness that quietly conflates two of them is impossible to catch by reading
its output.

So :class:`GameweekSimulation` keeps them separate *and* validates that they
reconcile. ``starters + autosub + captaincy == policy_points`` is checked on
construction, which means the decomposition cannot silently stop adding up.

**The vocabulary is deliberately outcome-shaped, not decision-shaped.** A
substitution records who came off, who came on, and from which bench slot,
because "why did this squad score what it scored" is a question about the
game's own mechanics. Skipped substitutions are recorded too, with a reason:
a bench player who never came on because the formation would have been illegal
is a different fact from one who was simply not needed, and a manager reading
their week wants to know which.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xg_alonso.contracts.constraints import SquadViolation
from xg_alonso.contracts.identifiers import EntryId, GameweekId, PlayerCode, Season
from xg_alonso.contracts.prediction import Position

__all__ = [
    "AutosubResult",
    "CaptainSource",
    "Captaincy",
    "GameweekSimulation",
    "SeasonSimulation",
    "SkipReason",
    "SkippedSubstitution",
    "Substitution",
    "SubstitutionReason",
]


class SubstitutionReason(StrEnum):
    """Why a substitute came on. One member today; FPL has one rule."""

    NO_MINUTES = "no_minutes"


class SkipReason(StrEnum):
    """Why a bench player did *not* come on."""

    BENCH_PLAYER_DID_NOT_PLAY = "bench_player_did_not_play"
    """He was on the bench and did not feature himself, so he cannot cover."""

    FORMATION_WOULD_BE_ILLEGAL = "formation_would_be_illegal"
    """Every vacancy he could fill would leave a shape the game forbids."""

    NO_VACANCY = "no_vacancy"
    """Every starter played. Nothing to cover."""


class CaptainSource(StrEnum):
    """Who ended up wearing the armband, and why."""

    CAPTAIN = "captain"
    VICE_CAPTAIN = "vice_captain"
    NONE = "none"
    """Neither played. FPL doubles nobody in that case."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Substitution(_Frozen):
    """One automatic substitution the game made after the deadline."""

    player_off: PlayerCode
    player_on: PlayerCode
    bench_slot: int = Field(ge=1, description="The incoming player's squad slot")
    off_position: Position
    on_position: Position
    reason: SubstitutionReason = SubstitutionReason.NO_MINUTES


class SkippedSubstitution(_Frozen):
    """A bench player who was considered and passed over, with the reason."""

    player_on: PlayerCode
    bench_slot: int = Field(ge=1)
    reason: SkipReason


class AutosubResult(_Frozen):
    """The eleven that actually scored, and how it got there."""

    final_xi: tuple[PlayerCode, ...]
    substitutions: tuple[Substitution, ...] = ()
    skipped: tuple[SkippedSubstitution, ...] = ()

    @model_validator(mode="after")
    def _no_player_twice(self) -> AutosubResult:
        if len(set(self.final_xi)) != len(self.final_xi):
            raise ValueError("the same player appears twice in the final eleven")
        off = [s.player_off for s in self.substitutions]
        if len(set(off)) != len(off):
            raise ValueError("the same player was substituted off more than once")
        on = [s.player_on for s in self.substitutions]
        if len(set(on)) != len(on):
            raise ValueError("the same player was substituted on more than once")
        return self


class Captaincy(_Frozen):
    """Who was doubled, and on what authority."""

    holder: PlayerCode | None
    source: CaptainSource
    multiplier: int = Field(ge=1)

    @model_validator(mode="after")
    def _source_matches_holder(self) -> Captaincy:
        if (self.source is CaptainSource.NONE) != (self.holder is None):
            raise ValueError(
                f"source {self.source} disagrees with holder {self.holder}: nobody "
                "wears the armband exactly when the source is 'none'"
            )
        if self.source is CaptainSource.NONE and self.multiplier != 1:
            raise ValueError(
                f"multiplier {self.multiplier} with no armband holder; an unworn "
                "armband multiplies nothing"
            )
        return self


class GameweekSimulation(_Frozen):
    """One gameweek, for one policy, with every quantity kept apart.

    The seven numbers the evaluation layer must never conflate live here as
    separate fields: the immediate decision delta, the realised weekly
    incremental, the hit cost, and the autosub, captaincy, bench and
    as-picked contributions to the total.
    """

    season: Season
    gameweek: GameweekId
    entry_id: EntryId

    # --- the decision ----------------------------------------------------
    transfers_made: int = Field(ge=0)
    player_out: PlayerCode | None = None
    player_in: PlayerCode | None = None
    free_transfers_before: int = Field(ge=0)
    free_transfers_after: int = Field(ge=0)
    hit_cost: int = Field(ge=0)
    rejected: tuple[SquadViolation, ...] = ()
    """Why a recommended transfer was refused. Empty when none was, or when it stood."""
    predicted_gain: float = 0.0
    decision_delta: int = 0
    """What the incoming player outscored the outgoing one by, *this* gameweek.

    This isolates the week's decision. ``incremental`` cannot: it compares two
    squads that have been diverging for weeks, so once the acting squad is
    ahead it wins every gameweek regardless of what was decided in it.
    """

    # --- realised points, decomposed; these three sum to policy_points ----
    starters_points: int = 0
    """The eleven picked at the deadline, armband counted once."""
    autosub_points: int = 0
    """Added by players the game brought on afterwards."""
    captaincy_points: int = 0
    """The *extra* from doubling, not the armband player's whole score."""

    policy_points: int = 0
    bench_points: int = 0
    """Left on the bench after autosubs. Never added to the total."""

    # --- the baseline, scored through the identical path ------------------
    baseline_name: str = "hold"
    baseline_points: int = 0

    # --- what the game did ------------------------------------------------
    autosubs: tuple[Substitution, ...] = ()
    skipped_autosubs: tuple[SkippedSubstitution, ...] = ()
    captaincy: Captaincy | None = None
    blank_players: tuple[PlayerCode, ...] = ()
    """Squad members whose club had no fixture. Distinct from 'played badly'."""
    double_players: tuple[PlayerCode, ...] = ()

    @model_validator(mode="after")
    def _decomposition_reconciles(self) -> GameweekSimulation:
        parts = self.starters_points + self.autosub_points + self.captaincy_points
        if parts != self.policy_points:
            raise ValueError(
                f"the decomposition does not reconcile: starters {self.starters_points} "
                f"+ autosubs {self.autosub_points} + captaincy {self.captaincy_points} "
                f"= {parts}, but policy_points is {self.policy_points}. These are "
                "reported separately precisely so they cannot drift apart."
            )
        return self

    @property
    def net_policy_points(self) -> int:
        """What the manager actually banks, after the hit."""
        return self.policy_points - self.hit_cost

    @property
    def incremental(self) -> int:
        """Points gained over the baseline, net of hits."""
        return self.net_policy_points - self.baseline_points

    @property
    def transfer_made(self) -> bool:
        return self.transfers_made > 0


class SeasonSimulation(_Frozen):
    """A completed walk. Metric definitions match ``BacktestResult`` exactly.

    They have to: until the new evaluation framework can reproduce the existing
    headline, the two must be comparable number for number.
    """

    season: Season
    entry_id: EntryId
    baseline_name: str = "hold"
    gameweeks: tuple[GameweekSimulation, ...] = ()

    @property
    def cumulative_incremental(self) -> tuple[int, ...]:
        running = 0
        totals: list[int] = []
        for week in self.gameweeks:
            running += week.incremental
            totals.append(running)
        return tuple(totals)

    @property
    def total_incremental(self) -> int:
        return sum(week.incremental for week in self.gameweeks)

    @property
    def total_hit_cost(self) -> int:
        return sum(week.hit_cost for week in self.gameweeks)

    @property
    def total_autosub_points(self) -> int:
        return sum(week.autosub_points for week in self.gameweeks)

    @property
    def total_captaincy_points(self) -> int:
        return sum(week.captaincy_points for week in self.gameweeks)

    @property
    def total_bench_points(self) -> int:
        return sum(week.bench_points for week in self.gameweeks)

    @property
    def transfers_made(self) -> int:
        return sum(1 for week in self.gameweeks if week.transfer_made)

    @property
    def decision_win_rate(self) -> float:
        """Share of transfers where the incoming player outscored the outgoing one.

        The honest per-decision measure. A coin-flipping policy lands near 50%;
        near 100% is a bug in the harness rather than a very good model.
        """
        acted = [w for w in self.gameweeks if w.transfer_made]
        if not acted:
            return 0.0
        return sum(1 for w in acted if w.decision_delta > 0) / len(acted)

    @property
    def mean_decision_delta(self) -> float:
        acted = [w for w in self.gameweeks if w.transfer_made]
        if not acted:
            return 0.0
        return sum(w.decision_delta for w in acted) / len(acted)

    @property
    def mean_regret(self) -> float:
        """Average loss on transfers that turned out worse than holding.

        Reported apart from the mean gain because the two are not symmetric:
        one badly-timed hit can erase several good weeks.
        """
        losses = [
            -w.decision_delta for w in self.gameweeks if w.transfer_made and w.decision_delta < 0
        ]
        if not losses:
            return 0.0
        return sum(losses) / len(losses)

    @property
    def calibration_error(self) -> float:
        """Mean absolute gap between predicted and realised gain."""
        acted = [w for w in self.gameweeks if w.transfer_made]
        if not acted:
            return 0.0
        return sum(abs(w.predicted_gain - w.decision_delta) for w in acted) / len(acted)
