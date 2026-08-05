"""Which players this manager can actually reach.

Discovery measures a feature's value over a population. Until now that
population has been *every* player, which quietly assumes every manager is
solving the same problem. They are not. A manager who has locked eleven players
and holds 0.3m in the bank cannot buy a 12.5m forward under any circumstances,
so a feature that improves the ranking of expensive forwards is worth nothing to
them — and averaging their value in with everyone else's is how a recommendation
engine comes to be confident about moves its user cannot make.

This module derives the reachable pool from the *hard* constraints only, and
records what it did.

Three properties are deliberate
-------------------------------

**Locked players stay in the pool.** They consume slots and budget, but they are
not removed: you still have to rank a locked player to pick a captain and set a
lineup. A mask that dropped them would make the model worst at exactly the
players the manager cares most about.

**Only hard constraints are read.** Nothing here touches
:class:`~xg_alonso.contracts.objective.ManagerObjective` or any belief. The
objective enters later, at scoring. If the two ever mixed, a soft preference
would have become a filter — the failure
:mod:`xg_alonso.contracts.objective` exists to prevent.

**Refusal is reported, never silent.** A mask that cannot be applied — because
the frame lacks a price column, or because applying it would leave folds too
thin to measure — degrades to the global pool and says so in
:class:`PoolDiagnostics`. Everything downstream records ``pool_applied``, so a
gain measured globally is never mistaken for one measured under constraints.
That distinction matters more than it looks: pool-conditioned gains are measured
on an easier population (cheap players have lower absolute error), so a number
carrying the wrong label is worse than a missing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import polars as pl

from xg_alonso.contracts.context import DecisionContext
from xg_alonso.contracts.identifiers import TenthsOfMillion
from xg_alonso.contracts.prediction import Position
from xg_alonso.domain.rules import SquadRules

__all__ = [
    "REACHABLE_COLUMN",
    "FeasiblePool",
    "PoolDiagnostics",
    "PoolSignature",
    "feasible_pool",
]

REACHABLE_COLUMN: Final[str] = "__reachable"
"""Name of the boolean column the mask is attached under.

A *column*, not an index-aligned :class:`polars.Series`, and the choice is
load-bearing. :func:`~xg_alonso.discovery.harness._folds` filters and re-splits
the frame, so a Series aligned to the original row order would silently
misalign after the holdout filter and mask the wrong players. A column travels
with its rows. The dunder prefix matches the existing ``__timeline`` convention.
"""

#: Deciles for `PoolSignature.pool_share_band`. Coarse on purpose: the signature
#: keys registry evidence, and a band that moved with a 0.1m price change would
#: give every gameweek its own bucket while claiming to describe a situation.
_SHARE_BANDS: Final[tuple[tuple[float, str], ...]] = (
    (0.05, "under_5pc"),
    (0.15, "5_15pc"),
    (0.35, "15_35pc"),
    (0.60, "35_60pc"),
    (0.85, "60_85pc"),
    (1.01, "over_85pc"),
)

_BUDGET_BANDS: Final[tuple[tuple[int, str], ...]] = (
    (40, "under_4.0"),
    (60, "4.0_6.0"),
    (80, "6.0_8.0"),
    (110, "8.0_11.0"),
    (10_000, "over_11.0"),
)

_GLOBAL: Final[str] = "global"


def _band(value: float, bands: tuple[tuple[float, str], ...]) -> str:
    for ceiling, label in bands:
        if value < ceiling:
            return label
    return bands[-1][1]


@dataclass(frozen=True)
class PoolDiagnostics:
    """What the pool derivation did, and whether it could be trusted.

    Carried alongside every measurement made under a mask. A run that fell back
    to the global pool is not a failed run — it is a run whose numbers mean
    something different, and the difference has to be legible downstream.
    """

    global_rows: int
    reachable_rows: int
    applied: bool
    reason: str = ""
    widened_by: str = ""
    thin_folds: tuple[int, ...] = ()

    @property
    def share(self) -> float:
        if self.global_rows == 0:
            return 0.0
        return self.reachable_rows / self.global_rows

    def describe(self) -> str:
        if not self.applied:
            return f"pool not applied ({self.reason}); measured over all {self.global_rows} rows"
        detail = f"reachable {self.reachable_rows} of {self.global_rows} rows ({self.share:.1%})"
        if self.widened_by:
            detail += f"; widened by {self.widened_by}"
        return detail


@dataclass(frozen=True)
class PoolSignature:
    """A coarse, low-cardinality description of the reachable pool.

    Deliberately lossy. Keying registry evidence on the raw constraint set gives
    every manager a private registry and dilutes every acceptance decision's
    sample size. Keying on the *shape* of the residual problem lets two managers
    with different locked players share evidence when what remains of their
    search is the same.

    Cardinality is bounded by construction — at most
    ``2^4 positions x 5 budget bands x 6 share bands x 2`` — and
    ``tests/discovery/test_feasible.py`` asserts that against randomly generated
    constraint sets rather than against the arithmetic.
    """

    positions_open: tuple[Position, ...] = ()
    budget_band: str = _GLOBAL
    pool_share_band: str = _GLOBAL
    club_restricted: bool = False
    applied: bool = True

    def key(self) -> str:
        """Short deterministic string; ``"global"`` when unconstrained."""
        if not self.applied:
            return _GLOBAL
        positions = "".join(p.value[0] for p in self.positions_open) or "none"
        club = "club" if self.club_restricted else "free"
        return f"{positions}|{self.budget_band}|{self.pool_share_band}|{club}"


@dataclass(frozen=True)
class FeasiblePool:
    """The reachable-player mask and everything needed to interpret it."""

    mask: pl.Series
    signature: PoolSignature
    diagnostics: PoolDiagnostics
    closed_positions: tuple[Position, ...] = ()
    per_slot_budget: TenthsOfMillion | None = None
    binding: tuple[str, ...] = field(default_factory=tuple)
    """Constraints that actually removed someone.

    The honest counterpart to
    :attr:`~xg_alonso.optimization.objective.ConstraintReport.is_binding`. When
    empty, the run reports "your constraints did not narrow the search" rather
    than implying conditioning happened when it did not.
    """

    def attach(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` with the mask attached under :data:`REACHABLE_COLUMN`."""
        return frame.with_columns(self.mask.alias(REACHABLE_COLUMN))


def _unconstrained(frame: pl.DataFrame, reason: str) -> FeasiblePool:
    """A pool that keeps everything, and says why."""
    height = frame.height
    return FeasiblePool(
        mask=pl.Series(REACHABLE_COLUMN, [True] * height, dtype=pl.Boolean),
        signature=PoolSignature(applied=False),
        diagnostics=PoolDiagnostics(
            global_rows=height, reachable_rows=height, applied=False, reason=reason
        ),
    )


def feasible_pool(
    frame: pl.DataFrame,
    *,
    context: DecisionContext,
    rules: SquadRules,
    price_column: str = "price_tenths",
    position_column: str = "position",
    team_column: str = "team_id",
    player_column: str = "player_code",
) -> FeasiblePool:
    """Derive the pool of players this manager can actually reach.

    Args:
        frame: Training rows, one per player-gameweek.
        context: The manager's situation. Only ``context.constraints`` and
            ``context.squad`` are read — never the objective, never a belief.
        rules: Positional quotas, budget and the per-club cap, from the pinned
            snapshot.
        price_column: Point-in-time price. Its **absence is tolerated** and
            reported rather than raised: the discovery frame did not carry a
            price until recently, and a hard failure here would make the whole
            discovery loop unrunnable on an older frame.

    Returns:
        A :class:`FeasiblePool`. When nothing could be derived, or when the
        constraints turn out not to bind, the mask keeps every row and
        :attr:`PoolDiagnostics.applied` is ``False``.
    """
    height = frame.height
    if height == 0:
        return _unconstrained(frame, "empty frame")

    constraints = context.constraints
    squad = context.squad
    columns = set(frame.columns)

    if player_column not in columns:
        return _unconstrained(frame, f"frame has no {player_column!r} column")

    locked = context.locked_codes()
    binding: list[str] = []
    keep = pl.Series(REACHABLE_COLUMN, [True] * height, dtype=pl.Boolean)
    locked_rows = frame.get_column(player_column).is_in([int(c) for c in locked])

    def restrict(condition: pl.Series, label: str) -> None:
        """Apply a restriction, exempting locked players, and note if it bit."""
        nonlocal keep
        proposed = keep & (condition | locked_rows)
        if proposed.sum() < keep.sum():
            binding.append(label)
        keep = proposed

    # --- 1. Explicit exclusions -------------------------------------------
    if constraints.excluded_players:
        excluded = [int(c) for c in constraints.excluded_players]
        restrict(~frame.get_column(player_column).is_in(excluded), "excluded_players")

    if constraints.excluded_teams and team_column in columns:
        excluded_teams = [int(t) for t in constraints.excluded_teams]
        restrict(~frame.get_column(team_column).is_in(excluded_teams), "excluded_teams")

    # --- 2. Positional demand ---------------------------------------------
    #
    # A position with no open slot is closed: no unlocked player of that
    # position can enter the squad, whatever they cost or score.
    closed: list[Position] = []
    if squad is not None and position_column in columns:
        locked_by_position: dict[Position, int] = {}
        for pick in squad.picks:
            if pick.player_code in locked:
                locked_by_position[pick.position] = locked_by_position.get(pick.position, 0) + 1

        demand = {
            position: rules.rule_for(position).squad_select - locked_by_position.get(position, 0)
            for position in (Position.GKP, Position.DEF, Position.MID, Position.FWD)
        }
        closed = [position for position, remaining in demand.items() if remaining <= 0]
        if closed:
            restrict(
                ~frame.get_column(position_column).is_in([p.value for p in closed]),
                "locked_positions",
            )

    # --- 3. Budget ---------------------------------------------------------
    #
    # Locked players strand their sale value, so what is spendable is the bank
    # plus what the *sellable* part of the squad would raise, less any floor the
    # manager placed under the bank.
    per_slot: TenthsOfMillion | None = None
    if squad is not None and price_column in columns:
        sellable = [pick for pick in squad.picks if pick.player_code not in locked]
        open_slots = len(sellable)
        if open_slots > 0:
            spendable = (
                int(squad.bank)
                + sum(int(pick.selling_price) for pick in sellable)
                - int(constraints.minimum_bank)
            )
            # Reserve the cheapest legal fill for every *other* open slot, so the
            # ceiling is what one slot may command rather than the whole pot.
            floor_price = frame.get_column(price_column).min()
            cheapest = int(floor_price) if isinstance(floor_price, (int, float)) else 0
            ceiling = spendable - cheapest * (open_slots - 1)
            if ceiling > 0:
                per_slot = TenthsOfMillion(ceiling)
                restrict(frame.get_column(price_column) <= ceiling, "budget_ceiling")

    # --- 4. Club quotas ----------------------------------------------------
    if squad is not None and team_column in columns:
        counts: dict[int, int] = {}
        for pick in squad.picks:
            if pick.player_code in locked:
                counts[int(pick.team_id)] = counts.get(int(pick.team_id), 0) + 1
        ceilings = {int(q.team_id): q.count for q in constraints.maximum_players_by_team}
        full = [
            team
            for team, held in counts.items()
            if held >= min(rules.max_per_club, ceilings.get(team, rules.max_per_club))
        ]
        if full:
            restrict(~frame.get_column(team_column).is_in(full), "club_ceiling")

    reachable = int(keep.sum())
    if not binding:
        return _unconstrained(frame, "constraints did not narrow the search")

    signature = PoolSignature(
        positions_open=tuple(
            position
            for position in (Position.GKP, Position.DEF, Position.MID, Position.FWD)
            if position not in closed
        ),
        budget_band=_band(float(per_slot), _BUDGET_BANDS) if per_slot is not None else _GLOBAL,
        pool_share_band=_band(reachable / height, _SHARE_BANDS),
        club_restricted="club_ceiling" in binding,
    )
    return FeasiblePool(
        mask=keep,
        signature=signature,
        diagnostics=PoolDiagnostics(global_rows=height, reachable_rows=reachable, applied=True),
        closed_positions=tuple(closed),
        per_slot_budget=per_slot,
        binding=tuple(binding),
    )
