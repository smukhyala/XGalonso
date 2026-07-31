"""Score a squad the way the game does, and decompose the result.

``score_squad`` answered one question: how many points did this squad take?
That is enough to rank two policies and not enough to understand either. A
policy that wins on captaincy and one that wins on selection produce the same
headline, and a harness that conflates them cannot be argued with.

So this module returns the parts. ``starters`` is the eleven picked at the
deadline with the armband counted once; ``autosub`` is what the game added
afterwards; ``captaincy`` is the *extra* from doubling, not the armband
player's whole score. They sum to the total, and
:class:`~xg_alonso.contracts.simulation.GameweekSimulation` validates that they
do — so the decomposition cannot quietly stop adding up.

**The order of operations is the correctness argument.** Select the eleven from
*predictions*, then score it against *outcomes*. Picking the eleven that turned
out best would measure hindsight and would flatter every policy equally, which
is worse than flattering one — it would look like a fair comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.prediction import PlayerPrediction
from xg_alonso.contracts.simulation import (
    AutosubResult,
    Captaincy,
    CaptainSource,
    SkippedSubstitution,
    Substitution,
)
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.autosubs import apply_autosubs, resolve_captaincy
from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringThresholds
from xg_alonso.optimization.lineup import best_starting_xi

__all__ = ["SquadScore", "simulate_squad"]


@dataclass(frozen=True)
class SquadScore:
    """One squad's gameweek, decomposed into parts that sum to the total."""

    starters_points: int
    autosub_points: int
    captaincy_points: int
    bench_points: int
    final_xi: tuple[PlayerCode, ...]
    substitutions: tuple[Substitution, ...]
    skipped: tuple[SkippedSubstitution, ...]
    captaincy: Captaincy

    @property
    def total(self) -> int:
        return self.starters_points + self.autosub_points + self.captaincy_points


def simulate_squad(
    squad: SquadState,
    points: Mapping[PlayerCode, int],
    *,
    predictions: Mapping[PlayerCode, PlayerPrediction] | None = None,
    rules: SquadRules | None = None,
    minutes: Mapping[PlayerCode, int] | None = None,
    captain_multiplier: int | None = None,
) -> SquadScore:
    """Score a squad against actual results, with the game's own mechanics.

    Args:
        squad: The fifteen, as they stood at the deadline.
        points: What each player actually scored.
        predictions: Expected points, used to *select* the eleven. Without them
            the squad's stored slots are used instead.
        rules: Positional bounds and sizes from the pinned snapshot. Required
            for selection and for autosubs.
        minutes: Minutes actually played. **Supplying this enables autosubs and
            vice-captaincy.** Omitting it reproduces the outcome-blind
            behaviour exactly: no substitutions, and the captain doubled
            whether or not he played.
        captain_multiplier: Defaults to the pinned ``ScoringThresholds`` value.

    Returns:
        The decomposed result. ``total`` is what ``score_squad`` returns.
    """
    multiplier = (
        ScoringThresholds().captain_multiplier if captain_multiplier is None else captain_multiplier
    )

    if predictions is not None and rules is not None:
        selection = best_starting_xi(squad.picks, predictions, rules)
        starters: Sequence[SquadPick] = selection.starters
        bench: Sequence[SquadPick] = selection.bench
        captain = selection.captain
        vice_captain = selection.vice_captain
    else:
        starters = squad.starters
        bench = squad.bench
        captain = next((p.player_code for p in squad.picks if p.is_captain), None)
        vice_captain = next((p.player_code for p in squad.picks if p.is_vice_captain), None)

    if minutes is None or rules is None:
        # The outcome-blind path. Preserved exactly so `score_squad` keeps its
        # meaning for every caller that has not yet been given minutes.
        autosubs = AutosubResult(final_xi=tuple(p.player_code for p in starters))
        armband = (
            Captaincy(holder=captain, source=CaptainSource.CAPTAIN, multiplier=multiplier)
            if captain is not None
            else Captaincy(holder=None, source=CaptainSource.NONE, multiplier=1)
        )
    else:
        autosubs = apply_autosubs(starters=starters, bench=bench, minutes=minutes, rules=rules)
        armband = resolve_captaincy(
            captain=captain,
            vice_captain=vice_captain,
            final_xi=autosubs.final_xi,
            minutes=minutes,
            rules=rules,
            multiplier=multiplier,
        )

    # As-picked eleven, armband counted once. A withdrawn starter contributes
    # zero here by construction: no minutes means no points.
    starters_points = sum(points.get(p.player_code, 0) for p in starters)

    autosub_points = sum(points.get(s.player_on, 0) for s in autosubs.substitutions)

    captaincy_points = 0
    if armband.holder is not None:
        captaincy_points = points.get(armband.holder, 0) * (armband.multiplier - 1)

    came_on = {s.player_on for s in autosubs.substitutions}
    bench_points = sum(points.get(p.player_code, 0) for p in bench if p.player_code not in came_on)

    return SquadScore(
        starters_points=starters_points,
        autosub_points=autosub_points,
        captaincy_points=captaincy_points,
        bench_points=bench_points,
        final_xi=autosubs.final_xi,
        substitutions=autosubs.substitutions,
        skipped=autosubs.skipped,
        captaincy=armband,
    )
