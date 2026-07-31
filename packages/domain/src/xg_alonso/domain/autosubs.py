"""Automatic substitutions and captaincy — what the game does after the deadline.

Three places in this repository already said this was missing and scored around
it: ``optimization/lineup.py`` recorded a vice-captain it never used,
``evaluation/backtest.py`` noted that a benched player never replaces a starter
who fails to play, and ``optimization/squad_builder.py`` justified a bench
tie-break by the absence of exactly this module.

**The reserve keeper needs no special case, and that is the whole design.** The
obvious implementation branches on position: outfield substitutes replace
outfield starters, the reserve goalkeeper replaces the goalkeeper. That branch
is unnecessary, because ``check_starting_xi`` already reads ``squad_min_play``
and ``squad_max_play`` from the pinned snapshot, where a goalkeeper is 1 and 1.
Swapping the reserve keeper for an outfielder produces two goalkeepers and
fails; swapping an outfielder for the keeper produces none and fails. So the
algorithm asks one question — *would this eleven be legal?* — and the keeper
rule falls out of the answer. One test asserts the skip reason is identical in
the keeper case and the defender case, which is what makes the absence of a
branch checkable rather than merely claimed.

**The loop is bench-first.** For each substitute in bench order, find a vacancy
they can legally fill. The alternative — for each vacancy, find a substitute —
agrees in every case with two non-playing starters and diverges only with three
or more in a tight shape. Bench order is the manager's stated priority, so it
is the order the game honours and the one used here. This is an assertion about
an engine FPL does not publish, so it is documented like the other
``VERIFY``-class facts rather than buried in a comment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.simulation import (
    AutosubResult,
    Captaincy,
    CaptainSource,
    SkippedSubstitution,
    SkipReason,
    Substitution,
)
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.domain.constraints import check_starting_xi
from xg_alonso.domain.rules import SquadRules

__all__ = ["apply_autosubs", "played", "resolve_captaincy"]


def played(minutes: Mapping[PlayerCode, int], code: PlayerCode) -> bool:
    """Whether a player featured at all.

    An absent key means no row for that player in that gameweek, which means no
    minutes. Treating a missing row as "unknown" and leaving the starter in
    place would silently disable autosubs for every blanking club.
    """
    return minutes.get(code, 0) > 0


def apply_autosubs(
    *,
    starters: Sequence[SquadPick],
    bench: Sequence[SquadPick],
    minutes: Mapping[PlayerCode, int],
    rules: SquadRules,
) -> AutosubResult:
    """Bring on substitutes for starters who did not play.

    Args:
        starters: The eleven picked at the deadline, in any order.
        bench: The substitutes in priority order — first is tried first.
        minutes: Minutes played per player this gameweek. Missing means none.
        rules: Supplies the positional bounds every candidate eleven is tested
            against, read from the pinned snapshot.

    Returns:
        The eleven that actually scored, every substitution made, and every
        bench player passed over with the reason why.

    A vacancy nobody can legally fill simply stays in the eleven on zero, which
    is what FPL does — a squad is never short a player, it is short a
    contribution.
    """
    xi = list(starters)
    vacancies = [pick for pick in xi if not played(minutes, pick.player_code)]
    vacancies.sort(key=lambda pick: pick.squad_slot)

    substitutions: list[Substitution] = []
    skipped: list[SkippedSubstitution] = []

    for substitute in bench:
        if not vacancies:
            skipped.append(
                SkippedSubstitution(
                    player_on=substitute.player_code,
                    bench_slot=substitute.squad_slot,
                    reason=SkipReason.NO_VACANCY,
                )
            )
            continue

        if not played(minutes, substitute.player_code):
            skipped.append(
                SkippedSubstitution(
                    player_on=substitute.player_code,
                    bench_slot=substitute.squad_slot,
                    reason=SkipReason.BENCH_PLAYER_DID_NOT_PLAY,
                )
            )
            continue

        target = _first_legal_vacancy(xi, vacancies, substitute, rules)
        if target is None:
            skipped.append(
                SkippedSubstitution(
                    player_on=substitute.player_code,
                    bench_slot=substitute.squad_slot,
                    reason=SkipReason.FORMATION_WOULD_BE_ILLEGAL,
                )
            )
            continue

        xi = [pick for pick in xi if pick.player_code != target.player_code] + [substitute]
        vacancies.remove(target)
        substitutions.append(
            Substitution(
                player_off=target.player_code,
                player_on=substitute.player_code,
                bench_slot=substitute.squad_slot,
                off_position=target.position,
                on_position=substitute.position,
            )
        )

    return AutosubResult(
        final_xi=tuple(pick.player_code for pick in xi),
        substitutions=tuple(substitutions),
        skipped=tuple(skipped),
    )


def _first_legal_vacancy(
    xi: Sequence[SquadPick],
    vacancies: Sequence[SquadPick],
    substitute: SquadPick,
    rules: SquadRules,
) -> SquadPick | None:
    """The lowest-slot vacancy this substitute can fill without breaking the shape.

    Legality is the only test. There is deliberately no position comparison
    here: a midfielder may replace a defender when the resulting shape is legal,
    and the reserve keeper is confined to the keeper's slot by the same rule
    that confines everyone else.
    """
    for vacancy in vacancies:
        candidate = [pick for pick in xi if pick.player_code != vacancy.player_code]
        candidate.append(substitute)
        if not check_starting_xi(candidate, rules=rules):
            return vacancy
    return None


def resolve_captaincy(
    *,
    captain: PlayerCode | None,
    vice_captain: PlayerCode | None,
    final_xi: Sequence[PlayerCode],
    minutes: Mapping[PlayerCode, int],
    rules: SquadRules,
    multiplier: int,
) -> Captaincy:
    """Decide who is doubled, after substitutions rather than before.

    The order matters twice. A captain who was substituted off is not in
    ``final_xi``, so the armband passes to the vice rather than being paid to
    somebody who did not play. And a vice who arrived *as* a substitute and
    played is eligible, which the membership test gives for free.

    Args:
        rules: Consulted for ``vice_captain_enabled``, which FPL publishes as
            ``sys_vice_captain_enabled`` and the code previously assumed.
        multiplier: From ``ScoringThresholds.captain_multiplier``.

    Returns:
        The armband holder and its provenance. When neither played the
        multiplier is 1 and nobody holds it — FPL doubles nothing rather than
        promoting the next best player.
    """
    eligible = set(final_xi)

    def in_play(code: PlayerCode | None) -> bool:
        return code is not None and code in eligible and played(minutes, code)

    if in_play(captain):
        assert captain is not None  # narrowed by in_play
        return Captaincy(holder=captain, source=CaptainSource.CAPTAIN, multiplier=multiplier)

    if rules.vice_captain_enabled and in_play(vice_captain):
        assert vice_captain is not None
        return Captaincy(
            holder=vice_captain, source=CaptainSource.VICE_CAPTAIN, multiplier=multiplier
        )

    return Captaincy(holder=None, source=CaptainSource.NONE, multiplier=1)


def formation_of(picks: Sequence[SquadPick]) -> tuple[int, int, int, int]:
    """Players per position in ``(GKP, DEF, MID, FWD)`` order.

    Useful for asserting that an eleven is still the shape it claims to be
    after substitutions, which is a different question from whether it is legal.
    """
    counts = dict.fromkeys((Position.GKP, Position.DEF, Position.MID, Position.FWD), 0)
    for pick in picks:
        counts[pick.position] += 1
    return (
        counts[Position.GKP],
        counts[Position.DEF],
        counts[Position.MID],
        counts[Position.FWD],
    )
