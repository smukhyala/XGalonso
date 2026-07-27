"""Render grounded recommendations as human-readable text.

**The no-fabrication guarantee is structural, not procedural.** A
:class:`~xg_alonso.contracts.reason_codes.Reason` refuses to be constructed
unless its evidence satisfies every placeholder in its template, so by the time
prose is rendered there is no gap left to fill. This module only arranges
sentences that were already grounded upstream.

If an LLM is later used to smooth the wording, it receives these rendered
sentences and may rephrase them. It is never handed the raw model output and
asked to explain it, because that is precisely the arrangement in which a
plausible invented cause becomes indistinguishable from a real one.
"""

from __future__ import annotations

from collections.abc import Mapping

from xg_alonso.contracts.identifiers import PlayerCode, TenthsOfMillion, format_money
from xg_alonso.contracts.recommendation import TransferRecommendation

__all__ = ["render_recommendation", "render_squad_summary"]

_RULE = "─" * 68


def _name(codes: Mapping[PlayerCode, str], code: PlayerCode) -> str:
    return codes.get(code, f"player {code}")


def render_recommendation(
    recommendation: TransferRecommendation,
    *,
    names: Mapping[PlayerCode, str],
    prices: Mapping[PlayerCode, int] | None = None,
) -> str:
    """Render a recommendation for a terminal.

    The output leads with the decision, then the arithmetic, then the evidence —
    in that order, because a user needs to know what to do before they need to
    know why, and needs to know why before they will trust it.
    """
    prices = prices or {}
    lines: list[str] = [_RULE]

    if recommendation.package.is_hold:
        lines += [
            f"  HOLD — no transfer beats keeping your squad (GW{recommendation.gameweek})",
            _RULE,
            "",
            f"  Projected points, current squad : {recommendation.comparison.baseline_expected_points:6.2f}",
            "",
            "  Every legal transfer was evaluated. None gained enough to justify",
            "  spending the transfer, so the recommendation is to roll it.",
            "",
            _RULE,
        ]
        return "\n".join(lines)

    move = recommendation.package.moves[0]
    out_name = _name(names, move.player_out)
    in_name = _name(names, move.player_in)

    lines += [
        f"  TRANSFER — GW{recommendation.gameweek}",
        _RULE,
        "",
        f"  OUT  {out_name:<28} {format_money(move.selling_price):>9}",
        f"  IN   {in_name:<28} {format_money(move.purchase_price):>9}",
        "",
    ]

    net = move.net_cost
    direction = "costs" if net > 0 else "frees"
    lines += [
        f"  Net {direction:<5}                     {format_money(TenthsOfMillion(abs(net))):>9}",
        f"  Bank after                        {format_money(recommendation.package.bank_after):>9}",
    ]
    if recommendation.package.hit_cost:
        lines.append(
            f"  Hit                                 -{recommendation.package.hit_cost} pts"
        )
    lines.append("")

    comparison = recommendation.comparison
    lines += [
        f"  Projected points, hold            {comparison.baseline_expected_points:9.2f}",
        f"  Projected points, after transfer  {comparison.candidate_expected_points:9.2f}",
        f"  Expected gain                     {recommendation.expected_points_gain:+9.2f} pts",
        f"  Uncertainty                       {recommendation.risk_score:9.2f} pts",
        "",
        "  Why:",
    ]

    for sentence in recommendation.render_reasons():
        lines.append(f"    • {sentence}")

    lines += [
        "",
        f"  Baseline: {comparison.baseline_name} over {comparison.horizon_gameweeks} GW"
        f"  ·  run {recommendation.run_id}",
        _RULE,
    ]
    return "\n".join(lines)


def render_squad_summary(
    *,
    entry_id: int,
    gameweek: int,
    squad_value: int,
    bank: int,
    free_transfers: int,
    starters: list[tuple[str, str, float]],
    bench: list[tuple[str, str, float]],
) -> str:
    """Render a squad with projected points per player.

    Args:
        starters: ``(name, position, expected_points)``, in XI order.
        bench: ``(name, position, expected_points)``, in bench order.
    """
    lines = [
        _RULE,
        f"  SQUAD — entry {entry_id}, GW{gameweek}",
        _RULE,
        "",
        f"  Squad value {format_money(TenthsOfMillion(squad_value)):>8}"
        f"   Bank {format_money(TenthsOfMillion(bank)):>7}"
        f"   Free transfers {free_transfers}",
        "",
        "  STARTING XI",
    ]
    for name, position, points in starters:
        lines.append(f"    {position:<4} {name:<28} {points:6.2f}")

    lines += ["", "  BENCH"]
    for index, (name, position, points) in enumerate(bench, start=1):
        lines.append(f"    {index}. {position:<3} {name:<27} {points:6.2f}")

    total = sum(points for _, _, points in starters)
    lines += ["", f"  Projected XI total {total:.2f}", _RULE]
    return "\n".join(lines)
