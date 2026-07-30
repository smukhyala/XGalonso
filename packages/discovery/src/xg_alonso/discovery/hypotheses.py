"""The seeded hypothesis library, and generation from measured weakness.

**These are candidates, not truths.** Every one states a mechanism and a
falsification condition, and several are expected to fail — a library where
everything is accepted is a library that was written after the results.

**What is honestly not computable here.** The brief's worked example is
``touch_dependency_rolling_5 x opponent_pressing_intensity``. Neither factor
exists: the official FPL API publishes no touch counts and no pressing metrics,
and decision D6 forbids any other source. Rather than ship a feature that reads
zeros and calls itself a pressing interaction, the affected hypotheses use the
nearest *computable* mechanism and say so in their rationale:

===============================  ==========================================
brief's mechanism                 computable substitute
===============================  ==========================================
opponent pressing intensity       opponent xG conceded, and points conceded
touch dependency                  creativity and influence per 90
shots / shots in the box          ``threat`` (Opta-derived, published)
key passes, passes into the box   ``creativity`` (Opta-derived, published)
set-piece share                   *none* — omitted rather than faked
===============================  ==========================================

That table is the honest version of "we implemented the hypothesis library".

**Generation is automated but not intelligent.** :func:`generate_from_residuals`
reads where the current model is measurably worst and instantiates mechanisms
that plausibly address that segment. It searches a declared space; it does not
reason about football, and nothing here calls an LLM — the repository has no
client and no key. :data:`GenerationSource.LLM` exists only so that a proposal,
if one is ever supplied through the adapter, stays permanently distinguishable
from a measured one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from xg_alonso.contracts.discovery import (
    FeatureHypothesis,
    GenerationSource,
    LeakageRisk,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.discovery.dsl import (
    Arith,
    ArithOp,
    EwmMean,
    FeatureProgram,
    GroupKey,
    GroupRel,
    GroupRelOp,
    Rolling,
    RollingAgg,
    ShrunkRate,
    Source,
    SourceScope,
    TimeSince,
    Trend,
    Unary,
    UnaryOp,
)

__all__ = [
    "SEEDED_HYPOTHESES",
    "SeededHypothesis",
    "generate_from_residuals",
    "seeded_programs",
]


@dataclass(frozen=True)
class SeededHypothesis:
    """A hypothesis paired with the program that tests it."""

    hypothesis: FeatureHypothesis
    program: FeatureProgram

    @property
    def name(self) -> str:
        return self.program.name


def _xg_rate(window: int = 5) -> ShrunkRate:
    return ShrunkRate(numerator="expected_goals", denominator="minutes", window=window)


def _rate(numerator: str, window: int = 5) -> ShrunkRate:
    return ShrunkRate(numerator=numerator, denominator="minutes", window=window)


def _h(
    hid: str,
    title: str,
    rationale: str,
    falsification: str,
    program: FeatureProgram,
    *,
    positions: tuple[Position, ...] = (),
    expected: str = "",
    risk: LeakageRisk = LeakageRisk.NONE,
    objectives: tuple[str, ...] = (),
) -> SeededHypothesis:
    return SeededHypothesis(
        hypothesis=FeatureHypothesis(
            id=hid,
            title=title,
            football_rationale=rationale,
            falsification_condition=falsification,
            expected_relationship=expected,
            required_raw_fields=program.columns(),
            transformation_plan=program.describe(),
            applicable_positions=positions,
            leakage_risk=risk,
            target_objectives=objectives,
            generation_source=GenerationSource.SEEDED,
        ),
        program=program,
    )


#: Fifteen mechanisms, mapped from the brief and restricted to what is computable.
SEEDED_HYPOTHESES: Final[tuple[SeededHypothesis, ...]] = (
    _h(
        "minutes.xg_x_expected_minutes",
        "Recent xG is more informative once scaled by expected minutes",
        "A per-90 rate says what a player does when on the pitch and says nothing "
        "about whether he will be on it. Multiplying the rate by expected minutes "
        "turns a conditional rate into an unconditional expectation, which is what "
        "a points projection actually needs.",
        "No improvement over the required set in at least 3 of 5 walk-forward folds.",
        FeatureProgram(
            name="xg_x_expected_minutes",
            root=Arith(
                op=ArithOp.MUL,
                left=_xg_rate(5),
                right=Rolling(child=Source(column="minutes"), window=5),
            ),
        ),
        expected="positive, strongest for forwards and attacking midfielders",
    ),
    _h(
        "fixture.difficulty_x_role",
        "Fixture difficulty interacts with role rather than shifting everyone equally",
        "A weak defence concedes more chances to the players who generate chances. "
        "A soft fixture should therefore lift a high-xG forward far more than a "
        "defensive midfielder, which a flat fixture adjustment cannot express.",
        "The interaction adds nothing beyond opponent strength entered on its own.",
        FeatureProgram(
            name="xg_x_opponent_weakness",
            root=Arith(
                op=ArithOp.MUL,
                left=_xg_rate(5),
                right=Source(column="opponent_conceded_xg_mean_5", scope=SourceScope.ENTITY),
            ),
        ),
        expected="positive; concentrated in MID and FWD",
    ),
    _h(
        "congestion.rest_x_minutes_load",
        "Rest matters most for players carrying a heavy minutes load",
        "A player who starts every week accumulates fatigue that a rotation option "
        "never does, so a short turnaround should cost the high-minute player more. "
        "Rest alone was already available to the model and did not help; the claim "
        "is that it only means something conditioned on load.",
        "No improvement, or improvement no better than rest days entered alone.",
        FeatureProgram(
            name="rest_x_minutes_load",
            root=Arith(
                op=ArithOp.MUL,
                left=TimeSince(event_column="minutes"),
                right=Rolling(child=Source(column="minutes"), window=5),
            ),
        ),
        expected="negative for short rest at high load",
    ),
    _h(
        "opponent.solidity_x_creators",
        "Creators lose output against defensively solid opponents",
        "**Substitute mechanism.** The brief proposes opponent pressing intensity "
        "against touch-dependent creators. Neither is published by the FPL API. "
        "The computable analogue is opponent defensive solidity (xG conceded) "
        "against a player's creativity rate — a different mechanism with a similar "
        "shape, and it is labelled as a substitute rather than as the original.",
        "No incremental value beyond creativity and opponent strength separately.",
        FeatureProgram(
            name="creativity_x_opponent_solidity",
            root=Arith(
                op=ArithOp.MUL,
                left=_rate("creativity", 5),
                right=Source(column="opponent_conceded_xg_mean_5", scope=SourceScope.ENTITY),
            ),
        ),
        expected="positive; creators gain more against leaky sides",
    ),
    _h(
        "team.share_of_threat",
        "A player's share of his side's attacking output predicts opportunity",
        "Teammate absence and tactical change both show up as a shift in who takes "
        "the shots. A player's share of team threat captures the focal role that an "
        "absolute rate misses — two forwards with the same xG are different assets "
        "if one is his team's only route to goal.",
        "Share adds nothing beyond the player's own absolute rate.",
        FeatureProgram(
            name="threat_share_of_team",
            root=GroupRel(
                op=GroupRelOp.SHARE,
                by=GroupKey.TEAM,
                child=Rolling(child=Source(column="threat"), window=5),
            ),
        ),
        expected="positive; strongest for FWD",
    ),
    _h(
        "market.ownership_momentum",
        "Transfer flow carries price information recent points do not",
        "Price changes are driven by net transfers, not by points. A player being "
        "bought heavily after a quiet week is on a different price trajectory from "
        "one being sold after the same week, and no points-based feature separates "
        "them.",
        "No incremental value for the team-value objective specifically.",
        FeatureProgram(
            name="transfer_momentum_5",
            root=Arith(
                op=ArithOp.SAFE_DIV,
                left=Rolling(child=Source(column="transfers_balance"), window=5),
                right=Rolling(child=Source(column="selected"), window=5),
            ),
        ),
        expected="positive for team_value_growth, near zero for expected_points",
        objectives=("team_value_growth",),
    ),
    _h(
        "venue.home_advantage_by_output",
        "Home advantage is larger for high-output attackers",
        "Playing at home lifts a side's chance creation, and the lift accrues to "
        "whoever converts chances. A flat home bonus spreads it evenly across the "
        "squad, including players whose returns do not depend on chance volume.",
        "The interaction adds nothing beyond a flat home indicator.",
        FeatureProgram(
            name="home_x_attacking_output",
            root=Arith(
                op=ArithOp.MUL,
                left=Source(column="is_home", scope=SourceScope.ENTITY),
                right=_rate("expected_goal_involvements", 5),
            ),
        ),
        expected="positive; concentrated in MID and FWD",
    ),
    _h(
        "captaincy.ceiling_x_security",
        "Captaincy value needs both a ceiling and the security to reach it",
        "A haul requires the player to be on the pitch long enough to produce one. "
        "Ceiling alone over-rates an explosive substitute; minutes alone over-rates "
        "a nailed defender. The product is what an armband decision actually weighs.",
        "No improvement under the captaincy-weighted objective.",
        FeatureProgram(
            name="ceiling_x_start_security",
            root=Arith(
                op=ArithOp.MUL,
                left=Rolling(child=Source(column="total_points"), window=5, agg=RollingAgg.MAX),
                right=Rolling(child=Source(column="starts"), window=5),
            ),
        ),
        expected="positive for captaincy_upside and rank-gain objectives",
        objectives=("mini_league_chase", "locked_premium_aggressive"),
    ),
    _h(
        "form.deviation_from_position_trend",
        "Deviation from the positional norm identifies role change earlier than a season average",
        "A season average is slow: it takes weeks to move after a player's role "
        "changes. Standing relative to his position right now moves immediately, so "
        "it should identify a newly-promoted starter before the average catches up.",
        "No improvement over the raw rolling mean it is derived from.",
        FeatureProgram(
            name="points_vs_position_norm",
            root=GroupRel(
                op=GroupRelOp.ZSCORE,
                by=GroupKey.POSITION,
                child=Rolling(child=Source(column="total_points"), window=5),
            ),
        ),
        expected="positive; a leading indicator rather than a level",
    ),
    _h(
        "opponent.shot_volume_vs_quality",
        "Opponent weakness splits into volume conceded and quality conceded",
        "Two defences that concede the same xG are not equivalent: one gives up many "
        "low-quality chances, the other few high-quality ones. A poacher benefits "
        "from the second and a volume shooter from the first, so collapsing them "
        "into one number loses the distinction that decides which forward to buy.",
        "The ratio adds nothing beyond total xG conceded.",
        FeatureProgram(
            name="opponent_chance_quality",
            root=Arith(
                op=ArithOp.SAFE_DIV,
                left=Source(column="opponent_conceded_xg_mean_5", scope=SourceScope.ENTITY),
                right=Source(column="opponent_conceded_goals_mean_5", scope=SourceScope.ENTITY),
            ),
        ),
        expected="positive for high-xG forwards",
    ),
    _h(
        "volatility.explosiveness",
        "Return volatility is an asset for a chaser and a liability for a defender of rank",
        "The same variance has opposite sign depending on the objective. This "
        "feature is expected to be *rejected globally and accepted for the "
        "rank-gain objective* — which, if it happens, is the clearest single piece "
        "of evidence that objective conditioning does something real.",
        "Accepted globally, or rejected under every objective alike.",
        FeatureProgram(
            name="points_volatility_10",
            root=Rolling(
                child=Source(column="total_points"),
                window=10,
                agg=RollingAgg.STD,
                min_periods=3,
            ),
        ),
        expected="objective-specific: positive for chase, negative for protection",
        objectives=("mini_league_chase", "rank_protection"),
    ),
    _h(
        "concentration.focal_attacker",
        "Attacking concentration raises the value of a side's focal attacker",
        "When a team's chances funnel through one player, that player's returns are "
        "less exposed to squad rotation elsewhere. Concentration is measurable as "
        "how far a player's threat sits above his own team's mean.",
        "No incremental value beyond the player's own threat rate.",
        FeatureProgram(
            name="threat_above_team_mean",
            root=GroupRel(
                op=GroupRelOp.DEV_FROM_MEAN,
                by=GroupKey.TEAM,
                child=Rolling(child=Source(column="threat"), window=5),
            ),
        ),
        expected="positive for FWD and attacking MID",
    ),
    _h(
        "overlap.attacking_dilution",
        "Attacking overlap among teammates reduces marginal returns",
        "The mirror of concentration. A forward in a side with three other high-xG "
        "attackers competes for the same chances, so his rate overstates what he "
        "will actually convert.",
        "No improvement, or improvement indistinguishable from the concentration feature.",
        FeatureProgram(
            name="xg_rank_within_team",
            root=GroupRel(
                op=GroupRelOp.RANK,
                by=GroupKey.TEAM,
                child=_xg_rate(5),
            ),
        ),
        expected="positive; the top attacker in a side outperforms his raw rate",
    ),
    _h(
        "substitution.late_minutes_pattern",
        "Substitution patterns interact with fixture congestion",
        "A player who is routinely withdrawn on 60 minutes loses the appearance "
        "bonus band and much of his chance volume, and congestion makes early "
        "withdrawal more likely. Minutes-when-played against fixture density "
        "separates a rested starter from a fading one.",
        "No improvement over expected minutes entered alone.",
        FeatureProgram(
            name="minutes_trend_x_congestion",
            root=Arith(
                op=ArithOp.MUL,
                left=Trend(child=Source(column="minutes"), window=5),
                right=Unary(
                    op=UnaryOp.CLIP,
                    child=TimeSince(event_column="minutes"),
                    lower=0.0,
                    upper=14.0,
                ),
            ),
        ),
        expected="negative when minutes are trending down under congestion",
    ),
    _h(
        "form.recency_weighted_output",
        "Exponentially weighted form beats a flat window",
        "A flat rolling mean treats a match five weeks ago as equal to last week's. "
        "Form does not work like that, and an exponential weighting expresses the "
        "decay a flat window cannot.",
        "No improvement over the flat rolling mean over the same window.",
        FeatureProgram(
            name="points_ewm_10",
            root=EwmMean(child=Source(column="total_points"), window=10, halflife=3.0),
        ),
        expected="positive but small; heavily correlated with the flat mean",
    ),
)


def seeded_programs() -> dict[str, FeatureProgram]:
    """Every seeded program, by name."""
    return {seed.program.name: seed.program for seed in SEEDED_HYPOTHESES}


# --- evidence-driven generation ----------------------------------------------

#: Mechanisms the generator can instantiate against a weak segment, as
#: ``(suffix, builder)``. Small and declared: an unbounded generator produces
#: syntactic variations of one idea rather than diverse ideas, which is the
#: failure the brief explicitly warns about.
_MECHANISMS: Final[tuple[tuple[str, str], ...]] = (
    ("x_minutes", "scaled by playing time"),
    ("x_opponent", "interacted with opponent weakness"),
    ("x_home", "interacted with venue"),
    ("vs_position", "expressed relative to the position"),
    ("trend", "as a trend rather than a level"),
    ("ceiling", "as a ceiling rather than a mean"),
)


def _mechanism_program(name: str, base: str, mechanism: str) -> FeatureProgram | None:
    """Instantiate one mechanism over one base metric."""
    rate = _rate(base, 5)
    rolling = Rolling(child=Source(column=base), window=5)

    if mechanism == "x_minutes":
        root: object = Arith(
            op=ArithOp.MUL, left=rate, right=Rolling(child=Source(column="minutes"), window=5)
        )
    elif mechanism == "x_opponent":
        root = Arith(
            op=ArithOp.MUL,
            left=rate,
            right=Source(column="opponent_conceded_xg_mean_5", scope=SourceScope.ENTITY),
        )
    elif mechanism == "x_home":
        root = Arith(
            op=ArithOp.MUL,
            left=rate,
            right=Source(column="is_home", scope=SourceScope.ENTITY),
        )
    elif mechanism == "vs_position":
        root = GroupRel(op=GroupRelOp.ZSCORE, by=GroupKey.POSITION, child=rolling)
    elif mechanism == "trend":
        root = Trend(child=Source(column=base), window=5)
    elif mechanism == "ceiling":
        root = Rolling(child=Source(column=base), window=5, agg=RollingAgg.MAX)
    else:  # pragma: no cover - the table above is closed
        return None

    try:
        return FeatureProgram(name=name, root=root)  # type: ignore[arg-type]
    except Exception:
        return None


def generate_from_residuals(
    *,
    weak_segments: Sequence[tuple[str, str, float]],
    base_metrics: Sequence[str],
    existing_names: Sequence[str] = (),
    known_versions: Sequence[str] = (),
    rejected_families: Sequence[str] = (),
    emphasis: Sequence[str] = (),
    limit: int = 8,
) -> list[SeededHypothesis]:
    """Propose hypotheses aimed at where the model is measurably worst.

    Args:
        weak_segments: ``(segment_kind, segment, relative_error)``, worst first.
            This is the *evidence*: generation is aimed at a measured failure
            rather than at generic football prose.
        base_metrics: Raw columns the mechanisms may be built over.
        existing_names / known_versions: Suppress duplicates by name and by
            semantics. Semantic suppression is the one that matters — a renamed
            copy of a rejected feature is still that feature.
        rejected_families: Hypothesis families already refuted. Skipped unless a
            mechanism differs materially, so the loop does not re-propose what it
            has already disproved.
        emphasis: Advisory hints from the objective, e.g. ``upside``.
        limit: Proposals per iteration. Bounded on purpose.

    Returns:
        Diverse proposals, at most one mechanism per (segment, metric) pair so a
        batch does not fill up with variations of a single idea.
    """
    taken = set(existing_names)
    versions = set(known_versions)
    refuted = set(rejected_families)
    out: list[SeededHypothesis] = []

    prefer_ceiling = any(tag in {"upside", "differential"} for tag in emphasis)
    mechanisms = list(_MECHANISMS)
    if prefer_ceiling:
        mechanisms.sort(key=lambda pair: 0 if pair[0] in {"ceiling", "trend"} else 1)

    used_pairs: set[tuple[str, str]] = set()

    for segment_kind, segment, error in weak_segments:
        for metric in base_metrics:
            if len(out) >= limit:
                return out
            for suffix, description in mechanisms:
                family = f"residual_{suffix}"
                if family in refuted:
                    continue
                if (segment, metric) in used_pairs:
                    continue
                name = f"{metric}_{suffix}"
                if name in taken:
                    continue

                program = _mechanism_program(name, metric, suffix)
                if program is None or program.version() in versions:
                    continue

                taken.add(name)
                versions.add(program.version())
                used_pairs.add((segment, metric))
                out.append(
                    SeededHypothesis(
                        hypothesis=FeatureHypothesis(
                            id=f"{family}.{metric}.{segment}",
                            title=f"{metric} {description}, for {segment_kind} {segment}",
                            football_rationale=(
                                f"The model's error is highest for {segment_kind} {segment} "
                                f"(relative error {error:.3f}). If {metric} carries signal that "
                                f"the current set uses only in aggregate, expressing it "
                                f"{description} may recover it for that segment specifically."
                            ),
                            falsification_condition=(
                                "No incremental value over the required set in a majority of "
                                f"folds, and no gain within {segment_kind} {segment} on at "
                                "least 200 validation rows."
                            ),
                            expected_relationship=f"positive within {segment}",
                            required_raw_fields=program.columns(),
                            transformation_plan=program.describe(),
                            generation_source=GenerationSource.RESIDUAL,
                            leakage_risk=LeakageRisk.NONE,
                        ),
                        program=program,
                    )
                )
                break
    return out
