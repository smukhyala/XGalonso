"""Can this check fail? Three negative controls say yes, in rising strength.

A calibration harness that never fires is worse than none, because it
manufactures confidence. So before any claim that a forecast is well calibrated,
this file proves the measurement can report the opposite:

1. **Synthetic mis-dispersion.** A distribution deliberately widened or narrowed
   by a known factor must move the dispersion ratio the matching way. A check
   that passes both is not a check.
2. **The incumbent.** A Gaussian built from the shipped ``expected_points_sd``
   must be flagged as badly under-covered. This is the strongest control
   available, because it is the code that ships today — if the check does not
   flag it, the check is broken.
3. **A point mass.** Zero width, so every metric must be marked uninterpretable
   rather than excellent.

The positive direction is covered too: a forecast that genuinely generates the
outcomes must come out calibrated, or the metrics are measuring noise.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from xg_alonso.evaluation.calibration import (
    Distribution,
    _cramer_von_mises,
    frame_from_distributions,
    gaussian_pmf,
    inflate_pmf,
    point_mass_pmf,
    randomised_pit,
    score_distributions,
)

# The measured population, from 113,270 player-gameweeks across four seasons.
_STATE_SHARES = (0.596, 0.128, 0.277)


def _truthful_distribution(rng: np.random.Generator) -> Distribution:
    """A three-state mixture roughly matching the real points distribution."""
    p_none, p_short, _ = _STATE_SHARES
    lam_short = 0.6 + rng.random()
    lam_long = 2.0 + 3.0 * rng.random()

    support = range(-2, 26)
    masses = []
    for k in support:
        mass = 0.0
        if k == 0:
            mass += p_none
        if k >= 0:
            mass += p_short * math.exp(-lam_short) * lam_short**k / math.factorial(k)
            mass += (1 - p_none - p_short) * math.exp(-lam_long) * lam_long**k / math.factorial(k)
        masses.append(mass)
    total = math.fsum(masses)
    return Distribution(support_min=-2, pmf=tuple(m / total for m in masses))


#: Starters blank far less often than the league at large and return more when
#: they play. Kept separate from `_STATE_SHARES` because conflating the two is
#: precisely the error this file's incumbent control exists to catch.
_STARTER_SHARES = (0.20, 0.10)


def _starter_distribution(rng: np.random.Generator) -> Distribution:
    """A player who is actually in the eleven.

    The incumbent ``expected_points_sd`` fails hardest here, and the reason is
    worth stating because it is not obvious. On the *whole* population the mean
    is about 1.1 points, and the formula's 0.8 floor produces an interval that —
    by luck rather than design — happens to straddle the 60% zero atom, so the
    coverage check reports something acceptable. Starters average nearer 3.7 with
    a true spread around 2.8, while the incumbent still claims about 1.2. That is
    where the understatement bites.

    An earlier version of the control drew from `_truthful_distribution` while
    its docstring said "starters only". The assertions were right about the
    incumbent and wrong about the population, so the strongest negative control
    in the suite was quietly measuring the wrong thing.
    """
    p_none, p_short = _STARTER_SHARES
    lam_short = 1.0 + rng.random()
    lam_long = 3.5 + 3.0 * rng.random()

    masses = []
    for k in range(-2, 26):
        mass = 0.0
        if k == 0:
            mass += p_none
        if k >= 0:
            mass += p_short * math.exp(-lam_short) * lam_short**k / math.factorial(k)
            mass += (1 - p_none - p_short) * math.exp(-lam_long) * lam_long**k / math.factorial(k)
        masses.append(mass)
    total = math.fsum(masses)
    return Distribution(support_min=-2, pmf=tuple(m / total for m in masses))


def _sample(distribution: Distribution, rng: np.random.Generator) -> int:
    return int(rng.choice(list(distribution.support), p=list(distribution.pmf)))


def _truthful_population(n: int = 3000, seed: int = 7) -> tuple[list[Distribution], list[int]]:
    """Forecasts that genuinely generated their own outcomes.

    The only population for which "calibrated" is knowable a priori, so it is
    what the positive assertions are made against.
    """
    rng = np.random.default_rng(seed)
    distributions = [_truthful_distribution(rng) for _ in range(n)]
    actuals = [_sample(d, rng) for d in distributions]
    return distributions, actuals


class TestTheDistributionItself:
    def test_a_pmf_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="not 1"):
            Distribution(support_min=0, pmf=(0.5, 0.2))

    def test_a_negative_mass_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative mass"):
            Distribution(support_min=0, pmf=(1.2, -0.2))

    def test_mean_and_sd_match_the_lattice(self) -> None:
        d = Distribution(support_min=0, pmf=(0.5, 0.0, 0.5))
        assert d.mean == pytest.approx(1.0)
        assert d.sd == pytest.approx(1.0)

    def test_the_cdf_saturates_outside_the_support(self) -> None:
        d = Distribution(support_min=2, pmf=(0.5, 0.5))
        assert d.cdf_at(1) == 0.0
        assert d.cdf_at(2) == pytest.approx(0.5)
        assert d.cdf_at(99) == pytest.approx(1.0)

    def test_the_central_interval_widens_with_the_level(self) -> None:
        d = gaussian_pmf(5.0, 3.0)
        narrow = d.interval(0.5)
        wide = d.interval(0.95)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


class TestTheRandomisedPit:
    def test_it_is_uniform_when_the_forecast_is_truthful(self) -> None:
        """The property the whole check rests on."""
        distributions, actuals = _truthful_population()
        rng = np.random.default_rng(11)
        pits = [
            randomised_pit(d, a, float(rng.random()))
            for d, a in zip(distributions, actuals, strict=True)
        ]
        assert min(pits) >= 0.0
        assert max(pits) <= 1.0
        # A uniform sample has mean 0.5 and sd 1/sqrt(12) ~ 0.289.
        assert sum(pits) / len(pits) == pytest.approx(0.5, abs=0.03)

    def test_the_naive_pit_is_not_uniform_on_this_lattice(self) -> None:
        """Why the randomisation is mandatory rather than a refinement.

        On a discrete forecast, ``F(y)`` is stochastically *larger* than uniform:
        it charges the actual with the whole probability atom sitting on it
        rather than a random slice of it. With ~60% of FPL scores at zero the
        effect is severe, and an implementer who skips the randomisation reads
        the resulting skew as a broken model rather than a broken diagnostic.

        Measured as a departure from uniformity, which is the claim. An earlier
        version of this test counted *distinct* PIT values, expecting the naive
        version to concentrate — but every player carries a different
        distribution, so both versions land on ~999 distinct values out of 1000
        and the count sees nothing. The mean and the Cramér-von Mises statistic
        separate them by a factor of roughly 1000.
        """
        distributions, actuals = _truthful_population(n=1000)
        naive = [d.cdf_at(a) for d, a in zip(distributions, actuals, strict=True)]
        randomised = [
            randomised_pit(d, a, float(v))
            for (d, a), v in zip(
                zip(distributions, actuals, strict=True),
                np.random.default_rng(3).random(len(actuals)),
                strict=True,
            )
        ]

        # Uniform has mean 0.5. The naive PIT sits far above it.
        assert sum(naive) / len(naive) > 0.6
        assert sum(randomised) / len(randomised) == pytest.approx(0.5, abs=0.03)

        # And the same conclusion via the statistic the report itself uses.
        assert _cramer_von_mises(naive) > 100 * _cramer_von_mises(randomised)

    def test_it_is_reproducible_across_runs(self) -> None:
        distributions, actuals = _truthful_population(n=200)
        frame = frame_from_distributions(distributions, actuals)
        first = score_distributions(frame)
        second = score_distributions(frame)
        assert first.overall.pit_uniformity == second.overall.pit_uniformity
        assert first.overall.pit_histogram == second.overall.pit_histogram


class TestThePositiveDirection:
    """A truthful forecast must come out calibrated, or the metrics are noise."""

    @pytest.fixture(scope="class")
    @staticmethod
    def report() -> object:
        distributions, actuals = _truthful_population()
        return score_distributions(frame_from_distributions(distributions, actuals))

    def test_the_dispersion_ratio_is_near_one(self, report: object) -> None:
        assert report.overall.dispersion_ratio == pytest.approx(1.0, abs=0.1)  # type: ignore[attr-defined]

    def test_coverage_is_near_nominal_at_every_level(self, report: object) -> None:
        for band in report.overall.coverage:  # type: ignore[attr-defined]
            # Lattice over-coverage is expected and one-sided; a shortfall is not.
            assert band.empirical >= band.nominal - 0.05, band.row()

    def test_it_is_not_flagged_degenerate(self, report: object) -> None:
        assert not report.overall.degenerate  # type: ignore[attr-defined]


class TestControlOneSyntheticMisdispersion:
    """A known factor must move the ratio the matching way. Both directions."""

    @staticmethod
    def _ratio_at(factor: float) -> float:
        distributions, actuals = _truthful_population(n=2000)
        skewed = [inflate_pmf(d, factor) for d in distributions]
        report = score_distributions(frame_from_distributions(skewed, actuals))
        return report.overall.dispersion_ratio

    def test_narrowed_forecasts_report_over_dispersion(self) -> None:
        """Halve the width and the outcomes are twice as spread as claimed."""
        assert self._ratio_at(0.5) > 1.5

    def test_widened_forecasts_report_under_dispersion(self) -> None:
        assert self._ratio_at(2.0) < 0.75

    def test_the_two_directions_are_distinguishable(self) -> None:
        """A check that reports the same thing for both is not a check."""
        assert self._ratio_at(0.5) > self._ratio_at(2.0)

    def test_narrowing_also_collapses_coverage(self) -> None:
        distributions, actuals = _truthful_population(n=2000)
        narrow = [inflate_pmf(d, 0.4) for d in distributions]
        report = score_distributions(frame_from_distributions(narrow, actuals))
        eighty = next(b for b in report.overall.coverage if b.nominal == 0.8)
        assert eighty.empirical < 0.8


class TestControlTwoTheIncumbent:
    """The strongest control: the code that ships today.

    `expected_points_sd = max(0.5, |total| * minutes_sd/90 + 0.8)` where
    `minutes_sd = 30*(1 - |mean-45|/45) + 6`. Fed through this check as a
    Gaussian, it must be flagged. If it is not, the check is broken — and the
    numbers below are pinned so a future change that silently stops flagging it
    fails here rather than in a report nobody reads.
    """

    @staticmethod
    def _incumbent_sd(total: float, expected_minutes: float) -> float:
        minutes_sd = 30.0 * (1.0 - abs(expected_minutes - 45.0) / 45.0) + 6.0
        return max(0.5, abs(total) * (minutes_sd / 90.0) + 0.8)

    @pytest.fixture(scope="class")
    @staticmethod
    def report() -> object:
        """Starters only — the population the incumbent fails hardest on."""
        rng = np.random.default_rng(5)
        distributions: list[Distribution] = []
        actuals: list[int] = []
        minutes: list[int] = []
        for _ in range(3000):
            truth = _starter_distribution(rng)
            outcome = _sample(truth, rng)
            expected_minutes = 80.0 + 10.0 * rng.random()
            sd = TestControlTwoTheIncumbent._incumbent_sd(truth.mean, expected_minutes)
            distributions.append(gaussian_pmf(truth.mean, sd))
            actuals.append(outcome)
            minutes.append(int(expected_minutes))
        return score_distributions(
            frame_from_distributions(distributions, actuals, extra={"minutes": minutes})
        )

    def test_it_is_flagged_as_badly_under_covered(self, report: object) -> None:
        eighty = next(b for b in report.overall.coverage if b.nominal == 0.8)  # type: ignore[attr-defined]
        assert eighty.empirical < 0.6, (
            f"the incumbent scored {eighty.empirical:.1%} at a nominal 80%; if this "
            "check reports acceptable coverage for it, the check is broken"
        )

    def test_the_dispersion_ratio_exposes_the_understated_width(self, report: object) -> None:
        assert report.overall.dispersion_ratio > 1.5  # type: ignore[attr-defined]

    def test_it_is_not_dismissed_as_degenerate(self, report: object) -> None:
        """The incumbent has width — it is wrong, not absent. Confusing the two
        would let a real failure be filed as 'no forecast to assess'."""
        assert not report.overall.degenerate  # type: ignore[attr-defined]

    def test_the_pit_is_visibly_non_uniform(self, report: object) -> None:
        assert report.overall.pit_uniformity > 0.5  # type: ignore[attr-defined]


class TestControlThreeADegenerateForecaster:
    @pytest.fixture(scope="class")
    @staticmethod
    def report() -> object:
        distributions, actuals = _truthful_population(n=500)
        points = [point_mass_pmf(round(d.mean)) for d in distributions]
        return score_distributions(frame_from_distributions(points, actuals))

    def test_every_metric_is_marked_uninterpretable(self, report: object) -> None:
        assert report.overall.degenerate  # type: ignore[attr-defined]
        assert report.overall.dispersion_ratio == 0.0  # type: ignore[attr-defined]

    def test_the_report_says_so_rather_than_reporting_excellence(self, report: object) -> None:
        assert any("no width" in w for w in report.warnings)  # type: ignore[attr-defined]

    def test_coverage_collapses(self, report: object) -> None:
        for band in report.overall.coverage:  # type: ignore[attr-defined]
            assert band.mean_width == pytest.approx(1.0)


class TestTheSlicesAreStructural:
    def test_appeared_only_is_not_optional(self) -> None:
        """The zero atom drags every pooled statistic toward looking fine, so
        the restricted population is part of the type."""
        distributions, actuals = _truthful_population(n=400)
        report = score_distributions(frame_from_distributions(distributions, actuals))
        assert report.appeared_only is not None
        assert report.appeared_only.n <= report.overall.n

    def test_a_missing_minutes_column_is_declared_not_faked(self) -> None:
        distributions, actuals = _truthful_population(n=200)
        report = score_distributions(frame_from_distributions(distributions, actuals))
        assert any("no `minutes` column" in w for w in report.warnings)

    def test_slices_appear_when_their_column_does(self) -> None:
        distributions, actuals = _truthful_population(n=400)
        rng = np.random.default_rng(2)
        frame = frame_from_distributions(
            distributions,
            actuals,
            extra={
                "position": [["GKP", "DEF", "MID", "FWD"][i % 4] for i in range(400)],
                "price": [int(v) for v in rng.integers(40, 140, 400)],
                "minutes": [int(v) for v in rng.integers(0, 91, 400)],
                "predicted": [d.mean for d in distributions],
            },
        )
        report = score_distributions(frame)
        assert {s.name for s in report.by_position} == {"GKP", "DEF", "MID", "FWD"}
        assert report.by_price_band
        assert {s.name for s in report.by_minutes_band} <= {
            "did not play",
            "cameo (1-59)",
            "started (60+)",
        }
        assert len(report.by_predicted_decile) == 10

    def test_the_pooled_number_can_hide_a_conditional_failure(self) -> None:
        """Half the population well calibrated, half badly. The pooled ratio
        lands near acceptable and `conditionally_inconsistent` fires anyway."""
        distributions, actuals = _truthful_population(n=2000)
        mixed = [d if i % 2 == 0 else inflate_pmf(d, 0.35) for i, d in enumerate(distributions)]
        minutes = [90 if i % 2 else 0 for i in range(len(mixed))]
        report = score_distributions(
            frame_from_distributions(mixed, actuals, extra={"minutes": minutes})
        )
        assert report.conditionally_inconsistent

    def test_a_missing_required_column_refuses_rather_than_guesses(self) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            score_distributions(pl.DataFrame({"actual": [1, 2]}))

    def test_an_empty_frame_reports_nothing_rather_than_zero_metrics(self) -> None:
        report = score_distributions(
            pl.DataFrame(
                {"actual": [], "pmf_support_min": [], "pmf": []},
                schema={
                    "actual": pl.Int64,
                    "pmf_support_min": pl.Int64,
                    "pmf": pl.List(pl.Float64),
                },
            )
        )
        assert report.overall.n == 0
        assert any("no rows" in w for w in report.warnings)


class TestCrpsAndPinball:
    def test_crps_is_zero_for_a_perfect_point_forecast(self) -> None:
        distributions = [point_mass_pmf(4)]
        report = score_distributions(frame_from_distributions(distributions, [4]))
        assert report.overall.crps == pytest.approx(0.0, abs=1e-9)

    def test_crps_rewards_the_better_forecast(self) -> None:
        """Proper: the truthful forecast must score lower than a shifted one."""
        distributions, actuals = _truthful_population(n=1500)
        shifted = [Distribution(support_min=d.support_min + 4, pmf=d.pmf) for d in distributions]
        truthful = score_distributions(frame_from_distributions(distributions, actuals))
        wrong = score_distributions(frame_from_distributions(shifted, actuals))
        assert truthful.overall.crps < wrong.overall.crps

    def test_pinball_is_reported_at_every_level(self) -> None:
        distributions, actuals = _truthful_population(n=200)
        report = score_distributions(frame_from_distributions(distributions, actuals))
        levels = [lvl for lvl, _ in report.overall.pinball]
        assert levels == [0.1, 0.25, 0.5, 0.75, 0.9]
        assert all(loss >= 0.0 for _, loss in report.overall.pinball)
