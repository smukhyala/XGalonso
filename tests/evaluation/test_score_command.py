"""The distributional report as `xg score` actually renders it.

`tests/evaluation/test_distribution_calibration.py` proves the *metrics* can
fail. This file is about the wiring: that the report reaches a user at all, that
it is honest about what it scored, and that the specific honesty this module
insists on — coverage never shown without the width that bought it, a degenerate
forecast marked uninterpretable rather than excellent — survives the trip
through the CLI's rendering instead of being flattened into a single number.

The command body is exercised through `_echo_distribution_report` rather than
through a full `xg score` invocation. The rest of that command needs a backfill
of four seasons on disk, which CI does not have; the frame it hands over is one
row per player-gameweek and is trivially constructible, so the boundary is drawn
there. What the full command does *before* that boundary is unchanged by this
work and is not what these tests are about.
"""

from __future__ import annotations

import inspect

import polars as pl
import pytest

from xg_alonso.cli.main import _echo_distribution_report, score


def _frame(
    *,
    sd: float,
    n: int = 400,
    seed: int = 20260804,
) -> pl.DataFrame:
    """A scoring frame shaped exactly as `xg score` assembles one.

    Outcomes are drawn from a genuinely dispersed process — a zero-inflated
    count, which is what FPL points are — so the reported `sd` can be right or
    wrong about it rather than being right by construction. Every player is
    given 90 minutes so `appeared only` is the whole frame: the headline check
    reads that slice, and a frame that was mostly non-appearances would let the
    zero atom cover for a bad `sd`, which is the exact failure the slice exists
    to prevent.
    """
    import numpy as np

    generator = np.random.default_rng(seed)
    actual = generator.poisson(lam=4.0, size=n)
    return pl.DataFrame(
        {
            "player_code": list(range(1, n + 1)),
            "predicted": [4.0] * n,
            "predicted_sd": [sd] * n,
            "position": ["MID"] * n,
            "actual": [int(v) for v in actual],
            "minutes": [90] * n,
            "price": [70] * n,
            "gameweek_id": [1] * n,
        }
    )


class TestTheReportSaysWhatItScored:
    def test_it_is_labelled_a_negative_control_not_a_forecast(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The one claim that must not be lost in the rendering.

        This prediction path emits a mean and a standard deviation, never a PMF.
        Scoring a Gaussian built from that pair is legitimate — it is what
        `gaussian_pmf` documents itself as, the strongest available negative
        control — but presenting it as "the distributional model, checked" would
        be a claim the repository cannot support.
        """
        _echo_distribution_report(_frame(sd=1.0))
        out = capsys.readouterr().out

        assert "NEGATIVE CONTROL" in out
        assert "nothing in this prediction path produces a predictive distribution" in (
            " ".join(out.lower().split())
        )
        assert "expected_points_sd" in out

    def test_the_pooled_number_never_appears_alone(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Slices are structural in the report type, and must reach the screen."""
        _echo_distribution_report(_frame(sd=1.0))
        out = capsys.readouterr().out

        assert "appeared only" in out
        assert "By minutes" in out
        assert "By position" in out
        assert "By price band" in out


class TestCoverageTravelsWithItsWidth:
    def test_every_coverage_line_carries_a_mean_width(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A check that covers by being wide is not a check.

        So a rendering that showed coverage without width would let a forecast
        buy a passing number with a twenty-point interval, and nothing on screen
        would say so.
        """
        _echo_distribution_report(_frame(sd=1.0))
        lines = [line for line in capsys.readouterr().out.splitlines() if "->" in line]

        assert lines, "no coverage rows were rendered at all"
        assert all("width=" in line for line in lines)

    def test_the_finding_quotes_the_width_it_was_measured_at(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _echo_distribution_report(_frame(sd=0.5))
        out = capsys.readouterr().out

        assert "FINDING" in out
        finding = next(line for line in out.splitlines() if "FINDING" in line)
        assert "mean" in finding
        assert "width" in finding


class TestTheFindingFiresOnlyWhenItShould:
    def test_an_over_narrow_sd_is_called_out(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The incumbent's failure mode, reproduced in miniature.

        Outcomes have a true spread of about two points; the forecast claims a
        half. Under-coverage of the 80% interval is then not an opinion, and the
        command has to say so rather than print the table and move on.
        """
        _echo_distribution_report(_frame(sd=0.5))
        out = capsys.readouterr().out

        assert "FINDING" in out
        assert "understates risk" in out

    def test_a_truthful_sd_is_not_called_out(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The other half of the control. A warning that always fires says nothing.

        Poisson(4) has a standard deviation of exactly two, so a forecast
        reporting two is telling the truth about its own spread and must pass.
        """
        _echo_distribution_report(_frame(sd=2.0))
        out = capsys.readouterr().out

        assert "FINDING" not in out

    def test_a_degenerate_forecast_is_uninterpretable_not_excellent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Zero width covers nothing, and must never read as a clean bill.

        `score_distributions` flags this itself; what is checked here is that
        the flag and its warning both survive into the rendered output, because
        a degenerate slice whose warning was swallowed would show a tiny CRPS
        and look like the best result in the file.
        """
        _echo_distribution_report(_frame(sd=0.0))
        out = capsys.readouterr().out

        assert "DEGENERATE" in out
        assert "uninterpretable rather than excellent" in out


class TestTheReportIsOnByDefault:
    def test_the_flag_defaults_to_on(self) -> None:
        """An audit you must opt into is an audit nobody runs.

        The default is the whole point of the wiring: `expected_points_sd` has
        shipped unchecked, and a flag that had to be remembered would have left
        it that way.
        """
        default = inspect.signature(score).parameters["distributions"].default
        assert default is True

    def test_the_point_estimate_report_is_not_conditional_on_it(self) -> None:
        """The distributional report is additive; nothing above it may change.

        Checked structurally: `score` still takes the same required inputs and
        the flag only gates the new call, so an existing invocation of
        `xg score` prints what it always did with the new section appended.
        """
        parameters = inspect.signature(score).parameters
        assert parameters["distributions"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        # The flag was appended, so every existing positional call site keeps
        # its meaning.
        assert list(parameters)[-1] == "distributions"
