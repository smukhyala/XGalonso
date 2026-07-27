"""Backtest report persistence tests.

The point of writing runs down is being able to answer "did that change help?".
That requires the record to survive, to carry enough provenance to know what
produced it, and to refuse to invent a baseline when there is no prior run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from xg_alonso.contracts.identifiers import GameweekId, PlayerCode, Season
from xg_alonso.evaluation import (
    BacktestReport,
    BacktestResult,
    GameweekOutcome,
    compare_to_previous,
    load_reports,
    write_report,
)

NOW = datetime(2026, 7, 27, 10, 0, tzinfo=UTC)


def _result(incremental: int, *, transfers: int = 2) -> BacktestResult:
    outcomes = []
    for i in range(max(transfers, 1)):
        outcomes.append(
            GameweekOutcome(
                season=Season("2025-26"),
                gameweek=GameweekId(6 + i),
                policy_points=50 + incremental,
                hold_points=50,
                hit_cost=0,
                transfer_made=i < transfers,
                player_out=PlayerCode(1) if i < transfers else None,
                player_in=PlayerCode(2) if i < transfers else None,
                predicted_gain=2.0,
                decision_delta=incremental,
            )
        )
    return BacktestResult(outcomes=outcomes)


def _report(run_id: str, totals: dict[str, int], created: datetime = NOW) -> BacktestReport:
    return BacktestReport(
        run_id=run_id,
        created_at=created,
        season="2025-26",
        first_gameweek=6,
        last_gameweek=25,
        code_version="a" * 40,
        model_fingerprint="b" * 64,
        model_trained_on=("2024-25",),
        results={name: _result(total) for name, total in totals.items()},
    )


class TestPersistence:
    def test_a_run_is_written_as_json_and_markdown(self, tmp_path: Path) -> None:
        destination = write_report(_report("r1", {"model": 5, "hold": 0}), tmp_path)
        assert (destination / "result.json").exists()
        assert (destination / "summary.md").exists()

    def test_the_record_carries_enough_provenance_to_reproduce_it(self, tmp_path: Path) -> None:
        destination = write_report(_report("r1", {"model": 5}), tmp_path)
        record = json.loads((destination / "result.json").read_text())
        assert record["code_version"] == "a" * 40
        assert record["model_fingerprint"] == "b" * 64
        assert record["model_trained_on"] == ["2024-25"]
        assert record["season"] == "2025-26"

    def test_per_gameweek_outcomes_survive(self, tmp_path: Path) -> None:
        """Totals alone cannot explain *why* a run moved."""
        destination = write_report(_report("r1", {"model": 5}), tmp_path)
        record = json.loads((destination / "result.json").read_text())
        assert record["policies"]["model"]["outcomes"]

    def test_directory_names_sort_chronologically(self, tmp_path: Path) -> None:
        earlier = write_report(
            _report("r1", {"model": 1}, datetime(2026, 1, 1, tzinfo=UTC)), tmp_path
        )
        later = write_report(
            _report("r2", {"model": 2}, datetime(2026, 6, 1, tzinfo=UTC)), tmp_path
        )
        assert earlier.name < later.name

    def test_markdown_orders_policies_by_result(self, tmp_path: Path) -> None:
        report = _report("r1", {"weak": 1, "strong": 9})
        body = report.to_markdown()
        assert body.index("strong") < body.index("weak")

    def test_markdown_keeps_the_sample_size_caveat(self, tmp_path: Path) -> None:
        """A margin without its caveat reads as more certain than it is."""
        assert "small sample" in _report("r1", {"model": 5}).to_markdown()


class TestComparison:
    def test_the_first_run_has_nothing_to_compare_against(self, tmp_path: Path) -> None:
        """Inventing a baseline would be worse than admitting there isn't one."""
        report = _report("r1", {"model": 5})
        write_report(report, tmp_path)
        assert compare_to_previous(report, tmp_path) is None

    def test_a_later_run_is_compared_to_the_previous_one(self, tmp_path: Path) -> None:
        write_report(_report("r1", {"model": 5}, datetime(2026, 1, 1, tzinfo=UTC)), tmp_path)
        current = _report("r2", {"model": 9}, datetime(2026, 6, 1, tzinfo=UTC))
        write_report(current, tmp_path)

        delta = compare_to_previous(current, tmp_path)
        assert delta is not None
        was, now = delta["model"]
        assert now > was

    def test_a_different_window_is_not_compared(self, tmp_path: Path) -> None:
        """Comparing GW6-25 against GW6-38 would be comparing different things."""
        other = _report("r1", {"model": 5}, datetime(2026, 1, 1, tzinfo=UTC))
        object.__setattr__(other, "last_gameweek", 38)
        write_report(other, tmp_path)

        current = _report("r2", {"model": 9}, datetime(2026, 6, 1, tzinfo=UTC))
        write_report(current, tmp_path)
        assert compare_to_previous(current, tmp_path) is None

    def test_a_run_is_not_compared_against_itself(self, tmp_path: Path) -> None:
        report = _report("r1", {"model": 5})
        write_report(report, tmp_path)
        write_report(report, tmp_path)
        assert compare_to_previous(report, tmp_path) is None


class TestLoading:
    def test_missing_directory_yields_no_history(self, tmp_path: Path) -> None:
        assert load_reports(tmp_path / "absent") == []

    def test_a_corrupt_record_does_not_hide_the_others(self, tmp_path: Path) -> None:
        """Partial history beats none."""
        write_report(_report("r1", {"model": 5}, datetime(2026, 1, 1, tzinfo=UTC)), tmp_path)
        broken = tmp_path / "20260201T000000Z-broken-gw1-1"
        broken.mkdir()
        (broken / "result.json").write_text("{not json")

        loaded = load_reports(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["run_id"] == "r1"

    def test_season_filter(self, tmp_path: Path) -> None:
        write_report(_report("r1", {"model": 5}), tmp_path)
        assert load_reports(tmp_path, season="2024-25") == []
        assert len(load_reports(tmp_path, season="2025-26")) == 1

    def test_newest_first(self, tmp_path: Path) -> None:
        write_report(_report("old", {"model": 1}, datetime(2026, 1, 1, tzinfo=UTC)), tmp_path)
        write_report(_report("new", {"model": 2}, datetime(2026, 6, 1, tzinfo=UTC)), tmp_path)
        assert load_reports(tmp_path)[0]["run_id"] == "new"
