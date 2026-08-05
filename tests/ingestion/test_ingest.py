"""Ingestion tests, run entirely against mocked HTTP.

No test here touches the live API. Ingest must be reproducible from stored
bytes, and a suite that depends on a third party's uptime stops being a signal.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from xg_alonso.contracts.identifiers import Season
from xg_alonso.domain import ScoringRules, SquadRules
from xg_alonso.domain.drift import diff_scoring_rules, diff_squad_rules
from xg_alonso.pipelines.ingestion import (
    FPL_BASE_URL,
    SOURCE_BOOTSTRAP,
    SOURCE_FIXTURES,
    FplApiClient,
    OfflineError,
    derive_available_time,
    detect_preseason_hazards,
    ingest_bootstrap,
)
from xg_alonso.storage import FileSystemBronzeStore

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def payload(bootstrap_payload: dict[str, Any]) -> dict[str, Any]:
    """The pinned snapshot. Every drift test below deep-copies before mutating."""
    return bootstrap_payload


class TestAvailableTimeDerivation:
    def test_date_minus_age(self) -> None:
        """The origin published at or before Date - Age."""
        available, _ = derive_available_time(
            {"date": "Mon, 27 Jul 2026 12:00:00 GMT", "age": "120"},
            fallback=NOW,
        )
        assert available == datetime(2026, 7, 27, 11, 58, tzinfo=UTC)

    def test_missing_age_treated_as_zero(self) -> None:
        available, _ = derive_available_time(
            {"date": "Mon, 27 Jul 2026 12:00:00 GMT"}, fallback=NOW
        )
        assert available == datetime(2026, 7, 27, 12, 0, tzinfo=UTC)

    def test_missing_date_falls_back(self) -> None:
        available, _ = derive_available_time({}, fallback=NOW)
        assert available == NOW

    def test_malformed_headers_fall_back_rather_than_crash(self) -> None:
        available, _ = derive_available_time(
            {"date": "not-a-date", "age": "not-a-number"}, fallback=NOW
        )
        assert available == NOW

    def test_negative_age_clamped(self) -> None:
        available, _ = derive_available_time(
            {"date": "Mon, 27 Jul 2026 12:00:00 GMT", "age": "-500"}, fallback=NOW
        )
        assert available == datetime(2026, 7, 27, 12, 0, tzinfo=UTC)

    def test_result_is_always_utc_aware(self) -> None:
        available, _ = derive_available_time(
            {"date": "Mon, 27 Jul 2026 12:00:00 +0500"}, fallback=NOW
        )
        assert available.tzinfo is not None
        assert available == datetime(2026, 7, 27, 7, 0, tzinfo=UTC)


class TestOfflineGuard:
    def test_offline_mode_blocks_network_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CI sets this so an accidental live call fails loudly instead of flaking."""
        monkeypatch.setenv("XG_ALONSO_OFFLINE", "1")
        with FplApiClient() as client, pytest.raises(OfflineError, match="reproducible"):
            client.bootstrap_static()


@respx.mock
def _mock_endpoints(payload: dict[str, Any]) -> None:
    respx.get(f"{FPL_BASE_URL}/bootstrap-static/").mock(
        return_value=httpx.Response(
            200,
            json=payload,
            headers={"date": "Mon, 27 Jul 2026 12:00:00 GMT", "age": "30"},
        )
    )


class TestIngestBootstrap:
    @respx.mock
    def test_writes_snapshots_and_parses_rules(
        self, tmp_path: Path, payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XG_ALONSO_OFFLINE", raising=False)
        respx.get(f"{FPL_BASE_URL}/bootstrap-static/").mock(
            return_value=httpx.Response(
                200, json=payload, headers={"date": "Mon, 27 Jul 2026 12:00:00 GMT", "age": "30"}
            )
        )
        respx.get(f"{FPL_BASE_URL}/fixtures/").mock(return_value=httpx.Response(200, json=[]))

        bronze = FileSystemBronzeStore(tmp_path)
        with FplApiClient() as client:
            result = ingest_bootstrap(
                client=client, bronze=bronze, season=Season("2026-27"), run_id="r1"
            )

        assert len(result.snapshots) == 2
        assert {s.source for s in result.snapshots} == {SOURCE_BOOTSTRAP, SOURCE_FIXTURES}
        assert not result.rules_drifted, "no pin supplied, so nothing can have drifted"

    @respx.mock
    def test_rerunning_does_not_duplicate_snapshots(
        self, tmp_path: Path, payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A half-finished ingest must be safe to simply run again."""
        monkeypatch.delenv("XG_ALONSO_OFFLINE", raising=False)
        respx.get(f"{FPL_BASE_URL}/bootstrap-static/").mock(
            return_value=httpx.Response(
                200, json=payload, headers={"date": "Mon, 27 Jul 2026 12:00:00 GMT"}
            )
        )
        respx.get(f"{FPL_BASE_URL}/fixtures/").mock(return_value=httpx.Response(200, json=[]))

        bronze = FileSystemBronzeStore(tmp_path)
        with FplApiClient() as client:
            ingest_bootstrap(client=client, bronze=bronze, season=Season("2026-27"), run_id="r1")
            ingest_bootstrap(client=client, bronze=bronze, season=Season("2026-27"), run_id="r2")

        assert len(bronze.list_snapshots(SOURCE_BOOTSTRAP)) == 1

    @respx.mock
    def test_rule_drift_is_detected(
        self, tmp_path: Path, payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silent revaluation would make every stored prediction wrong."""
        monkeypatch.delenv("XG_ALONSO_OFFLINE", raising=False)

        stale = json.loads(json.dumps(payload))
        stale["game_config"]["scoring"]["assists"] = 99
        pinned_scoring = ScoringRules.from_bootstrap(
            stale, version="2026-27", source_sha256="a" * 64, fetched_at=NOW
        )
        pinned_squad = SquadRules.from_bootstrap(payload, version="2026-27", source_sha256="a" * 64)

        respx.get(f"{FPL_BASE_URL}/bootstrap-static/").mock(
            return_value=httpx.Response(
                200, json=payload, headers={"date": "Mon, 27 Jul 2026 12:00:00 GMT"}
            )
        )
        respx.get(f"{FPL_BASE_URL}/fixtures/").mock(return_value=httpx.Response(200, json=[]))

        bronze = FileSystemBronzeStore(tmp_path)
        with FplApiClient() as client:
            result = ingest_bootstrap(
                client=client,
                bronze=bronze,
                season=Season("2026-27"),
                run_id="r1",
                pinned_scoring=pinned_scoring,
                pinned_squad=pinned_squad,
            )

        assert result.rules_drifted
        assert any(c.field == "assists" for c in result.rule_changes)


class TestDriftDiff:
    def test_identical_rules_report_no_change(self, payload: dict[str, Any]) -> None:
        a = ScoringRules.from_bootstrap(
            payload, version="2026-27", source_sha256="a" * 64, fetched_at=NOW
        )
        b = ScoringRules.from_bootstrap(
            payload, version="2026-27", source_sha256="b" * 64, fetched_at=NOW
        )
        assert diff_scoring_rules(a, b) == [], "hash and fetch time must not count as drift"

    def test_goalkeeper_revaluation_is_caught(self, payload: dict[str, Any]) -> None:
        changed = json.loads(json.dumps(payload))
        changed["game_config"]["scoring"]["goals_scored"]["GKP"] = 6
        pinned = ScoringRules.from_bootstrap(
            payload, version="x", source_sha256="a" * 64, fetched_at=NOW
        )
        live = ScoringRules.from_bootstrap(
            changed, version="x", source_sha256="b" * 64, fetched_at=NOW
        )
        changes = diff_scoring_rules(pinned, live)
        assert [c.field for c in changes] == ["goals_scored"]

    def test_squad_constraint_change_is_caught(self, payload: dict[str, Any]) -> None:
        changed = json.loads(json.dumps(payload))
        changed["game_config"]["rules"]["max_extra_free_transfers"] = 1
        pinned = SquadRules.from_bootstrap(payload, version="x", source_sha256="a" * 64)
        live = SquadRules.from_bootstrap(changed, version="x", source_sha256="b" * 64)
        changes = diff_squad_rules(pinned, live)
        assert any(c.field == "max_extra_free_transfers" for c in changes)


class TestPreseasonHazards:
    def test_zeroed_strength_fields_are_flagged(self, payload: dict[str, Any]) -> None:
        """Verified live: every team reads 0 on these before the season starts."""
        warnings = detect_preseason_hazards(payload)
        flagged = {w.field_name for w in warnings}
        assert "strength_attack_home" in flagged
        assert "strength_defence_home" in flagged

    def test_preseason_strength_scale_is_flagged(self, payload: dict[str, Any]) -> None:
        warnings = detect_preseason_hazards(payload)
        assert any(w.field_name == "strength_overall_home" for w in warnings)

    def test_in_season_payload_is_not_flagged(self, payload: dict[str, Any]) -> None:
        """The guard must not cry wolf once real values arrive."""
        in_season = json.loads(json.dumps(payload))
        for i, team in enumerate(in_season["teams"]):
            team["strength_attack_home"] = 1100 + i
            team["strength_attack_away"] = 1150 + i
            team["strength_defence_home"] = 1200 + i
            team["strength_defence_away"] = 1250 + i
            team["strength_overall_home"] = 1300 + i
        for element in in_season["elements"]:
            element["chance_of_playing_next_round"] = 100

        assert detect_preseason_hazards(in_season) == []

    def test_empty_payload_is_handled(self) -> None:
        assert detect_preseason_hazards({}) == []
