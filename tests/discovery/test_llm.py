"""The optional language-model proposer, and the gates that contain it.

**No test here makes a network call.** The adapter's job is to turn a model's
JSON into a validated program or drop it, and that is entirely testable offline
— which is the point: the safety of this path does not depend on the model
behaving, so it should not depend on the model being reachable either.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.discovery.conftest import SOURCE_COLUMNS
from xg_alonso.contracts.discovery import GenerationSource
from xg_alonso.discovery.compile import validate_program
from xg_alonso.discovery.llm import (
    LlmProposal,
    LlmUnavailableError,
    _compile_proposals,
    generate_with_llm,
    load_api_key,
)


def _proposal(name: str, program: object, **overrides: str) -> LlmProposal:
    base = {
        "title": "A claim",
        "football_rationale": "Because of a mechanism.",
        "falsification_condition": "No gain in three of five folds.",
        "expected_relationship": "positive",
        "program_name": name,
        "program_json": program if isinstance(program, str) else json.dumps(program),
    }
    base.update(overrides)
    return LlmProposal(**base)


def _rolling(
    column: str, window: int = 5, agg: str = "mean", min_periods: int = 1
) -> dict[str, object]:
    return {
        "kind": "rolling",
        "child": {"kind": "source", "column": column, "scope": "history"},
        "window": window,
        "agg": agg,
        "min_periods": min_periods,
    }


class TestProposalsAreParsedNotTrusted:
    def test_a_well_formed_proposal_becomes_a_hypothesis(self) -> None:
        [result] = _compile_proposals(
            [_proposal("mins_5", _rolling("minutes"))], already_tried=set()
        )
        assert result.program.name == "mins_5"
        assert result.program.describe() == "mean_5(minutes)"
        assert result.hypothesis.falsification_condition

    def test_every_llm_proposal_is_permanently_marked(self) -> None:
        """An LLM suggestion must never become indistinguishable from a measurement."""
        [result] = _compile_proposals(
            [_proposal("mins_5", _rolling("minutes"))], already_tried=set()
        )
        assert result.hypothesis.generation_source is GenerationSource.LLM

    def test_malformed_json_is_dropped(self) -> None:
        assert _compile_proposals([_proposal("x", "not json at all")], already_tried=set()) == []

    def test_an_unknown_node_kind_is_dropped(self) -> None:
        assert (
            _compile_proposals([_proposal("x", {"kind": "invented_node"})], already_tried=set())
            == []
        )

    def test_a_level_mismatch_is_dropped(self) -> None:
        """The DSL's own type system rejects it before anything is computed."""
        mixed = {
            "kind": "arith",
            "op": "mul",
            "epsilon": 1e-06,
            "left": _rolling("minutes"),
            "right": {"kind": "source", "column": "minutes", "scope": "history"},
        }
        assert _compile_proposals([_proposal("x", mixed)], already_tried=set()) == []

    def test_a_hallucinated_column_survives_parsing_and_fails_validation(self) -> None:
        """Two gates, not one: the DSL cannot know which columns exist."""
        [result] = _compile_proposals(
            [_proposal("ghost", _rolling("shots_in_box"))], already_tried=set()
        )
        issues = validate_program(result.program, available_columns=SOURCE_COLUMNS)
        assert any(issue.code == "unknown_column" for issue in issues)

    def test_reading_the_target_is_refused(self) -> None:
        [result] = _compile_proposals(
            [_proposal("cheat", _rolling("label_total_points"))], already_tried=set()
        )
        issues = validate_program(
            result.program,
            available_columns=(*SOURCE_COLUMNS, "label_total_points"),
            forbidden_columns=("label_total_points",),
        )
        assert any(issue.code == "target_leakage" for issue in issues)

    def test_a_program_is_never_repaired(self) -> None:
        """Dropping is correct; fixing would file our hypothesis under the model's name."""
        broken = {
            "kind": "rolling",
            "child": {"kind": "source", "column": "minutes", "scope": "history"},
            "window": 5,
            "agg": "std",
            "min_periods": 1,
        }  # std needs min_periods >= 2
        assert _compile_proposals([_proposal("x", broken)], already_tried=set()) == []

    def test_an_unsafe_program_name_is_dropped(self) -> None:
        for name in ("../escape", "with space", "", "a;b"):
            assert (
                _compile_proposals([_proposal(name, _rolling("minutes"))], already_tried=set())
                == []
            )

    def test_already_tried_names_are_skipped(self) -> None:
        assert (
            _compile_proposals([_proposal("mins_5", _rolling("minutes"))], already_tried={"mins_5"})
            == []
        )

    def test_duplicate_semantics_within_a_batch_are_collapsed(self) -> None:
        """Two differently-named proposals computing the same thing are one idea."""
        results = _compile_proposals(
            [_proposal("a", _rolling("minutes")), _proposal("b", _rolling("minutes"))],
            already_tried=set(),
        )
        assert len(results) == 1

    def test_a_mixed_batch_keeps_only_what_survives(self) -> None:
        results = _compile_proposals(
            [
                _proposal("good", _rolling("minutes")),
                _proposal("broken", "not json"),
                _proposal("also_good", _rolling("bps", window=10, agg="std", min_periods=3)),
            ],
            already_tried=set(),
        )
        assert [r.program.name for r in results] == ["good", "also_good"]


class TestCredentialHandling:
    def test_a_missing_key_is_an_explicit_failure_not_silence(self, tmp_path: Path) -> None:
        """A caller must tell "no ideas" apart from "never called"."""
        with pytest.raises(LlmUnavailableError, match="ANTHROPIC_API_KEY"):
            generate_with_llm(
                available_columns=SOURCE_COLUMNS,
                api_key=None,
                env_file=tmp_path / "absent.env",
            )

    def test_a_key_is_read_from_a_dotenv_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text('# a comment\nOTHER=1\nANTHROPIC_API_KEY="sk-ant-test"\n')
        assert load_api_key(env_file=env) == "sk-ant-test"

    def test_the_environment_wins_over_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=from-file\n")
        assert load_api_key(env_file=env) == "from-env"

    def test_a_file_without_the_key_yields_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("SOMETHING_ELSE=1\n")
        assert load_api_key(env_file=env) is None
