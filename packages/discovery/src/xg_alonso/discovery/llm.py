"""An optional language-model proposer, gated behind every check the loop already has.

**What the model is allowed to do.** Propose a hypothesis and a candidate program
*as data*. Nothing else. Its output is a JSON expression tree that goes through
exactly the same three gates a deterministic proposal does:

1. :func:`~xg_alonso.discovery.dsl.parse_program` — refuses an unknown node kind,
   a level mismatch, a malformed window. Never returns a partially-understood
   tree.
2. :func:`~xg_alonso.discovery.compile.validate_program` — checks every column
   against the real schema, blocks the target columns, enforces depth and size.
3. The point-in-time leakage harness, before the registry will accept it.

**Nothing is ever ``eval``'d or ``exec``'d, and no Python is generated.** The
model cannot express a computation the DSL cannot express, so the worst a bad or
adversarial proposal can do is fail validation. That is the property that makes
this safe to enable at all — not the model's good behaviour.

**The LLM path is optional and off by default.** Without an API key the
deterministic generator in :mod:`~xg_alonso.discovery.hypotheses` runs alone and
the loop is unchanged. Every proposal that does come from here is recorded with
``GenerationSource.LLM`` forever, so an LLM-suggested feature stays permanently
distinguishable from a measured one in the registry.

**A rationale is a hypothesis, not a finding.** The model's ``football_rationale``
is stored beside the fold-level evidence, never in place of it. Acceptance is
decided by the same walk-forward measurement and the same controls as everything
else — the model gets no vote.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from pydantic import BaseModel, Field

from xg_alonso.contracts.discovery import (
    FeatureHypothesis,
    GenerationSource,
    LeakageRisk,
)
from xg_alonso.discovery.dsl import FeatureProgram, ProgramError, parse_program
from xg_alonso.discovery.hypotheses import SeededHypothesis

__all__ = [
    "DEFAULT_MODEL",
    "LlmProposal",
    "LlmProposalBatch",
    "LlmUnavailableError",
    "api_key_origin",
    "generate_with_llm",
    "load_api_key",
]

#: Anthropic's most capable widely-available model. Hypothesis generation is a
#: low-volume, high-leverage call — a handful of proposals per experiment — so
#: the cost of the strongest model is negligible against the cost of testing a
#: bad hypothesis over five walk-forward folds.
DEFAULT_MODEL: Final[str] = "claude-opus-5"


class LlmUnavailableError(RuntimeError):
    """No API key, or the SDK is not installed.

    Raised rather than silently returning nothing: a caller that asked for LLM
    proposals and received none should be able to tell "the model had no ideas"
    apart from "the model was never called".
    """


def _read_key(path: Path) -> str | None:
    """Extract ``ANTHROPIC_API_KEY`` from one ``.env`` file, if it defines one."""
    try:
        text = path.read_text()
    except OSError:
        return None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() == "ANTHROPIC_API_KEY":
            return value.strip().strip("'\"") or None
    return None


def _find_key(env_file: Path | None = None) -> tuple[str, str] | None:
    """Locate a key and record where it came from.

    Search order is environment, then the nearest ``.env`` that actually
    *defines* the key, walking upward from the working directory.

    The upward walk is not a convenience. A git worktree gets its own copy of
    untracked files, so a key written to the main checkout is invisible from a
    worktree — and a *stale* ``.env`` left behind in the worktree shadows the
    real one silently, producing a 401 that looks like a bad credential rather
    than a bad path. Walking up means a worktree with no ``.env`` of its own
    inherits the repository's.

    A file that exists but defines no key is skipped rather than ending the
    search, since a ``.env`` holding unrelated settings should not mask one that
    holds the key.

    Returns ``(key, origin)`` or ``None``. The origin is a path or the word
    ``environment`` — never the key itself.
    """
    from_env = os.environ.get("ANTHROPIC_API_KEY")
    if from_env:
        return from_env, "environment"

    if env_file is not None:
        key = _read_key(env_file)
        return (key, str(env_file)) if key else None

    start = Path.cwd().resolve()
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            key = _read_key(candidate)
            if key:
                return key, str(candidate)
    return None


def load_api_key(*, env_file: Path | None = None) -> str | None:
    """Find an Anthropic API key, from the environment or a nearby ``.env``.

    The SDK reads ``ANTHROPIC_API_KEY`` from the environment and does not parse
    ``.env`` files, so this bridges the gap without taking a dependency on
    ``python-dotenv`` for what is a few lines of parsing.

    Returns ``None`` when no key is found. **Never logs or returns the key's
    value anywhere else** — it is passed straight to the client. Use
    :func:`api_key_origin` when you need to say *where* a key came from.
    """
    found = _find_key(env_file)
    return found[0] if found else None


def api_key_origin(*, env_file: Path | None = None) -> str | None:
    """Where :func:`load_api_key` would read its key from, or ``None``.

    Exists so an authentication failure is one line of output away from being
    diagnosed. "The key is invalid" and "you are reading a different file than
    you think" produce the same 401, and only the second is common.
    """
    found = _find_key(env_file)
    return found[1] if found else None


class LlmProposal(BaseModel):
    """One hypothesis the model proposes. Data only — never code."""

    title: str = Field(description="One line naming the mechanism")
    football_rationale: str = Field(
        description="The football mechanism claimed, in two or three sentences"
    )
    falsification_condition: str = Field(
        description="What measured result would refute this hypothesis"
    )
    expected_relationship: str = Field(
        description="Direction and shape expected, and for which players"
    )
    program_name: str = Field(description="snake_case column name the program produces")
    program_json: str = Field(
        description=(
            "The feature program as a JSON expression tree, matching the DSL "
            "grammar exactly. Parsed and validated before use; never executed."
        )
    )


class LlmProposalBatch(BaseModel):
    """The model's whole response. A batch, so diversity can be asked for."""

    proposals: list[LlmProposal] = Field(default_factory=list)


@dataclass(frozen=True)
class _Evidence:
    """What the model is shown. Deliberately narrow."""

    available_columns: tuple[str, ...]
    entity_columns: tuple[str, ...]
    required_features: tuple[str, ...]
    weak_segments: tuple[tuple[str, str, float], ...]
    already_tried: tuple[str, ...]
    lessons: tuple[str, ...]
    objective: str
    emphasis: tuple[str, ...]


_GRAMMAR: Final[str] = """\
A program is a JSON object. Every node has a "kind". The grammar, exactly:

ROW-level (one value per historical match):
  {"kind":"source","column":"<history column>","scope":"history"}
  {"kind":"const","value":<number>}

ENTITY-level (one value per player per deadline) - produced by aggregating ROW:
  {"kind":"rolling","child":<ROW>,"window":1-40,"agg":"mean|median|std|min|max|sum|count|percentile","min_periods":<int>,"quantile":<0-1 or omit>}
  {"kind":"lag","child":<ROW>,"periods":1-40}
  {"kind":"ewm_mean","child":<ROW>,"window":2-40,"halflife":<float>}
  {"kind":"trend","child":<ROW>,"window":3-40}
  {"kind":"shrunk_rate","numerator":"<col>","denominator":"<col>","window":1-40,"prior_strength":<float>,"scale":90.0}
  {"kind":"time_since","event_column":"<col>","require_positive":true}
  {"kind":"source","column":"<entity column>","scope":"entity"}

COMBINATORS:
  {"kind":"arith","op":"add|sub|mul|safe_div|min|max","left":<node>,"right":<node>,"epsilon":1e-06}
  {"kind":"unary","op":"log1p|neg|abs|clip|zscore|percentile_rank","child":<node>,"lower":<num or omit>,"upper":<num or omit>}
  {"kind":"group_rel","op":"rank|share|dev_from_mean|zscore","by":"position|team|opponent|all","child":<ENTITY>}

HARD RULES - a program breaking any of these is rejected:
  - The root must be ENTITY level.
  - A temporal node's child must be ROW level. You cannot aggregate an aggregate.
  - arith may only combine two nodes of the SAME level (a const counts as either).
  - group_rel and unary zscore/percentile_rank need an ENTITY child.
  - std needs min_periods >= 2. trend needs window >= 3.
  - There is no division operator: use "safe_div".
  - Max depth 8, max 40 nodes.
"""


def _system_prompt() -> str:
    return (
        "You propose testable hypotheses about Fantasy Premier League player "
        "performance, and express each as a feature program in a restricted DSL.\n\n"
        "You are one proposer inside a scientific loop. Every program you emit is "
        "parsed, statically validated, proven point-in-time safe, computed over "
        "four seasons of history, and backtested walk-forward against a noise "
        "control and a shuffled control. Your rationale is recorded as a "
        "hypothesis, never as a finding. You get no vote on whether a feature is "
        "accepted.\n\n"
        "Because of that, a wrong-but-testable hypothesis is useful and a vague "
        "one is worthless. State a mechanism specific enough to be refuted.\n\n"
        "Propose mechanisms that are genuinely DIFFERENT from each other and from "
        "what has already been tried. Several near-identical windows of the same "
        "metric is a wasted batch."
    )


def _user_prompt(evidence: _Evidence, count: int) -> str:
    weak = (
        "\n".join(
            f"  - {kind} {segment}: relative error {error:.3f}"
            for kind, segment, error in evidence.weak_segments
        )
        or "  (not measured)"
    )
    tried = "\n".join(f"  - {name}" for name in evidence.already_tried) or "  (none yet)"
    lessons = "\n".join(f"  - {lesson}" for lesson in evidence.lessons) or "  (none yet)"

    return f"""\
## The manager's objective
{evidence.objective}
Emphasis: {", ".join(evidence.emphasis) or "none stated"}

## Features that must stay in the model
{", ".join(evidence.required_features) or "(none)"}

Your job is to find signal these do NOT already carry. A feature that merely
restates a required one adds nothing.

## Where the current model is measurably worst
{weak}

## What has already been tried (do not repeat these)
{tried}

## What past experiments concluded
{lessons}

## Columns you may read

History columns (ROW level, inside a window):
{", ".join(evidence.available_columns)}

Entity columns (ENTITY level, already on the prediction row):
{", ".join(evidence.entity_columns)}

Nothing else exists. There are no shot locations, no touch counts, no pressing
metrics, no set-piece data — the official FPL API does not publish them. Do not
invent a column; a program naming one is rejected.

## The DSL
{_GRAMMAR}

## Task
Propose {count} DIVERSE hypotheses. For each, give the football mechanism, what
would refute it, and the program as JSON in `program_json` (a JSON string, not a
nested object).
"""


def generate_with_llm(
    *,
    available_columns: Sequence[str],
    entity_columns: Sequence[str] = (),
    required_features: Sequence[str] = (),
    weak_segments: Sequence[tuple[str, str, float]] = (),
    already_tried: Sequence[str] = (),
    lessons: Sequence[str] = (),
    objective: str = "maximise expected points",
    emphasis: Sequence[str] = (),
    count: int = 4,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    env_file: Path | None = None,
    effort: str = "high",
) -> list[SeededHypothesis]:
    """Ask a language model for hypotheses, and keep only the ones that parse.

    Returns proposals in exactly the same shape the deterministic generator
    produces, so the experiment runner does not know or care which produced them
    — except that each carries ``GenerationSource.LLM`` permanently.

    Raises:
        LlmUnavailableError: when the SDK is missing or no key is configured. A caller
            that asked for LLM proposals and got none must be able to tell an
            empty result from an absent one.

    A malformed or hallucinated program is **dropped silently and counted**, not
    repaired. Repairing a proposal would make it partly the machine's hypothesis
    and partly ours, and the registry would attribute the whole thing to the
    model.
    """
    # Key first, then the SDK. Both are legitimate reasons to be unavailable, but
    # the missing key is far the more common and it is the one a user can act on
    # without touching the install. Checking the import first also made the error
    # depend on the environment: with the extra installed you were told about the
    # key, and without it you were told about the package, for the same call.
    key = api_key or load_api_key(env_file=env_file)
    if not key:
        raise LlmUnavailableError(
            "no ANTHROPIC_API_KEY in the environment or in .env. The deterministic "
            "generator runs without one; set a key only to enable LLM proposals."
        )

    try:
        import anthropic
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
        raise LlmUnavailableError(
            "the `anthropic` package is not installed. Install the optional extra: "
            "`uv sync --extra llm`. The deterministic generator runs without it."
        ) from exc

    evidence = _Evidence(
        available_columns=tuple(available_columns),
        entity_columns=tuple(entity_columns),
        required_features=tuple(required_features),
        weak_segments=tuple(weak_segments),
        already_tried=tuple(already_tried),
        lessons=tuple(lessons),
        objective=objective,
        emphasis=tuple(emphasis),
    )

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        # The SDK types `effort` as a Literal; it arrives here as a plain str
        # from configuration, so the dict is narrowed at the call boundary.
        output_config=cast("Any", {"effort": effort}),
        system=_system_prompt(),
        messages=[{"role": "user", "content": _user_prompt(evidence, count)}],
        output_format=LlmProposalBatch,
    )

    # Check the stop reason before reading content. A safety refusal returns a
    # normal 200 with empty or partial content, and indexing into it blindly is
    # how an optional feature turns into a crash in the main loop.
    if response.stop_reason == "refusal":
        return []

    batch = response.parsed_output
    if batch is None:
        return []

    return _compile_proposals(batch.proposals, already_tried=set(already_tried))


def _compile_proposals(
    proposals: Sequence[LlmProposal], *, already_tried: set[str]
) -> list[SeededHypothesis]:
    """Parse each proposal's program. Drop anything that does not survive."""
    out: list[SeededHypothesis] = []
    seen_versions: set[str] = set()

    for index, proposal in enumerate(proposals):
        program = _parse(proposal)
        if program is None:
            continue
        if program.name in already_tried or program.version() in seen_versions:
            continue
        seen_versions.add(program.version())

        out.append(
            SeededHypothesis(
                hypothesis=FeatureHypothesis(
                    id=f"llm.{program.name}",
                    title=proposal.title.strip() or program.name,
                    football_rationale=proposal.football_rationale.strip()
                    or "No mechanism stated.",
                    falsification_condition=proposal.falsification_condition.strip()
                    or "No incremental value over the required set in a majority of folds.",
                    expected_relationship=proposal.expected_relationship.strip(),
                    required_raw_fields=program.columns(),
                    transformation_plan=program.describe(),
                    # Permanently marked. An LLM proposal must never become
                    # indistinguishable from a measured one in the registry.
                    generation_source=GenerationSource.LLM,
                    # Declared by the proposer and then checked mechanically. A
                    # model that declares NONE and is repeatedly caught is a
                    # proposer worth distrusting.
                    leakage_risk=LeakageRisk.LOW,
                ),
                program=program,
            )
        )
        del index

    return out


def _parse(proposal: LlmProposal) -> FeatureProgram | None:
    """Parse one proposal's program, or ``None`` if it is not valid.

    Never repairs. A program that does not parse is the model's mistake, and
    silently fixing it would file our hypothesis under its name.
    """
    name = proposal.program_name.strip()
    if not name or not name.replace("_", "").isalnum():
        return None
    try:
        payload: Any = proposal.program_json
        if isinstance(payload, str):
            payload = json.loads(payload)
        return parse_program(name, payload)
    except (ProgramError, json.JSONDecodeError, TypeError, ValueError):
        return None
