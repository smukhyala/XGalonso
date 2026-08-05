"""Fixtures every suite was otherwise rebuilding for itself.

Nineteen test modules independently reconstructed the path to the pinned
`bootstrap-static` snapshot, each with its own `Path(__file__).parents[N]` walk.
The literal `parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"`
appeared twenty times. That is not merely repetitive: the depth is encoded in
every copy, so moving one test file one directory deeper breaks it in a way that
reads as a missing fixture rather than as a wrong relative path, and adding a
directory level to `tests/` would break nineteen files at once.

**The snapshot is the reason the duplication mattered.** CLAUDE.md forbids
writing FPL scoring values and squad constraints as Python literals, so a test
that wants real rules has no alternative to loading this file. Making it easy to
load correctly is what stops the next test from typing `assists = 3` instead.

Three things are shared here and nothing else:

- `repo_root`, so a test that needs another repository file computes it once;
- `bootstrap_payload`, the raw snapshot;
- `scoring_rules` and `squad_rules`, built from it with the arguments
  `tests/domain/test_domain_rules.py` established.

Modules whose construction differs — a different `source_sha256`, a
`fetched_at` of `datetime.now(UTC)`, a payload deliberately mutated before
loading — keep their own fixtures. Forcing those onto a shared one would change
what they measure to make them look uniform, and uniformity is not the goal.

**`bootstrap_payload` is session-scoped and shared, so treat it as read-only.**
Every current consumer that mutates it already deep-copies first
(`json.loads(json.dumps(payload))`), which is what makes sharing safe; a test
that mutated it in place would silently corrupt every module that ran after it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest

from xg_alonso.domain.rules import SquadRules
from xg_alonso.domain.scoring import ScoringRules

#: The repository root. One walk, in the one file whose depth cannot change.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: The pinned `bootstrap-static` snapshot. Real `game_config.scoring` and
#: `game_config.rules`; the player list is truncated, so a test needing a full
#: 15-man squad synthesises a roster around these rules rather than using them
#: as a roster — see `tests/test_end_to_end.py` and `tests/api/conftest.py`.
BOOTSTRAP_FIXTURE: Final[Path] = REPO_ROOT / "data/fixtures/fpl/bootstrap_static_2026_27.json"

#: Provenance for rules built from the fixture. These are the values
#: `tests/domain/test_domain_rules.py` established and most modules copied: a
#: distinct hash per rule kind, so a test that confuses the two fails loudly.
RULES_VERSION: Final[str] = "2026-27"
SCORING_SHA256: Final[str] = "a" * 64
SQUAD_SHA256: Final[str] = "b" * 64

#: When the snapshot is treated as having been fetched. Pinned rather than
#: `datetime.now(UTC)`: `fetched_at` is provenance a drift check reads, and a
#: clock reading makes a test's inputs differ between two runs of it.
RULES_FETCHED_AT: Final[datetime] = datetime(2026, 7, 27, tzinfo=UTC)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, for tests that read a file other than the snapshot."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def bootstrap_payload() -> dict[str, Any]:
    """The pinned snapshot, parsed once.

    Shared, so do not mutate it. Deep-copy first — every existing caller that
    changes a rule to test drift detection already does.
    """
    payload: dict[str, Any] = json.loads(BOOTSTRAP_FIXTURE.read_text())
    return payload


@pytest.fixture(scope="session")
def scoring_rules(bootstrap_payload: dict[str, Any]) -> ScoringRules:
    """Scoring values read from the snapshot, never transcribed.

    A goalkeeper goal is 10 here because the payload says 10, which is the whole
    argument for loading it: the intuitive answer is 6 and it would survive
    review.
    """
    return ScoringRules.from_bootstrap(
        bootstrap_payload,
        version=RULES_VERSION,
        source_sha256=SCORING_SHA256,
        fetched_at=RULES_FETCHED_AT,
    )


@pytest.fixture(scope="session")
def squad_rules(bootstrap_payload: dict[str, Any]) -> SquadRules:
    """Squad size, positional quotas, budget and the per-club cap, from the snapshot."""
    return SquadRules.from_bootstrap(
        bootstrap_payload, version=RULES_VERSION, source_sha256=SQUAD_SHA256
    )
