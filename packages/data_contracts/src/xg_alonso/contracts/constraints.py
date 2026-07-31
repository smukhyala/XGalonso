"""Rule violations, as shared vocabulary.

:class:`SquadViolation` began in :mod:`xg_alonso.domain.constraints`, which is
where it is produced. It moved here because it is also *carried*: a simulation
records the transfers it refused, and ``contracts`` may not import ``domain``.

The alternative was a second type in ``contracts`` mirroring the first, which
``CLAUDE.md`` rules out directly — two names for one concept is exactly the
duplication the boundary exists to prevent. ``domain.constraints`` re-exports
it, so every existing import still resolves.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["SquadViolation"]


class SquadViolation(BaseModel):
    """One broken rule, with enough detail to explain it without re-deriving it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    detail: str
