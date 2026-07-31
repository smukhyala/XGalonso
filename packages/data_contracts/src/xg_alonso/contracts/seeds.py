"""One seed, and every other derived from it by name.

There were eight occurrences of the literal ``20260727`` across four packages —
an estimator's ``random_state``, a discovery harness, a permutation-importance
run, a backtest's policy RNG. Unrelated things reaching for the same number is
not reproducibility; it is a coincidence that looks like a scheme, and it has
two concrete costs. Two RNGs that should be independent can draw identical
sequences, and "which seed produced this run" has no answer beyond "that one".

:func:`derive_seed` replaces the scheme without replacing the numbers. A child
seed is a hash of the root and a labelled path, so
``derive_seed(ROOT_SEED, "policy", "random", "2024-25", 6, "abc123", 2)`` is
stable, reproducible, and *answerable* — the label says what it was for.

**The estimator keeps the bare root, deliberately.** Applying ``derive_seed`` to
``trained.py``'s ``random_state`` would change every fitted artifact and move
the headline before the evaluation framework can reproduce it. Naming a
constant is a refactor; renumbering it is an experiment. This module does the
first only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

__all__ = ["ROOT_SEED", "SeedLedger", "derive_seed"]

ROOT_SEED: Final[int] = 20260727
"""The one seed. Pinned by a test so a refactor cannot silently move it."""

#: numpy and scikit-learn both want a seed inside the 32-bit range.
_MASK: Final[int] = 2**32 - 1


def derive_seed(root: int, *parts: str | int) -> int:
    """A stable child seed from a root and a labelled path.

    Args:
        root: Normally :data:`ROOT_SEED`.
        parts: What this seed is *for*, most general first. Order matters and is
            part of the identity.

    Returns:
        A seed in ``[0, 2**32)``, identical across processes and machines.

    Uses SHA-256 rather than :func:`hash`, which is salted per process — a
    ``PYTHONHASHSEED``-dependent seed is the opposite of what this is for.
    """
    payload = "|".join([str(root), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & _MASK


@dataclass(frozen=True)
class SeedLedger:
    """Every seed actually drawn, in order, with what it was for.

    Recorded rather than recomputed. A run that reports its seeds can be
    reproduced by someone who does not have the code that derived them, which
    is the difference between reproducible and re-derivable.
    """

    entries: tuple[tuple[str, int], ...] = ()

    def record(self, label: str, value: int) -> SeedLedger:
        return SeedLedger(entries=(*self.entries, (label, value)))

    def for_label(self, label: str) -> int | None:
        return next((value for name, value in self.entries if name == label), None)
