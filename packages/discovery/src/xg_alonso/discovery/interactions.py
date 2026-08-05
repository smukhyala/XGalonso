"""Constructed interaction terms, and the control that makes them meaningful.

``docs/ml/04_interaction_discovery.md`` was marked ``Deferred (Post-MVP)`` and
:func:`~xg_alonso.discovery.search.beam_search` — the only bundle-capable search
in the repository — was **called from nowhere**. The DSL could already express a
product (:class:`~xg_alonso.discovery.dsl.ArithOp.MUL`), so what was missing was
not the capability but the wiring and, more importantly, the *control* that
makes an accepted interaction mean anything.

Two decisions here are load-bearing.

Inlining, not referencing
-------------------------

An interaction is built by **inlining the component programs' root nodes** into
a single ``Arith(MUL, left, right)``, never by referencing their materialised
columns through ``Source(name, scope=ENTITY)``.

This is the most important safety property in the module. With an entity-scope
reference, :func:`~xg_alonso.discovery.experiment._prove_point_in_time` would
compile a program whose leaves are columns *already computed and sitting on the
entity frame*. Appending future records cannot change an already-computed
number, so :func:`~xg_alonso.features.leakage.find_leakage` would come back
clean **having proved nothing** — a false ``LEAKAGE_PASSED`` that
:meth:`~xg_alonso.discovery.registry.DiscoveryRegistry.register_feature` would
then honour, because it trusts that flag. Inlining forces the harness to
re-derive both halves from ``player_stats`` at every cutoff, so the proof is
real. ``tests/discovery/test_interactions.py`` plants a leaky component and
requires the interaction to fail the harness.

The additive control
--------------------

The shuffled control is the *wrong null* for an interaction. Permuting
``f_i x f_j`` preserves the product's marginal distribution, which tests whether
the product carries information — but a product of two useful features is useful
without any interaction existing at all. Beating that control would show only
that two good features had been added.

So an interaction must beat ``required u {f_i, f_j}`` with the components
entered **separately**. That is the difference between "this product helps" and
"this product helps *beyond what its parts already gave you*", and only the
second sentence justifies the word interaction. It costs nothing:
:func:`~xg_alonso.discovery.harness.make_scorer` memoises on the sorted column
tuple and the pair was already scored during bundle search.

No triples
----------

Interactions are generated **only from the programs that survived round one**,
never from other interactions, so nesting cannot occur. That is a property of
the caller, and it is the real guarantee — it matches
``search.beam_search``'s own refusal of ``bundle_size > 2`` and keeps an already
sevenfold-inflated hypothesis space from squaring.

It is worth being precise about what does *not* guarantee it. ``MAX_DEPTH`` (8)
and ``MAX_NODES`` (40) do refuse a nested interaction built from realistically
deep components, but they do **not** refuse one built from shallow ones: a
product of two `Rolling(Source)` programs is 5 nodes at depth 3, and a product
of two of *those* is 11 nodes at depth 4 — comfortably legal. An earlier
version of this docstring claimed the depth limits were the protection. They are
a backstop for the pathological cases, not the mechanism.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from xg_alonso.discovery.dsl import Arith, ArithOp, FeatureProgram

__all__ = [
    "INTERACTION_FORMS",
    "InteractionCandidate",
    "interaction_pairs",
    "interaction_program",
]

INTERACTION_FORMS: Final[tuple[str, ...]] = ("mul", "safe_div")
"""The two forms proposed.

``mul`` for "these matter together" — xG matters more when a player starts.
``safe_div`` for "this per unit of that" — the only division the language has,
with an explicit epsilon, so a generator cannot emit a divide-by-zero because
there is no node that would mean one.

Deliberately not ``add`` or ``sub``: a gradient-boosted model already fits
additive structure, so proposing a sum is proposing a column the model can
already build for itself.
"""

_FORM_OPS: Final[dict[str, ArithOp]] = {
    "mul": ArithOp.MUL,
    "safe_div": ArithOp.SAFE_DIV,
}


@dataclass(frozen=True)
class InteractionCandidate:
    """One proposed interaction and the two programs it came from."""

    program: FeatureProgram
    left: str
    right: str
    form: str

    @property
    def components(self) -> tuple[str, ...]:
        """The component column names, for the additive control."""
        return (self.left, self.right)


def _canonical(
    left: FeatureProgram, right: FeatureProgram
) -> tuple[FeatureProgram, FeatureProgram]:
    """Order operands by name.

    ``FeatureProgram.version()`` hashes the canonical JSON, so ``MUL(a, b)`` and
    ``MUL(b, a)`` hash *differently* and would register as two distinct
    features — identical in meaning, duplicated in the registry, and each
    diluting the other's evidence. Multiplication is commutative and safe_div is
    not, but ordering both keeps one rule rather than two: for a division the
    order simply decides which program is the numerator, and the name records
    which was chosen.
    """
    return (left, right) if left.name <= right.name else (right, left)


def interaction_program(
    left: FeatureProgram, right: FeatureProgram, *, form: str
) -> FeatureProgram | None:
    """Build one interaction, or ``None`` if the pair cannot form a legal one.

    Returns ``None`` rather than raising for a self-pairing or an unknown form:
    the caller generates pairs in bulk and a single unusable combination is not
    an error in the run.

    The returned program is **not** validated here — it goes through
    :func:`~xg_alonso.discovery.compile.validate_program` and the full leakage
    harness exactly like every other candidate. A program that is too deep is
    refused there, which is the intended outcome.
    """
    op = _FORM_OPS.get(form)
    if op is None or left.name == right.name:
        return None

    first, second = _canonical(left, right)
    root = Arith(op=op, left=first.root, right=second.root)
    return FeatureProgram(name=f"{first.name}__{form}__{second.name}", root=root)


def interaction_pairs(
    programs: Sequence[FeatureProgram],
    *,
    cap: int = 8,
    forms: Sequence[str] = INTERACTION_FORMS,
) -> list[InteractionCandidate]:
    """Every legal interaction over the top ``cap`` programs.

    ``cap`` bounds the combinatorics: eight programs give twenty-eight pairs,
    which across two forms is fifty-six candidates. That is already a sevenfold
    expansion of a typical round's hypothesis space, and the reason the caller
    tightens its acceptance threshold and sizes its noise null against the
    number of candidates actually scored.

    Ordering is deterministic — the caller passes programs best-first and pairs
    come out in that order — so a truncated run is reproducible rather than
    arbitrary.
    """
    chosen = list(programs)[: max(0, cap)]
    out: list[InteractionCandidate] = []
    for left, right in combinations(chosen, 2):
        for form in forms:
            program = interaction_program(left, right, form=form)
            if program is None:
                continue
            first, second = _canonical(left, right)
            out.append(
                InteractionCandidate(program=program, left=first.name, right=second.name, form=form)
            )
    return out
