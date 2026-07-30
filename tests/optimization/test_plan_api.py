"""Contract tests for the requirements API the plan page talks to.

These assert the *shape and discipline* of the two endpoints rather than the
optimizer's arithmetic, which `test_requirements.py` already covers against a
controlled pool. What matters here is that the UI can trust what it gets: an
edited requirement set replaces the parse rather than being silently re-derived,
and a requirement that could not be honoured is visible rather than absent.
"""

from __future__ import annotations

import pytest

from xg_alonso.contracts.identifiers import PlayerCode, TeamId, TenthsOfMillion
from xg_alonso.contracts.objective import (
    Requirement,
    RequirementKind,
    SquadRequirements,
)


class TestRequirementInputRoundTrip:
    """A chip the UI sends back must rebuild the requirement it came from.

    Exercised through Pydantic's own serialisation, because that is the path the
    API actually takes — a hand-rolled dict would test a shape the wire never
    carries.
    """

    CASES = (
        Requirement(kind=RequirementKind.MUST_START, label="x", players=(PlayerCode(1),)),
        Requirement(kind=RequirementKind.MUST_INCLUDE, label="x", players=(PlayerCode(2),)),
        Requirement(kind=RequirementKind.MUST_EXCLUDE, label="x", players=(PlayerCode(3),)),
        Requirement(kind=RequirementKind.MUST_CAPTAIN, label="x", players=(PlayerCode(4),)),
        Requirement(kind=RequirementKind.CLUB_FLOOR, label="x", team_id=TeamId(1), count=3),
        Requirement(kind=RequirementKind.CLUB_CEILING, label="x", team_id=TeamId(2), count=1),
        Requirement(kind=RequirementKind.FORMATION, label="x", formation="3-5-2"),
        Requirement(kind=RequirementKind.BANK_FLOOR, label="x", amount=TenthsOfMillion(5)),
    )

    @pytest.mark.parametrize("requirement", CASES, ids=lambda r: r.kind.value)
    def test_a_requirement_survives_json(self, requirement: Requirement) -> None:
        rebuilt = Requirement.model_validate_json(requirement.model_dump_json())
        assert rebuilt == requirement

    def test_an_unknown_kind_is_refused(self) -> None:
        """A typo in the UI must not become a requirement that binds nothing."""
        with pytest.raises(ValueError, match="not a valid"):
            RequirementKind("must_bench")


class TestEditedSetSemantics:
    """Dropping a chip must actually drop the requirement."""

    def test_a_dropped_requirement_is_absent_from_the_bundle(self) -> None:
        kept = Requirement(
            kind=RequirementKind.MUST_START, label="keep me", players=(PlayerCode(1),)
        )
        dropped = Requirement(kind=RequirementKind.FORMATION, label="drop me", formation="3-5-2")

        bundle = SquadRequirements(requirements=(kept,))
        labels = {r.label for r in bundle.requirements}
        assert "keep me" in labels
        assert dropped.label not in labels

    def test_an_empty_edited_set_is_the_free_optimum(self) -> None:
        """Clearing every chip must not fall back to re-parsing the sentence."""
        assert SquadRequirements(requirements=()).requirements == ()


class TestRelaxationOrderIsStable:
    def test_the_ui_can_predict_what_gives_first(self) -> None:
        """The page tells a manager which requirement will break first, so the
        order the solver actually uses has to match the order shown."""
        low = Requirement(
            kind=RequirementKind.BANK_FLOOR, label="bank", amount=TenthsOfMillion(5), priority=0
        )
        mid = Requirement(
            kind=RequirementKind.FORMATION, label="shape", formation="3-5-2", priority=2
        )
        high = Requirement(
            kind=RequirementKind.MUST_START, label="player", players=(PlayerCode(1),), priority=5
        )
        bundle = SquadRequirements(requirements=(high, mid, low))
        assert [r.label for r in bundle.relaxation_order()] == ["bank", "shape", "player"]
