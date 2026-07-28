"""Points-derivation tests.

The property that matters is reconciliation. A derivation is only worth showing
if it reproduces the number beside it — prose that quietly disagrees with its
own total is worse than no prose, because it looks like a check that passed.
"""

from __future__ import annotations

import pytest

from tests.explanations.conftest import RULES, make_prediction
from xg_alonso.contracts.prediction import Position
from xg_alonso.explanations.derivation import derive_points


class TestReconciliation:
    @pytest.mark.parametrize("position", list(Position))
    def test_the_lines_reproduce_the_total(self, position: Position) -> None:
        derivation = derive_points(make_prediction(position=position), RULES)
        assert derivation.reconciles, f"{position} derivation does not sum to its own total"

    def test_the_reported_total_is_the_prediction_total(self) -> None:
        prediction = make_prediction()
        assert derive_points(prediction, RULES).total == pytest.approx(prediction.expected_points)


class TestArithmetic:
    def test_a_goal_line_is_expectation_times_the_positional_rate(self) -> None:
        """A forward is paid 4 for a goal and a defender 6; the line must show it."""
        for position, rate in ((Position.FWD, 4.0), (Position.DEF, 6.0)):
            derivation = derive_points(make_prediction(position=position), RULES)
            goals = next(line for line in derivation.lines if line.component == "Goals")
            assert goals.rate == rate
            assert goals.points == pytest.approx(goals.expectation * rate)

    def test_a_goalkeeper_goal_is_worth_ten(self) -> None:
        """The constant the project header names as the reason nothing is typed
        from memory. If this ever reads 6, the snapshot is not being read."""
        derivation = derive_points(make_prediction(position=Position.GKP), RULES)
        goals = next(line for line in derivation.lines if line.component == "Goals")
        assert goals.rate == 10.0

    def test_the_appearance_line_names_its_two_branches(self) -> None:
        """It is not a product, and saying it is would be a lie about the maths."""
        line = next(
            line
            for line in derive_points(make_prediction(), RULES).lines
            if line.component == "Appearing"
        )
        assert not line.is_exact_product
        assert "branches" in line.note

    def test_threshold_terms_declare_their_approximation(self) -> None:
        derivation = derive_points(make_prediction(position=Position.GKP), RULES)
        for component in ("Goals conceded", "Saves"):
            line = next(line for line in derivation.lines if line.component == component)
            assert line.note, f"{component} hides that it is paid per completed group"


class TestReadability:
    def test_material_lines_are_ordered_by_contribution(self) -> None:
        material = derive_points(make_prediction(), RULES).material()
        sizes = [abs(line.points) for line in material]
        assert sizes == sorted(sizes, reverse=True)

    def test_a_forward_is_not_shown_a_clean_sheet_line(self) -> None:
        """It is exactly zero and always will be; listing it teaches nothing."""
        material = derive_points(make_prediction(position=Position.FWD), RULES).material()
        assert all(line.component != "Clean sheet" for line in material)

    def test_every_line_renders_its_own_arithmetic(self) -> None:
        for line in derive_points(make_prediction(), RULES).material():
            rendered = line.explain()
            assert line.component in rendered
            assert "pts" in rendered
