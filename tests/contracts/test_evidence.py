"""Evidence panel and reason-vocabulary tests.

The property test over the whole vocabulary is the important one here. Reasons
are added by hand, and the failure mode is a code whose template asks for a
placeholder no builder supplies — which produces a reason that silently never
fires, exactly the defect that left fixtures out of every explanation the system
produced for its first four releases.
"""

from __future__ import annotations

import pytest

from xg_alonso.contracts.evidence import (
    EXPLANATORY_PANEL,
    FeatureEvidence,
    FeatureValue,
    panel_feature_names,
)
from xg_alonso.contracts.identifiers import PlayerCode
from xg_alonso.contracts.reason_codes import (
    REASON_TEMPLATES,
    Reason,
    ReasonCode,
    ReasonPolarity,
    template_placeholders,
)
from xg_alonso.features.catalogue import feature_names
from xg_alonso.features.opponent import OPPONENT_FEATURES


class TestPanel:
    def test_every_panel_feature_exists_in_the_catalogue(self) -> None:
        """A panel entry naming a column nobody builds is a reason that never fires."""
        available = set(feature_names()) | set(OPPONENT_FEATURES)
        missing = [name for name in panel_feature_names() if name not in available]
        assert not missing, f"panel names features the catalogue does not build: {missing}"

    def test_panel_names_are_unique(self) -> None:
        names = panel_feature_names()
        assert len(names) == len(set(names))

    def test_panel_is_small_enough_to_read(self) -> None:
        """Curation is the point. A panel of 171 explains nothing."""
        assert len(EXPLANATORY_PANEL) <= 20


class TestFeatureValue:
    def test_a_middling_percentile_is_not_notable(self) -> None:
        value = FeatureValue(name="x", label="x", family="f", value=1.0, percentile=0.51)
        assert not value.is_notable

    def test_an_extreme_percentile_is_notable(self) -> None:
        for percentile in (0.05, 0.95):
            value = FeatureValue(
                name="x", label="x", family="f", value=1.0, percentile=percentile
            )
            assert value.is_notable

    def test_a_missing_percentile_is_never_notable(self) -> None:
        value = FeatureValue(name="x", label="x", family="f", value=1.0, percentile=None)
        assert not value.is_notable

    def test_favourability_inverts_when_lower_is_better(self) -> None:
        """High volatility ranks high and argues against the player."""
        volatile = FeatureValue(
            name="v", label="v", family="f", value=3.0, percentile=0.9, higher_is_better=False
        )
        assert volatile.favourability == pytest.approx(0.1)

    def test_percentile_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            FeatureValue(name="x", label="x", family="f", value=1.0, percentile=1.4)


class TestNotableOrdering:
    def test_notable_values_come_back_most_distinguishing_first(self) -> None:
        evidence = FeatureEvidence(
            values=(
                FeatureValue(name="a", label="a", family="f", value=1.0, percentile=0.75),
                FeatureValue(name="b", label="b", family="f", value=1.0, percentile=0.99),
                FeatureValue(name="c", label="c", family="f", value=1.0, percentile=0.50),
                FeatureValue(name="d", label="d", family="f", value=1.0, percentile=0.02),
            )
        )
        assert [v.name for v in evidence.notable()] == ["b", "d", "a"]


class TestVocabulary:
    def test_every_code_has_a_template(self) -> None:
        missing = [code for code in ReasonCode if code not in REASON_TEMPLATES]
        assert not missing, f"codes without templates: {missing}"

    def test_no_template_is_orphaned(self) -> None:
        assert set(REASON_TEMPLATES) == set(ReasonCode)

    @pytest.mark.parametrize("code", list(ReasonCode))
    def test_every_template_renders_from_supplied_placeholders(self, code: ReasonCode) -> None:
        """Construct each code with exactly what its template asks for.

        Numeric slots go through evidence and descriptive ones through context,
        which is also a check that the split is declared consistently: a slot
        that needs a string but is fed a float renders as a number and reads as
        nonsense.
        """
        placeholders = template_placeholders(REASON_TEMPLATES[code])
        textual = {"position"}
        reason = Reason(
            code=code,
            polarity=ReasonPolarity.CONTEXT,
            subject=PlayerCode(1),
            evidence={name: 1.0 for name in placeholders - textual},
            context={name: "MID" for name in placeholders & textual},
            weight=0.0,
        )
        rendered = reason.render()
        assert rendered
        assert "{" not in rendered


class TestGrounding:
    def test_a_reason_missing_evidence_cannot_be_constructed(self) -> None:
        with pytest.raises(ValueError, match="missing evidence"):
            Reason(
                code=ReasonCode.EXPECTED_MINUTES_SECURE,
                polarity=ReasonPolarity.SUPPORTS_IN,
                subject=PlayerCode(1),
                evidence={"p_start": 0.9},  # expected_minutes omitted
                weight=1.0,
            )

    def test_evidence_the_template_never_renders_is_rejected(self) -> None:
        """Unread evidence is unchecked evidence, and usually a stale builder."""
        with pytest.raises(ValueError, match="never renders"):
            Reason(
                code=ReasonCode.HOME_FIXTURE,
                polarity=ReasonPolarity.CONTEXT,
                subject=PlayerCode(1),
                evidence={"unused": 1.0},
                weight=0.0,
            )

    def test_a_slot_defined_twice_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="both evidence and context"):
            Reason(
                code=ReasonCode.BONUS_MAGNET,
                polarity=ReasonPolarity.CONTEXT,
                subject=PlayerCode(1),
                evidence={"value": 1.0, "percentile": 0.9, "position": 1.0},
                context={"position": "MID"},
                weight=0.0,
            )

    def test_a_grounded_reason_renders_its_own_numbers(self) -> None:
        reason = Reason(
            code=ReasonCode.XG_RATE_HIGHER,
            polarity=ReasonPolarity.SUPPORTS_IN,
            subject=PlayerCode(7),
            evidence={"value": 0.61, "other": 0.18, "percentile": 0.84},
            context={"position": "FWD"},
            weight=1.0,
        )
        rendered = reason.render()
        assert "0.61" in rendered
        assert "0.18" in rendered
        assert "84%" in rendered
        assert "FWD" in rendered
