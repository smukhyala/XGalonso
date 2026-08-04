"""Context encoding: shape, blocks, and the two firewall properties.

:class:`TestIdentityIsNotAFeature` and :class:`TestConstraintsCannotBuyPoints`
are the reason this module exists. Everything else here is ordinary coverage;
those two are executable statements of the guarantees
:mod:`xg_alonso.contracts.objective` spends its module docstring insisting on.

The encoding is a pure function of stated inputs, so these can be exhaustive
rather than illustrative — there is no I/O to stub and no row order to depend
on.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from xg_alonso.contracts.context import DecisionContext
from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    TeamId,
    TenthsOfMillion,
)
from xg_alonso.contracts.objective import (
    BeliefEntity,
    BeliefProposition,
    ManagerConstraints,
    ManagerObjective,
    ObjectiveBundle,
    OwnershipPreference,
    PrimaryMetric,
    Requirement,
    RequirementKind,
    RiskPreference,
    SquadRequirements,
    UserBelief,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadPick, SquadState
from xg_alonso.domain.context_features import (
    CONTEXT_BLOCKS,
    CONTEXT_FEATURE_NAMES,
    CONTEXT_FEATURE_VERSION,
    ContextVector,
    encode_context,
)
from xg_alonso.domain.rules import SquadRules

_LAYOUT = ((Position.GKP, 2), (Position.DEF, 5), (Position.MID, 5), (Position.FWD, 3))
_LEGAL_XI_ORDER = (0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 1, 6, 11, 14)


@pytest.fixture(scope="module")
def rules() -> SquadRules:
    """Real quotas and budget from the pinned snapshot, never typed here.

    CLAUDE.md is explicit that squad constants load from the pinned
    ``bootstrap-static`` payload rather than from a developer's memory, and this
    module's normalisation divisors are derived from them.
    """
    fixture = (
        Path(__file__).resolve().parents[2] / "data/fixtures/fpl/bootstrap_static_2026_27.json"
    )
    return SquadRules.from_bootstrap(
        json.loads(fixture.read_text()), version="2026-27", source_sha256="b" * 64
    )


def _squad(
    *,
    bank: int = 10,
    free_transfers: int = 1,
    first_code: int = 100,
    clubs_per_player: int = 12,
    price: int = 50,
) -> SquadState:
    picks: list[SquadPick] = []
    slot = 1
    code = first_code
    for position, count in _LAYOUT:
        for _ in range(count):
            picks.append(
                SquadPick(
                    player_code=PlayerCode(code),
                    position=position,
                    team_id=TeamId(1 + code % clubs_per_player),
                    purchase_price=TenthsOfMillion(price),
                    current_price=TenthsOfMillion(price),
                    selling_price=TenthsOfMillion(price),
                    squad_slot=slot,
                )
            )
            slot += 1
            code += 1
    ordered = [
        picks[original].model_copy(update={"squad_slot": index + 1})
        for index, original in enumerate(_LEGAL_XI_ORDER)
    ]
    return SquadState(
        entry_id=EntryId(1),
        gameweek=GameweekId(8),
        picks=tuple(ordered),
        bank=TenthsOfMillion(bank),
        free_transfers=free_transfers,
    )


def _context(
    *,
    constraints: ManagerConstraints | None = None,
    requirements: SquadRequirements | None = None,
    squad: SquadState | None = None,
    beliefs: tuple[UserBelief, ...] = (),
    objective: ManagerObjective | None = None,
) -> DecisionContext:
    return DecisionContext(
        bundle=ObjectiveBundle(
            objective=objective
            or ManagerObjective(
                id="expected_points_balanced_h1_neutral",
                name="Maximise expected points",
                primary_metric=PrimaryMetric.EXPECTED_POINTS,
            ),
            constraints=constraints or ManagerConstraints(),
            beliefs=beliefs,
        ),
        requirements=requirements or SquadRequirements(),
        squad=squad,
        as_of_gameweek=GameweekId(8),
    )


class TestShapeAndNaming:
    def test_width_matches_declared_names(self, rules: SquadRules) -> None:
        vector = encode_context(_context(squad=_squad()), rules=rules)
        assert vector.values.shape == (len(CONTEXT_FEATURE_NAMES),)
        assert vector.names == CONTEXT_FEATURE_NAMES
        assert vector.version == CONTEXT_FEATURE_VERSION

    def test_blocks_tile_the_vector_without_gaps_or_overlap(self) -> None:
        """A gap would be a silently unused dimension; an overlap, a double count."""
        covered: list[int] = []
        for block in CONTEXT_BLOCKS.values():
            covered.extend(range(block.start, block.stop))
        assert sorted(covered) == list(range(len(CONTEXT_FEATURE_NAMES)))

    def test_names_are_unique(self) -> None:
        assert len(set(CONTEXT_FEATURE_NAMES)) == len(CONTEXT_FEATURE_NAMES)

    def test_width_is_fixed_regardless_of_how_much_is_known(self, rules: SquadRules) -> None:
        """Fixed width is what lets this be a model input at all."""
        sparse = encode_context(_context(), rules=rules)
        dense = encode_context(
            _context(
                squad=_squad(),
                constraints=ManagerConstraints(locked_players=(PlayerCode(100),)),
                beliefs=(
                    UserBelief(
                        entity_type=BeliefEntity.PLAYER,
                        entity_id=100,
                        proposition=BeliefProposition.WILL_RETURN,
                        confidence=0.7,
                    ),
                ),
            ),
            rules=rules,
        )
        assert sparse.values.shape == dense.values.shape

    def test_all_values_are_finite(self, rules: SquadRules) -> None:
        """A NaN here propagates silently into a distance or a gradient."""
        for squad in (None, _squad(bank=0, price=1), _squad(bank=900)):
            vector = encode_context(_context(squad=squad), rules=rules)
            assert np.all(np.isfinite(vector.values)), vector.as_mapping()

    def test_vector_is_read_only(self, rules: SquadRules) -> None:
        vector = encode_context(_context(squad=_squad()), rules=rules)
        with pytest.raises(ValueError, match="read-only"):
            vector.values[0] = 99.0

    def test_mismatched_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="diverged"):
            ContextVector(values=np.zeros(3), names=("a", "b"), version=CONTEXT_FEATURE_VERSION)

    def test_unknown_block_raises(self, rules: SquadRules) -> None:
        vector = encode_context(_context(), rules=rules)
        with pytest.raises(KeyError, match="unknown context block"):
            vector.block("nonsense")


class TestIdentityIsNotAFeature:
    """Relabelling every player must not move the vector by one bit.

    If this fails, a learned conditioning would memorise player codes and
    transfer to no manager who did not own those exact players — which would
    make the whole conditioning claim vacuous.
    """

    def test_permuting_player_codes_is_bitwise_invariant(self, rules: SquadRules) -> None:
        original = _squad(first_code=100)
        relabelled = _squad(first_code=7000)

        forwards_a = tuple(p.player_code for p in original.picks if p.position is Position.FWD)
        forwards_b = tuple(p.player_code for p in relabelled.picks if p.position is Position.FWD)

        first = encode_context(
            _context(constraints=ManagerConstraints(locked_players=forwards_a), squad=original),
            rules=rules,
        )
        second = encode_context(
            _context(constraints=ManagerConstraints(locked_players=forwards_b), squad=relabelled),
            rules=rules,
        )
        assert np.array_equal(first.values, second.values)
        assert first.fingerprint() == second.fingerprint()

    def test_which_forward_is_locked_does_not_matter_only_how_many(self, rules: SquadRules) -> None:
        """Two managers freezing different single forwards face the same shape."""
        squad = _squad()
        forwards = [p.player_code for p in squad.picks if p.position is Position.FWD]
        first = encode_context(
            _context(constraints=ManagerConstraints(locked_players=(forwards[0],)), squad=squad),
            rules=rules,
        )
        second = encode_context(
            _context(constraints=ManagerConstraints(locked_players=(forwards[1],)), squad=squad),
            rules=rules,
        )
        assert np.array_equal(first.block("slot_pressure"), second.block("slot_pressure"))


class TestConstraintsCannotBuyPoints:
    """The firewall, checked at the signature rather than in the body."""

    def test_encoder_accepts_no_points_bearing_argument(self) -> None:
        """No parameter from which expected points could be recovered.

        This is the structural guarantee that a constraint can never be traded
        against points: there is nothing to trade *with*. Asserted on the
        signature so adding such a parameter fails loudly rather than quietly
        opening the door.
        """
        parameters = set(inspect.signature(encode_context).parameters)
        assert parameters == {"context", "rules", "club_fixture_load"}

        forbidden = {"predictions", "prediction", "expected_points", "points", "scoring"}
        assert not (parameters & forbidden)

    def test_objective_block_moves_with_the_objective(self, rules: SquadRules) -> None:
        """Soft preferences change the vector — they are supposed to."""
        conservative = _context(
            objective=ManagerObjective(
                id="a",
                name="a",
                primary_metric=PrimaryMetric.DOWNSIDE_PROTECTION,
                risk_preference=RiskPreference.CONSERVATIVE,
                ownership_preference=OwnershipPreference.TEMPLATE,
            )
        )
        aggressive = _context(
            objective=ManagerObjective(
                id="b",
                name="b",
                primary_metric=PrimaryMetric.DIFFERENTIAL_YIELD,
                risk_preference=RiskPreference.AGGRESSIVE,
                ownership_preference=OwnershipPreference.DIFFERENTIAL,
            )
        )
        first = encode_context(conservative, rules=rules).block("objective")
        second = encode_context(aggressive, rules=rules).block("objective")
        assert not np.array_equal(first, second)

    def test_aggressive_risk_encodes_a_negative_variance_penalty(self, rules: SquadRules) -> None:
        """A deficit-chaser wants variance; the sign must survive encoding.

        Treating risk aversion as universally positive is the modelling error
        `RiskPreference.variance_sign` exists to prevent, so it is worth
        checking that the encoding does not quietly take an absolute value.
        """
        aggressive = _context(
            objective=ManagerObjective(
                id="b",
                name="b",
                primary_metric=PrimaryMetric.EXPECTED_POINTS,
                risk_preference=RiskPreference.AGGRESSIVE,
            )
        )
        vector = encode_context(aggressive, rules=rules)
        assert vector.as_mapping()["objective_signed_uncertainty_penalty"] < 0


class TestSlotPressure:
    def test_rises_with_locks_in_that_position(self, rules: SquadRules) -> None:
        squad = _squad()
        forwards = [p.player_code for p in squad.picks if p.position is Position.FWD]
        readings = []
        for count in range(len(forwards) + 1):
            vector = encode_context(
                _context(
                    constraints=ManagerConstraints(locked_players=tuple(forwards[:count])),
                    squad=squad,
                ),
                rules=rules,
            )
            readings.append(vector.as_mapping()["slot_pressure_fwd"])
        assert readings == sorted(readings)
        assert readings[0] == pytest.approx(0.0)
        assert readings[-1] == pytest.approx(1.0)

    def test_locking_forwards_leaves_other_positions_alone(self, rules: SquadRules) -> None:
        squad = _squad()
        forwards = tuple(p.player_code for p in squad.picks if p.position is Position.FWD)
        vector = encode_context(
            _context(constraints=ManagerConstraints(locked_players=forwards), squad=squad),
            rules=rules,
        )
        reading = vector.as_mapping()
        assert reading["slot_pressure_fwd"] == pytest.approx(1.0)
        assert reading["slot_pressure_def"] == pytest.approx(0.0)
        assert reading["slot_pressure_mid"] == pytest.approx(0.0)


class TestBudgetBlock:
    def test_bank_is_binding_when_at_the_floor(self, rules: SquadRules) -> None:
        context = _context(
            constraints=ManagerConstraints(minimum_bank=TenthsOfMillion(20)),
            squad=_squad(bank=10),
        )
        assert encode_context(context, rules=rules).as_mapping()["budget_bank_is_binding"] == 1.0

    def test_bank_is_not_binding_with_room(self, rules: SquadRules) -> None:
        context = _context(
            constraints=ManagerConstraints(minimum_bank=TenthsOfMillion(5)),
            squad=_squad(bank=50),
        )
        assert encode_context(context, rules=rules).as_mapping()["budget_bank_is_binding"] == 0.0

    def test_headroom_per_slot_rises_as_the_squad_freezes(self, rules: SquadRules) -> None:
        """Freezing removes slots faster than it removes money.

        This direction is deliberate and worth pinning, because it reads
        backwards at first glance. Locking twelve players strands their sale
        value — total spendable falls hard — but it also leaves only three slots
        to fill, so each *remaining* slot commands more. A manager who has
        protected most of their squad is not poor; they are constrained, and
        what money they have is concentrated.

        The other half of that trade is carried by ``slot_pressure_*``, which
        rises as options disappear. Encoding both is what lets a model tell
        "rich and flexible" from "rich but boxed in" — two situations that call
        for different recommendations and that a single budget number cannot
        separate.
        """
        squad = _squad(bank=30)
        codes = [p.player_code for p in squad.picks]
        free = encode_context(_context(squad=squad), rules=rules).as_mapping()
        frozen = encode_context(
            _context(constraints=ManagerConstraints(locked_players=tuple(codes[:12])), squad=squad),
            rules=rules,
        ).as_mapping()

        assert frozen["budget_headroom_per_open_slot"] > free["budget_headroom_per_open_slot"]
        # ...and the constraint really did bite, elsewhere in the vector.
        assert sum(frozen[f"slot_pressure_{p}"] for p in ("gkp", "def", "mid", "fwd")) > sum(
            free[f"slot_pressure_{p}"] for p in ("gkp", "def", "mid", "fwd")
        )

    def test_a_smaller_bank_lowers_headroom(self, rules: SquadRules) -> None:
        """Holding slots fixed, less money must read as less headroom."""
        poor = encode_context(_context(squad=_squad(bank=0)), rules=rules).as_mapping()
        rich = encode_context(_context(squad=_squad(bank=60)), rules=rules).as_mapping()
        assert poor["budget_headroom_per_open_slot"] < rich["budget_headroom_per_open_slot"]

    def test_from_scratch_reports_full_headroom(self, rules: SquadRules) -> None:
        reading = encode_context(_context(), rules=rules).as_mapping()
        assert reading["budget_headroom_per_open_slot"] == pytest.approx(1.0)
        assert reading["budget_squad_value_share"] == pytest.approx(0.0)


class TestFormationBlock:
    def test_unconstrained_formation_reports_free(self, rules: SquadRules) -> None:
        reading = encode_context(_context(squad=_squad()), rules=rules).as_mapping()
        assert reading["formation_is_fixed"] == 0.0
        assert reading["formation_reachable_share"] == pytest.approx(1.0)

    def test_fixed_formation_narrows_the_reachable_share(self, rules: SquadRules) -> None:
        shaped = SquadRequirements(
            requirements=(
                Requirement(kind=RequirementKind.FORMATION, label="play 3-5-2", formation="3-5-2"),
            )
        )
        reading = encode_context(
            _context(squad=_squad(), requirements=shaped), rules=rules
        ).as_mapping()
        assert reading["formation_is_fixed"] == 1.0
        assert reading["formation_defenders"] == pytest.approx(0.3)
        assert reading["formation_midfielders"] == pytest.approx(0.5)
        assert reading["formation_forwards"] == pytest.approx(0.2)
        assert 0.0 < reading["formation_reachable_share"] < 1.0


class TestFixtureBlock:
    def test_absent_fixture_data_is_flagged_not_silently_zero(self, rules: SquadRules) -> None:
        """A zero meaning 'no edge' and one meaning 'nobody told me' differ.

        The availability flag is the whole reason this block is four wide.
        """
        reading = encode_context(_context(squad=_squad()), rules=rules).as_mapping()
        assert reading["fixture_outlook_available"] == 0.0
        assert reading["fixture_locked_load"] == 0.0

    def test_present_fixture_data_sets_the_flag_and_the_gap(self, rules: SquadRules) -> None:
        squad = _squad(clubs_per_player=12)
        locked = tuple(p.player_code for p in squad.picks if p.team_id == TeamId(5))
        load = {TeamId(team): [1.0] * 5 for team in range(1, 13)}
        load[TeamId(5)] = [5.0] * 5

        reading = encode_context(
            _context(constraints=ManagerConstraints(locked_players=locked), squad=squad),
            rules=rules,
            club_fixture_load=load,
        ).as_mapping()

        assert reading["fixture_outlook_available"] == 1.0
        assert reading["fixture_locked_load"] == pytest.approx(5.0)
        assert reading["fixture_load_gap"] > 0.0


class TestBeliefBlock:
    def test_no_beliefs_is_all_zero(self, rules: SquadRules) -> None:
        assert np.array_equal(
            encode_context(_context(squad=_squad()), rules=rules).block("belief"),
            np.zeros(3),
        )

    def test_negative_belief_encodes_a_negative_mean(self, rules: SquadRules) -> None:
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=100,
            proposition=BeliefProposition.WILL_NOT_START,
            confidence=0.8,
        )
        reading = encode_context(
            _context(squad=_squad(), beliefs=(belief,)), rules=rules
        ).as_mapping()
        assert reading["belief_mean_signed_confidence"] == pytest.approx(-0.8)
        assert reading["belief_squad_share"] > 0.0

    def test_belief_about_a_player_outside_the_squad_touches_nothing(
        self, rules: SquadRules
    ) -> None:
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=99999,
            proposition=BeliefProposition.WILL_RETURN,
            confidence=0.5,
        )
        reading = encode_context(
            _context(squad=_squad(), beliefs=(belief,)), rules=rules
        ).as_mapping()
        assert reading["belief_squad_share"] == pytest.approx(0.0)
        assert reading["belief_count"] > 0.0


class TestDeterminism:
    def test_encoding_is_reproducible(self, rules: SquadRules) -> None:
        """'Every prediction must be reproducible' starts with its inputs."""
        context = _context(
            squad=_squad(),
            constraints=ManagerConstraints(locked_players=(PlayerCode(100),)),
        )
        first = encode_context(context, rules=rules)
        second = encode_context(context, rules=rules)
        assert np.array_equal(first.values, second.values)
        assert first.fingerprint() == second.fingerprint()

    def test_fingerprint_separates_different_situations(self, rules: SquadRules) -> None:
        squad = _squad()
        free = encode_context(_context(squad=squad), rules=rules)
        frozen = encode_context(
            _context(
                constraints=ManagerConstraints(
                    locked_players=tuple(p.player_code for p in squad.picks)
                ),
                squad=squad,
            ),
            rules=rules,
        )
        assert free.fingerprint() != frozen.fingerprint()
