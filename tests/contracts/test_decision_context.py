"""Decision-context identity, bucketing and the objective/constraint firewall.

Three of these tests are not ordinary unit tests but executable statements of
design claims made elsewhere in prose:

- :class:`TestFeasibilityDigest` checks the claim that the digest covers *only*
  what changes which players are reachable. That claim is what lets a pool
  computation be shared between two managers, so a silent widening of the field
  set would degrade to "no sharing" invisibly.
- :class:`TestBoundedCardinality` checks the claim that context-conditioning does
  not fragment the artifact space combinatorially. It is asserted against
  randomly generated constraint sets rather than against the arithmetic, because
  the arithmetic is exactly what a future edit would invalidate.
- :class:`TestIdentityIsNotAFeature` checks that player identity cannot leak into
  the bucket. The whole conditioning story rests on structural encoding
  generalising to managers who never owned a given player.
"""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from xg_alonso.contracts.context import (
    CONTEXT_VERSION,
    BeliefLoad,
    BudgetBand,
    ClubPressure,
    ContextBucket,
    DecisionContext,
    HitAppetite,
    LockPressure,
    LockShape,
    TransferFreedom,
)
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
    PrimaryMetric,
    Requirement,
    RequirementKind,
    SquadArea,
    SquadRequirements,
    TeamQuota,
    UserBelief,
)
from xg_alonso.contracts.prediction import Position
from xg_alonso.contracts.squad import SquadPick, SquadState

_LAYOUT = ((Position.GKP, 2), (Position.DEF, 5), (Position.MID, 5), (Position.FWD, 3))
_LEGAL_XI_ORDER = (0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 1, 6, 11, 14)


def _squad(
    *,
    bank: int = 10,
    free_transfers: int = 1,
    clubs_per_player: int = 12,
    first_code: int = 100,
) -> SquadState:
    """A legal fifteen with a 1-4-4-2 starting shape.

    ``clubs_per_player`` controls club concentration so the club-pressure band
    can be exercised: a small modulus forces several players onto one club.
    """
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
                    purchase_price=TenthsOfMillion(50),
                    current_price=TenthsOfMillion(50),
                    selling_price=TenthsOfMillion(50),
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


def _objective(**overrides: object) -> ManagerObjective:
    base: dict[str, object] = {
        "id": "expected_points_balanced_h1_neutral",
        "name": "Maximise expected points",
        "primary_metric": PrimaryMetric.EXPECTED_POINTS,
    }
    base.update(overrides)
    return ManagerObjective(**base)  # type: ignore[arg-type]


def _context(
    *,
    constraints: ManagerConstraints | None = None,
    requirements: SquadRequirements | None = None,
    squad: SquadState | None = None,
    beliefs: tuple[UserBelief, ...] = (),
    objective: ManagerObjective | None = None,
) -> DecisionContext:
    bundle = ObjectiveBundle(
        objective=objective or _objective(),
        constraints=constraints or ManagerConstraints(),
        beliefs=beliefs,
    )
    return DecisionContext(
        bundle=bundle,
        requirements=requirements or SquadRequirements(),
        squad=squad,
        as_of_gameweek=GameweekId(8),
    )


class TestReadThroughAccessors:
    """The three kinds of intent stay separately reachable, never merged."""

    def test_objective_constraints_and_beliefs_are_distinct_members(self) -> None:
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=100,
            proposition=BeliefProposition.WILL_RETURN,
            confidence=0.6,
        )
        constraints = ManagerConstraints(locked_players=(PlayerCode(100),))
        context = _context(constraints=constraints, beliefs=(belief,))

        assert context.objective.primary_metric is PrimaryMetric.EXPECTED_POINTS
        assert context.constraints.locked_players == (PlayerCode(100),)
        assert context.beliefs == (belief,)

    def test_context_is_frozen_and_rejects_unknown_fields(self) -> None:
        context = _context()
        with pytest.raises(ValidationError):
            context.bundle = ObjectiveBundle(objective=_objective())
        with pytest.raises(ValidationError):
            DecisionContext(
                bundle=ObjectiveBundle(objective=_objective()),
                nonsense=1,  # type: ignore[call-arg]
            )


class TestLockResolution:
    """Locks are stated four ways; a caller must not have to know which."""

    def test_locked_positions_expand_against_the_squad(self) -> None:
        squad = _squad()
        context = _context(
            constraints=ManagerConstraints(locked_positions=(Position.GKP,)),
            squad=squad,
        )
        keepers = {p.player_code for p in squad.picks if p.position is Position.GKP}
        assert context.locked_codes() == keepers

    def test_protected_areas_expand_against_the_squad(self) -> None:
        squad = _squad()
        context = _context(
            constraints=ManagerConstraints(protected_squad_areas=(SquadArea.ATTACK,)),
            squad=squad,
        )
        forwards = {p.player_code for p in squad.picks if p.position is Position.FWD}
        assert context.locked_codes() == forwards

    def test_explicit_locks_and_area_locks_union(self) -> None:
        squad = _squad()
        a_defender = next(p for p in squad.picks if p.position is Position.DEF)
        context = _context(
            constraints=ManagerConstraints(
                locked_players=(a_defender.player_code,),
                protected_squad_areas=(SquadArea.ATTACK,),
            ),
            squad=squad,
        )
        expected = {p.player_code for p in squad.picks if p.position is Position.FWD}
        expected.add(a_defender.player_code)
        assert context.locked_codes() == expected

    def test_without_a_squad_only_explicit_locks_resolve(self) -> None:
        """A from-scratch build has no squad, so an area lock names nobody yet.

        Returning the empty set here rather than raising is deliberate: building
        from scratch with a protected area declared is a coherent request, and
        the area still binds once a squad exists.
        """
        context = _context(
            constraints=ManagerConstraints(protected_squad_areas=(SquadArea.ATTACK,))
        )
        assert context.locked_codes() == frozenset()


class TestFeasibilityDigest:
    """Covers exactly what changes the reachable pool, and nothing else."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_points_hit", 12),
            ("max_transfers", 3),
            ("required_features", ("expected_goals_per90_5",)),
            ("excluded_features", ("selected_mean_5",)),
        ],
    )
    def test_invariant_to_fields_that_do_not_change_reachability(
        self, field: str, value: object
    ) -> None:
        """These change how many moves or which columns — never which players."""
        squad = _squad()
        base = _context(squad=squad)
        varied = _context(
            constraints=ManagerConstraints(**{field: value}),  # type: ignore[arg-type]
            squad=squad,
        )
        assert base.feasibility_digest() == varied.feasibility_digest()

    def test_invariant_to_beliefs(self) -> None:
        squad = _squad()
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=100,
            proposition=BeliefProposition.WILL_RETURN,
            confidence=0.9,
        )
        assert (
            _context(squad=squad).feasibility_digest()
            == _context(squad=squad, beliefs=(belief,)).feasibility_digest()
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("locked_players", (PlayerCode(100),)),
            ("excluded_players", (PlayerCode(900),)),
            ("minimum_bank", TenthsOfMillion(20)),
            ("maximum_budget", TenthsOfMillion(950)),
            ("excluded_teams", (TeamId(3),)),
            ("required_teams", (TeamId(4),)),
            ("maximum_players_by_team", (TeamQuota(team_id=TeamId(2), count=1),)),
        ],
    )
    def test_sensitive_to_fields_that_do_change_reachability(
        self, field: str, value: object
    ) -> None:
        squad = _squad()
        base = _context(squad=squad)
        varied = _context(
            constraints=ManagerConstraints(**{field: value}),  # type: ignore[arg-type]
            squad=squad,
        )
        assert base.feasibility_digest() != varied.feasibility_digest()

    def test_sensitive_to_formation(self) -> None:
        squad = _squad()
        shaped = SquadRequirements(
            requirements=(
                Requirement(kind=RequirementKind.FORMATION, label="play 3-5-2", formation="3-5-2"),
            )
        )
        assert (
            _context(squad=squad).feasibility_digest()
            != _context(squad=squad, requirements=shaped).feasibility_digest()
        )

    def test_sensitive_to_the_bank(self) -> None:
        """Budget headroom decides affordability, so it must move the digest."""
        assert (
            _context(squad=_squad(bank=1)).feasibility_digest()
            != _context(squad=_squad(bank=60)).feasibility_digest()
        )

    def test_stable_across_rebuilds(self) -> None:
        squad = _squad()
        assert (
            _context(squad=squad).feasibility_digest() == _context(squad=squad).feasibility_digest()
        )


class TestFingerprintReproducibility:
    """A provenance fingerprint that moves with the clock records nothing."""

    def test_two_identical_contexts_fingerprint_alike(self) -> None:
        """``ManagerObjective`` and ``UserBelief`` both carry ``created_at``.

        Built a microsecond apart they differ in a field that affects no result,
        so the fingerprint must ignore it or it cannot verify a faithful re-run.
        """
        belief_kwargs = {
            "entity_type": BeliefEntity.PLAYER,
            "entity_id": 100,
            "proposition": BeliefProposition.WILL_RETURN,
            "confidence": 0.5,
        }
        first = _context(
            objective=_objective(),
            beliefs=(UserBelief(**belief_kwargs),),  # type: ignore[arg-type]
            squad=_squad(),
        )
        second = _context(
            objective=_objective(),
            beliefs=(UserBelief(**belief_kwargs),),  # type: ignore[arg-type]
            squad=_squad(),
        )
        assert first.context_fingerprint() == second.context_fingerprint()

    def test_fingerprint_still_separates_real_differences(self) -> None:
        """Stripping timestamps must not blunt the fingerprint."""
        squad = _squad()
        assert (
            _context(squad=squad).context_fingerprint()
            != _context(
                squad=squad,
                constraints=ManagerConstraints(locked_players=(PlayerCode(100),)),
            ).context_fingerprint()
        )

    def test_fingerprint_is_finer_than_the_digest(self) -> None:
        """Hit appetite is invisible to the pool but belongs in provenance."""
        squad = _squad()
        base = _context(squad=squad)
        hits = _context(constraints=ManagerConstraints(max_points_hit=8), squad=squad)
        assert base.feasibility_digest() == hits.feasibility_digest()
        assert base.context_fingerprint() != hits.context_fingerprint()


class TestBucketBands:
    def test_lock_pressure_rises_as_the_squad_freezes(self) -> None:
        squad = _squad()
        codes = [p.player_code for p in squad.picks]
        pressures = [
            _context(constraints=ManagerConstraints(locked_players=tuple(codes[:n])), squad=squad)
            .bucket()
            .lock_pressure
            for n in (0, 5, 10, 14)
        ]
        assert pressures == [
            LockPressure.FREE,
            LockPressure.LIGHT,
            LockPressure.HEAVY,
            LockPressure.FROZEN,
        ]

    def test_lock_pressure_unknown_without_a_squad(self) -> None:
        assert _context().bucket().lock_pressure is LockPressure.UNKNOWN

    @pytest.mark.parametrize(
        ("bank", "band"),
        [
            (2, BudgetBand.BROKE),
            (10, BudgetBand.THIN),
            (30, BudgetBand.COMFORTABLE),
            (80, BudgetBand.RICH),
        ],
    )
    def test_budget_bands(self, bank: int, band: BudgetBand) -> None:
        assert _context(squad=_squad(bank=bank)).bucket().budget_band is band

    def test_lock_shape_is_positional_not_identity(self) -> None:
        squad = _squad()
        forwards = tuple(p.player_code for p in squad.picks if p.position is Position.FWD)
        context = _context(constraints=ManagerConstraints(locked_players=forwards), squad=squad)
        assert context.bucket().lock_shape is LockShape.FWD

    def test_lock_shape_mixed_across_positions(self) -> None:
        squad = _squad()
        mixed = (
            next(p.player_code for p in squad.picks if p.position is Position.DEF),
            next(p.player_code for p in squad.picks if p.position is Position.FWD),
        )
        context = _context(constraints=ManagerConstraints(locked_players=mixed), squad=squad)
        assert context.bucket().lock_shape is LockShape.MIXED

    def test_transfer_freedom_takes_the_tighter_of_squad_and_cap(self) -> None:
        """A manager with five free transfers who asked for one has one."""
        squad = _squad(free_transfers=5)
        assert _context(squad=squad).bucket().transfer_freedom is TransferFreedom.MANY
        capped = _context(constraints=ManagerConstraints(max_transfers=1), squad=squad)
        assert capped.bucket().transfer_freedom is TransferFreedom.ONE

    def test_club_pressure_detects_concentration(self) -> None:
        assert _context(squad=_squad(clubs_per_player=20)).bucket().club_pressure is (
            ClubPressure.SLACK
        )
        assert _context(squad=_squad(clubs_per_player=5)).bucket().club_pressure is (
            ClubPressure.TIGHT
        )

    def test_belief_load(self) -> None:
        belief = UserBelief(
            entity_type=BeliefEntity.PLAYER,
            entity_id=100,
            proposition=BeliefProposition.WILL_START,
            confidence=0.4,
        )
        assert _context().bucket().belief_load is BeliefLoad.NONE
        assert _context(beliefs=(belief,)).bucket().belief_load is BeliefLoad.SOME


class TestContextKey:
    def test_carries_objective_and_version(self) -> None:
        key = _context(squad=_squad()).context_key()
        assert key.startswith("expected_points_balanced_h1_neutral|")
        assert CONTEXT_VERSION in key

    def test_same_bucket_same_key(self) -> None:
        """Two managers whose situations bucket alike share cached work.

        This is the point of bucketing: different locked players, same shape of
        problem. If this ever fails, the cache has become per-manager.
        """
        squad = _squad()
        forwards = [p.player_code for p in squad.picks if p.position is Position.FWD]
        first = _context(constraints=ManagerConstraints(locked_players=(forwards[0],)), squad=squad)
        second = _context(
            constraints=ManagerConstraints(locked_players=(forwards[1],)), squad=squad
        )
        assert first.feasibility_digest() != second.feasibility_digest()
        assert first.context_key() == second.context_key()

    def test_different_bucket_different_key(self) -> None:
        squad = _squad()
        codes = [p.player_code for p in squad.picks]
        free = _context(squad=squad)
        frozen = _context(
            constraints=ManagerConstraints(locked_players=tuple(codes[:14])), squad=squad
        )
        assert free.context_key() != frozen.context_key()


def _theoretical_cells(objective_ids: int, formations: int) -> int:
    """Upper bound on distinct buckets, derived from the enums themselves.

    Computed rather than written down, so adding an enum member updates the
    bound instead of silently invalidating a hand-typed number.
    """
    return (
        objective_ids
        * len(TransferFreedom)
        * len(HitAppetite)
        * len(LockPressure)
        * len(LockShape)
        * len(ClubPressure)
        * len(BudgetBand)
        * formations
        * len(BeliefLoad)
    )


def _random_contexts(count: int, *, seed: int) -> list[DecisionContext]:
    rng = random.Random(seed)
    codes = [p.player_code for p in _squad().picks]
    contexts: list[DecisionContext] = []
    for _ in range(count):
        squad = _squad(bank=rng.randint(0, 90), free_transfers=rng.randint(1, 5))
        constraints = ManagerConstraints(
            locked_players=tuple(rng.sample(codes, rng.randint(0, 15))),
            max_points_hit=rng.choice([0, 4, 12]),
            max_transfers=rng.choice([None, 1, 2, 5]),
            minimum_bank=TenthsOfMillion(rng.randint(0, 30)),
        )
        contexts.append(_context(constraints=constraints, squad=squad))
    return contexts


class TestBoundedCardinality:
    """Context-conditioning must not fragment the artifact space.

    The claim is *boundedness*, not aggressive collapse. A bucket space with a
    few thousand reachable cells is fine — what would be fatal is a cache that
    grows one entry per manager, which is what keying on the raw constraint set
    would produce.
    """

    def test_keys_compress_against_raw_constraint_sets(self) -> None:
        """Distinct situations, far fewer cache entries.

        Every sampled context has its own feasibility digest — they genuinely
        face different pools. The point is that they *share* representations.
        """
        contexts = _random_contexts(600, seed=20260804)
        digests = {c.feasibility_digest() for c in contexts}
        keys = {c.context_key() for c in contexts}

        assert len(digests) > 500, "sampler produced too few distinct situations to be a test"
        assert len(keys) * 2 < len(digests), (
            f"{len(keys)} keys against {len(digests)} distinct pools is not compression"
        )

    def test_keys_saturate_rather_than_growing_with_sample_size(self) -> None:
        """Quadrupling the sample must not quadruple the cache.

        This is the property that distinguishes a bounded space from a merely
        large one, and it is the one a future edit would break by folding a
        continuous quantity into the bucket.
        """
        small = {c.context_key() for c in _random_contexts(500, seed=1)}
        large = {c.context_key() for c in _random_contexts(2000, seed=1)}
        assert len(large) < 2 * len(small), (
            f"{len(small)} keys at n=500 grew to {len(large)} at n=2000; "
            "the bucket space is not saturating"
        )

    def test_never_exceeds_the_enum_product(self) -> None:
        contexts = _random_contexts(2000, seed=7)
        keys = {c.context_key() for c in contexts}
        # One objective and one formation are exercised by the sampler.
        assert len(keys) <= _theoretical_cells(objective_ids=1, formations=1)

    def test_bucket_key_round_trips(self) -> None:
        bucket = _context(squad=_squad()).bucket()
        assert bucket.key().count("|") == len(ContextBucket.model_fields) - 1


class TestIdentityIsNotAFeature:
    """Relabelling every player must not change how a situation buckets.

    The conditioning story depends on structural encoding: "one of three forward
    slots is frozen" has to generalise to a manager who never owned that player.
    If identity leaked into the bucket, the learned conditioning would memorise
    player codes and transfer to nobody.
    """

    def test_permuting_player_codes_preserves_the_bucket(self) -> None:
        original = _squad(first_code=100)
        relabelled = _squad(first_code=5000)

        forwards_a = tuple(p.player_code for p in original.picks if p.position is Position.FWD)
        forwards_b = tuple(p.player_code for p in relabelled.picks if p.position is Position.FWD)

        first = _context(constraints=ManagerConstraints(locked_players=forwards_a), squad=original)
        second = _context(
            constraints=ManagerConstraints(locked_players=forwards_b), squad=relabelled
        )

        assert first.bucket() == second.bucket()
        assert first.context_key() == second.context_key()
        # ...while the pool identity still separates them, because they really do
        # face different players.
        assert first.feasibility_digest() != second.feasibility_digest()
