"""Archetype clustering tests.

Archetype labels end up in user-facing prose, so the properties that matter are
not "does k-means converge" but "does this say something true and stable". Three
things could each make the output actively misleading: labels that change
between runs, two clusters sharing a name, and a cluster fitted on so few
players that it describes nobody.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xg_alonso.features.archetypes import ARCHETYPE_VERSION, STYLE_AXES, build_archetypes


def _population(per_position: int = 60, seed: int = 3) -> tuple[pl.DataFrame, list[str], list[int]]:
    """Two genuinely separated groups per position, plus noise.

    Built with a real separation so the clustering has something to find: half
    of each position is high-output and attacking, half is low-output and
    defensive. A method that cannot recover that split cannot be trusted on
    real players, where the separation is far softer.
    """
    generator = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {}
    positions: list[str] = []
    codes: list[int] = []

    columns = sorted(
        {source for axes in STYLE_AXES.values() for axis in axes for source in axis.sources}
    )
    for column in columns:
        rows[column] = []

    code = 1
    for position in STYLE_AXES:
        for index in range(per_position):
            attacking = index < per_position // 2
            for column in columns:
                centre = 1.6 if attacking else 0.3
                rows[column].append(float(generator.normal(centre, 0.25)))
            positions.append(position)
            codes.append(code)
            code += 1

    return pl.DataFrame(rows), positions, codes


class TestShape:
    def test_every_player_is_placed(self) -> None:
        frame, positions, codes = _population()
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        assert {entry.player_code for entry in model.players} == set(codes)

    def test_a_player_is_placed_in_exactly_one_archetype(self) -> None:
        frame, positions, codes = _population()
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        seen = [entry.player_code for entry in model.players]
        assert len(seen) == len(set(seen))

    def test_archetypes_carry_the_version(self) -> None:
        frame, positions, codes = _population()
        assert build_archetypes(frame, positions=positions, player_codes=codes).version == (
            ARCHETYPE_VERSION
        )

    def test_cluster_sizes_account_for_every_player(self) -> None:
        frame, positions, codes = _population()
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        for position in STYLE_AXES:
            placed = [e for e in model.players if e.position == position]
            declared = sum(a.size for a in model.archetypes if a.position == position)
            assert declared == len(placed)


class TestLabels:
    def test_labels_are_unique_within_a_position(self) -> None:
        """Two clusters sharing a name cannot tell a reader which one he is in.

        The first run produced two goalkeeper archetypes both called
        "high-returning, bonus-magnet GKP".
        """
        frame, positions, codes = _population()
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        for position in STYLE_AXES:
            labels = [a.label for a in model.archetypes if a.position == position]
            assert len(labels) == len(set(labels)), f"duplicate labels in {position}: {labels}"

    def test_every_archetype_has_a_readable_label(self) -> None:
        frame, positions, codes = _population()
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        for archetype in model.archetypes:
            assert archetype.label
            assert archetype.position in archetype.label
            assert archetype.describe()

    def test_a_population_with_no_real_spread_is_not_clustered(self) -> None:
        """Naming a difference that is not there is worse than saying nothing.

        Standardising a near-constant axis rescales noise to unit variance, so
        after scaling it is indistinguishable from a genuinely varied one — and
        the labeller confidently called one cluster "goalscoring" and another
        "non-scoring" over differences of a couple of percent. The spread guard
        is measured on the raw values, before scaling, and here it correctly
        declines to produce archetypes at all.
        """
        generator = np.random.default_rng(1)
        columns = sorted(
            {source for axes in STYLE_AXES.values() for axis in axes for source in axis.sources}
        )
        rows = {c: [float(generator.normal(1.0, 0.005)) for _ in range(60)] for c in columns}
        model = build_archetypes(
            pl.DataFrame(rows),
            positions=["MID"] * 60,
            player_codes=list(range(60)),
        )
        assert model.archetypes == ()


class TestStability:
    def test_the_same_input_gives_the_same_archetypes(self) -> None:
        """Labels are shown to users; a reshuffle between runs would rename them."""
        frame, positions, codes = _population()
        first = build_archetypes(frame, positions=positions, player_codes=codes)
        second = build_archetypes(frame, positions=positions, player_codes=codes)
        assert [a.label for a in first.archetypes] == [a.label for a in second.archetypes]
        assert [(p.player_code, p.archetype) for p in first.players] == [
            (p.player_code, p.archetype) for p in second.players
        ]


class TestComparables:
    def test_comps_never_include_the_player_himself(self) -> None:
        frame, positions, codes = _population()
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        for entry in model.players:
            assert entry.player_code not in entry.comparables

    def test_comps_stay_within_position(self) -> None:
        """A centre-back compared to a striker is not a comparison."""
        frame, positions, codes = _population()
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        position_of = dict(zip(codes, positions, strict=True))
        for entry in model.players:
            for other in entry.comparables:
                assert position_of[other] == entry.position

    def test_comps_are_actually_the_nearest(self) -> None:
        """A player from the opposite group should not be anybody's closest comp."""
        frame, positions, codes = _population()
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        # Codes were laid out attacking-first within each position block, so a
        # low-index player in a block should draw comps from his own half.
        mids = [e for e in model.players if e.position == "MID"]
        attacking = {e.player_code for e in mids[: len(mids) // 2]}
        matched = sum(
            1
            for entry in mids
            if entry.player_code in attacking
            and entry.comparables
            and entry.comparables[0] in attacking
        )
        assert matched / max(len(attacking), 1) > 0.8


class TestGuards:
    def test_a_position_with_too_few_players_is_skipped(self) -> None:
        """An archetype fitted on nine players describes those nine and nobody else."""
        frame, positions, codes = _population(per_position=6)
        model = build_archetypes(frame, positions=positions, player_codes=codes)
        assert model.archetypes == ()
        assert model.players == ()

    def test_mismatched_inputs_are_refused(self) -> None:
        frame, positions, codes = _population()
        with pytest.raises(ValueError, match="same length"):
            build_archetypes(frame, positions=positions[:-1], player_codes=codes)

    def test_a_frame_without_style_columns_produces_nothing(self) -> None:
        """Better to say nothing than to cluster on one axis and call it a style."""
        model = build_archetypes(
            pl.DataFrame({"unrelated": [1.0] * 60}),
            positions=["DEF"] * 60,
            player_codes=list(range(60)),
        )
        assert model.archetypes == ()
