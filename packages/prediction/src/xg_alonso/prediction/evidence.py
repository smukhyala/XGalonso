"""Attach explanatory feature evidence to a batch of predictions.

Percentiles are the reason this is a batch operation rather than a per-player
one. A value only becomes evidence when it is ranked against the players it is
being compared to, and that ranking is a property of the whole population being
predicted — so it cannot be computed inside a loop over individual rows without
either recomputing it every time or quietly ranking against whatever subset
happened to be in scope.

Ranking is **within position**. A defender with 0.15 expected goals per 90 is
exceptional; a striker with the same rate is not, and a single league-wide
percentile would flatten that distinction into a number that misleads in both
directions.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from xg_alonso.contracts.evidence import (
    EVIDENCE_PANEL_VERSION,
    EXPLANATORY_PANEL,
    FeatureEvidence,
    FeatureValue,
    PanelEntry,
)
from xg_alonso.contracts.prediction import PlayerPrediction

__all__ = ["attach_feature_evidence", "build_feature_evidence"]

#: A position needs at least this many ranked players before a percentile means
#: anything. Below it the rank is reported as unknown rather than as a confident
#: position within a population of three.
_MIN_POPULATION: int = 8


def _percentiles(values: list[float | None]) -> list[float | None]:
    """Fractional rank of each non-null value among the non-null values.

    Ties share the midpoint of the range they span, so two identical values
    never rank differently. Computed with the plain average-rank definition
    rather than a library call because the semantics matter more here than the
    speed: a percentile that treats ties inconsistently would let two players
    with the same xG be described differently.
    """
    present = [(index, value) for index, value in enumerate(values) if value is not None]
    if len(present) < _MIN_POPULATION:
        return [None] * len(values)

    present.sort(key=lambda pair: pair[1])
    total = len(present)
    ranks: list[float | None] = [None] * len(values)

    position = 0
    while position < total:
        end = position
        while end + 1 < total and present[end + 1][1] == present[position][1]:
            end += 1
        # Average rank across the tied block, mapped into [0, 1].
        average_rank = (position + end) / 2.0
        percentile = average_rank / (total - 1) if total > 1 else 0.5
        for tied in range(position, end + 1):
            ranks[present[tied][0]] = percentile
        position = end + 1

    return ranks


def _column_values(frame: pl.DataFrame, entry: PanelEntry) -> list[float | None]:
    """A panel entry's values, from the first source column the frame carries.

    A missing column is not an error. The two feature sets spell the same
    concept differently and the baseline is much smaller than the catalogue, so
    an entry nothing supplies should read as "not measured" rather than break
    prediction — and one the baseline spells its own way should still resolve.
    """
    for name in entry.columns():
        if name in frame.columns:
            series = frame[name].cast(pl.Float64, strict=False)
            return [None if value is None else float(value) for value in series.to_list()]
    return [None] * frame.height


def build_feature_evidence(
    frame: pl.DataFrame,
    *,
    positions: Sequence[str],
    panel: Sequence[PanelEntry] = EXPLANATORY_PANEL,
    panel_version: str = EVIDENCE_PANEL_VERSION,
) -> list[FeatureEvidence]:
    """Materialise the panel for every row of a feature frame.

    Args:
        frame: The frame the model predicted on. Rows are positional; the
            returned list is aligned to it.
        positions: Each row's position, used to scope percentile ranking.
        panel: Which features to carry. Defaults to the declared panel.
        panel_version: Recorded on each result so a stored prediction can be
            checked against the panel that produced it.

    Returns:
        One :class:`FeatureEvidence` per row, in the frame's order.
    """
    if frame.height != len(positions):
        raise ValueError(
            f"positions has {len(positions)} entries for a frame of {frame.height} rows; "
            "percentiles would be scoped to the wrong players"
        )

    raw = {entry.name: _column_values(frame, entry) for entry in panel}

    # Rank inside each position separately, then scatter back into row order.
    by_position: dict[str, list[int]] = {}
    for index, position in enumerate(positions):
        by_position.setdefault(position, []).append(index)

    ranked: dict[str, list[float | None]] = {entry.name: [None] * frame.height for entry in panel}
    for indices in by_position.values():
        for entry in panel:
            column = raw[entry.name]
            group_percentiles = _percentiles([column[i] for i in indices])
            for slot, index in enumerate(indices):
                ranked[entry.name][index] = group_percentiles[slot]

    results: list[FeatureEvidence] = []
    for index in range(frame.height):
        values = tuple(
            FeatureValue(
                name=entry.name,
                label=entry.label,
                family=entry.family,
                value=raw[entry.name][index],
                percentile=ranked[entry.name][index],
                higher_is_better=entry.higher_is_better,
            )
            for entry in panel
        )
        results.append(FeatureEvidence(panel_version=panel_version, values=values))
    return results


def attach_feature_evidence(
    predictions: Sequence[PlayerPrediction],
    evidence: Sequence[FeatureEvidence],
) -> list[PlayerPrediction]:
    """Return predictions carrying their evidence.

    Kept separate from :func:`build_feature_evidence` because prediction skips
    rows whose position is unrecognised, so the two sequences are aligned by the
    caller rather than assumed to be the same length.
    """
    if len(predictions) != len(evidence):
        raise ValueError(
            f"{len(predictions)} predictions against {len(evidence)} evidence entries; "
            "an off-by-one here would attribute one player's features to another"
        )
    return [
        prediction.model_copy(update={"feature_evidence": found})
        for prediction, found in zip(predictions, evidence, strict=True)
    ]
