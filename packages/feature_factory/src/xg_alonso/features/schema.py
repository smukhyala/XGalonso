"""The model's feature set, defined once, and hashed so a change is detectable.

Two jobs.

**One definition.** ``prediction/dataset.py`` composed
``feature_names() + OPPONENT_FEATURES + CAREER_FEATURES + RECENCY_FEATURES``
inline, and ``features/assemble.py`` built exactly those four families in
exactly that order. Two places agreeing by convention is one refactor away from
two places disagreeing silently, and the symptom would be a model fitted on
columns the inference path does not build.

**A hash that moves when the meaning moves.** Hashing the *names* would miss
the case that matters most: changing ``prior_strength`` from 3.0 to 4.0 alters
every shrunk rate's values while altering no name at all. So whole
:class:`FeatureSpec` records are hashed, in declaration order, and a
reordering changes the digest even though the set does not.

**The honest gap.** Opponent, career and recency features come from
hand-written functions rather than declarative specs, so only their names and a
module version constant are hashable. Editing the arithmetic inside
``build_career_features`` without bumping ``CAREER_FEATURES_VERSION`` will not
change the hash. That is a real limitation, it is why a catalogue-hash mismatch
is a warning rather than a refusal, and it is the reason those version
constants exist at all.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Final

from xg_alonso.features.career import CAREER_FEATURES, CAREER_VERSION
from xg_alonso.features.catalogue import CATALOGUE_VERSION, FeatureSpec, catalogue_specs
from xg_alonso.features.opponent import OPPONENT_FEATURES
from xg_alonso.features.recency import RECENCY_FEATURES

__all__ = [
    "CATALOGUE_HASH_VERSION",
    "FAMILY_VERSIONS",
    "OPPONENT_FEATURES_VERSION",
    "RECENCY_FEATURES_VERSION",
    "catalogue_hash",
    "model_feature_names",
]

CATALOGUE_HASH_VERSION: Final[str] = "catalogue_hash_v1"

#: Bump when the *computation* changes, not merely when a name does. Nothing
#: enforces this — it is the same contract `CATALOGUE_VERSION` already carries,
#: and the same one it can be broken by.
OPPONENT_FEATURES_VERSION: Final[str] = "opponent_v1"
RECENCY_FEATURES_VERSION: Final[str] = "recency_v1"

FAMILY_VERSIONS: Final[tuple[tuple[str, str], ...]] = (
    ("catalogue", CATALOGUE_VERSION),
    ("opponent", OPPONENT_FEATURES_VERSION),
    ("career", CAREER_VERSION),
    ("recency", RECENCY_FEATURES_VERSION),
)


def model_feature_names() -> tuple[str, ...]:
    """The ordered trained-model feature set. The single definition."""
    return (
        tuple(feature_names_of_catalogue()) + OPPONENT_FEATURES + CAREER_FEATURES + RECENCY_FEATURES
    )


def feature_names_of_catalogue() -> list[str]:
    """Every catalogue feature name, in declaration order."""
    return [spec.name for spec in catalogue_specs()]


def _spec_record(spec: FeatureSpec) -> dict[str, object]:
    """One spec as hashable data.

    ``prior_strength`` goes through ``repr`` rather than being serialised as a
    float, because JSON float formatting is not stable enough to hash across
    interpreters and a digest that drifts is worse than none.
    """
    return {
        "name": spec.name,
        "generator": spec.generator,
        "source_column": spec.source_column,
        "window": spec.window,
        "aggregation": spec.aggregation,
        "denominator": spec.denominator,
        "prior_strength": repr(spec.prior_strength),
        "min_periods": spec.min_periods,
        "family": spec.family,
    }


def catalogue_hash(
    specs: Sequence[FeatureSpec] | None = None, *, extension: Sequence[str] = ()
) -> str:
    """A digest of the feature set's *definition*, in declaration order.

    Args:
        specs: Defaults to the whole catalogue.
        extension: Discovered features appended beyond the catalogue, in the
            order they were appended.

    The ``specs`` value is a JSON array, so ``sort_keys`` orders the keys
    *inside* each record without reordering the list. Reordering the metric
    grids therefore changes the digest, which is correct: it changes the column
    order every artifact will be selected by.
    """
    payload = {
        "manifest_version": CATALOGUE_HASH_VERSION,
        "families": [list(pair) for pair in FAMILY_VERSIONS],
        "specs": [_spec_record(s) for s in (specs if specs is not None else catalogue_specs())],
        "opponent": list(OPPONENT_FEATURES),
        "career": list(CAREER_FEATURES),
        "recency": list(RECENCY_FEATURES),
        "extension": list(extension),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
