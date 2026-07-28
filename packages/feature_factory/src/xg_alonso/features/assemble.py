"""The one place the trained-model feature set is assembled.

**Why this exists.** The catalogue, the opponent features and the career
features were joined together independently in five places — the recommend
pipeline, the squad builder, the backtest, the scoring command and the API. Each
was correct when written, and adding career features made two of them wrong,
because a model fitted with a feature it is never given at inference fails
inside `_matrix` with a list of column names and no explanation of which build
path forgot them.

A feature set is a single fact about a model. Assembling it in one function
means a new family cannot be half-adopted: every path gets it, or none compiles.
"""

from __future__ import annotations

import polars as pl

from xg_alonso.features.career import build_career_features
from xg_alonso.features.catalogue import build_catalogue
from xg_alonso.features.opponent import build_opponent_features, build_opponent_strength

__all__ = ["build_model_features"]


def build_model_features(
    entities: pl.DataFrame,
    *,
    player_stats: pl.DataFrame,
    opponent_strength: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Every feature a trained model expects, in one pass.

    Args:
        entities: One row per prediction, carrying ``player_code`` and a cutoff.
        player_stats: Canonical ``player_gameweek_stats``.
        opponent_strength: Precomputed opponent table. Derived when omitted;
            pass it when building several gameweeks so the inversion is not
            repeated per week.

    Returns:
        ``entities`` plus the catalogue, opponent and career features.
    """
    strength = (
        build_opponent_strength(player_stats) if opponent_strength is None else opponent_strength
    )
    features = build_catalogue(entities, player_stats=player_stats)
    features = build_opponent_features(features, opponent_strength=strength)
    return build_career_features(features, player_stats=player_stats)
