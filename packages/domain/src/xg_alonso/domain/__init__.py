"""Pure FPL and football rules — no database, no API, no dataframe engine.

The ``domain-purity`` contract in ``.importlinter`` enforces that emptiness of
dependencies, which is what lets every rule here be tested in microseconds with
no fixtures beyond a pinned JSON payload.

Two principles run through the package:

1. **Constants are read, never transcribed.** :class:`ScoringRules` and
   :class:`SquadRules` parse a pinned ``bootstrap-static`` snapshot. The few
   values FPL does not publish live in :class:`ScoringThresholds`, marked
   ``VERIFY``, so the line between vendor-supplied and self-asserted is visible.
2. **Points are assembled, never predicted directly.** :func:`assemble_points`
   is the D8 boundary: models emit component counts, and only this layer knows
   what a component is worth.
"""

from xg_alonso.domain.constraints import (
    SquadViolation,
    check_squad,
    check_starting_xi,
    is_legal_squad,
)
from xg_alonso.domain.pricing import selling_price, squad_value
from xg_alonso.domain.purchase_prices import (
    PurchasePriceResult,
    TransferRecord,
    parse_transfer_log,
    reconstruct_purchase_prices,
)
from xg_alonso.domain.realisation import (
    RealisedCounts,
    realised_points,
    realised_points_matrix,
)
from xg_alonso.domain.rules import PositionRule, SquadRules
from xg_alonso.domain.scoring import ScoringRules, ScoringThresholds, assemble_points

__all__ = [
    "PositionRule",
    "PurchasePriceResult",
    "RealisedCounts",
    "ScoringRules",
    "ScoringThresholds",
    "SquadRules",
    "SquadViolation",
    "TransferRecord",
    "assemble_points",
    "check_squad",
    "check_starting_xi",
    "is_legal_squad",
    "parse_transfer_log",
    "realised_points",
    "realised_points_matrix",
    "reconstruct_purchase_prices",
    "selling_price",
    "squad_value",
]
