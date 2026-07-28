"""HTTP surface over the decision system.

The second surface after the CLI, per decision D4. Deliberately thin — the
composition root already wires the packages together, so this layer parses
requests, calls it, and shapes responses. Every response carries provenance.
"""

from xg_alonso.api.service import DecisionService, ServiceConfig

__all__ = ["DecisionService", "ServiceConfig"]
