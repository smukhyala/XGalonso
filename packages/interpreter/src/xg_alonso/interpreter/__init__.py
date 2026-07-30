"""Language-model reading of a manager's request, resolved against real players.

Separate from :mod:`xg_alonso.domain` because it makes a network call and domain
is meant to be pure, and separate from :mod:`xg_alonso.discovery` because that
package's core is contractually forbidden from knowing about football. This one
is football-specific and calls out, so it gets its own layer and its own
optional dependency.

The deterministic parser in ``domain.squad_requests`` runs first and always.
This is the second pass, for intent a vocabulary cannot key on.
"""

from xg_alonso.interpreter.requests import (
    DEFAULT_MODEL,
    Interpretation,
    InterpretedRequest,
    InterpreterUnavailableError,
    ProposedRequirement,
    api_key_origin,
    interpret_request,
    load_api_key,
)

__all__ = [
    "DEFAULT_MODEL",
    "Interpretation",
    "InterpretedRequest",
    "InterpreterUnavailableError",
    "ProposedRequirement",
    "api_key_origin",
    "interpret_request",
    "load_api_key",
]
