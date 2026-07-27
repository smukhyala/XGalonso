"""Converts structured evidence into user-facing reasoning.

An LLM may rewrite a grounded sentence for readability. It may never invent a
cause or a statistic — and here that is a structural guarantee rather than a
policy. A Reason cannot be constructed unless its evidence satisfies every
placeholder in its template, so no renderer is ever handed a gap to fill.
"""

from xg_alonso.explanations.render import render_recommendation, render_squad_summary

__all__ = ["render_recommendation", "render_squad_summary"]
