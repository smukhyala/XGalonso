"""Objective-conditioned feature discovery.

The Feature Factory generates candidates. The Feature Scientist — this package —
decides which of them earn their place, and does so *conditional on what the
manager is trying to achieve*.

That conditioning is the whole idea. Traditional automated feature engineering
asks which features maximise predictive accuracy, and there is exactly one answer.
A manager forty points behind in a mini-league and a manager protecting a rank
need different answers, because they are making different decisions, and a
feature that improves an average while flattening the tail actively harms the
first one.

The loop, and the module that owns each step:

===========================================  =========================
step                                          module
===========================================  =========================
encode the objective                          ``contracts.objective``
generate a falsifiable hypothesis             ``hypotheses``
compile it to a safe program                  ``dsl``
validate statically, then for leakage         ``compile``
compute it historically                       ``compile``
backtest walk-forward                         ``experiment``
score it under the objective                  ``utility``
accept, reject or revise                      ``acceptance``
remember the result                           ``registry``, ``memory``
===========================================  =========================

**Nothing here executes generated code.** A proposal is an expression tree
(:mod:`~xg_alonso.discovery.dsl`), never Python source, so the worst a bad
proposal can do is fail validation.

**The core is domain-free.** ``dsl``, ``compile``, ``utility``, ``acceptance``,
``search`` and ``registry`` know about columns, windows and folds — not about
football. ``.importlinter`` enforces it, so the engine is reusable outside FPL
rather than merely claimed to be. The football lives in ``residuals``,
``hypotheses`` and ``experiment``.
"""

from xg_alonso.discovery.dsl import (
    MAX_DEPTH,
    MAX_NODES,
    MAX_WINDOW,
    Arith,
    ArithOp,
    ClusterRel,
    ClusterRelOp,
    Const,
    EwmMean,
    FeatureProgram,
    GroupKey,
    GroupRel,
    GroupRelOp,
    Lag,
    Level,
    ProgramError,
    Rolling,
    RollingAgg,
    ShrunkRate,
    Source,
    SourceScope,
    TimeSince,
    Trend,
    Unary,
    UnaryOp,
    parse_program,
    program_hash,
)

__all__ = [
    "MAX_DEPTH",
    "MAX_NODES",
    "MAX_WINDOW",
    "Arith",
    "ArithOp",
    "ClusterRel",
    "ClusterRelOp",
    "Const",
    "EwmMean",
    "FeatureProgram",
    "GroupKey",
    "GroupRel",
    "GroupRelOp",
    "Lag",
    "Level",
    "ProgramError",
    "Rolling",
    "RollingAgg",
    "ShrunkRate",
    "Source",
    "SourceScope",
    "TimeSince",
    "Trend",
    "Unary",
    "UnaryOp",
    "parse_program",
    "program_hash",
]
