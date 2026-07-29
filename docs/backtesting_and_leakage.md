# Backtesting and Leakage

Leakage does not crash anything. It produces a model that validates beautifully
and loses money. Every guard described here exists because the failure it
prevents is silent.

## The rule

> A feature computed for a prediction at time *T* may use a source record only
> when `available_time <= T`.

Not `event_time`. A match that was played is not the same as a match whose
statistics had been published.

## Folds

`xg_alonso.contracts.folds.walk_forward_folds` is the **only** sanctioned fold
constructor and is structurally incapable of shuffling — it validates that the
gameweek sequence is ascending and unique and raises otherwise. Random splits on
temporal data are not discouraged here; they are unrepresentable.

The discovery harness adds three properties on top:

1. **A global timeline across seasons.** Gameweek 3 of 2024-25 must sort *after*
   gameweek 30 of 2023-24. Without the offset a "walk-forward" split trains on
   the future at every season boundary.
2. **An embargo** between training and validation, so a rolling window's lookback
   cannot straddle the boundary.
3. **A holdout.** `HarnessConfig.holdout_seasons` is excluded from every fold, so
   acceptance decisions are never tuned against the period later used to claim
   the system works. Default: `2025-26`.

## Four leaks, four defences

| Leak | Defence |
|---|---|
| A program reads records not yet available | `features.leakage.assert_no_leakage`, run on every program before registration |
| A program reads its own target | `validate_program(forbidden_columns=...)`, a static name check |
| A scaler / projection / cluster is fitted on the rows it scores | **Structural**: `fit_embedding` and `fit_clusters` return immutable models whose stored statistics are the only way to transform new rows. There is no function that scales a frame using its own statistics. |
| A fold's validation overlaps its training | The frozen fold constructor, plus explicit tests |

## The negative control

A leakage harness that never fails is indistinguishable from a broken one, and
the broken version is *more* dangerous than no harness because it manufactures
confidence.

So every positive check is paired with a deliberately leaky twin that must be
caught. `features.leakage.assert_detects_leakage` does this for feature builders;
`tests/discovery/test_leakage.py` does it for cluster fit/apply isolation.

Writing that second control found a real problem. Its first version "drifted" the
data by adding a constant to every column — which, after z-scoring, changes
nothing. Both the positive test and its control would have passed vacuously, one
of them for entirely the wrong reason. The drift now changes distribution
*shape*.

## Controls for a positive result

Beyond leakage, a candidate must beat two controls or it is rejected:

- **Noise control** — the required set plus a matched-complexity random column.
- **Shuffled control** — the required set plus a permuted copy of the candidate,
  preserving its exact scale, skew and missingness while destroying only the
  association with the target.

A gradient-boosted model handed one more column often improves slightly whatever
it contains. Without these controls, "adding this feature helped" is a statement
about model capacity rather than about football.

## What is not claimed

**No statistical significance is claimed anywhere**, because none is tested.
Reports say "improved in 5 of 5 folds" and give the spread. They never say
"significant".
