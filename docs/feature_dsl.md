# The Feature Program DSL

A discovered feature is a **frozen expression tree**, not Python source.
`xg_alonso.discovery.dsl` defines it; `xg_alonso.discovery.compile` turns it into
a Polars plan.

**Nothing in this system ever calls `eval` or `exec`.** A generator — whether the
residual-driven search that ships, or a language model supplied later — can only
emit a tree. The worst a bad proposal can do is fail validation.

---

## The two levels

A feature is computed for a *prediction row* — one player at one deadline — from
many *history rows*. Those are different populations, and mixing them silently
produces a column that looks computed and is not.

**`ROW`** — an expression over one historical record. `minutes`,
`log1p(bps)`. One value per historical match.

**`ENTITY`** — an expression over one prediction row. Produced by aggregating a
`ROW` expression across the lookback window, or read from a fixture attribute
like `is_home`. One value per player.

The rule is mechanical:

> A temporal node consumes `ROW` and produces `ENTITY`. Arithmetic may combine
> two expressions only at the same level. A program's root must be `ENTITY`.

```python
# Rejected: one value per match times one value per player.
Arith(op=MUL, left=Rolling(child=Source("minutes"), window=5), right=Source("minutes"))
# ProgramError: cannot combine a entity-level left operand with a row-level right one
```

Constants are level-polymorphic, so `rolling_mean(x) * 2` is fine.

---

## Grammar

```
Program := Source(column, scope)              # a raw column
         | Const(value)
         | Rolling(child, window, agg, min_periods, quantile?)
         | Lag(child, periods)
         | EwmMean(child, window, halflife)
         | Trend(child, window)                # OLS slope, oldest to newest
         | ShrunkRate(numerator, denominator, window, prior_strength, scale)
         | TimeSince(event_column, require_positive)
         | Arith(op, left, right, epsilon)
         | Unary(op, child, lower?, upper?)
         | GroupRel(op, by, child)
         | ClusterRel(op, child, cluster_model_version, cluster_id?)
```

| Family | Members |
|---|---|
| Temporal | `Rolling` (mean, median, std, min, max, sum, count, percentile), `Lag`, `EwmMean`, `Trend`, `ShrunkRate`, `TimeSince` |
| Arithmetic | `add`, `sub`, `mul`, `safe_div`, `min`, `max` |
| Unary | `log1p`, `neg`, `abs`, `clip`, `zscore`, `percentile_rank` |
| Group | `rank`, `share`, `dev_from_mean`, `zscore` — within position, team, opponent or all |
| Cluster | `rank`, `dev_from_centroid`, `gate` |

### Interactions

There is no separate "interaction primitive" because there does not need to be:
an interaction is `Arith(MUL, ...)` over two entity-level expressions. Every
interaction the brief lists is expressible.

```python
# player metric x opponent metric
Arith(
    op=MUL,
    left=ShrunkRate("expected_goals", "minutes", window=5),
    right=Source("opponent_conceded_xg_mean_5", scope=ENTITY),
)

# minutes probability x per-90 output
Arith(
    op=MUL,
    left=ShrunkRate("expected_goals", "minutes", window=5),
    right=Rolling(child=Source("minutes"), window=5),
)
```

### Gating

`ClusterRel(op=GATE, cluster_id=k)` multiplies by the **soft membership
probability** of cluster *k*, not by a hard label. A player sitting between two
clusters contributes to both in proportion, which is what a boundary means.

---

## What cannot be expressed

Absent by construction, not by convention:

- **No division.** Only `safe_div`, which returns **null** when
  `|denominator| <= epsilon`. A ratio over a vanishing denominator is undefined,
  not enormous — and the tempting alternative, `left / (|right| + eps)`, turns a
  12-minute cameo into an elite per-90 rate, which is exactly the failure
  `shrunk_rate_as_of` exists to prevent.
- **No forward offset.** No node reads ahead. `Lag(periods=1)` is the previous
  appearance; there is no lead.
- **No raw column escape hatch.** Every leaf is a declared `Source` checked
  against the real schema.

---

## Static validation

`validate_program` runs before anything is computed and returns **every** issue,
not the first — a generator revising a proposal needs the whole list.

| Code | Rejects |
|---|---|
| `unknown_column` | a source the frame does not have |
| `target_leakage` | a column on the forbidden list (the labels) |
| `excessive_depth` | deeper than `MAX_DEPTH` (8) |
| `excessive_size` | more than `MAX_NODES` (40) |
| `duplicate_semantics` | a program hashing to an already-registered version |

Window bounds, `min_periods`, quantile coherence, clip bounds and level
agreement are enforced at *construction*, so a malformed node cannot be built at
all.

---

## Versioning: identity is semantic

`FeatureProgram.version()` is the SHA-256 of the tree's canonical, key-sorted
JSON. **The name is excluded.**

```python
FeatureProgram(name="xg_five",  root=xg).version() ==
FeatureProgram(name="whatever", root=xg).version()   # True
```

Two consequences, both wanted:

- A renamed copy of a rejected feature is still that feature, and the registry
  says so.
- An edited program becomes a *new version* rather than silently changing what
  an old evaluation referred to.

---

## Caching

Sub-expressions are keyed by subtree hash and computed once per batch. `xg_5 *
rest` and `xg_5 / price` in the same run stage and aggregate `xg_5` a single
time. This is not a micro-optimisation: a search evaluates hundreds of programs
over the same anchors.

---

## Point-in-time safety

The compiler does not implement its own window logic. It takes a `WindowStager`
protocol, which `xg_alonso.features.generators.stage_window` satisfies — the
implementation the leakage harness already proves correct.

Re-implementing it to keep the engine domain-free would have created a second
copy of the highest-risk function in the platform, and the two would eventually
disagree on exactly the boundary condition that causes silent leakage.

Every program is additionally pushed through `features.leakage.assert_no_leakage`
before it may be registered, and `tests/discovery/test_leakage.py` pairs each
check with a negative control.
