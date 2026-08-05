<!-- claims
package: packages/discovery
symbols: xg_alonso.discovery.dsl:Arith, xg_alonso.discovery.search:beam_search, xg_alonso.discovery.acceptance
-->

# Interaction Discovery Workflow

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Interaction Discovery |
| Version | 1.1 |
| Status | In progress — the expression exists, the search is not wired |
| Owner | ML Platform |
| Dependencies | [Feature Factory](02_feature_factory.md), [Feature Scientist](03_feature_scientist.md) |
| Last updated | 2026-08-04 |

> **Status correction, 2026-08-04.** This document said `Deferred (Post-MVP)`, which is no longer
> true, but "done" would overstate it in the other direction. Precisely:
>
> - **Interactions are expressible.** `Arith(MUL)` in the discovery DSL
>   (`packages/discovery/src/xg_alonso/discovery/dsl.py`) lets a hypothesis multiply two subtrees, so
>   a proposed interaction compiles, validates against the real schema, and passes through the same
>   leakage proof and walk-forward controls as any other candidate.
> - **A search over feature *sets* exists but is unreached.** `search.py::beam_search` carries the
>   best `k` partial sets forward and is defined, tested and called from nowhere. Wiring it into the
>   discovery loop is in progress.
> - **The five-stage evaluation pipeline below was not built.** Acceptance runs through
>   `discovery/acceptance.py` against criteria fixed in advance, not through the staged screen this
>   document specifies.
>
> Read section 2 with that correction in mind; it is left in place because the reasoning for
> *gating* interaction search — rather than generating combinatorially — is still the operative
> policy.

---

## 1. Purpose and Scope

This document specifies the workflow that turns raw interaction candidates into promoted production features.

Ownership is split across two subsystems:

- The [Feature Factory](02_feature_factory.md) **generates** interaction candidates. The generator contract, supported interaction forms, candidate sources, worked examples, and the blocked-combination rules live in §7.9 of that document.
- The [Feature Scientist](03_feature_scientist.md) **evaluates** those candidates and decides which are promoted, using the five stages below.

This document covers the evaluation and promotion path only. It does not restate the generator contract.

## 2. Gating

The precondition this section set has been met: point-in-time correctness is enforced mechanically
by the leakage harness in `xg_alonso.features.leakage`, and materialization is deterministic. That
is why interaction expression was unlocked.

The policy that survives is the restraint, not the deferral. Interaction search stays **narrow and
gated** rather than combinatorial: crossing every metric with every window would produce thousands
of columns, almost all noise, which is exactly what D12 caps against. An interaction earns its place
by the same route as any other candidate — a falsifiable hypothesis, a leakage proof, a walk-forward
backtest against noise and shuffled controls, and a utility score under a stated objective.

SHAP-driven interaction discovery was never built and is not planned; `xg importance` measures
out-of-sample contribution directly instead.

The stages below were specified before the work started and were not the design that shipped. They
are retained as the contract they were meant to be, not as a description of running code.

## 3. Stage 1: Eligibility

Features must pass:

- sufficient coverage;
- sufficient variance;
- acceptable leakage score;
- stable generation;
- semantic compatibility.

A feature that fails any eligibility criterion is excluded from candidate generation entirely, rather than generated and then screened out later.

## 4. Stage 2: Candidate Generation

Generate interactions among:

- top univariate features;
- complementary feature families;
- model residual drivers;
- domain-allowed pairs;
- targeted triples when pairwise evidence exists.

Triples are generated only where pairwise evidence already exists. The search is deliberately narrow: interaction discovery is controlled, not unrestricted.

## 5. Stage 3: Cheap Screening

Use:

- mutual information;
- univariate gain;
- permutation lift;
- residual correlation;
- small tree models;
- sampled cross-validation.

Screening exists to reduce the candidate set cheaply before any full walk-forward evaluation is spent on it.

## 6. Stage 4: Full Evaluation

Evaluate surviving interactions using walk-forward folds.

Track:

- mean validation lift;
- worst-fold lift;
- stability;
- added latency;
- missingness;
- correlation with existing features;
- target leakage risk.

## 7. Stage 5: Promotion

An interaction can enter a production feature set only when:

- average lift exceeds the minimum threshold;
- no major fold degrades beyond tolerance;
- importance is stable across retrains;
- lineage is complete;
- inference cost is acceptable.

Promotion is the only path into a versioned feature set. A promoted interaction carries a full Feature Card and lineage record exactly like a base feature, and models load it through the registry rather than by direct reference.

---

## Related documents

- [Feature Factory](02_feature_factory.md) — generates the candidates; §7.9 defines the interaction generator contract and blocked combinations
- [Feature Scientist](03_feature_scientist.md) — owns evaluation, selection, promotion, and retirement
- [Prediction Models](07_prediction_models.md) — consumes the promoted feature sets
- [Embeddings](06_embeddings.md) — source of the embedding-similarity interaction inputs
- [Player Clustering](05_player_clustering.md) — source of archetype and cluster terms used in interactions
