<!-- claims
package: packages/discovery
commands: xg discover, xg build-discovery-frame
routes: GET /features/discovered, GET /hypotheses, GET /clusters, GET /experiments, GET /players/{player_code}/cluster-history
symbols: xg_alonso.discovery.dsl:parse_program, xg_alonso.discovery.compile, xg_alonso.discovery.hypotheses, xg_alonso.discovery.harness, xg_alonso.discovery.utility, xg_alonso.discovery.acceptance, xg_alonso.discovery.registry, xg_alonso.discovery.llm, xg_alonso.discovery.experiment:residual_weakness, xg_alonso.features.leakage
-->

# Objective-Conditioned Feature Discovery

## The question this changes

Automated feature engineering normally asks:

> Which features maximise predictive accuracy?

That has one answer. This system asks a different question, which has several:

> Given this manager's objective, constraints, beliefs, planning horizon and
> required features — which representation of the player pool produces the best
> *decisions*?

The difference is not academic. A manager forty points behind in a mini-league
and a manager protecting a top-1% overall rank are not looking for the same
player, are not served by the same features, and are not well described by the
same notion of "similar player". A feature that sharpens the mean while
flattening the tail improves RMSE and makes the first manager strictly worse
off, because the tail is the only thing that moves rank.

---

## What is automated, what is deterministic, what uses an LLM

Stated first, because it is the claim most easily overstated.

| Component | Status |
|---|---|
| Feature DSL, compilation, validation | **Deterministic** |
| Leakage checking | **Deterministic**, mechanical |
| Walk-forward backtesting | **Deterministic**, seeded |
| Utility scoring, acceptance policy | **Deterministic**, thresholds declared in advance |
| Embeddings, clustering, k selection | **Deterministic**, seeded (`20260727` / `20260728`) |
| Objective compilation from text | **Deterministic** — regex patterns with fixed meanings |
| Hypothesis generation | **Automated, not intelligent** — searches a declared space of mechanisms, aimed at measured residual weakness |
| Language model | **Optional, off by default.** `--llm` enables an adapter; without a key the deterministic generator runs alone |

### The language-model proposer

`discovery/llm.py` lets a model propose hypotheses. What makes it safe is not the
model's good behaviour — it is that **the model can only emit data**:

- Its output is a JSON expression tree, parsed by `parse_program`. Nothing is
  ever `eval`'d or `exec`'d, and no Python is generated.
- It cannot express a computation the DSL cannot express. A hallucinated column,
  a level mismatch, a malformed window and a read of the target column are all
  refused before anything is computed.
- A proposal is **never repaired**. Fixing one would file our hypothesis under
  the model's name.
- Every proposal is recorded with `GenerationSource.LLM` permanently, so an
  LLM-suggested feature stays distinguishable from a measured one forever.
- The model **gets no vote on acceptance**. Its rationale is stored beside the
  fold-level evidence, never in place of it; the same walk-forward measurement
  and the same noise and shuffled controls decide.
- It is **additive and never fatal**. The deterministic generator runs either
  way, and a missing key, missing SDK or failed call degrades to "no extra
  proposals" rather than breaking the run.

- The **budget is split, not concatenated**. Appending the model's proposals to
  the deterministic pool and truncating from the front discards them entirely
  whenever the deterministic generators already fill the quota — the common
  case — so `--llm` would pay for a call and contribute nothing, silently.
  `split_budget` reserves up to half the budget for the model, and reports the
  count it had to drop rather than letting a cap read as "everything was tried".

Enable with `xg discover ... --llm` and an `ANTHROPIC_API_KEY` in the
environment or in a `.env`. Install the optional extra with
`uv sync --extra llm`.

The key is resolved by walking **up** from the working directory to the nearest
`.env` that defines it. That matters in a git worktree, which gets its own copy
of untracked files: a key written to the main checkout is otherwise invisible,
and a stale `.env` left in the worktree shadows it silently — producing a 401
that looks like a bad credential rather than a bad path. `api_key_origin()`
reports which file was read, so that failure is one line away from diagnosed.

Verified live on 2026-07-29 against `claude-opus-5`: four proposals returned,
all four parsed, all four passed static validation against the real schema and
forbidden-column list, all four were measured over five walk-forward folds — and
**none was accepted**. Two were marked `revise` and two `rejected`; the four
accepted features that run were all deterministic. That is the design working:
the model proposes, the measurement decides.

**Every relationship reported is a predictive association.** No causal claim is
made anywhere in the output vocabulary. A hypothesis's `football_rationale` is a
stated mechanism, not a finding — the finding is the fold-level evidence beside
it.

---

## The loop

```
compile objective ─► measure residual weakness ─► propose falsifiable hypotheses
      ▲                                                        │
      │                                                        ▼
  record lesson ◄── accept / reject ◄── score under objective ◄── compile to a
                                                                  safe program
                                                                       │
                                            walk-forward backtest ◄────┤
                                                                       ▼
                                                        prove point-in-time safe
```

Step by step, with the module that owns it:

| # | Step | Module |
|---|---|---|
| 1 | Encode the manager's objective, constraints and beliefs | `contracts.objective`, `domain.intent` |
| 2 | Measure where the required-feature model is worst | `discovery.experiment.residual_weakness` |
| 3 | Propose hypotheses with falsification conditions | `discovery.hypotheses` |
| 4 | Compile to a safe feature program | `discovery.dsl` |
| 5 | Validate statically against the real schema | `discovery.compile.validate_program` |
| 6 | **Prove point-in-time safe** | `features.leakage` |
| 7 | Compute historically | `discovery.compile` |
| 8 | Backtest walk-forward, paired per fold | `discovery.harness` |
| 9 | Compare against noise and shuffled controls | `discovery.harness` |
| 10 | Score utility under the objective | `discovery.utility` |
| 11 | Accept, reject, or mark for revision | `discovery.acceptance` |
| 12 | Register the verdict and the lesson | `discovery.registry` |

Step 6 cannot be skipped. A feature that leaks validates beautifully and loses
money, and `DiscoveryRegistry.register_feature` refuses any spec whose
`validation_status` is not `LEAKAGE_PASSED`.

---

## Objectives

Six presets ship (`domain.objectives`), and a user's text adapts one of them.
They differ in more than risk appetite, because the situations do:

| Preset | Primary metric | Risk | Horizon | Ownership |
|---|---|---|---|---|
| `expected_points` | expected points | balanced | 5 | neutral |
| `rank_protection` | downside protection | conservative | 2 | template |
| `mini_league_chase` | expected rank gain *(proxy)* | aggressive | 3 | differential |
| `team_value_growth` | transfer momentum *(not a price forecast)* | balanced | 3 | neutral |
| `wildcard_prep` | expected points + flexibility | balanced | 6 | neutral |
| `locked_premium_aggressive` | expected rank gain *(proxy)* | aggressive | 3 | differential |

### The signed variance term

`RiskPreference.variance_sign` returns `+1.0` for conservative, `+0.35` for
balanced and **`-0.5` for aggressive**. That negative value is the single most
important number in the objective layer.

Without it, "aggressive" degrades to "slightly less cautious" and every
objective converges on the same template squad. With it, a chaser is *rewarded*
for volatility — which is correct, because a certain average leaves them exactly
as far behind as they started.

### Objectives, constraints and beliefs are three different things

Confusing them is how an optimizer comes to sell a player the user said to keep:

- **Objective** — what to maximise. Soft, traded off against itself.
- **Constraint** — what is not negotiable. Hard, *never* traded off. Applied as
  a filter before scoring, not as a penalty. A penalty large enough to usually
  prevent a sale is still one a good enough alternative overcomes.
- **Belief** — what the manager thinks they know. Uncertain evidence, bounded,
  and never allowed to overwrite a prediction.

---

## User beliefs

`prediction.beliefs.apply_beliefs` returns a `BeliefAdjustment` carrying **both**
the raw and the adjusted prediction, plus the multiplier and its rationale. The
raw prediction is never mutated.

- Adjustments are **multiplicative and clamped** at `BELIEF_CLAMP = 0.25`. At
  maximum stated confidence a belief moves a projection by 25%, and no
  combination of beliefs exceeds that. A manager who is certain still does not
  overrule four seasons of data with a sentence.
- Components are scaled and the points **reassembled through the domain**, not
  multiplied at the total — the prediction contract requires the breakdown to
  sum to the total, and only the domain knows the scoring rules.
- `expected_points_sd` is deliberately **not** reduced. A belief is not
  evidence, and acting on one must not make the projection look more certain
  than the data made it.
- `belief_sensitivity` sweeps the confidence so a user can see where a
  recommendation would flip.

---

## The feature DSL

A feature program is a frozen Pydantic expression tree. **Nothing is ever
`eval`'d or `exec`'d**, and no Python source is generated. See
[`feature_dsl.md`](feature_dsl.md) for the full grammar.

Two things are *unrepresentable* rather than merely discouraged:

- **There is no division.** Only `safe_div`, which returns null when the
  denominator is negligible. A divide-by-zero cannot be written.
- **No node takes a forward offset.** Every temporal primitive describes a
  lookback over appearances already visible.

A **row/entity level system** rejects expressions that silently mix one value
per historical match with one value per player. `rolling_mean(minutes) * minutes`
type-checks as arithmetic and means nothing; it raises here.

---

## How features are accepted

`discovery.acceptance`. Gate order is deliberate:

1. **Leakage** — fatal, checked first. A leaking feature's measured value is
   fiction, so there is nothing to weigh it against. An *unchecked* feature is
   also rejected: absence of evidence is not evidence of absence.
2. **Sufficiency** — fewer than `min_folds` (default 3) is
   `INSUFFICIENT_DATA`, a distinct verdict from `REJECTED`. A verdict on two
   folds is absent, not negative, and recording it as negative would write the
   wrong lesson into `discovery_lessons`.
3. **Quality** — utility, fold win rate, incremental value, recent degradation,
   missingness, turnover, complexity. Collected rather than short-circuited, so
   a revision has the whole list.

Recent degradation is checked *separately* from the average, because a feature
that helped for two seasons and stopped helping this one is the most dangerous
kind: the model leans on it hardest exactly when it stopped working.

### The controls

A candidate must beat **two** controls or it is rejected regardless of its gain:

- **Noise control** — the required set plus a matched-complexity random column.
  A boosted model handed one more column often improves slightly whatever it
  contains.
- **Shuffled control** — the required set plus a permuted copy of the candidate
  itself. Stronger: it preserves the candidate's exact scale, skew and
  missingness and destroys only the association with the target.

Without them, "adding this feature helped" is a statement about model capacity,
not about football.

### Classifications

`GLOBALLY_COMPLEMENTARY`, `OBJECTIVE_SPECIFIC`, `CLUSTER_SPECIFIC`, `REDUNDANT`,
`UNSTABLE`, `LEAKAGE_SUSPECTED`, `INSUFFICIENT_HISTORY`.

The distinction between the first three is the product. A feature that helps
everywhere is a model improvement; one that helps only under an objective is
evidence the objective layer does something real; one that helps only for some
clusters should be *gated* rather than added globally, where it would dilute
into noise for everyone else.

---

## Clusters

See [`player_embeddings_and_clusters.md`](player_embeddings_and_clusters.md).
In short: clusters are keyed by gameweek (they move), membership is soft
(so features can be gated in proportion), and similarity is conditioned on the
objective — both by reweighted distance and by a supervised projection.

Measured on 2024-25 GW20 the conditioning changes the partition rather than
relabelling it: Rand index **0.76** between the unconditioned control and
mini-league chase, **0.89** against rank protection.

---

## Limitations

Stated plainly, because several are structural rather than temporary.

1. **`expected_rank_gain` is a proxy, not a rank.** True overall rank needs the
   global distribution of every manager's squad, which the public API does not
   publish. What is computed is an ownership-weighted points differential.
2. **Ownership is `selected`, not effective ownership.** EO accounts for
   captaincy and is not published.
3. **`team_value_growth` scores transfer momentum, not price changes.** D11
   defers the price model and no current-season price data exists at GW1.
4. **Several of the brief's mechanisms are not computable.** Touch counts,
   shots in the box, key passes, passes into the penalty area and pressing
   intensity are not in the official FPL API, and D6 forbids other sources. The
   affected hypotheses use documented substitutes; see
   `discovery.hypotheses` for the mapping. Nothing reads a zero and calls
   itself a pressing interaction.
5. **`defensive_contribution` is 73.7% null** — one season of four. Programs may
   reference it, and the missingness gate will normally reject them. That
   rejection is correct, not a bug.
6. **Chips are state, not logic** (D5). `chip_plan` is carried and validated;
   no chip is optimised.
7. ~~**Decision-quality metrics are wired but not yet driving acceptance.**~~
   **Closed 2026-08-04.** `discovery/decision_metrics.py` registers one metric
   per `PrimaryMetric` into the previously-empty `MetricRegistry`, so
   `objective_gain` is a real objective-specific measurement rather than a
   `predictive_gain` alias, and `decision_gain` and `turnover_penalty` are
   measured rather than permanently `unmeasured`.

   Every metric is computed on the **predicted top-k drawn from the reachable
   rows**, which is what makes them decision metrics rather than fit metrics —
   and what lets two managers get different `objective_gain` from an identical
   model and feature. Measured on the demo fixture, one candidate scores
   **+0.26 for captaincy upside and −0.47 for downside protection**: the same
   feature, opposite verdicts, which is precisely the property that was false
   while the registry was empty.

   Costs no extra model fits — `harness.fold_predictions` reads the cache the
   subgroup breakdown already populated. What is *not* done: `decision_gain` is
   top-k realised points, not a policy backtest through
   `evaluation.walk_forward`. Simulating squads would mean hundreds of solves
   per candidate inside a loop already fitting nine models per candidate, and an
   unusable metric is an unmeasured one with extra steps. Hit costs and squad
   legality beyond the reachability mask are still outside the term.
8. **A thin required-feature baseline makes complements easy to find.** When a
   user says "keep xG", the baseline is xG *alone*, so gains are measured
   against a one-feature model and are correspondingly large. That is the right
   answer to the question asked, but it is not the same as a gain over the full
   171-feature catalogue.

---

## Running one experiment end to end

```bash
# 1. Build the point-in-time training frame (the expensive step, run once).
uv run xg build-discovery-frame --seasons 2023-24,2024-25

# 2. See how a request parses, without running anything.
uv run xg discover "I am 40 points behind in my mini-league. Keep Haaland and
  my current defense. I want an aggressive three-gameweek strategy. Recent xG
  must remain in the model. Find signals that complement xG." --dry-run

# 3. Run it. The 2025-26 season is held out of every fold by default.
uv run xg discover "<same request>" --max-hypotheses 6 --clusters 5
```

The run prints the parsed intent, the residual weaknesses generation was aimed
at, every program refused before computation, the per-feature verdicts with
their reasons, the objective-conditioned clusters, the complementary search
path, and the lessons recorded. A machine-readable manifest is written to
`.data/reports/<experiment_id>.json`.

A run whose working tree is dirty is reported as **not reproducible**, because
the recorded commit does not describe the code that ran.

---

## The recommendation layer

`xg advise` answers a different question from `xg recommend`. That command asks
which transfer gains the most expected points; this one asks which transfer best
serves *what the manager is trying to do*, and shows what their own instructions
cost separately from what the model thinks.

The three inputs enter at three different points, deliberately:

**Constraints filter the search before anything is scored.**
`rank_single_transfers` takes a `sellable` set; a held player is simply not in
it. Expressing "keep Haaland" as a large negative score still lets a good enough
alternative buy him out — silently, since the recommendation would look like any
other. He still appears on the transfer board carrying a `CONSTRAINT_HELD`
reason, because "why not him?" must have an answer for every squad member, and
"you told me not to" is a better answer than silence.

**The objective re-prices the predictions.** `objective_valued` is one
substitution point that every scoring path already reads. Threading an objective
through `starting_xi_points`, captaincy and the bench separately would leave some
paths using it and others not — and the ones that did not would be exactly the
subtle cases, a captain chosen on raw points while the squad around him was
chosen on rank gain. This is the same argument `horizon_valued` already makes.

**Beliefs produce a second recommendation, never a replacement.** Both are
returned, and the output says explicitly whether the belief changed the answer.

Worked example against the sample squad:

```
What your instructions changed
  1 player(s) held by your instruction and never offered for sale
  no transfers considered in: DEF

What holding them cost
  keeping Haaland gave up 3.59 projected points (best alternative: Osula)
  keeping van Ewijk gave up 2.21 projected points (best alternative: O'Shea)

With your beliefs applied
  Haaland: belief raised the projection from 2.78 to 3.26 (+0.48)
  Your beliefs did not change the recommendation — they were weighed and the
  move stands either way.
```

`selected_by_percent` was added to the player normalisation for this. It is
published on every element and was previously dropped, which would have left the
ownership term — the thing that *defines* a differential or template objective —
silently scoring every player identically.

---

## Service layer

Read surfaces over what `xg discover` produced, plus the compiler:

```
POST /objectives/compile              parse a request; returns confidences and unparsed clauses
GET  /objectives                      the six presets
GET  /features/discovered?objective_id=...   every verdict, accepted and rejected
GET  /hypotheses                      each with its falsification condition
GET  /clusters?objective_id=...       with the statistical basis behind each label
GET  /players/{code}/cluster-history  a cluster is not an identity
GET  /experiments, /experiments/{id}  manifests
```

**Running an experiment is deliberately not exposed over HTTP.** A discovery run
fits hundreds of models and takes minutes, and there is no job queue in this
repository (D1 keeps everything local). A request that blocks for that long is
not an API, it is a timeout. `ExperimentStage` carries the vocabulary a queue
would report, so adding one later is a change of execution strategy rather than
of interface.

### What the web surface covers, and what it does not

**Partly built since this section was first written.** `apps/web/app/discovery/page.tsx` exists and
reads `GET /features/discovered`, `GET /hypotheses`, `GET /clusters` and `GET /experiments`, so the
feature-discovery lab and a cluster view ship. `apps/web/app/plan/page.tsx` covers the objective
builder by way of a typed request rather than a form of controls.

Still not built: a **recommendation comparison** — two objectives' recommendations for the same
squad, side by side, which is the view that would make the whole conditioning argument legible at a
glance. And the cluster surface is a list rather than an explorer: there is no way to walk from a
cluster to its members to a member's history in the UI, though
`GET /players/{player_code}/cluster-history` would serve it.

Stating this plainly is better than a scaffold that looks finished.

---

## Reproducibility

Every experiment emits an `ExperimentManifest` carrying the objective and its
version, hashes of the constraints and beliefs, the data cutoff, the seasons,
the cluster model version, the model configuration hash, **every named seed**,
**every fold definition**, the metrics, the code commit and whether the tree was
dirty.

See [`experiment_reproducibility.md`](experiment_reproducibility.md).
