<!-- claims
commands: xg models list, xg models verify, xg models backfill-manifest, xg models audit
symbols: xg_alonso.features.career:CAREER_VERSION, xg_alonso.features.catalogue:CATALOGUE_VERSION
-->

# Model Artifacts

**Status:** `Implemented`

What a fitted artifact is, what it must carry, how compatibility is decided, and what is currently
on disk.

---

## 1. Why this exists

Before manifests, `SavedModel` recorded four fields and nothing compared a pickle's fitted feature
schema against the active catalogue. Three failure modes followed, in increasing order of danger:

1. **A bare `KeyError` from `trained.py::_matrix`** when the artifact needed a column the active
   build no longer produces. The traceback named neither the column nor the fix.
2. **`X has 219 features, expecting 224`** from inside scikit-learn — accurate, and useless for
   working out which 5.
3. **No error at all.** A feature tuple of the *right length* in the *wrong order* raises nothing
   and returns plausible, silently wrong predictions. This is the case the whole design exists for.

---

## 2. Compatibility is satisfiability, not equality

`trained.py::_matrix` selects columns **by name** using the artifact's own tuple, so a feature
frame carrying *extra* columns is fine. The question a gate must ask is therefore:

> Can the active build supply, by name, every column this artifact was fitted on?

Not "are the two schemas identical". Measured against the 224-feature active schema, an
equality-based gate would classify **6 of 8 artifacts as broken on day one** and take `xg plan`,
`xg score`, `xg backtest`, `xg importance` and the API down with them. Every one of those six
predicts correctly.

Catalogue-hash equality is kept as a **separate, non-blocking staleness signal**. "Trained against
a different feature definition" is worth knowing; it is not worth refusing.

### Check order

Nothing below fires before everything above it resolves.

| # | check | severity | why |
|---|---|---|---|
| 1 | `artifact_version` read from raw JSON **before** pydantic | blocking | `extra="forbid"` would turn a future manifest into an unreadable validation dump |
| 2 | `feature_count` vs `len(feature_names)` vs each estimator's `n_features_in_` | blocking | pre-empts sklearn's uninformative arity message |
| 3 | `manifest.feature_names == models.feature_columns`, **exact and ordered** | blocking | the reordered-tuple case, which raises nothing and returns wrong numbers |
| 4 | satisfiability — `set(needed) - set(active) - set(extension)` | blocking | turns a bare `KeyError` into a named list plus the fix |
| 5 | unexpected features (active supplies what the artifact never saw) | **warning** | 6 of 8 artifacts are proper subsets and predict correctly |
| 6 | relative order of shared names moved | warning | selection is by name, so it cannot break inference |
| 7 | catalogue version / hash mismatch | warning if satisfiable, else blocking | staleness, not breakage |
| 8 | `rules_snapshot_hash` mismatch | blocking | components are priced with the *active* rules; a model fitted at GK-goal=6 and priced at 10 is unreconcilable |
| 9 | sklearn major/minor drift | warning | |

Checks run against **manifest fields alone**, so an incompatible artifact raises *before*
`pickle.load` is ever called. That is what makes "a raw `KeyError` must never be the first failure"
structural rather than lucky.

---

## 3. What a manifest is, and where it lives

Written **twice**: as a JSON sidecar (`<name>.pkl.manifest.json`) and embedded in the pickle.

The sidecar is the point — **reading provenance must not execute code**, and unpickling an unknown
file to read its manifest is exactly the risk being managed. The embedded copy survives a `.pkl`
being copied without its sidecar. Disagreement between the two on `payload_sha256` is blocking.

Recorded: artifact and code version, git commit and dirty flag, runtime versions, feature
catalogue version and hash, the **ordered** feature names, dropped and extension features, rules
snapshot hash, training-data manifest hash, training seasons/gameweeks/rows, fit start and end
times, estimator configuration, fold metrics, calibration metadata, model fingerprint, payload
digest and size.

Two hashes deserve their reasoning stated, because the obvious choices are both wrong:

- **`rules_snapshot_hash` is not `ScoringRules.source_sha256`.** That is the hash of the whole
  `bootstrap-static` payload — the same payload whose `elements[].now_cost` changes daily — so
  gating on it would mark every model rules-drifted within twenty-four hours of training. The hash
  covers what the rules *say*, excluding `source_sha256` and `fetched_at`. It **includes**
  `thresholds`, the block FPL does not publish: a change to `saves_per_point` re-prices every
  prediction and no drift check can catch it.
- **`training_data_manifest_hash` is not the parquet bytes.** Parquet is not byte-stable, so
  rewriting identical rows changes compression blocks and the digest would move on every
  `xg backfill`. It is a content *summary*: row count, ordered schema, season and gameweek
  coverage, latest `available_time`, player count, plus the bronze snapshots' own content hashes.
  Recorded, never blocking — a model trained on last week's snapshot is a legitimate model.

**`model_fingerprint` is not an identity.** It hashes name, version, sorted label keys, columns and
row count, so two models fitted on *different data* with the same shape collide.
`payload_sha256` is the true identity.

---

## 4. Current inventory

Audited against the 224-feature active schema (catalogue `8601b507`), 2026-07-31.

| artifact | features | missing | active-but-unseen | manifest | status |
|---|---|---|---|---|---|
| `fixed.pkl` | 224 | 0 | 0 | yes | **compatible** (exact) |
| `holdout.pkl` | 219 | 0 | 5 | yes | compatible (stale) |
| `recency.pkl` | 219 | 0 | 5 | yes | compatible (stale) |
| `career.pkl` | 214 | 0 | 10 | yes | compatible (stale) |
| `early.pkl` | 184 | 0 | 40 | yes | compatible (stale) |
| `late.pkl` | 184 | 0 | 40 | yes | compatible (stale) |
| `component_models.pkl` | 141 | 0 | 83 | yes | compatible (stale) |
| `component_models.expected_rank_gain_aggressive_h3_differential.pkl` | 206 | **5** | 23 | **no** | **migratable** |

The eighth artifact is the only genuinely broken one. It needs five interaction features the active
build no longer produces — `xg_x_expected_minutes`, `xg_x_opponent_weakness`,
`rest_x_minutes_load`, `creativity_x_opponent_solidity`, `home_x_attacking_output` — all of which
are regenerable via `xg train --objective …`, which is why it is *migratable* rather than
*archival*. It carries no manifest because it was produced before manifests existed and was never
re-saved.

> **Provenance caveat.** `.data/` is gitignored, so this table describes a local working directory
> and cannot be reproduced from a fresh checkout. `component_models.expected_rank_gain_…` existed
> only inside a git worktree that was subsequently removed; it was copied into `.data/models/`
> before removal, and that copy is the only one. Treat this section as a record of a moment, not a
> guarantee about any other machine.

`ARCHIVAL` is reserved for artifacts whose features **no** code can produce. Nothing currently
qualifies.

---

## 5. The index, and why it is append-only

`.data/models/_index.jsonl` — deliberately the same idiom as `storage/bronze.py::_manifest.jsonl`.
Status changes are **appended events, not edits**, so "no deletion, provenance preserved" falls out
of the file format rather than being a rule someone has to remember.

Entries are keyed on `artifact_id = payload_sha256[:16]`, which is stable across moves;
`previous_path` is set when an audit relocates a file.

---

## 6. Reading an artifact without executing it

Two stages, because either alone is insufficient.

**Stage 1 — static opcode scan** (`pickletools.genops`). Never resolves a global, never imports;
0.06–0.09 s for a 4 MB pickle. Protocol ≥4 puts module and class names on the stack as *string
constants*, so no module can be referenced without its name appearing as a string —
over-approximating is the safe direction. Allowlisted: `xg_alonso.*`, `sklearn.*`, `numpy.*`,
`scipy.*`, `builtins`, `copyreg`, `collections`, `datetime`, `_codecs`. Deliberately absent: `os`,
`subprocess`, `posix`, `pickle`, `functools`, `operator`.

**Stage 2 — `Unpickler.find_class` override**, consulted before each global actually resolves.

A precise `STACK_GLOBAL` stack simulator was deliberately *not* built: over-approximation plus
Stage 2 is simpler and strictly stronger. Scanning is capped by file size and opcode count, because
`genops` never executes but a crafted file can still declare a very large string.

---

## 7. Commands

```bash
xg models audit [--dry-run|--apply]   # --dry-run is the DEFAULT
xg models list [--all]                # default listing shows COMPATIBLE only
xg models verify <path>
xg models backfill-manifest <path>
```

Those four are the whole sub-app. An earlier draft of this section documented
`xg models activate|deactivate`; **neither was ever implemented**, and there is no notion of an
"active" artifact that a command can toggle. Compatibility is decided per artifact against the
current build, at the moment it is loaded, rather than by a stored activation flag — which is the
stronger design, because an activation flag can be true and wrong.

`--dry-run` is the default because `.data` is gitignored and a bad `--apply` is unrecoverable. The
audit **moves** rather than copy-then-deletes, and appends the index entry *before* the move, so a
crash mid-move still leaves a record.

Backfilled manifests use `artifact_version="legacy_0"` with unknowable fields left explicitly empty
(`git_commit=""`, `feature_catalogue_hash=""`, `rules_snapshot_hash=""`) rather than guessed. Empty
hashes are treated as *unknown* and downgrade to a warning — never a pass. **A backfilled manifest
is a description, not a certificate.**

---

## 8. Known gaps

- **The pre-merge baseline backtest was never recorded.** Workstream C changed free-transfer
  accrual, autosubs and vice-captaincy, all of which move every historical number, and the plan
  called for a pre-merge run so the delta would be attributable. It was not made, so the before/
  after table in the tranche description rests on the post-change runs alone.
- **`xg evaluate run` does not exist yet.** `run_one` is injected but never wired, so the
  `LEGACY_HEADLINE` reproduction gate against
  `.data/reports/20260727T211139Z-2024-25-gw6-25/` is **unrun, not passed**.
- **The catalogue hash cannot see imperative feature code.** Opponent, career and recency features
  come from hand-written functions rather than declarative specs, so only their names and a module
  version constant are hashable. Editing the arithmetic inside `build_career_features` without
  bumping `CAREER_VERSION` (`packages/feature_factory/src/xg_alonso/features/career.py`) will not
  change the digest. This is exactly why a catalogue-hash mismatch is a warning rather than a
  refusal, and why those version constants exist at all.

---

## Related documents

- [07_prediction_models.md](07_prediction_models.md) — what the models predict and how points are assembled
- [02_feature_factory.md](02_feature_factory.md) — the feature catalogue the schema hash covers
- [../backtesting_and_leakage.md](../backtesting_and_leakage.md) — walk-forward folds and freeze assertions
