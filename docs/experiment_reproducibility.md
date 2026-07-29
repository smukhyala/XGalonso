# Experiment Reproducibility

Every discovery run emits an `ExperimentManifest`
(`xg_alonso.contracts.discovery`), written to
`.data/reports/<experiment_id>.json`.

## What the manifest carries

| Field | Why |
|---|---|
| `experiment_id`, `stage`, `started_at`, `completed_at` | identity and lifecycle |
| `objective_id`, `objective_version` | *which question was asked* |
| `constraints_hash`, `beliefs_hash` | the constraints and beliefs, fingerprinted |
| `data_cutoff`, `seasons` | what data existed |
| `feature_set_version`, `cluster_model_version` | the representation used |
| `model_config_hash` | estimator hyperparameters |
| `seeds` | **every named RNG seed** |
| `fold_definitions` | every `(index, train_start, train_end, validate_start, validate_end)` |
| `hypotheses_proposed`, `features_compiled`, `features_accepted/rejected` | what happened |
| `metrics` | headline numbers |
| `code_version`, `git_dirty` | the code that ran |

## `reproducible` is a property, not a claim

```python
@property
def reproducible(self) -> bool:
    return bool(self.code_version) and not self.git_dirty
```

A run with uncommitted changes is reported as **not reproducible**, and the CLI
says so in yellow. The recorded commit does not describe the code that ran, so
the outputs cannot be regenerated from it. This mirrors
`RunManifest.promotable`, which already governs artifact promotion.

## Determinism

Every RNG in the discovery path is explicitly seeded:

| Seed | Where |
|---|---|
| `20260727` | estimator `random_state`, permutation importance, controls |
| `20260728` | k-means, embeddings |
| `seed + 1` | objective-conditioned cluster fitting |

The supervised projection is a closed-form ridge solution with no RNG at all.

## The evaluation history is immutable

`DiscoveryRegistry.record_evaluation` uses `append_table`. There is no update and
no delete. Re-measuring a feature adds a row; it never edits one.

That is what makes "what did we believe about this feature in October, and on
what evidence" answerable — and a discovery system whose past conclusions are
unfalsifiable is one whose present conclusions cannot be trusted either.

`accepted_features` reads the **latest** verdict per feature, not the best one. A
feature accepted in March and rejected in May is rejected; taking the best result
ever recorded would make the registry a highlight reel.

## Reproducing a run

1. Check out `code_version`.
2. Rebuild the frame for `seasons` with `xg build-discovery-frame`.
3. Re-run `xg discover` with the same objective text (or load the objective from
   `discovery_objectives` by `objective_id`).
4. Compare `metrics` and the per-feature `discovery_evaluations` rows.

Identical seeds, identical folds and a deterministic pipeline mean the numbers
should match exactly. If they do not, one of the recorded inputs is wrong — which
is itself the finding.
