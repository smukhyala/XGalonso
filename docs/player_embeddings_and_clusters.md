# Player Embeddings and Clusters

## Embeddings

`xg_alonso.discovery.embeddings`. Standardise the declared columns, then project
with PCA. Raw vectors here are heavily correlated — several windows of the same
metric — so a Euclidean distance over them is dominated by whichever metric
happens to have the most windows declared. The loadings are retained so a cluster
can be described in terms of original columns rather than "component 3".

**Fit and apply are separate, structurally.** `fit_embedding` returns an
immutable `EmbeddingModel`; `transform` uses the *stored* mean, scale and
loadings. A validation fold cannot enter the standardisation applied to it — not
because the caller remembered, but because the API offers no way to do it.

Missing columns are filled with the **fitted mean**, which becomes zero after
standardisation. That is the honest neutral value. Filling with a literal zero
before scaling would assert that `days_since_last_match` is "played today" and
that `selected` is "owned by nobody".

### Columns

Only what the official FPL API publishes (D6). Present: xG, xA, xGI, threat,
creativity, influence, minutes, starts, appearance rate, xGC, clean sheets,
saves, defensive contribution, BPS, points mean/max/std, ownership, transfer
balance, price, opponent strength, home, rest.

**Absent, and deliberately not approximated:** shots, shots in the box, big
chances, touches in the box, key passes, passes into the penalty area, set-piece
share, pressing intensity. None is published. `threat` proxies shot volume and
`creativity` chance creation — both Opta-derived and official.

## Clusters

`xg_alonso.discovery.clusters`. Three properties the existing
`features.archetypes` does not have:

**Clusters move.** Assignments are keyed by gameweek. A midfielder pushed forward
after an injury genuinely becomes a different asset, and a permanent label cannot
notice. `transition_history` returns only the *moves*, because the transition is
the event.

**Membership is soft.** A softmax over negative squared distance, so a player
between two centroids splits between them. This is what makes gated features
meaningful.

**Similarity is conditioned on the objective.** Two forwards with identical
expected points are interchangeable to a points-maximiser and completely
different to a rank-chaser if one is owned by 60% of the field and the other by
3%.

### Choosing k

Never from one unsupervised metric. Silhouette alone reliably produces three
clusters meaning "good", "average" and "bad" — a ranking with extra steps.
`select_k` combines:

- silhouette (0.40)
- seed-to-seed partition stability, by Rand index (0.30)
- downstream predictive utility, when the caller has measured it (0.30)
- a **disqualifying** penalty when the smallest cluster holds under 4% of the
  pool — a cluster describing 1% of players describes nobody

### Two objective-conditioned approaches

**A — objective-weighted distance.** `objective_weights` reweights each embedding
axis by how much the objective cares about it. Applied to a standardised space,
and carried through the PCA loadings so an emphasis meant for `selected_mean_5`
is not applied to whichever component sits in that slot. Columns the objective
does not name keep weight 1.0, not 0.0: "less relevant" is a weaker claim than
"irrelevant", and the evidence supports only the weaker one.

**B — supervised projection.** `fit_supervised_projection` fits a
ridge-regularised map onto the objective's own target and clusters in the leading
directions of that space. It **refuses** below 800 rows and returns `None`, and
the caller falls back to A. Refusing is correct behaviour, not degradation: a
projection fitted on a thin fold is noise with a matrix behind it, and it would
be indistinguishable from a real one in every downstream report.

### Measured effect

2024-25 GW20, 368 players, k=5, Rand index against the unconditioned control:

| Comparison | Agreement |
|---|---|
| unconditioned vs mini-league chase | 0.762 |
| unconditioned vs rank protection | 0.886 |
| chase vs protection | 0.797 |

The conditioning changes the partition rather than relabelling it. Chase groups
on ceiling and ownership; protection groups on minutes security, which puts
goalkeepers and nailed outfield starters together.

### Summaries

A cluster's `dominant_features` — (column, standardised centroid) pairs — is the
output. `label` is generated from those same numbers and is decoration. A label
without its statistical basis is a story.
