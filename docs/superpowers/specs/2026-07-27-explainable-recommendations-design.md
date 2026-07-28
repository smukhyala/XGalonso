# Explainable Recommendations — Design

**Date:** 2026-07-27
**Status:** Implemented

---

## Problem

The GW1 recommendation screen states a decision — *Sell João Pedro, buy Osula, +2.62 pts* —
and then fails to justify it. Three symptoms, one root cause and two independent gaps.

### Symptom 1: the justifications read as self-contradictory

The screen shows, as an undifferentiated list:

- "Minutes look secure: 100% chance of starting, around 70.2 minutes expected."
- "Minutes are a concern: 60% chance of starting, around 54.0 minutes expected."

Both sentences are true, and they are about *different players*. `Reason` carries
`subject: PlayerCode` and the API returns it, but `TheCall.tsx` renders only `reason.text`.
The attribution exists in the data and is discarded at the last hop.

### Symptom 2: no feature is ever named

"0.53 projected goal involvements" is `components.goals + components.assists` — the model's
own output for the gameweek, not an underlying statistic. The catalogue contains
`expected_goals_per90_*`, `expected_assists_per90_*`, `threat_per90_*` and 168 others, and
not one of them can appear in an explanation.

**This is the root cause.** `predict_with_models` accepts a feature frame of 171 columns,
emits nine component numbers, and discards the frame. `PlayerPrediction` has no link back to
the features that produced it, so the explanation layer has nothing to cite except model
output. Every other explanation failure is downstream of this.

### Symptom 3: the choice set is invisible

Hato is projected at 2.2 and stays; João Pedro at 2.6 and goes. That looks arbitrary until
you know that transfers are position-locked (`transfer.py:159`) — Hato is a DEF, Osula a FWD,
so Hato was never a candidate for that slot. The constraint is enforced in the optimizer and
stated nowhere in the product.

### Gap 1: only one transfer is offered

`rank_single_transfers` evaluates every legal move and returns them sorted.
`best_single_transfer` keeps `ranked[0]` and discards the rest. The work is already done and
thrown away.

### Gap 2: feature importance does not exist

`grep -rn importance` over the source tree returns documentation only. Nothing measures which
of the 171 catalogue features earn their place. `HistGradientBoosting` exposes no
`feature_importances_`, so this requires permutation importance and there is no code for it.

---

## Non-goals

- No LLM in the explanation path. Reasons stay template-rendered from validated evidence.
- No new features added to the catalogue. This work measures and cites what exists.
- No multi-transfer packages beyond what `rank_single_transfers` already scores.
- No change to the scoring rules, the pinned snapshot, or the drift check.

---

## A. Feature evidence on predictions

### The panel

A declared constant `EXPLANATORY_PANEL` in `xg_alonso.contracts.evidence`, versioned by
`EVIDENCE_PANEL_VERSION`. Roughly fourteen features chosen because they are both
interpretable to a manager and load-bearing in the model:

| Feature | What it answers |
|---|---|
| `expected_goals_per90_5` | Is he getting into scoring positions? |
| `expected_assists_per90_5` | Is he creating? |
| `expected_goal_involvements_per90_10` | Level, over a longer window |
| `threat_per90_5` | Shot volume proxy |
| `creativity_per90_5` | Chance creation proxy |
| `bps_per90_5` | Bonus-point likelihood |
| `minutes_mean_3` | Recent playing time |
| `starts_mean_5` | Is he a starter or a rotation risk? |
| `total_points_mean_5` | Recent return |
| `total_points_max_5` | Ceiling |
| `total_points_std_10` | Volatility |
| `opponent_conceded_xg_mean_5` | How leaky is the opponent? |
| `opponent_clean_sheets_against_mean_5` | How often does the opponent keep one out? |
| `is_home` | Venue |

Panel membership is **not** open-ended. Exposing all 171 per player would produce a wall of
correlated numbers that explains nothing; the panel is curated for interpretability and
cross-checked against the importance table (§E) so it cannot drift into citing a feature the
model does not use.

### The contract

```python
class FeatureValue(BaseModel):
    """One feature's value for one player, with the context that makes it legible."""

    name: str
    value: float | None  # None when the feature could not be computed
    percentile: float | None  # Rank within the same position, 0-1
    family: str  # From the catalogue spec


class FeatureEvidence(BaseModel):
    panel_version: str
    values: tuple[FeatureValue, ...]

    def get(self, name: str) -> FeatureValue | None: ...
```

`PlayerPrediction` gains `feature_evidence: FeatureEvidence | None = None`. Optional, because
the closed-form baseline path does not build the full catalogue and must not pretend it did.

### Percentiles

A raw value carries no meaning to a reader. `0.53 xG/90` is either elite or unremarkable
depending on position. Percentiles are computed **within position, across every player
predicted in that gameweek**, after prediction and before explanation. Computed once per
prediction batch, not per player.

### Population

`predict_with_models` gains an `evidence_panel` argument (defaults to the declared panel) and
attaches evidence from the same frame it predicts on. `predict_frame` (baseline) attaches
whatever slice-1 features overlap the panel and leaves the rest `None`. **A feature that
could not be computed is null, never imputed** — consistent with the existing treatment of
missing history as NaN rather than an invented average.

---

## B. Reason vocabulary

### Expansion

From seven codes to seventeen. The three currently emitted stay; the four defined-but-dead
codes start firing; ten are new.

**Currently emitted, kept:** `EXPECTED_MINUTES_SECURE`, `EXPECTED_MINUTES_DECLINE`,
`UNDERLYING_STATS_IMPROVING`.

**Defined but never emitted, now wired:** `FIXTURE_SWING_POSITIVE`, `FIXTURE_SWING_NEGATIVE`,
`UNDERLYING_STATS_DECLINING`, `AVAILABILITY_RISK_HIGH`.

**New, citing named features with value and percentile:**

| Code | Template shape |
|---|---|
| `XG_RATE_HIGHER` / `XG_RATE_LOWER` | "xG per 90 over the last 5 appearances: {value:.2f} — {percentile:.0%} among {position}s, against {other:.2f} for the alternative." |
| `XA_RATE_HIGHER` / `XA_RATE_LOWER` | as above, expected assists |
| `THREAT_HIGHER` | "Higher shot volume: threat per 90 of {value:.1f}, {percentile:.0%} among {position}s." |
| `CEILING_HIGHER` | "Bigger ceiling: best return in the last 5 appearances was {value:.0f} points against {other:.0f}." |
| `VOLATILITY_LOWER` | "Steadier: points standard deviation {value:.2f} against {other:.2f}." |
| `BONUS_MAGNET` | "Bonus-point profile: {value:.1f} BPS per 90, {percentile:.0%} among {position}s." |
| `PRICE_EFFICIENCY` | "Better value: {value:.2f} projected points per million against {other:.2f}." |
| `POINTS_BREAKDOWN` | "{total:.2f} projected = {appearance:.2f} appearance + {goals:.2f} goals + {assists:.2f} assists + {clean_sheets:.2f} clean sheet + {bonus:.2f} bonus." |
| `POSITION_LOCKED` | "A transfer must be like-for-like: only {candidate_count:.0f} {position}s were legal replacements." |
| `BUDGET_LOCKED` | "Budget caps the choice: {budget:.1f}m available against a market where the next upgrade costs {shortfall:.1f}m more." |

The last two exist specifically to answer *"why wasn't Hato considered?"* — the constraint is
part of the explanation, not an implementation detail.

### Grounding is unchanged

Every new code goes through the same `_evidence_satisfies_template` validator. A reason whose
evidence cannot fill its own template still refuses to be constructed. The vocabulary grows;
the guarantee does not weaken.

Templates stay **numeric-only** — no player names inside evidence dicts. The renderer attaches
the subject's name from the name map it already receives. This keeps the contract layer free
of presentation concerns and keeps `evidence: dict[str, float]` honestly typed.

### Position in templates

`{position}` is a string, and `evidence` is `dict[str, float]`. Rather than widening the
evidence type, `Reason` gains an optional `context: dict[str, str]` for non-numeric template
slots, validated by the same mechanism. Numeric claims stay in `evidence`; only labels live in
`context`, so nothing quantitative can enter prose unvalidated.

### API shape

`ReasonOut` gains `subject_name: str` and `polarity: str`. The UI groups reasons under the
player they concern, which is what dissolves the contradictory-looking list.

---

## C. Per-player justification

New module `packages/explanations/src/xg_alonso/explanations/player.py`.

```python
@dataclass(frozen=True)
class StartVerdict:
    is_starter: bool
    margin: float  # Points ahead of, or behind, the marginal XI place
    marginal_player: PlayerCode | None
    formation_note: str  # e.g. "the 3-defender minimum forces a DEF into the XI"


@dataclass(frozen=True)
class ReplacementOption:
    player_in: PlayerCode
    net_gain: float
    price_delta: int
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class PlayerExplanation:
    player_code: PlayerCode
    breakdown: PointsBreakdown
    evidence: tuple[FeatureValue, ...]  # Panel, sorted by |percentile - 0.5| desc
    start_verdict: StartVerdict
    replacements: tuple[ReplacementOption, ...]  # Top 3, may be empty
    no_replacement_reason: Reason | None  # Grounded, when replacements is empty
```

`explain_player` is a pure function over `(prediction, percentiles, squad, ranked_transfers,
rules)`. It performs no ranking of its own — it filters the board the optimizer already
produced. That keeps a single source of truth for which moves are legal and what they gain.

**Start verdict margin.** Computed by re-running `best_starting_xi` with the player forced out
of, or into, the XI and taking the difference. This is exact rather than a heuristic on
expected points, and it correctly reports "he starts because the three-defender minimum forces
a defender in" rather than implying he outscored a midfielder.

---

## D. Transfer board

### Contract

```python
@dataclass(frozen=True)
class TransferOption:
    player_out: PlayerCode
    player_in: PlayerCode
    gross_gain: float
    net_gain: float
    hit_cost: int
    risk_penalty: float
    selling_price: TenthsOfMillion
    purchase_price: TenthsOfMillion
    bank_after: TenthsOfMillion
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class TransferBoard:
    top: tuple[TransferOption, ...]  # Global best, default 8
    by_player: tuple[PlayerBestMove, ...]  # One entry per squad player, all 15
    candidates_considered: int
    legal_moves: int
```

`PlayerBestMove` holds either the best legal move for that player or a grounded reason there
is none. All fifteen appear, so a player with no upgrade is visibly accounted for rather than
silently absent — which is the difference between "the model ignored Hato" and "here is Hato's
best available move and it loses by 0.4."

### Reason construction

`_build_reasons` is extracted from `transfer.py` into `explanations/reasons.py` and rewritten
to take feature evidence. It becomes the single place reasons are built for both a transfer
option and a player explanation, so the two can never disagree.

### API

`/recommend/{entry_id}` gains `alternatives: list[TransferOptionOut]` and
`by_player: list[PlayerBestMoveOut]`. The headline recommendation is unchanged — it remains
`top[0]`, or an explicit hold.

---

## E. Feature importance

### Method

Out-of-sample permutation importance, in `packages/evaluation/src/xg_alonso/evaluation/importance.py`.

For each walk-forward fold, each label, and each feature: shuffle that column in the
**validation** rows, re-predict, and record the increase in MAE. Repeated `n_repeats=5` times
with a fixed seed; the mean increase is the importance and the standard deviation is the
stability.

**Why permutation and not a surrogate.** `HistGradientBoosting` has no native importance, and
a surrogate tree measures the surrogate. Permutation measures the deployed model directly.
Running it on validation folds rather than training rows is what makes it *evidence* rather
than a description of the fit — a feature the model memorised will show importance in-sample
and none out.

### Honest reporting

Two properties are surfaced rather than hidden:

- **Correlated features split their importance.** `goals_scored_mean_3` and
  `goals_scored_mean_5` each look weak because either can substitute for the other. Results
  are therefore reported **grouped by family** alongside the flat ranking, and the page says so.
- **Degenerate labels have no meaningful importance.** `ComponentModels.degenerate_labels()`
  already identifies constant-output models. Their importance rows are computed but flagged,
  because ranking features by their effect on a constant is meaningless.

### Aggregation

A single "which features matter most" number weights each label's importance by that label's
contribution to expected points, computed from the pinned scoring rules across the predicted
population. Without that weighting, `label_yellow_cards` — worth −1 point and rare — ranks
alongside `label_minutes`, which gates everything.

### Persistence

`.data/gold/feature_importance.parquet`:

```
feature_name, family, label, fold_index, mae_delta, mae_delta_std,
baseline_mae, rank_within_label, catalogue_version, model_fingerprint, computed_at
```

Versioned by catalogue and model fingerprint so a stale table cannot be served against a
retrained model — the API compares fingerprints and reports staleness rather than silently
showing old numbers.

### CLI

- `xg importance --model .data/models/late.pkl` — standalone run, writes the parquet.
- `xg train --importance` — opt-in flag, runs it as part of training.

Opt-in because the cost is real: 171 features × 9 labels × 5 repeats × folds. Measured on the
existing artifact before defaulting it on.

---

## F. Surfaces

### API

`GET /features/importance?label=&family=&limit=` returns the ranked table with per-label
breakdown, fold stability, degenerate flags, and a `stale: bool` when fingerprints disagree.

### Web

**New page `/features` — the Feature Lab.** Ranked horizontal bars coloured by family; a
label filter (overall, minutes, goals, assists, clean sheets, saves, bonus, cards); a
stability column showing rank variance across folds; degenerate labels flagged inline; and a
plain-language note on why correlated features split.

**Main page changes.**

- `TheCall.tsx` — reasons grouped under the named player they concern, with an OUT/IN marker.
- New `Alternatives` section — the top 8 board, each row expandable to its reasons.
- `Pitch.tsx` and the squad ledger — every player row expands into its `PlayerExplanation`:
  breakdown bar, feature panel with percentile markers, start verdict, top replacements.

---

## G. Testing

| Area | Test |
|---|---|
| Vocabulary | Property test: every `ReasonCode` has a template, and a constructed `Reason` with that code's evidence renders without `KeyError`. |
| Grounding | A `Reason` whose evidence omits a placeholder still fails construction, including for `context` slots. |
| Percentiles | Bounded to `[0, 1]`, monotone in value, computed within position. |
| Evidence | An explanation never cites a numeric value absent from its prediction's evidence or breakdown. |
| Importance | A pure-noise column added to a synthetic frame scores ≈0; a column the label is a function of ranks first. |
| Importance | Degenerate labels are flagged, not ranked. |
| Board | Every returned option passes the same legality checks as `rank_single_transfers`. |
| Board | `by_player` covers all fifteen picks exactly once. |
| Start verdict | For a squad where the 3-DEF minimum forces a low-scoring defender in, the verdict names the constraint. |
| API | `/features/importance` reports `stale: true` when the model fingerprint does not match. |

---

## Files

**New**

- `packages/data_contracts/src/xg_alonso/contracts/evidence.py`
- `packages/explanations/src/xg_alonso/explanations/reasons.py`
- `packages/explanations/src/xg_alonso/explanations/player.py`
- `packages/evaluation/src/xg_alonso/evaluation/importance.py`
- `apps/web/app/features/page.tsx`

**Changed**

- `packages/data_contracts/src/xg_alonso/contracts/reason_codes.py`
- `packages/data_contracts/src/xg_alonso/contracts/prediction.py`
- `packages/data_contracts/src/xg_alonso/contracts/recommendation.py`
- `packages/prediction/src/xg_alonso/prediction/inference.py`
- `packages/prediction/src/xg_alonso/prediction/baseline.py`
- `packages/optimization/src/xg_alonso/optimization/transfer.py`
- `packages/explanations/src/xg_alonso/explanations/render.py`
- `apps/api/src/xg_alonso/api/main.py`
- `apps/api/src/xg_alonso/api/service.py`
- `apps/cli/src/xg_alonso/cli/main.py`
- `apps/web/lib/api.ts`
- `apps/web/components/TheCall.tsx`
- `apps/web/components/Pitch.tsx`
- `apps/web/app/page.tsx`

---

## Ordering

1. §A evidence contract and population — everything else depends on it.
2. §B vocabulary and §D reason extraction — the two are one change.
3. §D board, §C player explanation.
4. §E importance subsystem and CLI.
5. §F API and web surfaces.

Each stage is independently testable, per the repository's testing philosophy.


---

## What changed during implementation

Recorded because the design was written before the code met the data, and three
things it assumed turned out to be wrong.

### The panel needed source aliases

The design assumed one feature set. There are two, and they spell the same
concept differently: the catalogue has `expected_goals_per90_5`, the closed-form
baseline has `xg_per90_shrunk`. Since the API defaults to the baseline — a
deliberate choice, so it cannot silently disagree with the CLI — every attacking
reason fell silent on the surface that actually runs.

`PanelEntry` therefore gained `sources`, an ordered list of columns that supply
the entry. The panel names a *concept*; each feature set resolves it however it
spells it. Panel version moved to `panel_v2`.

### `minutes_mean_3` became `minutes_mean_5`

The slice-1 baseline builds a five-appearance mean, not a three. Aliasing a
three-match label onto a five-match column would have put a false window in the
prose, so the panel entry moved to the window both sets share.

### Rank stability was measured wrongly, twice

The first implementation pooled every label into one ranking and reported the
spread of a feature's rank across *components*. That number was enormous for
every feature — a minutes feature legitimately ranks first for minutes and last
for saves — and rendered as an instability warning on the entire catalogue.

Fixed to rank within `(fold, label)` and compare across folds for the same
label. That exposed a second problem: the CLI measured only the final fold, so
there was nothing to compare and the metric returned zero, which reads as
*perfectly stable* rather than as *never checked*. `xg importance` now measures
every validation window, and `stability()` returns an empty mapping below two
folds so the surfaces can say "not measured" instead of printing a zero.

### The Feature Lab needed a non-linear axis

Importance follows a power law here: `minutes_mean_1` scores 3.5× the next
feature and several hundred times the tail. On a linear axis the top bar filled
its row and the other seventy-nine collapsed into invisibility. Bar lengths are
square-root scaled, order is preserved exactly, and the exact figures stay in a
column beside them.

## Measured result

Against `late.pkl`, 184 features x 9 labels x 6 walk-forward folds:

| Family | Weighted importance |
|---|---|
| player performance | 6.25e-2 |
| player volume | 6.5e-3 |
| player volatility | 5.9e-3 |
| opponent | 4.6e-3 |
| player rate | 3.0e-3 |
| player ceiling | 9.2e-4 |
| player floor | 5.9e-4 |
| fixture | 6.0e-5 |

42 of 184 features did not improve out-of-sample error at all.

`minutes_mean_1` leads by a distance, which is the expected shape — minutes gate
every other component. Note the caveat recorded in `LABEL_TO_BREAKDOWN`: the
label weighting credits minutes only with appearance points, so its true
importance is higher than this method can show. The figure is a floor.
