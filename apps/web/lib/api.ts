/**
 * Typed client for the decision API.
 *
 * Requests go through the Next rewrite to the local FastAPI, so the browser
 * stays same-origin and there is no public API surface (D1: local-only).
 */

export type Position = "GKP" | "DEF" | "MID" | "FWD";

export interface Provenance {
  model_name: string;
  model_version: string;
  feature_set_version: string;
  data_cutoff: string;
  generated_at: string;
  run_id: string;
}

export interface HistoryNote {
  kind: "opponent" | "gameweek" | "venue";
  text: string;
  is_positive: boolean;
}

export interface DerivationLine {
  component: string;
  expectation: number;
  unit: string;
  rate: number;
  points: number;
  note: string;
  sentence: string;
}

export interface GameweekProjection {
  gameweek: number;
  opponent: string | null;
  is_home: boolean | null;
  expected_points: number;
  weight: number;
}

export interface SeasonLine {
  season: string;
  appearances: number;
  minutes: number;
  goals: number;
  assists: number;
  clean_sheets: number;
  points: number;
  expected_goals: number | null;
  expected_assists: number | null;
  /** Null means the sample cannot support a rate — not that the rate is zero. */
  per_90: number | null;
  points_per_appearance: number | null;
  sentence: string;
}

export interface ScheduledFixture {
  gameweek: number;
  opponent: string;
  is_home: boolean;
  /** FPL's published 1-5 rating. Null preseason, when it is unset. */
  difficulty: number | null;
  label: string;
}

export interface FixtureRun {
  fixtures: ScheduledFixture[];
  mean_difficulty: number | null;
  home_count: number;
  blanks: number[];
  doubles: number[];
  sentence: string;
}

/** Retrieved, never modelled — this is what makes it checkable. */
export interface PlayerContext {
  seasons: SeasonLine[];
  run: FixtureRun | null;
}

export interface PlayerSummary {
  player_code: number;
  name: string;
  position: Position;
  team_id: number;
  price: number;
  status: string | null;
  expected_points: number;
  expected_points_sd: number;
  p_start: number;
  expected_minutes: number;
  history: HistoryNote[];
  derivation: DerivationLine[];
  horizon: GameweekProjection[];
  horizon_total: number | null;
  context: PlayerContext | null;
}

export interface SquadPlayer extends PlayerSummary {
  squad_slot: number;
  is_captain: boolean;
  is_vice_captain: boolean;
  is_starter: boolean;
  selling_price: number;
  purchase_price: number;
}

export interface SquadResponse {
  entry_id: number;
  gameweek: number;
  formation: string;
  squad_value: number;
  bank: number;
  free_transfers: number;
  projected_points: number;
  prices_assumed: boolean;
  players: SquadPlayer[];
  provenance: Provenance;
}

export type Polarity = "supports_in" | "supports_out" | "context";

export interface Reason {
  code: string;
  text: string;
  subject: number;
  subject_name: string;
  polarity: Polarity;
  weight: number;
}

export interface FeatureValue {
  name: string;
  label: string;
  family: string;
  value: number | null;
  percentile: number | null;
  higher_is_better: boolean;
}

export interface Breakdown {
  appearance: number;
  goals: number;
  assists: number;
  clean_sheets: number;
  goals_conceded: number;
  saves: number;
  cards: number;
  defensive_contribution: number;
  bonus: number;
  total: number;
}

export interface TransferOption {
  player_out: number;
  player_out_name: string;
  player_in: number;
  player_in_name: string;
  position: Position;
  selling_price: number;
  purchase_price: number;
  gross_gain: number;
  net_gain: number;
  hit_cost: number;
  risk_penalty: number;
  bank_after: number;
  reasons: Reason[];
  history_in: HistoryNote[];
  history_out: HistoryNote[];
}

export interface Comparable {
  player_code: number;
  name: string;
  expected_points: number;
  price: number | null;
}

export interface Archetype {
  label: string;
  size: number;
  rank_within: number;
  comparables: Comparable[];
  caveat: string;
}

export interface Swap {
  position: Position;
  player_in: number | null;
  player_in_name: string | null;
  player_out: number | null;
  player_out_name: string | null;
  points_in: number;
  points_out: number;
  delta: number;
  is_like_for_like: boolean;
}

export interface LineupComparison {
  yours_points: number;
  ours_points: number;
  total_delta: number;
  swap_delta: number;
  captain_delta: number;
  shape_delta: number;
  yours_formation: string;
  ours_formation: string;
  yours_captain_name: string | null;
  ours_captain_name: string | null;
  swaps: Swap[];
  is_identical: boolean;
  yours_is_better: boolean;
}

export interface PlayerExplanation {
  player_code: number;
  name: string;
  position: Position;
  expected_points: number;
  breakdown: Breakdown;
  evidence: FeatureValue[];
  reasons: Reason[];
  is_starter: boolean;
  start_margin: number;
  forced_by_quota: boolean;
  legal_replacements: number;
  replacements: TransferOption[];
  no_replacement_reasons: Reason[];
  archetype: Archetype | null;
  history: HistoryNote[];
  derivation: DerivationLine[];
  derivation_reconciles: boolean;
}

export interface SquadBuild {
  gameweek: number;
  formation: string;
  squad_value: number;
  bank: number;
  projected_points: number;
  players: SquadPlayer[];
  explanations: PlayerExplanation[];
  candidates_considered: number;
  provenance: Provenance;
}

export interface Recommendation {
  entry_id: number;
  gameweek: number;
  is_hold: boolean;
  player_out: PlayerSummary | null;
  player_in: PlayerSummary | null;
  hit_cost: number;
  bank_after: number;
  projected_hold: number;
  projected_after: number;
  expected_gain: number;
  risk: number;
  reasons: Reason[];
  alternatives: TransferOption[];
  players: PlayerExplanation[];
  candidates_considered: number;
  legal_moves: number;
  lineup: LineupComparison | null;
  provenance: Provenance;
}

export interface FeatureImportance {
  feature_name: string;
  family: string;
  importance: number;
  rank_stability: number | null;
  per_label: Record<string, number>;
}



// --- Requirements and planning -------------------------------------------

export type RequirementKind =
  | "must_start"
  | "must_include"
  | "must_exclude"
  | "must_captain"
  | "club_floor"
  | "club_ceiling"
  | "formation"
  | "bank_floor";

export interface Requirement {
  kind: RequirementKind;
  label: string;
  players: number[];
  player_names: string[];
  team_id: number | null;
  count: number | null;
  formation: string | null;
  amount: number | null;
  priority: number;
  /** How sure the parser was. Null when supplied directly rather than parsed. */
  confidence: number | null;
  /** The phrase that produced it — so you can see why it thinks you asked. */
  evidence: string;
  /** "matched" by a vocabulary rule, or "model" if a language model read it. */
  source: "matched" | "model";
}

/** A requirement sent back after editing. Replaces the parse, never adds to it. */
export interface RequirementInput {
  kind: RequirementKind;
  players?: number[];
  team_id?: number | null;
  count?: number | null;
  formation?: string | null;
  amount?: number | null;
  priority?: number;
}

export interface ParsedRequirements {
  objective_id: string;
  requirements: Requirement[];
  unparsed: string[];
  unresolved_names: string[];
  /** Contradictions found without solving, e.g. four players from one club. */
  problems: string[];
  overall_confidence: number;
  interpreted: boolean;
  /** Why the model was or was not consulted. Shown, never swallowed. */
  interpreter_note: string;
  /** A lean read from the request, e.g. "differential". Not a requirement. */
  ownership_preference: string;
  risk_preference: string;
  /** Understood but not expressible as a requirement. */
  model_notes: string[];
}

export interface RequirementOutcome {
  kind: RequirementKind;
  label: string;
  honoured: boolean;
  /** Points given up, holding the others fixed. Zero means it cost nothing. */
  cost: number | null;
  note: string;
}

export interface Plan {
  objective_id: string;
  model_note: string;
  gameweek: number;
  formation: string;
  bank: number;
  expected_points: number;
  unconstrained_points: number;
  total_cost: number;
  feasible_as_asked: boolean;
  players: SquadPlayer[];
  outcomes: RequirementOutcome[];
  parsed: ParsedRequirements | null;
}

// --- Objective-conditioned discovery -------------------------------------

export interface ObjectivePreset {
  id: string;
  name: string;
  primary_metric: string;
  risk_preference: string;
  planning_horizon: number;
  ownership_preference: string;
}

export interface DiscoveredFeature {
  feature: string;
  version: string;
  hypothesis_id: string;
  status: string;
  complementarity: string;
  utility: number;
  incremental_value: number;
  folds: number;
  folds_improved: number;
  stability: number;
  missingness: number;
  leakage_passed: boolean;
  reason: string;
}

export interface Hypothesis {
  id: string;
  title: string;
  football_rationale: string;
  falsification_condition: string;
  expected_relationship: string;
  generation_source: string;
  leakage_risk: string;
  status: string;
  required_raw_fields: string[];
}

export interface Cluster {
  cluster_model_version: string;
  cluster_id: number;
  objective_id: string;
  size: number;
  label: string;
  dominant_features: [string, number][];
}

export interface Experiment {
  experiment_id: string;
  objective_id: string | null;
  stage: string | null;
  hypotheses_proposed: number;
  features_compiled: number;
  features_accepted: number;
  features_rejected: number;
  code_version: string | null;
  git_dirty: boolean;
  completed_at: string | null;
  metrics: [string, number][];
}

export interface ImportanceResponse {
  features: FeatureImportance[];
  families: Record<string, number>;
  degenerate_labels: string[];
  labels: string[];
  label_weights: Record<string, number>;
  folds_measured: number;
  features_measured: number;
  /** Which slice these numbers describe. `ALL` is the pooled measurement. */
  position: string;
  /** Slices present in the table. A pre-v2 table offers `ALL` only. */
  positions: string[];
  /** Validation rows behind this slice. A positional slice is much smaller. */
  rows_measured: number;
  features_with_no_effect: number;
  catalogue_version: string;
  model_fingerprint: string;
  computed_at: string;
  stale: boolean;
}

export interface Health {
  status: string;
  season: string;
  next_gameweek: number;
  deadline: string;
  players_loaded: number;
  history_rows: number;
  model_loaded: boolean;
  stale: boolean;
}

/** How long any one call may stall before it is treated as a failure.
 *
 * Generous rather than tight: `/recommend` rebuilds the whole feature frame and
 * legitimately takes a second or two on a cold cache, and a timeout that fires
 * on a slow-but-working request is worse than none at all.
 *
 * The reason there is a ceiling: the page fetches once on mount and does not
 * retry, and `busy` is cleared in a `finally`. A request that never settles
 * therefore never clears it, and the UI sits on a skeleton with no error and no
 * way forward — the failure looks identical to "still loading", which is the
 * one thing a viewer cannot distinguish. A rejection is recoverable; a hang is
 * not.
 */
const REQUEST_TIMEOUT_MS = 30_000;

/** `fetch`, but a stall becomes a rejection with a message worth reading. */
async function withTimeout(path: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(`/api${path}`, {
      ...init,
      cache: "no-store",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (cause) {
    // A DOMException named TimeoutError is what AbortSignal.timeout throws;
    // anything else here is the API being unreachable, which reads the same to
    // a user and needs the same instruction.
    const timedOut = cause instanceof DOMException && cause.name === "TimeoutError";
    throw new Error(
      timedOut
        ? `The API did not respond within ${REQUEST_TIMEOUT_MS / 1000}s. Is it still running?`
        : "Could not reach the API on 127.0.0.1:8000. Start it with `make api`.",
    );
  }
}

/** The server's `detail` when there is one, and a useful guess when there is not.
 *
 * FastAPI returns `{"detail": ...}` for every error it raises itself, and that
 * text is always better than anything invented here — "the picks endpoint
 * returns 404 until that gameweek's deadline" tells a manager exactly what to
 * do. The fallback matters because the most common failure in local use is
 * `make web` without `make api`: the Next proxy answers 5xx with an HTML error
 * page, `detail` parses to nothing, and the honest reading of that pair is that
 * the upstream is not there.
 */
async function failureFrom(response: Response): Promise<Error> {
  const body = await response.json().catch(() => null);
  const detail = body?.detail;
  if (typeof detail === "string" && detail) {
    return new Error(detail);
  }
  if (response.status >= 500) {
    return new Error("Could not reach the API on 127.0.0.1:8000. Start it with `make api`.");
  }
  return new Error(response.statusText || `Request failed (${response.status})`);
}

async function get<T>(path: string): Promise<T> {
  const response = await withTimeout(path, {});
  if (!response.ok) {
    throw await failureFrom(response);
  }
  return response.json() as Promise<T>;
}

/** POST helper. Same error discipline as `get`: the server's detail wins. */
async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await withTimeout(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw await failureFrom(response);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => get<Health>("/health"),
  players: (limit = 20) => get<PlayerSummary[]>(`/players?limit=${limit}`),
  buildSquad: () => get<SquadResponse>("/build-squad"),
  squad: (entryId: number, squadFile?: string) =>
    get<SquadResponse>(
      `/squad/${entryId}${squadFile ? `?squad_file=${encodeURIComponent(squadFile)}` : ""}`,
    ),
  recommend: (entryId: number, squadFile?: string) =>
    get<Recommendation>(
      `/recommend/${entryId}${squadFile ? `?squad_file=${encodeURIComponent(squadFile)}` : ""}`,
    ),
  buildSquadExplained: () => get<SquadBuild>("/build-squad/explained"),
  parseRequirements: (text: string, preset = "expected_points", interpret = false) =>
    post<ParsedRequirements>("/requirements/parse", { text, preset, interpret }),

  plan: (body: {
    text: string;
    preset?: string;
    requirements?: RequirementInput[];
    interpret?: boolean;
  }) => post<Plan>("/squad/plan", body),

  objectives: () => get<ObjectivePreset[]>("/objectives"),

  experiments: () => get<Experiment[]>("/experiments"),

  discoveredFeatures: (objectiveId: string) =>
    get<DiscoveredFeature[]>(
      `/features/discovered?objective_id=${encodeURIComponent(objectiveId)}`,
    ),

  hypotheses: () => get<Hypothesis[]>("/hypotheses"),

  clusters: (objectiveId?: string) =>
    get<Cluster[]>(
      `/clusters${objectiveId ? `?objective_id=${encodeURIComponent(objectiveId)}` : ""}`,
    ),

  importance: (label?: string, limit = 60, position?: string) =>
    get<ImportanceResponse>(
      `/features/importance?limit=${limit}` +
        `${label ? `&label=${encodeURIComponent(label)}` : ""}` +
        `${position ? `&position=${encodeURIComponent(position)}` : ""}`,
    ),
};

/** Component labels read as `label_goals_scored`; people do not. */
export function labelName(label: string): string {
  return label.replace(/^label_/, "").replace(/_/g, " ");
}

/** A percentile as the phrase a person would say, not a decimal. */
export function percentileText(percentile: number | null): string | null {
  if (percentile === null) return null;
  return `${Math.round(percentile * 100)}${suffix(Math.round(percentile * 100))}`;
}

function suffix(value: number): string {
  if (value % 100 >= 11 && value % 100 <= 13) return "th";
  if (value % 10 === 1) return "st";
  if (value % 10 === 2) return "nd";
  if (value % 10 === 3) return "rd";
  return "th";
}

/**
 * Importance as a percentage a person can read.
 *
 * These values are a fraction of the model's baseline error, so a percentage is
 * the literal reading: 0.036 means shuffling the feature made the model 3.6%
 * worse. Exponent notation said the same thing and communicated nothing —
 * "3.6e-2" next to "5.9e-4" asks a reader to compare mantissas and exponents in
 * their head, which is exactly the arithmetic a chart exists to avoid.
 *
 * Small values keep two significant figures rather than rounding to "0.0%",
 * because the tail of this ranking is where the "did this feature do anything"
 * question actually lives.
 */
export function importanceText(value: number): string {
  const percent = value * 100;
  const magnitude = Math.abs(percent);
  if (magnitude === 0) return "0%";
  if (magnitude < 0.01) return `${percent < 0 ? "−" : ""}<0.01%`;
  const decimals = magnitude >= 10 ? 1 : magnitude >= 1 ? 2 : 3;
  return `${percent.toFixed(decimals)}%`;
}

/** Money is stored in tenths of a million everywhere in this system. */
export function money(tenths: number): string {
  return `£${(tenths / 10).toFixed(1)}m`;
}

export const POSITION_COLOR: Record<Position, string> = {
  GKP: "var(--color-gkp)",
  DEF: "var(--color-def)",
  MID: "var(--color-mid)",
  FWD: "var(--color-fwd)",
};
