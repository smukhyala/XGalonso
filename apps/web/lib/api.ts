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
  provenance: Provenance;
}

export interface FeatureImportance {
  feature_name: string;
  family: string;
  importance: number;
  rank_stability: number | null;
  per_label: Record<string, number>;
}

export interface ImportanceResponse {
  features: FeatureImportance[];
  families: Record<string, number>;
  degenerate_labels: string[];
  labels: string[];
  label_weights: Record<string, number>;
  folds_measured: number;
  features_measured: number;
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

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`, { cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
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
  importance: (label?: string, limit = 60) =>
    get<ImportanceResponse>(
      `/features/importance?limit=${limit}${label ? `&label=${encodeURIComponent(label)}` : ""}`,
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
