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

export interface Reason {
  code: string;
  text: string;
  subject: number;
  weight: number;
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
  provenance: Provenance;
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
};

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
