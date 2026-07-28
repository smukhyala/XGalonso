"use client";

import { useCallback, useEffect, useState } from "react";
import { Pitch } from "@/components/Pitch";
import { TheCall } from "@/components/TheCall";
import {
  api,
  money,
  POSITION_COLOR,
  type Health,
  type PlayerSummary,
  type Recommendation,
  type SquadResponse,
} from "@/lib/api";

// Your real GW1 squad, transcribed from the Pick Team screen. Swap back to
// example_squad.json for the synthetic fixture. Once the GW1 deadline passes,
// `entry/{id}/picks/` starts returning data and the Team ID box takes over.
const SAMPLE_SQUAD = "data/samples/my_squad.json";

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [squad, setSquad] = useState<SquadResponse | null>(null);
  const [call, setCall] = useState<Recommendation | null>(null);
  const [board, setBoard] = useState<PlayerSummary[]>([]);
  const [entryId, setEntryId] = useState("1234567");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  const load = useCallback(async (id: string) => {
    setBusy(true);
    setError(null);
    try {
      const [squadResult, callResult] = await Promise.all([
        api.squad(Number(id), SAMPLE_SQUAD),
        api.recommend(Number(id), SAMPLE_SQUAD),
      ]);
      setSquad(squadResult);
      setCall(callResult);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.players(12).then(setBoard).catch(() => setBoard([]));
    void load(entryId);
    // Intentionally runs once: the entry id is re-submitted explicitly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="mx-auto max-w-6xl px-6 pb-28 sm:px-10">
      <Masthead health={health} entryId={entryId} onEntryId={setEntryId} onSubmit={() => load(entryId)} />

      {error && (
        <p
          role="alert"
          className="mt-10 max-w-2xl text-[15px] leading-relaxed"
          style={{ color: "var(--color-loss)" }}
        >
          {error}
        </p>
      )}

      <div className="mt-14 grid gap-16 lg:grid-cols-[1.05fr_0.95fr] lg:gap-14">
        <div>
          {busy && !call ? <Skeleton /> : call && <TheCall recommendation={call} />}
          {squad && <Ledger squad={squad} />}
        </div>

        {squad && (
          <div className="rise" style={{ animationDelay: "0.2s" }}>
            <div className="flex items-baseline justify-between">
              <p className="eyebrow">The eleven</p>
              <p className="eyebrow">{squad.formation}</p>
            </div>
            <div className="mt-5">
              <Pitch players={squad.players} />
            </div>
          </div>
        )}
      </div>

      {board.length > 0 && <Board players={board} />}

      {squad && <Footnote squad={squad} />}
    </main>
  );
}

function Masthead({
  health,
  entryId,
  onEntryId,
  onSubmit,
}: {
  health: Health | null;
  entryId: string;
  onEntryId: (value: string) => void;
  onSubmit: () => void;
}) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-6 pt-10">
      <div className="flex items-baseline gap-4">
        <span
          style={{
            fontFamily: "var(--font-display)",
            fontWeight: 700,
            fontSize: "1.0625rem",
            letterSpacing: "0.02em",
          }}
        >
          XG ALONSO
        </span>
        {health && (
          <span className="eyebrow">
            {health.season} · GW{health.next_gameweek}
            {health.stale && <span style={{ color: "var(--color-loss)" }}> · snapshot is stale</span>}
          </span>
        )}
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
        className="flex items-center gap-2"
      >
        <label htmlFor="entry" className="eyebrow">
          Team ID
        </label>
        <input
          id="entry"
          value={entryId}
          onChange={(event) => onEntryId(event.target.value.replace(/\D/g, ""))}
          inputMode="numeric"
          className="tnum w-28 border-b bg-transparent pb-1 text-sm outline-none transition-colors"
          style={{ borderColor: "var(--color-line)" }}
        />
        <button
          type="submit"
          className="text-sm transition-opacity hover:opacity-70"
          style={{ color: "var(--color-mid)" }}
        >
          Load
        </button>
      </form>
    </header>
  );
}

function Ledger({ squad }: { squad: SquadResponse }) {
  return (
    <section className="rise mt-14" style={{ animationDelay: "0.15s" }}>
      <div className="hairline" />
      <dl className="mt-6 flex flex-wrap gap-x-10 gap-y-5">
        <Stat label="Projected" value={squad.projected_points.toFixed(2)} unit="pts" />
        <Stat label="Squad value" value={money(squad.squad_value)} />
        <Stat label="Bank" value={money(squad.bank)} />
        <Stat label="Free transfers" value={String(squad.free_transfers)} />
      </dl>
      {squad.prices_assumed && (
        <p className="mt-5 max-w-lg text-[13px] leading-relaxed text-dim">
          Purchase prices were assumed equal to current prices, so the budget shown is a
          lower bound. Real prices come from your transfer history once the season starts.
        </p>
      )}
    </section>
  );
}

function Stat({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className="tnum mt-1.5 text-lg" style={{ letterSpacing: "-0.02em" }}>
        {value}
        {unit && <span className="ml-1 text-[0.65em] text-dim">{unit}</span>}
      </dd>
    </div>
  );
}

function Board({ players }: { players: PlayerSummary[] }) {
  const ceiling = Math.max(...players.map((p) => p.expected_points));
  return (
    <section className="rise mt-24" style={{ animationDelay: "0.3s" }}>
      <div className="flex items-baseline justify-between">
        <p className="eyebrow">Best available</p>
        <p className="eyebrow hidden sm:block">Projected points · this gameweek</p>
      </div>
      <div className="hairline mt-5" />

      <ol className="mt-2">
        {players.map((player, index) => (
          <li
            key={player.player_code}
            className="grid grid-cols-[2.5rem_1fr_auto] items-center gap-4 border-b py-3.5 sm:grid-cols-[2.5rem_1fr_7rem_5rem_auto]"
            style={{ borderColor: "var(--color-line)" }}
          >
            <span className="tnum text-xs text-dim">{String(index + 1).padStart(2, "0")}</span>

            <span className="flex items-center gap-3">
              <span
                aria-hidden
                className="h-1.5 w-1.5 rounded-full"
                style={{ background: POSITION_COLOR[player.position] }}
              />
              <span className="text-[15px]">{player.name}</span>
              <span className="eyebrow hidden sm:inline">{player.position}</span>
            </span>

            {/* A bar, not a number in a box — the comparison is the point. */}
            <span className="hidden h-[3px] sm:block" style={{ background: "var(--color-line)" }}>
              <span
                className="block h-full"
                style={{
                  width: `${(player.expected_points / ceiling) * 100}%`,
                  background: POSITION_COLOR[player.position],
                  opacity: 0.75,
                }}
              />
            </span>

            <span className="tnum hidden text-sm text-muted sm:block">{money(player.price)}</span>

            <span className="tnum text-right text-sm">{player.expected_points.toFixed(2)}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Footnote({ squad }: { squad: SquadResponse }) {
  const { provenance } = squad;
  return (
    <footer className="mt-24">
      <div className="hairline" />
      <p className="eyebrow mt-6 leading-relaxed">
        {provenance.model_name} v{provenance.model_version} · {provenance.feature_set_version} ·
        data cutoff {new Date(provenance.data_cutoff).toISOString().slice(0, 16).replace("T", " ")}Z
        · run {provenance.run_id}
      </p>
      <p className="mt-3 max-w-2xl text-[13px] leading-relaxed text-dim">
        Every figure here traces to a model version, a feature set and a data cutoff. Nothing
        is stated that the evidence behind it cannot support.
      </p>
    </footer>
  );
}

function Skeleton() {
  return (
    <div className="space-y-4" aria-busy>
      <div className="h-3 w-24 animate-pulse" style={{ background: "var(--color-line)" }} />
      <div className="h-14 w-3/4 animate-pulse" style={{ background: "var(--color-line)" }} />
      <div className="h-14 w-2/3 animate-pulse" style={{ background: "var(--color-line)" }} />
    </div>
  );
}
