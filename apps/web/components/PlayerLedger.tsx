"use client";

import { useState } from "react";
import {
  POSITION_COLOR,
  money,
  percentileText,
  type PlayerExplanation,
  type Reason,
  type TransferOption,
} from "@/lib/api";

/**
 * Every squad member, with the case for and against him.
 *
 * The screen this replaces showed fifteen numbers and explained two of them —
 * the pair in the recommended move. So a projected 2.2 sat next to a projected
 * 2.6 with nothing to say why either was what it was, or why one was being sold
 * and the other kept. This is the answer to both, per player, on demand.
 *
 * Two encodings do the work, and each was chosen to match what it represents:
 *
 * - **The breakdown is a stacked bar**, drawn to a scale shared across the
 *   squad. A projection is literally a sum of component points, so a stacked
 *   bar *is* the arithmetic rather than an illustration of it, and a shared
 *   scale means two players' bars can be compared by eye.
 * - **Evidence is a marker on a track**, not a bar. Bar length reads as
 *   magnitude, and these are *ranks* — 0.53 expected goals per 90 means nothing
 *   until you know it is 84th percentile among forwards. A marker on a track
 *   says "position within a field", which is what a percentile is.
 */

const COMPONENTS = [
  { key: "appearance", label: "Appearing", tone: "var(--color-muted)" },
  { key: "goals", label: "Goals", tone: "var(--color-fwd)" },
  { key: "assists", label: "Assists", tone: "var(--color-mid)" },
  { key: "clean_sheets", label: "Clean sheet", tone: "var(--color-def)" },
  { key: "saves", label: "Saves", tone: "var(--color-gkp)" },
  { key: "defensive_contribution", label: "Defensive", tone: "var(--color-def)" },
  { key: "bonus", label: "Bonus", tone: "var(--color-chalk)" },
] as const;

export function PlayerLedger({ players }: { players: PlayerExplanation[] }) {
  const [open, setOpen] = useState<number | null>(null);
  if (players.length === 0) return null;

  // One scale for every bar, so the widths mean the same thing on each row.
  const ceiling = Math.max(...players.map((p) => Math.max(p.breakdown.total, 0.01)));
  const starters = players.filter((p) => p.is_starter);
  const bench = players.filter((p) => !p.is_starter);

  return (
    <section className="rise mt-24" style={{ animationDelay: "0.35s" }}>
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="eyebrow">The case for each player</p>
        <p className="eyebrow">Open a row for the evidence</p>
      </div>
      <div className="hairline mt-5" />

      <Group
        title="Starting"
        players={starters}
        ceiling={ceiling}
        open={open}
        onToggle={(code) => setOpen(open === code ? null : code)}
      />
      <Group
        title="Bench"
        players={bench}
        ceiling={ceiling}
        open={open}
        onToggle={(code) => setOpen(open === code ? null : code)}
      />
    </section>
  );
}

function Group({
  title,
  players,
  ceiling,
  open,
  onToggle,
}: {
  title: string;
  players: PlayerExplanation[];
  ceiling: number;
  open: number | null;
  onToggle: (code: number) => void;
}) {
  if (players.length === 0) return null;
  return (
    <>
      <p className="eyebrow mt-8 mb-1">{title}</p>
      <ul>
        {players.map((player) => (
          <Row
            key={player.player_code}
            player={player}
            ceiling={ceiling}
            isOpen={open === player.player_code}
            onToggle={() => onToggle(player.player_code)}
          />
        ))}
      </ul>
    </>
  );
}

function Row({
  player,
  ceiling,
  isOpen,
  onToggle,
}: {
  player: PlayerExplanation;
  ceiling: number;
  isOpen: boolean;
  onToggle: () => void;
}) {
  const colour = POSITION_COLOR[player.position];
  const upgrade = player.replacements[0];

  return (
    <li className="border-b" style={{ borderColor: "var(--color-line)" }}>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isOpen}
        className="grid w-full grid-cols-[0.75rem_1fr_auto] items-center gap-4 py-3.5 text-left transition-opacity hover:opacity-80 sm:grid-cols-[0.75rem_9rem_1fr_4.5rem_auto]"
      >
        <span
          aria-hidden
          className="h-1.5 w-1.5 rounded-full"
          style={{ background: colour }}
        />

        <span className="flex items-baseline gap-2">
          <span className="text-[15px]">{player.name}</span>
          {player.forced_by_quota && (
            <span className="eyebrow" title="A positional minimum put him in the XI">
              quota
            </span>
          )}
        </span>

        <span className="hidden sm:block">
          <BreakdownBar breakdown={player.breakdown} ceiling={ceiling} />
        </span>

        <span className="tnum hidden text-right text-xs sm:block" style={{ color: "var(--color-dim)" }}>
          {upgrade ? `${upgrade.net_gain >= 0 ? "+" : ""}${upgrade.net_gain.toFixed(2)}` : "—"}
        </span>

        <span className="tnum text-right text-sm">{player.expected_points.toFixed(2)}</span>
      </button>

      {isOpen && <Detail player={player} />}
    </li>
  );
}

function BreakdownBar({
  breakdown,
  ceiling,
}: {
  breakdown: PlayerExplanation["breakdown"];
  ceiling: number;
}) {
  const width = (Math.max(breakdown.total, 0) / ceiling) * 100;
  return (
    <span
      className="flex h-[3px] overflow-hidden"
      style={{ width: `${width}%`, background: "var(--color-line)" }}
    >
      {COMPONENTS.map(({ key, tone }) => {
        const value = breakdown[key];
        if (value <= 0) return null;
        return (
          <span
            key={key}
            className="block h-full"
            style={{
              width: `${(value / Math.max(breakdown.total, 0.01)) * 100}%`,
              background: tone,
              opacity: 0.8,
            }}
          />
        );
      })}
    </span>
  );
}

function Detail({ player }: { player: PlayerExplanation }) {
  const parts = COMPONENTS.map(({ key, label, tone }) => ({
    label,
    tone,
    value: player.breakdown[key],
  })).filter((part) => Math.abs(part.value) >= 0.005);

  return (
    <div className="grid gap-10 pb-9 pt-2 lg:grid-cols-2">
      <div>
        <p className="eyebrow">Where the {player.expected_points.toFixed(2)} comes from</p>
        <dl className="mt-4 space-y-2">
          {parts.map((part) => (
            <div key={part.label} className="flex items-center gap-3">
              <span
                aria-hidden
                className="h-2 w-2 shrink-0"
                style={{ background: part.tone, opacity: 0.8 }}
              />
              <dt className="text-[13px]" style={{ color: "var(--color-muted)" }}>
                {part.label}
              </dt>
              <span className="h-px flex-1" style={{ background: "var(--color-line)" }} />
              <dd className="tnum text-[13px]">{part.value.toFixed(2)}</dd>
            </div>
          ))}
        </dl>

        <p className="eyebrow mt-9">
          {player.is_starter ? "Why he starts" : "Why he sits"}
        </p>
        <p className="mt-3 max-w-md text-[14px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
          {player.is_starter ? (
            player.forced_by_quota ? (
              <>
                The shape needs him. Dropping him costs only{" "}
                <span className="tnum">{player.start_margin.toFixed(2)}</span> points, so it is
                the positional minimum keeping him in the eleven rather than his projection.
              </>
            ) : (
              <>
                Leaving him out would cost the eleven{" "}
                <span className="tnum">{player.start_margin.toFixed(2)}</span> points.
              </>
            )
          ) : (
            <>
              Playing him would cost the eleven{" "}
              <span className="tnum">{Math.abs(player.start_margin).toFixed(2)}</span> points,
              measured by re-picking the side around him.
            </>
          )}
        </p>
      </div>

      <div>
        {player.evidence.length > 0 && (
          <>
            <p className="eyebrow">What sets him apart</p>
            <ul className="mt-4 space-y-3.5">
              {player.evidence.slice(0, 5).map((value) => (
                <Evidence key={value.name} value={value} />
              ))}
            </ul>
          </>
        )}

        <p className="eyebrow mt-9">
          {player.replacements.length > 0 ? "Who could replace him" : "Why nobody replaces him"}
        </p>

        {player.replacements.length > 0 ? (
          <ul className="mt-4 space-y-3">
            {player.replacements.map((option) => (
              <Replacement key={option.player_in} option={option} />
            ))}
          </ul>
        ) : (
          <Reasons reasons={player.no_replacement_reasons} />
        )}

        {player.reasons.length > 0 && (
          <details className="mt-6">
            <summary
              className="eyebrow cursor-pointer transition-opacity hover:opacity-70"
              style={{ listStyle: "none" }}
            >
              Full reasoning
            </summary>
            <Reasons reasons={player.reasons} />
          </details>
        )}
      </div>
    </div>
  );
}

/**
 * One panel feature: the value, and where it sits among comparable players.
 *
 * The track runs worst-to-best *for this player*, which is why the marker
 * position uses the oriented percentile. Volatility ranks high when a player is
 * erratic, and showing that as a marker near the good end would invert the
 * argument the number is making.
 */
function Evidence({ value }: { value: PlayerExplanation["evidence"][number] }) {
  const oriented =
    value.percentile === null
      ? null
      : value.higher_is_better
        ? value.percentile
        : 1 - value.percentile;
  const rank = percentileText(value.percentile);

  return (
    <li>
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-[13px]" style={{ color: "var(--color-muted)" }}>
          {value.label}
        </span>
        <span className="tnum text-[13px]">
          {value.value === null ? "not measured" : value.value.toFixed(2)}
        </span>
      </div>
      {oriented !== null && (
        <div className="mt-1.5 flex items-center gap-3">
          <span className="relative h-[2px] flex-1" style={{ background: "var(--color-line)" }}>
            <span
              className="absolute top-1/2 h-2.5 w-[2px] -translate-y-1/2"
              style={{
                left: `${oriented * 100}%`,
                background: oriented >= 0.5 ? "var(--color-gain)" : "var(--color-loss)",
              }}
            />
          </span>
          <span className="eyebrow shrink-0">{rank}</span>
        </div>
      )}
    </li>
  );
}

function Replacement({ option }: { option: TransferOption }) {
  const price = option.purchase_price - option.selling_price;
  return (
    <li>
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-[14px]">{option.player_in_name}</span>
        <span
          className="tnum text-[13px]"
          style={{ color: option.net_gain >= 0 ? "var(--color-gain)" : "var(--color-loss)" }}
        >
          {option.net_gain >= 0 ? "+" : ""}
          {option.net_gain.toFixed(2)} pts
        </span>
      </div>
      <p className="eyebrow mt-1">
        {price === 0 ? "same price" : `${price > 0 ? "costs" : "frees"} ${money(Math.abs(price))}`}
        {option.hit_cost > 0 && ` · −${option.hit_cost} hit`}
      </p>
    </li>
  );
}

function Reasons({ reasons }: { reasons: Reason[] }) {
  if (reasons.length === 0) return null;
  return (
    <ul className="mt-4 space-y-2.5">
      {reasons.map((reason) => (
        <li
          key={`${reason.code}-${reason.subject}`}
          className="text-[13px] leading-relaxed"
          style={{ color: "var(--color-muted)" }}
        >
          {reason.text}
        </li>
      ))}
    </ul>
  );
}
