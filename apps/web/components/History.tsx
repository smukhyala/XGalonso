"use client";

import type { HistoryNote } from "@/lib/api";

/**
 * What a player has actually done in this situation.
 *
 * Set apart from the modelled reasons deliberately. Everything else on a player
 * row is derived — a projection, a percentile, a component — and says *how
 * much* without ever saying *when*. These are match results, with the season
 * and the score line attached, so they are the one part of the explanation a
 * reader can check against a scoreboard rather than take on trust.
 *
 * The marker carries the direction, so a strong record and a poor one are
 * distinguishable before the sentence is read. Nothing here feeds a projection
 * and the component says so where it matters.
 */
export function History({
  notes,
  compact = false,
}: {
  notes: HistoryNote[];
  compact?: boolean;
}) {
  if (notes.length === 0) return null;

  return (
    <ul className={compact ? "space-y-2" : "space-y-3"}>
      {notes.map((note) => (
        <li key={`${note.kind}-${note.text.slice(0, 24)}`} className="flex gap-3">
          <span
            aria-hidden
            className="mt-[0.5rem] h-1.5 w-1.5 shrink-0 rounded-full"
            style={{
              background: note.is_positive ? "var(--color-gain)" : "var(--color-loss)",
              opacity: 0.85,
            }}
          />
          <span
            className={compact ? "text-[13px] leading-relaxed" : "text-[14px] leading-relaxed"}
            style={{ color: "var(--color-muted)" }}
          >
            {note.text}
          </span>
        </li>
      ))}
    </ul>
  );
}
