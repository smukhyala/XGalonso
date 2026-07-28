"use client";

import type { DerivationLine, GameweekProjection } from "@/lib/api";

/**
 * The two things a projection needs beside it: how it was built, and what the
 * weeks after it look like.
 *
 * A single number for a single gameweek is the least useful form the answer
 * takes. It cannot say which part of a player's game produced it — a striker on
 * 5.5 from goals and a defender on 5.5 from clean sheets are different assets —
 * and it cannot say whether next week collapses. Both were computed already and
 * neither reached the board.
 */
export function Derivation({ lines }: { lines: DerivationLine[] }) {
  if (lines.length === 0) return null;
  return (
    <div>
      <p className="eyebrow">Where the points come from</p>
      <ul className="mt-3 space-y-3">
        {lines.map((line) => (
          <li key={line.component}>
            <div className="flex items-baseline gap-3">
              <span className="text-[13px]" style={{ color: "var(--color-chalk)" }}>
                {line.component}
              </span>
              <span className="h-px flex-1" style={{ background: "var(--color-line)" }} />
              <span
                className="tnum text-[13px]"
                style={{ color: line.points >= 0 ? "var(--color-chalk)" : "var(--color-loss)" }}
              >
                {line.points >= 0 ? "+" : ""}
                {line.points.toFixed(2)}
              </span>
            </div>
            {/* The arithmetic, so the number can be checked rather than trusted. */}
            <p className="tnum mt-1 text-[12px]" style={{ color: "var(--color-muted)" }}>
              {formatExpectation(line.expectation)} {line.unit} × {line.rate >= 0 ? "+" : ""}
              {line.rate} pts
            </p>
            {line.note && (
              <p
                className="mt-1 max-w-md text-[12px] leading-relaxed"
                style={{ color: "var(--color-dim)" }}
              >
                {line.note}
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * The weeks ahead, drawn to one scale so the run reads as a shape.
 *
 * Bar height is the projection and opacity is the discount, so a week that
 * counts for less looks like it counts for less. That pairing matters: a big
 * bar five weeks out is worth about half a big bar next week, and a chart that
 * drew them identically would argue for chasing distant fixtures.
 */
export function Horizon({
  weeks,
  total,
}: {
  weeks: GameweekProjection[];
  total: number | null;
}) {
  if (weeks.length === 0) return null;
  const ceiling = Math.max(...weeks.map((w) => w.expected_points), 0.01);

  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="eyebrow">The next {weeks.length} gameweeks</p>
        {total !== null && (
          <p className="eyebrow">
            <span className="tnum">{total.toFixed(2)}</span> discounted
          </p>
        )}
      </div>

      <ul className="mt-4 space-y-2.5">
        {weeks.map((week) => (
          <li key={week.gameweek} className="grid grid-cols-[2.5rem_1fr_auto] items-center gap-3">
            <span className="eyebrow">GW{week.gameweek}</span>
            <span className="flex items-center gap-2.5">
              <span
                className="h-[3px] flex-1"
                style={{ background: "var(--color-line)" }}
              >
                <span
                  className="block h-full"
                  style={{
                    width: `${(week.expected_points / ceiling) * 100}%`,
                    background: "var(--color-chalk)",
                    // Opacity is the discount: a later week counts for less and
                    // should not look like it counts the same.
                    opacity: 0.25 + week.weight * 0.75,
                  }}
                />
              </span>
              <span className="text-[12px] whitespace-nowrap" style={{ color: "var(--color-muted)" }}>
                {week.opponent ?? "no fixture"}
                {week.is_home !== null && (
                  <span style={{ color: "var(--color-dim)" }}>
                    {" "}
                    {week.is_home ? "(H)" : "(A)"}
                  </span>
                )}
              </span>
            </span>
            <span className="tnum text-right text-[13px]">
              {week.expected_points.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>

      <p className="mt-4 max-w-md text-[12px] leading-relaxed" style={{ color: "var(--color-dim)" }}>
        Faded bars count for less. A free transfer arrives every week, so holding him that
        long is a choice rather than a consequence — and a projection five weeks out is a
        longer extrapolation from the same evidence.
      </p>
    </div>
  );
}

function formatExpectation(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude >= 10) return value.toFixed(0);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}
