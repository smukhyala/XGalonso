"use client";

import { useState } from "react";
import { POSITION_COLOR, money, type Reason, type TransferOption } from "@/lib/api";

/**
 * The moves the recommendation beat.
 *
 * The optimizer has always scored every legal transfer and returned them
 * ranked; the product kept the first row and discarded the rest. A single
 * option with nothing behind it cannot be disagreed with intelligently — you
 * cannot tell whether it won by a distance or by a rounding error, and you
 * cannot see the move you were personally considering.
 *
 * Ordered by net gain, with the bar drawn against the best. A move that nearly
 * won should look like it nearly won.
 */
export function Alternatives({ options }: { options: TransferOption[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (options.length === 0) return null;

  const best = Math.max(...options.map((o) => Math.abs(o.net_gain)), 0.01);

  return (
    <section className="rise mt-24" style={{ animationDelay: "0.3s" }}>
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="eyebrow">What it beat</p>
        <p className="eyebrow">Net gain after hit and uncertainty</p>
      </div>
      <div className="hairline mt-5" />

      <ul className="mt-2">
        {options.map((option) => {
          const key = `${option.player_out}-${option.player_in}`;
          const isOpen = open === key;
          const colour = POSITION_COLOR[option.position] ?? "var(--color-muted)";

          return (
            <li key={key} className="border-b" style={{ borderColor: "var(--color-line)" }}>
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : key)}
                aria-expanded={isOpen}
                className="grid w-full grid-cols-[1fr_auto] items-center gap-4 py-3.5 text-left transition-opacity hover:opacity-80 sm:grid-cols-[1fr_8rem_5rem]"
              >
                <span className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
                  <span className="text-[15px]" style={{ color: "var(--color-muted)" }}>
                    {option.player_out_name}
                  </span>
                  <span aria-hidden style={{ color: "var(--color-dim)" }}>
                    →
                  </span>
                  <span className="text-[15px]" style={{ color: colour }}>
                    {option.player_in_name}
                  </span>
                  {option.hit_cost > 0 && (
                    <span className="eyebrow" style={{ color: "var(--color-loss)" }}>
                      −{option.hit_cost} hit
                    </span>
                  )}
                </span>

                <span className="hidden h-[3px] sm:block" style={{ background: "var(--color-line)" }}>
                  <span
                    className="block h-full"
                    style={{
                      width: `${(Math.abs(option.net_gain) / best) * 100}%`,
                      background: option.net_gain >= 0 ? "var(--color-gain)" : "var(--color-loss)",
                      opacity: 0.7,
                    }}
                  />
                </span>

                <span
                  className="tnum text-right text-sm"
                  style={{
                    color: option.net_gain >= 0 ? "var(--color-gain)" : "var(--color-loss)",
                  }}
                >
                  {option.net_gain >= 0 ? "+" : ""}
                  {option.net_gain.toFixed(2)}
                </span>
              </button>

              {isOpen && <Detail option={option} />}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Detail({ option }: { option: TransferOption }) {
  const cost = option.purchase_price - option.selling_price;
  const argued = option.reasons.filter((r) => r.polarity !== "context");

  return (
    <div className="grid gap-8 pb-8 pt-1 sm:grid-cols-[1fr_1.4fr]">
      <dl className="space-y-2">
        <Line label="Sell" value={money(option.selling_price)} />
        <Line label="Buy" value={money(option.purchase_price)} />
        <Line
          label={cost >= 0 ? "Costs" : "Frees"}
          value={money(Math.abs(cost))}
        />
        <Line label="Bank after" value={money(option.bank_after)} />
        <Line label="Before uncertainty" value={`${option.gross_gain.toFixed(2)} pts`} />
        <Line label="Uncertainty" value={`−${option.risk_penalty.toFixed(2)} pts`} />
      </dl>

      <div>
        {argued.length > 0 ? (
          <ul className="space-y-2.5">
            {argued.map((reason: Reason) => (
              <li
                key={`${reason.code}-${reason.subject}`}
                className="text-[13px] leading-relaxed"
                style={{ color: "var(--color-muted)" }}
              >
                <span style={{ color: "var(--color-chalk)" }}>{reason.subject_name}</span>{" "}
                — {reason.text}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[13px]" style={{ color: "var(--color-dim)" }}>
            This move gains points without any single statistic standing out.
          </p>
        )}
      </div>
    </div>
  );
}

function Line({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-3">
      <dt className="eyebrow">{label}</dt>
      <span className="h-px flex-1" style={{ background: "var(--color-line)" }} />
      <dd className="tnum text-[13px]">{value}</dd>
    </div>
  );
}
