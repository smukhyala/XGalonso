"use client";

import { POSITION_COLOR, money, type Recommendation } from "@/lib/api";

/**
 * The recommendation, stated as a sentence.
 *
 * The reflexive answer here is a big number in a card with a small label. That
 * would be wrong for this product: the number is not the point, the *decision*
 * is, and a manager reading it needs to know what to do before they need to
 * know by how much. So the headline is the instruction, the arithmetic sits
 * beneath it, and the evidence sits beneath that — in the order a person
 * actually asks for them.
 *
 * A hold is rendered with the same weight as a transfer. Most gameweeks the
 * right move is to do nothing, and a product that renders "hold" apologetically
 * teaches its user to ignore it.
 */
export function TheCall({ recommendation }: { recommendation: Recommendation }) {
  const { player_out: out, player_in: incoming } = recommendation;
  const gain = recommendation.expected_gain;

  if (recommendation.is_hold || !out || !incoming) {
    return (
      <section className="rise" style={{ animationDelay: "0.1s" }}>
        <p className="eyebrow">The call · GW{recommendation.gameweek}</p>
        <h1
          className="mt-4 text-balance"
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2.25rem, 6vw, 4rem)",
            fontWeight: 700,
            lineHeight: 0.98,
            letterSpacing: "-0.03em",
          }}
        >
          Keep your transfer.
        </h1>
        <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-muted">
          Every legal transfer was evaluated against your current eleven. None gained
          enough to be worth spending on, so the recommendation is to roll it.
        </p>
        <p className="tnum mt-6 text-sm text-dim">
          Projected {recommendation.projected_hold.toFixed(2)} pts either way
        </p>
      </section>
    );
  }

  return (
    <section className="rise" style={{ animationDelay: "0.1s" }}>
      <p className="eyebrow">The call · GW{recommendation.gameweek}</p>

      <h1
        className="mt-4 text-balance"
        style={{
          fontFamily: "var(--font-display)",
          fontSize: "clamp(2.25rem, 6vw, 4rem)",
          fontWeight: 700,
          lineHeight: 0.98,
          letterSpacing: "-0.03em",
        }}
      >
        Sell{" "}
        <span style={{ color: POSITION_COLOR[out.position] }}>{out.name}</span>.
        <br />
        Buy{" "}
        <span style={{ color: POSITION_COLOR[incoming.position] }}>{incoming.name}</span>.
      </h1>

      <dl className="mt-8 flex flex-wrap items-baseline gap-x-10 gap-y-4">
        <Figure
          label="Expected gain"
          value={`${gain >= 0 ? "+" : ""}${gain.toFixed(2)}`}
          unit="pts"
          tone={gain >= 0 ? "gain" : "loss"}
          emphasis
        />
        <Figure label="Hold" value={recommendation.projected_hold.toFixed(2)} unit="pts" />
        <Figure label="After" value={recommendation.projected_after.toFixed(2)} unit="pts" />
        <Figure label="Uncertainty" value={`±${recommendation.risk.toFixed(2)}`} unit="pts" />
        {recommendation.hit_cost > 0 && (
          <Figure label="Hit" value={`−${recommendation.hit_cost}`} unit="pts" tone="loss" />
        )}
        <Figure label="Bank after" value={money(recommendation.bank_after)} />
      </dl>

      {recommendation.reasons.length > 0 && (
        <div className="mt-9">
          <p className="eyebrow">Because</p>
          <ul className="mt-4 space-y-3">
            {recommendation.reasons.map((reason) => (
              <li key={`${reason.code}-${reason.subject}`} className="flex gap-3.5">
                <span
                  aria-hidden
                  className="mt-[0.55rem] h-px w-5 shrink-0"
                  style={{ background: "var(--color-dim)" }}
                />
                <span className="text-[15px] leading-relaxed text-chalk/90">{reason.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function Figure({
  label,
  value,
  unit,
  tone,
  emphasis,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: "gain" | "loss";
  emphasis?: boolean;
}) {
  const colour =
    tone === "gain" ? "var(--color-gain)" : tone === "loss" ? "var(--color-loss)" : undefined;
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd
        className="tnum mt-1.5"
        style={{
          fontSize: emphasis ? "1.875rem" : "1.25rem",
          fontWeight: emphasis ? 600 : 500,
          color: colour ?? "var(--color-chalk)",
          letterSpacing: "-0.02em",
        }}
      >
        {value}
        {unit && <span className="ml-1 text-[0.6em] text-dim">{unit}</span>}
      </dd>
    </div>
  );
}
