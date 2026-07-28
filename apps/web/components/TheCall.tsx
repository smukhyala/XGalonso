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

      <Because recommendation={recommendation} />
    </section>
  );
}

/**
 * The evidence, grouped under the player each piece is about.
 *
 * This is the fix for the screen that read as self-contradictory. The reasons
 * were always correct and always carried a subject, and the subject was thrown
 * away at render — so "minutes look secure: 100%" sat directly above "minutes
 * are a concern: 60%" in one flat list, with nothing to say they described
 * different people. Grouping under a named heading is the whole repair.
 */
function Because({ recommendation }: { recommendation: Recommendation }) {
  const { player_out: out, player_in: incoming } = recommendation;
  if (recommendation.reasons.length === 0) return null;

  const forIn = recommendation.reasons.filter((r) => r.polarity === "supports_in");
  const forOut = recommendation.reasons.filter((r) => r.polarity === "supports_out");
  const context = recommendation.reasons.filter((r) => r.polarity === "context");

  return (
    <div className="mt-10">
      <p className="eyebrow">Because</p>

      <div className="mt-5 grid gap-8 sm:grid-cols-2">
        <Side
          verb="Buying"
          name={incoming?.name ?? "the incoming player"}
          colour={incoming ? POSITION_COLOR[incoming.position] : "var(--color-mid)"}
          reasons={forIn}
        />
        <Side
          verb="Selling"
          name={out?.name ?? "the outgoing player"}
          colour={out ? POSITION_COLOR[out.position] : "var(--color-loss)"}
          reasons={forOut}
        />
      </div>

      {context.length > 0 && (
        <ul className="mt-8 space-y-2.5">
          {context.map((reason) => (
            <li
              key={`${reason.code}-${reason.subject}`}
              className="max-w-xl text-[13px] leading-relaxed"
              style={{ color: "var(--color-dim)" }}
            >
              {reason.text}
            </li>
          ))}
        </ul>
      )}

      {recommendation.legal_moves > 0 && (
        <p className="eyebrow mt-6">
          {recommendation.legal_moves.toLocaleString()} legal moves evaluated across{" "}
          {recommendation.candidates_considered.toLocaleString()} buyable players
        </p>
      )}
    </div>
  );
}

function Side({
  verb,
  name,
  colour,
  reasons,
}: {
  verb: string;
  name: string;
  colour: string;
  reasons: Recommendation["reasons"];
}) {
  return (
    <div>
      <p className="text-[13px]">
        <span className="eyebrow">{verb}</span>{" "}
        <span style={{ color: colour }}>{name}</span>
      </p>
      {reasons.length > 0 ? (
        <ul className="mt-3 space-y-3">
          {reasons.map((reason) => (
            <li key={`${reason.code}-${reason.subject}`} className="flex gap-3">
              <span
                aria-hidden
                className="mt-[0.5rem] h-px w-3.5 shrink-0"
                style={{ background: colour, opacity: 0.6 }}
              />
              <span className="text-[14px] leading-relaxed text-chalk/90">{reason.text}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-3 text-[13px]" style={{ color: "var(--color-dim)" }}>
          Nothing here argues either way.
        </p>
      )}
    </div>
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
