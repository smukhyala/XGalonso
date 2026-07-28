"use client";

import { POSITION_COLOR, type LineupComparison } from "@/lib/api";

/**
 * The eleven you have set, against the eleven we would field.
 *
 * A list of swaps is the obvious design and it is incomplete: two elevens from
 * the same fifteen differ in who plays, what shape they play, *and* who wears
 * the armband — and captaincy doubles a return, so moving it can outweigh every
 * substitution combined. The three terms below sum to the real gap, and the
 * residual is shown rather than folded away, because a decomposition whose last
 * term is "everything else" proves nothing unless you can see it.
 */
export function LineupDiff({ lineup }: { lineup: LineupComparison }) {
  if (lineup.is_identical) {
    return (
      <section className="rise mt-24" style={{ animationDelay: "0.28s" }}>
        <p className="eyebrow">Your eleven</p>
        <div className="hairline mt-5" />
        <p className="mt-6 max-w-xl text-[15px] leading-relaxed">
          You are already fielding the best eleven from this squad, in the right shape,
          with the right captain. Nothing to change.
        </p>
        <p className="tnum mt-4 text-sm text-dim">
          {lineup.ours_points.toFixed(2)} pts · {lineup.ours_formation}
        </p>
      </section>
    );
  }

  const better = !lineup.yours_is_better;
  const terms = [
    { label: "Different players", value: lineup.swap_delta },
    { label: "Captain", value: lineup.captain_delta },
    { label: "Shape", value: lineup.shape_delta },
  ].filter((term) => Math.abs(term.value) >= 0.005);

  return (
    <section className="rise mt-24" style={{ animationDelay: "0.28s" }}>
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="eyebrow">Your eleven vs ours</p>
        <p className="eyebrow">
          {lineup.yours_formation} → {lineup.ours_formation}
        </p>
      </div>
      <div className="hairline mt-5" />

      <dl className="mt-7 flex flex-wrap items-baseline gap-x-10 gap-y-4">
        <Figure label="Yours" value={lineup.yours_points.toFixed(2)} unit="pts" />
        <Figure label="Ours" value={lineup.ours_points.toFixed(2)} unit="pts" />
        <Figure
          label={better ? "You would gain" : "Ours is worse"}
          value={`${lineup.total_delta >= 0 ? "+" : ""}${lineup.total_delta.toFixed(2)}`}
          unit="pts"
          tone={lineup.total_delta >= 0 ? "gain" : "loss"}
          emphasis
        />
      </dl>

      <p className="eyebrow mt-9">Where the difference comes from</p>
      <ul className="mt-4 space-y-2.5">
        {terms.map((term) => (
          <li key={term.label} className="flex items-center gap-3">
            <span className="text-[14px]" style={{ color: "var(--color-muted)" }}>
              {term.label}
            </span>
            <span className="h-px flex-1" style={{ background: "var(--color-line)" }} />
            <span
              className="tnum text-[14px]"
              style={{
                color: term.value >= 0 ? "var(--color-gain)" : "var(--color-loss)",
              }}
            >
              {term.value >= 0 ? "+" : ""}
              {term.value.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-4 max-w-lg text-[13px] leading-relaxed" style={{ color: "var(--color-dim)" }}>
        These add up to the gap exactly. &ldquo;Shape&rdquo; is whatever the swaps and the
        armband do not account for — it appears when the two elevens play different
        formations, and it is shown rather than absorbed so you can see when the
        explanation is incomplete.
      </p>

      {lineup.yours_captain_name !== lineup.ours_captain_name && (
        <>
          <p className="eyebrow mt-9">Captain</p>
          <p className="mt-3 text-[15px]">
            <span style={{ color: "var(--color-muted)" }}>{lineup.yours_captain_name}</span>
            <span aria-hidden className="mx-3" style={{ color: "var(--color-dim)" }}>
              →
            </span>
            <span>{lineup.ours_captain_name}</span>
          </p>
        </>
      )}

      {lineup.swaps.length > 0 && (
        <>
          <p className="eyebrow mt-9">Changes</p>
          <ul className="mt-3">
            {lineup.swaps.map((swap, index) => (
              <li
                key={`${swap.player_in ?? "none"}-${swap.player_out ?? "none"}-${index}`}
                className="grid grid-cols-[1fr_auto] items-center gap-4 border-b py-3"
                style={{ borderColor: "var(--color-line)" }}
              >
                <span className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
                  <span
                    aria-hidden
                    className="h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ background: POSITION_COLOR[swap.position] }}
                  />
                  <span className="text-[15px]" style={{ color: "var(--color-muted)" }}>
                    {swap.player_out_name ?? "—"}
                  </span>
                  <span aria-hidden style={{ color: "var(--color-dim)" }}>
                    →
                  </span>
                  <span className="text-[15px]">{swap.player_in_name ?? "—"}</span>
                  {!swap.is_like_for_like && (
                    <span className="eyebrow" title="Half of a formation change">
                      shape
                    </span>
                  )}
                </span>
                <span
                  className="tnum text-right text-sm"
                  style={{ color: swap.delta >= 0 ? "var(--color-gain)" : "var(--color-loss)" }}
                >
                  {swap.delta >= 0 ? "+" : ""}
                  {swap.delta.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </>
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
