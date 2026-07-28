"use client";

import { Pitch } from "@/components/Pitch";
import { PlayerLedger } from "@/components/PlayerLedger";
import { money, type SquadBuild as SquadBuildData } from "@/lib/api";

/**
 * The gameweek-1 answer: which fifteen, and why each of them.
 *
 * Before the first deadline there is no squad to transfer from and transfers
 * are unlimited, so "which single move is best" is the wrong question — and it
 * was the only question this product could previously answer. The optimizer
 * already solved the right one exactly: pick fifteen, field the best eleven of
 * them, double the best of those. This surfaces that solve.
 *
 * There is deliberately no transfer board here. "Who would replace him" has no
 * meaning when every player was chosen freely and could be swapped at no cost,
 * so showing one would invent a constraint that does not exist yet.
 */
export function SquadBuild({ build }: { build: SquadBuildData }) {
  return (
    <>
      <section className="rise" style={{ animationDelay: "0.1s" }}>
        <p className="eyebrow">The squad · GW{build.gameweek}</p>

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
          Start these fifteen.
        </h1>

        <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-muted">
          Transfers are unlimited until the first deadline, so this is a squad built from
          scratch rather than a move from one you already own. Chosen by exact solve over{" "}
          {build.candidates_considered.toLocaleString()} available players — the fifteen whose
          best legal eleven scores highest, with the captain doubled.
        </p>

        <dl className="mt-9 flex flex-wrap items-baseline gap-x-10 gap-y-4">
          <Figure
            label="Projected"
            value={build.projected_points.toFixed(2)}
            unit="pts"
            emphasis
          />
          <Figure label="Squad value" value={money(build.squad_value)} />
          <Figure label="Bank" value={money(build.bank)} />
          <Figure label="Shape" value={build.formation} />
        </dl>

        <p className="mt-6 max-w-xl text-[13px] leading-relaxed text-dim">
          Money left in the bank is a readout, not a mistake. Bench players contribute
          nothing to the score, so at the optimum forcing every last 0.1m to be spent
          changes the projection by zero — spending it would buy a cosmetic number with
          real points.
        </p>
      </section>

      <section className="rise mt-14" style={{ animationDelay: "0.2s" }}>
        <div className="flex items-baseline justify-between">
          <p className="eyebrow">The eleven</p>
          <p className="eyebrow">{build.formation}</p>
        </div>
        <div className="mt-5">
          <Pitch players={build.players} />
        </div>
      </section>

      <PlayerLedger players={build.explanations} title="The case for each pick" />
    </>
  );
}

function Figure({
  label,
  value,
  unit,
  emphasis,
}: {
  label: string;
  value: string;
  unit?: string;
  emphasis?: boolean;
}) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd
        className="tnum mt-1.5"
        style={{
          fontSize: emphasis ? "1.875rem" : "1.25rem",
          fontWeight: emphasis ? 600 : 500,
          letterSpacing: "-0.02em",
        }}
      >
        {value}
        {unit && <span className="ml-1 text-[0.6em] text-dim">{unit}</span>}
      </dd>
    </div>
  );
}
