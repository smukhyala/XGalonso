"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  POSITION_COLOR,
  money,
  type Plan,
  type ParsedRequirements,
  type Requirement,
  type RequirementInput,
  type RequirementKind,
} from "@/lib/api";

/**
 * Say what the squad must contain, and get one that contains it.
 *
 * **The parse is a proposal, not a decision.** Natural language is ambiguous, so
 * what the parser understood is shown as editable chips *before* anything is
 * built, and building sends the edited set rather than the original sentence.
 * A system that acts on its own guess without showing it will eventually act on
 * a wrong one silently.
 *
 * **Requirements are hard.** Each one is a constraint on the solver rather than
 * a penalty, so it is either honoured exactly or reported as impossible. That is
 * why the results panel leads with what could *not* be honoured: when a set
 * cannot all hold, which requirement gave is the most useful thing on the page.
 *
 * **Every requirement carries a price.** What it costs is the difference
 * between the best squad with it and the best squad without, holding the others
 * fixed. Zero is a real answer and worth showing — it means the optimizer wanted
 * that player anyway, so the request was free.
 */

const EXAMPLES = [
  "I want Haaland starting, play 3-5-2 and leave 0.5 in the bank",
  "keep Haaland and Saka, at least 3 from Arsenal",
  "chase differentials aggressively over three gameweeks, Haaland starting",
  "captain Haaland, avoid Chelsea defenders, 3-4-3",
  "Haaland starting, prioritise the non-elite players",
];

/** How each requirement kind reads, and the colour it carries throughout. */
const KIND_META: Record<RequirementKind, { label: string; tone: string }> = {
  must_start: { label: "starts", tone: "rgba(126, 200, 148, 0.9)" },
  must_include: { label: "in squad", tone: "rgba(126, 176, 200, 0.9)" },
  must_exclude: { label: "excluded", tone: "rgba(224, 122, 122, 0.8)" },
  must_captain: { label: "captain", tone: "rgba(224, 186, 122, 0.9)" },
  club_floor: { label: "club min", tone: "rgba(170, 160, 200, 0.85)" },
  club_ceiling: { label: "club max", tone: "rgba(170, 160, 200, 0.85)" },
  formation: { label: "shape", tone: "rgba(200, 200, 200, 0.7)" },
  bank_floor: { label: "bank", tone: "rgba(200, 200, 200, 0.7)" },
};

function toInput(requirement: Requirement): RequirementInput {
  return {
    kind: requirement.kind,
    players: requirement.players,
    team_id: requirement.team_id,
    count: requirement.count,
    formation: requirement.formation,
    amount: requirement.amount,
    priority: requirement.priority,
  };
}

export default function PlanPage() {
  const [text, setText] = useState("");
  const [interpret, setInterpret] = useState(true);
  const [parsed, setParsed] = useState<ParsedRequirements | null>(null);
  const [dropped, setDropped] = useState<Set<number>>(new Set());
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);

  // Parsing is cheap — nothing is solved — so it runs as you type. Building is
  // a MILP plus one re-solve per requirement, so it only runs on request.
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    if (!text.trim()) {
      setParsed(null);
      return;
    }
    timer.current = setTimeout(() => {
      api
        .parseRequirements(text, "expected_points", interpret)
        .then((next) => {
          setParsed(next);
          setDropped(new Set());
        })
        .catch(() => setParsed(null));
      // Slower when the model is involved: that call costs money and a token
      // per keystroke is not a feature.
    }, interpret ? 900 : 300);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [text, interpret]);

  const kept = (parsed?.requirements ?? []).filter((_, index) => !dropped.has(index));

  const build = useCallback(async () => {
    setBuilding(true);
    setError(null);
    try {
      // The edited set is sent, not the sentence. Re-deriving from the text
      // here would silently discard whatever the manager just corrected.
      setPlan(await api.plan({ text, requirements: kept.map(toInput), interpret }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not build a squad.");
      setPlan(null);
    } finally {
      setBuilding(false);
    }
  }, [text, kept, interpret]);

  return (
    <main className="mx-auto max-w-6xl px-6 pb-28 sm:px-10">
      <Masthead />

      <header className="mt-14 max-w-3xl">
        <p className="eyebrow">Plan</p>
        <h1
          className="mt-4 text-balance"
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2rem, 5vw, 3.25rem)",
            fontWeight: 700,
            lineHeight: 1.0,
            letterSpacing: "-0.03em",
          }}
        >
          Say what the squad must contain.
        </h1>
        <p className="mt-5 text-[15px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
          Everything you ask for becomes a hard constraint, not a preference — so it is either
          honoured exactly or reported as impossible. Check what was understood before you
          build; the squad is assembled from the chips below, not from the sentence.
        </p>
      </header>

      <section className="rise mt-10" style={{ animationDelay: "0.08s" }}>
        <label htmlFor="request" className="eyebrow">
          What you want
        </label>
        <textarea
          id="request"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={2}
          placeholder="I want Haaland starting, play 3-5-2 and leave 0.5 in the bank"
          className="mt-3 w-full resize-none bg-transparent px-0 py-3 text-[17px] outline-none"
          style={{
            borderBottom: "1px solid var(--color-line)",
            fontFamily: "var(--font-display)",
          }}
        />

        <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3">
          <button
            type="button"
            onClick={() => setInterpret((on) => !on)}
            className="flex items-center gap-2.5 text-[12px] transition-opacity hover:opacity-100"
            style={{ color: interpret ? "var(--color-text)" : "var(--color-dim)" }}
            aria-pressed={interpret}
          >
            <span
              aria-hidden
              className="inline-block h-3 w-3 rounded-[2px]"
              style={{
                background: interpret ? "rgba(126, 200, 148, 0.9)" : "transparent",
                border: "1px solid var(--color-line)",
              }}
            />
            Read it with Claude too
          </button>
          <span className="text-[12px]" style={{ color: "var(--color-dim)" }}>
            Catches intent the vocabulary cannot — “prioritise the non-elite players”, “I’m bored
            of my team”. It never overrides a matched phrase.
          </span>
        </div>

        <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => setText(example)}
              className="text-[12px] transition-opacity hover:opacity-100"
              style={{ color: "var(--color-dim)", opacity: 0.85 }}
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      {parsed && (
        <Understood
          parsed={parsed}
          dropped={dropped}
          onToggle={(index) =>
            setDropped((current) => {
              const next = new Set(current);
              if (next.has(index)) next.delete(index);
              else next.add(index);
              return next;
            })
          }
        />
      )}

      {parsed && (
        <div className="mt-8 flex flex-wrap items-center gap-5">
          <button
            type="button"
            onClick={() => void build()}
            disabled={building}
            className="px-6 py-3 text-[14px] transition-opacity hover:opacity-80 disabled:opacity-40"
            style={{ border: "1px solid var(--color-text)" }}
          >
            {building ? "Building…" : `Build a squad around ${kept.length} requirement${kept.length === 1 ? "" : "s"}`}
          </button>
          <span className="text-[12px]" style={{ color: "var(--color-dim)" }}>
            Solves an exact optimisation and prices each requirement — a few seconds.
          </span>
        </div>
      )}

      {error && (
        <div className="mt-10 border p-6" style={{ borderColor: "rgba(224,122,122,0.4)" }}>
          <p className="text-[13px]" style={{ color: "var(--color-muted)" }}>
            {error}
          </p>
        </div>
      )}

      {plan && <Result plan={plan} />}
    </main>
  );
}

function Understood({
  parsed,
  dropped,
  onToggle,
}: {
  parsed: ParsedRequirements;
  dropped: Set<number>;
  onToggle: (index: number) => void;
}) {
  return (
    <section className="rise mt-12" style={{ animationDelay: "0.1s" }}>
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="eyebrow">What I understood</p>
        <p className="eyebrow hidden sm:block">
          {parsed.objective_id} · {Math.round(parsed.overall_confidence * 100)}% confident
          {parsed.interpreted ? " · read by claude" : ""}
        </p>
      </div>
      <div className="hairline mt-4" />

      {parsed.requirements.length === 0 ? (
        <p className="mt-4 text-[13px]" style={{ color: "var(--color-dim)" }}>
          No structural requirements yet — building now would give the free optimum.
        </p>
      ) : (
        <ul className="mt-4 space-y-px">
          {parsed.requirements.map((requirement, index) => {
            const meta = KIND_META[requirement.kind];
            const off = dropped.has(index);
            return (
              <li key={`${requirement.kind}-${requirement.label}-${index}`}>
                <button
                  type="button"
                  onClick={() => onToggle(index)}
                  className="grid w-full grid-cols-[6rem_1fr_auto] items-baseline gap-4 border-b py-3 text-left transition-opacity hover:opacity-80"
                  style={{
                    borderColor: "var(--color-line)",
                    opacity: off ? 0.35 : 1,
                  }}
                  title={off ? "Excluded from the build — click to restore" : "Click to drop"}
                >
                  <span className="flex flex-col gap-0.5">
                    <span className="eyebrow" style={{ color: meta.tone }}>
                      {meta.label}
                    </span>
                    {requirement.source === "model" && (
                      // Inferred, not matched. A reader has to be able to tell.
                      <span className="eyebrow" style={{ color: "var(--color-dim)" }}>
                        claude
                      </span>
                    )}
                  </span>
                  <span className="flex flex-col gap-0.5">
                    <span
                      className="text-[14px]"
                      style={{ textDecoration: off ? "line-through" : undefined }}
                    >
                      {requirement.label}
                    </span>
                    {requirement.evidence && (
                      // Showing the phrase is what lets a manager tell a correct
                      // reading from a plausible one.
                      <span className="text-[12px]" style={{ color: "var(--color-dim)" }}>
                        from “{requirement.evidence.trim()}”
                      </span>
                    )}
                  </span>
                  <span className="tnum text-[12px]" style={{ color: "var(--color-dim)" }}>
                    {requirement.confidence === null
                      ? "—"
                      : `${Math.round(requirement.confidence * 100)}%`}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {parsed.problems.length > 0 && (
        // Caught without solving. Saying so immediately beats a relaxation pass
        // that blames something else.
        <div className="mt-5 border p-4" style={{ borderColor: "rgba(224,122,122,0.4)" }}>
          <p className="eyebrow" style={{ color: "rgba(224,122,122,0.9)" }}>
            These cannot all hold
          </p>
          <ul className="mt-2 space-y-1">
            {parsed.problems.map((problem) => (
              <li key={problem} className="text-[13px]" style={{ color: "var(--color-muted)" }}>
                {problem}
              </li>
            ))}
          </ul>
        </div>
      )}

      {(parsed.ownership_preference || parsed.risk_preference) && (
        <div className="mt-5">
          <p className="eyebrow">Read as a lean, not a requirement</p>
          <p className="mt-2 text-[13px]" style={{ color: "var(--color-muted)" }}>
            {[parsed.ownership_preference, parsed.risk_preference].filter(Boolean).join(" · ")} —
            this shapes the objective rather than constraining the squad, so it does not appear
            as a chip.
          </p>
        </div>
      )}

      {parsed.model_notes.length > 0 && (
        <div className="mt-5">
          <p className="eyebrow">What Claude made of it</p>
          <ul className="mt-2 space-y-1.5">
            {parsed.model_notes.map((note) => (
              <li
                key={note}
                className="text-[13px] leading-relaxed"
                style={{ color: "var(--color-muted)" }}
              >
                {note}
              </li>
            ))}
          </ul>
        </div>
      )}

      {parsed.interpreter_note && !parsed.interpreted && (
        <p className="mt-5 text-[12px]" style={{ color: "var(--color-dim)" }}>
          Claude was not consulted: {parsed.interpreter_note}. The matched reading above stands.
        </p>
      )}

      {parsed.unparsed.length > 0 && (
        <div className="mt-5">
          <p className="eyebrow">Not understood</p>
          <p className="mt-2 text-[13px]" style={{ color: "var(--color-dim)" }}>
            {parsed.unparsed.join(" · ")} — ignored rather than guessed at.
          </p>
        </div>
      )}
    </section>
  );
}

function Result({ plan }: { plan: Plan }) {
  const starters = plan.players.filter((player) => player.is_starter);
  const bench = plan.players.filter((player) => !player.is_starter);
  const broken = plan.outcomes.filter((outcome) => !outcome.honoured);

  return (
    <section className="rise mt-16" style={{ animationDelay: "0.12s" }}>
      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <p className="eyebrow">The squad</p>
        <p className="eyebrow hidden sm:block">
          GW{plan.gameweek} · {plan.model_note}
        </p>
      </div>
      <div className="hairline mt-4" />

      <div className="mt-6 flex flex-wrap items-baseline gap-x-10 gap-y-3">
        <Figure value={plan.expected_points.toFixed(2)} label="expected points" />
        <Figure value={plan.formation} label="shape" />
        <Figure value={money(plan.bank)} label="in the bank" />
        <Figure
          value={plan.total_cost <= 0.005 ? "nothing" : `${plan.total_cost.toFixed(2)} pts`}
          label="your requirements cost"
        />
      </div>

      {plan.total_cost > 0.005 && (
        <p className="mt-3 text-[12px]" style={{ color: "var(--color-dim)" }}>
          The best squad ignoring everything you asked for scores{" "}
          {plan.unconstrained_points.toFixed(2)}.
        </p>
      )}

      {broken.length > 0 && (
        <div className="mt-6 border p-5" style={{ borderColor: "rgba(224,186,122,0.45)" }}>
          <p className="eyebrow" style={{ color: "rgba(224,186,122,0.95)" }}>
            Could not honour everything
          </p>
          <ul className="mt-2.5 space-y-1.5">
            {broken.map((outcome) => (
              <li key={outcome.label} className="text-[13px]">
                <span style={{ color: "var(--color-text)" }}>{outcome.label}</span>
                {outcome.note && (
                  <span style={{ color: "var(--color-dim)" }}> — {outcome.note}</span>
                )}
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[12px]" style={{ color: "var(--color-dim)" }}>
            These were dropped in priority order so a legal squad could exist. Everything else
            below was honoured exactly.
          </p>
        </div>
      )}

      <div className="mt-8 grid gap-10 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <p className="eyebrow">Starting eleven</p>
          <ul className="mt-3">
            {starters.map((player) => (
              <PlayerRow key={player.player_code} player={player} />
            ))}
          </ul>

          <p className="eyebrow mt-8">Bench</p>
          <ul className="mt-3">
            {bench.map((player) => (
              <PlayerRow key={player.player_code} player={player} />
            ))}
          </ul>
        </div>

        <div>
          <p className="eyebrow">What each requirement cost</p>
          <ul className="mt-3 space-y-px">
            {plan.outcomes.map((outcome) => (
              <li
                key={`${outcome.kind}-${outcome.label}`}
                className="flex items-baseline justify-between gap-4 border-b py-2.5"
                style={{ borderColor: "var(--color-line)" }}
              >
                <span
                  className="text-[13px]"
                  style={{ color: outcome.honoured ? undefined : "var(--color-dim)" }}
                >
                  {outcome.label}
                </span>
                <span className="tnum whitespace-nowrap text-[12px]">
                  {!outcome.honoured ? (
                    <span style={{ color: "rgba(224,122,122,0.9)" }}>dropped</span>
                  ) : outcome.cost === null ? (
                    <span style={{ color: "var(--color-dim)" }}>—</span>
                  ) : outcome.cost <= 0.005 ? (
                    <span style={{ color: "var(--color-dim)" }}>free</span>
                  ) : (
                    <span>−{outcome.cost.toFixed(2)}</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[12px] leading-relaxed" style={{ color: "var(--color-dim)" }}>
            Each cost is measured by rebuilding the squad without that requirement and holding
            the others fixed, so they are marginal — a captaincy lock can make its own start
            lock free, and the joint figure is the total above.
          </p>
        </div>
      </div>
    </section>
  );
}

function PlayerRow({ player }: { player: Plan["players"][number] }) {
  return (
    <li
      className="grid grid-cols-[1fr_auto_auto] items-center gap-4 border-b py-2.5"
      style={{ borderColor: "var(--color-line)" }}
    >
      <span className="flex items-center gap-3">
        <span
          aria-hidden
          className="h-1.5 w-1.5 rounded-full"
          style={{ background: POSITION_COLOR[player.position] }}
        />
        <span className="text-[14px]">{player.name}</span>
        {player.is_captain && <span className="eyebrow">C</span>}
        {player.is_vice_captain && <span className="eyebrow">V</span>}
      </span>
      <span className="tnum text-[12px]" style={{ color: "var(--color-dim)" }}>
        {money(player.price)}
      </span>
      <span className="tnum text-right text-[13px]">{player.expected_points.toFixed(2)}</span>
    </li>
  );
}

function Figure({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <p className="tnum text-[22px]" style={{ fontFamily: "var(--font-display)" }}>
        {value}
      </p>
      <p className="eyebrow mt-0.5">{label}</p>
    </div>
  );
}

function Masthead() {
  return (
    <header className="flex flex-wrap items-center justify-between gap-6 pt-10">
      <Link href="/" className="flex items-baseline gap-4 transition-opacity hover:opacity-70">
        <span
          style={{
            fontFamily: "var(--font-display)",
            fontWeight: 700,
            fontSize: "1.0625rem",
            letterSpacing: "0.02em",
          }}
        >
          XG Alonso
        </span>
      </Link>
      <nav className="flex gap-6">
        <Link href="/discovery" className="eyebrow transition-opacity hover:opacity-70">
          Discovery
        </Link>
        <Link href="/features" className="eyebrow transition-opacity hover:opacity-70">
          Feature lab
        </Link>
        <Link href="/" className="eyebrow transition-opacity hover:opacity-70">
          Back to the squad
        </Link>
      </nav>
    </header>
  );
}
