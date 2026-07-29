"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import {
  api,
  type Cluster,
  type DiscoveredFeature,
  type Experiment,
  type Hypothesis,
} from "@/lib/api";

/**
 * Objective-conditioned feature discovery, as a read surface.
 *
 * **Read-only, deliberately.** A discovery run fits hundreds of models over
 * five walk-forward folds and takes minutes. There is no job queue here, so a
 * button that launched one would be a request that blocks until it times out.
 * `xg discover` runs them; this reads what they produced.
 *
 * **Rejections are shown beside acceptances.** A page listing only the features
 * that worked is indistinguishable from one that never tested anything, and the
 * rejected rows are where most of the information is — a hypothesis that
 * improved five folds out of five but moved the ranking by nothing tells you
 * more about the model than another accepted feature does.
 *
 * **Objectives come from experiments, not from the preset list.** A run
 * compiles a preset plus the manager's own words into a derived objective —
 * `expected_rank_gain_aggressive_h3_differential`, not `expected_points` — and
 * verdicts are filed under the derived id. Listing presets here would query
 * objectives nothing was ever measured against and render an empty page that
 * looked broken.
 */
export default function DiscoveryPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [objective, setObjective] = useState<string | null>(null);
  const [features, setFeatures] = useState<DiscoveredFeature[]>([]);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    let live = true;
    Promise.all([api.experiments(), api.hypotheses()])
      .then(([runs, ideas]) => {
        if (!live) return;
        setExperiments(runs);
        setHypotheses(ideas);
        const measured = runs.map((run) => run.objective_id).filter((id): id is string => !!id);
        setObjective(measured[0] ?? null);
      })
      .catch((cause: unknown) => {
        if (!live) return;
        setError(cause instanceof Error ? cause.message : "Could not reach the discovery registry.");
      })
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, []);

  const loadObjective = useCallback(async (id: string) => {
    const [discovered, grouped] = await Promise.all([
      api.discoveredFeatures(id).catch(() => []),
      api.clusters(id).catch(() => []),
    ]);
    setFeatures(discovered);
    setClusters(grouped);
  }, []);

  useEffect(() => {
    if (objective) void loadObjective(objective);
  }, [objective, loadObjective]);

  const objectives = Array.from(
    new Set(experiments.map((run) => run.objective_id).filter((id): id is string => !!id)),
  );

  return (
    <main className="mx-auto max-w-6xl px-6 pb-28 sm:px-10">
      <Masthead />

      <header className="mt-14 max-w-3xl">
        <p className="eyebrow">Discovery</p>
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
          Features found for what you were trying to do.
        </h1>
        <p className="mt-5 text-[15px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
          The feature lab asks which columns predict points. This asks a different question:
          given an objective — chase a mini-league, protect a rank, grow team value — which
          representation of the player pool produces the best <em>decisions</em>. A feature
          that helps one objective can be worthless to another, so every verdict below is
          filed under the objective it was measured against.
        </p>
      </header>

      {error && <Empty message={error} />}
      {busy && !error && <Skeleton />}

      {!busy && !error && objectives.length === 0 && (
        <Empty message="No discovery run has been recorded yet. Run `xg discover` to produce one." />
      )}

      {objectives.length > 0 && (
        <>
          <nav className="rise mt-12" aria-label="Objective" style={{ animationDelay: "0.1s" }}>
            <p className="eyebrow">Measured against</p>
            <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2.5">
              {objectives.map((id) => (
                <Chip
                  key={id}
                  label={objectiveName(id)}
                  active={objective === id}
                  onClick={() => setObjective(id)}
                />
              ))}
            </div>
          </nav>

          <Verdicts features={features} hypotheses={hypotheses} />
          <Clusters clusters={clusters} />
          <Runs experiments={experiments.filter((run) => run.objective_id === objective)} />
        </>
      )}
    </main>
  );
}

/**
 * A derived objective id read back as English.
 *
 * `expected_rank_gain_aggressive_h3_differential` is exactly how the registry
 * files it and exactly how nobody speaks. The parts are positional and
 * generated by the intent compiler, so they can be unpacked rather than guessed.
 */
function objectiveName(id: string): string {
  const horizon = /_h(\d+)/.exec(id)?.[1];
  const risk = ["aggressive", "balanced", "conservative"].find((word) => id.includes(word));
  const metric = id.startsWith("expected_rank_gain")
    ? "Rank gain"
    : id.startsWith("expected_points")
      ? "Points"
      : id.split("_").slice(0, 2).join(" ");

  const parts = [metric];
  if (risk) parts.push(risk);
  if (horizon) parts.push(`${horizon} GW`);
  if (id.includes("differential")) parts.push("differential");
  return parts.join(" · ");
}

/** Verdict to how it should read. Rejections are not failures of the process. */
const STATUS_TONE: Record<string, { label: string; color: string }> = {
  accepted: { label: "Accepted", color: "rgba(126, 200, 148, 0.9)" },
  rejected: { label: "Rejected", color: "rgba(224, 122, 122, 0.75)" },
  revise: { label: "Needs revision", color: "rgba(224, 186, 122, 0.85)" },
};

function Verdicts({
  features,
  hypotheses,
}: {
  features: DiscoveredFeature[];
  hypotheses: Hypothesis[];
}) {
  const [open, setOpen] = useState<string | null>(null);
  const byId = new Map(hypotheses.map((h) => [h.id, h]));

  if (features.length === 0) {
    return (
      <section className="mt-16">
        <p className="eyebrow">Verdicts</p>
        <p className="mt-4 text-[13px]" style={{ color: "var(--color-dim)" }}>
          No feature has been measured against this objective yet.
        </p>
      </section>
    );
  }

  const accepted = features.filter((f) => f.status === "accepted").length;

  return (
    <section className="rise mt-16" style={{ animationDelay: "0.2s" }}>
      <div className="flex items-baseline justify-between">
        <p className="eyebrow">Verdicts</p>
        <p className="eyebrow hidden sm:block">
          {accepted} of {features.length} accepted
        </p>
      </div>
      <div className="hairline mt-5" />

      <ol className="mt-2">
        {features.map((feature) => {
          const tone = STATUS_TONE[feature.status] ?? {
            label: feature.status,
            color: "var(--color-line)",
          };
          const hypothesis = byId.get(feature.hypothesis_id);
          const isOpen = open === feature.version;

          return (
            <li
              key={feature.version}
              className="border-b"
              style={{ borderColor: "var(--color-line)" }}
            >
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : feature.version)}
                aria-expanded={isOpen}
                className="grid w-full grid-cols-[1fr_auto] items-center gap-4 py-3.5 text-left transition-opacity hover:opacity-80 sm:grid-cols-[1fr_9rem_5rem_4rem]"
              >
                <span className="flex flex-col gap-0.5">
                  <span className="flex items-center gap-3">
                    <span
                      aria-hidden
                      className="h-1.5 w-1.5 rounded-full"
                      style={{ background: tone.color }}
                    />
                    <span className="text-[15px]">{feature.feature}</span>
                  </span>
                  <span className="pl-[1.125rem] text-[12px]" style={{ color: "var(--color-dim)" }}>
                    {feature.reason}
                  </span>
                </span>

                <span className="eyebrow hidden sm:block">{tone.label}</span>

                {/* Folds improved, not a p-value. Five of five is a fact; a
                    significance claim would need a test nobody ran. */}
                <span className="tnum hidden text-sm text-muted sm:block">
                  {feature.folds_improved}/{feature.folds} folds
                </span>

                <span className="tnum text-right text-sm">{feature.utility.toFixed(3)}</span>
              </button>

              {isOpen && (
                <div className="grid gap-8 pb-8 pl-[1.125rem] pr-2 pt-1 lg:grid-cols-2">
                  <div className="space-y-5">
                    <Figure
                      label="Incremental value"
                      value={feature.incremental_value.toFixed(5)}
                      note="Gain over the required feature set alone."
                    />
                    <Figure
                      label="Stability"
                      value={feature.stability.toFixed(2)}
                      note="Agreement of its contribution across folds. Low means the mechanism may be right and the transformation wrong."
                    />
                    <Figure
                      label="Complementarity"
                      value={feature.complementarity.replace(/_/g, " ")}
                      note="Whether it adds something the existing set was missing, or only for some clusters."
                    />
                    <Figure
                      label="Missingness"
                      value={`${(feature.missingness * 100).toFixed(1)}%`}
                      note="Rows where the program produced no value."
                    />
                    <Figure
                      label="Leakage harness"
                      value={feature.leakage_passed ? "Passed" : "NOT PASSED"}
                      note="Rebuilt with future rows appended; a feature whose past values moved is refused."
                    />
                  </div>

                  {hypothesis && (
                    <div className="space-y-5">
                      <div>
                        <p className="eyebrow">The claim</p>
                        <p className="mt-2 text-[13px] leading-relaxed">{hypothesis.title}</p>
                      </div>
                      <div>
                        <p className="eyebrow">Why it might be true</p>
                        <p
                          className="mt-2 text-[13px] leading-relaxed"
                          style={{ color: "var(--color-muted)" }}
                        >
                          {hypothesis.football_rationale}
                        </p>
                      </div>
                      <div>
                        <p className="eyebrow">What would refute it</p>
                        <p
                          className="mt-2 text-[13px] leading-relaxed"
                          style={{ color: "var(--color-muted)" }}
                        >
                          {hypothesis.falsification_condition}
                        </p>
                      </div>
                      {hypothesis.generation_source === "llm" && (
                        // Kept visible permanently. A language model proposed
                        // this; the measurement accepted or rejected it, and the
                        // two must never be confused for one another.
                        <p className="text-[12px]" style={{ color: "var(--color-dim)" }}>
                          Proposed by a language model. The rationale above is a claim, not a
                          finding — the verdict beside it came from the same walk-forward
                          measurement as every other feature.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function Figure({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div>
      <p className="eyebrow">{label}</p>
      <p className="tnum mt-1 text-[15px]">{value}</p>
      <p className="mt-1 text-[12px] leading-relaxed" style={{ color: "var(--color-dim)" }}>
        {note}
      </p>
    </div>
  );
}

function Clusters({ clusters }: { clusters: Cluster[] }) {
  if (clusters.length === 0) return null;

  const total = clusters.reduce((sum, cluster) => sum + cluster.size, 0);

  return (
    <section className="rise mt-20" style={{ animationDelay: "0.25s" }}>
      <p className="eyebrow">How the player pool divided</p>
      <p className="mt-3 max-w-2xl text-[13px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
        Clusters are fitted <em>under the objective</em>, so the same players group differently
        depending on what you are trying to achieve. These are not positions.
      </p>

      <div className="mt-6 grid gap-px sm:grid-cols-2 lg:grid-cols-3">
        {clusters.map((cluster) => (
          <div
            key={`${cluster.cluster_model_version}-${cluster.cluster_id}`}
            className="border p-5"
            style={{ borderColor: "var(--color-line)" }}
          >
            <div className="flex items-baseline justify-between gap-3">
              <p className="eyebrow">Cluster {cluster.cluster_id}</p>
              <p className="tnum text-[12px]" style={{ color: "var(--color-dim)" }}>
                {cluster.size.toLocaleString()} ·{" "}
                {total > 0 ? `${Math.round((cluster.size / total) * 100)}%` : "—"}
              </p>
            </div>
            <p className="mt-2.5 text-[14px] leading-snug">{cluster.label}</p>
            <ul className="mt-3 space-y-1">
              {cluster.dominant_features.slice(0, 3).map(([name, weight]) => (
                <li
                  key={name}
                  className="flex items-baseline justify-between gap-2 text-[12px]"
                  style={{ color: "var(--color-dim)" }}
                >
                  <span>{name.replace(/_/g, " ")}</span>
                  <span className="tnum">{weight.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

function Runs({ experiments }: { experiments: Experiment[] }) {
  if (experiments.length === 0) return null;

  return (
    <section className="rise mt-20" style={{ animationDelay: "0.3s" }}>
      <p className="eyebrow">Runs</p>
      <div className="hairline mt-5" />
      <ul className="mt-2">
        {experiments.map((run) => (
          <li
            key={run.experiment_id}
            className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b py-3.5"
            style={{ borderColor: "var(--color-line)" }}
          >
            <span className="text-[13px]">{run.experiment_id}</span>
            <span className="tnum text-[12px]" style={{ color: "var(--color-dim)" }}>
              {run.hypotheses_proposed} proposed · {run.features_compiled} compiled ·{" "}
              {run.features_accepted} accepted · {run.features_rejected} rejected
            </span>
            {/* A dirty tree means the recorded commit does not describe the code
                that ran, so the run cannot be called reproducible. Saying so is
                the whole value of recording it. */}
            <span className="eyebrow" style={{ color: run.git_dirty ? "rgba(224, 186, 122, 0.9)" : undefined }}>
              {run.git_dirty ? "not reproducible" : "reproducible"}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Chip({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="text-[13px] transition-opacity hover:opacity-100"
      style={{
        color: active ? "var(--color-text)" : "var(--color-dim)",
        opacity: active ? 1 : 0.8,
        borderBottom: active ? "1px solid var(--color-text)" : "1px solid transparent",
        paddingBottom: "2px",
      }}
    >
      {label}
    </button>
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

function Empty({ message }: { message: string }) {
  return (
    <div className="mt-16 border p-8" style={{ borderColor: "var(--color-line)" }}>
      <p className="text-[13px]" style={{ color: "var(--color-muted)" }}>
        {message}
      </p>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="mt-16 space-y-3" aria-hidden>
      {[0, 1, 2, 3, 4].map((row) => (
        <div
          key={row}
          className="h-11 w-full"
          style={{ background: "var(--color-line)", opacity: 0.5 }}
        />
      ))}
    </div>
  );
}
