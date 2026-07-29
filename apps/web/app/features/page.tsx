"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import {
  api,
  importanceText,
  labelName,
  type FeatureImportance,
  type ImportanceResponse,
} from "@/lib/api";
import { explainFamily, explainFeature } from "@/lib/glossary";

/** The pooled slice. Mirrors `ALL_POSITIONS` in the evaluation package. */
const ALL_POSITIONS = "ALL";

/**
 * The Feature Lab: which of the catalogue's features actually earn their place.
 *
 * The measurement is permutation importance on walk-forward validation rows —
 * each feature is shuffled in turn and the model's out-of-sample error is
 * watched. So a feature that only ever helped by memorising shows nothing here.
 *
 * **Colour discipline.** The rest of this product colours by position and
 * nothing else, so a feature — which has no position — is drawn in chalk. What
 * varies instead is *solidity*: a feature whose rank swings between folds is
 * rendered fainter than one that lands in the same place every time. Certainty
 * is the second dimension worth seeing, and opacity carries it without
 * inventing a colour that would collide with the positional system.
 */
export default function FeaturesPage() {
  const [data, setData] = useState<ImportanceResponse | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [position, setPosition] = useState<string>(ALL_POSITIONS);
  // Held separately from `data.positions` so the selector does not vanish when a
  // slice returns nothing and `data` goes null.
  const [available, setAvailable] = useState<string[]>([ALL_POSITIONS]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  const load = useCallback(async (nextLabel: string | null, nextPosition: string) => {
    setBusy(true);
    setError(null);
    try {
      const response = await api.importance(nextLabel ?? undefined, 40, nextPosition);
      setData(response);
      if (response.positions.length > 0) setAvailable(response.positions);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load the table.");
      setData(null);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void load(label, position);
  }, [label, position, load]);

  return (
    <main className="mx-auto max-w-6xl px-6 pb-28 sm:px-10">
      <Masthead />

      <header className="mt-14 max-w-3xl">
        <p className="eyebrow">Feature lab</p>
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
          Which features earn their place.
        </h1>
        <p className="mt-5 text-[15px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
          Each feature is shuffled in turn on gameweeks the model never trained on, and the
          damage to its out-of-sample error is recorded. A feature that only helped by
          memorising the training set does nothing here — which is the point of measuring it
          this way rather than reading importances off the fit.
        </p>
      </header>

      {(error || data) && (
        <PositionFilter active={position} available={available} onChange={setPosition} />
      )}

      {error && <Empty message={error} />}

      {data && (
        <>
          <Summary data={data} />

          <Findings data={data} />

          <PointsMix data={data} />

          <Filters
            labels={data.labels}
            weights={data.label_weights}
            active={label}
            onChange={setLabel}
          />

          <Ranking data={data} busy={busy} />

          <Families data={data} />

          <Caveats data={data} />
        </>
      )}

      {!data && !error && busy && <Skeleton />}
    </main>
  );
}

/**
 * Family names, shortened for display.
 *
 * Every family but `opponent` and `fixture` is prefixed `player_`, so the
 * prefix distinguishes nothing and costs the column its width — enough that the
 * label ran into the figure beside it. Dropped here rather than in the data,
 * because the stored name is what `xg importance` prints and what the parquet
 * records.
 */
function familyName(family: string): string {
  return family.replace(/^player_/, "").replace(/_/g, " ");
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
          XG ALONSO
        </span>
        <span className="eyebrow">← back to the call</span>
      </Link>
      <nav className="flex gap-6">
        <Link href="/discovery" className="eyebrow transition-opacity hover:opacity-70">
          Discovery
        </Link>
      </nav>
    </header>
  );
}

function Summary({ data }: { data: ImportanceResponse }) {
  const measured = data.features_measured;
  const dead = data.features_with_no_effect;
  return (
    <section className="rise mt-12" style={{ animationDelay: "0.1s" }}>
      <div className="hairline" />
      <dl className="mt-6 flex flex-wrap gap-x-10 gap-y-5">
        <Stat label="Features measured" value={String(measured)} />
        <Stat label="No measurable effect" value={String(dead)} />
        <Stat label="Folds" value={String(data.folds_measured)} />
        <Stat label="Catalogue" value={data.catalogue_version} />
      </dl>
      {data.stale && (
        <p
          role="alert"
          className="mt-5 max-w-2xl text-[13px] leading-relaxed"
          style={{ color: "var(--color-loss)" }}
        >
          These numbers were measured against a different model than the one currently
          loaded. Re-run <code className="tnum">xg importance</code> before trusting them.
        </p>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className="tnum mt-1.5 text-lg" style={{ letterSpacing: "-0.02em" }}>
        {value}
      </dd>
    </div>
  );
}

/**
 * What the run actually found, before the table of numbers.
 *
 * The ranking was accurate and arrived without a thesis: eighty rows of column
 * names, ordered, with no statement of what the ordering *meant*. A reader had
 * to reverse-engineer the finding from the data, which is work the page should
 * have done.
 *
 * Every figure here is computed from the response rather than written down, so
 * the summary cannot drift from the table beneath it.
 */
function Findings({ data }: { data: ImportanceResponse }) {
  const ranked = data.features;
  if (ranked.length === 0) return null;

  const top = ranked[0];
  const topFamily = Object.entries(data.families).sort((a, b) => b[1] - a[1])[0];
  const total = Object.values(data.families).reduce((sum, value) => sum + value, 0);
  const familyShare = topFamily && total > 0 ? topFamily[1] / total : 0;

  // How concentrated the ranking is: what share of all measured importance the
  // top five features carry. A high number means a few features decide
  // everything, which is itself the finding.
  const headSum = ranked.slice(0, 5).reduce((sum, f) => sum + Math.max(f.importance, 0), 0);
  const allSum = ranked.reduce((sum, f) => sum + Math.max(f.importance, 0), 0);
  const concentration = allSum > 0 ? headSum / allSum : 0;

  const deadShare = data.features_measured
    ? data.features_with_no_effect / data.features_measured
    : 0;

  const unstable = ranked.filter(
    (f) => f.rank_stability !== null && f.rank_stability > data.features_measured / 8,
  ).length;

  return (
    <section className="rise mt-14" style={{ animationDelay: "0.12s" }}>
      <p className="eyebrow">What this run found</p>

      <div
        className="mt-5 max-w-3xl space-y-4 text-[15px] leading-relaxed"
        style={{ color: "var(--color-muted)" }}
      >
        <p>
          Each of the {data.features_measured} features was shuffled in turn across{" "}
          {data.folds_measured} walk-forward validation windows — gameweeks the model was
          never fitted on — and the damage to its out-of-sample error recorded. A feature
          that only helped by memorising the training set does nothing under that test,
          which is the point of running it this way.
        </p>

        <p>
          <span style={{ color: "var(--color-chalk)" }}>
            {top.feature_name} came first, at {importanceText(top.importance)}.
          </span>{" "}
          The top five carry {Math.round(concentration * 100)}% of all measured importance,
          so this is not a broad consensus of many small signals — a handful of features
          decide the projection and the rest adjust it.
          {topFamily && (
            <>
              {" "}
              The {familyName(topFamily[0])} family alone accounts for{" "}
              {Math.round(familyShare * 100)}% of the total.
            </>
          )}
        </p>

        <p>
          <span style={{ color: "var(--color-chalk)" }}>
            {data.features_with_no_effect} of {data.features_measured} features (
            {Math.round(deadShare * 100)}%) did not improve out-of-sample error at all.
          </span>{" "}
          That is expected rather than alarming, and there are three separate reasons for
          it — worth telling apart, because only one of them means a feature is useless.
        </p>

        <ul className="space-y-3 pt-1">
          <Cause
            title="It duplicates a feature that won"
            body="A rolling mean over three appearances and one over five carry nearly the same signal. Shuffling either leaves the model able to recover from the other, so both score near zero while the pair together matters a great deal. This is why the family totals are shown below — they are the only place that signal is visible."
          />
          <Cause
            title="It describes something the model already sees another way"
            body="Composite ratings overlap the counting stats they were built from. Once goals, assists and minutes are present, an index summarising them has little left to add."
          />
          <Cause
            title="It genuinely carries nothing"
            body="Rare events and market noise. A feature counting something that happens a handful of times a season cannot move a projection, however sensible it looks in a list."
          />
        </ul>

        {unstable > 0 && (
          <p>
            <span style={{ color: "var(--color-chalk)" }}>
              {unstable} features rank inconsistently between folds.
            </span>{" "}
            They are shown faded in the table below. A feature that places third on one
            window and near the bottom on the next has not been shown to matter — it has
            been shown to be noisy, and the two are easy to confuse when only an average
            is displayed.
          </p>
        )}
      </div>
    </section>
  );
}

function Cause({ title, body }: { title: string; body: string }) {
  return (
    <li className="flex gap-3.5">
      <span
        aria-hidden
        className="mt-[0.6rem] h-px w-4 shrink-0"
        style={{ background: "var(--color-dim)" }}
      />
      <span className="text-[14px] leading-relaxed">
        <span style={{ color: "var(--color-chalk)" }}>{title}.</span> {body}
      </span>
    </li>
  );
}

function Filters({
  labels,
  weights,
  active,
  onChange,
}: {
  labels: string[];
  weights: Record<string, number>;
  active: string | null;
  onChange: (label: string | null) => void;
}) {
  // Ordered by how much each component moves a points total, so the list reads
  // as a hierarchy of consequence rather than as alphabetical trivia.
  const ordered = [...labels].sort((a, b) => (weights[b] ?? 0) - (weights[a] ?? 0));

  return (
    <nav className="rise mt-14" style={{ animationDelay: "0.15s" }} aria-label="Component">
      <p className="eyebrow">Measured against</p>
      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2.5">
        <Chip label="Everything, weighted" active={active === null} onClick={() => onChange(null)} />
        {ordered.map((label) => (
          <Chip
            key={label}
            label={labelName(label)}
            active={active === label}
            onClick={() => onChange(label)}
          />
        ))}
      </div>
    </nav>
  );
}

/**
 * How a position reads to a person, and the order a squad is listed in.
 *
 * Both `GK` and `GKP` appear: the archive spells it `GK` and the live bootstrap
 * spells it `GKP`, so a table measured across seasons can carry either. Mapping
 * both beats normalising one away, which would make the label disagree with the
 * value the API actually filters on.
 */
const POSITION_LABELS: Record<string, string> = {
  ALL: "Every position",
  GK: "Goalkeepers",
  GKP: "Goalkeepers",
  DEF: "Defenders",
  MID: "Midfielders",
  FWD: "Forwards",
};

const POSITION_ORDER = ["ALL", "GK", "GKP", "DEF", "MID", "FWD"];

/**
 * Which players the ranking below actually describes.
 *
 * This is the most consequential control on the page and it is deliberately
 * first. A pooled ranking answers "what predicts a footballer's return", which
 * is a question nobody asks — minutes played dominates a striker's goals, and
 * for a defender the same minutes matter through clean sheets. The two are
 * different rankings, and reading one while thinking about the other is the
 * specific mistake this fixes.
 *
 * These are separate *measurements*, not a reweighting of pooled numbers: a
 * defender's ranking comes from permuting features on defenders' rows.
 */
function PositionFilter({
  active,
  available,
  onChange,
}: {
  active: string;
  available: string[];
  onChange: (position: string) => void;
}) {
  const ordered = POSITION_ORDER.filter((position) => available.includes(position));
  const extras = available.filter((position) => !POSITION_ORDER.includes(position));
  const shown = [...ordered, ...extras];

  if (shown.length <= 1) {
    return (
      <p className="mt-12 text-[13px]" style={{ color: "var(--color-muted)" }}>
        Measured across every position together. Re-run <code>xg importance</code> to
        measure each position separately.
      </p>
    );
  }

  return (
    <nav className="rise mt-12" aria-label="Position" style={{ animationDelay: "0.1s" }}>
      <p className="eyebrow">Ranking for</p>
      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2.5">
        {shown.map((position) => (
          <Chip
            key={position}
            label={POSITION_LABELS[position] ?? position}
            active={active === position}
            onClick={() => onChange(position)}
          />
        ))}
      </div>
    </nav>
  );
}

/**
 * What this position's points are actually made of.
 *
 * The ranking above is weighted by how much each component moves a points
 * total, and that mix is wildly different by position — a goalkeeper's points
 * are twelve percent saves and one percent goals; a forward's are the reverse.
 * Showing the mix is what makes the ranking above readable, and it is the
 * clearest evidence that these are separate measurements rather than one
 * ranking relabelled four times.
 *
 * It also explains the part that surprises people: minutes leads every
 * position. That is not a flattening bug — a player who does not play scores
 * nothing whatever his position, so minutes gates every other component. The
 * position-specific answer lives in the component filter below.
 */
function PointsMix({ data }: { data: ImportanceResponse }) {
  const mix = Object.entries(data.label_weights)
    .filter(([, weight]) => weight > 0.001)
    .sort((a, b) => b[1] - a[1]);

  if (mix.length === 0) return null;

  const appearance = mix
    .filter(([name]) => name === "label_minutes" || name === "label_starts")
    .reduce((sum, [, weight]) => sum + weight, 0);

  return (
    <section className="rise mt-12" style={{ animationDelay: "0.12s" }}>
      <p className="eyebrow">
        {data.position === ALL_POSITIONS
          ? "What points are made of"
          : `What ${(POSITION_LABELS[data.position] ?? data.position).toLowerCase()} are paid for`}
      </p>

      <div className="mt-4 flex h-2 w-full overflow-hidden" style={{ background: "var(--color-line)" }}>
        {mix.map(([name, weight], index) => (
          <span
            key={name}
            title={`${labelName(name)} — ${(weight * 100).toFixed(1)}%`}
            style={{
              width: `${weight * 100}%`,
              background: "var(--color-text)",
              opacity: 0.85 - index * 0.09,
            }}
          />
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5">
        {mix.slice(0, 6).map(([name, weight]) => (
          <span key={name} className="text-[12px]" style={{ color: "var(--color-muted)" }}>
            {labelName(name)}{" "}
            <span className="tnum" style={{ color: "var(--color-dim)" }}>
              {(weight * 100).toFixed(1)}%
            </span>
          </span>
        ))}
      </div>

      <p className="mt-3 max-w-2xl text-[12px] leading-relaxed" style={{ color: "var(--color-dim)" }}>
        Minutes and starts together are {(appearance * 100).toFixed(0)}% of the mix, which is
        why appearance features lead the ranking for every position — a player who does not
        play scores nothing whatever he is. Pick a component below to see what separates the
        positions.
      </p>
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
      aria-pressed={active}
      className="border-b pb-0.5 text-[13px] transition-colors"
      style={{
        color: active ? "var(--color-chalk)" : "var(--color-dim)",
        borderColor: active ? "var(--color-chalk)" : "transparent",
      }}
    >
      {label}
    </button>
  );
}

function Ranking({ data, busy }: { data: ImportanceResponse; busy: boolean }) {
  const [open, setOpen] = useState<string | null>(null);
  const ceiling = Math.max(...data.features.map((f) => f.importance), 1e-9);

  // Rank spread is in places out of the field measured; anything past a fifth of
  // the field is noise for display purposes and clamps to fully faint.
  const worst = Math.max(data.features_measured / 5, 1);

  /*
    Bar length is square-rooted.

    Importance here is a power law: minutes are worth several times the next
    feature and several hundred times the tail, because minutes gate every other
    component. On a linear scale the top bar fills the row and every other bar
    collapses to a sliver, so the chart shows one fact and hides the seventy-nine
    below it. A square root keeps the order exactly and makes the tail legible.

    The exact figures stay in the column on the right, so nothing is lost —
    only the visual ratio is compressed, and the page says so beneath.
  */
  const scale = (value: number) =>
    value <= 0 ? 0 : Math.sqrt(value / ceiling) * 100;

  return (
    <section className="mt-12" style={{ opacity: busy ? 0.4 : 1, transition: "opacity 0.2s" }}>
      <div className="hairline" />
      <ol className="mt-2">
        {data.features.map((feature, index) => {
          const width = scale(feature.importance);
          const solidity =
            feature.rank_stability === null
              ? 0.55
              : Math.max(0.18, 1 - Math.min(feature.rank_stability / worst, 1) * 0.82);

          const isOpen = open === feature.feature_name;
          return (
            <li
              key={feature.feature_name}
              className="border-b"
              style={{ borderColor: "var(--color-line)" }}
            >
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : feature.feature_name)}
                aria-expanded={isOpen}
                className="grid w-full grid-cols-[2.25rem_1fr_auto] items-center gap-x-5 gap-y-1 py-3 text-left transition-opacity hover:opacity-80 sm:grid-cols-[2.25rem_15rem_1fr_7rem_4.75rem]"
              >
                <span className="tnum text-xs" style={{ color: "var(--color-dim)" }}>
                  {String(index + 1).padStart(2, "0")}
                </span>

                <span className="tnum text-[13px] break-all">{feature.feature_name}</span>

                <span
                  className="hidden h-[3px] sm:block"
                  style={{ background: "var(--color-line)" }}
                >
                  <span
                    className="block h-full"
                    style={{
                      width: `${width}%`,
                      background: "var(--color-chalk)",
                      opacity: solidity,
                    }}
                  />
                </span>

                <span className="eyebrow hidden truncate sm:block" title={feature.family}>
                  {familyName(feature.family)}
                </span>

                <span className="tnum text-right text-[13px]">
                  {importanceText(feature.importance)}
                </span>
              </button>

              {isOpen && <FeatureDetail feature={feature} data={data} />}
            </li>
          );
        })}
      </ol>
      <p className="mt-5 max-w-2xl text-[13px] leading-relaxed" style={{ color: "var(--color-dim)" }}>
        The percentage is how much worse the model got at predicting points when that
        feature was shuffled — 3.6% means the error grew by 3.6%. Bars fade as a
        feature&rsquo;s rank moves between folds: a solid bar landed in roughly the same
        place every time, a faint one happened to look important once. Lengths are
        square-root scaled, since importance follows a power law here and on a linear
        axis everything below the top feature disappears.
      </p>
      {data.features_measured > data.features.length && (
        <p className="eyebrow mt-3">
          Showing the top {data.features.length} of {data.features_measured} measured
        </p>
      )}
    </section>
  );
}

/**
 * What a feature means, and why it moves a score.
 *
 * The ranking on its own was accurate and unreadable: `bps_std_20` above
 * `influence_per90_10` tells a reader nothing about either. The number says
 * which mattered; this says what they are, which is the part that makes the
 * ordering interpretable rather than trivia.
 */
function FeatureDetail({
  feature,
  data,
}: {
  feature: FeatureImportance;
  data: ImportanceResponse;
}) {
  const metric = explainFeature(feature.feature_name);
  const family = explainFamily(feature.family);
  const perLabel = Object.entries(feature.per_label).sort((a, b) => b[1] - a[1]);
  const ceiling = Math.max(...perLabel.map(([, v]) => Math.abs(v)), 1e-9);

  return (
    <div className="grid gap-8 pb-8 pt-1 lg:grid-cols-2">
      <div className="max-w-prose space-y-4 text-[14px] leading-relaxed">
        {metric ? (
          <>
            <p style={{ color: "var(--color-chalk)" }}>{metric.what}</p>
            <p style={{ color: "var(--color-muted)" }}>{metric.why}</p>
          </>
        ) : (
          <p style={{ color: "var(--color-muted)" }}>
            No description recorded for this feature yet. It belongs to the{" "}
            {familyName(feature.family)} family: {family.what.toLowerCase()}
          </p>
        )}

        <div className="pt-1">
          <p className="eyebrow">Family — {familyName(feature.family)}</p>
          <p className="mt-2 text-[13px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
            {family.why}
          </p>
        </div>

        <p className="text-[13px]" style={{ color: "var(--color-dim)" }}>
          {feature.rank_stability === null
            ? "Rank stability was not measured — that needs at least two folds with rows."
            : `Its rank moved by about ${feature.rank_stability.toFixed(0)} places between folds, out of ${data.features_measured} features.`}
        </p>
      </div>

      <div>
        <p className="eyebrow">Where it helps</p>
        <ul className="mt-4 space-y-2.5">
          {perLabel.map(([label, value]) => (
            <li key={label} className="grid grid-cols-[7.5rem_1fr_4rem] items-center gap-3">
              <span className="text-[13px]" style={{ color: "var(--color-muted)" }}>
                {labelName(label)}
              </span>
              <span className="h-[3px]" style={{ background: "var(--color-line)" }}>
                <span
                  className="block h-full"
                  style={{
                    width: `${(Math.abs(value) / ceiling) * 100}%`,
                    background: value >= 0 ? "var(--color-chalk)" : "var(--color-loss)",
                    opacity: 0.7,
                  }}
                />
              </span>
              <span className="tnum text-right text-[12px]">{importanceText(value)}</span>
            </li>
          ))}
        </ul>
        <p className="mt-4 max-w-xs text-[12px] leading-relaxed" style={{ color: "var(--color-dim)" }}>
          How much worse each component model got when this feature was shuffled. A red
          bar means the model predicted that component <em>better</em> without it — the
          feature was actively misleading it there.
        </p>
      </div>
    </div>
  );
}

function Families({ data }: { data: ImportanceResponse }) {
  const [openFamily, setOpenFamily] = useState<string | null>(null);
  const entries = Object.entries(data.families).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return null;
  const ceiling = Math.max(...entries.map(([, value]) => value), 1e-9);
  const scale = (value: number) => (value <= 0 ? 0 : Math.sqrt(value / ceiling) * 100);

  return (
    <section className="mt-24">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="eyebrow">By family</p>
        <p className="eyebrow">Open a row for what it means</p>
      </div>
      <div className="hairline mt-5" />

      <ul className="mt-2">
        {entries.map(([family, value]) => {
          const isOpen = openFamily === family;
          const meaning = explainFamily(family);
          return (
            <li key={family} className="border-b" style={{ borderColor: "var(--color-line)" }}>
              <button
                type="button"
                onClick={() => setOpenFamily(isOpen ? null : family)}
                aria-expanded={isOpen}
                className="grid w-full grid-cols-[1fr_auto] items-center gap-4 py-3.5 text-left transition-opacity hover:opacity-80 sm:grid-cols-[12rem_1fr_5.5rem]"
              >
                <span className="text-[14px]">{familyName(family)}</span>
                <span
                  className="hidden h-[3px] sm:block"
                  style={{ background: "var(--color-line)" }}
                >
                  <span
                    className="block h-full"
                    style={{
                      width: `${scale(value)}%`,
                      background: "var(--color-chalk)",
                      opacity: 0.7,
                    }}
                  />
                </span>
                <span className="tnum text-right text-[13px]">{importanceText(value)}</span>
              </button>
              {isOpen && (
                <div className="max-w-prose space-y-3 pb-7 pt-1 text-[14px] leading-relaxed">
                  <p style={{ color: "var(--color-chalk)" }}>{meaning.what}</p>
                  <p style={{ color: "var(--color-muted)" }}>{meaning.why}</p>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Caveats({ data }: { data: ImportanceResponse }) {
  return (
    <section className="mt-20 max-w-2xl">
      <div className="hairline" />
      <p className="eyebrow mt-6">How to read this</p>

      <div className="mt-4 space-y-4 text-[14px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
        <p>
          <span style={{ color: "var(--color-chalk)" }}>
            Correlated features split their score.
          </span>{" "}
          A rolling mean over three appearances and one over five carry almost the same
          signal, so shuffling either leaves the model able to recover from the other and
          both rank low. That is why the family totals are here: they say whether a
          <em> kind</em> of feature matters even when no single column looks decisive.
        </p>

        <p>
          <span style={{ color: "var(--color-chalk)" }}>Minutes are underrated by this method.</span>{" "}
          Weighting each component by its share of a points total credits minutes only with
          the appearance points, when in practice minutes gate every other component — a
          player who does not play scores nothing anywhere. Read the minutes features as a
          floor.
        </p>

        {data.degenerate_labels.length > 0 && (
          <p>
            <span style={{ color: "var(--color-loss)" }}>
              {data.degenerate_labels.length} component
              {data.degenerate_labels.length === 1 ? "" : "s"} output a constant
            </span>{" "}
            and {data.degenerate_labels.length === 1 ? "is" : "are"} excluded:{" "}
            {data.degenerate_labels.map(labelName).join(", ")}. Ranking features by their
            effect on a constant would be arithmetic without meaning.
          </p>
        )}

        <p style={{ color: "var(--color-dim)" }}>
          Measured on {data.folds_measured} walk-forward fold
          {data.folds_measured === 1 ? "" : "s"} against model{" "}
          <span className="tnum">{data.model_fingerprint.slice(0, 12)}</span> on{" "}
          {new Date(data.computed_at).toISOString().slice(0, 10)}.
        </p>
      </div>
    </section>
  );
}

function Empty({ message }: { message: string }) {
  return (
    <section className="mt-14 max-w-2xl">
      <div className="hairline" />
      <p className="mt-6 text-[15px] leading-relaxed">{message}</p>
      <p className="mt-4 text-[14px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
        Measure the catalogue with:
      </p>
      <pre
        className="tnum mt-3 overflow-x-auto p-4 text-[13px]"
        style={{ background: "var(--color-surface)", color: "var(--color-chalk)" }}
      >
        xg importance --model .data/models/late.pkl
      </pre>
    </section>
  );
}

function Skeleton() {
  return (
    <div className="mt-14 space-y-3" aria-busy>
      {Array.from({ length: 8 }).map((_, index) => (
        <div
          key={index}
          className="h-8 animate-pulse"
          style={{ background: "var(--color-line)", opacity: 1 - index * 0.1 }}
        />
      ))}
    </div>
  );
}
