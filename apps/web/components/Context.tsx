import type { FixtureRun, PlayerContext, SeasonLine } from "@/lib/api";

/**
 * Where a player is coming from, and what he is walking into.
 *
 * **Why this exists.** Every other number on the page is about next week. Two
 * players sitting on the same projection are not the same asset if one is
 * coming off eighteen goals and the other off eight, or if one opens against
 * the promoted sides and the other against last season's top four. The
 * projection knows all of that; the screen was not saying any of it, so a reader
 * comparing two rows had a decimal and nothing behind it.
 *
 * **These are retrievals, not projections.** Season lines are sums over matches
 * that happened; the run is the published schedule. That is why they are drawn
 * in chalk rather than in the positional colour — colour on this product means
 * a modelled quantity, and borrowing it here would present a count as a
 * forecast.
 */

const DIFFICULTY_SHADE: Record<number, string> = {
  1: "rgba(126, 200, 148, 0.55)",
  2: "rgba(126, 200, 148, 0.30)",
  3: "var(--color-line)",
  4: "rgba(224, 122, 122, 0.30)",
  5: "rgba(224, 122, 122, 0.55)",
};

export function SeasonSummary({ line }: { line: SeasonLine }) {
  return (
    <li className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1 py-1.5">
      <span className="tnum text-[13px] text-dim">{line.season}</span>
      <span className="text-[13px]" style={{ color: "var(--color-text)" }}>
        {line.sentence.replace(`${line.season}: `, "")}
      </span>
    </li>
  );
}

/**
 * The fixtures themselves, not a single difficulty average.
 *
 * An average of 3.0 covers both a flat run of even games and one that alternates
 * between the champions and a promoted side, and those call for different
 * decisions. Showing the sequence lets a reader see which one it is.
 */
export function FixtureTicker({ run }: { run: FixtureRun }) {
  if (run.fixtures.length === 0) return null;

  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {run.fixtures.map((fixture) => (
          <span
            key={`${fixture.gameweek}-${fixture.opponent}-${fixture.is_home ? "h" : "a"}`}
            className="tnum inline-flex flex-col items-center rounded-[3px] px-2 py-1"
            style={{
              background:
                fixture.difficulty === null
                  ? "var(--color-line)"
                  : (DIFFICULTY_SHADE[fixture.difficulty] ?? "var(--color-line)"),
            }}
            title={
              fixture.difficulty === null
                ? `GW${fixture.gameweek} — difficulty not yet published`
                : `GW${fixture.gameweek} — difficulty ${fixture.difficulty} of 5`
            }
          >
            <span className="text-[11px] leading-tight text-dim">GW{fixture.gameweek}</span>
            <span className="text-[12px] leading-tight">{fixture.label}</span>
          </span>
        ))}
      </div>

      <p className="mt-2.5 text-[12px]" style={{ color: "var(--color-muted)" }}>
        {run.mean_difficulty === null ? (
          // Preseason the API publishes no difficulty. Saying so beats printing
          // a zero, which would read as the easiest schedule in the league.
          <>Difficulty not yet published for this run.</>
        ) : (
          <>
            {run.home_count} of {run.fixtures.length} at home, {run.mean_difficulty} average
            difficulty.
          </>
        )}
        {run.blanks.length > 0 && <> Blank in GW{run.blanks.join(", GW")}.</>}
        {run.doubles.length > 0 && <> Double in GW{run.doubles.join(", GW")}.</>}
      </p>
    </div>
  );
}

export function ContextPanel({ context }: { context: PlayerContext | null }) {
  if (!context || (context.seasons.length === 0 && !context.run)) return null;

  // Most recent first: what he did last season is the question, and what he did
  // three seasons ago is the footnote.
  const seasons = [...context.seasons].reverse();

  return (
    <div className="space-y-7">
      {seasons.length > 0 && (
        <div>
          <p className="eyebrow">What he has actually produced</p>
          <ul className="mt-2.5 divide-y" style={{ borderColor: "var(--color-line)" }}>
            {seasons.map((line) => (
              <SeasonSummary key={line.season} line={line} />
            ))}
          </ul>
          {seasons.every((line) => line.per_90 === null) && (
            <p className="mt-2 text-[12px]" style={{ color: "var(--color-dim)" }}>
              Too few minutes to quote a per-90 rate.
            </p>
          )}
        </div>
      )}

      {context.run && (
        <div>
          <p className="eyebrow">What he is walking into</p>
          <div className="mt-2.5">
            <FixtureTicker run={context.run} />
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * One line for a collapsed row, so context is visible while *comparing* players
 * rather than only after opening one of them.
 */
export function LastSeasonLine({ context }: { context: PlayerContext | null }) {
  const seasons = context?.seasons ?? [];
  if (seasons.length === 0) return null;

  const last = seasons[seasons.length - 1];
  return (
    <span className="text-[12px]" style={{ color: "var(--color-dim)" }}>
      {last.sentence}
    </span>
  );
}
