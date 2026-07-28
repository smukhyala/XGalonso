"use client";

import { POSITION_COLOR, type SquadPlayer } from "@/lib/api";

/**
 * The squad on a pitch, in the formation the system actually chose.
 *
 * Formation is a decision this product makes, so it is drawn rather than
 * stated — a 3-5-2 and a 4-4-2 should look different at a glance. Each player
 * is a token with a *conviction ring*: the arc's completeness is the modelled
 * probability that he starts, which is the quantity that dominates FPL scoring
 * and the one most worth seeing before a transfer.
 */

/*
  Geometry. Attacking direction is up the page, so this is *our* half: the goal
  we defend sits at the bottom with the keeper inside his own penalty area, and
  the halfway line closes the top. Drawing the box at the far end instead would
  put the keeper nowhere near a goal.
*/
const ROWS: { position: SquadPlayer["position"]; y: number }[] = [
  { position: "GKP", y: 94 },
  { position: "DEF", y: 74 },
  { position: "MID", y: 50 },
  { position: "FWD", y: 24 },
];

const GOAL_LINE = 106;
const BENCH_LABEL_Y = 118;
const BENCH_Y = 125;

// Token size. Dropped from 4.6 because at that radius eleven tokens crowded the
// pitch and the shape — which is the whole reason the XI is drawn rather than
// listed — stopped being legible behind them.
const RADIUS = 3.4;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function Token({
  player,
  x,
  y,
  index,
}: {
  player: SquadPlayer;
  x: number;
  y: number;
  index: number;
}) {
  const colour = POSITION_COLOR[player.position];
  const conviction = Math.max(0, Math.min(1, player.p_start));
  const delay = 0.35 + index * 0.045;

  return (
    <g
      className="rise"
      style={{ animationDelay: `${delay}s` }}
      tabIndex={0}
      role="img"
      aria-label={`${player.name}, ${player.position}, ${player.expected_points.toFixed(
        1,
      )} projected points, ${Math.round(conviction * 100)}% chance of starting`}
    >
      <circle cx={x} cy={y} r={RADIUS + 1.1} fill="var(--color-pitch)" opacity={0.85} />

      {/* Conviction ring: how much of the circle is drawn is p(start). */}
      <circle
        cx={x}
        cy={y}
        r={RADIUS}
        fill="none"
        stroke="var(--color-line)"
        strokeWidth={0.9}
      />
      <circle
        cx={x}
        cy={y}
        r={RADIUS}
        fill="none"
        stroke={colour}
        strokeWidth={0.9}
        strokeLinecap="round"
        strokeDasharray={`${CIRCUMFERENCE * conviction} ${CIRCUMFERENCE}`}
        transform={`rotate(-90 ${x} ${y})`}
        style={{ transition: "stroke-dasharray 700ms cubic-bezier(0.16,1,0.3,1)" }}
      />

      <circle cx={x} cy={y} r={RADIUS - 0.85} fill={colour} opacity={0.13} />

      <text
        x={x}
        y={y + 0.9}
        textAnchor="middle"
        className="tnum"
        style={{ fontSize: 2.4, fill: colour, fontWeight: 600 }}
      >
        {player.expected_points.toFixed(1)}
      </text>

      <text
        x={x}
        y={y + RADIUS + 3.4}
        textAnchor="middle"
        style={{ fontSize: 2.15, fill: "var(--color-chalk)", letterSpacing: "0.02em" }}
      >
        {player.name.length > 12 ? `${player.name.slice(0, 11)}…` : player.name}
      </text>

      {(player.is_captain || player.is_vice_captain) && (
        <>
          <circle
            cx={x + RADIUS - 0.2}
            cy={y - RADIUS + 0.2}
            r={1.65}
            fill={player.is_captain ? colour : "var(--color-pitch)"}
            stroke={colour}
            strokeWidth={0.6}
          />
          <text
            x={x + RADIUS - 0.2}
            y={y - RADIUS + 0.85}
            textAnchor="middle"
            style={{
              fontSize: 1.75,
              fontWeight: 700,
              fill: player.is_captain ? "var(--color-pitch)" : colour,
            }}
          >
            {player.is_captain ? "C" : "V"}
          </text>
        </>
      )}
    </g>
  );
}

export function Pitch({ players }: { players: SquadPlayer[] }) {
  const starters = players.filter((p) => p.is_starter);
  const bench = players.filter((p) => !p.is_starter);

  let tokenIndex = 0;

  return (
    <figure className="w-full">
      <svg viewBox="0 0 100 137" className="w-full" aria-label="Starting eleven by position">
        <defs>
          <radialGradient id="turf" cx="50%" cy="35%">
            <stop offset="0%" stopColor="#13202a" />
            <stop offset="100%" stopColor="#0c151b" />
          </radialGradient>
        </defs>

        {/* The playing surface. Curved at the top so the frame is not a box. */}
        <path
          d={`M 6 ${GOAL_LINE} L 6 26 Q 6 6 26 6 L 74 6 Q 94 6 94 26 L 94 ${GOAL_LINE} Z`}
          fill="url(#turf)"
          stroke="var(--color-line)"
          strokeWidth={0.5}
        />

        {/* Our end: penalty area, six-yard box and the D, behind the keeper. */}
        <path
          d={`M 28 ${GOAL_LINE} L 28 84 L 72 84 L 72 ${GOAL_LINE}`}
          fill="none"
          stroke="var(--color-line)"
          strokeWidth={0.4}
        />
        <path
          d={`M 41 ${GOAL_LINE} L 41 98 L 59 98 L 59 ${GOAL_LINE}`}
          fill="none"
          stroke="var(--color-line)"
          strokeWidth={0.4}
        />
        <path d="M 41 84 A 9 9 0 0 1 59 84" fill="none" stroke="var(--color-line)" strokeWidth={0.4} />
        <path
          d={`M 44 ${GOAL_LINE} L 44 ${GOAL_LINE + 3} L 56 ${GOAL_LINE + 3} L 56 ${GOAL_LINE}`}
          fill="none"
          stroke="var(--color-dim)"
          strokeWidth={0.5}
        />

        {/* Halfway line closes the top; the rest of the pitch runs off-frame. */}
        <path d="M 38 6 A 12 12 0 0 0 62 6" fill="none" stroke="var(--color-line)" strokeWidth={0.4} />

        {ROWS.map(({ position, y }) => {
          const row = starters.filter((p) => p.position === position);
          return row.map((player, i) => {
            const span = 100 / (row.length + 1);
            const node = (
              <Token
                key={player.player_code}
                player={player}
                x={span * (i + 1)}
                y={y}
                index={tokenIndex}
              />
            );
            tokenIndex += 1;
            return node;
          });
        })}

        {/* Bench sits outside the touchline, which is where it belongs. */}
        <text
          x="6"
          y={BENCH_LABEL_Y}
          className="eyebrow"
          style={{ fontSize: 2.4, fill: "var(--color-dim)" }}
        >
          BENCH — IN SUBSTITUTION ORDER
        </text>
        {bench.map((player, i) => {
          const span = 100 / (bench.length + 1);
          const node = (
            <Token
              key={player.player_code}
              player={player}
              x={span * (i + 1)}
              y={BENCH_Y}
              index={tokenIndex}
            />
          );
          tokenIndex += 1;
          return node;
        })}
      </svg>
    </figure>
  );
}
