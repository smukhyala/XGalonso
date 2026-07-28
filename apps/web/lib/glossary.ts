/**
 * Plain-English explanations for every metric and family the product shows.
 *
 * The Feature Lab listed `bps_std_20` next to `influence_per90_10` and expected
 * a reader to know what either meant, let alone why one outranked the other.
 * The ranking was honest and unreadable — a table of jargon with numbers beside
 * it communicates precision, not meaning.
 *
 * Two things are defined here, because a reader needs both:
 *
 * - **What the metric is.** What it measures, in words, without reference to
 *   the column name.
 * - **Why it moves the score.** The mechanism by which it affects FPL points,
 *   which is the part that makes a ranking interpretable rather than trivia.
 *
 * Feature names are generated — `{metric}_{aggregation}_{window}` — so the
 * lookup decomposes a name rather than enumerating 180 entries, which would
 * fall out of date the first time a window was added.
 */

export interface Explanation {
  /** What it measures. */
  what: string;
  /** Why it moves an FPL score. */
  why: string;
}

/** The base metrics, before aggregation and window are applied. */
const METRICS: Record<string, Explanation> = {
  minutes: {
    what: "Minutes played.",
    why: "Minutes gate everything. A player who does not appear scores nothing at all, so this sits underneath every other component rather than beside it — which is why it dominates the ranking.",
  },
  starts: {
    what: "Whether he started, rather than came off the bench.",
    why: "A starter is far more likely to reach 60 minutes, which is the threshold for the second appearance point and for a defender's clean sheet.",
  },
  goals_scored: {
    what: "Goals actually scored.",
    why: "Worth 10 points for a goalkeeper, 6 for a defender, 5 for a midfielder and 4 for a forward. The single largest term available to an attacking player.",
  },
  assists: { what: "Assists actually made.", why: "Three points each, at every position." },
  clean_sheets: {
    what: "Matches where his team conceded nothing while he was on the pitch for 60 minutes.",
    why: "Four points for a goalkeeper or defender, one for a midfielder, nothing for a forward. Most of a defender's floor.",
  },
  goals_conceded: {
    what: "Goals his team let in while he played.",
    why: "Minus one point for every two conceded, for goalkeepers and defenders. It also destroys the clean sheet, so it costs twice.",
  },
  saves: {
    what: "Shots stopped.",
    why: "One point per three saves, goalkeepers only. A keeper behind a weak defence can out-score a keeper behind a strong one on saves alone.",
  },
  yellow_cards: { what: "Bookings.", why: "Minus one point each." },
  bonus: {
    what: "Bonus points awarded after the match.",
    why: "One to three points to the best performers in each fixture, decided by the bonus points system below.",
  },
  bps: {
    what: "The bonus points system score — a raw tally of everything he did in the match, good and bad.",
    why: "Not worth points itself. It decides who receives the bonus, so a consistently high BPS is a reliable extra point or two a week that a goals-and-assists view misses entirely.",
  },
  total_points: {
    what: "FPL points actually scored.",
    why: "The outcome itself. Useful as a summary of form, and circular if leaned on too hard — it is the thing being predicted.",
  },
  defensive_contribution: {
    what: "Tackles, interceptions, clearances and blocks, combined into one count.",
    why: "New in 2025/26: two points when a defender reaches ten in a match, or a midfielder or forward reaches twelve. It gives a defensive player a scoring route that did not exist before.",
  },
  expected_goals: {
    what: "The quality of the chances he had, in goals. A tap-in counts near one, a speculative shot near zero.",
    why: "A better predictor of future goals than past goals, because finishing swings wildly from week to week while chance quality persists.",
  },
  expected_assists: {
    what: "The quality of the chances he created for others.",
    why: "Same logic as expected goals: it survives a teammate having a bad finishing week, which a raw assist count does not.",
  },
  expected_goal_involvements: {
    what: "Expected goals and expected assists added together.",
    why: "One number for total attacking output, useful when a player contributes through both routes.",
  },
  expected_goals_conceded: {
    what: "The quality of chances his team gave up while he played.",
    why: "How likely a clean sheet is, before luck. A defence conceding low-quality chances keeps more clean sheets than its goals-conceded record suggests.",
  },
  influence: {
    what: "An Opta rating of how much he affected the outcome of the match — goals, assists, saves, defensive actions and decisive moments combined.",
    why: "Not worth points directly. It captures involvement that no counting stat sees, which is why it earns a place despite being a composite.",
  },
  creativity: {
    what: "An Opta rating of his chance creation: passes and crosses that led to shots, weighted by quality.",
    why: "A leading indicator for assists. It moves before the assists do, because it counts the chances created whether or not they were finished.",
  },
  threat: {
    what: "An Opta rating of his goal danger, built from shot volume, shot location and quality.",
    why: "A leading indicator for goals, and the cleanest way to separate an attacking full-back from a stay-at-home centre-half.",
  },
  ict_index: {
    what: "Influence, creativity and threat combined into a single index.",
    why: "A general involvement score. Broad by design, so it rarely beats its own components on a specific question.",
  },
  selected: {
    what: "How many managers own him.",
    why: "Not a football statistic. It carries information about what the market believes, which sometimes leads price and team news.",
  },
  transfers_in: {
    what: "Managers buying him this gameweek.",
    why: "Often moves before public team news does, so a sharp rise can mean the market knows something the statistics do not yet show.",
  },
  transfers_out: { what: "Managers selling him this gameweek.", why: "The same signal, inverted." },
  transfers_balance: {
    what: "Transfers in minus transfers out.",
    why: "Net market direction in one number.",
  },
  value: { what: "His price.", why: "Not a performance measure; it constrains what else the squad can afford." },
};

const AGGREGATIONS: Record<string, string> = {
  mean: "averaged over his last",
  sum: "totalled over his last",
  max: "his best single return in the last",
  min: "his worst single return in the last",
  std: "how much it varied over his last",
  median: "the middle value over his last",
  per90: "per 90 minutes, over his last",
};

export const FAMILIES: Record<string, Explanation> = {
  player_performance: {
    what: "Rolling averages of what a player actually did — minutes, goals, points and the rest.",
    why: "The plainest description of form. It dominates the ranking because minutes live here, and minutes decide whether anything else can happen.",
  },
  player_rate: {
    what: "Output expressed per 90 minutes, shrunk toward the average so a short cameo cannot look elite.",
    why: "Separates a player who is genuinely productive from one who simply plays a lot. Necessary for comparing a rotation player with an ever-present.",
  },
  player_volume: {
    what: "Totals rather than averages — minutes and points accumulated over a window.",
    why: "Captures durability. Two players averaging the same per match are different assets if one has missed half of them.",
  },
  player_volatility: {
    what: "How much a player's returns swing from week to week.",
    why: "Two players averaging five points are not the same if one alternates two and eight. Volatility is what makes a captaincy pick risky.",
  },
  player_ceiling: {
    what: "His best return in a recent window.",
    why: "What he is capable of on his best day, which a mean hides. The case for a captain rests here.",
  },
  player_floor: {
    what: "His worst return in a recent window.",
    why: "The downside. What a safe pick is bought for.",
  },
  opponent: {
    what: "How the upcoming opponent has performed defensively — what they concede and how often they keep a clean sheet.",
    why: "Fixture difficulty, measured rather than assumed. The same player is worth meaningfully more against a leaky defence.",
  },
  fixture: {
    what: "Where the match is played.",
    why: "Home advantage is real but small, so it moves a projection at the margin rather than deciding it.",
  },
  unknown: {
    what: "Features not yet assigned a family.",
    why: "Usually a sign the family map has fallen behind a new generator.",
  },
};

/** Explain a generated feature name by decomposing it. */
export function explainFeature(name: string): Explanation | null {
  const window = name.match(/_(\d+)$/)?.[1];
  let stem = window ? name.slice(0, -(window.length + 1)) : name;

  let aggregation: string | null = null;
  for (const key of Object.keys(AGGREGATIONS)) {
    if (stem.endsWith(`_${key}`)) {
      aggregation = key;
      stem = stem.slice(0, -(key.length + 1));
      break;
    }
  }

  const metric = METRICS[stem];
  if (!metric) return null;

  if (!aggregation || !window) return metric;

  const matches = window === "1" ? "match" : "matches";
  return {
    what: `${metric.what} Here, ${AGGREGATIONS[aggregation]} ${window} ${matches}.`,
    why: metric.why,
  };
}

/** Explain a family, falling back to a neutral description. */
export function explainFamily(family: string): Explanation {
  return (
    FAMILIES[family] ?? {
      what: `Features in the ${family.replace(/_/g, " ")} family.`,
      why: "No description recorded for this family yet.",
    }
  );
}
