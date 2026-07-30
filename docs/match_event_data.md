# Match event data

## What this adds

The official FPL API publishes a great deal about players and almost nothing
about teams. It carries per-player expected goals and its own Opta-derived
composite indices, but no team-level event counts at all — not shots, not shots
on target, not corners, not fouls.

That gap matters to the discovery engine specifically. A hypothesis like *"a
defender in a side that concedes few shots on target is undervalued relative to
one whose clean sheets came from luck"* is not expressible without team shot
volume. `expected_goals_conceded` is close, but it is a single modelled number;
the raw counts let a hypothesis separate volume from quality.

`team_match_events` supplies, per team per match:

| Column | Meaning |
|---|---|
| `shots` / `shots_against` | Total shots taken and faced |
| `shots_on_target` / `shots_on_target_against` | The on-target subset |
| `corners` / `corners_against` | Set-piece volume, both directions |
| `fouls_committed` / `fouls_suffered` | Fouls each way |
| `yellow_cards` / `red_cards` | Booking-point exposure |
| `goals_for` / `goals_against` | Final score, from the team's own side |

One row per team per match, not one row per match. A team's own row therefore
carries both what it did and what was done to it, so no downstream feature has
to branch on which side the player's club was — which is where sign errors live.

## Where it comes from, and why not somewhere better

Decision D6 originally read *"Official FPL API only, zero budget. No scraping,
no paid providers."* It was relaxed on 2026-07-29 to permit fetching free public
data the official API does not publish. The relaxation is narrower than it
sounds, because permission to fetch is not permission to fetch anything.

Three candidate sources were evaluated:

| Source | Verdict |
|---|---|
| **Understat** — shot-level, with xG per shot | **Refused.** Its `robots.txt` is `User-agent: * / Disallow: /` — an unambiguous machine-readable refusal covering every path |
| **FBref** — rich team and player match logs | **Refused.** Behind a Cloudflare interactive challenge, even on `/robots.txt`. Getting data would mean evading a control rather than passing one |
| **football-data.co.uk** — team-match counts, static CSV | **Used.** Publishes `User-agent: * / Disallow:` — explicit permission for everything |

Understat is the source the original brief pointed at, and it is strictly
richer: shot coordinates, situations, and an xG value per shot. It said no. That
refusal is honoured, and the cost is recorded rather than glossed: **shot-level
features are not buildable from the source we use.** We learn that a team took
18 shots and 7 were on target; we do not learn where they were taken from or
what each was worth. A feature claiming shot-location provenance would be
claiming lineage it does not have.

### The refusal is enforced, not remembered

`pipelines/ingestion/robots.py` gates every fetch. A comment saying "be polite"
would not have stopped a future adapter being written against Understat; a gate
does. Semantics follow RFC 9309:

- **2xx** — parse and obey.
- **4xx** — no rules exist, so unrestricted. FPL serves its SPA shell at
  `/robots.txt` and `raw.githubusercontent.com` returns 404; both are correctly
  read as permitting us.
- **5xx or unreachable** — assume **complete disallow**. An origin we cannot ask
  is an origin that has not said yes.

Rules are fetched once per origin and cached, and a `Pacer` enforces the
declared `Crawl-delay` — or a one-second floor when none is declared, because
nothing here is latency-sensitive.

The test suite carries a **negative control**: an origin serving Understat's
actual `robots.txt` must be refused, and the refusal must cost zero downloads.
Verified by mutation — replacing the gate with one that allows everything fails
exactly those two tests and no others.

## Point-in-time correctness

These are *post-match* statistics, which makes availability the most dangerous
field in the table. A row stamped with its own kickoff would let a model read a
match's shot count before the match was played.

The source refreshes a whole-season file periodically and publishes no per-match
timestamp, so there is nothing to observe. The strongest claim actually
defensible is **one day after kickoff**, and that is the bound used
(`PUBLICATION_LAG`). It is conservative by roughly a day and costs essentially
nothing: a gameweek's matches finish days before the next deadline.

Kickoffs are published in UK local time. They are converted through
`Europe/London`, not treated as UTC — parsing an August 20:00 kickoff as UTC
would place it an hour early, and for an availability bound early is the wrong
direction. Both a BST and a GMT fixture are asserted in the tests.

## Club identity

The source names clubs in prose — `Man United`, `Nott'm Forest`, `Tottenham` —
and FPL names them differently. **No club code is transcribed from memory.**
Names resolve against the `teams` table and `team_code` is read from there, so
the mapping stays correct as clubs are promoted, relegated or renamed.

Matching runs exact → alias → *unique* prefix. The prefix step lets `Ipswich`
reach `Ipswich Town` and `Coventry` reach `Coventry City` without an alias entry
per promoted club. It refuses to guess: a prefix matching two clubs leaves the
name unmatched rather than picking one.

`SOURCE_TEAM_ALIASES` carries spelling differences only. Its three entries —
`Man United`→`Man Utd`, `Tottenham`→`Spurs`, `Sheffield United`→`Sheffield Utd`
— were derived on 2026-07-29 by diffing all 25 distinct names across the
2022-23 to 2025-26 `E0` files against the FPL club lists for the same seasons.
They were the only names that failed both exact and prefix matching.

Historical club lists come from the archive's per-season `teams.csv`
(`fetch_archive_teams`), because the live bootstrap lists only the current
twenty and a backfill spans relegated clubs.

Unmatched names are **reported, never swallowed**. In the Premier League an
unmatched name means a bug, so it raises. In the Championship most clubs are
genuinely absent from FPL, so it reports — a silent drop there would look
identical to a club that simply took no shots.

## Divisions

`E0` (Premier League) is ingested by default. `E1` (Championship) is supported
but off, because a promoted side otherwise arrives with no history at all —
Coventry City and Hull City are in the 2026/27 FPL bootstrap and played no
Premier League football inside the D7 window.

Every row carries its `division`. Pooling Championship shot counts with Premier
League ones as though they were the same quantity is a modelling decision, and
it should have to be made explicitly rather than fall out of a data load.

## Usage

```bash
# The D7 range, Premier League
xg ingest-match-events

# One season, or the Championship
xg ingest-match-events --seasons 2024-25
xg ingest-match-events --seasons 2025-26 --division E1
```

Writes bronze snapshots per season-division and a
`silver/team_match_events.parquet`. Verified live on 2026-07-29: four seasons,
380 matches each, 3,040 team-match rows, all 20 clubs resolved per season and
no missing counts.

## What is not built yet

The table exists and is populated; it is **not yet joined into the discovery
frame**. Wiring it in means adding the columns to the DSL's source vocabulary
and joining on `(team_code, kickoff_time)` under the same point-in-time rule as
every other feature. Until that lands, no discovered feature reads these
columns — which is stated here rather than implied by their presence.
