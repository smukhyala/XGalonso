# XG Alonso — Web

| Field | Value |
|---|---|
| Project | XG Alonso |
| Document | Web Front End |
| Status | Active (MVP) |
| Depends on | `apps/api` |

The local front end for the decision system. It renders one gameweek's
recommendation, the resulting eleven, and the ranked player pool.

Per **D4** this is the third interface, after the CLI and the API. It holds no
modelling logic: every number on the page is computed in `packages/` and
delivered through `apps/api`, so the browser cannot disagree with `xg recommend`.

## Running it

Two processes. The API must be up first — the web app proxies `/api/*` to it.

```bash
make api           # 127.0.0.1:8000
make web-install   # once
make web           # 127.0.0.1:3000
```

## Design

| Concern | Decision |
|---|---|
| Palette | Floodlit night — `#0A1014` ground, chalk text |
| Accent | Positional, not decorative: GKP amber, DEF blue, MID green, FWD vermilion |
| Display face | Archivo |
| Body face | Instrument Sans |
| Figures | Geist Mono, tabular numerals |

Colour is an information channel here. A token's colour states the player's
position, so the palette is never spent on decoration.

### The two signature elements

**The call.** The hero is the recommendation written as a sentence — *"Sell X.
Buy Y."* — not a metric tile. The product's job is to say what to do; the
arithmetic supporting it sits underneath, and the evidence under that, in the
order a manager actually asks for them. A hold renders at equal weight, because
most gameweeks the right move is to do nothing and a front end that renders
"hold" apologetically teaches its user to ignore it.

**The pitch.** The eleven is drawn in the formation the optimizer actually
chose, so a 3-4-3 and a 4-4-2 look different at a glance. Each player is a token
carrying a *conviction ring* whose completeness is the modelled probability that
he starts — the quantity that dominates FPL scoring. Attacking direction is up
the page: this is our half, so the keeper stands inside his own penalty area and
the halfway line closes the top.

## Honesty constraints

These are load-bearing, not stylistic.

- Every response carries a `Provenance` record, and the footer prints it —
  model name and version, feature-set version, data cutoff, run id. A figure
  whose lineage cannot be shown does not belong on the page.
- Reason text is rendered by `packages/data_contracts`, which refuses to build a
  reason whose evidence cannot fill its own template. The front end never
  composes prose, so it cannot fabricate a justification.
- When purchase prices are assumed rather than reconstructed, the squad panel
  says so and states that the budget shown is a lower bound.
- `stale` on `/health` surfaces in the masthead rather than being swallowed.
