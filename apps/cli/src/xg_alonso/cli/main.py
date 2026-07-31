"""The ``xg`` command-line interface.

Per decision D4 the CLI is the first surface, ahead of the HTTP API and the web
app. That ordering is deliberate: a terminal command forces the recommendation
to stand on its own content rather than on presentation, and a recommendation
that is not convincing as plain text will not become convincing in a card.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer

from xg_alonso.cli.pipeline import (
    PRICES_WERE_ASSUMED,
    SliceContext,
    build_context,
    fetch_squad,
    load_squad_file,
    recommend,
    squad_from_payload,
)
from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    Season,
    TeamId,
    TenthsOfMillion,
    parse_season,
)
from xg_alonso.contracts.provenance import utc_now
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.explanations.render import render_recommendation, render_squad_summary
from xg_alonso.pipelines.ingestion import (
    SOURCE_BOOTSTRAP,
    SOURCE_FIXTURES,
    FplApiClient,
    OfflineError,
    git_manifest,
    ingest_bootstrap,
    ingest_element_summaries,
    read_element_summaries,
    season_end_time,
)
from xg_alonso.pipelines.normalization import (
    PLAYER_GAMEWEEK_STATS_SCHEMA,
    empty_frame,
    normalize_element_summary,
    normalize_history_past,
)
from xg_alonso.prediction.gameweek import collapse_by_player
from xg_alonso.storage import FileSystemBronzeStore, ParquetTableStore

app = typer.Typer(
    name="xg",
    help="XG Alonso — an ML-powered Fantasy Premier League decision system.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_DATA_ROOT = Path(".data")
DEFAULT_SEASON = "2026-27"

DataRoot = Annotated[
    Path, typer.Option("--data-root", help="Where bronze snapshots live.", show_default=True)
]
SeasonOpt = Annotated[str, typer.Option("--season", help="Season in YYYY-YY form.")]


def _bronze(data_root: Path) -> FileSystemBronzeStore:
    return FileSystemBronzeStore(data_root / "bronze")


def _load_context(data_root: Path, season: Season) -> SliceContext:
    """Rebuild the slice context from the most recent bronze snapshot.

    Reads stored bytes rather than the network, so a recommendation is
    reproducible after the fact and an offline run behaves identically.
    """
    bronze = _bronze(data_root)

    bootstrap_ref = bronze.latest(SOURCE_BOOTSTRAP)
    if bootstrap_ref is None:
        raise typer.BadParameter(
            f"no {SOURCE_BOOTSTRAP} snapshot under {data_root / 'bronze'}. Run `xg ingest` first."
        )
    bootstrap_payload = json.loads(bronze.read(bootstrap_ref))

    fixtures_ref = bronze.latest(SOURCE_FIXTURES)
    fixtures_payload = json.loads(bronze.read(fixtures_ref)) if fixtures_ref else []

    # Per-gameweek history needs element-summary or the archive backfill, which
    # is slice-2 work. An empty frame keeps every feature null-safe rather than
    # letting the pipeline pretend it has history it does not have.
    stats_path = data_root / "silver" / "player_gameweek_stats.parquet"
    player_stats = (
        pl.read_parquet(stats_path)
        if stats_path.exists()
        else empty_frame(PLAYER_GAMEWEEK_STATS_SCHEMA)
    )

    return build_context(
        bootstrap_payload,
        fixtures_payload=fixtures_payload,
        player_stats=player_stats,
        season=season,
        snapshot_sha256=bootstrap_ref.content_sha256,
        available_time=bootstrap_ref.timestamps.available_time,
    )


@dataclass(frozen=True)
class RulesResolution:
    """Which season's rules were used, and whether they were the right ones."""

    scoring: Any
    squad: Any
    source_season: str
    exact: bool

    @property
    def caveat(self) -> str:
        """One line for a report, empty when the rules are the season's own."""
        if self.exact:
            return ""
        return (
            f"scoring rules resolved from the {self.source_season} pinned snapshot; "
            "no snapshot exists for the evaluated season"
        )


def _resolve_rules(
    data_root: Path, season: Season, *, require_exact: bool = False
) -> RulesResolution:
    """Rules for ``season``, preferring its own pinned snapshot.

    `.data/pinned` holds only the current season, so a 2024-25 backtest was
    being scored under 2026-27 rules — three call sites reached for
    `DEFAULT_SEASON` and none of them said so. There is no historical snapshot
    to load, so the honest fix is not to pretend otherwise: fall back to the
    newest pinned season no later than the one asked for, fall back further to
    the newest available, and **record which**, so `rules_are_exact` reaches
    the report rather than the reader's imagination.

    Args:
        require_exact: Refuse the fallback instead of recording it.
    """
    pinned = sorted(
        p.stem.removeprefix("rules_") for p in (data_root / "pinned").glob("rules_*.json")
    )
    wanted = str(season)

    if wanted not in pinned and require_exact:
        raise typer.BadParameter(
            f"no pinned rules for {wanted}. Available: {pinned or 'none'}. "
            "Run `xg ingest` for that season, or drop --require-exact-rules."
        )

    if wanted in pinned:
        chosen = wanted
    else:
        earlier = [s for s in pinned if s <= wanted]
        chosen = earlier[-1] if earlier else (pinned[-1] if pinned else DEFAULT_SEASON)

    # Read the chosen season's *pinned* rules, not the latest bronze snapshot.
    #
    # This previously called `_load_context`, which parses whatever bootstrap
    # payload happens to be newest in bronze — so a 2022-23 resolution returned
    # `exact=True`, `source_season="2022-23"` and a `ScoringRules` whose
    # `version` said "2022-23" while its *values* came from 2026-27. It labelled
    # the answer with the season it was asked about and supplied another
    # season's rules, which is worse than the bug this function was written to
    # fix: it turned a silent substitution into an asserted one.
    scoring, squad = _load_pinned_rules(data_root, parse_season(chosen))
    if scoring is None or squad is None:
        # Nothing pinned at all — a first run. The snapshot is the only source
        # there is, and `exact=False` says so rather than implying otherwise.
        context = _load_context(data_root, season)
        return RulesResolution(
            scoring=context.scoring,
            squad=context.squad_rules,
            source_season=chosen,
            exact=False,
        )

    return RulesResolution(
        scoring=scoring, squad=squad, source_season=chosen, exact=chosen == wanted
    )


@app.command()
def ingest(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
) -> None:
    """Fetch official FPL data into immutable bronze snapshots."""
    parsed = parse_season(season)
    run_id = f"ingest-{uuid.uuid4().hex[:12]}"
    bronze = _bronze(data_root)

    # Load the pinned rules so drift can actually be detected. Without these
    # `result.rule_changes` is always empty and the check below is dead code.
    pinned_scoring, pinned_squad = _load_pinned_rules(data_root, parsed)

    with FplApiClient() as client:
        result = ingest_bootstrap(
            client=client,
            bronze=bronze,
            season=parsed,
            run_id=run_id,
            pinned_scoring=pinned_scoring,
            pinned_squad=pinned_squad,
        )

    typer.echo(f"run {run_id}  commit {result.manifest.git_commit[:8]}")
    for ref in result.snapshots:
        typer.echo(f"  {ref.source:<24} {ref.byte_size:>9,} bytes  {ref.content_sha256[:12]}")

    if result.rules_drifted:
        typer.secho("\n  RULE DRIFT DETECTED", fg=typer.colors.RED, bold=True)
        for change in result.rule_changes:
            typer.echo(f"    {change}")
        typer.echo("  Stored predictions were scored under the old rules.")

    if result.preseason_warnings:
        typer.secho("\n  Preseason data hazards:", fg=typer.colors.YELLOW)
        for warning in result.preseason_warnings:
            typer.echo(f"    {warning.field_name}: {warning.detail}")

    if pinned_scoring is None:
        _pin_rules(data_root, parsed, result.snapshots[0])
        typer.echo(f"\n  pinned the rules from this snapshot -> {_pinned_path(data_root, parsed)}")


@app.command(name="team-news")
def team_news_command(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    entry_id: Annotated[
        int | None, typer.Option("--entry", help="Your team id, so your squad is checked first.")
    ] = None,
    squad_file: Annotated[
        Path | None, typer.Option("--squad-file", help="Picks JSON, if the API has no squad yet.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="How many players to search.")] = 20,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the shortlist and stop, without searching.")
    ] = False,
) -> None:
    """Search for team news FPL has not published, and file it as form signals.

    **This is the second-class channel, deliberately.** `xg refresh` handles
    everything official — injuries, suspensions, chance of playing — and a
    re-poll is authoritative. This is for what no endpoint carries: a coach
    hinting somebody needs a rest, a returning player taking a place back, a
    striker who has not looked himself.

    Nothing found here can assert a projection. Every result becomes a form
    signal, which multiplies expected points within a hard clamp, must carry a
    source URL or it cannot be constructed at all, and expires. A claim with no
    link is discarded rather than filed at lower strength.

    Searches a shortlist rather than the league: your squad, then the highly
    owned and highly projected. Players FPL has already flagged are skipped —
    the official field is a fact, and an inference restating it would
    double-count while looking like corroboration.
    """
    import json

    from xg_alonso.contracts.form import SignalSet
    from xg_alonso.interpreter import (
        InterpreterUnavailableError,
        search_player_news,
        shortlist_from,
    )

    parsed = parse_season(season)
    context = _load_context(data_root, parsed)
    gameweek = context.next_gameweek()

    held: list[int] = []
    if entry_id is not None or squad_file is not None:
        try:
            state = _squad_state(context, entry_id or 0, squad_file)
            held = [int(p.player_code) for p in state.picks]
        except Exception as exc:
            typer.secho(f"  (no squad loaded: {exc})", fg=typer.colors.YELLOW)

    projected = {
        int(p.player_code): p.expected_points
        for p in _predict_all(context, gameweek, None, data_root=data_root)
    }
    rows = list(context.players.iter_rows(named=True))
    shortlist = shortlist_from(rows, squad=held, expected_points=projected, limit=limit)

    typer.echo(f"Shortlist ({len(shortlist)} of {len(rows)} players)")
    for entry in shortlist[:12]:
        typer.echo(f"  {entry.name:<18} {entry.why}")
    if len(shortlist) > 12:
        typer.echo(f"  ... and {len(shortlist) - 12} more")

    if dry_run:
        typer.echo("\n  (dry run — nothing searched)")
        return

    typer.echo("\n  searching ...")
    try:
        sweep = search_player_news(shortlist, now=utc_now())
    except InterpreterUnavailableError as exc:
        typer.secho(f"  {exc}", fg=typer.colors.YELLOW)
        raise typer.Exit(1) from exc

    typer.echo(f"  {sweep.summary()}\n")
    for signal in sweep.signals:
        colour = typer.colors.RED if signal.direction.value == "negative" else typer.colors.GREEN
        name = context.player_names().get(signal.player_code, str(signal.player_code))
        shift = (signal.multiplier - 1.0) * 100.0
        typer.secho(f"  {name:<18} {shift:+.1f}%  ({signal.strength.value})", fg=colour)
        typer.echo(f"                     {signal.summary}")
        for source in signal.sources[:2]:
            typer.echo(f"                     {source}")

    if sweep.discarded:
        typer.echo("\n  discarded:")
        for who, why in sweep.discarded:
            typer.echo(f"    {who}: {why}")

    if not sweep.signals:
        typer.echo("  nothing worth filing — the official data is the whole story today")
        return

    # A signal that lapses before the deadline it would apply to is filed and
    # never read. That is the correct expiry behaviour — form information is
    # perishable — but silently writing signals nothing will ever use is not,
    # so it is said out loud along with when to run this instead.
    deadline = context.deadline_for(gameweek)
    stale_on_arrival = [s for s in sweep.signals if not s.is_live(deadline)]
    if stale_on_arrival:
        days = (deadline - utc_now()).days
        typer.secho(
            f"\n  {len(stale_on_arrival)} of {len(sweep.signals)} expire before the "
            f"GW{int(gameweek)} deadline ({days} days away) and will never be applied.",
            fg=typer.colors.YELLOW,
        )
        typer.echo("  Team news is a near-deadline tool; run it in the days before one.")

    # Merged rather than replaced, and expiry does the pruning. Overwriting
    # would silently discard a signal from a source this sweep did not happen to
    # search, which is not the same as that signal having lapsed.
    path = data_root / "signals" / "form_signals.json"
    existing: list[Any] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("signals", [])
        except (OSError, ValueError):
            existing = []

    fresh = {(s.player_code, s.observed_at.isoformat()) for s in sweep.signals}
    kept = [
        record
        for record in existing
        if (record.get("player_code"), record.get("observed_at")) not in fresh
    ]
    merged = SignalSet(signals=tuple(sweep.signals), loaded_at=utc_now()).model_dump(mode="json")
    merged["signals"] = [*kept, *merged["signals"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    typer.echo(f"\n  {len(merged['signals'])} signal(s) on file -> {path}")


@app.command()
def refresh(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    entry_id: Annotated[
        int | None,
        typer.Option("--entry", help="Your FPL team id, so squad changes rank as critical."),
    ] = None,
    squad_file: Annotated[
        Path | None, typer.Option("--squad-file", help="Picks JSON, when the API has no squad yet.")
    ] = None,
    show_all: Annotated[
        bool, typer.Option("--all", help="List the minor events too, not just what matters.")
    ] = False,
) -> None:
    """Re-read the official payload and report what changed.

    **This is the answer to "the platform did not know".** A striker picked up a
    foot injury and FPL published `status='i'` with a dated news line against
    him; the recommendation kept naming him because nobody re-read the source.
    No amount of scraping fixes that — one re-poll does.

    Costs exactly one request for all 564 players however much moved, so it is
    a scheduled job rather than anything a query triggers. FPL sends no ETag,
    but `cache-control: max-age=300` means polling faster than five minutes
    returns CDN-cached bytes; run it on the deadline, and hourly on the days
    before one.

    Writes an append-only event log. Nothing downstream ever fetches — it reads
    what this wrote, and the API reports how old that is rather than presenting
    stale numbers with a confident face.
    """
    import json

    from xg_alonso.contracts.events import Materiality
    from xg_alonso.pipelines.ingestion import SOURCE_BOOTSTRAP, diff_bootstrap

    parsed = parse_season(season)
    run_id = f"refresh-{uuid.uuid4().hex[:12]}"
    bronze = _bronze(data_root)

    previous_ref = bronze.latest(SOURCE_BOOTSTRAP)
    previous = json.loads(bronze.read(previous_ref)) if previous_ref else None

    pinned_scoring, pinned_squad = _load_pinned_rules(data_root, parsed)
    with FplApiClient() as client:
        result = ingest_bootstrap(
            client=client,
            bronze=bronze,
            season=parsed,
            run_id=run_id,
            pinned_scoring=pinned_scoring,
            pinned_squad=pinned_squad,
        )

    current_ref = result.snapshots[0]
    current = json.loads(bronze.read(current_ref))
    typer.echo(
        f"  polled bootstrap-static  {current_ref.byte_size:,} bytes  "
        f"{current_ref.content_sha256[:12]}"
    )

    # Content-addressed bronze makes an unchanged payload free: identical bytes
    # write nothing, so the hash comparison ends the run before a single player
    # is examined.
    if previous_ref is not None and previous_ref.content_sha256 == current_ref.content_sha256:
        typer.echo("  unchanged since the last poll — nothing to do")
        _write_events(data_root, [], checked_at=utc_now())
        return

    held: list[int] = []
    if entry_id is not None or squad_file is not None:
        try:
            context = _load_context(data_root, parsed)
            state = _squad_state(context, entry_id or 0, squad_file)
            held = [int(p.player_code) for p in state.picks]
        except Exception as exc:
            typer.secho(f"  (no squad loaded: {exc})", fg=typer.colors.YELLOW)

    diff = diff_bootstrap(
        previous,
        current,
        detected_at=utc_now(),
        squad=held,
        previous_hash=previous_ref.content_sha256 if previous_ref else "",
        current_hash=current_ref.content_sha256,
        payload_bytes=current_ref.byte_size,
    )

    if previous is None:
        typer.echo("  first snapshot — no baseline to compare against yet")
        _write_events(data_root, [], checked_at=diff.compared_at)
        return

    surfaced = diff.worth_surfacing()
    minor = [e for e in diff.events if not e.materiality.is_worth_surfacing]
    typer.echo(f"  compared {diff.players_compared:,} players — {len(diff.events)} change(s)")

    if surfaced:
        typer.echo("")
        for event in surfaced:
            colour = (
                typer.colors.RED
                if event.materiality is Materiality.CRITICAL
                else typer.colors.YELLOW
            )
            typer.secho(f"  {event.materiality.value.upper():<8} {event.headline}", fg=colour)
            stamp = (
                event.source_reported_at.strftime("%d %b %H:%M")
                if event.source_reported_at
                else "no source timestamp"
            )
            typer.echo(f"           {event.reason} · reported {stamp}")
            if event.detail and event.detail not in event.headline:
                typer.echo(f'           "{event.detail}"')
    else:
        typer.echo("  nothing material")

    if minor:
        if show_all:
            typer.echo("\n  minor:")
            for event in minor:
                typer.echo(f"    {event.headline}  ({event.reason})")
        else:
            names = ", ".join(e.web_name for e in minor[:6])
            more = f" and {len(minor) - 6} more" if len(minor) > 6 else ""
            typer.echo(f"  minor ({len(minor)}): {names}{more}   --all to list")

    path = _write_events(data_root, list(diff.events), checked_at=diff.compared_at)
    typer.echo(f"\n  appended to {path}")


def _write_events(data_root: Path, events: list[Any], *, checked_at: datetime) -> Path:
    """Append events to the log and record when we last looked.

    **Append-only, and the timestamp is written even when nothing changed.**
    "We checked and nothing moved" and "we have not checked since Tuesday" are
    entirely different states, and a log that only grows on change cannot tell
    them apart — which is exactly the confusion that let a stale snapshot look
    current.
    """
    import json

    directory = data_root / "events"
    directory.mkdir(parents=True, exist_ok=True)

    log = directory / "player_events.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(event.model_dump_json() + "\n")

    (directory / "last_checked.json").write_text(
        json.dumps({"checked_at": checked_at.isoformat(), "events": len(events)}, indent=2),
        encoding="utf-8",
    )
    return log


def _pinned_path(data_root: Path, season: Season) -> Path:
    return data_root / "pinned" / f"rules_{season}.json"


def _load_pinned_rules(data_root: Path, season: Season):  # type: ignore[no-untyped-def]
    """Read the pinned rules for a season, or ``(None, None)`` on first run."""
    from xg_alonso.pipelines.ingestion import load_rules_from_snapshot

    path = _pinned_path(data_root, season)
    if not path.exists():
        return None, None

    record = json.loads(path.read_text())
    return load_rules_from_snapshot(
        record["payload"],
        season=season,
        source_sha256=record["source_sha256"],
        fetched_at=datetime.fromisoformat(record["fetched_at"]),
    )


def _pin_rules(data_root: Path, season: Season, ref: object) -> None:
    """Store the rule-bearing part of a snapshot as the drift reference.

    Only ``game_config`` and ``element_types`` are kept — the parts that define
    scoring and squad constraints. Pinning the whole payload would make every
    price change look like a rule change.
    """
    bronze = _bronze(data_root)
    payload = json.loads(bronze.read(ref))  # type: ignore[arg-type]
    path = _pinned_path(data_root, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_sha256": ref.content_sha256,  # type: ignore[attr-defined]
                "fetched_at": ref.timestamps.available_time.isoformat(),  # type: ignore[attr-defined]
                "payload": {
                    "game_config": payload["game_config"],
                    "element_types": payload["element_types"],
                },
            },
            indent=1,
            sort_keys=True,
        )
    )


@app.command(name="build-features")
def build_features_command(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    full: Annotated[
        bool,
        typer.Option("--full", help="Build the whole declared catalogue, not just the slice set."),
    ] = False,
) -> None:
    """Build the point-in-time feature set for the next gameweek."""
    from xg_alonso.cli.pipeline import build_entities
    from xg_alonso.features.slice1 import (
        SLICE1_FEATURES,
        build_slice1_features,
        build_team_gameweek_stats,
    )

    context = _load_context(data_root, parse_season(season))
    gameweek = context.next_gameweek()
    cutoff = context.deadline_for(gameweek)

    entities = build_entities(context, cutoff=cutoff)
    team_stats = build_team_gameweek_stats(context.player_stats, context.players)
    features = build_slice1_features(
        entities, player_stats=context.player_stats, team_stats=team_stats
    )

    reported: tuple[str, ...] = SLICE1_FEATURES
    if full:
        from xg_alonso.features.assemble import build_model_features
        from xg_alonso.features.career import CAREER_FEATURES
        from xg_alonso.features.catalogue import feature_names
        from xg_alonso.features.opponent import OPPONENT_FEATURES
        from xg_alonso.features.recency import RECENCY_FEATURES

        # Built through the shared assembler so the artifact this command
        # writes is exactly what a trained model consumes. Assembling it
        # separately is how it previously came to be missing thirteen opponent
        # columns, and later the career and recency ones.
        features = build_model_features(features, player_stats=context.player_stats)
        reported = (
            tuple(SLICE1_FEATURES)
            + tuple(feature_names())
            + OPPONENT_FEATURES
            + CAREER_FEATURES
            + RECENCY_FEATURES
        )

    out_dir = data_root / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"features_gw{gameweek}.parquet"
    features.write_parquet(destination)

    typer.echo(f"GW{gameweek}  cutoff {cutoff.isoformat()}")
    typer.echo(f"  {features.height:,} rows x {len(reported)} features -> {destination}")

    if full:
        # Too many to list; report the distribution and anything unusable.
        coverages = [
            1.0 - (features[n].null_count() / max(features.height, 1))
            for n in reported
            if n in features.columns
        ]
        thin = [
            n
            for n in reported
            if n in features.columns
            and 1.0 - (features[n].null_count() / max(features.height, 1)) < 0.5
        ]
        constant = [
            n
            for n in reported
            if n in features.columns
            and features[n].drop_nulls().len() > 1
            and features[n].drop_nulls().n_unique() == 1
        ]
        typer.echo(f"    mean coverage {sum(coverages) / max(len(coverages), 1):6.1%}")
        typer.echo(f"    below 50% coverage: {len(thin)}")
        typer.echo(f"    constant (no signal): {len(constant)}")
        if constant:
            typer.secho(f"    {constant[:6]}", fg=typer.colors.YELLOW)
        return

    for name in reported:
        if name in features.columns:
            coverage = 1.0 - (features[name].null_count() / max(features.height, 1))
            typer.echo(f"    {name:<24} coverage {coverage:6.1%}")


@app.command()
def squad(
    entry_id: Annotated[int, typer.Argument(help="Public FPL entry (manager) id.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    squad_file: Annotated[
        Path | None,
        typer.Option(
            "--squad-file",
            help="Read the squad from a picks JSON file instead of the API. "
            "Required before a gameweek deadline, when picks/ returns 404.",
        ),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option(
            "--model",
            help="Use trained component models instead of the closed-form baseline.",
        ),
    ] = None,
) -> None:
    """Show a squad with projected points per player."""
    context = _load_context(data_root, parse_season(season))
    recommendation, predictions = _run(context, entry_id, squad_file, model_path)
    names = context.player_names()

    state = _squad_state(context, entry_id, squad_file)

    def rows(picks: tuple[SquadPick, ...], /) -> list[tuple[str, str, float]]:
        out = []
        for pick in picks:
            prediction = predictions.get(pick.player_code)
            out.append(
                (
                    names.get(pick.player_code, str(pick.player_code)),
                    pick.position.value,
                    prediction.expected_points if prediction else 0.0,
                )
            )
        return out

    typer.echo(
        render_squad_summary(
            entry_id=entry_id,
            gameweek=int(state.gameweek),
            squad_value=int(state.squad_value),
            bank=int(state.bank),
            free_transfers=state.free_transfers,
            starters=rows(state.starters),
            bench=rows(state.bench),
        )
    )
    del recommendation


def _squad_state(context: SliceContext, entry_id: int, squad_file: Path | None):  # type: ignore[no-untyped-def]
    """Load a squad, preferring the live API and falling back to a file.

    ``entry/{id}/event/{gw}/picks/`` returns 404 before a gameweek's deadline —
    the squad genuinely does not exist yet — so the file path is a necessary
    fallback rather than a convenience, and it is what makes the whole flow
    testable before a season starts.
    """
    import httpx

    gameweek = context.next_gameweek()

    if squad_file is not None:
        payload = load_squad_file(squad_file)
        return squad_from_payload(
            payload, context=context, entry_id=EntryId(entry_id), gameweek=gameweek
        )

    try:
        with FplApiClient() as client:
            return fetch_squad(
                client=client,
                context=context,
                entry_id=EntryId(entry_id),
                gameweek=gameweek,
            )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise typer.BadParameter(
                f"entry {entry_id} has no squad for GW{gameweek} yet — the picks "
                "endpoint returns 404 until that gameweek's deadline passes. "
                "Pass --squad-file to work from a local squad in the meantime."
            ) from exc
        raise
    except OfflineError as exc:
        raise typer.BadParameter(
            f"cannot fetch entry {entry_id} while offline. Pass --squad-file."
        ) from exc


def _run(  # type: ignore[no-untyped-def]
    context: SliceContext, entry_id: int, squad_file: Path | None, model_path: Path | None = None
):
    state = _squad_state(context, entry_id, squad_file)
    manifest = git_manifest("recommend", run_id=f"rec-{uuid.uuid4().hex[:12]}")

    models = None
    if model_path is not None:
        from xg_alonso.prediction import load_models

        models = load_models(model_path).models

    return recommend(
        context=context,
        squad=state,
        entry_id=EntryId(entry_id),
        run_id=manifest.run_id,
        code_version=manifest.git_commit,
        generated_at=utc_now(),
        models=models,
    )


@app.command(name="recommend")
def recommend_command(
    entry_id: Annotated[int, typer.Argument(help="Public FPL entry (manager) id.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    squad_file: Annotated[
        Path | None,
        typer.Option(
            "--squad-file",
            help="Read the squad from a picks JSON file instead of the API. "
            "Required before a gameweek deadline, when picks/ returns 404.",
        ),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option(
            "--model",
            help="Use trained component models instead of the closed-form baseline.",
        ),
    ] = None,
) -> None:
    """Recommend the best legal single transfer, or advise holding."""
    context = _load_context(data_root, parse_season(season))
    recommendation, _ = _run(context, entry_id, squad_file, model_path)

    prices = {
        PlayerCode(int(r["player_code"])): int(r["current_price"])
        for r in context.players.iter_rows(named=True)
    }
    typer.echo(render_recommendation(recommendation, names=context.player_names(), prices=prices))

    if PRICES_WERE_ASSUMED.get(entry_id):
        typer.secho(
            "\n  Note: purchase prices were assumed equal to current prices, so the\n"
            "  budget above is a lower bound. Real prices are reconstructed from the\n"
            "  transfer log, which stays empty until the season produces transfers.",
            fg=typer.colors.YELLOW,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()


@app.command(name="ingest-history")
def ingest_history_command(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Fetch only the first N players (for a quick check)."),
    ] = None,
) -> None:
    """Fetch each player's history into bronze.

    Without this every player looks identical to the model at gameweek 1, because
    the current season has no matches yet. This is roughly 500 sequential polite
    requests and takes a couple of minutes; it is safe to interrupt and re-run,
    since snapshots are keyed by content hash.
    """
    context = _load_context(data_root, parse_season(season))
    element_ids = [int(r["element_id"]) for r in context.players.iter_rows(named=True)]
    if limit is not None:
        element_ids = element_ids[:limit]

    run_id = f"history-{uuid.uuid4().hex[:12]}"
    bronze = _bronze(data_root)

    typer.echo(f"fetching {len(element_ids)} player histories (run {run_id})")
    with FplApiClient() as client, typer.progressbar(element_ids, label="  players") as bar:
        for element_id in bar:
            ingest_element_summaries(
                client=client, bronze=bronze, element_ids=[element_id], run_id=run_id
            )

    _materialize_history(data_root, context)


def _materialize_history(data_root: Path, context: SliceContext) -> None:
    """Turn stored player histories into the silver stats table."""
    bronze = _bronze(data_root)
    code_by_element = {
        int(r["element_id"]): int(r["player_code"]) for r in context.players.iter_rows(named=True)
    }
    payloads = read_element_summaries(bronze, list(code_by_element))

    frames = []
    for element_id, payload in payloads.items():
        frames.append(
            normalize_history_past(
                payload,
                player_code=code_by_element[element_id],
                season_end_lookup=season_end_time,
            )
        )
        frames.append(
            normalize_element_summary(
                payload,
                player_code=code_by_element[element_id],
                season=context.season,
                available_time=context.available_time,
            )
        )

    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        typer.secho("  no history rows found", fg=typer.colors.YELLOW)
        return

    stats = pl.concat(frames, how="vertical")
    out_dir = data_root / "silver"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "player_gameweek_stats.parquet"

    # Merge rather than overwrite. `xg backfill` writes the same file, and an
    # earlier version clobbered four seasons of archive history with whatever
    # the API happened to return for the current one.
    added = stats.height
    if destination.exists():
        existing = pl.read_parquet(destination)
        stats = pl.concat([existing, stats], how="diagonal_relaxed").unique(
            subset=["player_code", "season", "gameweek_id", "fixture_id"],
            keep="last",
            maintain_order=True,
        )
        typer.echo(f"  merged {added:,} rows into {existing.height:,} existing")

    stats.write_parquet(destination)
    typer.echo(f"  {stats.height:,} total history rows -> {destination}")


@app.command()
def backfill(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    seasons: Annotated[
        str | None,
        typer.Option("--seasons", help="Comma-separated seasons. Defaults to the D7 range."),
    ] = None,
) -> None:
    """Backfill per-gameweek history from the community archive.

    The live API is current-season only, so this is the only route to the
    history a model needs. The archive mirrors official API responses verbatim.
    """
    import io

    from xg_alonso.pipelines.ingestion import (
        BACKFILL_SEASONS,
        SOURCE_ARCHIVE_GW,
        SOURCE_ARCHIVE_PLAYERS,
        fetch_archive_season,
    )
    from xg_alonso.pipelines.normalization import (
        build_players_history,
        normalize_archive_season,
    )

    wanted = [s.strip() for s in seasons.split(",")] if seasons else list(BACKFILL_SEASONS)
    run_id = f"backfill-{uuid.uuid4().hex[:12]}"
    bronze = _bronze(data_root)

    frames: list[pl.DataFrame] = []
    player_frames: list[pl.DataFrame] = []
    for season_name in wanted:
        typer.echo(f"  {season_name} ...", nl=False)
        fetch_archive_season(season_name, bronze=bronze, run_id=run_id)

        gw_ref = bronze.latest(f"{SOURCE_ARCHIVE_GW}.{season_name}")
        players_ref = bronze.latest(f"{SOURCE_ARCHIVE_PLAYERS}.{season_name}")
        if gw_ref is None or players_ref is None:
            typer.secho(" missing after fetch", fg=typer.colors.RED)
            continue

        merged = pl.read_csv(
            io.BytesIO(bronze.read(gw_ref)), infer_schema_length=None, ignore_errors=True
        )
        players_raw = pl.read_csv(
            io.BytesIO(bronze.read(players_ref)), infer_schema_length=None, ignore_errors=True
        )
        result = normalize_archive_season(merged, players_raw, season=parse_season(season_name))
        frames.append(result.stats)
        player_frames.append(
            build_players_history(merged, players_raw, season=parse_season(season_name))
        )

        defensive = "with DC" if result.has_defensive_contributions else "no DC"
        typer.echo(
            f" {result.rows_out:>6,} rows ({defensive})"
            f"  dropped: {result.rows_dropped_unresolved_element} unresolved,"
            f" {result.rows_dropped_bad_kickoff} bad kickoff,"
            f" {result.rows_dropped_managers} managers"
        )

    if not frames:
        typer.secho("no seasons backfilled", fg=typer.colors.RED)
        raise typer.Exit(1)

    stats = pl.concat(frames, how="vertical")
    out_dir = data_root / "silver"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "player_gameweek_stats.parquet"
    stats.write_parquet(destination)
    typer.echo(f"\n  {stats.height:,} total rows -> {destination}")

    if player_frames:
        players_history = pl.concat(player_frames, how="vertical")
        players_dest = out_dir / "players_history.parquet"
        players_history.write_parquet(players_dest)
        typer.echo(f"  {players_history.height:,} player-seasons -> {players_dest}")


models_app = typer.Typer(
    name="models",
    help="Inspect, verify and audit saved model artifacts.",
    no_args_is_help=True,
)
app.add_typer(models_app)


def _manifest_for(saved: Any, path: Path, data_root: Path, season: Season) -> Any:
    """Assemble a manifest from every layer that owns part of it.

    `contracts` carries provenance and never computes it, so the catalogue hash
    comes from `features`, the rules hash from `domain` and the data hash from
    `storage`. This is the one place all three are in scope.
    """
    import platform

    from xg_alonso.contracts.artifacts import (
        ArtifactManifest,
        ComponentMetrics,
        RuntimeVersions,
    )
    from xg_alonso.domain.scoring import rules_snapshot_hash
    from xg_alonso.features.catalogue import CATALOGUE_VERSION
    from xg_alonso.features.schema import catalogue_hash
    from xg_alonso.prediction.trained import TRAINED_MODEL_NAME, TRAINED_MODEL_VERSION
    from xg_alonso.storage.training_manifest import training_data_manifest_hash

    resolved = _resolve_rules(data_root, season)
    skill = saved.models.skill_by_label()
    bias = saved.models.mean_bias_by_label()
    degenerate = set(saved.models.degenerate_labels())

    data_hash = ""
    silver = data_root / "silver" / "player_gameweek_stats.parquet"
    if silver.exists():
        data_hash, _ = training_data_manifest_hash(silver_path=silver)

    return ArtifactManifest(
        model_type=TRAINED_MODEL_NAME,
        model_name=path.stem,
        model_version=TRAINED_MODEL_VERSION,
        created_at=saved.saved_at,
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        code_version=_git_commit(),
        runtime=RuntimeVersions(python=platform.python_version()),
        feature_catalogue_version=CATALOGUE_VERSION,
        feature_catalogue_hash=catalogue_hash(),
        feature_names=tuple(saved.models.feature_columns),
        feature_count=len(saved.models.feature_columns),
        dropped_features=tuple(getattr(saved.models, "dropped_features", ()) or ()),
        rules_snapshot_hash=rules_snapshot_hash(resolved.scoring, resolved.squad),
        scoring_rules_version=resolved.scoring.version,
        training_data_manifest_hash=data_hash,
        training_seasons=tuple(saved.trained_seasons),
        training_gameweeks=tuple(saved.trained_gameweeks),
        training_rows=saved.models.trained_on_rows,
        component_metrics=tuple(
            ComponentMetrics(
                label=label,
                mean_skill=round(value, 6),
                mean_bias=round(bias.get(label, 0.0), 6),
                degenerate=label in degenerate,
                label_mean=round(saved.models.label_means.get(label, 0.0), 6),
                folds=len(saved.models.folds),
            )
            for label, value in sorted(skill.items())
        ),
        label_columns=tuple(sorted(saved.models.models)),
        model_fingerprint=saved.models.fingerprint(),
    )


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    """Whether the working tree has uncommitted changes.

    `ArtifactManifest.reproducible` is `bool(code_version) and not git_dirty`,
    and `git_dirty` defaults to `False` — so leaving it unset made every
    manifest claim reproducibility, including one written from a tree with
    uncommitted edits. That inverts the field's purpose: it exists precisely to
    say "the commit recorded here does not describe the code that ran".

    Uses the same probe as `pipelines/ingestion/bootstrap.py` and
    `discovery/experiment.py`, both of which already recorded this correctly.
    On failure it returns `True`, because an unknown tree state is not a clean
    one and the safe direction is to under-claim.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return True
    return bool(out.stdout.strip())


def _active_schema(data_root: Path, season: Season) -> Any:
    """The active feature schema, carrying the active rules hash.

    Every call site used the bare `ActiveSchema.from_catalogue()`, which leaves
    `rules_snapshot_hash` empty — and the rules-drift check is guarded by
    `if active.rules_snapshot_hash and manifest.rules_snapshot_hash and ...`.
    So the one **blocking** check that catches a model fitted under one scoring
    system and priced under another could never fire on any of them.

    Resolution is non-exact-tolerant: `_resolve_rules` records which season's
    snapshot it used, and a hash from the wrong season is still a real hash to
    compare against. Failing to resolve rules at all is not fatal here — it
    degrades to the previous behaviour, with the check inert, rather than
    refusing to list artifacts.
    """
    from xg_alonso.domain.scoring import rules_snapshot_hash
    from xg_alonso.prediction.artifacts import ActiveSchema

    try:
        resolved = _resolve_rules(data_root, season)
        rules_hash = rules_snapshot_hash(resolved.scoring, resolved.squad)
    except Exception:
        rules_hash = ""
    return ActiveSchema.from_catalogue(rules_snapshot_hash=rules_hash)


@models_app.command("list")
def models_list(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    show_all: Annotated[
        bool, typer.Option("--all", help="Include artifacts that cannot be used.")
    ] = False,
) -> None:
    """Every artifact and whether it can be used with the active build."""
    from xg_alonso.prediction.artifacts import read_manifest

    active = _active_schema(data_root, parse_season(season))
    paths = sorted((data_root / "models").glob("*.pkl"))
    if not paths:
        typer.echo(f"  no artifacts under {data_root / 'models'}")
        return

    typer.echo(
        f"  active schema: {len(active.feature_names)} features, {active.catalogue_version}\n"
    )
    for path in paths:
        verdict = _verify_one(path, active)
        usable = verdict.compatible
        if not usable and not show_all:
            typer.secho(f"  {path.name:<58} {verdict.status}", fg=typer.colors.RED)
            continue
        colour = typer.colors.GREEN if usable else typer.colors.RED
        manifest = read_manifest(path)
        detail = "" if manifest else "  (no manifest)"
        typer.secho(f"  {path.name:<58} {verdict.status}{detail}", fg=colour)


def _verify_one(path: Path, active: Any) -> Any:
    """Check one artifact without letting a refusal stop the listing."""
    from xg_alonso.prediction.artifacts import (
        ArtifactCompatibilityError,
        check_compatibility,
        payload_digest,
        read_manifest,
    )
    from xg_alonso.prediction.inference import load_models

    try:
        manifest = read_manifest(path)
    except ArtifactCompatibilityError as exc:
        return exc.compatibility

    try:
        saved = load_models(path, on_incompatible="ignore")
        columns: Any = saved.models.feature_columns
    except Exception:
        columns = None

    return check_compatibility(
        manifest,
        active=active,
        artifact_path=path,
        artifact_feature_columns=columns,
        payload_sha256=payload_digest(path) if manifest else None,
    )


@models_app.command("verify")
def models_verify(
    artifact: Annotated[Path, typer.Argument(help="Path to a .pkl artifact.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
) -> None:
    """Explain in full whether one artifact can be used, and why not."""
    if not artifact.exists():
        raise typer.BadParameter(f"no artifact at {artifact}")

    verdict = _verify_one(artifact, _active_schema(data_root, parse_season(season)))
    typer.echo(verdict.explain())
    if verdict.schema_diff is not None:
        diff = verdict.schema_diff
        typer.echo(
            f"\n  needs {len(diff.expected_order)} features; "
            f"{len(diff.missing)} unavailable, {len(diff.unexpected)} unused"
        )
    if not verdict.compatible:
        raise typer.Exit(1)


@models_app.command("backfill-manifest")
def models_backfill_manifest(
    artifact: Annotated[Path, typer.Argument(help="Path to a .pkl artifact.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
) -> None:
    """Write a manifest for an artifact saved before manifests existed.

    A backfilled manifest is a *description*, never a certificate. Fields that
    cannot be recovered — the git commit it was trained at, the data hash if
    the silver table has since changed — are left empty rather than guessed,
    and the compatibility check treats an empty hash as unknown rather than as
    a pass.
    """
    from xg_alonso.prediction.artifacts import payload_digest, write_manifest
    from xg_alonso.prediction.inference import load_models, replace_manifest

    if not artifact.exists():
        raise typer.BadParameter(f"no artifact at {artifact}")

    saved = load_models(artifact, on_incompatible="ignore")
    manifest = _manifest_for(saved, artifact, data_root, parse_season(season))
    manifest = replace_manifest(
        manifest,
        artifact_version="artifact_manifest_v1",
        payload_sha256=payload_digest(artifact),
        payload_bytes=artifact.stat().st_size,
    )
    written = write_manifest(artifact, manifest)
    typer.echo(
        f"  {artifact.name}: {manifest.feature_count} features, "
        f"trained on {', '.join(manifest.training_seasons) or 'unknown'}"
    )
    typer.echo(f"  wrote {written}")


@models_app.command("audit")
def models_audit(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    apply: Annotated[
        bool, typer.Option("--apply", help="Write manifests. Off by default.")
    ] = False,
) -> None:
    """Classify every artifact, and optionally give each one a manifest.

    Dry by default. `.data` is gitignored, so a mistake here has no safety net,
    and the expensive error is destroying provenance rather than failing to
    write it.
    """
    active = _active_schema(data_root, parse_season(season))
    paths = sorted((data_root / "models").glob("*.pkl"))
    if not paths:
        typer.echo(f"  no artifacts under {data_root / 'models'}")
        return

    mode = "APPLY" if apply else "dry run — nothing will be written"
    typer.echo(f"  auditing {len(paths)} artifact(s) [{mode}]\n")

    counts: dict[str, int] = {}
    for path in paths:
        verdict = _verify_one(path, active)
        counts[str(verdict.status)] = counts.get(str(verdict.status), 0) + 1
        colour = typer.colors.GREEN if verdict.compatible else typer.colors.RED
        typer.secho(f"  {path.name:<58} {verdict.status}", fg=colour)
        for finding in verdict.blocking:
            typer.secho(f"      {finding.reason}: {finding.detail[:100]}", fg=typer.colors.RED)

        if apply and verdict.compatible:
            models_backfill_manifest(path, data_root=data_root, season=season)

    typer.echo("")
    for status, count in sorted(counts.items()):
        typer.echo(f"  {count:>3}  {status}")
    if not apply:
        typer.echo("\n  re-run with --apply to write manifests")


evaluate_app = typer.Typer(
    name="evaluate",
    help="Multi-season, multi-squad policy evaluation.",
    no_args_is_help=True,
)
app.add_typer(evaluate_app)


@evaluate_app.command("check")
def evaluate_check(
    preset: Annotated[str, typer.Option("--preset", help="smoke, legacy or full.")] = "smoke",
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    model_path: Annotated[
        Path | None, typer.Option("--model", help="Artifact to freeze-check.")
    ] = None,
) -> None:
    """Run the freeze assertions and stop.

    These guard failures that produce *better*-looking numbers — a model
    trained on the season it is evaluated over does not crash, it wins — so
    they are worth running before an experiment rather than after it.
    """
    from xg_alonso.contracts.evaluation import PRESETS
    from xg_alonso.evaluation.frozen import FreezeViolation, assert_frozen
    from xg_alonso.prediction import load_models

    config = PRESETS.get(preset)
    if config is None:
        raise typer.BadParameter(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")

    models: dict[str, Any] = {spec.name: None for spec in config.models}
    if model_path is not None:
        saved = load_models(model_path)
        for spec in config.models:
            if spec.artifact_path is not None or spec.name != "closed_form":
                models[spec.name] = saved

    typer.echo(f"  {config.experiment_id}")
    typer.echo(f"  seasons: {', '.join(config.evaluation_seasons)}")
    try:
        checks = assert_frozen(config, models=models)
    except FreezeViolation as exc:
        typer.secho(f"\n  FROZEN CHECK FAILED\n  {exc}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(1) from exc

    typer.echo("")
    for check in checks:
        colour = typer.colors.GREEN if check.passed else typer.colors.YELLOW
        typer.secho(f"  {check.row()}", fg=colour)
    typer.echo(f"\n  {sum(c.passed for c in checks)}/{len(checks)} checks passed")


@evaluate_app.command("plan")
def evaluate_plan(
    preset: Annotated[str, typer.Option("--preset", help="smoke, legacy or full.")] = "smoke",
    squads: Annotated[int, typer.Option("--squads", help="How many starting squads.")] = 1,
) -> None:
    """Show what an experiment would run, without running any of it.

    The unit list is pure and total, so the cost of a grid is knowable before
    committing to it.
    """
    from xg_alonso.contracts.evaluation import PRESETS
    from xg_alonso.evaluation.runner import plan_units

    config = PRESETS.get(preset)
    if config is None:
        raise typer.BadParameter(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")

    units = plan_units(config, [f"squad-{i}" for i in range(squads)])
    by_policy: dict[str, int] = {}
    for unit in units:
        by_policy[unit.policy] = by_policy.get(unit.policy, 0) + 1

    typer.echo(f"  {config.experiment_id}")
    typer.echo(f"  seasons     {', '.join(config.evaluation_seasons)}")
    typer.echo(f"  conditions  {len({u.condition for u in units})}")
    typer.echo(f"  units       {len(units)}\n")
    for policy, count in sorted(by_policy.items(), key=lambda kv: -kv[1]):
        note = "  (replicated: stochastic)" if count > len(units) // len(by_policy) else ""
        typer.echo(f"    {policy:<18}{count:>5}{note}")


@evaluate_app.command("status")
def evaluate_status(
    experiment_id: Annotated[str, typer.Argument(help="Experiment directory name.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
) -> None:
    """How far along an experiment is, from its run files alone."""
    directory = data_root / "experiments" / experiment_id
    if not directory.exists():
        raise typer.BadParameter(f"no experiment at {directory}")

    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        typer.echo(f"  {manifest['experiment_id']}")
        typer.echo(f"  {manifest['runs_completed']}/{manifest['runs_planned']} runs")
        typer.echo(f"  reproducible: {'yes' if manifest['reproducible'] else 'no'}")
        for item in manifest.get("limitations", []):
            typer.echo(f"    ! {item}")
    else:
        completed = len(list((directory / "runs").glob("*.json")))
        typer.echo(f"  {completed} run(s) recorded, no manifest yet")


@evaluate_app.command("report")
def evaluate_report(
    experiment_id: Annotated[str, typer.Argument(help="Experiment directory name.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
) -> None:
    """Re-render every artifact from the run files.

    Nothing accumulates: the aggregate, the comparisons and the summary are
    recomputed each time, so a resumed experiment cannot emit a
    half-aggregated report that reads as complete.
    """
    from xg_alonso.contracts.evaluation import ExperimentConfig
    from xg_alonso.evaluation.experiment_report import (
        build_report,
        generate_limitations,
        load_runs,
        write_report,
    )

    directory = data_root / "experiments" / experiment_id
    config_path = directory / "config.json"
    if not config_path.exists():
        raise typer.BadParameter(f"no config at {config_path}")

    config = ExperimentConfig.model_validate_json(config_path.read_text())
    runs = load_runs(directory)
    if not runs:
        raise typer.BadParameter(f"no completed runs under {directory}")

    limitations = generate_limitations(config, runs)
    report = build_report(config, runs, limitations=limitations)
    written = write_report(directory, report)

    typer.echo((directory / "summary.md").read_text())
    typer.echo(f"  wrote {', '.join(sorted(written))} -> {directory}")


@app.command()
def backtest(
    season: Annotated[
        str, typer.Option("--season", help="Season to walk, e.g. 2025-26.")
    ] = "2025-26",
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    start_gw: Annotated[int, typer.Option("--from", help="First gameweek to evaluate.")] = 6,
    end_gw: Annotated[int, typer.Option("--to", help="Last gameweek to evaluate.")] = 38,
    model_path: Annotated[
        Path | None,
        typer.Option("--model", help="A fitted model to evaluate alongside the baseline."),
    ] = None,
) -> None:
    """Walk a past season, measuring recommendations against holding.

    This is the headline metric. Two identical squads walk the season — one
    takes every recommendation, one never transfers — and both are scored on the
    same actual results. A model that ranks players well but recommends badly
    looks good on MAE and bad here, which is the point.

    Starts at GW6 by default so the rolling features have in-season history to
    work with; earlier gameweeks lean entirely on the prior season.
    """
    import random

    from xg_alonso.contracts.identifiers import TeamId
    from xg_alonso.contracts.prediction import Position
    from xg_alonso.evaluation import (
        POLICIES,
        BacktestReport,
        actual_points,
        compare_to_previous,
        gameweek_deadlines,
        run_policy,
        walk_forward,
        write_report,
    )
    from xg_alonso.features.opponent import build_opponent_strength
    from xg_alonso.features.slice1 import build_slice1_features, build_team_gameweek_stats
    from xg_alonso.optimization.transfer import Candidate
    from xg_alonso.prediction.baseline import predict_frame

    parsed = parse_season(season)
    silver = data_root / "silver"
    stats_path = silver / "player_gameweek_stats.parquet"
    players_path = silver / "players_history.parquet"
    if not stats_path.exists() or not players_path.exists():
        raise typer.BadParameter("no backfill found. Run `xg backfill` first.")

    all_stats = pl.read_parquet(stats_path)
    history = pl.read_parquet(players_path).filter(pl.col("season") == str(parsed))
    if history.is_empty():
        raise typer.BadParameter(f"no player history for {season}. Backfill it first.")

    # The archive labels goalkeepers "GK"; the rest of the system uses "GKP".
    team_ids = {name: i + 1 for i, name in enumerate(sorted(history["team_name"].unique()))}
    players = history.with_columns(
        pl.when(pl.col("position") == "GK")
        .then(pl.lit("GKP"))
        .otherwise(pl.col("position"))
        .alias("position"),
        pl.col("team_name").replace_strict(team_ids, default=0).alias("team_id"),
        pl.col("opening_price").alias("current_price"),
    ).filter(pl.col("position").is_in([p.value for p in Position]))

    deadlines = gameweek_deadlines(all_stats).filter(pl.col("season") == str(parsed))
    deadline_by_gw = {int(r["gameweek_id"]): r["deadline"] for r in deadlines.iter_rows(named=True)}

    prices = {
        PlayerCode(int(r["player_code"])): TenthsOfMillion(int(r["current_price"]))
        for r in players.iter_rows(named=True)
    }
    positions = {
        PlayerCode(int(r["player_code"])): str(r["position"]) for r in players.iter_rows(named=True)
    }
    teams = {
        PlayerCode(int(r["player_code"])): int(r["team_id"]) for r in players.iter_rows(named=True)
    }
    names = {
        PlayerCode(int(r["player_code"])): str(r["web_name"]) for r in players.iter_rows(named=True)
    }

    # One resolution rather than two full context rebuilds, and it records
    # which season's rules these actually are.
    resolved = _resolve_rules(data_root, parsed)
    squad_rules = resolved.squad
    scoring = resolved.scoring
    if not resolved.exact:
        typer.secho(f"  note: {resolved.caveat}", fg=typer.colors.YELLOW)

    trained = None
    if model_path is not None:
        from xg_alonso.prediction import load_models

        trained = load_models(model_path)
        evaluated = tuple(range(start_gw, end_gw + 1))
        if trained.overlaps(str(parsed), evaluated):
            raise typer.BadParameter(
                f"{model_path} was trained on {season} gameweeks that overlap "
                f"GW{start_gw}-{end_gw}. Backtesting a model on its own training "
                "data measures memorisation, not skill. Train on an earlier season."
            )
        typer.echo(
            f"  model {trained.models.fingerprint()[:12]} "
            f"trained on {', '.join(trained.trained_seasons)}"
        )

    initial = _opening_squad(players, squad_rules, parsed, start_gw)
    typer.echo(
        f"  initial squad: {len(initial.picks)} players, "
        f"bank {int(initial.bank) / 10:.1f}m, {season} GW{start_gw}-{end_gw}"
    )

    prediction_cache: dict[int, Any] = {}

    def _inputs_for(gameweek: GameweekId) -> tuple[Any, Any, Any]:
        """Predictions and candidates for one gameweek, computed once.

        Cached so every policy is evaluated against byte-identical inputs — a
        control that saw different predictions would not be a control.
        """
        cached = prediction_cache.get(int(gameweek))
        if cached is not None:
            result: tuple[Any, Any, Any] = cached
            return result

        cutoff = deadline_by_gw.get(int(gameweek))
        if cutoff is None:
            raise KeyError(f"no deadline for GW{gameweek}")

        entities = players.select(
            "player_code", "position", "team_id", "current_price", "web_name"
        ).with_columns(pl.lit(cutoff).alias("prediction_timestamp"))

        team_stats = build_team_gameweek_stats(all_stats, players)
        features = build_slice1_features(entities, player_stats=all_stats, team_stats=team_stats)
        predictions = predict_frame(
            features,
            rules=scoring,
            from_gameweek=gameweek,
            data_cutoff=cutoff,
            predicted_at=cutoff,
            run_id="backtest",
            code_version="backtest",
            feature_set_version="slice1_v1",
        )
        by_code = collapse_by_player(predictions)
        candidates = [
            Candidate(
                player_code=p.player_code,
                position=p.position,
                team_id=TeamId(teams[p.player_code]),
                price=prices[p.player_code],
                prediction=p,
            )
            for p in predictions
            if p.player_code in prices
        ]
        computed: tuple[Any, Any, Any] = (by_code, candidates, cutoff)
        prediction_cache[int(gameweek)] = computed
        return computed

    opponent_strength = (
        build_opponent_strength(all_stats) if trained is not None else pl.DataFrame()
    )
    trained_cache: dict[tuple[int, bool], Any] = {}

    def _trained_inputs_for(gameweek: GameweekId, shrink: bool = False) -> tuple[Any, Any, Any]:
        """The same entities and cutoff, but predicted by the fitted model.

        Sharing the entity frame and the legality check means a comparison
        between the closed-form and trained policies isolates prediction
        quality — nothing else differs.
        """
        cached = trained_cache.get((int(gameweek), shrink))
        if cached is not None:
            result: tuple[Any, Any, Any] = cached
            return result

        assert trained is not None
        from xg_alonso.features.assemble import build_model_features
        from xg_alonso.features.catalogue import CATALOGUE_VERSION
        from xg_alonso.prediction import predict_with_models

        _, _, cutoff = _inputs_for(gameweek)

        # Who each player faces this gameweek. Fixtures are published well
        # before the deadline, so this is an input rather than a leak — only
        # the result of the match is withheld.
        fixture = (
            all_stats.filter(
                (pl.col("season") == str(parsed)) & (pl.col("gameweek_id") == int(gameweek))
            )
            .select("player_code", "opponent_team_id", "was_home")
            .unique(subset=["player_code"], keep="first", maintain_order=True)
        )
        entities = (
            players.select("player_code", "position", "team_id", "current_price", "web_name")
            .join(fixture, on="player_code", how="left")
            .with_columns(pl.lit(cutoff).alias("prediction_timestamp"))
        )
        features = build_model_features(
            entities, player_stats=all_stats, opponent_strength=opponent_strength
        )

        predictions = predict_with_models(
            features,
            models=trained.models,
            rules=scoring,
            from_gameweek=gameweek,
            data_cutoff=cutoff,
            predicted_at=cutoff,
            run_id="backtest-trained",
            code_version="backtest",
            feature_set_version=CATALOGUE_VERSION,
            shrink_by_skill=shrink,
        )
        by_code = collapse_by_player(predictions)
        candidates = [
            Candidate(
                player_code=p.player_code,
                position=p.position,
                team_id=TeamId(teams[p.player_code]),
                price=prices[p.player_code],
                prediction=p,
            )
            for p in predictions
            if p.player_code in prices
        ]
        computed: tuple[Any, Any, Any] = (by_code, candidates, cutoff)
        trained_cache[(int(gameweek), shrink)] = computed
        return computed

    gameweeks = [
        GameweekId(gw)
        for gw in range(start_gw, end_gw + 1)
        if gw in deadline_by_gw and actual_points(all_stats, season=parsed, gameweek=GameweekId(gw))
    ]

    policies = dict(POLICIES)
    if trained is not None:
        # Same selection rule as `model`, different predictions. The gap between
        # the two is exactly the value the fitted model adds.
        policies["trained"] = POLICIES["model"]
        # Same model, same selection rule — the only difference is whether weak
        # components are allowed to speak as loudly as strong ones.
        policies["trained_shrunk"] = POLICIES["model"]

    results: dict[str, Any] = {}
    for policy_name, selector in policies.items():
        rng = random.Random(20260727)

        def recommend_at(
            squad_state: Any,
            gameweek: GameweekId,
            season_arg: Any,
            _sel: Any = selector,
            _rng: Any = rng,
            _name: str = policy_name,
        ) -> Any:
            if _name.startswith("trained"):
                by_code, candidates, cutoff = _trained_inputs_for(
                    gameweek, _name == "trained_shrunk"
                )
            else:
                by_code, candidates, cutoff = _inputs_for(gameweek)
            recommendation = run_policy(
                squad_state,
                selector=_sel,
                candidates=candidates,
                predictions=by_code,
                rules=squad_rules,
                entry_id=EntryId(0),
                gameweek=gameweek,
                generated_at=cutoff,
                run_id=f"backtest-{_name}",
                rng=_rng,
                policy_name=_name,
            )
            return recommendation, by_code

        typer.echo(f"  {policy_name} ...", nl=False)
        results[policy_name] = walk_forward(
            initial_squad=initial,
            season=parsed,
            gameweeks=gameweeks,
            recommend_fn=recommend_at,
            player_stats=all_stats,
            prices=prices,
            positions=positions,
            teams=teams,
            rules=squad_rules,
        )
        typer.echo(" done")

    typer.echo("\n" + "─" * 72)
    typer.echo(f"  BACKTEST — {season} GW{start_gw}-{end_gw}, {len(gameweeks)} gameweeks")
    typer.echo("─" * 72 + "\n")
    typer.echo(
        f"  {'policy':<16}{'vs hold':>10}{'transfers':>11}"
        f"{'win rate':>10}{'pts/xfer':>10}{'pred err':>10}"
    )
    typer.echo("  " + "-" * 67)
    for policy_name, result in sorted(results.items(), key=lambda kv: -kv[1].total_incremental):
        typer.echo(
            f"  {policy_name:<16}{result.total_incremental:>+10d}{result.transfers_made:>11}"
            f"{result.decision_win_rate:>9.0%}{result.mean_decision_delta:>+10.2f}"
            f"{result.calibration_error:>10.2f}"
        )

    headline = max(
        (n for n in results if n != "hold"),
        key=lambda n: results[n].total_incremental,
    )
    model = results[headline]
    best_control = max(
        (r.total_incremental for n, r in results.items() if n not in (headline, "hold")),
        default=0,
    )
    typer.echo("\n" + "─" * 72)
    if model.total_incremental > best_control:
        typer.secho(
            f"  '{headline}' beats every other policy by "
            f"{model.total_incremental - best_control:+d} pts.",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            f"  '{headline}' does NOT beat the best alternative ({best_control:+d} pts). "
            "It is adding nothing over a simpler policy.",
            fg=typer.colors.RED,
        )
    typer.echo(
        "  Note: a single season and one starting squad is a small sample.\n"
        "  Treat the ordering as a signal, not the margin."
    )

    report = BacktestReport(
        run_id=f"bt-{uuid.uuid4().hex[:12]}",
        created_at=utc_now(),
        season=str(parsed),
        first_gameweek=start_gw,
        last_gameweek=end_gw,
        code_version=git_manifest("backtest", run_id="report").git_commit,
        model_fingerprint=trained.models.fingerprint() if trained else None,
        model_trained_on=trained.trained_seasons if trained else (),
        results=results,
    )
    reports_root = data_root / "reports"
    delta = compare_to_previous(report, reports_root)
    destination = write_report(report, reports_root)

    if delta:
        typer.echo("\n  Change since the previous run on this window:")
        for name, (was, now) in sorted(delta.items(), key=lambda kv: -kv[1][1]):
            arrow = "+" if now > was else ("=" if now == was else "")
            typer.echo(f"    {name:<16}{was:>+6d} -> {now:>+6d}   ({now - was:+d}) {arrow}")
    else:
        typer.echo("\n  No previous run on this window to compare against.")
    typer.echo(f"  saved -> {destination}")

    worst = sorted((o for o in model.outcomes if o.transfer_made), key=lambda o: o.decision_delta)
    if worst:
        typer.echo(f"\n  {headline} weakest decisions:")
        for outcome in worst[:3]:
            out_name = names.get(outcome.player_out, "?") if outcome.player_out else "?"
            in_name = names.get(outcome.player_in, "?") if outcome.player_in else "?"
            typer.echo(
                f"    GW{outcome.gameweek:<3}{out_name[:20]:>21} -> {in_name[:20]:<21}"
                f"{outcome.decision_delta:+4d} pts (predicted {outcome.predicted_gain:+.1f})"
            )


def _opening_squad(players: pl.DataFrame, squad_rules, season, gameweek):  # type: ignore[no-untyped-def]
    """Build a realistic opening squad for a backtest.

    **This is load-bearing for the metric, not setup detail.** An earlier version
    picked the cheapest legal 15, which left 36m unspent and produced a hold
    baseline of players who barely appear. Everything beats that, so the
    backtest reported a 100% beat-hold rate and hundreds of incremental points —
    a number that measured the starting squad, not the recommendations.

    A credible baseline spends the budget the way a manager would: the most
    expensive legal squad that fits, built from opening prices only, so it
    encodes no knowledge of how the season actually went.
    """
    from xg_alonso.contracts.identifiers import TeamId
    from xg_alonso.contracts.prediction import Position
    from xg_alonso.contracts.squad import SquadPick, SquadState

    budget = int(squad_rules.total_budget)
    club_count: dict[int, int] = {}
    chosen: dict[str, list[dict[str, Any]]] = {}

    # Reserve a floor for the positions not yet filled, so spending early does
    # not leave the squad unable to complete itself legally.
    order = sorted(squad_rules.positions, key=lambda r: -r.squad_select)
    remaining_slots = sum(r.squad_select for r in squad_rules.positions)
    spend = 0

    for rule in order:
        pool = list(
            players.filter(pl.col("position") == rule.position.value)
            .sort("current_price", descending=True)
            .iter_rows(named=True)
        )
        cheapest = min((int(r["current_price"]) for r in pool), default=40)
        picked: list[dict[str, Any]] = []
        for row in pool:
            if len(picked) == rule.squad_select:
                break
            team = int(row["team_id"])
            if club_count.get(team, 0) >= squad_rules.max_per_club:
                continue
            price = int(row["current_price"])
            # Leave enough to fill every remaining slot at the cheapest price.
            slots_after = remaining_slots - len(picked) - 1
            if spend + price + slots_after * cheapest > budget:
                continue
            club_count[team] = club_count.get(team, 0) + 1
            picked.append(row)
            spend += price
        if len(picked) < rule.squad_select:
            raise typer.BadParameter(
                f"could not fill {rule.position.value} within budget; "
                f"only {len(picked)} of {rule.squad_select} affordable"
            )
        remaining_slots -= rule.squad_select
        chosen[rule.position.value] = picked

    starting = chosen["GKP"][:1] + chosen["DEF"][:4] + chosen["MID"][:4] + chosen["FWD"][:2]
    bench = chosen["GKP"][1:] + chosen["DEF"][4:] + chosen["MID"][4:] + chosen["FWD"][2:]

    picks, total = [], 0
    for slot, row in enumerate(starting + bench, start=1):
        price = TenthsOfMillion(int(row["current_price"]))
        picks.append(
            SquadPick(
                player_code=PlayerCode(int(row["player_code"])),
                position=Position(str(row["position"])),
                team_id=TeamId(int(row["team_id"])),
                purchase_price=price,
                current_price=price,
                selling_price=price,
                squad_slot=slot,
                # No captain is set here. The XI and captain are chosen from
                # predictions at decision time; baking one in at squad
                # construction is how every earlier run ended up captaining
                # the goalkeeper.
                is_captain=False,
                is_vice_captain=False,
            )
        )
        total += int(price)

    return SquadState(
        entry_id=EntryId(0),
        gameweek=GameweekId(gameweek),
        picks=tuple(picks),
        bank=TenthsOfMillion(budget - total),
        free_transfers=1,
    )


def _objective_feature_columns(
    data_root: Path,
    objective_id: str,
    frame: pl.DataFrame,
    player_stats: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Append an objective's accepted discovered features to a frame.

    **This is the wire that makes discovery matter.** Without it the registry
    records which features serve which objective and nothing downstream reads
    it, so every objective is scored by the same fixed catalogue and a squad
    built to chase a mini-league is identical to one built to protect a rank.

    It lives in the CLI rather than in `prediction` because the import lattice
    puts discovery *above* prediction: the prediction layer must not know that
    feature discovery exists. Composing the two is an application concern, and
    this is the application.

    Returns the frame with the extra columns and the names that were added.
    A feature whose program fails to compile against this frame is skipped with
    a warning rather than aborting the run — one bad program in a registry of
    many should not cost a manager their squad — but the skip is announced,
    because a silently thinner feature set is a silently different model.
    """
    from xg_alonso.discovery.compile import CompileContext, compile_program
    from xg_alonso.discovery.dsl import ProgramError, parse_program
    from xg_alonso.discovery.registry import DiscoveryRegistry
    from xg_alonso.features.generators import stage_window
    from xg_alonso.storage.parquet_store import ParquetTableStore

    directory = data_root / "discovery"
    if not directory.exists():
        return frame, []

    registry = DiscoveryRegistry(ParquetTableStore(directory))
    specs = registry.accepted_features(objective_id)
    if not specs:
        return frame, []

    context = CompileContext(player_stats=player_stats, stage=stage_window)
    added: list[str] = []
    for spec in specs:
        if spec.name in frame.columns:
            continue
        try:
            program = parse_program(spec.name, spec.program)
            frame = compile_program(program, frame, context)
        except (ProgramError, ValueError, KeyError) as exc:
            typer.secho(f"    skipped {spec.name}: {exc}", fg=typer.colors.YELLOW)
            continue
        added.append(spec.name)

    return frame, added


@app.command(name="plan")
def plan_command(
    request: Annotated[str, typer.Argument(help="What you want, in plain English.")],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: Annotated[
        str, typer.Option("--season", help="Season in YYYY-YY form.")
    ] = DEFAULT_SEASON,
    model: Annotated[Path | None, typer.Option("--model", help="Trained models to use.")] = None,
    preset: Annotated[
        str, typer.Option("--preset", help="Objective to start from before your words modify it.")
    ] = "expected_points",
    show_unconstrained: Annotated[
        bool,
        typer.Option(
            "--show-free", help="Also print the squad you would get with no requirements."
        ),
    ] = False,
) -> None:
    """Build a squad around requirements you type in plain English.

    "I want Haaland starting, keep Saliba, play 3-5-2 and leave 0.5 in the bank"

    Everything you ask for becomes a hard constraint on the solver, never a
    penalty — so a requirement is either honoured exactly or reported as
    impossible. When a set cannot all hold, the lowest-priority requirements are
    dropped until one can and every dropped one is named, because "no legal
    squad exists" is not something a manager can act on.

    The parsed interpretation is printed **before** anything is built. Natural
    language is ambiguous, and a system that acts on its own guess without
    showing it will eventually act on a wrong one silently.
    """
    from xg_alonso.contracts.identifiers import EntryId, TeamId
    from xg_alonso.domain.intent import build_name_index, compile_intent
    from xg_alonso.optimization.squad_builder import SquadCandidate, build_constrained_squad

    parsed = parse_season(season)
    context = _load_context(data_root, parsed)
    gameweek = context.next_gameweek()

    names = context.player_names()
    name_index = build_name_index({int(code): name for code, name in names.items()})
    team_index: dict[str, int] = {}
    for row in context.teams.iter_rows(named=True):
        team_index[str(row["name"])] = int(row["team_id"])
        if row.get("short_name"):
            team_index[str(row["short_name"])] = int(row["team_id"])

    intent = compile_intent(
        request,
        players=name_index,
        teams=team_index,
        base_preset=preset,
        next_gameweek=int(gameweek),
    )

    typer.echo("Parsed request")
    typer.echo(f"  objective        {intent.bundle.objective.id}")
    if intent.requirements.requirements:
        for requirement in intent.requirements.requirements:
            typer.echo(f"  {requirement.kind.value:<14} {requirement.label}")
    else:
        typer.echo("  (no structural requirements — building the free optimum)")
    if intent.unparsed:
        typer.secho(f"  not understood   {list(intent.unparsed)}", fg=typer.colors.YELLOW)
    typer.echo(f"  confidence       {intent.overall_confidence:.0%}")

    # An objective-trained model is preferred when one exists, because it was
    # fitted on the features discovery found to serve *this* goal. Falling back
    # to the general model is correct rather than an error — it just means the
    # squad is chosen by the catalogue alone, and the run says so.
    from xg_alonso.prediction import load_models

    objective_model_path = (
        data_root / "models" / f"component_models.{intent.bundle.objective.id}.pkl"
    )
    models = None
    model_note = "general model"
    chosen_model = model
    if chosen_model is None:
        objective_model = (
            data_root / "models" / f"component_models.{intent.bundle.objective.id}.pkl"
        )
        general = data_root / "models" / "component_models.pkl"
        if objective_model.exists():
            chosen_model = objective_model
            model_note = f"model trained for {intent.bundle.objective.id}"
        elif general.exists():
            chosen_model = general
    if chosen_model is not None:
        models = load_models(chosen_model).models
    else:
        model_note = "closed-form baseline (no trained model found)"

    predictions = _predict_all(
        context,
        gameweek,
        models,
        data_root=data_root,
        objective_id=intent.bundle.objective.id if chosen_model == objective_model_path else "",
    )

    # The objective chooses the feature set. This is what makes the same
    # requirements produce a different squad under a different goal.
    frame_note = f"  scoring with the {model_note}"

    available = {
        int(r["player_code"]): r
        for r in context.players.iter_rows(named=True)
        if r.get("status") in (None, "a", "d")
    }
    candidates = [
        SquadCandidate(
            player_code=p.player_code,
            position=p.position,
            team_id=TeamId(int(available[int(p.player_code)]["team_id"])),
            price=TenthsOfMillion(int(available[int(p.player_code)]["current_price"])),
            prediction=p,
        )
        for p in predictions
        if int(p.player_code) in available
    ]

    typer.echo(f"\nChoosing from {len(candidates):,} available players for GW{int(gameweek)}")
    if frame_note:
        typer.echo(frame_note)

    built = build_constrained_squad(
        candidates,
        rules=context.squad_rules,
        entry_id=EntryId(0),
        gameweek=gameweek,
        requirements=intent.requirements,
    )

    typer.echo(
        f"\nSquad  {built.expected_points:.2f} pts"
        f"  ({built.selection.formation_label}, {int(built.squad.bank) / 10:.1f}m banked)"
    )
    if built.total_cost > 0:
        typer.echo(
            f"  Your requirements cost {built.total_cost:.2f} points against the "
            f"free optimum of {built.unconstrained_points:.2f}."
        )

    starters = ", ".join(
        names.get(p.player_code, str(p.player_code))
        + ("(C)" if p.is_captain else "(V)" if p.is_vice_captain else "")
        for p in built.selection.starters
    )
    bench = ", ".join(names.get(p.player_code, str(p.player_code)) for p in built.selection.bench)
    typer.echo(f"  XI     {starters}")
    typer.echo(f"  Bench  {bench}")

    if built.outcomes:
        typer.echo("\nWhat you asked for")
        for outcome in built.outcomes:
            colour = typer.colors.RED if not outcome.honoured else None
            typer.secho(f"  {outcome.summary}", fg=colour)

    if not built.feasible_as_asked:
        typer.secho(
            "\n  Not everything could hold at once. The requirements above marked "
            "unhonoured are the ones that had to give.",
            fg=typer.colors.YELLOW,
        )

    if show_unconstrained:
        free = build_constrained_squad(
            candidates,
            rules=context.squad_rules,
            entry_id=EntryId(0),
            gameweek=gameweek,
            price_requirements=False,
        )
        free_xi = ", ".join(
            names.get(p.player_code, str(p.player_code)) for p in free.selection.starters
        )
        typer.echo(f"\nWithout your requirements  {free.expected_points:.2f} pts")
        typer.echo(f"  XI     {free_xi}")


@app.command()
def train(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    objective: Annotated[
        str,
        typer.Option(
            "--objective",
            help=(
                "Train on this objective's accepted discovered features as well as "
                "the catalogue. Saves alongside the objective id so `xg plan` can find it."
            ),
        ),
    ] = "",
    seasons: Annotated[
        str, typer.Option("--seasons", help="Comma-separated training seasons.")
    ] = "2024-25",
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to write the fitted model.")
    ] = None,
    min_gameweek: Annotated[
        int, typer.Option("--min-gw", help="Skip opening gameweeks with empty windows.")
    ] = 4,
) -> None:
    """Fit component models on historical seasons.

    Train on seasons *before* the one you intend to evaluate. A model fitted on
    the period it is later backtested over measures memorisation, and the result
    looks entirely reasonable — which is why the saved artifact records what it
    was trained on and the backtest refuses to use an overlapping model.
    """
    from xg_alonso.prediction import (
        SavedModel,
        build_training_frame,
        model_summary,
        save_models,
        train_component_models,
    )

    stats_path = data_root / "silver" / "player_gameweek_stats.parquet"
    if not stats_path.exists():
        raise typer.BadParameter("no backfill found. Run `xg backfill` first.")

    wanted = [s.strip() for s in seasons.split(",")]
    stats = pl.read_parquet(stats_path)

    typer.echo(f"  building training frame from {', '.join(wanted)} ...")
    data = build_training_frame(stats, seasons=wanted, min_gameweek=min_gameweek)
    typer.echo(f"    {data.rows:,} rows x {len(data.feature_columns)} features")

    # Objective-conditioned training. Discovery decides which features serve
    # which goal; this is where that decision reaches the model. Without it the
    # registry is a filing cabinet — every objective would be scored by the same
    # fixed catalogue and "chase a mini-league" would return the same squad as
    # "protect my rank".
    frame = data.frame
    feature_columns = list(data.feature_columns)
    if objective:
        frame, discovered = _objective_feature_columns(data_root, objective, frame, stats)
        if discovered:
            feature_columns.extend(discovered)
            typer.echo(
                f"    + {len(discovered)} discovered for {objective}: {', '.join(discovered)}"
            )
        else:
            typer.secho(
                f"    no accepted features for {objective}; "
                "run `xg discover` first or this is just the catalogue",
                fg=typer.colors.YELLOW,
            )

    typer.echo("  fitting component models ...")
    models = train_component_models(
        frame,
        feature_columns=tuple(feature_columns),
        label_columns=data.label_columns,
    )

    saved = SavedModel(
        models=models,
        trained_seasons=data.seasons,
        trained_gameweeks=data.gameweeks,
        saved_at=utc_now(),
    )
    default_name = f"component_models.{objective}.pkl" if objective else "component_models.pkl"
    destination = out or (data_root / "models" / default_name)
    save_models(saved, destination)

    summary = model_summary(saved)
    typer.echo(
        f"\n  {len(summary['labels'])} models, {summary['folds']} walk-forward folds"
        f", fingerprint {summary['fingerprint']}"
    )
    degenerate = models.degenerate_labels()
    bias = models.mean_bias_by_label()

    typer.echo("\n  Out-of-sample skill (degenerate models report zero):")
    for label, skill in sorted(summary["skill"].items(), key=lambda kv: -kv[1]):
        note = ""
        if label in degenerate:
            note = "   <- CONSTANT OUTPUT, carries no information"
        elif skill <= 0.02:
            note = "   <- no better than a constant"
        typer.echo(f"    {label:<26}{skill:>+8.1%}   bias {bias.get(label, 0.0):+7.3f}{note}")

    if degenerate:
        typer.secho(
            f"\n  WARNING: {len(degenerate)} component(s) output a constant: "
            f"{', '.join(degenerate)}.\n"
            "  They add only an offset to expected points, so rankings are driven\n"
            "  entirely by whatever else remains. Check the loss function.",
            fg=typer.colors.RED,
            bold=True,
        )
    typer.echo(f"\n  saved -> {destination}")


@app.command(name="build-squad")
def build_squad_command(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    model_path: Annotated[
        Path | None,
        typer.Option("--model", help="Use trained component models."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the squad as a picks JSON file.")
    ] = None,
) -> None:
    """Build a squad from scratch — the gameweek-1 answer.

    At GW1 nobody has a squad, so there is nothing to transfer from. This picks
    a legal fifteen maximising the expected points of the eleven it would field,
    which is not the same as maximising the squad total: four expensive bench
    players score nothing.
    """
    from xg_alonso.optimization import SquadCandidate, build_squad

    context = _load_context(data_root, parse_season(season))
    gameweek = context.next_gameweek()

    models = None
    if model_path is not None:
        from xg_alonso.prediction import load_models

        models = load_models(model_path).models

    predictions = _predict_all(context, gameweek, models)
    by_code = collapse_by_player(predictions)

    available = {
        int(r["player_code"]): r
        for r in context.players.iter_rows(named=True)
        if r.get("status") in (None, "a", "d")
    }
    candidates = [
        SquadCandidate(
            player_code=p.player_code,
            position=p.position,
            team_id=TeamId(int(available[int(p.player_code)]["team_id"])),
            price=TenthsOfMillion(int(available[int(p.player_code)]["current_price"])),
            prediction=p,
        )
        for p in predictions
        if int(p.player_code) in available
    ]
    typer.echo(f"  choosing from {len(candidates):,} available players for GW{gameweek}")

    squad, selection = build_squad(
        candidates,
        rules=context.squad_rules,
        entry_id=EntryId(0),
        gameweek=gameweek,
        predictions=by_code,
    )

    names = context.player_names()

    def rows(picks: tuple[SquadPick, ...]) -> list[tuple[str, str, float]]:
        return [
            (
                names.get(p.player_code, str(p.player_code))
                + (" (C)" if p.is_captain else " (V)" if p.is_vice_captain else ""),
                p.position.value,
                by_code[p.player_code].expected_points if p.player_code in by_code else 0.0,
            )
            for p in picks
        ]

    typer.echo(
        render_squad_summary(
            entry_id=0,
            gameweek=int(gameweek),
            squad_value=int(squad.squad_value),
            bank=int(squad.bank),
            free_transfers=squad.free_transfers,
            starters=rows(squad.starters),
            bench=rows(squad.bench),
        )
    )
    typer.echo(
        f"\n  formation {selection.formation_label}   projected {selection.expected_points:.2f} pts"
    )

    if out is not None:
        by_element = {
            int(r["player_code"]): int(r["element_id"])
            for r in context.players.iter_rows(named=True)
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "picks": [
                        {
                            "element": by_element[int(p.player_code)],
                            "position": p.squad_slot,
                            "purchase_price": int(p.purchase_price),
                            "is_captain": p.is_captain,
                            "is_vice_captain": p.is_vice_captain,
                        }
                        for p in squad.picks
                    ],
                    "bank": int(squad.bank),
                    "free_transfers": squad.free_transfers,
                },
                indent=1,
            )
        )
        typer.echo(f"  saved -> {out}")


def _predict_all(  # type: ignore[no-untyped-def]
    context: SliceContext,
    gameweek: GameweekId,
    models: object | None,
    *,
    data_root: Path | None = None,
    objective_id: str = "",
):
    """Predict every player, via the trained models when supplied.

    ``objective_id`` materialises that objective's accepted discovered features
    onto the prediction frame. A model trained with them expects them, and a
    prediction frame built from the catalogue alone would be missing columns —
    so this is the other half of the wire that `xg train --objective` opens.
    Both halves read the same registry through the same helper, which is what
    keeps training and inference describing the same feature set.
    """
    from xg_alonso.cli.pipeline import build_entities
    from xg_alonso.features.assemble import build_model_features
    from xg_alonso.features.catalogue import CATALOGUE_VERSION
    from xg_alonso.features.slice1 import build_slice1_features, build_team_gameweek_stats
    from xg_alonso.prediction import predict_with_models
    from xg_alonso.prediction.baseline import predict_frame

    cutoff = context.deadline_for(gameweek)
    entities = build_entities(context, cutoff=cutoff)

    def _temper(predictions: list[Any]) -> list[Any]:
        """Apply what is known about a player beyond the model's features.

        Delegates to `prediction.adjustments` so this path, `xg recommend` and
        the API apply the same adjustments in the same order. They previously
        did not, and the divergence was invisible: the same team id produced
        three different projections depending on which surface was asked.

        Price calibration and availability come from the payload already loaded,
        so they always apply. Signals need a file, so without a data root they
        are simply absent rather than the whole adjustment being skipped.
        """
        from xg_alonso.prediction.adjustments import adjust_predictions
        from xg_alonso.prediction.form import load_signals

        rows = list(context.players.iter_rows(named=True))
        prices = {
            PlayerCode(int(row["player_code"])): TenthsOfMillion(int(row["current_price"]))
            for row in rows
            if row.get("current_price") is not None
        }
        chances = {
            PlayerCode(int(row["player_code"])): row.get("chance_of_playing_next_round")
            for row in rows
        }
        signals = (
            load_signals(data_root / "signals" / "form_signals.json")
            if data_root is not None
            else None
        )
        return adjust_predictions(
            predictions, prices=prices, chances=chances, signals=signals, at=cutoff
        )

    if models is not None:
        features = build_model_features(entities, player_stats=context.player_stats)
        if objective_id and data_root is not None:
            features, _ = _objective_feature_columns(
                data_root, objective_id, features, context.player_stats
            )
        return _temper(
            predict_with_models(
                features,
                models=models,  # type: ignore[arg-type]
                rules=context.scoring,
                from_gameweek=gameweek,
                data_cutoff=cutoff,
                predicted_at=cutoff,
                run_id="build-squad",
                code_version="cli",
                feature_set_version=CATALOGUE_VERSION,
            )
        )

    features = build_slice1_features(
        entities,
        player_stats=context.player_stats,
        team_stats=build_team_gameweek_stats(context.player_stats, context.players),
    )
    return _temper(
        predict_frame(
            features,
            rules=context.scoring,
            from_gameweek=gameweek,
            data_cutoff=cutoff,
            predicted_at=cutoff,
            run_id="build-squad",
            code_version="cli",
            feature_set_version="slice1_v1",
        )
    )


@app.command()
def score(
    season: Annotated[
        str, typer.Option("--season", help="Season to score, e.g. 2025-26.")
    ] = "2025-26",
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    start_gw: Annotated[int, typer.Option("--from", help="First gameweek to score.")] = 6,
    end_gw: Annotated[int, typer.Option("--to", help="Last gameweek to score.")] = 38,
    model_path: Annotated[
        Path | None,
        typer.Option("--model", help="Score a fitted model instead of the closed-form baseline."),
    ] = None,
) -> None:
    """Score assembled expected points against what actually happened.

    Every other accuracy command here scores a *component*. This one scores the
    number the interface displays and the optimizer consumes, which no metric in
    the repository previously touched — the chain could have been sound at every
    link and wrong end to end.

    Both error and ranking are reported, because they answer different
    questions. A constant bias ruins MAE while leaving every transfer decision
    intact; a well-calibrated model that ranks nobody correctly is useless to
    the optimizer. Slices by position and price band come with it, since a
    pooled number hides a model that is excellent at goalkeepers and blind to
    forwards.
    """
    from xg_alonso.contracts.prediction import Position
    from xg_alonso.evaluation import (
        actual_points,
        gameweek_deadlines,
        score_predictions,
    )
    from xg_alonso.features.opponent import build_opponent_strength
    from xg_alonso.features.slice1 import build_slice1_features, build_team_gameweek_stats
    from xg_alonso.prediction.baseline import predict_frame

    parsed = parse_season(season)
    silver = data_root / "silver"
    stats_path = silver / "player_gameweek_stats.parquet"
    players_path = silver / "players_history.parquet"
    if not stats_path.exists() or not players_path.exists():
        raise typer.BadParameter("no backfill found. Run `xg backfill` first.")

    all_stats = pl.read_parquet(stats_path)
    history = pl.read_parquet(players_path).filter(pl.col("season") == str(parsed))
    if history.is_empty():
        raise typer.BadParameter(f"no player history for {season}. Backfill it first.")

    team_ids = {name: i + 1 for i, name in enumerate(sorted(history["team_name"].unique()))}
    players = history.with_columns(
        pl.when(pl.col("position") == "GK")
        .then(pl.lit("GKP"))
        .otherwise(pl.col("position"))
        .alias("position"),
        pl.col("team_name").replace_strict(team_ids, default=0).alias("team_id"),
        pl.col("opening_price").alias("current_price"),
    ).filter(pl.col("position").is_in([p.value for p in Position]))

    deadlines = gameweek_deadlines(all_stats).filter(pl.col("season") == str(parsed))
    deadline_by_gw = {int(r["gameweek_id"]): r["deadline"] for r in deadlines.iter_rows(named=True)}

    resolved = _resolve_rules(data_root, parsed)
    scoring = resolved.scoring
    if not resolved.exact:
        typer.secho(f"  note: {resolved.caveat}", fg=typer.colors.YELLOW)

    trained = None
    if model_path is not None:
        from xg_alonso.prediction import load_models

        trained = load_models(model_path)
        evaluated = tuple(range(start_gw, end_gw + 1))
        if trained.overlaps(str(parsed), evaluated):
            raise typer.BadParameter(
                f"{model_path} was trained on {season} gameweeks that overlap "
                f"GW{start_gw}-{end_gw}. Scoring a model on its own training data "
                "measures memorisation. Train on an earlier season."
            )
        typer.echo(
            f"  model {trained.models.fingerprint()[:12]} "
            f"trained on {', '.join(trained.trained_seasons)}"
        )
    else:
        typer.echo("  closed-form baseline")

    opponent_strength = (
        build_opponent_strength(all_stats) if trained is not None else pl.DataFrame()
    )
    team_stats = (
        build_team_gameweek_stats(all_stats, players) if trained is None else pl.DataFrame()
    )

    # Minutes travel with the outcome so the report can separate players who
    # took the field from those whose zero was never in doubt.
    outcome_rows = (
        all_stats.filter(pl.col("season") == str(parsed))
        .group_by(["gameweek_id", "player_code"])
        .agg(
            pl.col("total_points").sum().alias("actual"),
            pl.col("minutes").sum().alias("minutes"),
        )
    )

    collected: list[pl.DataFrame] = []
    for gw in range(start_gw, end_gw + 1):
        gameweek = GameweekId(gw)
        cutoff = deadline_by_gw.get(gw)
        if cutoff is None or not actual_points(all_stats, season=parsed, gameweek=gameweek):
            continue

        if trained is None:
            entities = players.select(
                "player_code", "position", "team_id", "current_price", "web_name"
            ).with_columns(pl.lit(cutoff).alias("prediction_timestamp"))
            features = build_slice1_features(
                entities, player_stats=all_stats, team_stats=team_stats
            )
            predictions = predict_frame(
                features,
                rules=scoring,
                from_gameweek=gameweek,
                data_cutoff=cutoff,
                predicted_at=cutoff,
                run_id="score",
                code_version="cli",
                feature_set_version="slice1_v1",
            )
        else:
            from xg_alonso.features.assemble import build_model_features
            from xg_alonso.features.catalogue import CATALOGUE_VERSION
            from xg_alonso.prediction import predict_with_models

            fixture = (
                all_stats.filter((pl.col("season") == str(parsed)) & (pl.col("gameweek_id") == gw))
                .select("player_code", "opponent_team_id", "was_home")
                .unique(subset=["player_code"], keep="first", maintain_order=True)
            )
            entities = (
                players.select("player_code", "position", "team_id", "current_price", "web_name")
                .join(fixture, on="player_code", how="left")
                .with_columns(pl.lit(cutoff).alias("prediction_timestamp"))
            )
            features = build_model_features(
                entities, player_stats=all_stats, opponent_strength=opponent_strength
            )
            predictions = predict_with_models(
                features,
                models=trained.models,
                rules=scoring,
                from_gameweek=gameweek,
                data_cutoff=cutoff,
                predicted_at=cutoff,
                run_id="score-trained",
                code_version="cli",
                feature_set_version=CATALOGUE_VERSION,
            )

        predicted = pl.DataFrame(
            {
                "player_code": [int(p.player_code) for p in predictions],
                "predicted": [float(p.expected_points) for p in predictions],
                "position": [str(p.position.value) for p in predictions],
            }
        )
        gw_outcomes = outcome_rows.filter(pl.col("gameweek_id") == gw).drop("gameweek_id")

        # An inner join scores only players with both a prediction and a
        # recorded outcome. Players who left the league mid-season have neither.
        collected.append(
            predicted.join(gw_outcomes, on="player_code", how="inner")
            .join(
                players.select("player_code", pl.col("current_price").alias("price")),
                on="player_code",
                how="left",
            )
            .with_columns(pl.lit(gw).alias("gameweek_id"))
        )

    if not collected:
        raise typer.BadParameter(f"no scorable gameweeks in {season} GW{start_gw}-{end_gw}")

    frame = pl.concat(collected)
    typer.echo(
        f"  {frame.height:,} player-gameweeks across {frame['gameweek_id'].n_unique()} gameweeks\n"
    )

    report = score_predictions(frame)
    typer.echo(report.summary())

    if report.overall.degenerate:
        typer.secho(
            "\n  WARNING: expected points is constant across players. The optimizer "
            "cannot rank anybody on this output.",
            fg=typer.colors.RED,
            bold=True,
        )


def _positions_by_season(data_root: Path) -> pl.DataFrame:
    """Each player's position in each season, for slicing a measurement.

    A player can change position between seasons — FPL reclassifies wingers and
    wing-backs regularly — so this is keyed on season as well as player. Falling
    back to a season-agnostic map would file a 2024-25 midfielder's rows under
    whatever he is listed as today.
    """
    from xg_alonso.contracts.prediction import Position

    path = data_root / "silver" / "players_history.parquet"
    if not path.exists():
        return pl.DataFrame(
            schema={"player_code": pl.Int64, "season": pl.Utf8, "position": pl.Utf8}
        )
    frame = pl.read_parquet(path)
    if not {"player_code", "season", "position"}.issubset(frame.columns):
        return pl.DataFrame(
            schema={"player_code": pl.Int64, "season": pl.Utf8, "position": pl.Utf8}
        )
    # The archive spells goalkeeper `GK`; the contracts' `Position` enum — and
    # therefore every prediction — spells it `GKP`. Normalised here so the two
    # sides of the join agree. Without this the goalkeeper slice silently fails
    # to match any prediction, its label weights fall back to the pooled ones,
    # and the ranking it produces is cosmetic rather than positional.
    return frame.select(
        pl.col("player_code").cast(pl.Int64),
        pl.col("season").cast(pl.Utf8),
        pl.col("position").cast(pl.Utf8).replace({"GK": Position.GKP.value}).alias("position"),
    ).unique(subset=["player_code", "season"])


def _attach_position(rows: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame | None:
    """Join a position onto measurement rows, by season where one is available."""
    if "label_season" in rows.columns:
        return rows.join(
            positions.rename({"season": "label_season"}),
            on=["player_code", "label_season"],
            how="left",
        )
    latest = positions.sort("season").unique(subset=["player_code"], keep="last").drop("season")
    return rows.join(latest, on="player_code", how="left")


@app.command()
def importance(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    model_path: Annotated[
        Path | None, typer.Option("--model", help="Fitted model to measure.")
    ] = None,
    seasons: Annotated[
        str, typer.Option("--seasons", help="Comma-separated seasons to measure on.")
    ] = "2024-25",
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to write the importance table.")
    ] = None,
    repeats: Annotated[int, typer.Option("--repeats", help="Shuffles per feature.")] = 5,
    top: Annotated[int, typer.Option("--top", help="How many features to print.")] = 25,
    min_gameweek: Annotated[
        int, typer.Option("--min-gw", help="Skip opening gameweeks with empty windows.")
    ] = 4,
) -> None:
    """Measure which features actually earn their place.

    The catalogue declares 171 features and nothing measured whether any of them
    helped. This shuffles each one in turn on rows the model was **not** fitted
    on and records what happens to out-of-sample error — so a feature that only
    ever helped by memorising shows nothing here, which is the point.

    Read the family totals alongside the flat ranking. Correlated features
    substitute for one another, so each scores low individually while the family
    they belong to matters a great deal; a flat ranking alone would say the
    opposite.
    """
    from xg_alonso.contracts.provenance import utc_now
    from xg_alonso.evaluation.importance import (
        ALL_POSITIONS,
        ImportanceTable,
        label_weights_from_predictions,
        permutation_importance,
        write_importance,
    )
    from xg_alonso.features.catalogue import CATALOGUE_VERSION, catalogue_specs
    from xg_alonso.features.opponent import OPPONENT_FEATURES
    from xg_alonso.prediction import build_training_frame, load_models

    stats_path = data_root / "silver" / "player_gameweek_stats.parquet"
    if not stats_path.exists():
        raise typer.BadParameter("no backfill found. Run `xg backfill` first.")

    resolved_model = model_path or (data_root / "models" / "component_models.pkl")
    if not resolved_model.exists():
        raise typer.BadParameter(f"no model at {resolved_model}. Run `xg train` first.")

    saved = load_models(resolved_model)
    models = saved.models

    wanted = [s.strip() for s in seasons.split(",")]
    typer.echo(f"  building measurement frame from {', '.join(wanted)} ...")
    data = build_training_frame(
        pl.read_parquet(stats_path), seasons=wanted, min_gameweek=min_gameweek
    )
    typer.echo(f"    {data.rows:,} rows x {len(data.feature_columns)} features")

    # Measure on each walk-forward validation window in turn. Those are the only
    # rows the model genuinely did not train on — the refit at the end of
    # training saw everything, so measuring on the full frame would describe the
    # fit and quietly call it evidence.
    #
    # Every fold rather than just the last, because a single fold cannot show
    # whether a feature's importance holds up: rank stability across folds is
    # what separates a real effect from one lucky window.
    ordered = sorted(data.seasons)
    offset = {season: index * 100 for index, season in enumerate(ordered)}
    timeline = pl.col("label_season").replace_strict(offset, default=0) + pl.col("label_gameweek")

    windows: list[tuple[int, pl.DataFrame]] = []
    for fold in models.folds:
        rows = data.frame.filter(timeline.is_between(fold.validate_start, fold.validate_end))
        if not rows.is_empty():
            windows.append((fold.fold_index, rows))

    if not windows:
        typer.secho(
            "  WARNING: no validation window has rows for these seasons; falling back "
            "to the full frame, which overlaps training and is not out-of-sample.",
            fg=typer.colors.YELLOW,
        )
        windows = [(0, data.frame)]

    typer.echo(
        f"    measuring on {len(windows)} fold(s), "
        f"{sum(rows.height for _, rows in windows):,} held-out rows"
    )

    # Opponent features come from a separate generator, so a catalogue-only map
    # files them under "unknown" — which then reads as a family that does not
    # matter, when in fact it is the only fixture information the model has.
    families = {spec.name: spec.family for spec in catalogue_specs()}
    for name in OPPONENT_FEATURES:
        families[name] = "fixture" if name == "is_home" else "opponent"

    # Weight each label by how much it actually moves a points total, measured
    # from a predicted population rather than assumed from the scoring values.
    context = _load_context(data_root, parse_season(DEFAULT_SEASON))
    gameweek = context.next_gameweek()
    predictions = _predict_all(context, gameweek, models)
    weights = label_weights_from_predictions(predictions)

    typer.echo(
        f"  permuting {len(models.feature_columns)} features x "
        f"{len(models.models)} labels x {repeats} repeats x {len(windows)} fold(s) ..."
    )
    computed_at = utc_now()

    # Measured once pooled, then once per position.
    #
    # The pooled slice answers "does this feature earn its place in the
    # catalogue". The positional slices answer the question a manager actually
    # asks, which is different for every position: minutes drives a striker's
    # goals, but for a defender the same minutes matter through clean sheets,
    # and a feature that ranks third overall can rank first for goalkeepers and
    # last for forwards. Averaging those into one ranking describes none of them.
    #
    # A slice with too few rows is skipped rather than measured, because
    # permutation importance on a handful of rows is noise wearing a number's
    # clothes.
    min_rows_for_a_slice = 150
    slices: list[tuple[str, int, pl.DataFrame]] = [
        (ALL_POSITIONS, fold_index, rows) for fold_index, rows in windows
    ]

    # The training frame is keyed on `player_code` and carries no position — it
    # is a feature matrix, and position is an attribute of the player, not of
    # the row. Joined in here rather than added to the frame, so the model still
    # never sees it as an input.
    #
    # Read from `players_history`, not from the live bootstrap. The bootstrap
    # lists only this season's squads, so joining it against measurement rows
    # from an earlier season silently drops every player who has since left —
    # which showed up as two positions being measured instead of four.
    positions = _positions_by_season(data_root)

    for fold_index, rows in windows:
        if "player_code" not in rows.columns or positions.is_empty():
            continue
        labelled = _attach_position(rows, positions)
        if labelled is None:
            continue
        for position in sorted(labelled["position"].unique().drop_nulls().to_list()):
            subset = labelled.filter(pl.col("position") == position).drop("position")
            if subset.height >= min_rows_for_a_slice:
                slices.append((str(position), fold_index, subset))

    measured_positions = sorted({name for name, _, _ in slices if name != ALL_POSITIONS})
    if measured_positions:
        typer.echo(f"  positional slices: {', '.join(measured_positions)}")
    else:
        typer.echo("  no positional slice had enough rows; measuring pooled only")

    # Weights are sliced with the rows. Measuring defenders against the pooled
    # definition of what points are made of would flatten their ranking toward
    # appearance exactly as the global weighting does — the slice would be
    # cosmetic. A clean sheet is four points to a defender and zero to a forward,
    # and the ranking has to know that.
    weights_for: dict[str, dict[str, float]] = {ALL_POSITIONS: weights}
    for position in measured_positions:
        sliced = label_weights_from_predictions(predictions, position=position)
        weights_for[position] = sliced or weights

    tables = [
        permutation_importance(
            models,
            rows,
            label_columns=data.label_columns,
            families=families,
            label_weights=weights_for.get(position, weights),
            catalogue_version=CATALOGUE_VERSION,
            computed_at=computed_at,
            n_repeats=repeats,
            fold_index=fold_index,
            position=position,
        )
        for position, fold_index, rows in slices
    ]
    table = ImportanceTable(
        rows=tuple(row for measured in tables for row in measured.rows),
        catalogue_version=CATALOGUE_VERSION,
        model_fingerprint=models.fingerprint(),
        computed_at=computed_at,
        label_weights=weights,
    )

    destination = out or (data_root / "gold" / "feature_importance.parquet")
    write_importance(table, destination)

    ranked = sorted(table.by_feature().items(), key=lambda kv: -kv[1])
    stability = table.stability()

    typer.echo(f"\n  Top {min(top, len(ranked))} features, weighted across components:")
    if stability:
        typer.echo(f"    {'feature':<38}{'importance':>12}{'rank sd':>10}")
        for name, value in ranked[:top]:
            typer.echo(f"    {name:<38}{value:>12.5f}{stability.get(name, 0.0):>10.1f}")
    else:
        # Only one fold had rows, so there is nothing to compare a rank against.
        # Printing a zero here would read as "perfectly stable".
        typer.echo(f"    {'feature':<38}{'importance':>12}")
        for name, value in ranked[:top]:
            typer.echo(f"    {name:<38}{value:>12.5f}")
        typer.echo("    (rank stability needs at least two folds with rows)")

    typer.echo("\n  By family (correlated features split, so read these too):")
    for family, value in sorted(table.by_family().items(), key=lambda kv: -kv[1]):
        typer.echo(f"    {family:<38}{value:>12.5f}")

    dead = [name for name, value in ranked if value <= 0.0]
    if dead:
        typer.echo(
            f"\n  {len(dead)} of {len(ranked)} features did not improve out-of-sample error."
        )

    degenerate = table.degenerate_labels()
    if degenerate:
        typer.secho(
            f"\n  NOTE: {len(degenerate)} label(s) output a constant and are excluded "
            f"from the ranking: {', '.join(degenerate)}.\n"
            "  Ranking features by their effect on a constant would be arithmetic "
            "without meaning.",
            fg=typer.colors.YELLOW,
        )

    typer.echo(f"\n  saved -> {destination}")


@app.command(name="refresh-plan")
def refresh_plan_command(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    entry_id: Annotated[
        int | None, typer.Option("--entry", help="Weight your own players more heavily.")
    ] = None,
    squad_file: Annotated[
        Path | None, typer.Option("--squad-file", help="Squad payload instead of the API.")
    ] = None,
    budget: Annotated[int, typer.Option("--budget", help="Maximum lookups.")] = 20,
    model_path: Annotated[
        Path | None, typer.Option("--model", help="Use a fitted model for relevance.")
    ] = None,
) -> None:
    """Show which clubs are worth looking up for form, and which are not.

    The naive alternative is one lookup per player per gameweek — about 600 a
    week, most of them about players nobody will own. This asks per club
    instead, because team news is published that way, and skips anyone whose
    form cannot change a decision: too low-scoring to start, too expensive to
    buy, or already covered by a signal that has not gone stale.

    Prints the plan without fetching anything. What fetches the answers lives
    outside this repository, which is what keeps the prediction path free of a
    network dependency.
    """
    from xg_alonso.contracts.identifiers import TeamId
    from xg_alonso.prediction.form import load_signals
    from xg_alonso.prediction.refresh import plan_refresh

    context = _load_context(data_root, parse_season(season))
    gameweek = context.next_gameweek()
    cutoff = context.deadline_for(gameweek)

    models = None
    if model_path is not None and model_path.exists():
        from xg_alonso.prediction import load_models

        models = load_models(model_path).models

    predictions = _predict_all(context, gameweek, models)
    rows = {int(r["player_code"]): r for r in context.players.iter_rows(named=True)}

    teams = {PlayerCode(code): TeamId(int(row["team_id"])) for code, row in rows.items()}
    prices = {
        PlayerCode(code): TenthsOfMillion(int(row["current_price"])) for code, row in rows.items()
    }
    team_names = {int(r["team_id"]): str(r["name"]) for r in context.teams.iter_rows(named=True)}

    owned: frozenset[PlayerCode] = frozenset()
    affordable: TenthsOfMillion | None = None
    if entry_id is not None or squad_file is not None:
        state = _squad_state(context, entry_id or 0, squad_file)
        owned = frozenset(pick.player_code for pick in state.picks)
        affordable = TenthsOfMillion(
            max(int(pick.selling_price) for pick in state.picks) + int(state.bank)
        )

    plan = plan_refresh(
        predictions,
        teams=teams,
        prices=prices,
        signals=load_signals(data_root / "signals" / "form_signals.json"),
        at=cutoff,
        owned=owned,
        affordable_at=affordable,
        budget=budget,
    )

    typer.echo(f"\n  {plan.describe()}\n")
    if not plan.requests:
        typer.echo("  Nothing worth looking up: every relevant player already has a")
        typer.echo("  signal that has not gone stale.")
        return

    typer.echo(f"    {'club':<24}{'players':>9}   why")
    for request in plan.requests:
        club = team_names.get(int(request.team_id), f"team {int(request.team_id)}")
        typer.echo(f"    {club:<24}{len(request.players):>9}   {request.reason}")

    naive = plan.total_players
    if naive > plan.lookups:
        typer.echo(
            f"\n  {plan.lookups} lookups instead of {naive} "
            f"({naive / max(plan.lookups, 1):.0f}x fewer)."
        )


# ---------------------------------------------------------------------------
# Objective-conditioned feature discovery
# ---------------------------------------------------------------------------


@app.command(name="discover")
def discover_command(
    request: Annotated[
        str,
        typer.Argument(
            help=(
                "What you want, in plain English. e.g. 'I am 40 points behind in my "
                "mini-league, keep Haaland, aggressive three-gameweek strategy, keep xG "
                "and find complementary signals'."
            )
        ),
    ],
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    frame_path: Annotated[
        Path | None,
        typer.Option(
            "--frame",
            help="Training frame from `xg build-discovery-frame`. Defaults to the gold copy.",
        ),
    ] = None,
    preset: Annotated[
        str,
        typer.Option("--preset", help="Objective preset to start from before the text adjusts it."),
    ] = "expected_points",
    max_hypotheses: Annotated[
        int, typer.Option("--max-hypotheses", help="Proposals to test this run.")
    ] = 6,
    holdout: Annotated[
        str,
        typer.Option("--holdout", help="Seasons excluded from every fold. Never optimised on."),
    ] = "2025-26",
    clusters: Annotated[
        int, typer.Option("--clusters", help="Objective-conditioned clusters.")
    ] = 5,
    no_controls: Annotated[
        bool,
        typer.Option("--no-controls", help="Skip the noise and shuffled controls (much faster)."),
    ] = False,
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm",
            help=(
                "Also ask a language model for hypotheses. Needs ANTHROPIC_API_KEY in the "
                "environment or in .env. Its proposals pass exactly the same parse, "
                "validation and leakage gates as every other."
            ),
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the parsed intent and stop.")
    ] = False,
) -> None:
    """Compile a request, discover features that serve it, and report the verdicts.

    Prints the parsed intent **before** running anything. Natural language is
    ambiguous and a system that acts on its own guess without showing it will
    eventually act on a wrong one silently, so the structured interpretation and
    everything the parser did *not* understand are surfaced first.
    """
    from xg_alonso.contracts.discovery import ExperimentStage
    from xg_alonso.discovery.experiment import ExperimentConfig, run_discovery
    from xg_alonso.discovery.harness import HarnessConfig
    from xg_alonso.discovery.registry import DiscoveryRegistry
    from xg_alonso.domain.intent import build_name_index, compile_intent
    from xg_alonso.features.generators import stage_window

    frame_file = frame_path or (data_root / "gold" / "discovery_training.parquet")
    if not frame_file.exists():
        typer.secho(
            f"No training frame at {frame_file}. Run `xg build-discovery-frame` first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    frame = pl.read_parquet(frame_file)
    stats = pl.read_parquet(data_root / "silver" / "player_gameweek_stats.parquet")
    history = pl.read_parquet(data_root / "silver" / "players_history.parquet")
    # Indexed by full name *and* by unambiguous surname: the stored name is
    # "Erling Haaland" and a manager types "Haaland".
    names = build_name_index(
        dict(
            zip(
                history["player_code"].to_list(),
                history["web_name"].to_list(),
                strict=True,
            )
        )
    )

    intent = compile_intent(request, players=names, base_preset=preset)
    bundle = intent.bundle
    objective = bundle.objective

    typer.secho("\nParsed intent", bold=True)
    typer.echo(f"  objective        {objective.id}")
    typer.echo(f"  primary metric   {objective.primary_metric.value}")
    typer.echo(
        f"  risk             {objective.risk_preference.value} "
        f"(variance term {objective.signed_uncertainty_penalty:+.2f})"
    )
    typer.echo(f"  horizon          {objective.planning_horizon} gameweek(s)")
    typer.echo(f"  ownership        {objective.ownership_preference.value}")
    if bundle.constraints.locked_players:
        held = ", ".join(str(int(c)) for c in bundle.constraints.locked_players)
        typer.echo(f"  locked players   {held}")
    if bundle.constraints.locked_position_set():
        frozen = ", ".join(sorted(p.value for p in bundle.constraints.locked_position_set()))
        typer.echo(f"  frozen positions {frozen}")
    typer.echo(f"  max points hit   {bundle.constraints.max_points_hit}")
    typer.echo(f"  required features {list(bundle.constraints.required_features) or '(none)'}")
    for belief in bundle.beliefs:
        typer.echo(
            f"  belief           player {belief.entity_id} "
            f"{belief.proposition.value} @ {belief.confidence:.0%}"
        )
    typer.echo(f"  confidence       {intent.overall_confidence:.0%}")
    if intent.unparsed:
        typer.secho("  not understood:", fg=typer.colors.YELLOW)
        for clause in intent.unparsed:
            typer.secho(f"    - {clause}", fg=typer.colors.YELLOW)

    if dry_run:
        typer.echo("\n(dry run — nothing was executed)")
        return

    # Parquet rather than DuckDB: the composition root may not import a database
    # driver (`.importlinter`), and the registry reads whole tables and filters
    # in Polars, so nothing here wants SQL.
    store = ParquetTableStore(data_root / "discovery")
    registry = DiscoveryRegistry(store)

    def on_stage(stage: ExperimentStage, detail: str) -> None:
        typer.echo(f"  [{stage.value:<19}] {detail}")

    typer.secho("\nRunning", bold=True)
    try:
        result = run_discovery(
            bundle=bundle,
            training=frame,
            player_stats=stats,
            stage_window=stage_window,
            registry=registry,
            config=ExperimentConfig(
                harness=HarnessConfig(
                    holdout_seasons=tuple(s.strip() for s in holdout.split(",") if s.strip()),
                    max_folds=5,
                ),
                max_hypotheses=max_hypotheses,
                cluster_k=clusters,
                run_controls=not no_controls,
                use_llm=use_llm,
            ),
            on_stage=on_stage,
        )
    finally:
        pass

    typer.secho("\n" + result.summary(), bold=True)

    if result.weak_segments:
        typer.secho("\nWhere the required feature set is weakest", bold=True)
        for kind, segment, error in result.weak_segments:
            typer.echo(f"  {kind:<10} {segment:<8} relative error {error:.3f}")

    if result.rejected_programs:
        typer.secho("\nRefused before computation", fg=typer.colors.YELLOW)
        for name, issues in result.rejected_programs:
            typer.echo(f"  {name}: {'; '.join(str(i) for i in issues)[:120]}")

    report = registry.acceptance_report(objective.id)
    if report.height:
        typer.secho("\nVerdicts", bold=True)
        for row in report.iter_rows(named=True):
            colour = (
                typer.colors.GREEN
                if row["status"].startswith("accepted")
                else typer.colors.YELLOW
                if row["status"] in {"revise", "insufficient_data"}
                else typer.colors.RED
            )
            typer.secho(
                f"  {row['status']:<26} {row['feature']:<34} "
                f"utility {row['utility']:+.4f}  incremental {row['incremental_value']:+.5f}  "
                f"folds {row['folds_improved']}/{row['folds']}  stability {row['stability']:.2f}",
                fg=colour,
            )
            typer.echo(f"      {row['complementarity']} — {row['reason'][:110]}")

    if result.cluster_summaries:
        typer.secho("\nObjective-conditioned clusters", bold=True)
        for summary in result.cluster_summaries:
            typer.echo(f"  c{summary.cluster_id} n={summary.size:<6} {summary.label}")

    if result.search is not None:
        search = result.search
        typer.secho("\nComplementary search", bold=True)
        typer.echo(
            f"  baseline MAE {search.baseline.metric:.4f} -> {search.final.metric:.4f} "
            f"({search.total_gain:+.2%}) over {search.evaluations} evaluations"
        )
        for step in search.steps:
            typer.echo(f"    + {step.describe()}")
        if search.rejected:
            passed = ", ".join(f"{n} ({g:+.2%})" for n, g in search.rejected[:5])
            typer.echo(f"    passed over: {passed}")
        if search.truncated:
            typer.secho(
                "    (a cap stopped the search — this is not proof nothing more exists)",
                fg=typer.colors.YELLOW,
            )

    if result.lessons:
        typer.secho("\nLessons recorded", bold=True)
        for lesson in result.lessons:
            typer.echo(f"  [{lesson.hypothesis_family}] {lesson.result}")
            if lesson.promising_direction:
                typer.echo(f"      -> {lesson.promising_direction}")

    manifest_path = data_root / "reports" / f"{result.manifest.experiment_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(result.manifest.model_dump_json(indent=2))
    typer.echo(f"\nManifest written to {manifest_path}")
    if not result.manifest.reproducible:
        typer.secho(
            "This run is NOT reproducible: the working tree is dirty or the commit is "
            "unknown, so the recorded code version does not describe what ran.",
            fg=typer.colors.YELLOW,
        )


@app.command(name="build-discovery-frame")
def build_discovery_frame_command(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    seasons: Annotated[
        str, typer.Option("--seasons", help="Comma-separated seasons to build.")
    ] = "2023-24,2024-25",
    min_gameweek: Annotated[
        int, typer.Option("--min-gw", help="Skip the opening gameweeks of each season.")
    ] = 2,
) -> None:
    """Build the point-in-time training frame the discovery loop runs on.

    Separate from `xg discover` because it is the expensive step — the feature
    window is recomputed per gameweek rather than once over the whole frame,
    which is the price of a dataset that is not silently poisoned.
    """
    from xg_alonso.prediction.dataset import build_training_frame

    wanted = [s.strip() for s in seasons.split(",") if s.strip()]
    stats = pl.read_parquet(data_root / "silver" / "player_gameweek_stats.parquet")
    history = pl.read_parquet(data_root / "silver" / "players_history.parquet")

    typer.echo(f"Building features as of every deadline in {', '.join(wanted)}...")
    data = build_training_frame(stats, seasons=wanted, min_gameweek=min_gameweek)
    frame = data.frame

    # The component labels carry no points total by design (D8: models predict
    # components and the domain prices them). A discovered feature is judged on
    # the number a manager reads, so attach it here.
    totals = (
        stats.filter(pl.col("season").is_in(wanted))
        .group_by(["player_code", "season", "gameweek_id"])
        .agg(pl.col("total_points").sum().alias("label_total_points"))
    )
    frame = frame.join(
        totals,
        left_on=["player_code", "label_season", "label_gameweek"],
        right_on=["player_code", "season", "gameweek_id"],
        how="left",
    )

    meta = history.select("player_code", "season", "position", "team_name").unique(
        subset=["player_code", "season"]
    )
    frame = frame.join(
        meta,
        left_on=["player_code", "label_season"],
        right_on=["player_code", "season"],
        how="left",
    ).filter(pl.col("label_total_points").is_not_null() & pl.col("position").is_not_null())

    destination = data_root / "gold" / "discovery_training.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination)
    typer.secho(
        f"Wrote {frame.height:,} rows x {frame.width} columns to {destination}",
        fg=typer.colors.GREEN,
    )


@app.command(name="advise")
def advise_command(
    request: Annotated[str, typer.Argument(help="What you want, in plain English.")],
    entry_id: Annotated[int, typer.Option("--entry", help="Public FPL entry id.")] = 0,
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
    squad_file: Annotated[
        Path | None,
        typer.Option("--squad-file", help="Read the squad from a picks JSON file."),
    ] = None,
    model_path: Annotated[
        Path | None, typer.Option("--model", help="Use trained component models.")
    ] = None,
    preset: Annotated[str, typer.Option("--preset", help="Objective to start from.")] = (
        "expected_points"
    ),
) -> None:
    """Recommend a transfer under your objective, constraints and beliefs.

    The difference from `xg recommend`: that command answers "which transfer
    gains the most expected points". This one answers "which transfer best
    serves *what you are trying to do*" — and shows what your own instructions
    cost you, separately from what the model thinks.
    """
    from xg_alonso.cli.pipeline import recommend_for_objective
    from xg_alonso.domain.intent import build_name_index, compile_intent
    from xg_alonso.prediction.inference import load_models

    context = _load_context(data_root, parse_season(season))
    state = _squad_state(context, entry_id, squad_file)

    names = build_name_index(
        {int(r["player_code"]): str(r["web_name"]) for r in context.players.iter_rows(named=True)}
    )
    intent = compile_intent(
        request,
        players=names,
        base_preset=preset,
        next_gameweek=int(state.gameweek),
        current_squad=[int(p.player_code) for p in state.picks],
    )
    bundle = intent.bundle

    typer.secho("Your request, as understood", bold=True)
    typer.echo(f"  objective        {bundle.objective.id}")
    typer.echo(
        f"  risk             {bundle.objective.risk_preference.value} "
        f"(variance term {bundle.objective.signed_uncertainty_penalty:+.2f})"
    )
    typer.echo(f"  horizon          {bundle.objective.planning_horizon} gameweek(s)")
    typer.echo(f"  ownership        {bundle.objective.ownership_preference.value}")
    typer.echo(f"  confidence       {intent.overall_confidence:.0%}")
    for clause in intent.unparsed:
        typer.secho(f"  not understood:  {clause}", fg=typer.colors.YELLOW)

    models = None
    if model_path is not None:
        models = load_models(model_path).models

    run_id = f"advise-{uuid.uuid4().hex[:12]}"
    manifest = git_manifest("advise", run_id=run_id)

    result = recommend_for_objective(
        context=context,
        squad=state,
        bundle=bundle,
        entry_id=EntryId(entry_id),
        run_id=run_id,
        code_version=manifest.git_commit,
        generated_at=utc_now(),
        models=models,
    )

    for problem in result.constraint_problems:
        typer.secho(f"  ! {problem}", fg=typer.colors.RED)

    names_by_code = context.player_names()
    prices = {
        PlayerCode(int(r["player_code"])): int(r["current_price"])
        for r in context.players.iter_rows(named=True)
    }

    typer.secho("\nRecommendation", bold=True)
    typer.echo(render_recommendation(result.raw, names=names_by_code, prices=prices))

    if result.constraint_report.is_binding:
        typer.secho("\nWhat your instructions changed", bold=True)
        for line in result.constraint_report.explain():
            typer.echo(f"  {line}")

    if result.opportunity_costs:
        typer.secho("\nWhat holding them cost", bold=True)
        for held, alternative, forgone in result.opportunity_costs:
            target = names_by_code.get(alternative, "?") if alternative else "nobody"
            typer.echo(
                f"  keeping {names_by_code.get(held, held)} gave up {forgone:.2f} "
                f"projected points (best alternative: {target})"
            )
    elif result.constraint_report.locked_players:
        typer.echo(
            "\n  Holding your locked players cost nothing: no affordable alternative scored higher."
        )

    if result.adjusted is not None:
        typer.secho("\nWith your beliefs applied", bold=True)
        for adjustment in result.belief_adjustments:
            typer.echo(
                f"  {names_by_code.get(adjustment.player_code, '?')}: {adjustment.explain()}"
            )
        if result.beliefs_changed_the_answer:
            typer.secho("  Your beliefs changed the recommendation:", fg=typer.colors.YELLOW)
            typer.echo(render_recommendation(result.adjusted, names=names_by_code, prices=prices))
        else:
            typer.echo(
                "  Your beliefs did not change the recommendation — they were "
                "weighed and the move stands either way."
            )
        typer.secho(
            "  Shown separately on purpose: a belief is your judgement, not evidence.",
            fg=typer.colors.BLUE,
        )
