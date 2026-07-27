"""The ``xg`` command-line interface.

Per decision D4 the CLI is the first surface, ahead of the HTTP API and the web
app. That ordering is deliberate: a terminal command forces the recommendation
to stand on its own content rather than on presentation, and a recommendation
that is not convincing as plain text will not become convincing in a card.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from xg_alonso.cli.pipeline import (
    PRICES_WERE_ASSUMED,
    SliceContext,
    build_context,
    load_squad_file,
    recommend,
    squad_from_payload,
)
from xg_alonso.contracts.identifiers import EntryId, PlayerCode, Season, parse_season
from xg_alonso.contracts.provenance import utc_now
from xg_alonso.contracts.squad import SquadPick
from xg_alonso.explanations.render import render_recommendation, render_squad_summary
from xg_alonso.pipelines.ingestion import (
    SOURCE_BOOTSTRAP,
    SOURCE_FIXTURES,
    FplApiClient,
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
from xg_alonso.storage import FileSystemBronzeStore

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


@app.command()
def ingest(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
) -> None:
    """Fetch official FPL data into immutable bronze snapshots."""
    parsed = parse_season(season)
    run_id = f"ingest-{uuid.uuid4().hex[:12]}"
    bronze = _bronze(data_root)

    with FplApiClient() as client:
        result = ingest_bootstrap(client=client, bronze=bronze, season=parsed, run_id=run_id)

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


@app.command(name="build-features")
def build_features_command(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
    season: SeasonOpt = DEFAULT_SEASON,
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

    out_dir = data_root / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"features_gw{gameweek}.parquet"
    features.write_parquet(destination)

    typer.echo(f"GW{gameweek}  cutoff {cutoff.isoformat()}")
    typer.echo(f"  {features.height:,} rows x {len(SLICE1_FEATURES)} features -> {destination}")
    for name in SLICE1_FEATURES:
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
) -> None:
    """Show a squad with projected points per player."""
    context = _load_context(data_root, parse_season(season))
    recommendation, predictions = _run(context, entry_id, squad_file)
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
    gameweek = context.next_gameweek()
    if squad_file is None:
        raise typer.BadParameter(
            "--squad-file is required for now. The public picks endpoint returns 404 "
            "before a gameweek deadline, so there is no squad to fetch until the "
            "season is under way."
        )
    payload = load_squad_file(squad_file)
    return squad_from_payload(
        payload, context=context, entry_id=EntryId(entry_id), gameweek=gameweek
    )


def _run(context: SliceContext, entry_id: int, squad_file: Path | None):  # type: ignore[no-untyped-def]
    state = _squad_state(context, entry_id, squad_file)
    manifest = git_manifest("recommend", run_id=f"rec-{uuid.uuid4().hex[:12]}")
    return recommend(
        context=context,
        squad=state,
        entry_id=EntryId(entry_id),
        run_id=manifest.run_id,
        code_version=manifest.git_commit,
        generated_at=utc_now(),
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
) -> None:
    """Recommend the best legal single transfer, or advise holding."""
    context = _load_context(data_root, parse_season(season))
    recommendation, _ = _run(context, entry_id, squad_file)

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
    stats.write_parquet(destination)
    typer.echo(f"  {stats.height:,} history rows -> {destination}")


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
    from xg_alonso.pipelines.normalization import normalize_archive_season

    wanted = [s.strip() for s in seasons.split(",")] if seasons else list(BACKFILL_SEASONS)
    run_id = f"backfill-{uuid.uuid4().hex[:12]}"
    bronze = _bronze(data_root)

    frames: list[pl.DataFrame] = []
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

        defensive = "with DC" if result.has_defensive_contributions else "no DC"
        typer.echo(
            f" {result.rows_out:>6,} rows ({defensive})"
            f"  dropped: {result.rows_dropped_unresolved_element} unresolved,"
            f" {result.rows_dropped_bad_kickoff} bad kickoff"
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
