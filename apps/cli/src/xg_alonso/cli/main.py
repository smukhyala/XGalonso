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
from typing import Annotated, Any

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
from xg_alonso.contracts.identifiers import (
    EntryId,
    GameweekId,
    PlayerCode,
    Season,
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
        from xg_alonso.features.catalogue import build_catalogue, feature_names

        features = build_catalogue(features, player_stats=context.player_stats)
        reported = tuple(SLICE1_FEATURES) + tuple(feature_names())

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
        actual_points,
        gameweek_deadlines,
        run_policy,
        walk_forward,
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

    squad_rules = _load_context(data_root, parse_season(DEFAULT_SEASON)).squad_rules
    scoring = _load_context(data_root, parse_season(DEFAULT_SEASON)).scoring

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
        by_code = {p.player_code: p for p in predictions}
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
    trained_cache: dict[int, Any] = {}

    def _trained_inputs_for(gameweek: GameweekId) -> tuple[Any, Any, Any]:
        """The same entities and cutoff, but predicted by the fitted model.

        Sharing the entity frame and the legality check means a comparison
        between the closed-form and trained policies isolates prediction
        quality — nothing else differs.
        """
        cached = trained_cache.get(int(gameweek))
        if cached is not None:
            result: tuple[Any, Any, Any] = cached
            return result

        assert trained is not None
        from xg_alonso.features.catalogue import CATALOGUE_VERSION, build_catalogue
        from xg_alonso.features.opponent import build_opponent_features
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
        features = build_catalogue(entities, player_stats=all_stats)
        features = build_opponent_features(features, opponent_strength=opponent_strength)

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
        )
        by_code = {p.player_code: p for p in predictions}
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
        trained_cache[int(gameweek)] = computed
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
            source = _trained_inputs_for if _name == "trained" else _inputs_for
            by_code, candidates, cutoff = source(gameweek)
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

    headline = "trained" if "trained" in results else "model"
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
                is_captain=slot == 1,
                is_vice_captain=slot == 2,
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


@app.command()
def train(
    data_root: DataRoot = DEFAULT_DATA_ROOT,
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

    typer.echo("  fitting component models ...")
    models = train_component_models(
        data.frame,
        feature_columns=data.feature_columns,
        label_columns=data.label_columns,
    )

    saved = SavedModel(
        models=models,
        trained_seasons=data.seasons,
        trained_gameweeks=data.gameweeks,
        saved_at=utc_now(),
    )
    destination = out or (data_root / "models" / "component_models.pkl")
    save_models(saved, destination)

    summary = model_summary(saved)
    typer.echo(
        f"\n  {len(summary['labels'])} models, {summary['folds']} walk-forward folds"
        f", fingerprint {summary['fingerprint']}"
    )
    typer.echo("\n  Out-of-sample skill vs predicting the mean:")
    for label, skill in sorted(summary["skill"].items(), key=lambda kv: -kv[1]):
        flag = "" if skill > 0.02 else "   <- no better than a constant"
        typer.echo(f"    {label:<26}{skill:>+8.1%}{flag}")
    typer.echo(f"\n  saved -> {destination}")
