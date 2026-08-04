"""Rebuild the committed demo fixtures under ``data/fixtures`` from a local ``.data``.

Why both the generator *and* its output are committed
-----------------------------------------------------

A stranger who clones this repository has no ``.data`` — the whole store is
gitignored, and per D6 the only way to refill it is the official FPL API, which
needs network access and several minutes. So ``xg demo`` reads committed
fixtures instead. That makes the fixtures a *build artifact living in git*,
which is normally a smell, so the smell is paid for the only way it can be: the
program that produced them is committed beside them, and a provenance sidecar
records exactly which local bytes it read, when, and with which seed. The
fixtures are therefore auditable and reproducible without being a build step.

What is sampled, and why it is stratified
-----------------------------------------

``.data/silver/player_gameweek_stats.parquet`` is 2.9 MB — an order of
magnitude over the 256 KB per-file ceiling enforced by
``check-added-large-files`` in ``.pre-commit-config.yaml``. A uniform random
sample of players would fit but would be useless: filling a legal 15 needs
2 GKP / 5 DEF / 5 MID / 3 FWD under a maximum of three players per club and a
1000-tenths budget, and a uniform sample concentrates on whichever position has
the largest population. So the sample is stratified by club **and** position,
taking the highest-minutes players in each cell. Highest-minutes rather than
random because a fringe player with four appearances carries no usable window,
and because the regulars are the ones whose prices span the real range.

The sample is real FPL data, not synthetic. Only real price, position and club
spread can fill a legal 15, and FPL's data is public, so deriving from it is
legitimate. Nothing here invents a number.

Usage::

    uv run python tools/build_demo_fixture.py            # from .data into data/fixtures
    uv run python tools/build_demo_fixture.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Bumped whenever the sampling rule changes, so a stale fixture is identifiable
#: from its sidecar alone rather than by re-deriving it.
GENERATOR_VERSION: Final[str] = "1"

#: The two seasons the demo trains and discovers over. ``xg train`` and
#: ``xg build-discovery-frame`` both need at least two so a walk-forward fold
#: and a holdout are distinguishable rather than nominal.
FIXTURE_SEASONS: Final[tuple[str, ...]] = ("2023-24", "2024-25")

#: Players kept per club per position per season.
#:
#: The arithmetic, because the ceiling is what sets it: the source averages
#: ~34.5 gameweek rows per player-season and compresses to ~27 bytes a row at
#: zstd-22, so the budget is roughly 256 KB / 27 / 34.5 ≈ 270 player-seasons.
#: Five per club over twenty clubs and two seasons is 200, plus the supplement
#: 212 — 7,998 rows, 218 KB, about 17% under the ceiling. Six per club measured
#: 242 KB, which fits but leaves nothing for a future parquet writer that
#: compresses a few percent worse, so five is the shape that stays safe.
#:
#: Two midfielders rather than two defenders because MID is both the largest
#: real FPL population and the position the xG features are most about.
CLUB_QUOTA: Final[Mapping[str, int]] = {"GK": 1, "DEF": 1, "MID": 2, "FWD": 1}

#: Extra player-seasons drawn at random per season, outside the quota. They
#: widen the price and ownership spread that the quota's minutes-ranking would
#: otherwise flatten, and they are the only place the seed is consumed.
RANDOM_SUPPLEMENT: Final[int] = 6

#: Hard ceiling per committed fixture file, matching
#: ``.pre-commit-config.yaml``'s ``check-added-large-files --maxkb=256``.
MAX_FIXTURE_BYTES: Final[int] = 256 * 1024

#: zstd at its maximum level. Fixtures are written once and read forever, so
#: compression time is irrelevant and every byte of headroom is worth having.
_COMPRESSION_LEVEL: Final[int] = 22

_BOOTSTRAP_SOURCE: Final[str] = "fpl.bootstrap_static"
_FIXTURES_SOURCE: Final[str] = "fpl.fixtures"

PROVENANCE_NAME: Final[str] = "PROVENANCE.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _portable(path: Path) -> str:
    """A path fit to commit: relative to the repository, never someone's home.

    ``/Users/<name>/projects/...`` in a checked-in sidecar is both noise in
    review and a small privacy leak, and it is meaningless on the machine that
    reads it back.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _echo(message: str) -> None:
    sys.stdout.write(message + "\n")


def silver_schema_digest() -> str:
    """A digest of the silver contract the gameweek fixture was cut from.

    Ordered, and over dtypes as well as names: a column re-typed from ``Int64``
    to ``Float64`` keeps every name intact while changing what the pipeline
    computes, and that is exactly the change a fixture must not survive
    unnoticed.
    """
    from xg_alonso.pipelines.normalization import PLAYER_GAMEWEEK_STATS_SCHEMA

    payload = json.dumps(
        [[name, str(dtype)] for name, dtype in PLAYER_GAMEWEEK_STATS_SCHEMA.items()],
        separators=(",", ":"),
    )
    return _sha256(payload.encode("utf-8"))


def _catalogue_hash() -> str:
    """The feature catalogue's definition digest at the moment of extraction."""
    from xg_alonso.features.schema import catalogue_hash

    return str(catalogue_hash())


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def select_player_seasons(
    stats: pl.DataFrame,
    history: pl.DataFrame,
    *,
    seasons: Sequence[str],
    quota: Mapping[str, int],
    supplement: int,
    seed: int,
) -> pl.DataFrame:
    """Choose which ``(player_code, season)`` pairs the fixture keeps.

    Args:
        stats: The full silver gameweek table.
        history: The full silver per-season player table, for position and club.
        seasons: Seasons to sample. Every other season is dropped entirely.
        quota: Players kept per club per position, keyed by the ``position``
            values that ``players_history`` uses (``GK``/``DEF``/``MID``/``FWD``).
        supplement: Extra player-seasons drawn at random per season.
        seed: Consumed only by ``supplement``. The quota selection is a pure
            ranking, so the bulk of the fixture is reproducible without an RNG.

    Returns:
        A two-column frame of ``player_code`` and ``season``, sorted.
    """
    from xg_alonso.contracts.seeds import derive_seed

    minutes = stats.group_by(["player_code", "season"]).agg(
        pl.col("minutes").sum().alias("total_minutes")
    )
    meta = history.join(minutes, on=["player_code", "season"], how="left").with_columns(
        pl.col("total_minutes").fill_null(0)
    )

    chosen: list[pl.DataFrame] = []
    for season in seasons:
        for position, keep in quota.items():
            cell = (
                meta.filter((pl.col("season") == season) & (pl.col("position") == position))
                # player_code ascending is the tie-break, so two players with
                # identical minutes never swap between runs.
                .sort(
                    ["team_name", "total_minutes", "player_code"],
                    descending=[False, True, False],
                )
                .group_by("team_name", maintain_order=True)
                .head(keep)
            )
            chosen.append(cell.select("player_code", "season"))

    selected = pl.concat(chosen).unique()

    if supplement:
        for season in seasons:
            pool = (
                meta.filter(
                    (pl.col("season") == season)
                    # Only players with a real season behind them; a two-cameo
                    # player widens nothing and costs the same rows.
                    & (pl.col("total_minutes") >= 450)
                )
                .join(selected, on=["player_code", "season"], how="anti")
                .sort("player_code")
            )
            codes = pool["player_code"].to_list()
            if not codes:
                continue
            # derive_seed rather than `seed ^ hash(season)`: builtin hash is
            # salted per process, so the second form would draw a different
            # sample on every run and quietly break reproducibility.
            rng = random.Random(derive_seed(seed, "supplement", season))
            drawn = rng.sample(codes, k=min(supplement, len(codes)))
            selected = pl.concat(
                [
                    selected,
                    pl.DataFrame(
                        {"player_code": sorted(drawn), "season": [season] * len(drawn)},
                        schema={"player_code": pl.Int64, "season": pl.String},
                    ),
                ]
            ).unique()

    return selected.sort(["season", "player_code"])


def sample_gameweek_stats(
    stats: pl.DataFrame,
    selected: pl.DataFrame,
) -> pl.DataFrame:
    """Restrict the gameweek table to the chosen player-seasons."""
    return stats.join(selected, on=["player_code", "season"], how="inner").sort(
        ["season", "player_code", "gameweek_id"]
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_parquet(frame: pl.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # ``statistics=False`` because per-row-group min/max on 37 columns is pure
    # overhead for a file this small, and it is several kilobytes of it.
    frame.write_parquet(
        destination,
        compression="zstd",
        compression_level=_COMPRESSION_LEVEL,
        statistics=False,
    )


def _latest_bronze_payload(data_root: Path, source: str) -> tuple[bytes, dict[str, Any]] | None:
    """The newest raw payload for ``source``, with the timestamps it was seen under.

    Read through the real bronze store so the hash check on the way out of
    bronze is exercised here too: a fixture derived from a corrupted snapshot
    would be an excellent way to ship a silent lie.
    """
    from xg_alonso.storage import FileSystemBronzeStore

    store = FileSystemBronzeStore(data_root / "bronze")
    ref = store.latest(source)
    if ref is None:
        return None
    payload = store.read(ref)
    return payload, {
        "source": ref.source,
        "content_sha256": ref.content_sha256,
        "byte_size": ref.byte_size,
        "http_status": ref.http_status,
        "run_id": ref.run_id,
        "timestamps": json.loads(ref.timestamps.model_dump_json()),
    }


def build(
    *,
    source_root: Path,
    out_root: Path,
    seasons: Sequence[str] = FIXTURE_SEASONS,
    quota: Mapping[str, int] = CLUB_QUOTA,
    supplement: int = RANDOM_SUPPLEMENT,
    seed: int | None = None,
) -> dict[str, Any]:
    """Regenerate every fixture, returning the provenance record that was written."""
    from xg_alonso.contracts.seeds import ROOT_SEED, derive_seed

    resolved_seed = seed if seed is not None else derive_seed(ROOT_SEED, "fixtures", "demo", "v1")

    stats_source = source_root / "silver" / "player_gameweek_stats.parquet"
    history_source = source_root / "silver" / "players_history.parquet"
    rules_source = source_root / "pinned" / "rules_2026-27.json"
    for path in (stats_source, history_source, rules_source):
        if not path.exists():
            raise SystemExit(
                f"missing {path}. The generator reads a populated local store; "
                "run `xg ingest`, `xg backfill` and `xg build-features` first."
            )

    stats = pl.read_parquet(stats_source)
    history = pl.read_parquet(history_source)

    selected = select_player_seasons(
        stats,
        history,
        seasons=seasons,
        quota=quota,
        supplement=supplement,
        seed=resolved_seed,
    )
    sampled = sample_gameweek_stats(stats, selected)

    files: list[dict[str, Any]] = []

    stats_out = out_root / "silver" / "player_gameweek_stats.parquet"
    _write_parquet(sampled, stats_out)
    files.append(
        _describe(
            stats_out,
            out_root,
            source=_portable(stats_source),
            rows=sampled.height,
            columns=sampled.width,
            note=(
                f"club-and-position stratified sample: {dict(quota)} per club per season "
                f"plus {supplement} random per season, seasons {list(seasons)}"
            ),
            extra={
                "player_seasons": selected.height,
                "distinct_players": int(selected["player_code"].n_unique()),
                "seasons": sorted(sampled["season"].unique().to_list()),
                "clubs": int(
                    history.join(selected, on=["player_code", "season"], how="semi")[
                        "team_name"
                    ].n_unique()
                ),
                "source_rows": stats.height,
            },
        )
    )

    history_out = out_root / "silver" / "players_history.parquet"
    _write_parquet(history.sort(["season", "player_code"]), history_out)
    files.append(
        _describe(
            history_out,
            out_root,
            source=_portable(history_source),
            rows=history.height,
            columns=history.width,
            note="copied whole — every season, every player; the file is already small",
        )
    )

    rules_out = out_root / "pinned" / "rules_2026-27.json"
    rules_out.parent.mkdir(parents=True, exist_ok=True)
    rules_out.write_bytes(rules_source.read_bytes())
    files.append(
        _describe(
            rules_out,
            out_root,
            source=_portable(rules_source),
            note=(
                "byte-for-byte copy of the pinned rules snapshot. CLAUDE.md requires "
                "scoring constants to load from a pinned snapshot with a drift check, so "
                "the demo goes through that path rather than around it"
            ),
        )
    )

    bronze_dir = out_root / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    bronze_records: list[dict[str, Any]] = []
    for source in (_BOOTSTRAP_SOURCE, _FIXTURES_SOURCE):
        found = _latest_bronze_payload(source_root, source)
        if found is None:
            raise SystemExit(f"no {source} snapshot under {source_root / 'bronze'}")
        payload, record = found
        destination = bronze_dir / f"{source}.json.gz"
        # mtime=0 so identical bytes produce an identical file, the same
        # guarantee FileSystemBronzeStore makes.
        destination.write_bytes(gzip.compress(payload, mtime=0))
        record["fixture_path"] = destination.relative_to(out_root).as_posix()
        bronze_records.append(record)
        files.append(
            _describe(
                destination,
                out_root,
                source=_portable(source_root / "bronze" / source),
                note="gzipped official payload, replayed into a bronze store by `xg demo`",
                extra={"uncompressed_bytes": len(payload), "payload_sha256": _sha256(payload)},
            )
        )

    provenance: dict[str, Any] = {
        "generator": "tools/build_demo_fixture.py",
        "generator_version": GENERATOR_VERSION,
        "extracted_at": datetime.now(UTC).isoformat(),
        "source_root": _portable(source_root),
        "seed": resolved_seed,
        "seasons": list(seasons),
        "club_quota": dict(quota),
        "random_supplement_per_season": supplement,
        "max_fixture_bytes": MAX_FIXTURE_BYTES,
        # The two things that make a fixture go stale without anyone touching
        # it: the silver contract it was cut from, and the feature set that
        # will be computed over it. Recorded rather than re-derived, so a test
        # can detect the drift from the sidecar alone.
        "silver_schema_sha256": silver_schema_digest(),
        "catalogue_hash": _catalogue_hash(),
        "bronze_snapshots": bronze_records,
        "files": files,
        "warning": (
            "Real, public FPL data, sampled — not synthetic and not complete. "
            f"Anything computed from it describes {selected['player_code'].n_unique()} players "
            f"over {len(seasons)} seasons, so it demonstrates that the pipeline runs. "
            "It is not evidence about football."
        ),
    }

    (out_root / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def _describe(
    path: Path,
    out_root: Path,
    *,
    source: str,
    rows: int | None = None,
    columns: int | None = None,
    note: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = path.read_bytes()
    record: dict[str, Any] = {
        "path": path.relative_to(out_root).as_posix(),
        "derived_from": source,
        "byte_size": len(payload),
        "sha256": _sha256(payload),
        "note": note,
    }
    if rows is not None:
        record["rows"] = rows
    if columns is not None:
        record["columns"] = columns
    if extra:
        record.update(extra)
    return record


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def check(out_root: Path) -> list[str]:
    """Every way the committed fixtures disagree with their own sidecar."""
    problems: list[str] = []
    sidecar = out_root / PROVENANCE_NAME
    if not sidecar.exists():
        return [f"no {sidecar}"]

    provenance = json.loads(sidecar.read_text())
    files = provenance.get("files", [])
    if not files:
        problems.append("provenance records no files")

    for record in files:
        path = out_root / str(record["path"])
        if not path.exists():
            problems.append(f"{record['path']}: recorded but missing")
            continue
        payload = path.read_bytes()
        if len(payload) != record["byte_size"]:
            problems.append(
                f"{record['path']}: {len(payload)} bytes on disk, {record['byte_size']} recorded"
            )
        if _sha256(payload) != record["sha256"]:
            problems.append(f"{record['path']}: sha256 differs from the recorded digest")
        if len(payload) > MAX_FIXTURE_BYTES:
            problems.append(
                f"{record['path']}: {len(payload)} bytes exceeds the "
                f"{MAX_FIXTURE_BYTES} pre-commit ceiling"
            )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=REPO_ROOT / ".data")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "fixtures")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed fixtures against their sidecar and exit.",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = check(args.out)
        for problem in problems:
            _echo(f"  FAIL {problem}")
        _echo("fixtures OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    provenance = build(source_root=args.source, out_root=args.out)
    _echo(f"seed {provenance['seed']}  generator v{provenance['generator_version']}")
    for record in provenance["files"]:
        size = int(record["byte_size"])
        headroom = MAX_FIXTURE_BYTES - size
        _echo(f"  {record['path']:<48} {size:>8,} bytes  ({headroom:+,} vs ceiling)")
    problems = check(args.out)
    for problem in problems:
        _echo(f"  FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
