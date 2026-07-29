"""Normalize football-data.co.uk season CSVs into ``team_match_events``.

Three things here are load-bearing and each is a place a silent error would
otherwise live.

**Identity.** The source names teams in prose — ``"Man United"``, ``"Nott'm
Forest"``, ``"Tottenham"`` — and FPL names them differently. Codes are never
transcribed from memory: the mapping resolves *names* against the ``teams``
silver table and takes ``team_code`` from there, so it stays correct when a club
is promoted, relegated or renamed. The alias table below carries spelling
differences only, and every one of its entries was derived by diffing the two
vocabularies rather than recalled.

**Availability.** These are post-match statistics. A row stamped with its
kickoff would let a model read a match's shot count before the match had been
played, which is precisely the leak the four-timestamp rule exists to prevent.
The source refreshes the whole-season file periodically and publishes no
per-match timestamp, so the strongest claim we can actually defend is *one day
after kickoff*. That is the bound used. It is conservative by roughly a day and
costs nothing in FPL terms, because a gameweek's matches finish days before the
next deadline.

**Unmatched teams.** Dropped rows are returned in a report, never swallowed. A
Championship file maps only the handful of clubs FPL knows about, and a silent
drop there would look identical to a club that simply took no shots.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

from xg_alonso.pipelines.normalization.schema import TEAM_MATCH_EVENTS_SCHEMA, conform, empty_frame

__all__ = [
    "PUBLICATION_LAG",
    "SOURCE_TEAM_ALIASES",
    "MatchEventsReport",
    "TeamMappingError",
    "TeamNameMapping",
    "map_team_names",
    "normalize_match_events",
]

#: Kickoffs are published in UK local time, which is BST for most of a season
#: and GMT for the rest. Parsing them as UTC would put every August kickoff an
#: hour early — and an hour early is, for an availability bound, the wrong
#: direction.
_UK = ZoneInfo("Europe/London")

#: How long after kickoff we are willing to claim the row was available.
#:
#: Not an estimate of when the file actually updated — we cannot observe that.
#: It is the weakest claim the source's behaviour supports, chosen because an
#: availability bound that is too early is a leak and one that is too late is
#: merely a small loss of history.
PUBLICATION_LAG: Final[timedelta] = timedelta(days=1)

#: Kickoff time used when the source omits ``Time``. Late in the day, so the
#: derived availability can only move later, never earlier.
_FALLBACK_KICKOFF = (23, 59)

#: Spelling differences between the source's team names and FPL's, and nothing
#: else. Derived on 2026-07-29 by diffing all 25 distinct names across the
#: 2022-23 to 2025-26 ``E0`` files against the FPL ``teams`` tables for the same
#: seasons; these three were the only names that failed both exact and
#: prefix matching.
SOURCE_TEAM_ALIASES: Final[dict[str, str]] = {
    "man united": "man utd",
    "tottenham": "spurs",
    "sheffield united": "sheffield utd",
}


class TeamMappingError(RuntimeError):
    """Source team names could not be resolved to FPL clubs."""


def _normalise(name: str) -> str:
    """Fold a team name to a comparable key.

    Punctuation only — no word removal. Stripping ``"united"`` or ``"city"``
    would collapse Manchester's two clubs and Hull onto Coventry, so the
    differences those words carry are handled by the alias table instead.
    """
    lowered = name.strip().lower().replace("'", "").replace(".", "")
    if lowered.endswith(" fc"):
        lowered = lowered[:-3]
    return " ".join(lowered.split())


@dataclass(frozen=True)
class TeamNameMapping:
    """Resolved source names, and the ones nothing matched."""

    resolved: dict[str, int]
    unmatched: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.unmatched


def map_team_names(names: list[str], teams: pl.DataFrame) -> TeamNameMapping:
    """Resolve source team names to stable FPL ``team_code`` values.

    Matching runs exact, then alias, then *unique* prefix. The prefix step is
    what lets ``"Ipswich"`` reach ``"Ipswich Town"`` and ``"Coventry"`` reach
    ``"Coventry City"`` without an alias entry per promoted club. It refuses to
    guess: a prefix matching two clubs leaves the name unmatched rather than
    picking one.

    Args:
        names: Distinct team names as the source spells them.
        teams: The ``teams`` silver frame, with ``name`` and ``team_code``.
    """
    if teams.is_empty():
        return TeamNameMapping(resolved={}, unmatched=tuple(sorted(set(names))))

    by_key: dict[str, int] = {}
    for row in teams.select("name", "team_code").iter_rows(named=True):
        fpl_name = row["name"]
        code = row["team_code"]
        if fpl_name is None or code is None:
            continue
        by_key[_normalise(str(fpl_name))] = int(code)

    resolved: dict[str, int] = {}
    unmatched: list[str] = []

    for name in sorted(set(names)):
        key = _normalise(name)

        if key in by_key:
            resolved[name] = by_key[key]
            continue

        alias = SOURCE_TEAM_ALIASES.get(key)
        if alias is not None and alias in by_key:
            resolved[name] = by_key[alias]
            continue

        candidates = [
            code
            for fpl_key, code in by_key.items()
            if fpl_key.startswith(key) or key.startswith(fpl_key)
        ]
        if len(candidates) == 1:
            resolved[name] = candidates[0]
            continue

        unmatched.append(name)

    return TeamNameMapping(resolved=resolved, unmatched=tuple(unmatched))


@dataclass(frozen=True)
class MatchEventsReport:
    """What one normalization run produced, and what it could not."""

    frame: pl.DataFrame
    season: str
    division: str
    matches: int
    unmatched_teams: tuple[str, ...]
    skipped_rows: int

    @property
    def complete(self) -> bool:
        """Whether every source row became two silver rows."""
        return not self.unmatched_teams and self.skipped_rows == 0


def _parse_kickoff(date_text: str, time_text: str | None) -> datetime | None:
    """Parse the source's ``dd/mm/yyyy`` date and ``HH:MM`` UK-local time."""
    stamp = date_text.strip()
    if not stamp:
        return None

    parsed: datetime | None = None
    for pattern in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            parsed = datetime.strptime(stamp, pattern)  # noqa: DTZ007
        except ValueError:
            continue
        break
    if parsed is None:
        return None

    hour, minute = _FALLBACK_KICKOFF
    if time_text:
        try:
            clock = datetime.strptime(time_text.strip(), "%H:%M")  # noqa: DTZ007
        except ValueError:
            pass
        else:
            hour, minute = clock.hour, clock.minute

    local = parsed.replace(hour=hour, minute=minute, tzinfo=_UK)
    return local.astimezone(UTC)


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_match_events(
    payload: bytes,
    *,
    season: str,
    division: str,
    teams: pl.DataFrame,
    required_columns: tuple[str, ...],
    strict: bool = True,
) -> MatchEventsReport:
    """Turn one season-division CSV into team-match rows.

    Args:
        payload: The raw CSV bytes as stored in bronze.
        season: The season the file covers, e.g. ``"2024-25"``.
        division: Source division code, e.g. ``"E0"``.
        teams: The ``teams`` silver frame for this season.
        required_columns: Columns whose absence is a hard error.
        strict: Raise when a team name cannot be resolved. Set ``False`` only
            for a division FPL does not fully cover, such as ``E1``, where most
            clubs are genuinely absent from FPL rather than mis-spelled.

    Raises:
        TeamMappingError: on an unresolvable team name while ``strict``.
        ValueError: when a required column is missing. A partially-populated
            column reads downstream as signal, so this fails rather than nulls.
    """
    text = payload.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = [row for row in reader if (row.get("HomeTeam") or "").strip()]

    if not rows:
        return MatchEventsReport(
            frame=empty_frame(TEAM_MATCH_EVENTS_SCHEMA),
            season=season,
            division=division,
            matches=0,
            unmatched_teams=(),
            skipped_rows=0,
        )

    present = set(reader.fieldnames or [])
    missing = [column for column in required_columns if column not in present]
    if missing:
        message = (
            f"{division} {season}: source file is missing required columns {missing}. "
            "Refusing to normalize — a missing count would arrive downstream as a "
            "null column that looks like a legitimately quiet team."
        )
        raise ValueError(message)

    names = [row[side] for row in rows for side in ("HomeTeam", "AwayTeam")]
    mapping = map_team_names([name for name in names if name], teams)

    if strict and not mapping.complete:
        message = (
            f"{division} {season}: could not resolve team names "
            f"{list(mapping.unmatched)} against the FPL teams table. "
            "Add a spelling to SOURCE_TEAM_ALIASES rather than dropping the rows."
        )
        raise TeamMappingError(message)

    records: list[dict[str, object]] = []
    skipped = 0

    for row in rows:
        home_name = (row.get("HomeTeam") or "").strip()
        away_name = (row.get("AwayTeam") or "").strip()
        home_code = mapping.resolved.get(home_name)
        away_code = mapping.resolved.get(away_name)
        kickoff = _parse_kickoff(row.get("Date") or "", row.get("Time"))

        if home_code is None or away_code is None or kickoff is None:
            skipped += 1
            continue

        available = kickoff + PUBLICATION_LAG

        # Home perspective, then away. `_side` reads the source's H*/A* pairs
        # once per direction so the two rows cannot drift apart.
        for is_home, own, opponent in ((True, "H", "A"), (False, "A", "H")):
            records.append(
                {
                    "team_code": home_code if is_home else away_code,
                    "opponent_team_code": away_code if is_home else home_code,
                    "season": season,
                    "division": division,
                    "kickoff_time": kickoff,
                    "was_home": is_home,
                    "goals_for": _as_int(row.get(f"FT{own}G")),
                    "goals_against": _as_int(row.get(f"FT{opponent}G")),
                    "shots": _as_int(row.get(f"{own}S")),
                    "shots_against": _as_int(row.get(f"{opponent}S")),
                    "shots_on_target": _as_int(row.get(f"{own}ST")),
                    "shots_on_target_against": _as_int(row.get(f"{opponent}ST")),
                    "corners": _as_int(row.get(f"{own}C")),
                    "corners_against": _as_int(row.get(f"{opponent}C")),
                    "fouls_committed": _as_int(row.get(f"{own}F")),
                    "fouls_suffered": _as_int(row.get(f"{opponent}F")),
                    "yellow_cards": _as_int(row.get(f"{own}Y")),
                    "red_cards": _as_int(row.get(f"{own}R")),
                    "available_time": available,
                }
            )

    frame = conform(pl.DataFrame(records), TEAM_MATCH_EVENTS_SCHEMA)
    return MatchEventsReport(
        frame=frame,
        season=season,
        division=division,
        matches=len(rows) - skipped,
        unmatched_teams=mapping.unmatched,
        skipped_rows=skipped,
    )
