"""HTTP client for the official FPL API.

This is the only module permitted to import ``httpx`` (enforced by the
``http-isolation`` contract). Everything downstream reads from bronze snapshots,
which is what makes a feature build reproducible offline.

**Deriving ``available_time``.** We take the response ``Date`` header minus
``Age``, which tells us the origin produced this body at or before that instant.
That is a *lower bound* on true availability, not an observation: it admits data
slightly earlier than reality and therefore slightly increases leakage exposure.
It is still the right rule — the alternative, trusting the client clock, is
worse — but the bound is recorded honestly rather than described as conservative.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Final

import httpx

from xg_alonso.contracts.provenance import SourceTimestamps, TimeSource, utc_now

__all__ = [
    "FPL_BASE_URL",
    "FplApiClient",
    "FplResponse",
    "OfflineError",
    "derive_available_time",
]

FPL_BASE_URL: Final[str] = "https://fantasy.premierleague.com/api"

#: Set in CI so a test that reaches for the network fails loudly instead of flaking.
OFFLINE_ENV_VAR: Final[str] = "XG_ALONSO_OFFLINE"


class OfflineError(RuntimeError):
    """Raised when a network call is attempted while offline mode is set."""


class FplResponse:
    """A raw response plus the timestamps derived from it."""

    __slots__ = ("payload", "status_code", "timestamps", "url")

    def __init__(
        self,
        *,
        url: str,
        payload: bytes,
        status_code: int,
        timestamps: SourceTimestamps,
    ) -> None:
        self.url = url
        self.payload = payload
        self.status_code = status_code
        self.timestamps = timestamps


def derive_available_time(
    headers: httpx.Headers | dict[str, str], *, fallback: datetime
) -> tuple[datetime, TimeSource]:
    """Derive when the origin published this body.

    Returns ``Date - Age`` when both headers are usable, else the fallback.
    """
    raw_date = headers.get("date")
    if not raw_date:
        return fallback, TimeSource.HTTP_DATE_MINUS_AGE

    try:
        served_at = parsedate_to_datetime(raw_date)
    except (TypeError, ValueError):
        return fallback, TimeSource.HTTP_DATE_MINUS_AGE

    if served_at.tzinfo is None:
        served_at = served_at.replace(tzinfo=UTC)
    served_at = served_at.astimezone(UTC)

    age_seconds = 0
    raw_age = headers.get("age")
    if raw_age:
        try:
            age_seconds = max(0, int(raw_age))
        except ValueError:
            age_seconds = 0

    return served_at - timedelta(seconds=age_seconds), TimeSource.HTTP_DATE_MINUS_AGE


class FplApiClient:
    """Thin, polite client over the public FPL endpoints.

    No authentication, ever. Decision D3 keeps us on public endpoints only, so
    there are no credentials to store or leak.
    """

    def __init__(
        self,
        *,
        base_url: str = FPL_BASE_URL,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                # Identify ourselves rather than impersonating a browser.
                "User-Agent": "xg-alonso/0.1 (+https://github.com/smukhyala; research)",
                "Accept": "application/json",
            },
            follow_redirects=True,
        )

    def __enter__(self) -> FplApiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def get(self, path: str, *, event_time: datetime | None = None) -> FplResponse:
        """Fetch one endpoint and stamp it with all four timestamps.

        Args:
            path: Endpoint path, e.g. ``"bootstrap-static/"``.
            event_time: When the underlying real-world event occurred. Defaults
                to the derived availability time, which is correct for
                continuously-updated resources like ``bootstrap-static``.

        Raises:
            OfflineError: when ``XG_ALONSO_OFFLINE`` is set.
            httpx.HTTPStatusError: on a non-2xx response.
        """
        if os.environ.get(OFFLINE_ENV_VAR):
            raise OfflineError(
                f"refusing to fetch {path!r}: {OFFLINE_ENV_VAR} is set. "
                "Feature builds must be reproducible from stored snapshots."
            )

        url = f"{self._base_url}/{path.lstrip('/')}"
        observed_at = utc_now()
        response = self._client.get(url)
        response.raise_for_status()

        available_time, time_source = derive_available_time(response.headers, fallback=observed_at)
        processed_at = utc_now()

        return FplResponse(
            url=url,
            payload=response.content,
            status_code=response.status_code,
            timestamps=SourceTimestamps(
                event_time=event_time or available_time,
                observed_time=observed_at,
                available_time=available_time,
                processed_time=processed_at,
                time_source=time_source,
            ),
        )

    # -- endpoints --------------------------------------------------------

    def bootstrap_static(self) -> FplResponse:
        """Players, teams, gameweeks, scoring rules and squad constraints."""
        return self.get("bootstrap-static/")

    def fixtures(self) -> FplResponse:
        """Every fixture, including unplayed ones."""
        return self.get("fixtures/")

    def element_summary(self, element_id: int) -> FplResponse:
        """One player's gameweek history and prior-season aggregates."""
        return self.get(f"element-summary/{element_id}/")

    def entry(self, entry_id: int) -> FplResponse:
        """A manager's public summary."""
        return self.get(f"entry/{entry_id}/")

    def entry_history(self, entry_id: int) -> FplResponse:
        """A manager's per-gameweek history, bank and squad value."""
        return self.get(f"entry/{entry_id}/history/")

    def entry_transfers(self, entry_id: int) -> FplResponse:
        """A manager's transfer history.

        This is what makes decision D3 work: the response carries
        ``element_in_cost`` per move, which is the only public route to a
        player's purchase price and therefore to their selling price.
        """
        return self.get(f"entry/{entry_id}/transfers/")

    def entry_picks(self, entry_id: int, gameweek: int) -> FplResponse:
        """A manager's squad for one gameweek.

        Returns 404 before that gameweek's deadline — the picks do not exist
        yet. Callers must handle that rather than treating it as an error, which
        is why the CLI also accepts ``--squad-file``.
        """
        return self.get(f"entry/{entry_id}/event/{gameweek}/picks/")
