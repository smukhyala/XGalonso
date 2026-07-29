"""A ``robots.txt`` gate that runs before any non-API fetch.

Decision D6 was relaxed to permit fetching free public match data (see
``docs/match_event_data.md``). Permission to fetch is not permission to fetch
*anything*, so the relaxation ships with its own boundary: this module refuses a
URL whose origin disallows us, and it refuses it in code rather than in a
comment a future adapter can forget to read.

The rule exists because the first candidate source we evaluated failed it.
Understat publishes exactly the shot-level data the brief asked for, and its
``robots.txt`` is::

    User-agent: *
    Disallow: /

That is an unambiguous machine-readable refusal covering every path. A comment
saying "be polite" would not have stopped an adapter being written against it;
a gate does.

**Semantics** follow RFC 9309:

- ``2xx`` — parse and obey the rules.
- ``4xx`` — no rules exist, so everything is allowed. FPL serves its SPA shell
  at ``/robots.txt`` and ``raw.githubusercontent.com`` returns 404; both are
  correctly read as unrestricted.
- ``5xx`` or unreachable — assume **complete disallow**. This is the one place
  the RFC is stricter than convenience would like, and it is followed: an
  origin we cannot ask is an origin that has not said yes.

Rules are fetched once per origin per gate instance and cached, so a season
backfill costs one extra request in total.
"""

from __future__ import annotations

import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

__all__ = [
    "USER_AGENT",
    "Pacer",
    "RobotsDisallowedError",
    "RobotsGate",
    "RobotsUnavailableError",
]

#: One identity for every outbound request. We identify ourselves rather than
#: impersonating a browser: a source that wants to refuse us must be able to.
USER_AGENT = "xg-alonso/0.1 (+https://github.com/smukhyala; research)"

#: Floor on the gap between requests to one origin when robots.txt names no
#: crawl-delay. Chosen to be slower than a human clicking, since nothing here is
#: latency-sensitive — a season backfill is a handful of requests.
DEFAULT_CRAWL_DELAY_SECONDS = 1.0


class RobotsDisallowedError(RuntimeError):
    """The origin's ``robots.txt`` refuses this URL for our user agent."""


class RobotsUnavailableError(RuntimeError):
    """The origin's ``robots.txt`` could not be read, so consent is unknown.

    Separate from :class:`RobotsDisallowedError` because the remedies differ: a
    disallow is final and means find another source, while an unavailable
    ``robots.txt`` is usually transient and worth retrying.
    """


class Pacer:
    """Enforces a minimum gap between requests to one origin.

    Deliberately wall-clock based and deliberately not injectable with a fake
    clock: a test that wanted to skip the wait would be testing an adapter that
    does not pace, which is the behaviour worth preventing. Tests set the delay
    to zero instead, which is honest about what they are skipping.
    """

    __slots__ = ("_delay", "_last")

    def __init__(self, delay_seconds: float = DEFAULT_CRAWL_DELAY_SECONDS) -> None:
        self._delay = max(0.0, delay_seconds)
        self._last: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last is not None:
            remaining = self._delay - (now - self._last)
            if remaining > 0:
                time.sleep(remaining)
        self._last = time.monotonic()


def _origin(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        message = f"cannot check robots.txt for a non-absolute URL: {url!r}"
        raise ValueError(message)
    return f"{parts.scheme}://{parts.netloc}"


class RobotsGate:
    """Per-origin ``robots.txt`` rules, fetched once and cached."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        user_agent: str = USER_AGENT,
        timeout: float = 15.0,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout
        self._parsers: dict[str, RobotFileParser | None] = {}

    def _parser_for(self, origin: str) -> RobotFileParser | None:
        """Return the origin's rules, or ``None`` when it publishes none."""
        if origin in self._parsers:
            return self._parsers[origin]

        try:
            response = self._client.get(f"{origin}/robots.txt", timeout=self._timeout)
        except httpx.HTTPError as exc:
            message = (
                f"could not read {origin}/robots.txt ({exc}). "
                "RFC 9309 treats an unreachable robots.txt as a complete disallow, "
                "so the fetch is refused rather than attempted."
            )
            raise RobotsUnavailableError(message) from exc

        if response.status_code >= 500:
            message = (
                f"{origin}/robots.txt returned {response.status_code}. "
                "RFC 9309 treats a server error as a complete disallow, so the "
                "fetch is refused rather than attempted."
            )
            raise RobotsUnavailableError(message)

        if response.status_code >= 400:
            # No rules published. Unrestricted, per RFC 9309.
            self._parsers[origin] = None
            return None

        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        self._parsers[origin] = parser
        return parser

    def crawl_delay(self, url: str) -> float:
        """The origin's requested delay, or the default floor when it names none."""
        parser = self._parser_for(_origin(url))
        if parser is None:
            return DEFAULT_CRAWL_DELAY_SECONDS
        declared = parser.crawl_delay(self._user_agent)
        if declared is None:
            return DEFAULT_CRAWL_DELAY_SECONDS
        return max(DEFAULT_CRAWL_DELAY_SECONDS, float(declared))

    def allows(self, url: str) -> bool:
        """Whether the origin permits our user agent to fetch ``url``."""
        parser = self._parser_for(_origin(url))
        if parser is None:
            return True
        return parser.can_fetch(self._user_agent, url)

    def require(self, url: str) -> None:
        """Raise unless the origin permits this fetch.

        Raises:
            RobotsDisallowedError: the origin refuses this path.
            RobotsUnavailableError: consent could not be established.
        """
        if not self.allows(url):
            message = (
                f"refusing to fetch {url}: the origin's robots.txt disallows it for "
                f"{self._user_agent!r}. This is a machine-readable refusal, not a "
                "rate limit — find a source that permits automated access."
            )
            raise RobotsDisallowedError(message)
