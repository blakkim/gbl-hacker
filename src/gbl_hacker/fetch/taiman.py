"""Taiman Party Great League meta fetcher.

Retrieves raw upstream payloads from the two backend AJAX endpoints that
``pokemongo-get.com/taimanparty/`` calls under the hood — bypassing the
Vue.js SPA and going straight to the JSON / HTML the page itself consumes.

The two endpoints (verified live 2026-05-13):

1. ``getSeasonLeague.php`` (POST ``season=<n>``) — returns a large JSON
   array, one entry per league for the given season, each carrying a
   ``list`` of per-Pokémon usage counts.

2. ``BattlePvpRecommendNewVue.php`` (GET, query string with
   ``season, league, between, pokemon_first, search_type, ary_search,
   en_flg``) — returns an HTML fragment with the top recommended 3v3
   teams for that league/season slice.

This fetcher is deliberately dumb: it does not parse, does not cache,
and does not interpret league/bracket semantics. Those concerns belong
to ``parse_great_league_meta`` so that the network boundary stays narrow.

History note: a prior v0.1 attempt fetched the SPA root page with a
fabricated ``?league=`` query string. That call returned a Vue.js shell
with no usable data. The friendly-looking ``taiman_great_league_sample.html``
fixture from that era is preserved for archaeological reference but is
not the wire format any real refresh ever sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

import httpx

# Backend AJAX endpoints. These are the URLs the SPA itself hits — not
# the human-facing /taimanparty/ page.
AJAX_BASE: Final[str] = (
    "https://pokemongo-get.com/wp-content/themes/simplicity2-child/ajax"
)
SEASON_LEAGUE_URL: Final[str] = f"{AJAX_BASE}/gblleague/getSeasonLeague.php"
RECOMMEND_URL: Final[str] = (
    f"{AJAX_BASE}/battlelog_240822/battleparty/BattlePvpRecommendNewVue.php"
)
SITE_REFERER: Final[str] = "https://pokemongo-get.com/taimanparty/"

# Great League league_id is 0 in Taiman Party's internal table.
GREAT_LEAGUE_ID: Final[int] = 0

# Current GBL season as of fixture capture (2026-05-13). The site itself
# hard-codes this in its Vue state; we accept it as a default override
# and let CLI callers pin a different season explicitly. A future helper
# can scrape the current season number out of the SPA HTML.
DEFAULT_SEASON: Final[int] = 27

DEFAULT_TIMEOUT_SECONDS: Final[float] = 15.0

# Source caveat surfaced on every meta_snapshot rendering per the
# data_honesty evaluation principle. Carried alongside the raw bytes so
# downstream stages cannot drop it accidentally.
TAIMAN_SOURCE_CAVEAT: Final[str] = (
    "Taiman Party is a report-density-weighted upper-bracket meta feed. "
    "Sample density drops sharply past the upper bracket — top-500-specific "
    "slices are NOT reliable and must be interpreted as crowd-trend, not "
    "ground-truth top-rank usage."
)


class TaimanFetchError(Exception):
    """Base class for all Taiman Party fetcher failures."""


class TaimanNetworkError(TaimanFetchError):
    """The request never produced an HTTP response (DNS, TLS, timeout)."""

    def __init__(self, message: str, *, url: str) -> None:
        super().__init__(message)
        self.url = url


class TaimanHTTPError(TaimanFetchError):
    """Upstream returned a non-2xx HTTP response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        url: str,
        body: bytes = b"",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body = body


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of a single successful (2xx) Taiman Party fetch.

    Kept as a thin record around the wire response so downstream parsing
    can be tested without a network round-trip.

    Attributes:
        url: URL that was actually fetched.
        status_code: HTTP status (always 2xx for a returned result).
        content: Raw response body bytes, unparsed.
        content_type: ``Content-Type`` header value, or ``""`` when absent.
        fetched_at: UTC timestamp captured at response receipt.
        source_caveat: Static data-honesty string carried with the bytes.
    """

    url: str
    status_code: int
    content: bytes
    content_type: str
    fetched_at: datetime
    source_caveat: str = field(default=TAIMAN_SOURCE_CAVEAT)


@dataclass(frozen=True, slots=True)
class TaimanRawSnapshot:
    """A pair of raw responses comprising one Great League meta refresh.

    The Taiman Party site renders its team-recommend view by combining
    two independent backend calls:

    - ``season_league``: per-league Pokémon usage counts (JSON)
    - ``recommend``: ranked top-N 3v3 teams for the chosen league (HTML)

    Both are needed to populate the engine's ``MetaSnapshot``. A snapshot
    pair is the atomic unit a refresh persists and a parser consumes.
    """

    season: int
    league_id: int
    season_league: FetchResult
    recommend: FetchResult


def fetch_great_league_meta(
    *,
    season: int = DEFAULT_SEASON,
    league_id: int = GREAT_LEAGUE_ID,
    between: int = 1,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = "gbl-hacker/0.1 (+https://github.com/local/gbl-hacker)",
) -> TaimanRawSnapshot:
    """Fetch both Taiman Party backend payloads for a Great League refresh.

    Args:
        season: GBL season number (Taiman Party internal). Defaults to the
            value pinned at fixture capture; callers should override when
            the site rolls to a new season.
        league_id: League identifier in the Taiman Party table. Great
            League is ``0``; other values are reserved for Ultra/Master
            and event cups (out of scope for v0.1).
        between: Aggregation window selector (``1`` = 直近の集計 / recent;
            site UI also exposes ``2`` = 全ての集計 and ``3`` = リアルタイム).
        client: Optional pre-built ``httpx.Client`` — the seam tests use
            to inject ``httpx.MockTransport``. Caller owns its lifecycle.
        timeout: Per-request timeout. Ignored when an explicit client is
            supplied (the client's own timeout wins).
        user_agent: ``User-Agent`` header value. Identifying the engine
            is a courtesy to the upstream operator.

    Returns:
        ``TaimanRawSnapshot`` carrying both responses + the season /
        league_id used.

    Raises:
        TaimanHTTPError: Either upstream call returned non-2xx.
        TaimanNetworkError: Either upstream call never returned a response
            (DNS, connection, TLS, timeout).
    """

    owns_client = client is None
    active_client = client if client is not None else httpx.Client(timeout=timeout)

    try:
        season_result = _fetch_season_league(
            client=active_client,
            season=season,
            user_agent=user_agent,
        )
        recommend_result = _fetch_recommend(
            client=active_client,
            season=season,
            league_id=league_id,
            between=between,
            user_agent=user_agent,
        )
    finally:
        if owns_client:
            active_client.close()

    return TaimanRawSnapshot(
        season=season,
        league_id=league_id,
        season_league=season_result,
        recommend=recommend_result,
    )


def _fetch_season_league(
    *,
    client: httpx.Client,
    season: int,
    user_agent: str,
) -> FetchResult:
    """POST ``getSeasonLeague.php`` with ``season=<n>``.

    The backend expects form-encoded body, not query string. Response is
    a JSON array of league entries; we return it as raw bytes for the
    parser to decode.
    """

    headers = _headers(user_agent, accept="application/json, text/plain, */*")
    headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"

    return _safe_request(
        client,
        method="POST",
        url=SEASON_LEAGUE_URL,
        headers=headers,
        data={"season": str(season)},
    )


def _fetch_recommend(
    *,
    client: httpx.Client,
    season: int,
    league_id: int,
    between: int,
    user_agent: str,
) -> FetchResult:
    """GET ``BattlePvpRecommendNewVue.php`` with the full query schema.

    Query parameters captured live (2026-05-13):
        season, league, between, pokemon_first, search_type, ary_search,
        en_flg
    """

    params = {
        "season": str(season),
        "league": str(league_id),
        "between": str(between),
        "pokemon_first": "0",
        "search_type": "0",
        "ary_search": ",,",
        "en_flg": "false",
    }
    headers = _headers(user_agent, accept="text/html, */*; q=0.01")

    return _safe_request(
        client,
        method="GET",
        url=RECOMMEND_URL,
        headers=headers,
        params=params,
    )


def _headers(user_agent: str, *, accept: str) -> dict[str, str]:
    """Default request headers — matches the SPA's own XHR signature."""
    return {
        "User-Agent": user_agent,
        "Accept": accept,
        "Accept-Language": "ja,en;q=0.8",
        "Referer": SITE_REFERER,
        "Origin": "https://pokemongo-get.com",
        "X-Requested-With": "XMLHttpRequest",
    }


def _safe_request(
    client: httpx.Client,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    params: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
) -> FetchResult:
    """Issue a single HTTP request and normalize errors to TaimanFetchError."""

    try:
        response = client.request(method, url, headers=headers, params=params, data=data)
    except httpx.TimeoutException as exc:
        raise TaimanNetworkError(
            f"Timed out fetching Taiman Party endpoint {url}: {exc}",
            url=url,
        ) from exc
    except httpx.TransportError as exc:
        raise TaimanNetworkError(
            f"Network error fetching Taiman Party endpoint {url}: {exc}",
            url=url,
        ) from exc
    except httpx.HTTPError as exc:
        raise TaimanNetworkError(
            f"HTTP transport failure fetching {url}: {exc}",
            url=url,
        ) from exc

    if not (200 <= response.status_code < 300):
        raise TaimanHTTPError(
            f"Taiman Party fetch failed: HTTP {response.status_code} for {url}",
            status_code=response.status_code,
            url=url,
            body=response.content,
        )

    return FetchResult(
        url=str(response.request.url) if response.request else url,
        status_code=response.status_code,
        content=response.content,
        content_type=response.headers.get("content-type", ""),
        fetched_at=datetime.now(tz=timezone.utc),
    )


__all__ = [
    "DEFAULT_SEASON",
    "GREAT_LEAGUE_ID",
    "RECOMMEND_URL",
    "SEASON_LEAGUE_URL",
    "TAIMAN_SOURCE_CAVEAT",
    "FetchResult",
    "TaimanFetchError",
    "TaimanHTTPError",
    "TaimanNetworkError",
    "TaimanRawSnapshot",
    "fetch_great_league_meta",
]
