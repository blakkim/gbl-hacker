"""HTTP ingestion layer for upstream meta sources.

Sub-modules:
    taiman: Taiman Party (pokemongo-get.com) backend AJAX fetcher.

The fetch layer is intentionally thin: it pulls raw bytes from the
upstream URLs, surfaces transport/HTTP errors as typed exceptions, and
leaves parsing to a downstream module. This keeps the network boundary
auditable and lets parsing tests run entirely against fixture bytes.
"""

from gbl_hacker.fetch.taiman import (
    DEFAULT_SEASON,
    GREAT_LEAGUE_ID,
    RECOMMEND_URL,
    SEASON_LEAGUE_URL,
    TAIMAN_SOURCE_CAVEAT,
    FetchResult,
    TaimanFetchError,
    TaimanHTTPError,
    TaimanNetworkError,
    TaimanRawSnapshot,
    fetch_great_league_meta,
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
