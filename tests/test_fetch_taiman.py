"""Tests for the Taiman Party Great League fetcher (live backend XHR pair).

All network I/O is replaced with ``httpx.MockTransport`` so the suite is
fully offline. The fetcher issues two requests per refresh — a POST to
``getSeasonLeague.php`` and a GET to ``BattlePvpRecommendNewVue.php`` —
and the tests pin both contracts:

* On 2xx for both calls, ``fetch_great_league_meta`` returns a
  ``TaimanRawSnapshot`` carrying the raw bytes of both responses.
* On non-2xx from either call, it raises ``TaimanHTTPError`` carrying
  the status code, URL, and body.
* On transport failure, it raises ``TaimanNetworkError``.
* The mock transport actually receives both requests — guards against a
  regression where the fetcher might silently short-circuit a call.
* The data-honesty source caveat is attached so it cannot be dropped.
"""

from __future__ import annotations

import httpx
import pytest

from gbl_hacker.fetch.taiman import (
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


SEASON_BODY = b'[{"gsl_league_id":"0","name_en":"Great League","list":{}}]'
RECOMMEND_BODY = (
    '<div class="league_party_sc flex-list">'
    '<span class="font-s12">1</span>位'
    '<div class="lps_pokemon"></div>'
    '<div data-party="X-0,Y-0,Z-0"></div>'
    '<span class="font-s12">10</span>件'
    "</div>"
).encode("utf-8")


def _client_with(handler) -> httpx.Client:
    """Build an httpx.Client whose transport is a MockTransport."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _make_dual_handler(
    *,
    season_status: int = 200,
    season_body: bytes = SEASON_BODY,
    recommend_status: int = 200,
    recommend_body: bytes = RECOMMEND_BODY,
    captured: list[httpx.Request] | None = None,
):
    """Build a MockTransport handler that routes by request URL.

    Returns the configured status/body for each of the two endpoints so a
    single test can pin both calls' behavior.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        url = str(request.url)
        if url.startswith(SEASON_LEAGUE_URL):
            return httpx.Response(
                season_status,
                content=season_body,
                headers={"content-type": "text/html; charset=UTF-8"},
            )
        if url.startswith(RECOMMEND_URL):
            return httpx.Response(
                recommend_status,
                content=recommend_body,
                headers={"content-type": "text/html; charset=UTF-8"},
            )
        raise AssertionError(f"unexpected request URL: {url}")

    return handler


def test_fetch_great_league_meta_returns_both_payloads_on_2xx() -> None:
    """200/200 → ``TaimanRawSnapshot`` carrying both raw bytes."""

    captured: list[httpx.Request] = []
    handler = _make_dual_handler(captured=captured)

    with _client_with(handler) as client:
        raw = fetch_great_league_meta(client=client)

    # Both endpoints must actually have been hit (guards against silent
    # short-circuit or canned-data regressions).
    methods_and_paths = sorted(
        (r.method, str(r.url).split("?")[0]) for r in captured
    )
    assert methods_and_paths == sorted(
        [
            ("POST", SEASON_LEAGUE_URL),
            ("GET", RECOMMEND_URL),
        ]
    )

    assert isinstance(raw, TaimanRawSnapshot)
    assert raw.league_id == GREAT_LEAGUE_ID
    assert isinstance(raw.season_league, FetchResult)
    assert isinstance(raw.recommend, FetchResult)
    assert raw.season_league.content == SEASON_BODY
    assert raw.recommend.content == RECOMMEND_BODY
    assert raw.season_league.source_caveat == TAIMAN_SOURCE_CAVEAT
    assert raw.recommend.source_caveat == TAIMAN_SOURCE_CAVEAT
    assert raw.season_league.fetched_at is not None
    assert raw.recommend.fetched_at is not None


def test_fetch_sends_form_encoded_season_to_get_season_league() -> None:
    """``getSeasonLeague.php`` must receive ``season=<n>`` as form body."""

    captured: list[httpx.Request] = []
    handler = _make_dual_handler(captured=captured)

    with _client_with(handler) as client:
        fetch_great_league_meta(client=client, season=27)

    season_req = next(
        r for r in captured if str(r.url).startswith(SEASON_LEAGUE_URL)
    )
    assert season_req.method == "POST"
    content_type = season_req.headers.get("content-type", "")
    assert "application/x-www-form-urlencoded" in content_type
    body = season_req.content.decode("utf-8")
    assert "season=27" in body


def test_fetch_uses_full_query_schema_on_recommend() -> None:
    """The recommend GET must carry the captured-live query schema."""

    captured: list[httpx.Request] = []
    handler = _make_dual_handler(captured=captured)

    with _client_with(handler) as client:
        fetch_great_league_meta(
            client=client,
            season=27,
            league_id=GREAT_LEAGUE_ID,
            between=1,
        )

    recommend_req = next(
        r for r in captured if str(r.url).startswith(RECOMMEND_URL)
    )
    params = dict(recommend_req.url.params)
    assert params["season"] == "27"
    assert params["league"] == "0"
    assert params["between"] == "1"
    # All four "static-but-required" parameters must be present, matching
    # what the site itself sends so we don't get a different code path
    # on the backend.
    assert params["pokemon_first"] == "0"
    assert params["search_type"] == "0"
    assert params["ary_search"] == ",,"
    assert params["en_flg"] == "false"


def test_fetch_sets_user_agent_and_referer() -> None:
    """Both calls must identify themselves and look like SPA XHRs."""

    captured: list[httpx.Request] = []
    handler = _make_dual_handler(captured=captured)

    with _client_with(handler) as client:
        fetch_great_league_meta(client=client)

    for req in captured:
        ua = req.headers.get("user-agent", "")
        assert "gbl-hacker" in ua.lower()
        # The SPA sends Referer + X-Requested-With on its own XHRs; we
        # mirror those so the backend treats us as the same caller.
        assert req.headers.get("referer", "").startswith(
            "https://pokemongo-get.com/taimanparty"
        )
        assert req.headers.get("x-requested-with") == "XMLHttpRequest"


@pytest.mark.parametrize("status_code", [400, 403, 404, 429, 500, 502, 503])
def test_fetch_raises_typed_http_error_on_season_non_2xx(status_code: int) -> None:
    """Non-2xx from ``getSeasonLeague.php`` raises TaimanHTTPError."""

    error_body = b'{"error":"season upstream said no"}'
    handler = _make_dual_handler(
        season_status=status_code,
        season_body=error_body,
    )

    with _client_with(handler) as client:
        with pytest.raises(TaimanHTTPError) as excinfo:
            fetch_great_league_meta(client=client)

    err = excinfo.value
    assert isinstance(err, TaimanFetchError)
    assert err.status_code == status_code
    assert err.url.startswith(SEASON_LEAGUE_URL)
    assert err.body == error_body
    assert str(status_code) in str(err)


@pytest.mark.parametrize("status_code", [400, 403, 404, 429, 500, 502, 503])
def test_fetch_raises_typed_http_error_on_recommend_non_2xx(
    status_code: int,
) -> None:
    """Non-2xx from the recommend endpoint raises TaimanHTTPError."""

    error_body = b'<div class="error">recommend upstream said no</div>'
    handler = _make_dual_handler(
        recommend_status=status_code,
        recommend_body=error_body,
    )

    with _client_with(handler) as client:
        with pytest.raises(TaimanHTTPError) as excinfo:
            fetch_great_league_meta(client=client)

    err = excinfo.value
    assert err.status_code == status_code
    assert err.url.startswith(RECOMMEND_URL)
    assert err.body == error_body


def test_fetch_raises_network_error_on_transport_failure() -> None:
    """Transport-level failure (no HTTP response) → TaimanNetworkError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated DNS / connection failure")

    with _client_with(handler) as client:
        with pytest.raises(TaimanNetworkError) as excinfo:
            fetch_great_league_meta(client=client)

    assert isinstance(excinfo.value, TaimanFetchError)
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


def test_fetch_raises_network_error_on_timeout() -> None:
    """A request timeout is reported as TaimanNetworkError, not HTTPError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated timeout")

    with _client_with(handler) as client:
        with pytest.raises(TaimanNetworkError):
            fetch_great_league_meta(client=client)


def test_fetch_treats_299_as_success_and_300_as_http_error() -> None:
    """Boundary check on the 2xx success window."""

    handler_ok = _make_dual_handler(
        season_status=299,
        recommend_status=299,
    )
    with _client_with(handler_ok) as client:
        raw = fetch_great_league_meta(client=client)
    assert raw.season_league.status_code == 299
    assert raw.recommend.status_code == 299

    handler_3xx = _make_dual_handler(season_status=301, season_body=b"")
    with _client_with(handler_3xx) as client:
        with pytest.raises(TaimanHTTPError) as excinfo:
            fetch_great_league_meta(client=client)
    assert excinfo.value.status_code == 301
