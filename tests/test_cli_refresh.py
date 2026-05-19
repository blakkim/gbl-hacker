"""Tests for the ``gblh refresh`` CLI subcommand.

Sub-AC 4 contract: a CLI command entrypoint wires fetcher → parser →
persistence behind a single subcommand. The headline test invokes the CLI
via its ``main()`` runner with a *mocked fetch* (no network) and asserts:

    * exit code is 0 on success
    * the persistence layer was called with the *parsed snapshot*
      (i.e. fetch → parse → persist actually ran in order, and the object
      handed to persistence is the parser's output, not the raw fetch
      payload)

Additional guard tests pin the surrounding contract:

    * the parsed snapshot fed into persistence reflects the fetched bytes
      (parser actually ran on the mocked content)
    * the data-honesty caveat is echoed to stdout on success
    * fetch / parse / persist failures map to distinct non-zero exit codes
    * a bare ``gblh`` (no subcommand) returns the argparse usage exit code

The tests are fully offline — no httpx, no real cache directory writes
outside ``tmp_path``.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gbl_hacker.cli import (
    EXIT_FETCH,
    EXIT_OK,
    EXIT_PARSE,
    EXIT_PERSIST,
    EXIT_USAGE,
    main,
)
from gbl_hacker.fetch.taiman import (
    GREAT_LEAGUE_ID,
    RECOMMEND_URL,
    SEASON_LEAGUE_URL,
    TAIMAN_SOURCE_CAVEAT,
    FetchResult,
    TaimanHTTPError,
    TaimanNetworkError,
    TaimanRawSnapshot,
)
from gbl_hacker.parse.taiman import (
    MetaSnapshot,
    PokemonUsage,
    TaimanParseError,
    TeamUsage,
    parse_great_league_meta,
)
from gbl_hacker.persist.snapshot import (
    SCHEMA_VERSION,
    SnapshotPersistError,
    StoredSnapshot,
    list_snapshots,
    read_snapshot,
    write_snapshot,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


SEASON_FIXTURE = (
    Path(__file__).parent / "fixtures" / "taiman_season_league.json"
)
RECOMMEND_FIXTURE = (
    Path(__file__).parent / "fixtures" / "taiman_recommend_great_league.html"
)


def _fixture_raw_snapshot() -> TaimanRawSnapshot:
    """Build a ``TaimanRawSnapshot`` carrying the recorded live fixture pair.

    Using the real captured backend responses means the parser exercises
    the live wire format pinned by ``test_parse_taiman.py``. The CLI
    tests therefore prove the *wiring*, not a parallel parser
    implementation.
    """

    when = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
    return TaimanRawSnapshot(
        season=27,
        league_id=GREAT_LEAGUE_ID,
        season_league=FetchResult(
            url=SEASON_LEAGUE_URL,
            status_code=200,
            content=SEASON_FIXTURE.read_bytes(),
            content_type="text/html; charset=UTF-8",
            fetched_at=when,
            source_caveat=TAIMAN_SOURCE_CAVEAT,
        ),
        recommend=FetchResult(
            url=RECOMMEND_URL + "?season=27&league=0&between=1",
            status_code=200,
            content=RECOMMEND_FIXTURE.read_bytes(),
            content_type="text/html; charset=UTF-8",
            fetched_at=when,
            source_caveat=TAIMAN_SOURCE_CAVEAT,
        ),
    )


def _hand_built_snapshot() -> MetaSnapshot:
    """A minimal, hand-built snapshot used by some negative-path tests."""

    return MetaSnapshot(
        league="great_league",
        rating_bracket="upper",
        fetched_at=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        source_url=RECOMMEND_URL,
        source_caveat=TAIMAN_SOURCE_CAVEAT,
        pokemon_usage=(PokemonUsage(species="azumarill", usage_pct=14.7, rank=1),),
        team_usage=(
            TeamUsage(
                members=("Azumarill", "Annihilape", "Registeel"),
                usage_pct=3.4,
                rank=1,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Sub-AC 4 central assertion
# ---------------------------------------------------------------------------


def test_refresh_exits_zero_and_persistence_receives_parsed_snapshot(
    tmp_path: Path,
) -> None:
    """``gblh refresh`` returns 0 and hands persistence the parsed snapshot.

    This is the headline Sub-AC 4 assertion. We:

    1. Mock the *fetcher* to return canned fixture bytes (no network).
    2. Let the real parser run on those bytes.
    3. Spy on the persister and capture what it was called with.
    4. Assert exit code == 0.
    5. Assert the persister got a ``MetaSnapshot`` whose content reflects
       the fixture (proving the parser ran in between).
    6. Assert the persister got the cache directory the CLI resolved.
    7. Assert a file actually landed on disk (proving persistence was not
       a no-op — the spy still calls through to ``write_snapshot``).
    """

    fake_raw = _fixture_raw_snapshot()

    fetcher_calls: list[None] = []

    def spy_fetcher() -> TaimanRawSnapshot:
        fetcher_calls.append(None)
        return fake_raw

    captured: dict[str, Any] = {}

    def spy_persister(snapshot: MetaSnapshot, cache_dir: Path) -> StoredSnapshot:
        # Record the exact object handed in so the test can prove the
        # parser's output was passed verbatim to persistence.
        captured["snapshot"] = snapshot
        captured["cache_dir"] = cache_dir
        # Call through to the real writer so we also verify the file
        # actually lands on disk — Sub-AC 4 asks for "wires through
        # persistence", not "wires through a no-op stub".
        return write_snapshot(snapshot, cache_dir=cache_dir)

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["refresh", "--cache-dir", str(tmp_path)],
        fetcher=spy_fetcher,
        persister=spy_persister,
        stdout=stdout,
        stderr=stderr,
    )

    # (1) Exit code 0 on success.
    assert exit_code == EXIT_OK, (
        f"expected exit 0; got {exit_code}. stderr={stderr.getvalue()!r}"
    )

    # (2) Fetcher actually ran.
    assert len(fetcher_calls) == 1, "fetcher was not invoked exactly once"

    # (3) Persister got *something* — Sub-AC 4 wiring assertion.
    assert "snapshot" in captured, (
        "persistence layer was never called — fetch → parse → persist "
        "wiring is broken"
    )

    # (4) Persister got a parsed MetaSnapshot, not a raw FetchResult.
    snapshot_arg = captured["snapshot"]
    assert isinstance(snapshot_arg, MetaSnapshot), (
        "persister received a "
        f"{type(snapshot_arg).__name__}; expected MetaSnapshot. "
        "fetch → parse → persist wiring is broken."
    )

    # (5) The snapshot content came from the *parsed* fixture, not from
    # some accidental default. The fixture pins specific rows.
    species_seen = {p.species for p in snapshot_arg.pokemon_usage}
    assert species_seen, "parser produced zero Pokémon rows"
    assert "ヌオー" in species_seen or "デカヌチャン" in species_seen, f"expected at least one top-meta species in {species_seen!r}"
    assert len(snapshot_arg.team_usage) > 0, "parser produced zero team rows"

    # (6) The cache_dir passed through verbatim.
    assert captured["cache_dir"] == tmp_path

    # (7) A file actually landed on disk.
    files = list_snapshots(tmp_path)
    assert len(files) == 1, f"expected one snapshot file; got {files}"

    # (8) And the round-tripped snapshot equals the one persistence saw.
    reloaded = read_snapshot(files[0])
    assert reloaded == snapshot_arg

    # (9) Data-honesty caveat surfaces in stdout per the evaluation
    # principle — the CLI is the most user-visible surface.
    assert "report-density" in stdout.getvalue().lower(), (
        "data-honesty caveat must be echoed on successful refresh"
    )


# ---------------------------------------------------------------------------
# Wiring guard — verify the persister is called *with the parser's exact
# output*, by sentinel identity (not just structural equality).
# ---------------------------------------------------------------------------


def test_refresh_passes_parser_output_identity_to_persister(
    tmp_path: Path,
) -> None:
    """The object handed to persistence IS the object the parser returned.

    This is the strictest version of "persistence was called with the
    parsed snapshot" — same Python identity, not just equal value.
    """

    sentinel_snapshot = _hand_built_snapshot()

    def spy_fetcher() -> TaimanRawSnapshot:
        return _fixture_raw_snapshot()

    def stub_parser(_: TaimanRawSnapshot) -> MetaSnapshot:
        # Replace the real parser so we can use identity comparison.
        return sentinel_snapshot

    captured: dict[str, Any] = {}

    def spy_persister(snapshot: MetaSnapshot, cache_dir: Path) -> StoredSnapshot:
        captured["snapshot"] = snapshot
        return write_snapshot(snapshot, cache_dir=cache_dir)

    exit_code = main(
        ["refresh", "--cache-dir", str(tmp_path)],
        fetcher=spy_fetcher,
        parser=stub_parser,
        persister=spy_persister,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    # Same object — wiring is direct, no defensive copy in between.
    assert captured["snapshot"] is sentinel_snapshot


# ---------------------------------------------------------------------------
# Non-zero exit paths — each ingestion stage maps to its own exit code.
# ---------------------------------------------------------------------------


def test_refresh_returns_fetch_exit_code_on_fetch_failure(
    tmp_path: Path,
) -> None:
    """A ``TaimanFetchError`` from the fetcher yields ``EXIT_FETCH``."""

    def failing_fetcher() -> TaimanRawSnapshot:
        raise TaimanNetworkError(
            "simulated DNS failure", url=RECOMMEND_URL
        )

    persister_called: list[None] = []

    def spy_persister(*_: Any, **__: Any) -> StoredSnapshot:
        persister_called.append(None)
        raise AssertionError("persister must not be called on fetch failure")

    stderr = io.StringIO()

    exit_code = main(
        ["refresh", "--cache-dir", str(tmp_path)],
        fetcher=failing_fetcher,
        persister=spy_persister,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_FETCH
    assert persister_called == []
    assert "fetch failed" in stderr.getvalue().lower()


def test_refresh_returns_parse_exit_code_on_parse_failure(
    tmp_path: Path,
) -> None:
    """A ``TaimanParseError`` from the parser yields ``EXIT_PARSE``."""

    def stub_fetcher() -> TaimanRawSnapshot:
        return _fixture_raw_snapshot()

    def failing_parser(_: TaimanRawSnapshot) -> MetaSnapshot:
        raise TaimanParseError("simulated DOM drift", url=RECOMMEND_URL)

    persister_called: list[None] = []

    def spy_persister(*_: Any, **__: Any) -> StoredSnapshot:
        persister_called.append(None)
        raise AssertionError("persister must not be called on parse failure")

    stderr = io.StringIO()

    exit_code = main(
        ["refresh", "--cache-dir", str(tmp_path)],
        fetcher=stub_fetcher,
        parser=failing_parser,
        persister=spy_persister,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_PARSE
    assert persister_called == []
    assert "parse failed" in stderr.getvalue().lower()


def test_refresh_returns_persist_exit_code_on_persist_failure(
    tmp_path: Path,
) -> None:
    """A ``SnapshotPersistError`` from the persister yields ``EXIT_PERSIST``."""

    def stub_fetcher() -> TaimanRawSnapshot:
        return _fixture_raw_snapshot()

    def failing_persister(_s: MetaSnapshot, _d: Path) -> StoredSnapshot:
        raise SnapshotPersistError("simulated disk-full failure")

    stderr = io.StringIO()

    exit_code = main(
        ["refresh", "--cache-dir", str(tmp_path)],
        fetcher=stub_fetcher,
        # Real parser runs on the fixture content.
        parser=parse_great_league_meta,
        persister=failing_persister,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_PERSIST
    assert "persist failed" in stderr.getvalue().lower()


def test_refresh_returns_fetch_exit_code_on_http_failure(
    tmp_path: Path,
) -> None:
    """An HTTP error (4xx/5xx from upstream) also maps to ``EXIT_FETCH``."""

    def failing_fetcher() -> TaimanRawSnapshot:
        raise TaimanHTTPError(
            "simulated 503", status_code=503, url=RECOMMEND_URL, body=b""
        )

    exit_code = main(
        ["refresh", "--cache-dir", str(tmp_path)],
        fetcher=failing_fetcher,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_FETCH


def test_refresh_debug_flag_reraises_underlying_exception(
    tmp_path: Path,
) -> None:
    """``--debug`` re-raises the underlying exception for traceback debugging."""

    def failing_fetcher() -> TaimanRawSnapshot:
        raise TaimanNetworkError("boom", url=RECOMMEND_URL)

    with pytest.raises(TaimanNetworkError):
        main(
            ["refresh", "--cache-dir", str(tmp_path), "--debug"],
            fetcher=failing_fetcher,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )


# ---------------------------------------------------------------------------
# Usage / argparse paths
# ---------------------------------------------------------------------------


def test_bare_invocation_returns_usage_exit_code() -> None:
    """``gblh`` with no subcommand returns the argparse usage code (2)."""

    # argparse writes its usage error to *its own* stderr (not the
    # stream we pass into ``cmd_refresh``), so we capture it via the
    # ``capsys`` fixture-free path: ``main`` translates SystemExit to int.
    exit_code = main(
        [],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert exit_code == EXIT_USAGE


def test_filename_override_lands_at_exact_path(tmp_path: Path) -> None:
    """``--filename latest.json`` writes the snapshot at that exact name."""

    def stub_fetcher() -> TaimanRawSnapshot:
        return _fixture_raw_snapshot()

    exit_code = main(
        [
            "refresh",
            "--cache-dir",
            str(tmp_path),
            "--filename",
            "latest.json",
        ],
        fetcher=stub_fetcher,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    target = tmp_path / "latest.json"
    assert target.exists()
    # And it's a valid snapshot file.
    reloaded = read_snapshot(target)
    assert reloaded.league == "great_league"
    assert reloaded.source_caveat == TAIMAN_SOURCE_CAVEAT


def test_refresh_writes_schema_versioned_payload(tmp_path: Path) -> None:
    """The persisted file carries the current schema_version on disk."""

    def stub_fetcher() -> TaimanRawSnapshot:
        return _fixture_raw_snapshot()

    exit_code = main(
        ["refresh", "--cache-dir", str(tmp_path)],
        fetcher=stub_fetcher,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    files = list_snapshots(tmp_path)
    assert len(files) == 1
    import json as _json

    data = _json.loads(files[0].read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
