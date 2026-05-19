"""Tests for the ``gblh report-rating`` CLI subcommand.

Sub-AC 7.3 contract: a CLI subcommand accepts a team-id and pre/post-
rating arguments, constructs a :class:`gbl_hacker.rating_log.RatingLogEntry`,
and writes it via the rating-log store. The headline test invokes the
CLI runner against a temp store and asserts the resulting JSONL log
contents match what was passed on the command line.

Additional guard tests pin the surrounding contract:

    * The store path resolution honors ``--rating-log-path``,
      ``--cache-dir`` (joined with ``DEFAULT_STORE_FILENAME``), and the
      OS default fallback — in that precedence order.
    * ``--timestamp`` is parsed as ISO 8601 and used verbatim; default
      timestamp comes from the injected ``now`` clock.
    * Validation failures (negative rating, blank team-id, malformed
      timestamp) map to ``EXIT_RATING_LOG`` with a stderr explanation —
      no traceback unless ``--debug`` is set.
    * Argparse failures (missing required ``--team-id``, non-int ``--pre``)
      yield ``EXIT_USAGE``.
    * The writer DI seam receives the exact ``RatingLogEntry`` instance
      built by the CLI (sentinel identity, not just structural equality).
    * Korean ``--notes`` survive the round-trip through the JSONL file
      byte-for-byte.

The tests are fully offline and never touch the operator's real cache
directory — every store path lives under ``tmp_path``.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gbl_hacker.cli import (
    EXIT_OK,
    EXIT_RATING_LOG,
    EXIT_USAGE,
    main,
)
from gbl_hacker.rating_log import (
    DEFAULT_STORE_FILENAME,
    RatingLogEntry,
    RatingLogValidationError,
    read_entries,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


FIXED_NOW = datetime(2026, 5, 13, 21, 30, 0, tzinfo=timezone.utc)
"""A deterministic 'wall-clock now' for tests that don't pass --timestamp."""


def _fixed_clock() -> datetime:
    """Inject a frozen UTC clock so default-timestamp tests are deterministic."""

    return FIXED_NOW


# ---------------------------------------------------------------------------
# Sub-AC 7.3 central assertion
# ---------------------------------------------------------------------------


def test_report_rating_writes_entry_to_temp_store(tmp_path: Path) -> None:
    """The headline assertion: the JSONL store contains the reported entry.

    Invokes ``gblh report-rating`` against a temp rating-log path, then
    reads the file back via ``rating_log.read_entries`` and asserts the
    decoded entry matches every field that was supplied on the command
    line (with the timestamp resolved by the injected clock).
    """

    store = tmp_path / DEFAULT_STORE_FILENAME
    assert not store.exists()

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "azu-anni-regi",
            "--pre",
            "2400",
            "--post",
            "2425",
            "--notes",
            "won 3-2, closer carried",
            "--rating-log-path",
            str(store),
        ],
        now=_fixed_clock,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == EXIT_OK, (
        f"expected exit 0; got {exit_code}. stderr={stderr.getvalue()!r}"
    )

    # The file actually landed on disk.
    assert store.exists(), "rating-log store file was not created"

    # The JSONL file holds exactly one entry matching the CLI arguments.
    entries = read_entries(store)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.team_id == "azu-anni-regi"
    assert entry.pre_rating == 2400
    assert entry.post_rating == 2425
    assert entry.delta == 25
    assert entry.notes == "won 3-2, closer carried"
    assert entry.timestamp == FIXED_NOW

    # And the stdout confirmation mentions the entry's fields so the
    # operator can sanity-check what landed.
    out = stdout.getvalue()
    assert "azu-anni-regi" in out
    assert "2400" in out
    assert "2425" in out
    assert "+25" in out  # delta is rendered with a sign


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_report_rating_resolves_store_under_cache_dir(tmp_path: Path) -> None:
    """``--cache-dir`` makes the store land at ``DIR / DEFAULT_STORE_FILENAME``.

    The rating log lives at the cache *root*, not in the ``snapshots/``
    sub-dir. This isolates it from snapshot files when an operator
    inspects the cache by hand.
    """

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-a",
            "--pre",
            "1900",
            "--post",
            "1950",
            "--cache-dir",
            str(tmp_path),
        ],
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    expected_path = tmp_path / DEFAULT_STORE_FILENAME
    assert expected_path.exists()

    entries = read_entries(expected_path)
    assert len(entries) == 1
    assert entries[0].team_id == "team-a"


def test_report_rating_rating_log_path_wins_over_cache_dir(
    tmp_path: Path,
) -> None:
    """``--rating-log-path`` takes precedence when both flags are passed."""

    explicit = tmp_path / "explicit" / "ratings.jsonl"
    cache = tmp_path / "cache"
    cache.mkdir()

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-b",
            "--pre",
            "2000",
            "--post",
            "1990",  # losing session
            "--rating-log-path",
            str(explicit),
            "--cache-dir",
            str(cache),
        ],
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    # The explicit path got the entry…
    assert explicit.exists()
    entries = read_entries(explicit)
    assert len(entries) == 1
    assert entries[0].delta == -10
    # …and the cache-dir-resolved path did NOT get one.
    cache_path = cache / DEFAULT_STORE_FILENAME
    assert not cache_path.exists()


def test_report_rating_creates_missing_parent_directories(tmp_path: Path) -> None:
    """A nested explicit path triggers parent ``mkdir`` from the store layer."""

    nested = tmp_path / "a" / "b" / "c" / "rating_log.jsonl"
    assert not nested.parent.exists()

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-c",
            "--pre",
            "1500",
            "--post",
            "1500",
            "--rating-log-path",
            str(nested),
        ],
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    assert nested.exists()
    assert nested.parent.is_dir()


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------


def test_report_rating_uses_explicit_timestamp_when_provided(
    tmp_path: Path,
) -> None:
    """``--timestamp`` is parsed as ISO 8601 and used verbatim."""

    explicit_ts = "2026-05-12T03:45:00+00:00"
    store = tmp_path / "log.jsonl"

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-d",
            "--pre",
            "2100",
            "--post",
            "2130",
            "--timestamp",
            explicit_ts,
            "--rating-log-path",
            str(store),
        ],
        # Inject a clock that would fail the test if it were used —
        # provides a sharper assertion than "is some-datetime".
        now=lambda: datetime(1999, 1, 1, tzinfo=timezone.utc),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    entries = read_entries(store)
    assert len(entries) == 1
    assert entries[0].timestamp == datetime.fromisoformat(explicit_ts)


def test_report_rating_uses_injected_now_when_timestamp_omitted(
    tmp_path: Path,
) -> None:
    """Without ``--timestamp``, the entry timestamp is whatever ``now`` returns."""

    sentinel_ts = datetime(2030, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    store = tmp_path / "log.jsonl"

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-e",
            "--pre",
            "1800",
            "--post",
            "1820",
            "--rating-log-path",
            str(store),
        ],
        now=lambda: sentinel_ts,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    entries = read_entries(store)
    assert entries[0].timestamp == sentinel_ts


def test_report_rating_rejects_malformed_timestamp(tmp_path: Path) -> None:
    """A non-ISO timestamp string yields ``EXIT_RATING_LOG`` with a clear message."""

    store = tmp_path / "log.jsonl"
    stderr = io.StringIO()

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-f",
            "--pre",
            "2000",
            "--post",
            "2010",
            "--timestamp",
            "not-a-real-timestamp",
            "--rating-log-path",
            str(store),
        ],
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_RATING_LOG
    assert "timestamp" in stderr.getvalue().lower()
    # Nothing was written.
    assert not store.exists()


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_report_rating_rejects_negative_rating(tmp_path: Path) -> None:
    """A negative ``--pre`` surfaces as ``EXIT_RATING_LOG`` (validation failure)."""

    store = tmp_path / "log.jsonl"
    stderr = io.StringIO()

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-g",
            "--pre",
            "-1",
            "--post",
            "100",
            "--rating-log-path",
            str(store),
        ],
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_RATING_LOG
    assert "pre_rating" in stderr.getvalue()
    assert not store.exists()


def test_report_rating_rejects_whitespace_team_id(tmp_path: Path) -> None:
    """A whitespace-only ``--team-id`` triggers a validation error."""

    store = tmp_path / "log.jsonl"
    stderr = io.StringIO()

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "   ",
            "--pre",
            "2000",
            "--post",
            "2010",
            "--rating-log-path",
            str(store),
        ],
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_RATING_LOG
    assert "team_id" in stderr.getvalue()
    assert not store.exists()


def test_report_rating_debug_flag_reraises_validation_error(tmp_path: Path) -> None:
    """``--debug`` re-raises the underlying ``RatingLogValidationError``."""

    store = tmp_path / "log.jsonl"

    with pytest.raises(RatingLogValidationError):
        main(
            [
                "report-rating",
                "--team-id",
                "team-h",
                "--pre",
                "2000",
                "--post",
                "-5",  # invalid
                "--rating-log-path",
                str(store),
                "--debug",
            ],
            now=_fixed_clock,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )


# ---------------------------------------------------------------------------
# Argparse failures
# ---------------------------------------------------------------------------


def test_report_rating_missing_required_team_id_returns_usage_code() -> None:
    """Omitting ``--team-id`` (required) yields ``EXIT_USAGE`` from argparse."""

    exit_code = main(
        ["report-rating", "--pre", "2000", "--post", "2010"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert exit_code == EXIT_USAGE


def test_report_rating_non_int_pre_returns_usage_code() -> None:
    """A non-integer ``--pre`` is rejected by argparse's type=int."""

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-i",
            "--pre",
            "not-a-number",
            "--post",
            "2010",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert exit_code == EXIT_USAGE


# ---------------------------------------------------------------------------
# Writer DI seam — identity assertion
# ---------------------------------------------------------------------------


def test_report_rating_writer_di_receives_constructed_entry(tmp_path: Path) -> None:
    """The injected writer receives the exact RatingLogEntry the CLI built.

    Spies on (entry, path) without writing to disk; the strict assertion
    is that the CLI constructed a ``RatingLogEntry`` whose fields match
    the command-line args and handed it to the writer.
    """

    captured: dict[str, Any] = {}

    def spy_writer(entry: RatingLogEntry, path: Path) -> None:
        captured["entry"] = entry
        captured["path"] = path

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "spy-team",
            "--pre",
            "2200",
            "--post",
            "2250",
            "--notes",
            "spied",
            "--rating-log-path",
            str(tmp_path / "spy.jsonl"),
        ],
        rating_log_writer=spy_writer,
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    assert "entry" in captured, "writer was not invoked"
    entry = captured["entry"]
    assert isinstance(entry, RatingLogEntry)
    assert entry.team_id == "spy-team"
    assert entry.pre_rating == 2200
    assert entry.post_rating == 2250
    assert entry.notes == "spied"
    assert entry.timestamp == FIXED_NOW
    assert captured["path"] == tmp_path / "spy.jsonl"

    # The spy did NOT call through to the store, so nothing on disk.
    assert not (tmp_path / "spy.jsonl").exists()


def test_report_rating_writer_failure_returns_rating_log_exit(
    tmp_path: Path,
) -> None:
    """A writer that raises ``OSError`` surfaces as ``EXIT_RATING_LOG``."""

    def failing_writer(_entry: RatingLogEntry, _path: Path) -> None:
        raise OSError("simulated disk-full failure")

    stderr = io.StringIO()
    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-j",
            "--pre",
            "1500",
            "--post",
            "1500",
            "--rating-log-path",
            str(tmp_path / "log.jsonl"),
        ],
        rating_log_writer=failing_writer,
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == EXIT_RATING_LOG
    assert "write failed" in stderr.getvalue().lower()


# ---------------------------------------------------------------------------
# Round-trip / append-on-existing-store
# ---------------------------------------------------------------------------


def test_report_rating_appends_to_existing_store_in_order(tmp_path: Path) -> None:
    """Two consecutive invocations append two entries to the same JSONL file."""

    store = tmp_path / DEFAULT_STORE_FILENAME

    rc1 = main(
        [
            "report-rating",
            "--team-id",
            "team-1",
            "--pre",
            "2000",
            "--post",
            "2020",
            "--rating-log-path",
            str(store),
        ],
        now=lambda: datetime(2026, 5, 13, 20, 0, 0, tzinfo=timezone.utc),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc1 == EXIT_OK

    rc2 = main(
        [
            "report-rating",
            "--team-id",
            "team-2",
            "--pre",
            "2020",
            "--post",
            "2005",
            "--rating-log-path",
            str(store),
        ],
        now=lambda: datetime(2026, 5, 13, 21, 0, 0, tzinfo=timezone.utc),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc2 == EXIT_OK

    entries = read_entries(store)
    assert [e.team_id for e in entries] == ["team-1", "team-2"]
    assert [e.delta for e in entries] == [20, -15]


def test_report_rating_preserves_unicode_notes(tmp_path: Path) -> None:
    """Korean ``--notes`` survive the JSONL round-trip without mojibake.

    Pins the data-honesty + Korean-reporting project memory items:
    the operator can write notes in their preferred language and the
    engine preserves them verbatim.
    """

    store = tmp_path / "log.jsonl"

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "한국팀",
            "--pre",
            "2400",
            "--post",
            "2440",
            "--notes",
            "레지스틸 클로저로 3-2 승",
            "--rating-log-path",
            str(store),
        ],
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    entries = read_entries(store)
    assert entries[0].team_id == "한국팀"
    assert entries[0].notes == "레지스틸 클로저로 3-2 승"

    # And the on-disk JSONL line still contains the literal Korean
    # characters (ensure_ascii=False on the entry encoder).
    raw = store.read_text(encoding="utf-8")
    assert "한국팀" in raw
    assert "레지스틸" in raw


def test_report_rating_omitted_notes_yields_none(tmp_path: Path) -> None:
    """Omitting ``--notes`` produces a JSONL line whose ``notes`` is ``null``."""

    store = tmp_path / "log.jsonl"

    exit_code = main(
        [
            "report-rating",
            "--team-id",
            "team-k",
            "--pre",
            "1700",
            "--post",
            "1700",
            "--rating-log-path",
            str(store),
        ],
        now=_fixed_clock,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    entries = read_entries(store)
    assert entries[0].notes is None

    # Round-trip the on-disk JSON line through ``json.loads`` to pin the
    # explicit ``null`` representation (not ``""``, not missing key).
    line = store.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["notes"] is None
