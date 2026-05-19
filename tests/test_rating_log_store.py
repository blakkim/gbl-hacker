"""Tests for ``gbl_hacker.rating_log.store``.

Sub-AC 7.2 contract: the JSONL rating-log store appends entries to a
file and reads them back in insertion order, with sensible behavior on
an empty / missing store. The suite is organized into:

1. empty-store behavior — missing file, empty file, whitespace-only
2. single append + round-trip
3. multi-append round-trip preserves insertion order
4. unicode / negative-delta / None-notes round-trip
5. write-side guards (non-entry argument, parent-dir creation)
6. read-side strictness (corrupt line, line-numbered error, blank-line
   tolerance, EOF without trailing newline)
7. ``count_entries`` convenience
8. ``append_entries`` bulk helper

The tests use ``tmp_path`` exclusively — the store never touches the
operator's real cache directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gbl_hacker.rating_log import (
    DEFAULT_STORE_FILENAME,
    RatingLogDecodeError,
    RatingLogEntry,
    append_entries,
    append_entry,
    count_entries,
    entry_to_json,
    read_entries,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _entry(
    *,
    team_id: str = "azu-anni-regi",
    pre_rating: int = 2400,
    post_rating: int = 2425,
    timestamp: datetime | None = None,
    notes: str | None = "won 3-2",
) -> RatingLogEntry:
    """Build a fully-populated entry. Mirrors the helper in the entry suite."""

    return RatingLogEntry(
        team_id=team_id,
        pre_rating=pre_rating,
        post_rating=post_rating,
        timestamp=timestamp or datetime(2026, 5, 13, 21, 30, 0, tzinfo=timezone.utc),
        notes=notes,
    )


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """A non-existent JSONL store path under ``tmp_path``.

    Returns ``tmp_path / "rating_log.jsonl"`` — the file does NOT exist
    yet, which is the canonical "fresh install" precondition for the
    empty-store tests.
    """

    return tmp_path / "rating_log.jsonl"


# ---------------------------------------------------------------------------
# 1. empty-store behavior
# ---------------------------------------------------------------------------


def test_read_entries_returns_empty_for_missing_file(store_path: Path) -> None:
    """A missing store file reads as an empty list, not an error.

    Fresh engine installs and brand-new caches must not raise — the
    CLI 'show me my rating history' code path calls ``read_entries``
    unconditionally on startup.
    """

    assert not store_path.exists()
    assert read_entries(store_path) == []


def test_read_entries_returns_empty_for_empty_file(store_path: Path) -> None:
    """A 0-byte file reads as an empty list.

    Distinct from "missing" because an operator may have manually
    created an empty file (e.g., ``touch rating_log.jsonl``) to mark
    intent before running their first GBL set.
    """

    store_path.touch()
    assert store_path.stat().st_size == 0
    assert read_entries(store_path) == []


def test_read_entries_returns_empty_for_whitespace_only_file(
    store_path: Path,
) -> None:
    """A file containing only blank lines reads as an empty list.

    Blank lines are valid JSONL — they are the separator convention.
    A file of nothing-but-separators carries zero records.
    """

    store_path.write_text("\n\n   \n\t\n\n", encoding="utf-8")
    assert read_entries(store_path) == []


def test_count_entries_zero_for_missing_file(store_path: Path) -> None:
    """``count_entries`` mirrors ``read_entries`` on the empty-store edge."""

    assert count_entries(store_path) == 0


def test_count_entries_zero_for_empty_file(store_path: Path) -> None:
    """``count_entries`` returns zero for a 0-byte file."""

    store_path.touch()
    assert count_entries(store_path) == 0


# ---------------------------------------------------------------------------
# 2. single append + round-trip
# ---------------------------------------------------------------------------


def test_append_entry_creates_file_with_one_jsonl_line(store_path: Path) -> None:
    """First append creates the file and writes exactly one terminated line."""

    entry = _entry()
    append_entry(entry, path=store_path)

    assert store_path.exists()
    raw = store_path.read_text(encoding="utf-8")
    # The file ends in a newline and contains exactly one record line.
    assert raw.endswith("\n")
    lines = [line for line in raw.split("\n") if line]
    assert len(lines) == 1


def test_append_entry_creates_missing_parent_directories(tmp_path: Path) -> None:
    """A store path under nested missing dirs is created on first append.

    The CLI may resolve the default path to
    ``~/.cache/gbl-hacker/rating_log.jsonl`` on a fresh install where
    neither ``.cache`` nor ``gbl-hacker`` exists. The store must not
    force every caller to mkdir defensively.
    """

    nested = tmp_path / "a" / "b" / "c" / "rating_log.jsonl"
    assert not nested.parent.exists()

    append_entry(_entry(), path=nested)

    assert nested.exists()
    assert nested.parent.is_dir()


def test_single_append_round_trips_equal(store_path: Path) -> None:
    """``read_entries`` returns the entry that was appended, unchanged."""

    original = _entry()
    append_entry(original, path=store_path)

    reloaded = read_entries(store_path)
    assert reloaded == [original]


def test_appended_line_is_decodable_as_entry_json(store_path: Path) -> None:
    """The line written to disk matches ``entry_to_json`` byte-for-byte.

    Locks the on-disk format to the entry module's ser/de — any change
    to the JSON shape goes through the entry module, not the store.
    """

    original = _entry()
    append_entry(original, path=store_path)

    raw = store_path.read_text(encoding="utf-8")
    assert raw == entry_to_json(original) + "\n"


# ---------------------------------------------------------------------------
# 3. multi-append round-trip preserves insertion order
# ---------------------------------------------------------------------------


def test_multiple_appends_preserve_insertion_order(store_path: Path) -> None:
    """Reading back N appends yields exactly the N entries, in order."""

    entries = [
        _entry(
            team_id=f"team-{i}",
            pre_rating=1000 + i * 10,
            post_rating=1000 + (i + 1) * 10,
            timestamp=datetime(2026, 5, 13, 12, i, 0, tzinfo=timezone.utc),
            notes=f"run #{i}",
        )
        for i in range(5)
    ]
    for entry in entries:
        append_entry(entry, path=store_path)

    reloaded = read_entries(store_path)
    assert reloaded == entries


def test_append_after_read_continues_to_grow_file(store_path: Path) -> None:
    """The store is true append-only — a later append extends, not replaces."""

    first = _entry(team_id="team-1")
    append_entry(first, path=store_path)
    assert read_entries(store_path) == [first]

    second = _entry(team_id="team-2", post_rating=2450)
    append_entry(second, path=store_path)

    reloaded = read_entries(store_path)
    assert reloaded == [first, second]


def test_duplicate_entries_are_both_stored(store_path: Path) -> None:
    """The store does not deduplicate — two identical sessions are valid history.

    The operator may legitimately run the same team twice in a single
    evening with identical pre/post ratings; the store must not pretend
    only one happened.
    """

    entry = _entry()
    append_entry(entry, path=store_path)
    append_entry(entry, path=store_path)

    reloaded = read_entries(store_path)
    assert reloaded == [entry, entry]
    assert count_entries(store_path) == 2


# ---------------------------------------------------------------------------
# 4. unicode / negative-delta / None-notes round-trip
# ---------------------------------------------------------------------------


def test_unicode_notes_round_trip(store_path: Path) -> None:
    """Korean notes survive append + read without escaping or mojibake."""

    entry = _entry(notes="레지스틸 클로저로 3-2 승")
    append_entry(entry, path=store_path)

    reloaded = read_entries(store_path)
    assert reloaded == [entry]
    assert reloaded[0].notes == "레지스틸 클로저로 3-2 승"


def test_negative_delta_round_trip(store_path: Path) -> None:
    """A losing session (negative delta) round-trips intact."""

    entry = _entry(pre_rating=2500, post_rating=2410)
    assert entry.delta == -90
    append_entry(entry, path=store_path)

    reloaded = read_entries(store_path)
    assert reloaded == [entry]
    assert reloaded[0].delta == -90


def test_none_notes_round_trip(store_path: Path) -> None:
    """``None`` notes survive the JSONL round-trip as ``None`` (not "")."""

    entry = _entry(notes=None)
    append_entry(entry, path=store_path)

    reloaded = read_entries(store_path)
    assert reloaded == [entry]
    assert reloaded[0].notes is None


def test_empty_string_notes_round_trip(store_path: Path) -> None:
    """Empty-string notes are preserved as empty-string, distinct from None."""

    entry = _entry(notes="")
    append_entry(entry, path=store_path)

    reloaded = read_entries(store_path)
    assert reloaded[0].notes == ""


# ---------------------------------------------------------------------------
# 5. write-side guards
# ---------------------------------------------------------------------------


def test_append_entry_rejects_non_entry_argument(store_path: Path) -> None:
    """``append_entry`` requires a ``RatingLogEntry`` — raw dicts rejected.

    Validation lives in the dataclass; admitting raw dicts would
    bypass it and let malformed records reach disk.
    """

    with pytest.raises(TypeError, match="RatingLogEntry"):
        append_entry({"team_id": "t1"}, path=store_path)  # type: ignore[arg-type]


def test_append_entry_does_not_create_file_when_argument_invalid(
    store_path: Path,
) -> None:
    """A rejected argument must not have side-effects on disk."""

    with pytest.raises(TypeError):
        append_entry("not-an-entry", path=store_path)  # type: ignore[arg-type]

    # The file must not exist — the type check happens before any I/O.
    # If a future refactor accidentally moves the mkdir before the
    # type guard, this test will catch it.
    assert not store_path.exists()


# ---------------------------------------------------------------------------
# 6. read-side strictness
# ---------------------------------------------------------------------------


def test_read_entries_propagates_corrupt_line_with_line_number(
    store_path: Path,
) -> None:
    """A non-blank line that fails to decode raises with the line number.

    Wraps as ``RatingLogDecodeError`` regardless of the underlying
    cause (decode / schema / validation) — at the store level, "any
    bad line" means "this file is not safely consumable".
    """

    # Write one valid entry, then a junk line.
    append_entry(_entry(), path=store_path)
    with open(store_path, "a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")

    with pytest.raises(RatingLogDecodeError, match="line 2"):
        read_entries(store_path)


def test_read_entries_wraps_schema_version_mismatch_with_line_number(
    store_path: Path,
) -> None:
    """A schema-version mismatch on a line surfaces as a store decode error."""

    # Hand-craft a line carrying an impossible future schema version.
    bad_line = (
        '{"delta": 25, "notes": null, "post_rating": 2425, '
        '"pre_rating": 2400, "schema_version": 999, "team_id": "t1", '
        '"timestamp": "2026-05-13T21:30:00+00:00"}'
    )
    store_path.write_text(bad_line + "\n", encoding="utf-8")

    with pytest.raises(RatingLogDecodeError, match="line 1") as excinfo:
        read_entries(store_path)
    # The underlying cause is preserved for debuggability.
    assert excinfo.value.__cause__ is not None


def test_read_entries_skips_blank_lines_between_records(
    store_path: Path,
) -> None:
    """Blank lines between records do not affect entry count or order.

    JSONL convention permits arbitrary blank separators. The store
    writer does not produce them, but a human editor / version-control
    merge might, and the reader must tolerate them.
    """

    first = _entry(team_id="team-1")
    second = _entry(team_id="team-2")
    raw = entry_to_json(first) + "\n\n   \n" + entry_to_json(second) + "\n"
    store_path.write_text(raw, encoding="utf-8")

    reloaded = read_entries(store_path)
    assert reloaded == [first, second]


def test_read_entries_tolerates_missing_trailing_newline(
    store_path: Path,
) -> None:
    """A file whose last line is not newline-terminated still reads cleanly.

    Some editors strip the final newline. The reader should not lose
    the last record.
    """

    entry = _entry()
    # Write WITHOUT a trailing newline.
    store_path.write_text(entry_to_json(entry), encoding="utf-8")
    assert not store_path.read_text(encoding="utf-8").endswith("\n")

    reloaded = read_entries(store_path)
    assert reloaded == [entry]


def test_read_entries_line_number_is_one_indexed(store_path: Path) -> None:
    """The line number in a decode error matches what a human sees in ``less``."""

    # First two lines are valid; third is corrupt.
    append_entry(_entry(team_id="t1"), path=store_path)
    append_entry(_entry(team_id="t2"), path=store_path)
    with open(store_path, "a", encoding="utf-8") as fh:
        fh.write("garbage\n")

    with pytest.raises(RatingLogDecodeError, match="line 3"):
        read_entries(store_path)


# ---------------------------------------------------------------------------
# 7. count_entries convenience
# ---------------------------------------------------------------------------


def test_count_entries_matches_read_entries_length(store_path: Path) -> None:
    """``count_entries`` is consistent with ``len(read_entries(...))``."""

    for i in range(7):
        append_entry(_entry(team_id=f"team-{i}"), path=store_path)

    assert count_entries(store_path) == 7
    assert count_entries(store_path) == len(read_entries(store_path))


def test_count_entries_ignores_blank_lines(store_path: Path) -> None:
    """Blank lines don't inflate the count."""

    entry = _entry()
    raw = "\n" + entry_to_json(entry) + "\n\n\n" + entry_to_json(entry) + "\n\n"
    store_path.write_text(raw, encoding="utf-8")

    assert count_entries(store_path) == 2


# ---------------------------------------------------------------------------
# 8. append_entries bulk helper
# ---------------------------------------------------------------------------


def test_append_entries_writes_each_entry_in_order(store_path: Path) -> None:
    """The bulk helper writes the same lines as a loop of ``append_entry``."""

    entries = [_entry(team_id=f"team-{i}") for i in range(4)]
    written = append_entries(entries, path=store_path)

    assert written == 4
    assert read_entries(store_path) == entries


def test_append_entries_empty_iterable_is_a_noop(store_path: Path) -> None:
    """No entries → no file. The store remains in its empty state."""

    written = append_entries([], path=store_path)

    assert written == 0
    # The store helper should not create an empty file when nothing
    # was written — fewer surprises for tools that ``stat`` the file.
    assert not store_path.exists() or store_path.read_text(encoding="utf-8") == ""


def test_append_entries_chains_with_existing_appends(store_path: Path) -> None:
    """Bulk-append after single-append continues the JSONL stream."""

    first = _entry(team_id="team-0")
    append_entry(first, path=store_path)

    rest = [_entry(team_id=f"team-{i}") for i in range(1, 4)]
    append_entries(rest, path=store_path)

    assert read_entries(store_path) == [first, *rest]


# ---------------------------------------------------------------------------
# 9. DEFAULT_STORE_FILENAME constant
# ---------------------------------------------------------------------------


def test_default_store_filename_is_jsonl(tmp_path: Path) -> None:
    """The constant has the ``.jsonl`` suffix and is non-empty.

    Downstream CLI code resolves the rating-log path as
    ``cache_dir / DEFAULT_STORE_FILENAME``; this test guards the
    constant against accidental drift to ``.json`` (pretty-printed) or
    ``.txt`` (which would confuse tooling).
    """

    assert DEFAULT_STORE_FILENAME
    assert DEFAULT_STORE_FILENAME.endswith(".jsonl")
    # Sanity: usable as a real filename component without escaping.
    (tmp_path / DEFAULT_STORE_FILENAME).touch()
