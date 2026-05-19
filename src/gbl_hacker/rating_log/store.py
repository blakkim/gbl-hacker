"""Append-only JSONL store for ``RatingLogEntry`` records.

Sub-AC 7.2 contract: persist a series of rating-log entries as one JSON
object per line and read them back in insertion order. The store is the
on-disk tail of the long-loop validation feedback path
(``exit_conditions.long_loop_validation`` in ``seed.yaml``) — a flat,
human-readable file that survives engine restarts and version bumps.

Design choices and why they exist
---------------------------------

* **JSONL, not SQLite.** The store is single-writer (one operator,
  manually-entered runs), entries are immutable once written, and the
  engine never queries them by anything other than "all of them, in
  insertion order". JSONL satisfies every requirement with no schema
  migrations and no binary format. It is also ``grep``-able, which
  matters for the long-loop debugging workflow — the operator should be
  able to ``cat ~/.cache/gbl-hacker/rating_log.jsonl`` and read history
  with the naked eye.

* **Append-only.** Each ``append_entry`` call opens the file in ``"a"``
  mode and writes exactly one line. POSIX guarantees small appends are
  atomic at the kernel level, so a future multi-process scenario would
  interleave at line boundaries rather than within a line. v0.1 is
  single-writer but this property is free.

* **One JSON object per line.** ``entry_to_json`` with no ``indent``
  argument produces a single-line, ``sort_keys=True``,
  ``ensure_ascii=False`` payload — so a Korean note round-trips byte-
  for-byte without escaping. The reader splits on ``\\n``, skips blank
  lines, and defers per-line decode to ``entry_from_json``.

* **Strict reads, line-numbered errors.** A line that cannot be decoded
  is wrapped as a ``RatingLogDecodeError`` with the offending line
  number, preserving the underlying ``__cause__``. Silently skipping a
  corrupt line would let a typo in the log eat real history — and the
  long-loop validation evaluation principle would be undermined by
  silent data loss.

* **Empty-store affordance.** A missing file and an empty file both
  return ``[]``. The first ``append_entry`` creates the file (and any
  missing parent directories). A fresh engine install can therefore
  call ``read_entries`` before any rating run has been logged without
  raising — which matters for the CLI's ``rating-log show`` path and
  for any startup banner that displays "0 runs recorded".

* **fsync on append.** After each write the file is flushed and
  ``fsync``-d. This is the right trade-off for a manually-triggered,
  low-volume log — durability matters more than throughput when the
  operator is hand-entering data and would be furious to lose an entry
  to a crash between two GBL sets.

The module is import-cycle safe: it depends only on
``rating_log.entry`` (the data model) and the standard library.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from gbl_hacker.rating_log.entry import (
    RatingLogDecodeError,
    RatingLogEntry,
    RatingLogError,
    entry_from_json,
    entry_to_json,
)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


DEFAULT_STORE_FILENAME: Final[str] = "rating_log.jsonl"
"""Default basename for the rating-log store file.

Mirrors ``persist.snapshot.DEFAULT_CACHE_SUBDIR`` in spirit: a single
named constant that downstream CLI / config code resolves under the
cache root, so a caller never has to invent a filename. The ``.jsonl``
suffix is the conventional indicator that the file is line-delimited
JSON (one record per line), distinguishing it from the pretty-printed
``.json`` snapshots written by the persistence layer.
"""


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


def append_entry(entry: RatingLogEntry, *, path: Path) -> None:
    """Append one ``RatingLogEntry`` as a JSONL line to the store at ``path``.

    Parameters
    ----------
    entry:
        The rating-log entry to persist. Must be a ``RatingLogEntry``
        instance — raw dicts are rejected to keep the validation surface
        centralized in the dataclass's ``__post_init__``.
    path:
        Absolute or relative path to the JSONL store file. Created (and
        any missing parent directories created) if it does not exist.

    Raises
    ------
    TypeError
        ``entry`` is not a ``RatingLogEntry``.
    ValueError
        The serialized entry contains an embedded newline — would
        corrupt the JSONL line boundary.
    OSError
        I/O error creating directories or writing the file. Bubbled
        through unchanged so the caller can react to disk-full /
        permission errors directly.
    """

    if not isinstance(entry, RatingLogEntry):
        raise TypeError(
            f"append_entry requires RatingLogEntry, got {type(entry).__name__}"
        )

    # Lazily create the parent directory so a caller pointing at
    # ``~/.cache/gbl-hacker/rating_log.jsonl`` does not have to manage
    # directory creation themselves. ``exist_ok=True`` makes repeated
    # appends a no-op on the directory side.
    path.parent.mkdir(parents=True, exist_ok=True)

    encoded = entry_to_json(entry)
    # ``entry_to_json`` with no indent never embeds a newline at the
    # JSON-structural level — but ``notes`` is free-form, and a literal
    # newline inside notes would be JSON-escaped to ``\n`` and remain
    # safe. Defensively assert anyway: if a future change to
    # ``entry_to_json`` ever produced a multi-line payload, the JSONL
    # line boundary would be silently destroyed. Failing loudly here is
    # cheaper than chasing a corrupted log later.
    if "\n" in encoded:
        raise ValueError(
            "serialized rating-log entry contains an embedded newline; "
            "JSONL requires single-line records"
        )

    # Open in append mode, write one line, flush + fsync. The kernel
    # serializes the seek-to-end + write under the file's append lock
    # on POSIX, so concurrent writers (if ever introduced) interleave
    # at line boundaries instead of mid-line.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(encoded)
        fh.write("\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            # Some filesystems (tmpfs in CI sandboxes) reject ``fsync``
            # on regular files. The Python-level buffer flush above is
            # still enforced, so durability degrades to "buffered" on
            # those systems — acceptable for v0.1 since the CI tmpfs
            # case is not a real operator's disk.
            pass


def append_entries(entries: Iterable[RatingLogEntry], *, path: Path) -> int:
    """Append multiple entries in order; return the count written.

    Convenience for bulk migration / import code paths (e.g. replaying
    a backup into a fresh store). Order is preserved — the iterable is
    consumed left-to-right and each entry goes through the same
    ``append_entry`` code path.
    """

    count = 0
    for entry in entries:
        append_entry(entry, path=path)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


def read_entries(path: Path) -> list[RatingLogEntry]:
    """Read every ``RatingLogEntry`` from the JSONL store at ``path``.

    Returns the entries in insertion order — the order in which
    ``append_entry`` wrote them. A missing file and an empty file both
    return an empty list (see "empty-store affordance" in the module
    docstring). Blank lines anywhere in the file are skipped — they
    are allowed by the JSONL convention and may appear at end-of-file
    after a trailing newline.

    Parameters
    ----------
    path:
        Path to the JSONL store file.

    Returns
    -------
    list[RatingLogEntry]
        All entries in the store, oldest first.

    Raises
    ------
    RatingLogDecodeError
        A non-blank line is not valid JSON, or fails to decode into a
        ``RatingLogEntry``. The exception message carries the offending
        line number; the underlying cause (decode / schema / validation
        error from ``entry_from_json``) is preserved via ``__cause__``.
    OSError
        I/O error reading the file. Bubbled through unchanged.
    """

    if not path.exists():
        return []

    entries: list[RatingLogEntry] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                # JSONL tolerates blank lines as separators / EOF
                # padding. Skip without affecting the entry count.
                continue
            try:
                entries.append(entry_from_json(stripped))
            except RatingLogError as exc:
                # At the store level, any per-entry failure (decode,
                # schema mismatch, domain validation) means "this file
                # is not consumable as a rating log". Wrap as a single
                # store-level decode error so the caller has one
                # exception type to catch, but preserve the specific
                # cause via ``__cause__`` for debuggability.
                raise RatingLogDecodeError(
                    f"rating-log store {path!s} line {lineno}: {exc}"
                ) from exc
    return entries


def count_entries(path: Path) -> int:
    """Return the number of entries currently in the store at ``path``.

    Convenience for the long-loop validation exit-condition check —
    ``exit_conditions.long_loop_validation`` requires at least one
    logged run, and a caller may want to verify that quickly without
    materializing every entry into memory. A missing or empty file
    returns ``0``.

    The implementation reads the file line-by-line and counts
    non-blank lines, so the result is consistent with
    ``len(read_entries(path))`` but does not allocate any entry
    objects. JSON validity is *not* checked here — that is
    ``read_entries``'s job. Use ``read_entries`` when you need the
    parsed objects and want strict validation; use this when you only
    need to know "is there at least one entry yet?".
    """

    if not path.exists():
        return 0

    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            if raw_line.strip():
                count += 1
    return count


__all__ = [
    "DEFAULT_STORE_FILENAME",
    "append_entries",
    "append_entry",
    "count_entries",
    "read_entries",
]
