"""Rating-log entry data model — the long-loop validation feedback unit.

Sub-AC 7.1 contract: a single ``RatingLogEntry`` captures one real-life
GBL run of a recommended team — the team identifier, the rating before
and after, when it happened, and optional operator notes. The engine's
long-loop validation feedback (exit_conditions.long_loop_validation in
seed.yaml) is the persisted append-only series of these entries; later
sub-ACs add the on-disk store and the CLI hookup. This module exposes
the data model and its JSON ser/de surface only.

Design choices and why they exist
---------------------------------

* **Immutable, frozen dataclass.** Once a real-life run has been
  logged, its rating delta is history — the engine must never mutate
  it. To fix a typo'd entry, construct a new value rather than editing
  in place.

* **``delta`` is derived, not stored.** ``delta == post_rating -
  pre_rating`` is a pure function of the other fields. Storing it as
  an independent field invites silent drift — a hand-edited log
  containing ``pre=2400, post=2425, delta=+30`` would be a lie that
  the engine would happily ingest. Exposing ``delta`` as a property
  preserves the invariant by construction. The JSON encoder still
  emits it for human readers and for analytics tools that scan the
  log directly; the decoder cross-checks it for consistency on read.

* **Timestamps normalized to UTC.** The fetch layer writes UTC and
  the snapshot persist layer normalizes naive datetimes to UTC; this
  module follows the same convention so a rating log is comparable
  across sessions without timezone surprises.

* **Validation is strict, but actionable.** Each field that can be
  malformed raises a precise error message. ``team_id`` must be
  non-empty (whitespace doesn't count). Ratings must be ``int`` (bools
  are explicitly rejected because Python's ``True == 1`` is a footgun
  in this domain) and ``>= 0``. ``notes`` is either ``None`` or a
  string — an empty string is allowed because the operator may
  deliberately leave the field blank.

* **Schema-versioned JSON.** The on-disk shape carries a
  ``schema_version`` field, mirroring ``persist.snapshot``. A reader
  that meets an unknown version fails fast rather than silently
  re-interpreting bytes from a future writer.

The module is import-cycle safe: it depends only on the standard
library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


SCHEMA_VERSION: Final[int] = 1
"""Current on-disk schema version for a serialized ``RatingLogEntry``.

Increment when the JSON payload structure changes in a way that an
older reader cannot interpret. A reader that encounters a newer
``schema_version`` raises ``RatingLogSchemaError`` rather than guessing.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RatingLogError(Exception):
    """Base class for rating-log domain errors.

    Distinct from the snapshot persistence errors so a caller can
    handle "the rating-log feedback path failed" separately from "the
    meta snapshot cache failed" — the two failure modes have different
    user remediations (re-enter the rating change vs. re-fetch).
    """


class RatingLogValidationError(RatingLogError, ValueError):
    """Raised when a field of a ``RatingLogEntry`` fails validation.

    Subclasses ``ValueError`` so callers that already catch
    ``ValueError`` from a dataclass ``__post_init__`` (the convention
    elsewhere in this codebase — see ``parse.taiman``) continue to
    work.
    """


class RatingLogSchemaError(RatingLogError):
    """Raised when a serialized payload's ``schema_version`` is unusable.

    Covers three distinct failures, all of which mean "this payload is
    not one this build can interpret":

    * ``schema_version`` is missing.
    * ``schema_version`` is not an ``int``.
    * ``schema_version`` is a value this build does not write/read.

    Carries the offending version so the caller can log it and decide
    whether to upgrade the engine.
    """

    def __init__(self, message: str, *, found_version: Any | None = None) -> None:
        super().__init__(message)
        self.found_version = found_version


class RatingLogDecodeError(RatingLogError):
    """Raised when a serialized rating-log payload is structurally bad.

    Distinguishes "payload is corrupt" (this exception) from "payload
    is for a different schema" (``RatingLogSchemaError``) — the
    operator's response differs: corruption → re-enter the entry;
    schema mismatch → upgrade the engine.
    """


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RatingLogEntry:
    """One reported real-life GBL run of a recommended team.

    Attributes
    ----------
    team_id:
        Stable identifier of the recommended team that was run. Free-
        form string by design — the rating log is engine-agnostic
        about how teams are named (the engine output's team-id format
        may change between versions; old log lines must keep parsing).
        Must be non-empty and not just whitespace.
    pre_rating:
        Operator's GBL rating *before* the run began. Non-negative
        integer.
    post_rating:
        Operator's GBL rating *after* the run ended. Non-negative
        integer. May be greater than, equal to, or less than
        ``pre_rating`` — all three are valid outcomes that the engine
        wants to learn from.
    timestamp:
        Moment the run finished, as a tz-aware datetime. Naive
        datetimes are normalized to UTC on construction (consistent
        with the snapshot persistence layer's convention).
    notes:
        Free-form operator notes — e.g., "lost lead matchup, closer
        carried 2-0". ``None`` means no notes were provided; the empty
        string is also valid and is treated as "operator left it
        blank" (not "operator wrote whitespace").

    Notes
    -----
    ``delta`` is exposed as a property — see the module docstring for
    why it is computed rather than stored.
    """

    team_id: str
    pre_rating: int
    post_rating: int
    timestamp: datetime
    notes: str | None = None

    def __post_init__(self) -> None:
        # team_id ----------------------------------------------------------
        if not isinstance(self.team_id, str):
            raise RatingLogValidationError(
                f"team_id must be str, got {type(self.team_id).__name__}"
            )
        if not self.team_id.strip():
            raise RatingLogValidationError(
                "team_id must be non-empty and not just whitespace"
            )

        # pre_rating / post_rating ----------------------------------------
        for field_name, value in (
            ("pre_rating", self.pre_rating),
            ("post_rating", self.post_rating),
        ):
            # Bool is a subclass of int in Python — reject it explicitly
            # because True == 1 is a footgun in a numeric domain like
            # rating tracking.
            if isinstance(value, bool) or not isinstance(value, int):
                raise RatingLogValidationError(
                    f"{field_name} must be int, got {type(value).__name__}"
                )
            if value < 0:
                raise RatingLogValidationError(
                    f"{field_name} must be >= 0, got {value}"
                )

        # timestamp -------------------------------------------------------
        if not isinstance(self.timestamp, datetime):
            raise RatingLogValidationError(
                f"timestamp must be datetime, got {type(self.timestamp).__name__}"
            )
        if self.timestamp.tzinfo is None:
            # Frozen dataclass: bypass the immutability guard once,
            # only to coerce naive → UTC. After this assignment the
            # instance is sealed for the rest of its lifetime.
            object.__setattr__(
                self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc)
            )

        # notes -----------------------------------------------------------
        if self.notes is not None and not isinstance(self.notes, str):
            raise RatingLogValidationError(
                f"notes must be str or None, got {type(self.notes).__name__}"
            )

    @property
    def delta(self) -> int:
        """Rating change for this run: ``post_rating - pre_rating``.

        Positive means a winning session, zero means a wash, negative
        means a loss. Computed on every access so it cannot drift away
        from the underlying pre/post values.
        """

        return self.post_rating - self.pre_rating


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def entry_to_dict(entry: RatingLogEntry) -> dict[str, Any]:
    """Convert a ``RatingLogEntry`` to a JSON-serializable dict.

    The returned dict carries the ``schema_version`` so a reader can
    reject unknown versions. ``delta`` is included even though it is
    derived — log-scanning tools that consume the file directly
    benefit from not having to recompute it per line.
    """

    if not isinstance(entry, RatingLogEntry):
        raise TypeError(
            f"entry_to_dict requires RatingLogEntry, got {type(entry).__name__}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "team_id": entry.team_id,
        "pre_rating": entry.pre_rating,
        "post_rating": entry.post_rating,
        "delta": entry.delta,
        "timestamp": _to_utc(entry.timestamp).isoformat(),
        "notes": entry.notes,
    }


def entry_from_dict(data: dict[str, Any]) -> RatingLogEntry:
    """Rebuild a ``RatingLogEntry`` from a previously-serialized dict.

    Parameters
    ----------
    data:
        The dict produced by ``entry_to_dict`` (or read from a JSON
        file written by an upcoming rating-log store sub-AC).

    Raises
    ------
    RatingLogSchemaError
        ``schema_version`` is missing, non-integer, or unsupported.
    RatingLogDecodeError
        A required field is missing, has the wrong type, or carries
        an inconsistent ``delta``.
    RatingLogValidationError
        The decoded values are well-typed but violate the
        ``RatingLogEntry`` domain constraints (e.g. negative rating).
    """

    if not isinstance(data, dict):
        raise RatingLogDecodeError(
            f"rating-log payload must be a JSON object, got {type(data).__name__}"
        )

    version = data.get("schema_version")
    if version is None:
        raise RatingLogSchemaError(
            "rating-log payload is missing 'schema_version'",
            found_version=None,
        )
    if isinstance(version, bool) or not isinstance(version, int):
        raise RatingLogSchemaError(
            f"rating-log 'schema_version' must be int, got {type(version).__name__}",
            found_version=version,
        )
    if version != SCHEMA_VERSION:
        raise RatingLogSchemaError(
            (
                f"Unsupported rating-log schema_version={version!r}; "
                f"this build writes/reads version {SCHEMA_VERSION}"
            ),
            found_version=version,
        )

    # Required fields (other than notes, which is optional).
    try:
        team_id = data["team_id"]
        pre_rating = data["pre_rating"]
        post_rating = data["post_rating"]
        timestamp_raw = data["timestamp"]
    except KeyError as exc:
        raise RatingLogDecodeError(
            f"rating-log payload missing required field: {exc}"
        ) from exc

    notes = data.get("notes")

    # Timestamp parse. The dataclass validates that it ends up tz-aware
    # (or coerces naive → UTC), but we still need to parse the string
    # representation here.
    if not isinstance(timestamp_raw, str):
        raise RatingLogDecodeError(
            "rating-log 'timestamp' must be an ISO 8601 string, got "
            f"{type(timestamp_raw).__name__}"
        )
    try:
        timestamp = datetime.fromisoformat(timestamp_raw)
    except ValueError as exc:
        raise RatingLogDecodeError(
            f"rating-log 'timestamp' is not a valid ISO 8601 string: {timestamp_raw!r}"
        ) from exc

    # Defer domain validation to the dataclass __post_init__ — single
    # source of truth for the field invariants.
    try:
        entry = RatingLogEntry(
            team_id=team_id,
            pre_rating=pre_rating,
            post_rating=post_rating,
            timestamp=timestamp,
            notes=notes,
        )
    except RatingLogValidationError:
        # Already a precise domain error — preserve it.
        raise
    except (TypeError, ValueError) as exc:
        raise RatingLogDecodeError(
            f"rating-log payload failed validation: {exc}"
        ) from exc

    # If the payload carried an explicit ``delta``, cross-check it
    # against the derived value. Catches hand-edited inconsistencies
    # before they corrupt downstream analytics.
    if "delta" in data:
        encoded_delta = data["delta"]
        if isinstance(encoded_delta, bool) or not isinstance(encoded_delta, int):
            raise RatingLogDecodeError(
                f"rating-log 'delta' must be int, got {type(encoded_delta).__name__}"
            )
        if encoded_delta != entry.delta:
            raise RatingLogDecodeError(
                (
                    f"rating-log 'delta' is inconsistent with pre/post: "
                    f"payload says {encoded_delta}, computed {entry.delta}"
                )
            )

    return entry


def entry_to_json(entry: RatingLogEntry, *, indent: int | None = None) -> str:
    """Render a ``RatingLogEntry`` as a JSON string.

    Uses ``sort_keys=True`` so the output is byte-stable for tests and
    diffs. ``ensure_ascii=False`` keeps non-ASCII notes readable in the
    log file (the user has Korean reporting enabled — see project
    memory).
    """

    return json.dumps(
        entry_to_dict(entry),
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
    )


def entry_from_json(raw: str) -> RatingLogEntry:
    """Parse a JSON string into a ``RatingLogEntry``.

    Raises
    ------
    RatingLogDecodeError
        The string is not valid JSON, or its top-level value is not a
        JSON object.
    RatingLogSchemaError
        ``schema_version`` is missing or unsupported.
    RatingLogValidationError
        Fields decode but fail domain validation.
    """

    if not isinstance(raw, (str, bytes, bytearray)):
        raise RatingLogDecodeError(
            f"entry_from_json requires str/bytes, got {type(raw).__name__}"
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RatingLogDecodeError(
            f"rating-log payload is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RatingLogDecodeError(
            f"rating-log JSON root must be an object, got {type(data).__name__}"
        )

    return entry_from_dict(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_utc(dt: datetime) -> datetime:
    """Return ``dt`` as a tz-aware UTC datetime.

    Naive datetimes are interpreted as UTC (matches the project-wide
    convention established by ``persist.snapshot._normalize_to_utc``).
    Aware datetimes are converted in-place.
    """

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


__all__ = [
    "RatingLogDecodeError",
    "RatingLogEntry",
    "RatingLogError",
    "RatingLogSchemaError",
    "RatingLogValidationError",
    "SCHEMA_VERSION",
    "entry_from_dict",
    "entry_from_json",
    "entry_to_dict",
    "entry_to_json",
]
