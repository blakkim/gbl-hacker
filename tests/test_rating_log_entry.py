"""Tests for ``gbl_hacker.rating_log.entry``.

Sub-AC 7.1 contract: the ``RatingLogEntry`` data model validates its
fields, computes ``delta`` from ``post_rating - pre_rating``, and
round-trips through a versioned JSON form. Malformed inputs are
rejected with precise, distinguishable error types.

The suite is grouped into:

1. valid construction
2. delta computation (positive / zero / negative)
3. rejection of malformed inputs (per field)
4. immutability
5. JSON / dict round-trip
6. schema-version / decode error paths
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from gbl_hacker.rating_log import (
    SCHEMA_VERSION,
    RatingLogDecodeError,
    RatingLogEntry,
    RatingLogSchemaError,
    RatingLogValidationError,
    entry_from_dict,
    entry_from_json,
    entry_to_dict,
    entry_to_json,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _sample_entry(
    *,
    team_id: str = "azu-anni-regi",
    pre_rating: int = 2400,
    post_rating: int = 2425,
    timestamp: datetime | None = None,
    notes: str | None = "won 3-2, lead picked up free shields",
) -> RatingLogEntry:
    """Build a fully-populated entry for round-trip / scenario testing."""

    return RatingLogEntry(
        team_id=team_id,
        pre_rating=pre_rating,
        post_rating=post_rating,
        timestamp=timestamp
        or datetime(2026, 5, 13, 21, 30, 0, tzinfo=timezone.utc),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 1. valid construction
# ---------------------------------------------------------------------------


def test_valid_entry_construction_succeeds() -> None:
    """A well-formed entry constructs without raising."""

    entry = _sample_entry()

    assert entry.team_id == "azu-anni-regi"
    assert entry.pre_rating == 2400
    assert entry.post_rating == 2425
    assert entry.timestamp == datetime(2026, 5, 13, 21, 30, 0, tzinfo=timezone.utc)
    assert entry.notes == "won 3-2, lead picked up free shields"


def test_notes_default_to_none() -> None:
    """``notes`` is optional and defaults to ``None``."""

    entry = RatingLogEntry(
        team_id="t1",
        pre_rating=1000,
        post_rating=1000,
        timestamp=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )

    assert entry.notes is None


def test_empty_string_notes_allowed() -> None:
    """An explicitly-empty ``notes`` string is preserved (vs. None)."""

    entry = _sample_entry(notes="")
    assert entry.notes == ""


def test_naive_timestamp_normalized_to_utc() -> None:
    """A naive ``datetime`` is coerced to UTC on construction.

    Matches the project-wide convention from ``persist.snapshot`` —
    naive ISO timestamps are interpreted as UTC rather than rejected.
    """

    entry = RatingLogEntry(
        team_id="t1",
        pre_rating=1500,
        post_rating=1510,
        timestamp=datetime(2026, 5, 13, 21, 30, 0),  # naive
    )

    assert entry.timestamp.tzinfo is timezone.utc
    assert entry.timestamp == datetime(2026, 5, 13, 21, 30, 0, tzinfo=timezone.utc)


def test_aware_non_utc_timestamp_is_preserved() -> None:
    """A tz-aware non-UTC datetime is not converted on construction.

    The on-the-wire serialization step normalizes to UTC; the in-memory
    value keeps its original tz so a caller that passes ``KST`` sees
    ``KST`` back.
    """

    from datetime import timedelta

    kst = timezone(timedelta(hours=9))
    ts = datetime(2026, 5, 14, 6, 30, 0, tzinfo=kst)

    entry = _sample_entry(timestamp=ts)
    assert entry.timestamp.tzinfo == kst
    assert entry.timestamp == ts


def test_team_id_with_internal_whitespace_is_kept_verbatim() -> None:
    """The validator strips for non-emptiness check but never mutates the value."""

    entry = _sample_entry(team_id="team A-B-C")
    assert entry.team_id == "team A-B-C"


# ---------------------------------------------------------------------------
# 2. delta computation
# ---------------------------------------------------------------------------


def test_delta_is_positive_when_post_greater_than_pre() -> None:
    """``delta`` reflects a winning session."""

    entry = _sample_entry(pre_rating=2400, post_rating=2425)
    assert entry.delta == 25


def test_delta_is_zero_when_post_equals_pre() -> None:
    """``delta`` is zero for an exactly even session."""

    entry = _sample_entry(pre_rating=2500, post_rating=2500)
    assert entry.delta == 0


def test_delta_is_negative_when_post_less_than_pre() -> None:
    """``delta`` is negative for a losing session — and that's valid data."""

    entry = _sample_entry(pre_rating=2500, post_rating=2410)
    assert entry.delta == -90


def test_delta_is_int_not_float() -> None:
    """``delta`` preserves int-ness — downstream sums should not coerce to float."""

    entry = _sample_entry(pre_rating=2400, post_rating=2425)
    assert isinstance(entry.delta, int)


def test_delta_recomputes_after_replace() -> None:
    """Constructing a new entry with different ratings recomputes delta.

    Even though entries are frozen, ``dataclasses.replace``-style
    construction (a new instance) produces a fresh delta — confirming
    the property is not memoized in a way that survives the source
    fields changing.
    """

    a = _sample_entry(pre_rating=2400, post_rating=2425)
    b = RatingLogEntry(
        team_id=a.team_id,
        pre_rating=a.pre_rating,
        post_rating=2500,
        timestamp=a.timestamp,
        notes=a.notes,
    )

    assert a.delta == 25
    assert b.delta == 100


# ---------------------------------------------------------------------------
# 3. rejection of malformed inputs — per field
# ---------------------------------------------------------------------------


def test_reject_empty_team_id() -> None:
    """Empty ``team_id`` is rejected at construction time."""

    with pytest.raises(RatingLogValidationError, match="team_id"):
        _sample_entry(team_id="")


def test_reject_whitespace_only_team_id() -> None:
    """A whitespace-only ``team_id`` is treated as empty."""

    with pytest.raises(RatingLogValidationError, match="team_id"):
        _sample_entry(team_id="   \t  ")


def test_reject_non_string_team_id() -> None:
    """``team_id`` must be a string — ints, lists, None all rejected."""

    for bad in (123, None, ["t"], 1.5):
        with pytest.raises(RatingLogValidationError, match="team_id"):
            _sample_entry(team_id=bad)  # type: ignore[arg-type]


def test_reject_non_int_pre_rating() -> None:
    """``pre_rating`` must be an int — floats and strings rejected."""

    for bad in (2400.5, "2400", None):
        with pytest.raises(RatingLogValidationError, match="pre_rating"):
            _sample_entry(pre_rating=bad)  # type: ignore[arg-type]


def test_reject_bool_pre_rating() -> None:
    """Bools are rejected even though they are int subclasses in Python.

    ``True == 1`` is a footgun in a rating-tracking domain — reject
    explicitly so a typo'd boolean does not silently log a 1-point
    rating.
    """

    with pytest.raises(RatingLogValidationError, match="pre_rating"):
        _sample_entry(pre_rating=True)  # type: ignore[arg-type]


def test_reject_negative_pre_rating() -> None:
    """Negative ratings are domain-invalid."""

    with pytest.raises(RatingLogValidationError, match="pre_rating"):
        _sample_entry(pre_rating=-1)


def test_zero_pre_rating_is_accepted() -> None:
    """A literal 0 rating is accepted — the lower bound is inclusive."""

    entry = RatingLogEntry(
        team_id="t1",
        pre_rating=0,
        post_rating=10,
        timestamp=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert entry.pre_rating == 0
    assert entry.delta == 10


def test_reject_non_int_post_rating() -> None:
    """Same int-only rule applies to ``post_rating``."""

    for bad in (2400.0, "2400", [2400]):
        with pytest.raises(RatingLogValidationError, match="post_rating"):
            _sample_entry(post_rating=bad)  # type: ignore[arg-type]


def test_reject_bool_post_rating() -> None:
    """Bool is rejected for ``post_rating`` for the same reason as ``pre_rating``."""

    with pytest.raises(RatingLogValidationError, match="post_rating"):
        _sample_entry(post_rating=False)  # type: ignore[arg-type]


def test_reject_negative_post_rating() -> None:
    """Negative ``post_rating`` is domain-invalid even if delta is plausible."""

    with pytest.raises(RatingLogValidationError, match="post_rating"):
        _sample_entry(pre_rating=10, post_rating=-5)


def test_reject_non_datetime_timestamp() -> None:
    """``timestamp`` must be a ``datetime``; strings and ints rejected.

    Construct ``RatingLogEntry`` directly here rather than via
    ``_sample_entry`` — the helper substitutes a default when the
    timestamp is falsy (``None``), which would mask a real failure.
    """

    for bad in ("2026-05-13", 1_700_000_000, None, 1700000000.0):
        with pytest.raises(RatingLogValidationError, match="timestamp"):
            RatingLogEntry(
                team_id="t1",
                pre_rating=1000,
                post_rating=1000,
                timestamp=bad,  # type: ignore[arg-type]
            )


def test_reject_non_string_notes() -> None:
    """``notes`` must be ``None`` or a string."""

    for bad in (123, ["note"], {"note": "x"}):
        with pytest.raises(RatingLogValidationError, match="notes"):
            _sample_entry(notes=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. immutability
# ---------------------------------------------------------------------------


def test_entry_is_frozen() -> None:
    """The dataclass is frozen — attribute assignment fails after construction."""

    entry = _sample_entry()

    with pytest.raises(FrozenInstanceError):
        entry.pre_rating = 9999  # type: ignore[misc]


def test_entry_is_hashable() -> None:
    """Frozen dataclass is hashable — usable as a dict key or set member."""

    entry = _sample_entry()
    # Just exercising the hash; we don't depend on a specific value.
    assert hash(entry) == hash(entry)
    assert {entry: 1}[entry] == 1


# ---------------------------------------------------------------------------
# 5. dict / JSON round-trip
# ---------------------------------------------------------------------------


def test_entry_to_dict_includes_schema_version_and_delta() -> None:
    """The serialized dict carries the schema version and the derived delta."""

    entry = _sample_entry(pre_rating=2400, post_rating=2480)
    payload = entry_to_dict(entry)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["team_id"] == entry.team_id
    assert payload["pre_rating"] == 2400
    assert payload["post_rating"] == 2480
    assert payload["delta"] == 80  # derived
    assert payload["notes"] == entry.notes
    # Timestamp serialized in ISO 8601, normalized to UTC.
    assert payload["timestamp"] == "2026-05-13T21:30:00+00:00"


def test_round_trip_via_dict_layer_is_equal() -> None:
    """``entry_from_dict(entry_to_dict(e)) == e`` for the sample entry."""

    original = _sample_entry()
    payload = entry_to_dict(original)
    reloaded = entry_from_dict(payload)

    assert reloaded == original
    assert reloaded.delta == original.delta


def test_round_trip_via_json_layer_is_equal() -> None:
    """``entry_from_json(entry_to_json(e)) == e`` for the sample entry."""

    original = _sample_entry()
    encoded = entry_to_json(original)
    # Output must be valid JSON.
    parsed = json.loads(encoded)
    assert parsed["schema_version"] == SCHEMA_VERSION

    reloaded = entry_from_json(encoded)
    assert reloaded == original


def test_round_trip_with_none_notes() -> None:
    """``None`` notes round-trip through JSON as ``null``."""

    original = _sample_entry(notes=None)
    encoded = entry_to_json(original)
    assert '"notes": null' in encoded or '"notes":null' in encoded
    reloaded = entry_from_json(encoded)
    assert reloaded == original
    assert reloaded.notes is None


def test_round_trip_with_unicode_notes() -> None:
    """Non-ASCII notes survive round-trip (Korean reporting preference)."""

    original = _sample_entry(notes="레지스틸 클로저로 3-2 승")
    encoded = entry_to_json(original)
    reloaded = entry_from_json(encoded)
    assert reloaded.notes == original.notes


def test_round_trip_preserves_negative_delta() -> None:
    """A losing session round-trips with its negative delta intact."""

    original = _sample_entry(pre_rating=2500, post_rating=2410)
    assert original.delta == -90

    reloaded = entry_from_dict(entry_to_dict(original))
    assert reloaded == original
    assert reloaded.delta == -90


def test_to_json_indent_is_supported() -> None:
    """``entry_to_json(..., indent=2)`` produces a pretty-printed string."""

    encoded = entry_to_json(_sample_entry(), indent=2)
    assert "\n" in encoded
    # Sort_keys is on — first key in sorted order is "delta".
    assert encoded.lstrip().startswith("{")


# ---------------------------------------------------------------------------
# 6. schema-version / decode error paths
# ---------------------------------------------------------------------------


def test_from_dict_rejects_unknown_schema_version() -> None:
    """A newer ``schema_version`` raises rather than being re-interpreted."""

    payload = entry_to_dict(_sample_entry())
    payload["schema_version"] = SCHEMA_VERSION + 999

    with pytest.raises(RatingLogSchemaError) as excinfo:
        entry_from_dict(payload)
    assert excinfo.value.found_version == SCHEMA_VERSION + 999


def test_from_dict_rejects_missing_schema_version() -> None:
    """A payload with no ``schema_version`` is rejected."""

    payload = entry_to_dict(_sample_entry())
    payload.pop("schema_version")

    with pytest.raises(RatingLogSchemaError):
        entry_from_dict(payload)


def test_from_dict_rejects_non_int_schema_version() -> None:
    """A non-int ``schema_version`` (e.g. "1") is rejected."""

    payload = entry_to_dict(_sample_entry())
    payload["schema_version"] = "1"  # type: ignore[assignment]

    with pytest.raises(RatingLogSchemaError):
        entry_from_dict(payload)


def test_from_dict_rejects_bool_schema_version() -> None:
    """Bool ``schema_version`` is rejected — same int-subclass footgun."""

    payload = entry_to_dict(_sample_entry())
    payload["schema_version"] = True  # type: ignore[assignment]

    with pytest.raises(RatingLogSchemaError):
        entry_from_dict(payload)


def test_from_dict_rejects_missing_required_field() -> None:
    """Each required field is checked; ``post_rating`` is representative."""

    payload = entry_to_dict(_sample_entry())
    payload.pop("post_rating")

    with pytest.raises(RatingLogDecodeError, match="post_rating"):
        entry_from_dict(payload)


def test_from_dict_rejects_inconsistent_delta() -> None:
    """A payload whose stored ``delta`` disagrees with pre/post is rejected.

    Guards against silently-corrupted log files where someone has
    hand-edited the rating but forgotten to update the cached delta.
    """

    payload = entry_to_dict(_sample_entry(pre_rating=2400, post_rating=2425))
    assert payload["delta"] == 25
    payload["delta"] = 99  # deliberate lie

    with pytest.raises(RatingLogDecodeError, match="delta"):
        entry_from_dict(payload)


def test_from_dict_accepts_payload_without_explicit_delta() -> None:
    """``delta`` is optional in the input — only cross-checked when present."""

    payload = entry_to_dict(_sample_entry(pre_rating=2400, post_rating=2425))
    payload.pop("delta")

    entry = entry_from_dict(payload)
    assert entry.delta == 25


def test_from_dict_rejects_non_string_timestamp() -> None:
    """A numeric or null ``timestamp`` is a decode error, not a domain error."""

    payload = entry_to_dict(_sample_entry())
    payload["timestamp"] = 1_700_000_000  # type: ignore[assignment]

    with pytest.raises(RatingLogDecodeError, match="timestamp"):
        entry_from_dict(payload)


def test_from_dict_rejects_invalid_iso_timestamp() -> None:
    """A malformed ISO string is a decode error."""

    payload = entry_to_dict(_sample_entry())
    payload["timestamp"] = "not-a-timestamp"

    with pytest.raises(RatingLogDecodeError, match="timestamp"):
        entry_from_dict(payload)


def test_from_dict_naive_timestamp_loaded_as_utc() -> None:
    """A naive ISO timestamp in JSON loads as UTC (round-trip tolerance)."""

    payload = entry_to_dict(_sample_entry())
    payload["timestamp"] = "2026-05-13T21:30:00"  # strip the +00:00

    entry = entry_from_dict(payload)
    assert entry.timestamp == datetime(2026, 5, 13, 21, 30, 0, tzinfo=timezone.utc)


def test_from_dict_propagates_domain_validation_error() -> None:
    """A payload whose decoded values fail domain validation surfaces as a validation error.

    The decoder defers to ``RatingLogEntry``'s ``__post_init__``, so a
    payload that decodes cleanly but breaks an invariant (e.g.
    negative rating) surfaces as a ``RatingLogValidationError`` — not
    a decode error. This lets callers distinguish "the file is
    corrupt" from "the file describes an impossible run".
    """

    payload = entry_to_dict(_sample_entry())
    payload["pre_rating"] = -100
    # Drop the now-inconsistent delta so we hit validation, not the delta check.
    payload.pop("delta", None)

    with pytest.raises(RatingLogValidationError, match="pre_rating"):
        entry_from_dict(payload)


def test_from_dict_rejects_non_dict_root() -> None:
    """A non-dict root payload is a decode error."""

    with pytest.raises(RatingLogDecodeError):
        entry_from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_from_json_rejects_invalid_json() -> None:
    """Corrupt JSON is a decode error."""

    with pytest.raises(RatingLogDecodeError):
        entry_from_json("{not valid json")


def test_from_json_rejects_non_object_root() -> None:
    """A JSON array at the root is a decode error."""

    with pytest.raises(RatingLogDecodeError):
        entry_from_json("[1, 2, 3]")


def test_to_dict_rejects_non_entry_argument() -> None:
    """``entry_to_dict`` refuses to encode unrelated objects."""

    with pytest.raises(TypeError):
        entry_to_dict({"team_id": "t1"})  # type: ignore[arg-type]
