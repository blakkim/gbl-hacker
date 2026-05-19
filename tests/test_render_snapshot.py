"""Tests for :mod:`gbl_hacker.render.snapshot`.

Sub-AC 6 contract — the data-quality caveat about Taiman Party report
density is visible in *every* meta snapshot output, not hidden behind a
flag. These tests pin that contract at the module level:

    * The renderer always emits a ``DATA HONESTY`` block containing
      the snapshot's ``source_caveat`` verbatim.
    * The renderer also emits the caveat in the one-line footer so a
      reader who only scans the bottom of the output still sees it.
    * The renderer signature has NO suppression parameter (``quiet``,
      ``include_caveat``, ``no_caveat`` …) — the structural guarantee
      is that suppression is *not expressible* in the API.
    * An empty ``source_caveat`` raises ``MissingCaveatError`` rather
      than silently producing a caveat-less rendering.
    * The pinned banner label ``DATA HONESTY — Taiman Party caveat``
      is rendered verbatim so downstream tooling can grep for it.

The tests are deliberately structural (sig inspection, exception
contract) and not just textual, so a future refactor cannot regress
AC 6 by accidentally adding a ``quiet`` kwarg.
"""

from __future__ import annotations

import inspect
import io
from datetime import datetime, timezone
from typing import Any

import pytest

from gbl_hacker.fetch.taiman import RECOMMEND_URL, TAIMAN_SOURCE_CAVEAT
from gbl_hacker.parse.taiman import MetaSnapshot, PokemonUsage, TeamUsage
from gbl_hacker.render.snapshot import (
    CAVEAT_HEADER_LABEL,
    MissingCaveatError,
    format_caveat_block,
    format_snapshot_header,
    render_meta_snapshot,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _sample_snapshot(
    *,
    source_caveat: str = TAIMAN_SOURCE_CAVEAT,
    pokemon_rows: int = 3,
    team_rows: int = 2,
) -> MetaSnapshot:
    """Build a small snapshot fully populated with usage rows.

    ``pokemon_rows`` and ``team_rows`` let tests probe truncation
    behaviour. The default values keep all rows visible under the
    renderer's display caps so the basic tests don't need to think
    about ``...`` lines.
    """

    # Usage percentages are positive and within [0, 100]; the formula
    # leaves room for ``pokemon_rows`` up to ~50 without producing a
    # negative ``usage_pct`` (which the dataclass would reject).
    pokemon_usage = tuple(
        PokemonUsage(
            species=f"species_{i}",
            usage_pct=max(0.1, 30.0 - i * 0.5),
            rank=i + 1,
        )
        for i in range(pokemon_rows)
    )
    team_usage = tuple(
        TeamUsage(
            members=(f"lead_{i}", f"safe_swap_{i}", f"closer_{i}"),
            usage_pct=max(0.1, 15.0 - i * 0.25),
            rank=i + 1,
        )
        for i in range(team_rows)
    )
    return MetaSnapshot(
        league="great_league",
        rating_bracket="upper",
        fetched_at=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        source_url=RECOMMEND_URL,
        source_caveat=source_caveat,
        pokemon_usage=pokemon_usage,
        team_usage=team_usage,
    )


# ---------------------------------------------------------------------------
# AC 6 — caveat is always present in the rendering
# ---------------------------------------------------------------------------


def test_render_emits_caveat_header_label_verbatim() -> None:
    """The pinned ``DATA HONESTY`` banner label appears in every rendering.

    Downstream tooling (the long-loop validation log, audit scripts)
    may grep for this label to verify a snapshot rendering has not
    silently dropped the caveat. Pinning the label here means changing
    it requires a deliberate update.
    """

    stream = io.StringIO()
    render_meta_snapshot(_sample_snapshot(), stream=stream)

    assert CAVEAT_HEADER_LABEL in stream.getvalue()


def test_render_emits_full_caveat_text_in_header_block() -> None:
    """The verbatim ``source_caveat`` string appears in the rendering.

    Not just a paraphrase or a "see source_caveat" pointer — the actual
    multi-sentence caveat must be readable in the output.
    """

    snapshot = _sample_snapshot()
    stream = io.StringIO()
    render_meta_snapshot(snapshot, stream=stream)

    output = stream.getvalue()

    # The caveat is wrapped across lines, so search for distinctive
    # substrings rather than the entire multi-line caveat.
    assert "report-density" in output.lower()
    assert "upper-bracket" in output.lower()
    assert "top-500" in output.lower()


def test_render_emits_caveat_in_footer_line_as_well() -> None:
    """The renderer footer ALSO includes ``source_caveat: ...``.

    Header + footer redundancy is intentional — a user piping the
    output through ``| tail -1`` (or grepping for ``source_caveat``)
    still sees the warning.
    """

    snapshot = _sample_snapshot()
    stream = io.StringIO()
    render_meta_snapshot(snapshot, stream=stream)

    output = stream.getvalue()
    # Find the footer line specifically.
    footer_lines = [
        line for line in output.splitlines() if line.startswith("source_caveat:")
    ]
    assert footer_lines, (
        "renderer must emit a `source_caveat: ...` footer line so that "
        "grepping a single line of output still surfaces the caveat"
    )
    assert TAIMAN_SOURCE_CAVEAT in footer_lines[-1]


def test_render_signature_has_no_caveat_suppression_kwarg() -> None:
    """The renderer API offers no parameter that hides the caveat.

    This is the architectural enforcement of AC 6: suppression is not
    *expressible* in the function signature. A future maintainer who
    tries to add a ``quiet`` or ``include_caveat`` knob will trip this
    test.
    """

    forbidden_names = {
        "quiet",
        "silent",
        "no_caveat",
        "skip_caveat",
        "include_caveat",
        "hide_caveat",
        "suppress_caveat",
        "minimal",
    }
    params = set(inspect.signature(render_meta_snapshot).parameters.keys())
    leak = params & forbidden_names
    assert not leak, (
        f"render_meta_snapshot must not accept a caveat-suppression knob; "
        f"found: {sorted(leak)} (violates AC 6 — data_honesty caveat must "
        f"not be hidden behind a flag)"
    )


def test_render_raises_when_caveat_is_empty_string() -> None:
    """An empty caveat string is a violation; the renderer must refuse it.

    ``MetaSnapshot.__post_init__`` already enforces a non-empty
    ``source_caveat``, but we mirror the check at the rendering layer
    so a hand-built snapshot (or a partial mock in some future test)
    cannot accidentally produce a caveat-less rendering.
    """

    # Construct via object.__setattr__ to bypass dataclass validation;
    # this mimics a hypothetical bug where the dataclass guard was
    # accidentally weakened. The renderer's defence-in-depth check
    # must still catch it.
    snapshot = _sample_snapshot()
    object.__setattr__(snapshot, "source_caveat", "   ")

    with pytest.raises(MissingCaveatError):
        render_meta_snapshot(snapshot, stream=io.StringIO())

    with pytest.raises(MissingCaveatError):
        format_caveat_block(snapshot)


def test_format_caveat_block_emits_horizontal_rules_for_visibility() -> None:
    """The caveat block is visually framed so it can't be mistaken for prose.

    The intent is that a casual reader sees the banner and stops; a
    snapshot rendering that buries the caveat inside a paragraph would
    fail this test.
    """

    block = format_caveat_block(_sample_snapshot())
    # At least the top and bottom rules ⇒ two horizontal rule lines.
    rule_lines = [line for line in block.splitlines() if set(line) == {"="}]
    assert len(rule_lines) >= 2


def test_render_preserves_data_honesty_caveat_on_custom_caveat_text() -> None:
    """A non-default caveat (e.g. a future bracket-specific note) still surfaces.

    AC 6 is about the *caveat* on the snapshot, not about pinning a
    single hard-coded string. If the engine ever attaches a more
    specific caveat for a particular bracket, the renderer must
    surface that string verbatim too.
    """

    custom = "CUSTOM CAVEAT — ace-bracket sample size is below threshold."
    snapshot = _sample_snapshot(source_caveat=custom)
    stream = io.StringIO()
    render_meta_snapshot(snapshot, stream=stream)
    output = stream.getvalue()

    assert custom in output, "custom caveat must round-trip into the rendering"


# ---------------------------------------------------------------------------
# Non-caveat content sanity — the renderer is still useful as a renderer
# ---------------------------------------------------------------------------


def test_render_includes_snapshot_metadata() -> None:
    """The rendering surfaces league / bracket / fetched_at / counts."""

    snapshot = _sample_snapshot()
    stream = io.StringIO()
    render_meta_snapshot(snapshot, stream=stream)
    output = stream.getvalue()

    assert "great_league" in output
    assert "upper" in output
    assert "2026-05-13" in output
    assert RECOMMEND_URL in output


def test_render_includes_pokemon_and_team_usage_tables() -> None:
    """Per-Pokémon and per-team rows show up in the rendering."""

    snapshot = _sample_snapshot(pokemon_rows=2, team_rows=2)
    stream = io.StringIO()
    render_meta_snapshot(snapshot, stream=stream)
    output = stream.getvalue()

    assert "species_0" in output
    assert "species_1" in output
    assert "lead_0" in output
    assert "closer_1" in output


def test_render_truncates_long_pokemon_table_with_marker() -> None:
    """Beyond the display cap, the renderer notes how many rows were elided.

    Truncation is OK — but the rendering must say so, not silently
    swallow rows. (Display caps are a UX choice, not a data choice; the
    snapshot retains every row for analytical consumption.)
    """

    snapshot = _sample_snapshot(pokemon_rows=25, team_rows=1)
    stream = io.StringIO()
    render_meta_snapshot(snapshot, stream=stream, pokemon_limit=5)
    output = stream.getvalue()

    assert "20 more not shown" in output


def test_format_snapshot_header_excludes_caveat_to_keep_separation_of_concerns() -> None:
    """``format_snapshot_header`` renders metadata only — the caveat is
    rendered by ``format_caveat_block`` and orchestrated by
    ``render_meta_snapshot``. Splitting them means a caller cannot
    accidentally print the header alone and skip the caveat.
    """

    header = format_snapshot_header(_sample_snapshot())
    assert "report-density" not in header.lower(), (
        "the metadata header must not subsume the caveat — keep the two "
        "concerns separated so render_meta_snapshot is the only path that "
        "emits both"
    )


def test_render_writes_to_supplied_stream_not_to_stdout() -> None:
    """The renderer is testable in isolation — never bypasses the stream."""

    stream = io.StringIO()
    render_meta_snapshot(_sample_snapshot(), stream=stream)
    # If anything leaked to stdout, the StringIO would be empty here.
    assert stream.getvalue(), "renderer must write to the provided stream"


# ---------------------------------------------------------------------------
# Multiple invocations — caveat must appear on EVERY rendering, not just first
# ---------------------------------------------------------------------------


def test_caveat_appears_on_each_rendering_in_sequence() -> None:
    """Re-rendering the same snapshot does not "cache" away the caveat.

    A naive implementation that memoizes the caveat could fail the AC's
    "every meta snapshot output" wording on a second call. The
    renderer is stateless; this test pins that.
    """

    snapshot = _sample_snapshot()
    for _ in range(3):
        stream = io.StringIO()
        render_meta_snapshot(snapshot, stream=stream)
        assert "report-density" in stream.getvalue().lower()
        assert CAVEAT_HEADER_LABEL in stream.getvalue()
