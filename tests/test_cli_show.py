"""Tests for the ``gblh show`` CLI subcommand.

Sub-AC 6 contract: every meta-snapshot rendering surfaces the
data-honesty caveat. ``gblh show`` is the re-display surface — a user
who already has a snapshot cached should be able to look at it again
*without* being able to suppress the caveat.

These tests cover:

    * Loading the latest snapshot from a cache directory and rendering
      it — caveat appears unconditionally.
    * Loading an explicit ``--path`` file and rendering — caveat
      appears unconditionally.
    * Empty cache directory ⇒ ``EXIT_NO_SNAPSHOT`` (distinct from
      ``EXIT_PERSIST``) so callers can branch on the cause.
    * The CLI has no ``--quiet`` / ``--no-caveat`` flag — argparse
      rejects such arguments.

Tests are fully offline: every snapshot is written into ``tmp_path``
via the real persistence layer, so the same on-disk format ``gblh
refresh`` writes is exercised.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

from gbl_hacker.cli import (
    EXIT_NO_SNAPSHOT,
    EXIT_OK,
    EXIT_PERSIST,
    build_arg_parser,
    main,
)
from gbl_hacker.fetch.taiman import RECOMMEND_URL, TAIMAN_SOURCE_CAVEAT
from gbl_hacker.parse.taiman import MetaSnapshot, PokemonUsage, TeamUsage
from gbl_hacker.persist.snapshot import write_snapshot
from gbl_hacker.render.snapshot import CAVEAT_HEADER_LABEL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    fetched_at: datetime | None = None,
    rating_bracket: str = "upper",
) -> MetaSnapshot:
    """Build a minimal but valid snapshot.

    The renderer doesn't care about realistic data — it cares about the
    caveat. Two rows on each axis is enough to exercise table
    rendering without making the test brittle.
    """

    return MetaSnapshot(
        league="great_league",
        rating_bracket=rating_bracket,
        fetched_at=fetched_at
        or datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        source_url=RECOMMEND_URL,
        source_caveat=TAIMAN_SOURCE_CAVEAT,
        pokemon_usage=(
            PokemonUsage(species="azumarill", usage_pct=14.7, rank=1),
            PokemonUsage(species="registeel", usage_pct=8.2, rank=2),
        ),
        team_usage=(
            TeamUsage(
                members=("Azumarill", "Annihilape", "Registeel"),
                usage_pct=3.4,
                rank=1,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Sub-AC 6 — show surfaces the caveat unconditionally
# ---------------------------------------------------------------------------


def test_show_renders_latest_snapshot_with_caveat_banner(tmp_path: Path) -> None:
    """``gblh show`` loads the latest snapshot and emits the caveat banner.

    The single critical assertion: the rendered output contains the
    ``DATA HONESTY`` banner label AND the report-density warning text.
    Without those, the data-honesty principle is violated.
    """

    write_snapshot(_snapshot(), cache_dir=tmp_path, filename="latest.json")

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["show", "--cache-dir", str(tmp_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == EXIT_OK, (
        f"show should succeed with one snapshot present; got "
        f"{exit_code}, stderr={stderr.getvalue()!r}"
    )
    output = stdout.getvalue()
    assert CAVEAT_HEADER_LABEL in output, (
        "show output must contain the DATA HONESTY banner per AC 6"
    )
    assert "report-density" in output.lower(), (
        "show output must include the verbatim report-density warning"
    )


def test_show_with_explicit_path_renders_that_snapshot_with_caveat(
    tmp_path: Path,
) -> None:
    """``gblh show --path FILE`` renders an explicit snapshot, caveat included.

    Useful when the operator wants to replay a *historical* snapshot
    (e.g. for the long-loop validation log). The caveat must still
    appear — re-rendering an old snapshot doesn't make the data more
    reliable.
    """

    stored = write_snapshot(_snapshot(), cache_dir=tmp_path, filename="pinned.json")

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["show", "--path", str(stored.path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == EXIT_OK, stderr.getvalue()
    assert CAVEAT_HEADER_LABEL in stdout.getvalue()
    assert "report-density" in stdout.getvalue().lower()


def test_show_returns_no_snapshot_exit_code_when_cache_empty(
    tmp_path: Path,
) -> None:
    """An empty cache dir yields the dedicated ``EXIT_NO_SNAPSHOT`` code."""

    empty = tmp_path / "cache"
    empty.mkdir()

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["show", "--cache-dir", str(empty)],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == EXIT_NO_SNAPSHOT
    # Stdout should not carry a bogus "rendered nothing" output —
    # the error goes to stderr.
    assert stdout.getvalue() == ""
    assert "no snapshots" in stderr.getvalue().lower()


def test_show_returns_persist_exit_code_on_corrupt_path(tmp_path: Path) -> None:
    """A corrupt/invalid snapshot file routes to ``EXIT_PERSIST``."""

    bad = tmp_path / "broken.json"
    bad.write_text("{this is not valid json", encoding="utf-8")

    exit_code = main(
        ["show", "--path", str(bad)],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_PERSIST


def test_show_subcommand_rejects_quiet_flag(tmp_path: Path) -> None:
    """Argparse must reject a ``--quiet`` / ``--no-caveat`` argument.

    AC 6 is about structural enforcement. If a future maintainer adds a
    silencing flag, this test breaks before any production code can
    quietly hide the warning.
    """

    write_snapshot(_snapshot(), cache_dir=tmp_path, filename="latest.json")

    # Each forbidden flag should be rejected (exit code != 0) AND
    # leave stdout empty.
    for forbidden in ("--quiet", "--no-caveat", "--silent", "--hide-caveat"):
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main(
            ["show", "--cache-dir", str(tmp_path), forbidden],
            stdout=stdout,
            stderr=stderr,
        )

        assert exit_code != EXIT_OK, (
            f"show must reject {forbidden!r} (AC 6 — no caveat-suppression "
            f"flag is allowed); got success exit"
        )
        assert stdout.getvalue() == "", (
            f"show should not emit any rendering when {forbidden!r} is "
            "rejected — otherwise a partial suppression channel exists"
        )


def test_show_parser_has_no_caveat_suppression_options() -> None:
    """The argparse parser for ``show`` exposes no caveat-suppression flag.

    Structural complement to ``test_show_subcommand_rejects_quiet_flag``:
    even with all argparse's loose-match behaviour, the *registered*
    options for ``show`` must not include a suppression knob.
    """

    arg_parser = build_arg_parser()
    # Inspect by re-parsing --help on the show subcommand and grepping
    # the option strings. argparse's _actions is private but stable.
    sub = next(
        action
        for action in arg_parser._actions  # type: ignore[attr-defined]
        if action.dest == "command"
    )
    show_parser = sub.choices["show"]  # type: ignore[attr-defined]

    forbidden_substrings = (
        "quiet",
        "silent",
        "no-caveat",
        "no_caveat",
        "hide-caveat",
        "hide_caveat",
        "skip-caveat",
        "skip_caveat",
        "suppress-caveat",
        "suppress_caveat",
    )

    for action in show_parser._actions:  # type: ignore[attr-defined]
        for opt in action.option_strings:
            lowered = opt.lower()
            for needle in forbidden_substrings:
                assert needle not in lowered, (
                    f"`gblh show` registered option {opt!r} contains "
                    f"caveat-suppression substring {needle!r} — "
                    "violates AC 6 (data_honesty caveat must not be "
                    "hidden behind a flag)"
                )


def test_show_falls_back_to_latest_when_path_not_given(tmp_path: Path) -> None:
    """When multiple snapshots exist, ``show`` picks the newest by mtime."""

    import os

    older = write_snapshot(
        _snapshot(
            fetched_at=datetime(2026, 5, 12, 0, 0, 0, tzinfo=timezone.utc),
            rating_bracket="upper",
        ),
        cache_dir=tmp_path,
        filename="older.json",
    )
    newer = write_snapshot(
        _snapshot(
            fetched_at=datetime(2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc),
            rating_bracket="ace",
        ),
        cache_dir=tmp_path,
        filename="newer.json",
    )
    os.utime(older.path, (1_700_000_000, 1_700_000_000))
    os.utime(newer.path, (1_800_000_000, 1_800_000_000))

    stdout = io.StringIO()
    exit_code = main(
        ["show", "--cache-dir", str(tmp_path)],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert exit_code == EXIT_OK
    output = stdout.getvalue()
    assert "ace" in output, "show should render the newer (ace-bracket) snapshot"
    # Caveat still present — sanity check on the renderer wiring.
    assert CAVEAT_HEADER_LABEL in output


# ---------------------------------------------------------------------------
# Cross-surface invariant — refresh and show emit the SAME caveat content
# ---------------------------------------------------------------------------


def test_refresh_and_show_produce_consistent_caveat_text(tmp_path: Path) -> None:
    """``gblh refresh`` and ``gblh show`` use the same renderer → same caveat.

    If a future change forks the rendering path, this guard catches the
    drift before the two surfaces can diverge on AC 6.
    """

    from gbl_hacker.fetch.taiman import (
        GREAT_LEAGUE_ID,
        SEASON_LEAGUE_URL,
        FetchResult,
        TaimanRawSnapshot,
    )

    season_bytes = (
        Path(__file__).parent / "fixtures" / "taiman_season_league.json"
    ).read_bytes()
    recommend_bytes = (
        Path(__file__).parent / "fixtures" / "taiman_recommend_great_league.html"
    ).read_bytes()

    def stub_fetcher() -> TaimanRawSnapshot:
        when = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
        return TaimanRawSnapshot(
            season=27,
            league_id=GREAT_LEAGUE_ID,
            season_league=FetchResult(
                url=SEASON_LEAGUE_URL,
                status_code=200,
                content=season_bytes,
                content_type="text/html; charset=UTF-8",
                fetched_at=when,
                source_caveat=TAIMAN_SOURCE_CAVEAT,
            ),
            recommend=FetchResult(
                url=RECOMMEND_URL + "?season=27&league=0&between=1",
                status_code=200,
                content=recommend_bytes,
                content_type="text/html; charset=UTF-8",
                fetched_at=when,
                source_caveat=TAIMAN_SOURCE_CAVEAT,
            ),
        )

    # Refresh path
    refresh_out = io.StringIO()
    rc1 = main(
        ["refresh", "--cache-dir", str(tmp_path)],
        fetcher=stub_fetcher,
        stdout=refresh_out,
        stderr=io.StringIO(),
    )
    assert rc1 == EXIT_OK

    # Show path
    show_out = io.StringIO()
    rc2 = main(
        ["show", "--cache-dir", str(tmp_path)],
        stdout=show_out,
        stderr=io.StringIO(),
    )
    assert rc2 == EXIT_OK

    for output in (refresh_out.getvalue(), show_out.getvalue()):
        assert CAVEAT_HEADER_LABEL in output
        assert TAIMAN_SOURCE_CAVEAT in output
