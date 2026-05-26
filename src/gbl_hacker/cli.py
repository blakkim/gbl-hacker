"""Command-line entry point for the v0.1 GBL Hacker engine.

Sub-AC 4 contract: a single subcommand (``gblh refresh``) wires the three
ingestion stages together end-to-end:

    fetch  → parse  → persist

Sub-AC 6 contract: a companion ``gblh show`` subcommand renders a
previously-cached snapshot via the canonical
:func:`gbl_hacker.render.snapshot.render_meta_snapshot` so that the
data-honesty caveat surfaces on *every* snapshot output — refresh,
re-display, replay — not just the initial fetch.

Sub-AC 7.3 contract: a ``gblh report-rating`` subcommand accepts a
team-id and pre/post-rating arguments, constructs a
:class:`gbl_hacker.rating_log.RatingLogEntry`, and persists it via
:func:`gbl_hacker.rating_log.append_entry`. This is the operator-facing
hook for the long-loop validation feedback path (see
``exit_conditions.long_loop_validation`` in ``seed.yaml``) — every
real-life GBL set the operator runs against a recommended team gets one
``report-rating`` invocation, and the appended JSONL line is what makes
``rating_change_log entries >= 1`` measurable.

The CLI is intentionally small. v0.1's analytical surface lives in the
simulator and the (future) Pareto ranker; this module only owns the
ingestion-side plumbing so the operator can refresh the local meta cache
and report rating outcomes without dropping into a Python REPL.

Design notes
------------

* **Dependency injection over patching.** ``main()`` takes optional
  ``fetcher`` / ``parser`` / ``persister`` callables. Production wires the
  real implementations from :mod:`gbl_hacker.fetch`, :mod:`gbl_hacker.parse`,
  and :mod:`gbl_hacker.persist`; tests swap in mocks/spies and observe
  what the CLI calls them with. This keeps the test for Sub-AC 4 honest:
  it pins the *wiring*, not just the side effects.

* **Typed exit codes.** ``main()`` always returns an ``int`` so it can be
  asserted on by callers (the test runner uses this rather than catching
  ``SystemExit``). The wrapper :func:`run` lifts it to ``sys.exit`` for
  the ``gblh`` console-script entry point in ``pyproject.toml``.

* **Data-honesty caveat surfaced structurally.** ``refresh`` and ``show``
  both delegate snapshot rendering to
  :func:`gbl_hacker.render.snapshot.render_meta_snapshot`, which has no
  suppression flag. Per the ``data_honesty`` evaluation principle
  (AC 6), the report-density warning is structurally guaranteed to
  appear on every rendering — not gated on a verbose flag.

* **Errors are translated, not leaked.** ``TaimanFetchError``,
  ``TaimanParseError`` and ``SnapshotPersistError`` are caught and
  turned into a short stderr line + non-zero exit code. The
  ``--debug`` flag re-raises so the operator can see the full traceback
  when something genuinely unexpected happens.

The default cache layout matches what :mod:`gbl_hacker.persist.snapshot`
documents: ``~/.cache/gbl-hacker/snapshots/{league}__{bracket}__{ts}.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence, TextIO

from gbl_hacker.build_registry import (
    build_registry_for_meta,
    build_registry_pvpoke_top,
    synthesize_pvpoke_opponent_meta,
)
from gbl_hacker.dex import PokedexRegistry, load_default_registry
from gbl_hacker.fetch.taiman import (
    TaimanFetchError,
    TaimanRawSnapshot,
    fetch_great_league_meta,
)
from gbl_hacker.parse.taiman import (
    MetaSnapshot,
    TaimanParseError,
    parse_great_league_meta,
)
from gbl_hacker.score.expected_win_rate import (
    CandidateTeam,
    default_set_win_rate,
    expected_win_rate,
    materialize_opponent_team,
    set_driver_win_rate,
)
from gbl_hacker.score.meta_coverage import meta_coverage
from gbl_hacker.score.pareto import Score, ScoredTeam, pareto_filter
from gbl_hacker.score.rank import rank_top_k
from gbl_hacker.score.worst_case_robustness import worst_case_robustness
from gbl_hacker.persist.snapshot import (
    DEFAULT_CACHE_SUBDIR,
    SnapshotPersistError,
    StoredSnapshot,
    latest_snapshot,
    list_snapshots,
    read_snapshot,
    write_snapshot,
)
from gbl_hacker.rating_log import (
    DEFAULT_STORE_FILENAME,
    RatingLogEntry,
    RatingLogError,
    RatingLogValidationError,
    append_entry,
)
from gbl_hacker.reference import (
    DEFAULT_THRESHOLD as VERIFY_DEFAULT_THRESHOLD,
    ReferenceLoadError,
    format_verdict_summary,
    load_recommendations_fixture,
    load_reference_team_list,
    verify_overlap,
)
from gbl_hacker.render.recommendation import render_recommendation_table
from gbl_hacker.render.snapshot import render_meta_snapshot

# ---------------------------------------------------------------------------
# Type aliases for the injectable stages
# ---------------------------------------------------------------------------

FetcherFn = Callable[[], TaimanRawSnapshot]
"""Callable that yields a fresh raw payload pair from Taiman Party.

Production binds ``fetch_great_league_meta()`` which performs two HTTP
requests (``getSeasonLeague.php`` JSON + ``BattlePvpRecommendNewVue.php``
HTML) and bundles them into a ``TaimanRawSnapshot``. Tests bind a
no-network stub that returns canned bytes captured from a real refresh."""

ParserFn = Callable[[TaimanRawSnapshot], MetaSnapshot]
"""Callable that converts a raw fetch pair into a normalized snapshot.

Production binds ``parse_great_league_meta``. Tests usually bind the same
real parser so they exercise the wiring end-to-end while still controlling
the input bytes."""

PersisterFn = Callable[[MetaSnapshot, Path], StoredSnapshot]
"""Callable that writes a snapshot to a versioned local store.

Production binds a thin wrapper around ``write_snapshot``. Tests usually
bind a spy that records the arguments — the Sub-AC 4 assertion is exactly
"persistence was called with the parsed snapshot", so the spy is the
point of the test."""

RatingLogWriterFn = Callable[[RatingLogEntry, Path], None]
"""Callable that appends a ``RatingLogEntry`` to the JSONL store at a path.

Production binds a thin wrapper around
:func:`gbl_hacker.rating_log.append_entry`. Tests bind a spy that
captures the entry + path so the Sub-AC 7.3 assertion ("a rating-log
entry was built from the CLI args and handed to the store") can be made
without round-tripping through real disk I/O — though for the headline
test we DO call through to the real append_entry against ``tmp_path``,
so the JSONL line on disk is part of the verified contract."""

NowFn = Callable[[], datetime]
"""Callable that returns the current tz-aware datetime.

Production binds ``lambda: datetime.now(timezone.utc)`` — the operator's
wall-clock moment of "I just finished this set". Tests inject a fixed
clock so the persisted entry has a deterministic timestamp."""


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_cache_root() -> Path:
    """Resolve the on-disk cache root for ``gblh`` snapshots.

    Resolution order:

    1. ``$GBL_HACKER_CACHE`` (escape hatch for power users / CI).
    2. ``$XDG_CACHE_HOME/gbl-hacker`` if XDG is set.
    3. ``~/.cache/gbl-hacker`` (Linux/macOS default).

    The returned path is a *root* — actual snapshot files land in the
    ``snapshots/`` sub-directory (see ``DEFAULT_CACHE_SUBDIR``).
    """

    env_override = os.environ.get("GBL_HACKER_CACHE")
    if env_override:
        return Path(env_override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "gbl-hacker"
    return Path.home() / ".cache" / "gbl-hacker"


def _default_fetcher() -> TaimanRawSnapshot:
    """Production fetcher binding — performs both backend HTTP requests."""

    return fetch_great_league_meta()


def _default_parser(raw: TaimanRawSnapshot) -> MetaSnapshot:
    """Production parser binding — converts the raw pair into a snapshot."""

    return parse_great_league_meta(raw)


def _default_persister(snapshot: MetaSnapshot, cache_dir: Path) -> StoredSnapshot:
    """Production persister binding — writes the snapshot to disk."""

    return write_snapshot(snapshot, cache_dir=cache_dir)


def _default_rating_log_writer(entry: RatingLogEntry, path: Path) -> None:
    """Production rating-log writer — appends one entry to the JSONL store."""

    append_entry(entry, path=path)


def _default_now() -> datetime:
    """Production clock — wall-clock UTC at the moment the CLI ran.

    Returns a tz-aware ``datetime``. The rating-log entry's
    ``__post_init__`` coerces naive datetimes to UTC, but we are explicit
    here so the on-disk timestamp is unambiguously UTC even before the
    dataclass normalization fires.
    """

    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Exit codes
#
# These are stable so shell scripts (and the long-loop validation log) can
# branch on them. We do NOT use os.EX_* — those are POSIX-only and not
# Windows-friendly.
# ---------------------------------------------------------------------------

EXIT_OK: int = 0
"""Refresh ran end-to-end and a snapshot was persisted."""

EXIT_USAGE: int = 2
"""argparse-style usage error (mirrors argparse's own convention)."""

EXIT_FETCH: int = 3
"""Upstream fetch failed (network or non-2xx HTTP)."""

EXIT_PARSE: int = 4
"""Upstream payload could not be parsed (likely DOM drift)."""

EXIT_PERSIST: int = 5
"""Persistence layer failed (I/O, permissions, full disk)."""

EXIT_NO_SNAPSHOT: int = 6
"""``gblh show`` was invoked but no snapshot exists in the cache directory."""

EXIT_VERIFY_FAIL: int = 7
"""``gblh verify-reference`` ran end-to-end but the verdict was ``"fail"``.

The verdict ran cleanly — fixtures loaded, overlap was computed — but the
observed Jaccard on the chosen axis was below the configured threshold.
This is a *successful execution with a negative outcome*, distinct from
the loader / IO failure modes that also map to non-zero codes. CI gates
can branch on this exact value to surface "engine drifted from
reference" vs "the verification step itself broke"."""

EXIT_VERIFY_LOAD: int = 8
"""``gblh verify-reference`` could not load one of its input fixtures.

Distinct from :data:`EXIT_VERIFY_FAIL` so a CI gate's failure surface
clearly separates "engine output disagrees with reference" from
"fixtures were missing / malformed"."""

EXIT_RECOMMEND: int = 10
"""``gblh recommend`` failed to produce a recommendation list.

Covers the operator-visible failure modes that are distinct from the
snapshot-load codes:

* No snapshot available in the cache (mirror of ``EXIT_NO_SNAPSHOT``
  but kept separate so a CI gate can branch ``recommend`` failures
  from ``show`` failures).
* Empty or fully-unresolvable meta — the build registry came back
  empty so no team could be scored.
"""


EXIT_RATING_LOG: int = 9
"""``gblh report-rating`` failed to construct or persist the entry.

Distinct from :data:`EXIT_VERIFY_FAIL` (7) and :data:`EXIT_VERIFY_LOAD`
(8) so shell wrappers / CI gates can branch deterministically on the
literal exit code without having to disambiguate "verify-reference
fail" from "report-rating failure".

Covers two operator-visible failure modes:

* Validation failure (e.g. negative rating, whitespace-only ``team_id``,
  malformed ``--timestamp`` string) — the entry never made it past the
  ``RatingLogEntry`` constructor.
* Append-time I/O failure (disk full, permission error on the rating-log
  path, embedded newline in serialized payload).

Either way the operator's correct response is the same: read the stderr
message, fix the input or the disk condition, re-run the command. The
shared exit code lets shell wrappers branch on "rating-log entry was
not recorded" without distinguishing the cause."""


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level ``gblh`` argument parser.

    The parser uses sub-commands so future stages (``recommend``, ``log``,
    ``replay``, …) can be added without breaking the v0.1 contract.
    """

    parser = argparse.ArgumentParser(
        prog="gblh",
        description=(
            "GBL Hacker — Great League team recommendation engine. "
            "Set-state-aware simulator over a Taiman Party meta feed."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    refresh = sub.add_parser(
        "refresh",
        help=(
            "Fetch a fresh Taiman Party Great League snapshot, parse it, "
            "and persist it to the local cache."
        ),
        description=(
            "Manually refresh the local meta cache. Wires the fetch, parse, "
            "and persistence layers end-to-end. The data-honesty caveat is "
            "printed on success so it cannot be silently suppressed."
        ),
    )
    refresh.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Override the cache directory. Default: "
            "$GBL_HACKER_CACHE, $XDG_CACHE_HOME/gbl-hacker, or "
            "~/.cache/gbl-hacker — with the 'snapshots' sub-dir appended."
        ),
    )
    refresh.add_argument(
        "--filename",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Override the on-disk filename (default: "
            "{league}__{bracket}__{timestamp}.json)."
        ),
    )
    refresh.add_argument(
        "--lang",
        choices=("ja", "ko", "en"),
        default="ko",
        help=(
            "Display language for rendered species names: 'ja' (Taiman "
            "Party's original katakana), 'ko' (Korean), 'en' (English). "
            "Default: ko. The cached snapshot file always retains the "
            "original Japanese names plus the dex id — this flag only "
            "affects the human-readable output."
        ),
    )
    refresh.add_argument(
        "--debug",
        action="store_true",
        help="Re-raise underlying exceptions instead of catching and exiting.",
    )

    show = sub.add_parser(
        "show",
        help=(
            "Render a previously-cached snapshot to stdout. Always surfaces "
            "the Taiman Party data-honesty caveat — no flag suppresses it."
        ),
        description=(
            "Read a cached MetaSnapshot from disk and pretty-print it for a "
            "human reader. The output ALWAYS includes the report-density "
            "caveat block at the top and a footer line that re-states the "
            "caveat — per the data_honesty evaluation principle (AC 6) "
            "there is no flag, env var, or argument that hides the caveat."
        ),
    )
    show.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Override the cache directory to read from. Default matches the "
            "refresh subcommand's resolution order."
        ),
    )
    show.add_argument(
        "--path",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Read the snapshot from this explicit JSON path instead of the "
            "latest entry in the cache directory."
        ),
    )
    show.add_argument(
        "--lang",
        choices=("ja", "ko", "en"),
        default="ko",
        help=(
            "Display language for species names. Default: ko. Cached "
            "snapshot bytes are untouched — flag only affects rendering."
        ),
    )

    # ------------------------------------------------------------------
    # report-rating — Sub-AC 7.3 long-loop validation feedback hook.
    # ------------------------------------------------------------------
    rr = sub.add_parser(
        "report-rating",
        help=(
            "Log a real-life GBL run of a recommended team: pre/post "
            "rating, optional notes. Appends one JSONL line to the "
            "rating-log store."
        ),
        description=(
            "Record one real-life GBL run of a recommended team. "
            "Constructs a RatingLogEntry from the arguments and appends "
            "it to the rating-log JSONL store. This is the long-loop "
            "validation feedback path — the engine reads these entries "
            "back to learn whether its recommendations actually paid off."
        ),
    )
    rr.add_argument(
        "--team-id",
        required=True,
        type=str,
        metavar="ID",
        help=(
            "Stable identifier of the recommended team that was run. "
            "Free-form string — the rating log is engine-agnostic about "
            "team naming. Must be non-empty and not just whitespace."
        ),
    )
    rr.add_argument(
        "--pre",
        required=True,
        type=int,
        dest="pre_rating",
        metavar="RATING",
        help="Operator's GBL rating BEFORE the run began. Non-negative int.",
    )
    rr.add_argument(
        "--post",
        required=True,
        type=int,
        dest="post_rating",
        metavar="RATING",
        help=(
            "Operator's GBL rating AFTER the run ended. Non-negative int. "
            "May be greater than, equal to, or less than --pre — all three "
            "are valid outcomes the engine wants to learn from."
        ),
    )
    rr.add_argument(
        "--timestamp",
        type=str,
        default=None,
        metavar="ISO8601",
        help=(
            "Override the run's completion timestamp (ISO 8601, "
            "tz-aware). Default: wall-clock now in UTC."
        ),
    )
    rr.add_argument(
        "--notes",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "Free-form operator notes about the run. Korean / non-ASCII "
            "text is preserved verbatim in the JSONL line."
        ),
    )
    rr.add_argument(
        "--rating-log-path",
        type=Path,
        default=None,
        dest="rating_log_path",
        metavar="FILE",
        help=(
            "Explicit JSONL store file. Default: --cache-dir "
            f"resolved, then '{DEFAULT_STORE_FILENAME}'."
        ),
    )
    rr.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Cache directory that holds the rating-log file. Default "
            "resolution order matches the refresh subcommand; the "
            "rating-log file lives at the cache root (no 'snapshots' "
            "sub-dir)."
        ),
    )
    rr.add_argument(
        "--debug",
        action="store_true",
        help="Re-raise underlying exceptions instead of catching and exiting.",
    )

    # ------------------------------------------------------------------
    # verify-reference — Sub-AC 5.3 recommendation-vs-reference gate.
    # ------------------------------------------------------------------
    verify = sub.add_parser(
        "verify-reference",
        help=(
            "Compare a frozen engine recommendation list against an "
            "independent top-tier reference and emit a pass/fail verdict "
            "against a configurable Jaccard threshold (AC 5 gate)."
        ),
        description=(
            "AC 5 gate. Loads a frozen recommendations JSON fixture and a "
            "reference team-list JSON fixture, computes the symmetric "
            "Jaccard overlap on the chosen axis (team / pokemon), and "
            "emits 'pass' iff the observed Jaccard is >= --threshold. "
            "Exit codes: 0 on pass, EXIT_VERIFY_FAIL (7) on a "
            "clean-execution fail, EXIT_VERIFY_LOAD (8) on a fixture "
            "load error — so a CI gate can branch deterministically."
        ),
    )
    verify.add_argument(
        "--recommendations",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to the frozen engine-recommendations JSON fixture.",
    )
    verify.add_argument(
        "--reference",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to the independent reference-team-list JSON fixture.",
    )
    verify.add_argument(
        "--threshold",
        type=float,
        default=VERIFY_DEFAULT_THRESHOLD,
        metavar="JACCARD",
        help=(
            f"Minimum Jaccard for a 'pass' verdict (default "
            f"{VERIFY_DEFAULT_THRESHOLD}). Must be in [0.0, 1.0]."
        ),
    )
    verify.add_argument(
        "--axis",
        choices=("team", "pokemon"),
        default="team",
        help=(
            "Which Jaccard drives the verdict: 'team' (unordered species "
            "triples, default) or 'pokemon' (individual species rosters)."
        ),
    )

    # ------------------------------------------------------------------
    # recommend — top-K Pareto recommendation from the latest snapshot.
    # ------------------------------------------------------------------
    recommend = sub.add_parser(
        "recommend",
        help=(
            "Score every team in the latest cached snapshot through the "
            "set-state-aware simulator, take the Pareto frontier across "
            "(expected_win_rate, worst_case_robustness, meta_coverage), "
            "and print the top-K — with member names localized to --lang."
        ),
        description=(
            "Engine output — Sub-AC 2.4/2.5 surface. Loads the latest "
            "snapshot from the cache, materializes every meta species via "
            "the PvPoke gamemaster, scores every team_usage entry on the "
            "three axes, filters to the Pareto-optimal subset, and ranks "
            "the top-K by a weighted-sum tiebreaker. Output ALWAYS includes "
            "the Taiman Party data-honesty caveat (AC 6)."
        ),
    )
    recommend.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Cache directory to load the latest snapshot from.",
    )
    recommend.add_argument(
        "--path",
        type=Path,
        default=None,
        metavar="FILE",
        help="Read the snapshot from this explicit JSON path instead.",
    )
    recommend.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        metavar="K",
        help="Number of teams to surface (default: 5).",
    )
    recommend.add_argument(
        "--lang",
        choices=("ja", "ko", "en"),
        default="ko",
        help="Display language for species names (default: ko).",
    )
    recommend.add_argument(
        "--engine",
        choices=("set-driver", "9-pairing"),
        default="set-driver",
        help=(
            "Per-set win-rate aggregator. 'set-driver' (default) runs "
            "the full 3v3 set with energy/HP/shield carry-over and "
            "faint-driven switching. '9-pairing' uses the legacy max-"
            "entropy slot-pairing average (cheaper, less accurate)."
        ),
    )
    recommend.add_argument(
        "--candidates",
        choices=("ranking", "pool", "meta"),
        default="ranking",
        help=(
            "Candidate generation strategy. 'ranking' (default) "
            "enumerates ordered 3-combos from PvPoke's top-N GL "
            "ranking — the simulator's view of what's strongest "
            "regardless of JP-site popularity. Catches niche picks the "
            "meta underuses (the '마스카나 case'). 'pool' uses the top-N "
            "Taiman-Party most-used species. 'meta' rescores only the "
            "30 teams the site recommends."
        ),
    )
    recommend.add_argument(
        "--pool-size",
        type=int,
        default=12,
        dest="pool_size",
        metavar="N",
        help=(
            "Pool size for --candidates pool/ranking. For 'pool' it "
            "selects top-N most-used Taiman species; for 'ranking' it "
            "selects top-N from PvPoke's GL overall ranking. Default "
            "12 (1320 ordered 3-combos, ~70s). Larger pools scale "
            "roughly cubically: 15→~150s, 20→~6 min, 30→~22 min."
        ),
    )
    recommend.add_argument(
        "--opponents",
        choices=("meta", "ranking"),
        default="ranking",
        help=(
            "Opponent set the candidate is scored against. 'ranking' "
            "(default) synthesizes opponent lineups from PvPoke's "
            "top-N GL ranking (type-diverse teammates) so the score "
            "reflects ladder-shape robustness — including niche threats "
            "like rock leads that the Taiman 30-team feed underweights. "
            "'meta' uses Taiman Party's 30 recommended teams (true "
            "popularity weighting, but blind to anything outside that "
            "set)."
        ),
    )
    recommend.add_argument(
        "--opponents-size",
        type=int,
        default=30,
        dest="opponents_size",
        metavar="N",
        help=(
            "Synthetic opponent count for --opponents ranking. Default "
            "30. Each opponent is one PvPoke top-N lead + 2 type-"
            "diverse PvPoke top-20 teammates."
        ),
    )
    recommend.add_argument(
        "--stochastic-samples",
        type=int,
        default=5,
        dest="stochastic_samples",
        metavar="N",
        help=(
            "Number of stochastic simulator samples per (candidate, "
            "opponent) pair. ``5`` (default) averages 5 RNG-seeded "
            "simulations so the shield-decision stochasticity (non-lethal "
            "threats shielded with 50%% probability) yields a meaningful "
            "win-rate distribution — fixes the wcr 0%%/100%% collapse. "
            "Pass ``1`` for the legacy deterministic baseline (faster but "
            "WCR collapses to 0/100). Scoring time scales linearly in N."
        ),
    )
    recommend.add_argument(
        "--active-switch",
        action="store_true",
        dest="active_switch",
        help=(
            "Enable timer-aware active switching in the set driver. "
            "Default off — the heuristic adds significant per-set cost "
            "and over large candidate pools dominates the runtime. "
            "Recommended for small pools (e.g. --pool-size 6) when you "
            "want the set driver to consider mid-set swap value."
        ),
    )
    recommend.add_argument(
        "--win-mode",
        choices=["ko", "resource"],
        default="ko",
        dest="win_mode",
        help=(
            "How a set is scored when neither side is fully KO'd by the "
            "turn-budget cap. 'ko' (default) calls such stalls a draw. "
            "'resource' awards the stall to the side ahead on residual "
            "resources (alive Pokémon > shields > HP > energy) — real GBL "
            "has no draws, so this credits bulky 'force-shield then farm "
            "down on resource lead' lines that KO scoring misses."
        ),
    )
    recommend.add_argument(
        "--exclude",
        type=str,
        default="",
        metavar="LIST",
        help=(
            "Comma-separated list of Japanese species names (with form "
            "in parens when applicable, e.g. 'サニーゴ(ガラル),コノヨザル') to "
            "exclude from the CANDIDATE pool. Use when you don't own "
            "those Pokémon or don't want to use them. Opponent set is "
            "not filtered — you still need to handle them on ladder."
        ),
    )
    recommend.add_argument(
        "--debug",
        action="store_true",
        help="Re-raise underlying exceptions instead of catching and exiting.",
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand: refresh
# ---------------------------------------------------------------------------


def cmd_refresh(
    *,
    cache_dir: Path | None,
    filename: str | None,
    debug: bool,
    fetcher: FetcherFn,
    parser: ParserFn,
    persister: PersisterFn,
    stdout: TextIO,
    stderr: TextIO,
    lang: str = "ko",
    dex_registry: PokedexRegistry | None = None,
) -> int:
    """Execute the ``refresh`` subcommand.

    Pipeline (each stage isolated so the test can spy on any one):

        1. ``fetcher()``  → ``FetchResult``
        2. ``parser(result)`` → ``MetaSnapshot``
        3. ``persister(snapshot, cache_dir)`` → ``StoredSnapshot``

    Returns the appropriate exit code. Always non-raising unless
    ``debug`` is set, in which case the underlying exception bubbles.
    """

    target_dir = cache_dir if cache_dir is not None else (
        default_cache_root() / DEFAULT_CACHE_SUBDIR
    )

    # Stage 1: fetch
    try:
        raw = fetcher()
    except TaimanFetchError as exc:
        if debug:
            raise
        print(f"gblh refresh: fetch failed: {exc}", file=stderr)
        return EXIT_FETCH

    # Stage 2: parse
    try:
        snapshot = parser(raw)
    except TaimanParseError as exc:
        if debug:
            raise
        print(f"gblh refresh: parse failed: {exc}", file=stderr)
        return EXIT_PARSE

    # Stage 3: persist
    try:
        if filename is not None:
            # The persister type does not accept a filename — callers that
            # want one can wrap. Here we call ``write_snapshot`` directly
            # so the override stays on the CLI surface.
            stored = write_snapshot(
                snapshot, cache_dir=target_dir, filename=filename
            )
        else:
            stored = persister(snapshot, target_dir)
    except SnapshotPersistError as exc:
        if debug:
            raise
        print(f"gblh refresh: persist failed: {exc}", file=stderr)
        return EXIT_PERSIST

    # Success path. Two outputs go to stdout:
    #
    #   (a) A short ``wrote snapshot`` line with the on-disk path and
    #       schema version — operational confirmation for the operator.
    #   (b) The canonical snapshot rendering via
    #       ``render_meta_snapshot``, which is the SINGLE sanctioned
    #       path for displaying a snapshot. It unconditionally emits the
    #       data-honesty caveat block (top) and the caveat footer line
    #       (bottom). There is no flag here to suppress the caveat —
    #       AC 6 is enforced structurally, not by convention.
    print(
        (
            "gblh refresh: wrote snapshot\n"
            f"  path:           {stored.path}\n"
            f"  schema_version: {stored.schema_version}"
        ),
        file=stdout,
    )
    stdout.write("\n")
    render_meta_snapshot(
        snapshot,
        stream=stdout,
        lang=lang,
        dex_registry=dex_registry or load_default_registry(),
    )
    return EXIT_OK


def cmd_show(
    *,
    cache_dir: Path | None,
    path: Path | None,
    stdout: TextIO,
    stderr: TextIO,
    lang: str = "ko",
    dex_registry: PokedexRegistry | None = None,
) -> int:
    """Execute the ``show`` subcommand.

    Loads a snapshot from disk and renders it via the canonical
    :func:`gbl_hacker.render.snapshot.render_meta_snapshot`. The renderer
    structurally guarantees the data-honesty caveat appears in the
    output — AC 6 cannot be violated by this code path because there is
    no flag, env var, or argument that hides the caveat.

    Resolution order for the source snapshot:

    1. ``--path FILE`` if supplied — exact JSON file is read.
    2. Otherwise, the *latest* snapshot in the resolved cache directory.

    Returns
    -------
    int
        ``EXIT_OK`` on success; ``EXIT_NO_SNAPSHOT`` when no snapshot
        exists; ``EXIT_PERSIST`` when the snapshot file is unreadable
        or corrupt.
    """

    registry = dex_registry or load_default_registry()
    if path is not None:
        try:
            snapshot = read_snapshot(path)
        except SnapshotPersistError as exc:
            print(f"gblh show: could not read snapshot: {exc}", file=stderr)
            return EXIT_PERSIST
        render_meta_snapshot(snapshot, stream=stdout, lang=lang, dex_registry=registry)
        return EXIT_OK

    target_dir = cache_dir if cache_dir is not None else (
        default_cache_root() / DEFAULT_CACHE_SUBDIR
    )
    files = list_snapshots(target_dir)
    if not files:
        print(
            (
                f"gblh show: no snapshots found in {target_dir} — "
                "run `gblh refresh` first."
            ),
            file=stderr,
        )
        return EXIT_NO_SNAPSHOT

    try:
        snapshot = latest_snapshot(target_dir)
    except SnapshotPersistError as exc:
        print(f"gblh show: could not read snapshot: {exc}", file=stderr)
        return EXIT_PERSIST

    if snapshot is None:  # pragma: no cover — defensive, list_snapshots > 0 ⇒ not None
        print(f"gblh show: snapshot disappeared mid-read in {target_dir}", file=stderr)
        return EXIT_NO_SNAPSHOT

    render_meta_snapshot(snapshot, stream=stdout, lang=lang, dex_registry=registry)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Subcommand: report-rating
# ---------------------------------------------------------------------------


def _resolve_rating_log_path(
    *,
    rating_log_path: Path | None,
    cache_dir: Path | None,
) -> Path:
    """Resolve the on-disk path for the rating-log JSONL store.

    Precedence (highest first):

    1. ``--rating-log-path FILE`` — explicit file, used as-is.
    2. ``--cache-dir DIR`` — store lives at ``DIR / DEFAULT_STORE_FILENAME``.
    3. ``default_cache_root()`` — the OS-conventional cache root, joined
       with ``DEFAULT_STORE_FILENAME``.

    Note that ``report-rating`` does NOT use ``DEFAULT_CACHE_SUBDIR``
    (the ``snapshots/`` sub-directory). The rating log is a different
    artifact from the meta snapshots — putting it next to them, not
    inside the snapshot sub-dir, mirrors the project memory entry on
    long-loop validation and matches what a human would expect when
    they ``ls`` the cache root.
    """

    if rating_log_path is not None:
        return rating_log_path
    if cache_dir is not None:
        return cache_dir / DEFAULT_STORE_FILENAME
    return default_cache_root() / DEFAULT_STORE_FILENAME


def cmd_report_rating(
    *,
    team_id: str,
    pre_rating: int,
    post_rating: int,
    timestamp: str | None,
    notes: str | None,
    rating_log_path: Path | None,
    cache_dir: Path | None,
    debug: bool,
    writer: RatingLogWriterFn,
    now: NowFn,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Execute the ``report-rating`` subcommand.

    Pipeline:

        1. Resolve store path from ``--rating-log-path`` / ``--cache-dir``
           / OS default.
        2. Parse ``--timestamp`` (or default to ``now()``).
        3. Construct a ``RatingLogEntry`` — validation happens inside
           the dataclass and surfaces as ``RatingLogValidationError``.
        4. Hand it to ``writer(entry, path)`` for append.
        5. Print a one-line confirmation to stdout.

    Returns ``EXIT_OK`` on success; ``EXIT_RATING_LOG`` on any
    validation or persistence failure. ``--debug`` re-raises the
    underlying exception instead.
    """

    target_path = _resolve_rating_log_path(
        rating_log_path=rating_log_path,
        cache_dir=cache_dir,
    )

    # Stage 1: timestamp parsing. We do this BEFORE constructing the
    # entry so a malformed --timestamp string yields a precise error
    # message ("'--timestamp' is not valid ISO 8601") rather than a
    # generic dataclass TypeError.
    if timestamp is not None:
        try:
            ts = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            if debug:
                raise
            print(
                (
                    "gblh report-rating: '--timestamp' is not a valid "
                    f"ISO 8601 datetime: {timestamp!r} ({exc})"
                ),
                file=stderr,
            )
            return EXIT_RATING_LOG
    else:
        ts = now()

    # Stage 2: construct the entry. Validation lives in the dataclass —
    # we catch RatingLogValidationError here and turn it into a stderr
    # line so the operator sees "pre_rating must be >= 0" instead of a
    # Python traceback.
    try:
        entry = RatingLogEntry(
            team_id=team_id,
            pre_rating=pre_rating,
            post_rating=post_rating,
            timestamp=ts,
            notes=notes,
        )
    except RatingLogValidationError as exc:
        if debug:
            raise
        print(f"gblh report-rating: invalid entry: {exc}", file=stderr)
        return EXIT_RATING_LOG

    # Stage 3: persist. Any RatingLogError (including a future
    # ValueError raised by the store's embedded-newline guard) maps to
    # EXIT_RATING_LOG; OSError bubbles up here too because disk-full /
    # permission-denied is in the same "operator must fix and retry"
    # bucket as a validation error.
    try:
        writer(entry, target_path)
    except RatingLogError as exc:
        if debug:
            raise
        print(f"gblh report-rating: write failed: {exc}", file=stderr)
        return EXIT_RATING_LOG
    except (OSError, ValueError) as exc:
        if debug:
            raise
        print(f"gblh report-rating: write failed: {exc}", file=stderr)
        return EXIT_RATING_LOG

    # Operational confirmation. We echo the resolved fields so the
    # operator can sanity-check what landed on disk without re-reading
    # the JSONL file.
    print(
        (
            "gblh report-rating: recorded entry\n"
            f"  path:        {target_path}\n"
            f"  team_id:     {entry.team_id}\n"
            f"  pre_rating:  {entry.pre_rating}\n"
            f"  post_rating: {entry.post_rating}\n"
            f"  delta:       {entry.delta:+d}\n"
            f"  timestamp:   {entry.timestamp.isoformat()}"
        ),
        file=stdout,
    )
    return EXIT_OK


def _team_display_name(team, dex_registry, lang: str) -> str:
    """Localized ``lead / safe_swap / closer`` name for trust readouts."""
    from gbl_hacker.render.recommendation import _localize

    shadow_marker = {"ko": "(섀도우)", "ja": "(シャドウ)", "en": "(shadow)"}
    names = []
    for build in team.slots:
        loc = _localize(build.species, registry=dex_registry, lang=lang)
        if build.form_id and "(" not in loc:
            loc += shadow_marker.get(lang, shadow_marker["ja"])
        names.append(loc)
    return " / ".join(names)


def _render_trust_signals(
    top,
    *,
    frontier_size: int,
    opponents: str,
    primary_label: str,
    snapshot,
    registry: dict,
    opponents_size: int,
    set_fn,
    lang: str,
    dex_registry,
    stream: TextIO,
    debug: bool,
) -> None:
    """Render opponent-pool sensitivity + frontier-fragility trust signals.

    Re-scores the top-K teams against the opponent pool *not* used for the
    headline score (ranking ↔ meta) so the divergence is visible, and raises
    a fragility alarm when the Pareto frontier collapsed.
    """
    from gbl_hacker.trust import TrustRow, format_trust_table, pareto_alarm

    # The cross-check pool is whichever pool the primary score did NOT use.
    cross_snapshot = None
    cross_registry: dict = {}
    if opponents == "ranking":
        cross_label = "Taiman meta"
        cross_snapshot, cross_registry = snapshot, registry
    else:
        cross_label = "PvPoke-synthetic"
        try:
            cross_snapshot, cross_synth = synthesize_pvpoke_opponent_meta(
                top_n=opponents_size, meta_snapshot=snapshot
            )
            cross_registry = dict(registry)
            cross_registry.update(cross_synth)
        except Exception:
            if debug:
                raise
            cross_snapshot = None

    rows: list[TrustRow] = []
    if cross_snapshot is not None and cross_snapshot.team_usage:
        for st in top:
            try:
                ewr_cross = expected_win_rate(
                    st.team,
                    cross_snapshot,
                    build_registry=cross_registry,
                    on_missing_build="skip",
                    set_win_rate_fn=set_fn,
                )
            except Exception:
                if debug:
                    raise
                continue
            rows.append(
                TrustRow(
                    name=_team_display_name(st.team, dex_registry, lang),
                    ewr_primary=st.score.expected_win_rate,
                    ewr_cross=ewr_cross,
                )
            )

    if rows:
        print(file=stream)
        print(
            format_trust_table(
                rows, primary_label=primary_label, cross_label=cross_label
            ),
            file=stream,
        )
    alarm = pareto_alarm(frontier_size, opponents_label=primary_label)
    if alarm:
        print(file=stream)
        print(alarm, file=stream)


def cmd_recommend(
    *,
    cache_dir: Path | None,
    path: Path | None,
    top_k: int,
    lang: str,
    debug: bool,
    stdout: TextIO,
    stderr: TextIO,
    dex_registry: PokedexRegistry | None = None,
    engine: str = "set-driver",
    candidates: str = "ranking",
    pool_size: int = 12,
    opponents: str = "ranking",
    opponents_size: int = 30,
    stochastic_samples: int = 5,
    active_switch: bool = False,
    win_mode: str = "ko",
    exclude: str = "",
) -> int:
    """Execute the ``recommend`` subcommand.

    Pipeline:

    1. Load latest cached snapshot (or ``--path``).
    2. Materialize every meta species via the PvPoke gamemaster
       (``build_registry_for_meta``).
    3. Score every ``team_usage`` entry on the three axes.
    4. Pareto-filter the resulting score set.
    5. Rank the Pareto subset and slice to top-K.
    6. Render with the data-honesty caveat surfaced (AC 6).
    """

    if top_k <= 0:
        print(
            f"gblh recommend: --top-k must be >= 1, got {top_k}",
            file=stderr,
        )
        return EXIT_USAGE

    # Stage 1: load snapshot.
    if path is not None:
        try:
            snapshot = read_snapshot(path)
        except SnapshotPersistError as exc:
            print(f"gblh recommend: could not read snapshot: {exc}", file=stderr)
            return EXIT_PERSIST
    else:
        target_dir = cache_dir if cache_dir is not None else (
            default_cache_root() / DEFAULT_CACHE_SUBDIR
        )
        files = list_snapshots(target_dir)
        if not files:
            print(
                (
                    f"gblh recommend: no snapshots found in {target_dir} — "
                    "run `gblh refresh` first."
                ),
                file=stderr,
            )
            return EXIT_NO_SNAPSHOT
        try:
            loaded = latest_snapshot(target_dir)
        except SnapshotPersistError as exc:
            print(f"gblh recommend: could not read snapshot: {exc}", file=stderr)
            return EXIT_PERSIST
        if loaded is None:  # pragma: no cover - defensive
            print("gblh recommend: snapshot disappeared mid-read", file=stderr)
            return EXIT_NO_SNAPSHOT
        snapshot = loaded

    # Stage 2a: build registry from the Taiman snapshot. Always used as
    # the basis for ``--candidates pool`` lookups, and as the fallback
    # opponent-build source.
    try:
        registry = build_registry_for_meta(snapshot)
    except Exception as exc:
        if debug:
            raise
        print(f"gblh recommend: build registry failed: {exc}", file=stderr)
        return EXIT_RECOMMEND

    # Stage 2b: pick the evaluation snapshot. When ``--opponents
    # ranking``, synthesize a ladder-shape opponent set from PvPoke's
    # ranking so wcr/cov reflect what the simulator thinks of the
    # broader meta — not just the 30 most-popular Taiman lineups.
    if opponents == "ranking":
        try:
            # Pass the live Taiman snapshot so the synthetic opponent
            # set blends PvPoke leads with Taiman top-N most-used
            # species — closes the sample-bias gap where species like
            # Quagsire (Taiman cnt 1019, PvPoke leads list absent) get
            # under-represented as opponents.
            eval_snapshot, synthetic_registry = synthesize_pvpoke_opponent_meta(
                top_n=opponents_size,
                meta_snapshot=snapshot,
            )
        except Exception as exc:
            if debug:
                raise
            print(
                f"gblh recommend: synthetic opponent set failed: {exc}",
                file=stderr,
            )
            return EXIT_RECOMMEND
        # Merge so opponent-team materialization finds its builds first.
        eval_registry: dict = dict(registry)
        eval_registry.update(synthetic_registry)
    else:
        eval_snapshot = snapshot
        eval_registry = registry

    if not registry:
        print(
            (
                "gblh recommend: build registry is empty — none of the meta "
                "species could be materialized from the PvPoke gamemaster."
            ),
            file=stderr,
        )
        return EXIT_RECOMMEND

    if not eval_snapshot.team_usage:
        print(
            "gblh recommend: opponent snapshot has no team_usage rows.",
            file=stderr,
        )
        return EXIT_RECOMMEND

    # Stage 3: build the candidate list.
    if engine == "set-driver":
        if stochastic_samples > 1 or active_switch or win_mode != "ko":
            def _set_fn_configured(a, b):  # type: ignore[no-untyped-def]
                return set_driver_win_rate(
                    a,
                    b,
                    stochastic_samples=stochastic_samples,
                    active_switch=active_switch,
                    win_mode=win_mode,
                )

            set_fn = _set_fn_configured
        else:
            set_fn = set_driver_win_rate
    else:
        set_fn = default_set_win_rate
    scored: list[ScoredTeam] = []
    skipped: list[tuple[str, ...]] = []

    candidate_teams: list[tuple[CandidateTeam, tuple[str, ...]]] = []

    # Parse --exclude into a set of species labels. Match against both
    # the raw Japanese species_ja (e.g. ``サニーゴ(ガラル)``) and the
    # localized Korean / English names from the dex registry so an
    # operator can type "코산호" or "Corsola (Galarian)" too.
    exclude_set: set[str] = set()
    if exclude.strip():
        registry_dex_for_exclude = dex_registry or load_default_registry()
        raw_tokens = [t.strip() for t in exclude.split(",") if t.strip()]
        exclude_set.update(raw_tokens)
        # Build a label → species_ja reverse index over the dex registry
        # so localized labels resolve to the JA name used by builds.
        for entry in registry_dex_for_exclude.by_dex.values():
            if entry.ko in raw_tokens or entry.en in raw_tokens:
                exclude_set.add(entry.ja)

    def _is_excluded(build: "CombatantBuild") -> bool:  # type: ignore[name-defined]
        if not exclude_set:
            return False
        # The build's species is the JA base name. Direct match wins.
        if build.species in exclude_set:
            return True
        # Strip parenthesized variants for a softer match too —
        # "サニーゴ" excludes both base and (ガラル).
        bare = build.species.split("(")[0].strip()
        if bare and bare in exclude_set:
            return True
        return False

    if candidates in ("pool", "ranking"):
        if pool_size <= 2:
            print(
                f"gblh recommend: --pool-size must be >= 3, got {pool_size}",
                file=stderr,
            )
            return EXIT_USAGE
        from gbl_hacker.build_registry import registry_key as _reg_key

        pool_builds: list[tuple[str, "CombatantBuild"]] = []  # type: ignore[name-defined]

        if candidates == "ranking":
            # PvPoke top-N GL ranking — the simulator's view of what is
            # strongest in the abstract, independent of Taiman Party
            # popularity. Picks niche carries the JP meta underuses.
            # Walk a deeper slice of the ranking when exclude trims hits.
            scan_top_n = pool_size + 4 * max(1, len(exclude_set))
            for label, _species_id, build in build_registry_pvpoke_top(
                top_n=scan_top_n
            ):
                if _is_excluded(build):
                    continue
                pool_builds.append((label, build))
                if len(pool_builds) >= pool_size:
                    break
        else:
            # Top-N Taiman Party-used species. Form-aware lookup so
            # shadow / regional variants are picked up correctly.
            for entry in snapshot.pokemon_usage:
                key = _reg_key(entry.species, entry.form_id or 0)
                build = registry.get(key)
                if build is None and (entry.form_id or 0):
                    build = registry.get(entry.species)
                if build is None:
                    continue
                if _is_excluded(build):
                    continue
                pool_builds.append((entry.species, build))
                if len(pool_builds) >= pool_size:
                    break
        if len(pool_builds) < 3:
            print(
                (
                    "gblh recommend: candidate pool too thin — fewer than 3 "
                    f"species materialized from --pool-size {pool_size}."
                ),
                file=stderr,
            )
            return EXIT_RECOMMEND
        # Ordered 3-combos with **dex-unique** picks per the GBL rule —
        # base + shadow + regional variants of the same dex cannot share
        # a team. When a pool build has dex_id=0 (hand-built fixtures or
        # legacy entries), we fall back to species-name dedup to stay
        # compatible with older tests.
        for i, (s1, b1) in enumerate(pool_builds):
            for j, (s2, b2) in enumerate(pool_builds):
                if j == i:
                    continue
                if b1.dex_id and b2.dex_id and b1.dex_id == b2.dex_id:
                    continue
                if not (b1.dex_id and b2.dex_id) and s1 == s2:
                    continue
                for k, (s3, b3) in enumerate(pool_builds):
                    if k == i or k == j:
                        continue
                    if b3.dex_id and (
                        b3.dex_id == b1.dex_id or b3.dex_id == b2.dex_id
                    ):
                        continue
                    if not b3.dex_id and s3 in {s1, s2}:
                        continue
                    cand = CandidateTeam(lead=b1, safe_swap=b2, closer=b3)
                    candidate_teams.append((cand, (s1, s2, s3)))
    else:
        for team_usage in snapshot.team_usage:
            try:
                cand = materialize_opponent_team(team_usage, registry)
            except Exception:
                skipped.append(team_usage.members)
                continue
            candidate_teams.append((cand, team_usage.members))

    if not candidate_teams:
        print(
            "gblh recommend: no candidate team could be built.",
            file=stderr,
        )
        return EXIT_RECOMMEND

    # Stage 4: score every candidate.
    opponent_label = (
        "Taiman meta" if opponents == "meta" else "PvPoke-synthetic"
    )
    print(
        (
            f"gblh recommend: scoring {len(candidate_teams)} candidate(s) "
            f"via {engine} engine against {len(eval_snapshot.team_usage)} "
            f"{opponent_label} opponent team(s)…"
        ),
        file=stderr,
    )
    for candidate, label_members in candidate_teams:
        try:
            ewr = expected_win_rate(
                candidate,
                eval_snapshot,
                build_registry=eval_registry,
                on_missing_build="skip",
                set_win_rate_fn=set_fn,
            )
            wcr = worst_case_robustness(
                candidate,
                eval_snapshot,
                build_registry=eval_registry,
                on_missing_build="skip",
                set_win_rate_fn=set_fn,
            )
            cov = meta_coverage(
                candidate,
                eval_snapshot,
                build_registry=eval_registry,
                on_missing_build="skip",
                set_win_rate_fn=set_fn,
            )
        except Exception as exc:
            if debug:
                raise
            print(
                f"gblh recommend: scoring failed for {label_members}: {exc}",
                file=stderr,
            )
            continue
        scored.append(
            ScoredTeam(
                team=candidate,
                score=Score(
                    expected_win_rate=ewr,
                    worst_case_robustness=wcr,
                    meta_coverage=cov,
                ),
            )
        )

    if not scored:
        print(
            "gblh recommend: no candidate team could be scored.",
            file=stderr,
        )
        return EXIT_RECOMMEND

    # Stage 4 + 5: Pareto filter, then top-K rank.
    frontier = pareto_filter(scored)
    top = rank_top_k(frontier, k=top_k)

    # Stage 6: compute meta-popular team scores separately so the render
    # layer can surface the simulator's verdict on the JP site's most-
    # used teams next to the Pareto-optimal pool recommendation. In
    # ``--candidates meta`` mode this is the same list as ``scored``,
    # so we reuse it; in ``--candidates pool`` mode we re-score the 30
    # meta teams in addition to the pool combos.
    meta_scored: list[ScoredTeam]
    if candidates == "meta":
        meta_scored = scored
    else:
        meta_scored = []
        for team_usage in snapshot.team_usage:
            try:
                meta_cand = materialize_opponent_team(team_usage, registry)
            except Exception:
                continue
            try:
                ewr = expected_win_rate(
                    meta_cand,
                    eval_snapshot,
                    build_registry=eval_registry,
                    on_missing_build="skip",
                    set_win_rate_fn=set_fn,
                )
                wcr = worst_case_robustness(
                    meta_cand,
                    eval_snapshot,
                    build_registry=eval_registry,
                    on_missing_build="skip",
                    set_win_rate_fn=set_fn,
                )
                cov = meta_coverage(
                    meta_cand,
                    eval_snapshot,
                    build_registry=eval_registry,
                    on_missing_build="skip",
                    set_win_rate_fn=set_fn,
                )
            except Exception:
                continue
            meta_scored.append(
                ScoredTeam(
                    team=meta_cand,
                    score=Score(
                        expected_win_rate=ewr,
                        worst_case_robustness=wcr,
                        meta_coverage=cov,
                    ),
                )
            )

    registry_dex = dex_registry or load_default_registry()
    render_recommendation_table(
        top,
        snapshot=snapshot,
        stream=stdout,
        lang=lang,  # type: ignore[arg-type]
        dex_registry=registry_dex,
        pareto_size=len(frontier),
        all_scored=meta_scored,
    )

    # Trust signals: opponent-pool sensitivity + frontier fragility.
    try:
        _render_trust_signals(
            top,
            frontier_size=len(frontier),
            opponents=opponents,
            primary_label=opponent_label,
            snapshot=snapshot,
            registry=registry,
            opponents_size=opponents_size,
            set_fn=set_fn,
            lang=lang,
            dex_registry=registry_dex,
            stream=stdout,
            debug=debug,
        )
    except Exception as exc:  # never fail the run over a trust readout
        if debug:
            raise
        print(f"gblh recommend: trust signal skipped: {exc}", file=stderr)

    if skipped:
        print(
            (
                f"gblh recommend: skipped {len(skipped)} team(s) the gamemaster "
                "could not resolve (first 3: "
                + ", ".join("/".join(m) for m in skipped[:3])
                + ")"
            ),
            file=stderr,
        )

    return EXIT_OK


def cmd_verify_reference(
    *,
    recommendations: Path,
    reference: Path,
    threshold: float,
    axis: str,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Execute the ``verify-reference`` subcommand (Sub-AC 5.3).

    Loads the frozen recommendation fixture and the reference team-list
    fixture, runs :func:`gbl_hacker.reference.verify_overlap`, prints a
    human-readable verdict block, and returns an exit code that reflects
    the verdict (``EXIT_OK`` on pass, :data:`EXIT_VERIFY_FAIL` on
    fail). Fixture-load failures map to :data:`EXIT_VERIFY_LOAD` so CI
    gates can distinguish "engine disagrees with reference" from
    "fixtures were missing or malformed".

    Parameters
    ----------
    recommendations:
        Path to the frozen engine-recommendations JSON fixture.
    reference:
        Path to the independent reference-team-list JSON fixture.
    threshold:
        Minimum Jaccard for a ``"pass"`` verdict; must be in
        ``[0.0, 1.0]``.
    axis:
        Either ``"team"`` (unordered species triples) or
        ``"pokemon"`` (individual species rosters). argparse already
        rejects other values, so we only assert defensively below.
    stdout, stderr:
        Output streams.

    Returns
    -------
    int
        ``EXIT_OK`` on pass; ``EXIT_VERIFY_FAIL`` on fail;
        ``EXIT_VERIFY_LOAD`` on a fixture-load error;
        ``EXIT_USAGE`` for an out-of-range threshold or unknown axis
        (defense-in-depth — argparse already filters these).
    """

    if axis not in ("team", "pokemon"):
        # Defense-in-depth: argparse already filters via ``choices=``.
        print(
            f"gblh verify-reference: invalid --axis {axis!r}",
            file=stderr,
        )
        return EXIT_USAGE

    if not (0.0 <= threshold <= 1.0):
        print(
            (
                "gblh verify-reference: --threshold out of range: "
                f"{threshold} (must be in [0.0, 1.0])"
            ),
            file=stderr,
        )
        return EXIT_USAGE

    # Stage 1: load recommendations fixture.
    try:
        recs = load_recommendations_fixture(recommendations)
    except ReferenceLoadError as exc:
        print(
            f"gblh verify-reference: could not load recommendations: {exc}",
            file=stderr,
        )
        return EXIT_VERIFY_LOAD

    # Stage 2: load reference fixture.
    try:
        ref = load_reference_team_list(reference)
    except ReferenceLoadError as exc:
        print(
            f"gblh verify-reference: could not load reference: {exc}",
            file=stderr,
        )
        return EXIT_VERIFY_LOAD

    # Stage 3: compute verdict.
    verdict = verify_overlap(
        recs.teams,
        ref,
        threshold=threshold,
        axis=axis,  # type: ignore[arg-type]  # validated above
    )

    # Stage 4: render. Always render — pass and fail both go to stdout
    # so a CI log captures the same shape regardless of outcome.
    summary = format_verdict_summary(
        verdict,
        reference_source=ref.source,
        recommendation_source=recs.source,
    )
    print(summary, file=stdout)

    return EXIT_OK if verdict.passed else EXIT_VERIFY_FAIL


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    fetcher: FetcherFn | None = None,
    parser: ParserFn | None = None,
    persister: PersisterFn | None = None,
    rating_log_writer: RatingLogWriterFn | None = None,
    now: NowFn | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return an exit code.

    Parameters
    ----------
    argv:
        Argument list excluding the program name. ``None`` uses
        ``sys.argv[1:]``.
    fetcher, parser, persister:
        Optional DI seams for the refresh pipeline. ``None`` binds the
        production implementations. Tests inject spies / fakes here.
    rating_log_writer:
        Optional DI seam for the ``report-rating`` store append. ``None``
        binds ``rating_log.append_entry``. Tests inject a spy that
        captures ``(entry, path)`` to assert what the CLI wrote.
    now:
        Optional DI seam for "wall-clock now". ``None`` binds UTC now.
        Tests inject a fixed clock so the persisted timestamp is
        deterministic.
    stdout, stderr:
        Output streams. ``None`` binds ``sys.stdout`` / ``sys.stderr``.
        Tests typically pass an ``io.StringIO()`` to inspect what the CLI
        wrote.

    Returns
    -------
    int
        A stable exit code (see ``EXIT_*`` constants).
    """

    argv_list = list(sys.argv[1:] if argv is None else argv)
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    arg_parser = build_arg_parser()

    # argparse calls sys.exit on usage error; capture that so main() always
    # returns an int — the test runner asserts on the int, not on SystemExit.
    try:
        args = arg_parser.parse_args(argv_list)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else EXIT_USAGE

    if args.command == "refresh":
        return cmd_refresh(
            cache_dir=args.cache_dir,
            filename=args.filename,
            debug=args.debug,
            fetcher=fetcher or _default_fetcher,
            parser=parser or _default_parser,
            persister=persister or _default_persister,
            stdout=out,
            stderr=err,
            lang=args.lang,
        )

    if args.command == "show":
        return cmd_show(
            cache_dir=args.cache_dir,
            path=args.path,
            stdout=out,
            stderr=err,
            lang=args.lang,
        )

    if args.command == "report-rating":
        return cmd_report_rating(
            team_id=args.team_id,
            pre_rating=args.pre_rating,
            post_rating=args.post_rating,
            timestamp=args.timestamp,
            notes=args.notes,
            rating_log_path=args.rating_log_path,
            cache_dir=args.cache_dir,
            debug=args.debug,
            writer=rating_log_writer or _default_rating_log_writer,
            now=now or _default_now,
            stdout=out,
            stderr=err,
        )

    if args.command == "recommend":
        return cmd_recommend(
            cache_dir=args.cache_dir,
            path=args.path,
            top_k=args.top_k,
            lang=args.lang,
            debug=args.debug,
            stdout=out,
            stderr=err,
            engine=args.engine,
            candidates=args.candidates,
            pool_size=args.pool_size,
            opponents=args.opponents,
            opponents_size=args.opponents_size,
            stochastic_samples=args.stochastic_samples,
            active_switch=args.active_switch,
            win_mode=args.win_mode,
            exclude=args.exclude,
        )

    if args.command == "verify-reference":
        return cmd_verify_reference(
            recommendations=args.recommendations,
            reference=args.reference,
            threshold=args.threshold,
            axis=args.axis,
            stdout=out,
            stderr=err,
        )

    # Defensive — argparse would have rejected an unknown subcommand
    # already, so this is unreachable under normal invocation.
    print(f"gblh: unknown command: {args.command!r}", file=err)
    return EXIT_USAGE


def run() -> None:
    """``console_scripts`` entry point — wraps ``main()`` in ``sys.exit``."""

    sys.exit(main())


__all__ = [
    "EXIT_FETCH",
    "EXIT_NO_SNAPSHOT",
    "EXIT_OK",
    "EXIT_PARSE",
    "EXIT_PERSIST",
    "EXIT_RATING_LOG",
    "EXIT_USAGE",
    "EXIT_VERIFY_FAIL",
    "EXIT_VERIFY_LOAD",
    "FetcherFn",
    "NowFn",
    "ParserFn",
    "PersisterFn",
    "RatingLogWriterFn",
    "build_arg_parser",
    "cmd_refresh",
    "cmd_report_rating",
    "cmd_show",
    "cmd_verify_reference",
    "default_cache_root",
    "main",
    "run",
]
