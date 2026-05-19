"""Predicted-vs-actual joiner for the long-loop validation feedback path.

Sub-AC 7.4 contract: this module **joins** logged
:class:`~gbl_hacker.rating_log.RatingLogEntry` records (the operator's
real-life GBL outcomes) with their originating recommended team's
predicted scores (the engine's pre-game :class:`Score`) and emits a
**per-team predicted-vs-actual summary** plus a population-level
aggregate. The summary is what closes the long-loop validation loop the
seed pins down in ``exit_conditions.long_loop_validation``:

    "User has run at least one recommended team in actual GBL and
    reported the resulting rating change back to the engine log."

Without this aggregator the rating-log JSONL stream is a write-only
journal — the operator can append entries, but the engine never reads
them back against the recommendations they were paired with. With the
aggregator the engine can answer "did the recommended team perform as
predicted?" — which is the only honest definition of "the engine helped
the user climb" for v0.1.

Design choices and why they exist
---------------------------------

* **Join key is ``team_id``, by Mapping.** The rating log records a
  free-form ``team_id`` string (the operator decides how to name the
  team — see :class:`RatingLogEntry`). The engine's recommendation list
  is a sequence of :class:`ScoredTeam` instances that, in principle,
  could be canonicalized to a derived ``team_id`` in many ways
  (hyphen-joined species, slot tuple, hash, …). Rather than commit this
  module to one canonicalization, the join is driven by a
  caller-supplied ``Mapping[str, ScoredTeam]``. The caller picks the
  team-id convention; the aggregator does not invent one. This keeps
  the module compatible with both today's hand-typed CLI entries and a
  future CLI command that auto-generates the team-id from the
  recommendation table.

* **Orphans are surfaced, not silently dropped.** An entry whose
  ``team_id`` is not in the recommendation mapping ends up in
  :attr:`ValidationLogAggregate.orphaned_entries`, NOT in the join
  output. Silently dropping it would be the data-honesty anti-pattern
  the seed forbids: an operator who typed a wrong team-id should see
  that they typed a wrong team-id, not have the engine pretend the
  session never happened.

* **Recommendations without entries are surfaced too.** A
  :class:`ScoredTeam` that the operator has *not* run yet still gets a
  :class:`TeamValidationSummary` — with ``run_count == 0``,
  ``mean_delta is None``, and the predicted scores intact. This makes
  the aggregator a natural backing store for a "which of these
  recommended teams have I actually tested?" CLI view, without forcing
  the caller to maintain a separate "tested?" flag.

* **Residuals are deltas, not ratios.** ``win_rate_residual =
  actual_session_win_rate - predicted_win_rate``. A positive residual
  means the team beat its predicted EV; a negative residual means it
  under-performed. Using a ratio (``actual / predicted``) would blow up
  on ``predicted == 0`` and would obscure the sign of the gap.

* **Session-level "actual win rate" is honestly approximate.** A GBL
  *session* (one ``RatingLogEntry``) is typically a best-of-5 set of
  matchmaking sets — i.e. one entry represents multiple in-game sets.
  The engine's ``predicted_win_rate`` is a per-set EV. They are not
  byte-comparable, and this module documents the gap rather than
  hiding it: :attr:`TeamValidationSummary.actual_session_win_rate` is
  the fraction of *sessions* (not sets) with positive delta. The
  residual is a useful debugging gauge for "is the team's predicted
  vs. observed performance trending the wrong way", not a calibrated
  probabilistic comparison. The docstring on each property restates
  this so a downstream UI does not silently over-claim precision.

* **Deterministic order.** The output ``summaries`` follow the
  ``recommendations`` mapping's iteration order (Python 3.7+ dict
  insertion order is guaranteed). The ``orphaned_entries`` follow the
  input ``entries`` iteration order. This means two runs over the same
  inputs produce byte-identical output — which matters for the
  rationale-card / CLI rendering layer and for any snapshot-based
  regression test.

* **Frozen dataclasses + immutable tuples.** Both
  :class:`TeamValidationSummary` and :class:`ValidationLogAggregate`
  are ``frozen=True, slots=True``. Their ``entries`` /
  ``orphaned_entries`` / ``summaries`` are stored as ``tuple`` so a
  caller cannot accidentally mutate the aggregate after construction.
  This mirrors the convention in :mod:`gbl_hacker.score.pareto` and
  :mod:`gbl_hacker.rating_log.entry`.

* **Pure function, no I/O.** :func:`aggregate_validation_log` is a
  pure transformation of in-memory data. It does **not** read the
  rating-log JSONL store (callers feed it
  :func:`gbl_hacker.rating_log.read_entries`'s output) and does **not**
  re-simulate the team (callers feed it pre-computed
  :class:`ScoredTeam` instances from the score pipeline). Keeping the
  aggregator pure means the unit tests are deterministic and the same
  function can run over a CLI's in-memory state, a frozen test
  fixture, or a future replay tool's reconstructed history without
  modification.

Public surface
--------------

* :class:`TeamValidationSummary`  — per-team predicted-vs-actual record.
* :class:`ValidationLogAggregate` — population-level join container.
* :func:`aggregate_validation_log` — the headline Sub-AC 7.4 function.

The module is import-cycle safe: it depends on
:mod:`gbl_hacker.rating_log.entry` (data model only) and
:mod:`gbl_hacker.score.pareto` (the ``ScoredTeam`` /``Score`` data
classes only). Both upstreams are leaf modules with respect to the
engine's import graph, so no cycle is introduced.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from gbl_hacker.rating_log.entry import RatingLogEntry
from gbl_hacker.score.pareto import ScoredTeam


# ---------------------------------------------------------------------------
# Per-team summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamValidationSummary:
    """Predicted-vs-actual aggregate for one team in the recommendation set.

    The summary joins a single :class:`ScoredTeam` (the engine's
    pre-game prediction) with the matching subset of
    :class:`RatingLogEntry` records (the operator's post-game reports).
    "Matching" means the entry's ``team_id`` equals the key the caller
    used for this team in the recommendation mapping passed to
    :func:`aggregate_validation_log`.

    Attributes
    ----------
    team_id:
        The join key. Matches both the entries' ``team_id`` and the
        recommendation mapping's key — preserved on the summary so a
        downstream renderer can label rows without re-threading the key
        through a separate channel.
    scored_team:
        The :class:`ScoredTeam` from the recommendation mapping, or
        ``None`` when the summary was produced for an orphaned-team
        construction path (currently unused by
        :func:`aggregate_validation_log` itself, but the field is
        ``None``-able so callers can hand-construct a summary for an
        out-of-band rendering — e.g. "this team_id has entries but no
        prediction").
    entries:
        Tuple of every :class:`RatingLogEntry` whose ``team_id``
        matches this summary's join key. Empty tuple is legal — that
        is exactly the "recommended but not yet tested" case the
        long-loop validation log is supposed to surface.

    Notes
    -----
    Every "actual"-side property returns ``None`` when ``entries`` is
    empty — there is no statistic to report yet. Every "predicted"-side
    property returns ``None`` when ``scored_team is None``. Residuals
    return ``None`` when *either* side is missing. The ``None`` -versus-
    zero distinction matters: a team with ``run_count == 0`` is not the
    same as a team that ran zero winning sessions.
    """

    team_id: str
    scored_team: ScoredTeam | None
    entries: tuple[RatingLogEntry, ...]

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.team_id, str):
            raise TypeError(
                f"team_id must be str, got {type(self.team_id).__name__}"
            )
        if self.scored_team is not None and not isinstance(
            self.scored_team, ScoredTeam
        ):
            raise TypeError(
                "scored_team must be ScoredTeam or None, got "
                f"{type(self.scored_team).__name__}"
            )
        if not isinstance(self.entries, tuple):
            raise TypeError(
                "entries must be a tuple of RatingLogEntry, got "
                f"{type(self.entries).__name__}"
            )
        for idx, entry in enumerate(self.entries):
            if not isinstance(entry, RatingLogEntry):
                raise TypeError(
                    f"entries[{idx}] must be RatingLogEntry, got "
                    f"{type(entry).__name__}"
                )

    # ------------------------------------------------------------------
    # Count-based "actual" statistics
    #
    # These are direct counts and ratios over ``self.entries``. They
    # are *not* probability estimates of per-set win rate; see the
    # module docstring's "Session-level actual win rate is honestly
    # approximate" note.
    # ------------------------------------------------------------------

    @property
    def run_count(self) -> int:
        """Number of logged sessions for this team. ``0`` is legal."""

        return len(self.entries)

    @property
    def win_count(self) -> int:
        """Sessions whose ``delta > 0`` — net-winning sessions."""

        return sum(1 for e in self.entries if e.delta > 0)

    @property
    def loss_count(self) -> int:
        """Sessions whose ``delta < 0`` — net-losing sessions."""

        return sum(1 for e in self.entries if e.delta < 0)

    @property
    def even_count(self) -> int:
        """Sessions whose ``delta == 0`` — net-even sessions.

        A net-even session is rare in real GBL Elo math but is a valid
        outcome on a short session that the operator chose to log.
        Surfacing it separately means the win/loss counts add up to
        ``run_count`` without ambiguity::

            run_count == win_count + loss_count + even_count
        """

        return sum(1 for e in self.entries if e.delta == 0)

    @property
    def total_delta(self) -> int:
        """Sum of ``delta`` across all logged sessions for this team.

        Integer-typed: ``RatingLogEntry.delta`` is integer and the sum
        of integers is an integer. Downstream callers that want a
        float (mean, residual) should use the dedicated property.
        """

        return sum(e.delta for e in self.entries)

    @property
    def mean_delta(self) -> float | None:
        """Mean rating delta per session, or ``None`` if no sessions.

        Floating-point intentionally — averaging integer deltas
        produces a fractional value for any session count > 1. The
        ``None`` -versus-``0.0`` distinction is meaningful: an
        un-tested team has no mean to report; a team with one
        zero-delta session has ``0.0``.
        """

        if not self.entries:
            return None
        return self.total_delta / len(self.entries)

    @property
    def actual_session_win_rate(self) -> float | None:
        """Fraction of *sessions* (not sets) with positive delta.

        ``win_count / run_count`` when at least one session exists,
        ``None`` otherwise. NOT directly comparable to
        :attr:`predicted_win_rate` — a session is multiple in-game
        sets, while the predicted EV is per-set. The residual is a
        debugging gauge, not a calibrated probability comparison; see
        the module docstring.
        """

        if not self.entries:
            return None
        return self.win_count / len(self.entries)

    @property
    def actual_non_loss_rate(self) -> float | None:
        """Fraction of sessions with non-negative delta (wins + evens).

        Used as the actual-side counterpart for
        :attr:`predicted_robustness`. The interpretation is "how often
        the team avoided a losing session" — a robustness-style metric
        rather than a win-rate-style metric, matching the engine's
        worst-case-robustness axis. Returns ``None`` on an un-tested
        team.
        """

        if not self.entries:
            return None
        return (self.win_count + self.even_count) / len(self.entries)

    # ------------------------------------------------------------------
    # Predicted side
    # ------------------------------------------------------------------

    @property
    def predicted_win_rate(self) -> float | None:
        """Engine's ``expected_win_rate`` for this team, or ``None``."""

        if self.scored_team is None:
            return None
        return self.scored_team.score.expected_win_rate

    @property
    def predicted_robustness(self) -> float | None:
        """Engine's ``worst_case_robustness`` for this team, or ``None``."""

        if self.scored_team is None:
            return None
        return self.scored_team.score.worst_case_robustness

    @property
    def predicted_meta_coverage(self) -> float | None:
        """Engine's ``meta_coverage`` for this team, or ``None``."""

        if self.scored_team is None:
            return None
        return self.scored_team.score.meta_coverage

    # ------------------------------------------------------------------
    # Residuals (actual - predicted)
    #
    # Positive residual ⇒ the team beat the engine's prediction.
    # Negative residual ⇒ the team under-performed the prediction.
    # ``None`` ⇒ either side is missing; no comparison is possible.
    # ------------------------------------------------------------------

    @property
    def win_rate_residual(self) -> float | None:
        """``actual_session_win_rate - predicted_win_rate``, or ``None``.

        Honest-approximation caveat: see the module docstring. Sign is
        what matters: + means "team beat the engine's call", - means
        "team trailed the engine's call". Magnitude is a debugging
        gauge, not a calibrated probability gap.
        """

        actual = self.actual_session_win_rate
        predicted = self.predicted_win_rate
        if actual is None or predicted is None:
            return None
        return actual - predicted

    @property
    def robustness_residual(self) -> float | None:
        """``actual_non_loss_rate - predicted_robustness``, or ``None``.

        Same sign convention as :attr:`win_rate_residual`. The actual
        side here uses the non-loss rate (wins + evens) to align with
        the engine's robustness-style axis — "the team avoided a
        losing session at rate R". A team whose
        ``actual_non_loss_rate`` is well below
        ``predicted_robustness`` is one whose worst-case prediction
        the operator should re-examine.
        """

        actual = self.actual_non_loss_rate
        predicted = self.predicted_robustness
        if actual is None or predicted is None:
            return None
        return actual - predicted


# ---------------------------------------------------------------------------
# Population-level aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationLogAggregate:
    """Population-level container — every per-team summary + orphans.

    Returned by :func:`aggregate_validation_log`. Carries one
    :class:`TeamValidationSummary` per team in the recommendation
    mapping (in the mapping's iteration order) plus a tuple of
    :class:`RatingLogEntry` records whose ``team_id`` did not match
    any team in the mapping (in the entries iterator's input order).

    Attributes
    ----------
    summaries:
        One per team in the recommendation mapping. A team without any
        matching entries still appears here with empty ``entries`` —
        the un-tested-team case is a feature, not a missing record.
    orphaned_entries:
        Entries the join could not place. Surfaced (not dropped) so
        the operator can spot typo'd ``team_id`` values or
        recommendations that have rolled out of the current
        recommendation set.

    Notes
    -----
    The aggregate properties (``total_runs``, ``total_delta``, …)
    include orphans where it makes sense, with explicit docstrings on
    each property. The motivation is the long-loop validation
    exit-condition: ``rating_change_log entries >= 1`` is satisfied by
    *any* logged session, including orphans — but a Pareto-recommended
    team's predicted-vs-actual gap is only meaningful for the joined
    slice. The aggregate exposes both views without conflating them.
    """

    summaries: tuple[TeamValidationSummary, ...]
    orphaned_entries: tuple[RatingLogEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.summaries, tuple):
            raise TypeError(
                "summaries must be a tuple of TeamValidationSummary, got "
                f"{type(self.summaries).__name__}"
            )
        for idx, summary in enumerate(self.summaries):
            if not isinstance(summary, TeamValidationSummary):
                raise TypeError(
                    f"summaries[{idx}] must be TeamValidationSummary, got "
                    f"{type(summary).__name__}"
                )
        if not isinstance(self.orphaned_entries, tuple):
            raise TypeError(
                "orphaned_entries must be a tuple of RatingLogEntry, got "
                f"{type(self.orphaned_entries).__name__}"
            )
        for idx, entry in enumerate(self.orphaned_entries):
            if not isinstance(entry, RatingLogEntry):
                raise TypeError(
                    f"orphaned_entries[{idx}] must be RatingLogEntry, got "
                    f"{type(entry).__name__}"
                )

    # ------------------------------------------------------------------
    # Joined-slice aggregates (exclude orphans)
    # ------------------------------------------------------------------

    @property
    def joined_run_count(self) -> int:
        """Total sessions across all teams in the recommendation set.

        Excludes :attr:`orphaned_entries`. This is the count that
        feeds "did the engine's recommendations get tested?" — a
        non-zero value satisfies the seed's
        ``rating_change_log entries >= 1`` exit-condition for the
        joined slice specifically.
        """

        return sum(s.run_count for s in self.summaries)

    @property
    def joined_total_delta(self) -> int:
        """Sum of ``delta`` across joined-slice sessions only.

        Positive value ⇒ the operator's recommended-team sessions net
        rating-positive. Negative value ⇒ the recommendations are
        under-performing on net.
        """

        return sum(s.total_delta for s in self.summaries)

    @property
    def joined_mean_delta(self) -> float | None:
        """Mean ``delta`` over joined-slice sessions, or ``None`` if zero.

        ``None`` rather than ``0.0`` for the empty-input case — same
        rationale as :attr:`TeamValidationSummary.mean_delta`. An
        un-tested recommendation set has no mean to report.
        """

        n = self.joined_run_count
        if n == 0:
            return None
        return self.joined_total_delta / n

    @property
    def teams_with_entries(self) -> int:
        """Number of teams with at least one matched entry."""

        return sum(1 for s in self.summaries if s.run_count > 0)

    @property
    def teams_without_entries(self) -> int:
        """Number of teams with no matched entries — i.e. un-tested."""

        return sum(1 for s in self.summaries if s.run_count == 0)

    # ------------------------------------------------------------------
    # Full-log aggregates (include orphans)
    # ------------------------------------------------------------------

    @property
    def total_runs(self) -> int:
        """Every logged session, including orphans.

        This is the count that feeds the seed's long-loop validation
        exit-condition (``rating_change_log entries >= 1``) regardless
        of whether the team_id was joinable — any logged session
        counts toward "the user has run at least one team and reported
        the result".
        """

        return self.joined_run_count + len(self.orphaned_entries)

    @property
    def total_delta(self) -> int:
        """Sum of ``delta`` across every logged session, including orphans.

        Includes orphans because the rating change is real regardless
        of whether the engine can pair the session with a prediction.
        """

        orphan_delta = sum(e.delta for e in self.orphaned_entries)
        return self.joined_total_delta + orphan_delta

    @property
    def orphaned_count(self) -> int:
        """Number of entries whose ``team_id`` was not in the join mapping."""

        return len(self.orphaned_entries)


# ---------------------------------------------------------------------------
# Headline aggregator
# ---------------------------------------------------------------------------


def aggregate_validation_log(
    entries: Iterable[RatingLogEntry],
    recommendations: Mapping[str, ScoredTeam],
) -> ValidationLogAggregate:
    """Join entries to recommendations by ``team_id`` and return the aggregate.

    The headline Sub-AC 7.4 function. Walks ``entries`` once, partitions
    them into a per-``team_id`` bucket using ``recommendations`` as the
    join lookup, then constructs one
    :class:`TeamValidationSummary` per team in the recommendation
    mapping (in mapping iteration order) plus a tuple of orphaned
    entries (those whose ``team_id`` was not in the mapping, in input
    order).

    Parameters
    ----------
    entries:
        Iterable of :class:`RatingLogEntry` records — typically the
        output of :func:`gbl_hacker.rating_log.read_entries`. Consumed
        exactly once; works on generators.
    recommendations:
        Mapping ``team_id`` → :class:`ScoredTeam`. The caller decides
        what counts as a ``team_id`` (the engine does not impose a
        canonicalization in v0.1 — see the module docstring's "Join
        key" note). Iterate-order is preserved into the output
        ``summaries``.

    Returns
    -------
    ValidationLogAggregate
        The per-team summaries plus the orphan bucket.

    Raises
    ------
    TypeError
        - ``recommendations`` is not a :class:`collections.abc.Mapping`.
        - Any key in ``recommendations`` is not a ``str``.
        - Any value in ``recommendations`` is not a
          :class:`ScoredTeam`.
        - Any element of ``entries`` is not a
          :class:`RatingLogEntry`.

    Notes
    -----
    The function is **pure**: it returns a fresh
    :class:`ValidationLogAggregate` and never mutates its inputs.
    Re-running it over the same ``entries`` and ``recommendations``
    produces byte-identical output (modulo Python's hash randomization
    not affecting tuple/dict iteration order).

    No defensive copy of ``recommendations`` is taken before the
    join — if the caller mutates the mapping concurrently with this
    call the result is undefined; concurrent mutation is not a
    supported pattern.
    """

    # Type-check the recommendation mapping FIRST so a caller who passed
    # ``None`` or a list-by-mistake gets a precise error before the
    # entries iterator is consumed.
    if not isinstance(recommendations, Mapping):
        raise TypeError(
            "recommendations must be a Mapping[str, ScoredTeam], got "
            f"{type(recommendations).__name__}"
        )
    for key, value in recommendations.items():
        if not isinstance(key, str):
            raise TypeError(
                "recommendations keys must be str, got "
                f"{type(key).__name__} for value {value!r}"
            )
        if not isinstance(value, ScoredTeam):
            raise TypeError(
                f"recommendations[{key!r}] must be ScoredTeam, got "
                f"{type(value).__name__}"
            )

    # Walk ``entries`` once. The bucket-by-team_id dict gets the join
    # successes; the orphan list gets everything else. Both preserve
    # input order (dict insertion order in Python 3.7+ and list-append
    # respectively).
    by_team_id: dict[str, list[RatingLogEntry]] = {}
    orphans: list[RatingLogEntry] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, RatingLogEntry):
            raise TypeError(
                f"entries[{idx}] must be RatingLogEntry, got "
                f"{type(entry).__name__}"
            )
        if entry.team_id in recommendations:
            by_team_id.setdefault(entry.team_id, []).append(entry)
        else:
            orphans.append(entry)

    # Build per-team summaries in recommendation-mapping iteration
    # order. A team with no matched entries still gets a summary — the
    # "recommended but not yet tested" case is a feature.
    summaries: list[TeamValidationSummary] = []
    for team_id, scored_team in recommendations.items():
        team_entries = tuple(by_team_id.get(team_id, ()))
        summaries.append(
            TeamValidationSummary(
                team_id=team_id,
                scored_team=scored_team,
                entries=team_entries,
            )
        )

    return ValidationLogAggregate(
        summaries=tuple(summaries),
        orphaned_entries=tuple(orphans),
    )


__all__ = [
    "TeamValidationSummary",
    "ValidationLogAggregate",
    "aggregate_validation_log",
]
