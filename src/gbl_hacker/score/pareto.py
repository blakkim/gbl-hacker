"""Pareto-frontier filter (Sub-AC 2.4).

``pareto_filter(scored_teams)`` takes candidate teams paired with their
three score axes — ``expected_win_rate`` (Sub-AC 2.1),
``worst_case_robustness`` (Sub-AC 2.2), and ``meta_coverage`` (Sub-AC
2.3) — and returns only the **non-dominated** subset: the set of teams
no other candidate beats on every axis simultaneously.

Why a Pareto frontier, not a weighted-sum ranking?
--------------------------------------------------

The seed pins ``pareto_correctness`` as a top-level evaluation principle:

    "Output spans the Pareto frontier across expected win rate,
    worst-case robustness, and meta coverage — not collapsed onto a
    single metric."

A weighted-sum ranking (e.g. ``0.5·EV + 0.3·worst + 0.2·coverage``)
implicitly commits to a single trade-off ratio between the axes. That is
exactly the failure mode the seed is rejecting: a 0.55-mean / 0.30-worst
team and a 0.50-mean / 0.45-worst team are *both* valid choices that a
top-rank player picks between based on the day's matchups and how much
variance they want to absorb. Collapsing them onto one number throws
that decision away.

The Pareto frontier keeps every candidate that is not strictly dominated
on all three axes simultaneously. Two teams with identical mean win
rates but different robustness/coverage profiles both survive — and the
rationale-card layer (Sub-AC 3) explains *why* each one is there.

Dominance semantics
-------------------

For two scores ``a`` and ``b`` (each a 3-tuple ``(EV, worst, cov)``,
each axis "higher is better"):

* ``a`` **dominates** ``b`` iff
  ``a_i >= b_i for all i`` AND ``a_j >  b_j for some j``.
* ``a`` and ``b`` are **non-dominated** iff neither dominates the other
  (incomparable on the frontier).
* ``a`` and ``b`` are **equal** (all three axes byte-equal) iff neither
  the ``>=`` nor the ``>`` condition produces a strict winner; under the
  dominance rule above, neither dominates the other, so **both survive
  the filter**. This is the standard Pareto convention — equality does
  not eliminate either point. Downstream rationale-card rendering can
  dedupe by ``(team.species, score)`` if a presentation layer wants to
  collapse them, but the math should not.

The implementation is deliberately O(n²) — for ``v0.1`` ``K`` (the
number of candidate teams scored) is small (default proposal 5; even an
upper bound of a few hundred is comfortable). The algorithm walks each
candidate once and asks "does any *other* candidate dominate this one?"
If not, the candidate is on the frontier. This is the most readable
implementation and is exact for the axis count fixed by the seed (3).

Tie / NaN / out-of-range handling
---------------------------------

Each axis is contractually in ``[0.0, 1.0]`` (enforced upstream by the
three score aggregators). ``pareto_filter`` does **not** silently
re-clamp out-of-range values; it raises :class:`ValueError`. This is
defense-in-depth against a future score regression silently slipping a
``NaN`` or a ``-0.1`` into the frontier — a frontier consumer would
then see a "winner" that has no real meaning, which is worse than a
loud failure here.

``NaN`` is explicitly rejected. Standard IEEE-754 comparison rules make
``NaN >= x`` and ``NaN <= x`` both false, which means ``NaN`` is
*incomparable* to everything — under the dominance rule it would land
on the frontier despite being nonsense. Rejecting it at the boundary
keeps the frontier semantically meaningful.

Public surface
--------------

* :class:`Score`        — 3-axis scorecard (``expected_win_rate``,
                            ``worst_case_robustness``, ``meta_coverage``)
* :class:`ScoredTeam`   — ``(team, score)`` pairing
* :func:`dominates`     — strict-dominance predicate on two scores
* :func:`pareto_filter` — return the non-dominated subset
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from gbl_hacker.score.expected_win_rate import CandidateTeam


# Module-level constant: the dominance / value range each score axis is
# contractually bounded to. Kept as a module constant so a future
# rescaling decision lands in exactly one place.
_AXIS_LO = 0.0
_AXIS_HI = 1.0


def _validate_axis(value: float, name: str) -> None:
    """Reject NaN and out-of-range axis values.

    ``Score.__post_init__`` and :func:`pareto_filter`'s defensive
    validation both call this. Centralized so the error messages — and
    the contract — stay consistent.
    """

    if math.isnan(value):
        raise ValueError(f"{name} is NaN; score axes must be real numbers")
    if not (_AXIS_LO <= value <= _AXIS_HI):
        raise ValueError(
            f"{name} out of range: {value} "
            f"(must be in [{_AXIS_LO}, {_AXIS_HI}])"
        )


@dataclass(frozen=True, slots=True)
class Score:
    """Three-axis scorecard for a candidate team.

    Mirrors the ``score`` concept in the seed ontology. Each axis is
    "higher is better" and contractually in ``[0.0, 1.0]`` — the value
    range produced by :func:`expected_win_rate`,
    :func:`worst_case_robustness`, and :func:`meta_coverage`.

    Validation runs at construction time so out-of-range or NaN axes
    raise before they reach :func:`pareto_filter`. This keeps the
    frontier-math invariants local: any ``Score`` instance you hold is
    already comparable.

    Attributes
    ----------
    expected_win_rate:
        Usage-weighted mean win rate over the meta (Sub-AC 2.1).
    worst_case_robustness:
        Usage-weighted low-quantile win rate over the meta (Sub-AC 2.2).
    meta_coverage:
        Usage-weighted fraction of the meta handled ≥ threshold
        (Sub-AC 2.3).
    """

    expected_win_rate: float
    worst_case_robustness: float
    meta_coverage: float

    def __post_init__(self) -> None:
        _validate_axis(self.expected_win_rate, "expected_win_rate")
        _validate_axis(self.worst_case_robustness, "worst_case_robustness")
        _validate_axis(self.meta_coverage, "meta_coverage")

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """Axis values in canonical order (EV, worst, coverage).

        Convenience for callers (rationale cards, debug logs) that want
        the three axes as a flat tuple. The order matches the seed
        ontology's listing order and the order in this module's
        docstrings.
        """

        return (
            self.expected_win_rate,
            self.worst_case_robustness,
            self.meta_coverage,
        )


@dataclass(frozen=True, slots=True)
class ScoredTeam:
    """A candidate team paired with its three-axis score.

    The ``pareto_filter`` operates on iterables of these. Keeping
    ``team`` and ``score`` together (rather than as parallel lists)
    means a rationale-card renderer downstream can iterate the frontier
    once and produce per-team cards without re-joining the two.
    """

    team: CandidateTeam
    score: Score


def dominates(a: Score, b: Score) -> bool:
    """Return ``True`` iff ``a`` strictly Pareto-dominates ``b``.

    ``a`` dominates ``b`` iff::

        a.x_i >= b.x_i for every axis i
        AND
        a.x_j >  b.x_j for at least one axis j

    Equivalently: ``a`` is at least as good on everything *and* strictly
    better on something. Equal scores do not dominate each other (the
    strict-inequality condition fails for both directions); two
    incomparable scores also do not dominate each other.

    The function is asymmetric (``dominates(a, b)`` is not in general
    the same as ``dominates(b, a)``) and irreflexive
    (``dominates(a, a) is False`` by the strict-inequality clause).
    """

    a_t = a.as_tuple
    b_t = b.as_tuple
    strictly_better_somewhere = False
    for a_i, b_i in zip(a_t, b_t, strict=True):
        if a_i < b_i:
            # ``a`` is worse than ``b`` on this axis — cannot dominate.
            return False
        if a_i > b_i:
            strictly_better_somewhere = True
    return strictly_better_somewhere


def pareto_filter(scored_teams: Iterable[ScoredTeam]) -> list[ScoredTeam]:
    """Return the Pareto-optimal (non-dominated) subset of the input.

    A :class:`ScoredTeam` is retained iff no *other* member of
    ``scored_teams`` strictly dominates its score on the three axes
    (``expected_win_rate``, ``worst_case_robustness``, ``meta_coverage``
    — all "higher is better"). Equal scores survive together by the
    Pareto convention (neither strictly dominates the other).

    Output order matches input order. The frontier itself has no
    canonical single-axis sort (that is the whole point of the Pareto
    abstraction), so input order is the most honest default; callers
    that want a presentation sort (e.g. by EV descending for the
    rationale-card table) apply it on the returned list.

    Parameters
    ----------
    scored_teams:
        Iterable of :class:`ScoredTeam`. Consumed exactly once; works
        on generators. Empty input → empty output.

    Returns
    -------
    list[ScoredTeam]
        The non-dominated subset, in input order. Length is in
        ``[0, len(input)]``.

    Raises
    ------
    TypeError
        If any element is not a :class:`ScoredTeam` instance.

    Notes
    -----
    Validation of axis values (``[0, 1]``, no NaN) happens at
    :class:`Score` construction — instances reaching this function are
    already well-formed. Should an out-of-band caller construct a
    ``Score`` via :func:`object.__new__` (bypassing ``__post_init__``),
    the dominance predicate still operates correctly on real numbers in
    any range; only NaN would cause incomparability. Defense-in-depth
    here would re-validate every axis, which is wasted work for the
    common path — the upstream constructor is the right enforcement
    point.

    Complexity
    ----------
    O(n²) in the number of scored teams. For v0.1 ``K`` ≈ 5–50 this is
    instant. If ``K`` grows to thousands in a later AC, a Kung-Luccio-
    Preparata divide-and-conquer ``O(n log² n)`` algorithm can replace
    this body without changing the public contract.
    """

    # Materialize once: we walk the list twice (once for output order,
    # once per-candidate for the dominance check). A generator would
    # work but force us to keep two copies — explicit list is clearer.
    candidates = list(scored_teams)

    for idx, entry in enumerate(candidates):
        if not isinstance(entry, ScoredTeam):
            raise TypeError(
                f"pareto_filter input element at index {idx} is not a "
                f"ScoredTeam: got {type(entry).__name__}"
            )

    frontier: list[ScoredTeam] = []
    for i, candidate in enumerate(candidates):
        is_dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                # A point cannot dominate itself; ``dominates`` is
                # irreflexive, but skipping the self-pair is cheaper and
                # makes the intent explicit.
                continue
            if dominates(other.score, candidate.score):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(candidate)
    return frontier


__all__ = [
    "Score",
    "ScoredTeam",
    "dominates",
    "pareto_filter",
]
