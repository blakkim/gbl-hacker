"""Top-K ranker over the Pareto frontier (Sub-AC 2.5).

``rank_top_k(pareto_set, k)`` is the **final** stage of the v0.1 score
pipeline. The pipeline so far is:

1. Simulator yields per-matchup outcomes (Sub-AC 1).
2. ``expected_win_rate`` / ``worst_case_robustness`` / ``meta_coverage``
   each fold those outcomes into a single ``[0.0, 1.0]`` scalar
   (Sub-ACs 2.1–2.3).
3. ``pareto_filter`` keeps only the non-dominated subset of scored
   candidate teams across the three axes (Sub-AC 2.4).
4. **This module** orders that non-dominated subset and returns the
   top-K entries as the engine's final ranked recommendation list.

Why a separate ranker exists *after* the Pareto filter
------------------------------------------------------

The Pareto filter exposes the **frontier shape** — the set of teams
nobody dominates. By construction the frontier has no intrinsic
single-axis order: that is exactly the property the seed pins down as
``pareto_correctness`` ("output spans the frontier, not collapsed onto a
single metric"). So why rank at all?

Because the engine's final output is a *presentation*: a ranked list of
``K`` rationale cards (Sub-AC 3) for a top-rank operator to read in
priority order. A presentation needs an order; a frontier does not. The
two responsibilities are kept in separate modules so that:

* ``pareto_filter`` cannot be silently degraded into a weighted-sum
  ranker (the seed's ``pareto_correctness`` failure mode).
* ``rank_top_k`` cannot accidentally re-introduce dominated teams (it
  trusts its input to already be Pareto-filtered).
* The aggregation policy used for presentation (equal-weight sum,
  lexicographic tie-break) is named and overridable in *one* file.

Ranking policy
--------------

The default ranker scores each :class:`ScoredTeam` by a **weighted sum
of its three axes**, with all weights equal::

    rank_score(t) = w_ev · t.score.expected_win_rate
                  + w_wcr · t.score.worst_case_robustness
                  + w_cov · t.score.meta_coverage

Default ``weights = (1.0, 1.0, 1.0)``. Equal weights honor the seed's
``pareto_correctness`` principle: at the *frontier* stage we explicitly
refuse to collapse onto a single metric; at the *presentation* stage we
need *some* order, and the least-biased order is the one that treats
all three axes equally. A caller who wants to favor (say) a robustness-
heavy day can pass ``weights = (1.0, 2.0, 0.5)`` — the function does
not impose a normalization, so any non-negative triple works.

The mathematical fine print
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each axis is already in ``[0.0, 1.0]``, so the un-normalized weighted
sum is a valid ranking key even when weights are unequal — only the
*relative* magnitude of weights matters for the sort. We deliberately
do **not** divide by ``Σ weights``: the rank order is invariant under
that division and skipping it keeps the floating-point arithmetic
simple. Callers who want to compare ranks across runs with different
weights should divide by ``Σ weights`` themselves.

Tie-breaking
------------

Two ``ScoredTeam`` entries can have byte-equal weighted sums (e.g. on a
flat-equal-weight frontier where one team has score ``(0.6, 0.5, 0.4)``
and another has ``(0.5, 0.5, 0.5)``). Sorting by weighted sum alone is
non-deterministic for ties; ``rank_top_k`` resolves ties with a
**lexicographic descending** key over the three axes in canonical
ontology order ``(expected_win_rate, worst_case_robustness,
meta_coverage)``:

1. Higher weighted sum wins.
2. If tied, higher ``expected_win_rate`` wins.
3. If still tied, higher ``worst_case_robustness`` wins.
4. If still tied, higher ``meta_coverage`` wins.
5. If still tied (genuine score-byte-equal duplicates), input order is
   preserved — Python's ``list.sort`` is stable.

The tie-break order matches the seed ontology's listed axis order, not
a derived "which axis matters most" heuristic — keeping a tunable
"importance" out of the tie-break keeps the policy honest and easy to
audit.

Boundary behavior
-----------------

* ``k <= 0`` → empty list (no team requested). ``k = 0`` is **not** an
  error; it is a legitimate "rank but don't return anything yet" shape
  used by upstream pipelines that want to know the function would not
  crash on the current input.
* ``k > len(pareto_set)`` → return all teams in ranked order. We never
  pad with placeholder entries; the contract is "up to K teams in
  ranked order".
* Empty input → empty list, regardless of ``k``.
* Negative ``k`` is rejected with :class:`ValueError`. Negative slicing
  in NumPy / Pandas land is a common foot-gun source; we reject it
  loudly here to keep semantics unambiguous.

Public surface
--------------

* :func:`rank_top_k`            — return the top-K teams from a
                                    (Pareto-optimal) score frontier.
* :func:`_rank_sort_key`        — internal canonical sort key (exposed
                                    only via ``__all__`` for tests).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from gbl_hacker.score.pareto import ScoredTeam


# Module-level constant: canonical default weights. Equal weights match
# the seed's ``pareto_correctness`` principle — at the presentation
# stage we refuse to bias the ranking toward any single axis. Kept as a
# named constant so a future policy change lands in one place and is
# greppable from the rationale-card renderer.
_DEFAULT_WEIGHTS: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _validate_weights(weights: tuple[float, float, float]) -> None:
    """Reject NaN / negative / structurally-wrong weight triples.

    Centralized so the error messages — and the contract — stay
    consistent between :func:`rank_top_k`'s body and any future caller
    that wants to construct rank-equivalent sort keys directly.
    """

    if not isinstance(weights, tuple) or len(weights) != 3:
        raise ValueError(
            f"weights must be a 3-tuple (ev, wcr, cov); got {weights!r}"
        )
    names = ("ev_weight", "wcr_weight", "cov_weight")
    for name, value in zip(names, weights, strict=True):
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a real number; got {type(value).__name__}"
            )
        if math.isnan(float(value)):
            raise ValueError(f"{name} is NaN; weights must be real numbers")
        if value < 0.0:
            raise ValueError(
                f"{name} is negative ({value}); weights must be non-negative"
            )
    if sum(weights) == 0.0:
        # Zero-weight triple produces a constant ranking key — every
        # team ties on the primary criterion and the lexicographic
        # tiebreak silently takes over. That is almost certainly a
        # caller bug (they probably meant ``weights=(1,1,1)``). Reject
        # loudly rather than silently down-grading to lex-only.
        raise ValueError(
            "weights sum to zero; at least one axis must have positive weight"
        )


def _rank_sort_key(
    scored: ScoredTeam,
    weights: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Build the canonical sort key for one :class:`ScoredTeam`.

    Returns a 4-tuple ``(-weighted_sum, -ev, -wcr, -cov)``. The negation
    lets us call ``sorted(..., key=...)`` in *ascending* order and
    still get a *descending* rank — Python's sort is stable, so input
    order breaks final ties.

    Exposed module-internally (with a leading underscore) so tests can
    verify the key shape directly without re-implementing the ordering.
    """

    s = scored.score
    weighted = (
        weights[0] * s.expected_win_rate
        + weights[1] * s.worst_case_robustness
        + weights[2] * s.meta_coverage
    )
    # Negate every component so ascending sort → descending rank. Each
    # axis is in [0, 1] so the negation cannot produce NaN/inf for any
    # well-formed ``Score`` instance.
    return (
        -weighted,
        -s.expected_win_rate,
        -s.worst_case_robustness,
        -s.meta_coverage,
    )


def rank_top_k(
    pareto_set: Iterable[ScoredTeam],
    k: int,
    *,
    weights: tuple[float, float, float] = _DEFAULT_WEIGHTS,
) -> list[ScoredTeam]:
    """Return the top-K ranked teams from a Pareto-optimal score set.

    This is the Sub-AC 2.5 headline function — the engine's final
    output transformation. It takes the non-dominated subset produced
    by :func:`~gbl_hacker.score.pareto_filter` and orders it for human
    consumption by the rationale-card layer (Sub-AC 3).

    Parameters
    ----------
    pareto_set:
        Iterable of :class:`ScoredTeam` instances. The function does
        **not** re-verify Pareto-optimality — it trusts the upstream
        ``pareto_filter`` for that. Passing a non-Pareto-filtered set is
        not an error: the ranker will produce a deterministic order
        anyway. Consumed exactly once; works on generators.
    k:
        Number of teams to return. ``0`` returns an empty list. Values
        greater than the input length return all input teams in ranked
        order (no padding). Negative values raise :class:`ValueError`.
    weights:
        Per-axis weights for the weighted-sum ranking key, in canonical
        order ``(ev, wcr, cov)``. Default ``(1.0, 1.0, 1.0)`` —
        equal-weight, matching the seed's pareto-correctness intent.
        Each weight must be a non-negative real; their sum must be
        positive. Magnitudes need not be normalized (only the
        *relative* magnitude matters).

    Returns
    -------
    list[ScoredTeam]
        Up to ``min(k, len(input))`` teams, sorted descending by
        weighted-sum score with deterministic lexicographic tiebreak
        (see module docstring for the tie-break rule).

    Raises
    ------
    ValueError
        If ``k`` is negative, ``weights`` is not a length-3 tuple,
        any weight is negative / NaN / non-numeric, or the weights sum
        to zero.
    TypeError
        If any element of ``pareto_set`` is not a :class:`ScoredTeam`.

    Notes
    -----
    The function is **pure**: it returns a fresh list and never mutates
    the input. Even though ``ScoredTeam`` is frozen by construction, the
    iterable itself (e.g. a list passed by the caller) is left
    untouched — this matters when the caller wants to keep the
    pre-rank Pareto frontier around for debugging or for a second
    presentation pass with different weights.

    Complexity
    ----------
    ``O(n log n)`` in the size of the input, dominated by the sort. For
    v0.1's expected frontier size (≤ 50 teams), this is instant. The
    ``top_k`` slicing happens after a full sort — a partial-sort
    (e.g. ``heapq.nsmallest`` on the sort key) is asymptotically
    cheaper for ``k << n`` but not worth the readability cost at this
    scale. Swap-in point is documented for future scaling work.
    """

    # Validate k early — negative values are almost certainly a
    # downstream bug (slicing-style negative indexing leaked in).
    if not isinstance(k, int) or isinstance(k, bool):
        # bool is a subclass of int in Python; reject it explicitly so
        # ``rank_top_k(pareto, True)`` does not silently become k=1.
        raise ValueError(f"k must be an int, got {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    _validate_weights(weights)

    # Materialize once. We need to iterate twice (type-check loop +
    # sort), and a generator would force us to keep two copies.
    candidates = list(pareto_set)

    for idx, entry in enumerate(candidates):
        if not isinstance(entry, ScoredTeam):
            raise TypeError(
                f"rank_top_k input element at index {idx} is not a "
                f"ScoredTeam: got {type(entry).__name__}"
            )

    if k == 0 or not candidates:
        # Short-circuit: nothing to rank, or caller asked for nothing.
        # The first branch avoids a wasted sort; the second avoids a
        # sort over the empty list (harmless but explicit is clearer).
        return []

    # Python's ``sorted`` is stable, so equal sort-keys preserve input
    # order — this is the final layer of the tie-break contract.
    ranked = sorted(candidates, key=lambda st: _rank_sort_key(st, weights))

    # Slice to top-K. Slicing past the end is harmless in Python; the
    # ``k > len(candidates)`` branch falls out naturally.
    return ranked[:k]


__all__ = [
    "rank_top_k",
]
