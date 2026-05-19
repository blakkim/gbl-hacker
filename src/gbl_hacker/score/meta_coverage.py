"""Meta-coverage aggregator (Sub-AC 2.3).

``meta_coverage(team, meta)`` collapses the same per-matchup simulator
outputs that feed :func:`expected_win_rate` and
:func:`worst_case_robustness` into a *third*, structurally different
scalar in ``[0.0, 1.0]`` — the **fraction of the meta's usage-weight
that the team handles at or above a configurable win-rate threshold**.

Where ``expected_win_rate`` answers "how do you do on average" and
``worst_case_robustness`` answers "how badly can the meta hurt you in
practice", ``meta_coverage`` answers "how much of the meta are you
actually solving". The three axes are deliberately complementary — that
is exactly the Pareto-frontier shape the seed's ``score`` ontology pins:
none of them dominates the other two, and a team optimal on one axis can
be inferior on the others. The rationale card consumes ``meta_coverage``
as the "% of meta this team covers" line — the headline interpretability
quantity that persuades a top-rank player they are not picking a team
that loses to a large unseen slice of the field.

Threshold-based, not rate-based — why?
--------------------------------------

A natural alternative is "weighted-mean win rate over teams you beat",
but that re-derives a win-rate-shaped scalar already provided by
``expected_win_rate``, just on a sub-slice. Coverage is qualitatively
different: it is a **mass** statistic, not a rate. The question it
answers — "how much of the meta does this team handle?" — is a count
(weighted) of opponents above the threshold, normalized by the realized
total. That makes it monotone in the threshold (lower threshold ⇒ more
opponents counted) and orthogonal to the rate-shaped scores: a team can
have a high mean win rate yet narrow coverage (it dominates the few
opponents it sees, loses to the rest) or a wide coverage with a modest
mean (it edges out many opponents without crushing any).

Definition
----------

Let ``H = { opp_k : set_win_rate(team, opp_k) >= threshold }``::

    meta_coverage(team, M) =
        Σ_{k ∈ H}  usage_pct_k
        ──────────────────────
        Σ_k  usage_pct_k       (over the *covered* slice of the meta)

The denominator is the **realized** total usage — the same convention
used by ``expected_win_rate`` (truncation- and skip-robust). A meta that
sums to 30 %, with the team handling 20 % of that 30 %, scores
20 / 30 ≈ 0.667 — *not* 20 / 100 = 0.2. We are reporting the fraction
of *the scored slice* the team covers, never silently degrading by
conflating "couldn't score" with "lost to". Operators who want absolute
mass should multiply by ``Σ usage_pct / 100``.

The default ``threshold = 0.5`` is "any non-losing matchup counts" — the
weakest non-trivial bar. Teams with mean win rate above 0.5 against an
opponent are scoring at least neutrally; this captures the broadest
interpretation of "the team can handle this matchup". A stricter ranker
can pass ``threshold = 0.6`` (clear edge) or ``0.7`` (decisive edge).

Tie-handling
------------

The comparison is ``rate >= threshold`` (closed lower bound), not
``rate > threshold``. With the default ``tie_value = 0.5`` an exact-tie
simulator outcome maps to a 0.5 win fraction, and at the default
``threshold = 0.5`` that opponent **is counted**. Treating a true tie as
"handled" matches the GBL operator's intuition: a 50/50 matchup is not a
loss. Callers who specifically want strictly-better-than-coin-flip
coverage can pass ``threshold = 0.501``.

PvPoke-bug-avoidance contract
-----------------------------

Identical to its siblings: the per-set callable (``set_win_rate_fn``) is
expected to be set-state-aware (entry energy, asymmetric shields,
switch-energy carry). This module owns aggregation only. It *does*
enforce that the per-set win rate is in ``[0.0, 1.0]``; an out-of-range
return value from a future set simulator raises immediately rather than
silently corrupting the coverage count.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from gbl_hacker.parse.taiman import MetaSnapshot
from gbl_hacker.score.expected_win_rate import (
    CandidateTeam,
    MissingBuildError,
    SetWinRateFn,
    default_set_win_rate,
    materialize_opponent_team,
)
from gbl_hacker.simulator import MAX_SHIELDS, CombatantBuild


def meta_coverage(
    team: CandidateTeam,
    meta: MetaSnapshot,
    *,
    build_registry: Mapping[str, CombatantBuild],
    threshold: float = 0.5,
    starting_shields: int = MAX_SHIELDS,
    tie_value: float = 0.5,
    on_missing_build: Literal["skip", "raise"] = "raise",
    set_win_rate_fn: SetWinRateFn | None = None,
) -> float:
    """Usage-weighted fraction of the meta the ``team`` handles ≥ ``threshold``.

    This is the Sub-AC 2.3 headline function. It complements
    :func:`expected_win_rate` (usage-weighted *mean* rate) and
    :func:`worst_case_robustness` (usage-weighted low-quantile *rate*)
    with a usage-weighted *mass* — the share of the realized meta mass
    whose set-level win rate meets or exceeds ``threshold``.

    Parameters
    ----------
    team:
        The candidate team to score.
    meta:
        Parsed Taiman Party snapshot. Only ``meta.team_usage`` is
        consulted; per-Pokémon usage is informative for the rationale
        card's breakdown, not for the coverage scalar itself.
    build_registry:
        Maps species names → :class:`CombatantBuild`. Required for
        materializing opponent teams.
    threshold:
        Win-rate cutoff in ``[0.0, 1.0]``. An opponent contributes to the
        coverage numerator iff ``set_win_rate(team, opp) >= threshold``.
        Default ``0.5`` — "any non-losing matchup counts" (a tied
        matchup at the default ``tie_value = 0.5`` is included). Strict
        callers can pass ``0.501`` to require a strict edge, or ``0.6``
        / ``0.7`` for a "clear" / "decisive" edge.
    starting_shields:
        Forwarded to the default set-level aggregator. Ignored when a
        custom ``set_win_rate_fn`` is supplied.
    tie_value:
        Forwarded to the default set-level aggregator. Ignored when a
        custom ``set_win_rate_fn`` is supplied.
    on_missing_build:
        Strategy when ``build_registry`` is missing an opponent's
        species. ``"raise"`` (default) or ``"skip"``. Same semantics as
        :func:`expected_win_rate`: a skipped opponent leaves the
        denominator (total usage) smaller; ``"raise"`` keeps the
        operator honest about uncovered slices.
    set_win_rate_fn:
        Optional injection. ``None`` (default) → use
        :func:`default_set_win_rate` with the keyword args above.

    Returns
    -------
    float
        Coverage fraction in ``[0.0, 1.0]`` — the share of the realized
        meta usage mass for which the team's win rate meets or exceeds
        ``threshold``.

    Boundary behavior
    -----------------

    * **Empty meta** (``meta.team_usage`` empty) → ``0.0``. No signal.
    * **All opponents skipped** under ``on_missing_build="skip"`` →
      ``0.0``. No covered slice to compute a coverage fraction over.
    * **All opponents at usage_pct == 0** → ``0.0``. Zero realized total
      weight, no division-by-zero, no covered slice.
    * **Threshold = 0.0** → every in-range win rate qualifies; the
      function returns ``1.0`` on any non-empty, non-zero-weight meta
      (every realized opponent is "handled" by the broadest definition).
    * **Threshold = 1.0** → only perfect-win opponents qualify; this is
      the strictest interpretation of "handled".
    * **Out-of-range win rate** from ``set_win_rate_fn`` →
      :class:`ValueError`. Defense-in-depth against a future set
      simulator regressing.
    * **Out-of-range ``threshold``** (outside ``[0.0, 1.0]``) →
      :class:`ValueError`.
    * **Out-of-range ``on_missing_build``** → :class:`ValueError`.

    Why ``>=`` (not ``>``)?
        A 50/50 matchup at ``threshold = 0.5`` is "handled" in the GBL
        operator's sense — neither side has an edge, the team can still
        steal the set with correct play. Treating it as covered matches
        the rationale-card intent (which calls these matchups "even").
        Callers wanting strict dominance use ``threshold = 0.501``.

    Why denominator = realized usage, not 100?
        Same reason as :func:`expected_win_rate`. Taiman Party truncates
        its team-usage list; un-scored mass is *not* mass we lost to.
        Reporting "fraction of the scored slice" is the only honest
        statistic when truncation is the norm. The data-honesty caveat
        on the snapshot already surfaces the uncovered-mass problem to
        the operator; the score does not double-count it.
    """

    if on_missing_build not in ("skip", "raise"):
        raise ValueError(
            f"on_missing_build must be 'skip' or 'raise', got {on_missing_build!r}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold out of range: {threshold} (must be in [0.0, 1.0])"
        )

    if not meta.team_usage:
        return 0.0

    if set_win_rate_fn is None:

        def _default(a: CandidateTeam, b: CandidateTeam) -> float:
            return default_set_win_rate(
                a,
                b,
                starting_shields=starting_shields,
                tie_value=tie_value,
            )

        set_win_rate_fn = _default

    covered_weight = 0.0
    total_weight = 0.0
    for opp_usage in meta.team_usage:
        try:
            opp_team = materialize_opponent_team(opp_usage, build_registry)
        except MissingBuildError:
            if on_missing_build == "raise":
                raise
            # Skipped opponents leave the denominator smaller — same as
            # expected_win_rate. The "uncovered slice" is *not* the
            # numerator's job to flag; the operator's data-honesty
            # caveat (already on every snapshot rendering) is.
            continue
        rate = set_win_rate_fn(team, opp_team)
        if not (0.0 <= rate <= 1.0):
            raise ValueError(
                "set_win_rate_fn returned out-of-range value: "
                f"{rate} (must be in [0.0, 1.0])"
            )
        total_weight += opp_usage.usage_pct
        if rate >= threshold:
            covered_weight += opp_usage.usage_pct

    if total_weight == 0.0:
        # Either every opponent was at usage_pct == 0 (degenerate
        # upstream), or every opponent was skipped under "skip" policy.
        # Either way, there is no covered slice to report a fraction of.
        return 0.0

    coverage = covered_weight / total_weight
    # Defensive clamp. covered_weight and total_weight are both
    # non-negative and covered_weight <= total_weight by construction,
    # so the ratio is in [0, 1] absent floating-point pathology — the
    # clamp catches that pathology without changing well-formed results.
    if coverage < 0.0:
        return 0.0
    if coverage > 1.0:
        return 1.0
    return coverage


__all__ = ["meta_coverage"]
