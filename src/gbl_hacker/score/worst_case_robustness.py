"""Worst-case robustness aggregator (Sub-AC 2.2).

``worst_case_robustness(team, meta)`` collapses the same per-matchup
simulator outputs that feed :func:`expected_win_rate` into a *different*
scalar in ``[0.0, 1.0]`` — the team's **worst-case win rate** across the
meta. Where ``expected_win_rate`` answers "how do you do on average",
this answers "how badly can the meta hurt you in the matchups you
actually expect to face".

Why a usage-weighted quantile, not a naive minimum?
---------------------------------------------------

The simplest worst-case statistic is ``min(win_rate_over_meta)`` — the
single lowest 1v1-aggregate outcome across all opponent teams in the
meta. That works, but it has two well-known failure modes:

1. **Long-tail noise.** A 0.4 %-usage exotic counter that nukes you is
   not a Pareto-relevant robustness signal — you will almost never see
   it. A naive min is dominated by these tail outliers.
2. **Zero-weight artifacts.** A meta row reported at ``usage_pct == 0``
   should not lower the team's worst-case score, but a naive min over
   the rate column would still pick its win rate up.

The fix is a **usage-weighted low quantile**: sort opponents by win rate
ascending, walk the cumulative usage CDF, and return the rate at the
``quantile``-th fractile of the *usage* mass. Concretely::

    rates = [(usage_pct_k, set_win_rate(team, O_k))  for opponent k]
    sort rates ascending by win_rate
    cumulative = 0
    target     = quantile · total_usage
    return the first rate whose cumulative_usage ≥ target

This gives three useful behaviours from a single parameter:

* ``quantile = 0.0`` — the **lowest win rate** among opponents with
  non-zero usage. Zero-usage rows are skipped because they cannot push
  the cumulative weight forward; this is the natural way to filter them
  out without a separate special case.
* ``quantile = 0.1`` — the **10th-percentile** win rate over the
  realized usage distribution; the canonical robustness metric. Robust
  to the long-tail-noise problem above: a 0.4 %-usage counter can only
  set the value if you have already lost to enough usage mass to reach
  the 10th-percentile threshold.
* ``quantile = 0.5`` — the **usage-weighted median**.

The default ``quantile = 0.1`` matches the Pareto-ranker's intent: pin
the *practical* worst case (the 10 % of the meta most hostile to you),
not the *theoretical* one (a zero-weight or ultra-rare counter that may
never appear in a real session). Callers who specifically want
"absolute worst single matchup" can pass ``quantile = 0.0``.

PvPoke-bug-avoidance contract
-----------------------------

Identical to :mod:`expected_win_rate`: the per-set callable
(``set_win_rate_fn``) is expected to be set-state-aware (entry energy,
asymmetric shields, switch-energy carry). This module owns aggregation
only — it does not re-enforce per-matchup combat semantics. It *does*
enforce that the per-set win rate is in ``[0.0, 1.0]``; an out-of-range
return value from a future set simulator raises immediately.
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


def worst_case_robustness(
    team: CandidateTeam,
    meta: MetaSnapshot,
    *,
    build_registry: Mapping[str, CombatantBuild],
    quantile: float = 0.1,
    starting_shields: int = MAX_SHIELDS,
    tie_value: float = 0.5,
    on_missing_build: Literal["skip", "raise"] = "raise",
    set_win_rate_fn: SetWinRateFn | None = None,
) -> float:
    """Usage-weighted low-quantile win rate of ``team`` over the meta.

    This is the Sub-AC 2.2 headline function. It complements
    :func:`expected_win_rate` (a usage-weighted *mean*) with a usage-
    weighted *low quantile* — the win rate the team sustains across the
    bottom ``quantile`` fraction of the meta's usage mass when sorted by
    matchup outcome.

    Parameters
    ----------
    team:
        The candidate team to score.
    meta:
        Parsed Taiman Party snapshot. Only ``meta.team_usage`` is
        consulted; per-Pokémon usage feeds ``meta_coverage`` (Sub-AC
        2.3), not robustness.
    build_registry:
        Maps species names → :class:`CombatantBuild`. Required for
        materializing opponent teams.
    quantile:
        Position in ``[0.0, 1.0]`` along the *usage-sorted-by-win-rate*
        CDF at which to read the win rate. Default ``0.1`` (the 10th-
        percentile worst opponent by usage). ``0.0`` recovers the naive
        "lowest non-zero-weight opponent" minimum; ``0.5`` is the usage-
        weighted median; ``1.0`` is the usage-weighted maximum (almost
        never the right thing for a *robustness* metric, but provided
        for symmetry with the quantile parameter — see "Boundary
        behavior" below).
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
        Worst-case (usage-weighted ``quantile``-th) win rate in
        ``[0.0, 1.0]``.

    Boundary behavior
    -----------------

    * **Empty meta** (``meta.team_usage`` empty) → ``0.0``. No signal.
    * **All opponents skipped** under ``on_missing_build="skip"`` →
      ``0.0``. No covered slice to read the quantile from.
    * **All opponents at usage_pct == 0** → ``0.0``. Same reason.
    * **Out-of-range win rate** from ``set_win_rate_fn`` →
      :class:`ValueError`.
    * **Out-of-range ``quantile``** (outside ``[0.0, 1.0]``) →
      :class:`ValueError`.
    * **Out-of-range ``on_missing_build``** → :class:`ValueError`.

    Why does ``quantile=0.0`` skip zero-usage rows?
        Zero-usage rows do not advance the cumulative-usage walk, so
        they never become the "first row whose cumulative usage exceeds
        ``0.0``". This is the right behavior — a 0 %-usage row is
        upstream noise and should not lower the team's robustness
        floor. The implementation falls out of the CDF semantics; it
        is not a hard-coded special case.
    """

    if on_missing_build not in ("skip", "raise"):
        raise ValueError(
            f"on_missing_build must be 'skip' or 'raise', got {on_missing_build!r}"
        )
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(
            f"quantile out of range: {quantile} (must be in [0.0, 1.0])"
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

    # Collect (usage_pct, win_rate) pairs across the scored slice of the
    # meta. Out-of-range rates raise immediately — defense in depth
    # against a regressing set simulator (mirrors expected_win_rate).
    rate_rows: list[tuple[float, float]] = []
    for opp_usage in meta.team_usage:
        try:
            opp_team = materialize_opponent_team(opp_usage, build_registry)
        except MissingBuildError:
            if on_missing_build == "raise":
                raise
            continue
        rate = set_win_rate_fn(team, opp_team)
        if not (0.0 <= rate <= 1.0):
            raise ValueError(
                "set_win_rate_fn returned out-of-range value: "
                f"{rate} (must be in [0.0, 1.0])"
            )
        rate_rows.append((opp_usage.usage_pct, rate))

    if not rate_rows:
        return 0.0

    # Drop zero-usage rows up front: they contribute no information to
    # the CDF and they would otherwise spuriously match ``cumulative >=
    # target`` at ``quantile == 0.0`` (where ``target == 0.0``) on the
    # very first iteration, regardless of whether they advanced the
    # cumulative weight. Pre-filtering keeps the CDF walk semantics
    # honest: only rows with realized usage mass can set the quantile.
    non_zero_rows = [row for row in rate_rows if row[0] > 0.0]
    if not non_zero_rows:
        return 0.0

    total_weight = sum(usage for usage, _ in non_zero_rows)
    # ``total_weight`` is strictly positive here because we filtered out
    # the zero-usage rows, but we keep the explicit check for paranoia
    # against future refactors.
    if total_weight == 0.0:  # pragma: no cover - defensive
        return 0.0

    # Sort ascending by win rate; ties are resolved deterministically by
    # arrival order (Python's sort is stable). Two opponents at the same
    # win rate are interchangeable — their order in the CDF walk does
    # not change the returned quantile value.
    non_zero_rows.sort(key=lambda row: row[1])

    target = quantile * total_weight
    cumulative = 0.0
    last_rate = non_zero_rows[-1][1]  # for the ``quantile == 1.0`` case
    for usage, rate in non_zero_rows:
        cumulative += usage
        if cumulative >= target:
            return rate
    # Numerical fallback: floating-point accumulation can leave us a
    # hair short of ``target`` even when ``quantile == 1.0``. Return
    # the highest rate in that boundary case.
    return last_rate


__all__ = ["worst_case_robustness"]
