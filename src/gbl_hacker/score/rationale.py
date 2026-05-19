"""Rationale-card selection helpers (Sub-AC 3.1+).

The rationale card (seed ontology: ``rationale_card``) is the
interpretability surface attached to each recommended team. It lists
**key favorable matchups, key unfavorable matchups, the meta-coverage
breakdown, and a short prose justification** — the seed's
``interpretability`` evaluation principle made concrete.

This module owns the *selection* primitives that drive the rationale
card. The score axes (``expected_win_rate`` / ``worst_case_robustness``
/ ``meta_coverage``) already fold per-matchup outcomes into single
scalars; rationale-card rendering, by contrast, needs the **per-opponent
breakdown** the scalars threw away. That breakdown lives in
:class:`MetaMatchupResult` records (one per opponent in the meta), and
the selectors in this module project the right slice of those records
into the card's bullet lists.

Sub-AC 3.1 (initial entry):
    * :class:`MetaMatchupResult` — one opponent's set-level outcome
      paired with its meta-usage weight.
    * :func:`select_favorable_matchups` — top-N records ranked
      descending by ``win_rate`` (favorable = higher win rate).

Sub-AC 3.2:
    * :func:`select_unfavorable_matchups` — bottom-N records ranked
      ascending by ``win_rate`` (unfavorable = lower win rate). The
      rationale card's "key losses" bullet list pulls from this.

Sub-AC 3.3:
    * :func:`compute_meta_coverage` — usage-weighted fraction of the
      meta the team handles at or above ``win_threshold``, computed
      from pre-simulated :class:`MetaMatchupResult` records joined
      against the canonical :class:`MetaSnapshot` usage table. The
      rationale card's "% of meta covered" line pulls from this.

      Sibling-but-distinct from :func:`gbl_hacker.score.meta_coverage`
      (Sub-AC 2.3): the score-axis version owns the simulator-driven
      pipeline (team → simulator → scalar) and feeds the Pareto ranker;
      this version owns the rationale-card pipeline (per-opponent
      results → coverage scalar) and shares its inputs byte-for-byte
      with the favorable / unfavorable selectors above. Sharing inputs
      means the rationale card's coverage line cannot drift from its
      "favorable matchups" and "unfavorable matchups" bullets — they
      are all derived from the same ``MetaMatchupResult`` list.

Sub-AC 3.4 (this entry):
    * :class:`RationaleCard` — frozen container that assembles the
      three structured pieces produced by the earlier sub-ACs into the
      seed-ontology ``rationale_card`` shape (team + favorable list +
      unfavorable list + coverage slice).
    * :func:`build_rationale_card` — validating constructor over the
      three pre-computed pieces. It is the **single seam** the
      rationale-card pipeline funnels through, so any future renderer
      (CLI table, JSON exporter, web UI) consumes the same structurally
      validated object instead of re-deriving it from raw inputs.

Future sub-ACs will add:
    * Prose-justification composition over the structured pieces.

Why a separate module?
----------------------

The scalar-score modules (``expected_win_rate``, ``meta_coverage``, …)
are *aggregations* — they consume per-matchup data and emit one number.
The rationale card runs in the opposite direction: it consumes the same
per-matchup data and *preserves* it as a per-opponent list. Keeping
that selection logic out of the aggregation modules avoids a "what does
this module return" ambiguity and keeps the score-axis files focused.

PvPoke-bug-avoidance contract
-----------------------------

Identical to the score-axis modules: the upstream callable that
produced each ``MetaMatchupResult.win_rate`` is expected to be
set-state-aware (entry energy, asymmetric shields, switch-energy
carry). This module owns *selection*, not per-matchup combat. It does
enforce that every ``win_rate`` it receives is in ``[0.0, 1.0]`` — an
out-of-range value from a future set simulator raises immediately
rather than silently corrupting the rationale card's "top wins" list.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from gbl_hacker.parse.taiman import MetaSnapshot
from gbl_hacker.score.expected_win_rate import CandidateTeam


@dataclass(frozen=True, slots=True)
class MetaMatchupResult:
    """One opponent's set-level outcome from a candidate team's perspective.

    The score axes fold many of these into a single scalar each. The
    rationale card consumes the same records to render its
    "favorable / unfavorable matchups" bullet lists, so the record
    shape is the natural shared currency between the two layers.

    Attributes
    ----------
    opponent:
        The opposing :class:`CandidateTeam`. Carries the 3-slot lineup
        the card prints (lead/safe_swap/closer species names).
    win_rate:
        The candidate team's set-level win rate against ``opponent``,
        in ``[0.0, 1.0]``. Produced by the set-state-aware simulator
        (or any other compliant ``set_win_rate_fn`` injection).
    usage_pct:
        Optional report-density weight in ``[0.0, 100.0]``. ``None``
        when the caller has no meta-usage context (e.g. ad-hoc
        comparisons). Preserved verbatim so that downstream cards can
        render "weight: X%" alongside each line.
    """

    opponent: CandidateTeam
    win_rate: float
    usage_pct: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError(
                f"MetaMatchupResult.win_rate out of range: {self.win_rate} "
                f"(must be in [0.0, 1.0])"
            )
        # usage_pct semantics: Taiman Party reports shares as
        # percentages in [0, 100]. We accept None for the no-context
        # case but reject negatives / NaN if a real number was provided.
        if self.usage_pct is not None:
            if self.usage_pct != self.usage_pct:  # NaN check (NaN != NaN)
                raise ValueError(
                    "MetaMatchupResult.usage_pct is NaN; must be a real "
                    "number in [0, 100] or None"
                )
            if self.usage_pct < 0.0:
                raise ValueError(
                    f"MetaMatchupResult.usage_pct is negative "
                    f"({self.usage_pct}); must be in [0, 100] or None"
                )


def select_favorable_matchups(
    team: CandidateTeam,
    meta_matchup_results: Iterable[MetaMatchupResult],
    *,
    n: int = 3,
) -> list[MetaMatchupResult]:
    """Return the top-N favorable matchups for ``team`` (opponent + score).

    "Favorable" = highest ``win_rate``. The selector sorts the input
    records in **descending** win-rate order and returns the first
    ``n`` entries. If fewer than ``n`` records are supplied, the
    selector returns all of them in ranked order — it never pads the
    output. This is the Sub-AC 3.1 headline function and the first
    primitive feeding the rationale card's "favorable matchups" list.

    Parameters
    ----------
    team:
        The candidate team the records are evaluated against. Held for
        contract / downstream-rendering context — the rationale card
        will print "your team vs X" lines that need the subject team
        identity preserved through the pipeline. The function does not
        re-derive the per-opponent win rates from ``team``; that is
        the upstream simulator's job (Sub-AC 1 / Sub-AC 2.x).
    meta_matchup_results:
        Iterable of :class:`MetaMatchupResult`, each representing one
        opponent's set-level outcome from ``team``'s perspective.
        Consumed exactly once; works on generators.
    n:
        Number of records to return. Defaults to ``3`` — the rationale
        card convention of "top three" matchups. ``n = 0`` returns
        ``[]`` (legitimate "select nothing yet" shape used by upstream
        pipelines doing dry-run wiring). ``n`` larger than the input
        length returns every record in ranked order (no padding).
        Negative ``n`` raises :class:`ValueError`.

    Returns
    -------
    list[MetaMatchupResult]
        Up to ``min(n, len(input))`` records, sorted **descending** by
        ``win_rate``. Ties on ``win_rate`` preserve input order — the
        sort is stable, which matters when two opponents end up at the
        exact tie value (e.g. 0.5 double-KO).

    Raises
    ------
    ValueError
        If ``n`` is negative or non-int, or if any record's
        ``win_rate`` is out of ``[0, 1]``. The per-record validation
        runs at :class:`MetaMatchupResult` construction time too; the
        repeated check here is defense-in-depth against a caller that
        constructs records via ``object.__new__`` and bypasses
        ``__post_init__``.
    TypeError
        If any input element is not a :class:`MetaMatchupResult`.

    Notes
    -----
    The function is **pure**: it returns a fresh list and never
    mutates its input. Even when the caller passes a list it remains
    in its original order after the call — important when the caller
    keeps the full meta-matchup list around for the "unfavorable"
    selection pass that will land in Sub-AC 3.2.

    Why descending sort, not ``heapq.nlargest``?
        For v0.1's expected meta size (tens to low hundreds of teams)
        the ``O(n log n)`` full sort is instant and reads more clearly.
        A future scaling AC can swap in ``heapq.nlargest`` without
        changing the contract; the descending-stable-sort behavior is
        identical for tie-breaking purposes.

    Why does ``team`` not influence the math?
        The contract is "given a team's matchup results, pick the top
        N". The team-identity is already baked into the records by
        construction (the caller produced them for *this* team). The
        ``team`` parameter is held in the signature so the rationale-
        card renderer downstream can stitch "team T vs opponent O"
        lines without re-threading the team identity through a
        separate channel — and so a future audit hook can assert that
        every record was indeed produced against ``team``.
    """

    if not isinstance(n, int) or isinstance(n, bool):
        # bool is an int subclass in Python; reject explicitly so
        # ``select_favorable_matchups(team, results, n=True)`` does
        # not silently collapse to ``n=1``.
        raise ValueError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    # ``team`` is part of the contract but currently unused in the
    # math (see docstring "Why does team not influence the math?").
    # The no-op reference keeps linters from flagging the parameter
    # and documents the intentional pass-through.
    _ = team

    records = list(meta_matchup_results)

    for idx, rec in enumerate(records):
        if not isinstance(rec, MetaMatchupResult):
            raise TypeError(
                f"meta_matchup_results[{idx}] is not a MetaMatchupResult: "
                f"got {type(rec).__name__}"
            )
        # Defense-in-depth: re-validate the win_rate range even though
        # ``MetaMatchupResult.__post_init__`` already does. Catches
        # callers that bypass the dataclass constructor.
        if not (0.0 <= rec.win_rate <= 1.0):
            raise ValueError(
                f"meta_matchup_results[{idx}].win_rate out of range: "
                f"{rec.win_rate} (must be in [0.0, 1.0])"
            )

    if n == 0 or not records:
        # Short-circuits matching the documented contract:
        #   * n=0 → caller asked for nothing.
        #   * empty input → no records to rank.
        return []

    # Stable descending sort: ``-win_rate`` as the key turns ascending
    # sort into descending rank; Python's ``sorted`` is stable so
    # tied win rates preserve input order. This matches the seed's
    # interpretability principle — a deterministic order means the
    # rationale card looks the same on repeat runs over the same data.
    ranked = sorted(records, key=lambda r: -r.win_rate)

    # Slicing past the end is harmless; the ``n > len(records)`` branch
    # falls out of the same slice without a separate code path.
    return ranked[:n]


def select_unfavorable_matchups(
    team: CandidateTeam,
    meta_matchup_results: Iterable[MetaMatchupResult],
    *,
    n: int = 3,
) -> list[MetaMatchupResult]:
    """Return the top-N unfavorable matchups for ``team`` (opponent + score).

    "Unfavorable" = lowest ``win_rate``. The selector sorts the input
    records in **ascending** win-rate order and returns the first ``n``
    entries. If fewer than ``n`` records are supplied, the selector
    returns all of them in ranked order — it never pads the output.
    This is the Sub-AC 3.2 headline function and feeds the rationale
    card's "key losses" bullet list (the mirror of the favorable list).

    The signature, contract, and edge-case behavior are intentionally
    symmetric with :func:`select_favorable_matchups`:

      * Same record type / same ``n`` semantics / same purity guarantee.
      * Same defense-in-depth validation on ``n`` and ``win_rate``.
      * Same stability guarantee — ties on ``win_rate`` preserve input
        order, which matters when multiple opponents are pinned at the
        worst-loss tie value (e.g. the 0.0 "auto-loss" scenarios).

    Parameters
    ----------
    team:
        The candidate team the records are evaluated against. Held for
        contract / downstream-rendering context only — the function
        does not re-derive per-opponent win rates from ``team``.
    meta_matchup_results:
        Iterable of :class:`MetaMatchupResult`, each representing one
        opponent's set-level outcome from ``team``'s perspective.
        Consumed exactly once; works on generators.
    n:
        Number of records to return. Defaults to ``3`` — the rationale
        card convention of "bottom three" matchups. ``n = 0`` returns
        ``[]``. ``n`` larger than the input length returns every record
        in ranked order (no padding). Negative ``n`` raises
        :class:`ValueError`.

    Returns
    -------
    list[MetaMatchupResult]
        Up to ``min(n, len(input))`` records, sorted **ascending** by
        ``win_rate``. Ties on ``win_rate`` preserve input order — the
        sort is stable.

    Raises
    ------
    ValueError
        If ``n`` is negative or non-int, or if any record's
        ``win_rate`` is out of ``[0, 1]``.
    TypeError
        If any input element is not a :class:`MetaMatchupResult`.

    Notes
    -----
    The function is **pure**: it returns a fresh list and never mutates
    its input. Calling :func:`select_favorable_matchups` and
    :func:`select_unfavorable_matchups` on the *same* list (the typical
    rationale-card pipeline) is safe and order-independent.

    Why not just call ``select_favorable_matchups`` and slice the tail?
        A trailing slice would only happen to coincide when ``n``
        equals the full input length. For a general meta of size M
        with ``n < M``, the bottom-``n`` ascending list is **not** the
        reverse of the top-``n`` descending list. A dedicated function
        with a clear contract is the only honest answer.

    Why ascending sort, not ``heapq.nsmallest``?
        Same reasoning as the favorable selector: for v0.1's expected
        meta size (tens to low hundreds of teams) the ``O(n log n)``
        full sort is instant and reads more clearly. A future scaling
        AC can swap in ``heapq.nsmallest`` without changing the
        contract; the ascending-stable-sort behavior is identical for
        tie-breaking purposes.
    """

    if not isinstance(n, int) or isinstance(n, bool):
        # bool is an int subclass in Python; reject explicitly so
        # ``select_unfavorable_matchups(team, results, n=True)`` does
        # not silently collapse to ``n=1``.
        raise ValueError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    # ``team`` is part of the contract but currently unused in the
    # math, mirroring ``select_favorable_matchups``. The no-op
    # reference keeps linters from flagging the parameter.
    _ = team

    records = list(meta_matchup_results)

    for idx, rec in enumerate(records):
        if not isinstance(rec, MetaMatchupResult):
            raise TypeError(
                f"meta_matchup_results[{idx}] is not a MetaMatchupResult: "
                f"got {type(rec).__name__}"
            )
        # Defense-in-depth: re-validate the win_rate range even though
        # ``MetaMatchupResult.__post_init__`` already does.
        if not (0.0 <= rec.win_rate <= 1.0):
            raise ValueError(
                f"meta_matchup_results[{idx}].win_rate out of range: "
                f"{rec.win_rate} (must be in [0.0, 1.0])"
            )

    if n == 0 or not records:
        return []

    # Stable ascending sort: ``+win_rate`` as the key. Python's
    # ``sorted`` is stable so tied win rates preserve input order.
    # Matches the seed's interpretability principle — a deterministic
    # order means the rationale card looks the same on repeat runs.
    ranked = sorted(records, key=lambda r: r.win_rate)

    return ranked[:n]


def compute_meta_coverage(
    team: CandidateTeam,
    meta_usage_table: MetaSnapshot,
    matchup_results: Iterable[MetaMatchupResult],
    *,
    win_threshold: float = 0.5,
) -> float:
    """Usage-weighted fraction of the meta ``team`` covers (wins ≥ ``win_threshold``).

    This is the Sub-AC 3.3 headline function and the rationale-card
    sibling of :func:`gbl_hacker.score.meta_coverage` (Sub-AC 2.3). The
    score-axis version owns the simulator-driven pipeline; this version
    consumes the **pre-simulated** :class:`MetaMatchupResult` records
    the rationale card already holds, and joins them with the canonical
    usage weights carried on :class:`MetaSnapshot`.

    Coverage definition
    -------------------

    Let ``H = { opp_k ∈ meta : win_rate(team, opp_k) ≥ win_threshold }``::

        compute_meta_coverage(team, M, R) =
            Σ_{k ∈ H}  usage_pct_k
            ──────────────────────       (over the *realized* meta slice)
            Σ_k  usage_pct_k

    The denominator is the realized total usage of opponents for which a
    matchup result exists — matching the convention of
    :func:`expected_win_rate`, :func:`worst_case_robustness`, and
    :func:`gbl_hacker.score.meta_coverage`. An opponent listed in
    ``meta_usage_table.team_usage`` with **no** corresponding result in
    ``matchup_results`` is excluded from *both* numerator and denominator
    — symmetrically to the score-axis's ``on_missing_build="skip"``
    policy. The reason is the same: silently penalizing the team for an
    un-simulated opponent would conflate "we have no signal here" with
    "we lose here", which is the data-honesty anti-pattern the seed
    explicitly forbids.

    Tie-handling
    ------------

    The comparison is ``win_rate >= win_threshold`` (closed lower
    bound). At the default ``win_threshold = 0.5``, an exact 0.5 set-win
    rate (the typical tie value from the simulator) counts as
    "covered" — a GBL operator treats an even matchup as handleable, not
    as a loss. Callers wanting a strict edge pass ``win_threshold =
    0.501``; ``0.6`` is "clear edge", ``0.7`` "decisive edge".

    Parameters
    ----------
    team:
        The candidate team the records were produced for. Held for
        contract / downstream-rendering context — mirrors
        :func:`select_favorable_matchups` /
        :func:`select_unfavorable_matchups`. The function does *not*
        re-derive win rates from ``team``; that is the upstream
        simulator's job.
    meta_usage_table:
        The Taiman Party :class:`MetaSnapshot` whose ``team_usage``
        provides the canonical opponent species tuples *and* their
        usage weights. The function joins ``matchup_results`` against
        this table by opponent species tuple. Using the snapshot as the
        canonical weight source (rather than
        :attr:`MetaMatchupResult.usage_pct`) means a rationale-card
        rendering and a Pareto-ranker rendering will never disagree on
        which weights they used.
    matchup_results:
        Iterable of :class:`MetaMatchupResult`, each pinning a
        per-opponent set-level win rate. The opponent's
        ``species`` tuple is the join key. Consumed exactly once;
        works on generators. Records whose opponent does not appear in
        ``meta_usage_table.team_usage`` are silently ignored —
        rationale-card scratch lookups (e.g. a "what if we faced opp
        X" probe) must not pollute the coverage scalar.
    win_threshold:
        Win-rate cutoff in ``[0.0, 1.0]``. An opponent contributes to
        the coverage numerator iff its result meets or exceeds this
        threshold. Default ``0.5`` — "any non-losing matchup counts".

    Returns
    -------
    float
        Coverage fraction in ``[0.0, 1.0]`` — share of the realized
        meta usage mass with ``win_rate >= win_threshold``.

    Boundary behavior
    -----------------

    * **Empty meta** (``meta_usage_table.team_usage`` empty) → ``0.0``.
    * **All meta entries unmatched by results** → ``0.0``. (Denominator
      is 0; nothing to report a fraction of.)
    * **All matched results with usage_pct == 0** → ``0.0``. (Same
      reason; no realized weight.)
    * **win_threshold = 0.0** → every in-range win rate qualifies; the
      function returns ``1.0`` on any non-empty, non-zero-weight meta.
    * **win_threshold = 1.0** → only perfect-win opponents qualify.
    * **Out-of-range ``win_threshold``** → :class:`ValueError`.
    * **Out-of-range ``win_rate``** on any record → :class:`ValueError`.
    * **Non-:class:`MetaMatchupResult` entries** → :class:`TypeError`.

    Raises
    ------
    ValueError
        ``win_threshold`` ∉ ``[0.0, 1.0]``, or any record's
        ``win_rate`` ∉ ``[0.0, 1.0]``.
    TypeError
        Any entry of ``matchup_results`` is not a
        :class:`MetaMatchupResult`.

    Notes
    -----
    The function is **pure**: it returns a fresh scalar and does not
    mutate its inputs. Running the favorable / unfavorable selectors
    and ``compute_meta_coverage`` over the *same*
    ``matchup_results`` list (the standard rationale-card flow) is
    safe.

    Why is the meta snapshot the canonical weight source (not the
    record's ``usage_pct``)?
        ``MetaMatchupResult.usage_pct`` is optional — the data class
        accepts ``None`` for ad-hoc record construction. Pulling
        weights from :class:`MetaSnapshot` instead of the record keeps
        the rationale-card pipeline immune to upstream callers who
        construct records without weights. It also pins the rationale
        card to the *exact same* weights the Pareto ranker consumed,
        avoiding cross-pipeline drift.

    Why ignore unmatched records instead of treating them as 0?
        A record whose opponent is not in the meta snapshot is, by
        construction, off-meta — the operator may have probed a
        non-meta opponent for diagnostic purposes. Letting that probe
        contribute to the meta-coverage scalar would silently corrupt
        the headline interpretability number with off-meta data.

    Why ignore meta entries unmatched by results instead of treating
    them as 0-win-rate (uncovered)?
        Symmetric to the score-axis ``on_missing_build="skip"``: an
        un-simulated opponent is "no signal", not "auto-loss".
        Reporting "X of the *scored* slice" is the only honest
        statistic when partial simulation is the norm. The data-honesty
        caveat already attached to every snapshot rendering surfaces
        the uncovered slice to the operator separately; the coverage
        scalar does not double-count it.
    """

    if not (0.0 <= win_threshold <= 1.0):
        raise ValueError(
            f"win_threshold out of range: {win_threshold} "
            f"(must be in [0.0, 1.0])"
        )

    # Materialize once — the iterable may be a generator. The validation
    # pass below would otherwise consume it before the join loop runs.
    records = list(matchup_results)

    for idx, rec in enumerate(records):
        if not isinstance(rec, MetaMatchupResult):
            raise TypeError(
                f"matchup_results[{idx}] is not a MetaMatchupResult: "
                f"got {type(rec).__name__}"
            )
        # Defense-in-depth — MetaMatchupResult.__post_init__ already
        # range-checks ``win_rate``, but a caller bypassing the
        # dataclass constructor (``object.__new__`` etc.) could still
        # slip an out-of-range value through. Catch it here so the
        # coverage scalar is never silently corrupted.
        if not (0.0 <= rec.win_rate <= 1.0):
            raise ValueError(
                f"matchup_results[{idx}].win_rate out of range: "
                f"{rec.win_rate} (must be in [0.0, 1.0])"
            )

    # ``team`` is part of the contract but not used in the math — the
    # team identity is already baked into the records by construction.
    # The no-op reference keeps linters quiet and documents the
    # intentional pass-through (mirrors siblings).
    _ = team

    # Index results by opponent species tuple — the natural join key
    # against ``MetaSnapshot.team_usage[i].members`` (both are
    # ``tuple[str, str, str]``). On duplicate keys (the simulator
    # produced multiple results for the same opponent) the *last* one
    # wins, matching ``dict`` semantics and the most-recent-data
    # convention used elsewhere in the engine.
    results_by_opponent: dict[tuple[str, str, str], MetaMatchupResult] = {
        rec.opponent.species: rec for rec in records
    }

    covered_weight = 0.0
    total_weight = 0.0
    for usage in meta_usage_table.team_usage:
        key = usage.members  # already a tuple[str, str, str]
        result = results_by_opponent.get(key)
        if result is None:
            # Un-simulated opponent — see "Why ignore meta entries
            # unmatched by results" in the docstring.
            continue
        total_weight += usage.usage_pct
        if result.win_rate >= win_threshold:
            covered_weight += usage.usage_pct

    if total_weight == 0.0:
        # Either the meta is empty, or every meta entry was either
        # unmatched or at usage_pct == 0. Either way there is no
        # covered slice to report a fraction of; ``0.0`` matches the
        # score-axis sibling's degenerate-input contract.
        return 0.0

    coverage = covered_weight / total_weight
    # Defensive clamp. ``covered_weight`` and ``total_weight`` are both
    # non-negative and ``covered_weight <= total_weight`` by
    # construction, so the ratio is in [0, 1] absent floating-point
    # pathology — the clamp catches that pathology without changing
    # well-formed results.
    if coverage < 0.0:
        return 0.0
    if coverage > 1.0:
        return 1.0
    return coverage


@dataclass(frozen=True, slots=True)
class RationaleCard:
    """Per-team explanation container — the seed-ontology ``rationale_card``.

    Mirrors the ``rationale_card`` concept in the seed ontology: a
    per-team explanation containing **key favorable matchups, key
    unfavorable matchups, and the meta-coverage breakdown**. (The
    ontology also names a "short prose justification"; that field is
    deferred to a later sub-AC and is *not* required by Sub-AC 3.4 —
    see the module docstring.)

    The dataclass is :attr:`frozen <dataclasses.dataclass.frozen>` and
    uses :attr:`slots <dataclasses.dataclass.slots>` for the same
    reasons the rest of the score package does: structural immutability
    (the rationale card is consumed by renderers that must not mutate
    it) and a flat memory layout that makes ``Score`` / ``ScoredTeam``
    / ``RationaleCard`` cheap to materialize in bulk.

    The favorable / unfavorable lists are stored as :class:`tuple` —
    not :class:`list` — so that ``RationaleCard`` instances are
    hashable-friendly and structurally immutable. Renderers that need
    to iterate them treat them as ordered sequences; renderers that
    want a list call ``list(card.favorable)`` at the boundary.

    Attributes
    ----------
    team:
        The subject :class:`CandidateTeam` — the team this card
        explains. Pinned at construction so a downstream renderer can
        print "team T's rationale: …" without re-threading the team
        identity through a parallel channel.
    favorable:
        Ordered tuple of :class:`MetaMatchupResult` representing the
        team's key wins. By convention (the
        :func:`select_favorable_matchups` contract) this tuple is in
        **descending ``win_rate``** order; this dataclass does not
        re-sort it, so an upstream caller that built the tuple by
        hand-picking records preserves their picked order exactly.
    unfavorable:
        Ordered tuple of :class:`MetaMatchupResult` representing the
        team's key losses. By convention (the
        :func:`select_unfavorable_matchups` contract) this tuple is
        in **ascending ``win_rate``** order; this dataclass does not
        re-sort it.
    coverage:
        Scalar in ``[0.0, 1.0]`` — the meta-coverage slice the team
        handles. Produced by :func:`compute_meta_coverage` upstream.

    Invariants
    ----------

    * ``team`` is a :class:`CandidateTeam`.
    * Every entry of ``favorable`` and ``unfavorable`` is a
      :class:`MetaMatchupResult`.
    * ``coverage`` is a real number in ``[0.0, 1.0]`` (``NaN`` rejected).

    These invariants are enforced in :meth:`__post_init__` so any
    ``RationaleCard`` instance in hand is already well-formed — the
    downstream renderer never has to re-validate.

    Why no prose justification (yet)?
        Sub-AC 3.4's explicit minimum deliverable is "favorable list,
        unfavorable list, coverage slice". The prose justification
        named in the ontology is a downstream-rendering concern that
        composes *over* this structured card; landing it as a separate
        sub-AC keeps the structural contract auditable in isolation.
    """

    team: CandidateTeam
    favorable: tuple[MetaMatchupResult, ...]
    unfavorable: tuple[MetaMatchupResult, ...]
    coverage: float

    def __post_init__(self) -> None:
        if not isinstance(self.team, CandidateTeam):
            raise TypeError(
                f"RationaleCard.team must be a CandidateTeam, "
                f"got {type(self.team).__name__}"
            )

        # Both bullet lists must be *tuples* of MetaMatchupResult. The
        # tuple shape is part of the structural-immutability contract;
        # accepting a list would let a downstream renderer mutate the
        # card's bullet ordering.
        if not isinstance(self.favorable, tuple):
            raise TypeError(
                f"RationaleCard.favorable must be a tuple, "
                f"got {type(self.favorable).__name__}"
            )
        if not isinstance(self.unfavorable, tuple):
            raise TypeError(
                f"RationaleCard.unfavorable must be a tuple, "
                f"got {type(self.unfavorable).__name__}"
            )
        for idx, rec in enumerate(self.favorable):
            if not isinstance(rec, MetaMatchupResult):
                raise TypeError(
                    f"RationaleCard.favorable[{idx}] is not a "
                    f"MetaMatchupResult: got {type(rec).__name__}"
                )
        for idx, rec in enumerate(self.unfavorable):
            if not isinstance(rec, MetaMatchupResult):
                raise TypeError(
                    f"RationaleCard.unfavorable[{idx}] is not a "
                    f"MetaMatchupResult: got {type(rec).__name__}"
                )

        # ``coverage`` must be a real number in [0, 1]. NaN is rejected
        # explicitly — NaN comparisons all return False, so a NaN here
        # would silently pass an ``if not (0 <= cov <= 1):`` check on
        # some Python builds. Match the score-axis convention of
        # ``_validate_axis`` in pareto.py.
        if not isinstance(self.coverage, (int, float)) or isinstance(
            self.coverage, bool
        ):
            raise TypeError(
                f"RationaleCard.coverage must be a real number, "
                f"got {type(self.coverage).__name__}"
            )
        if math.isnan(self.coverage):
            raise ValueError(
                "RationaleCard.coverage is NaN; must be a real number "
                "in [0.0, 1.0]"
            )
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError(
                f"RationaleCard.coverage out of range: {self.coverage} "
                f"(must be in [0.0, 1.0])"
            )


def build_rationale_card(
    team: CandidateTeam,
    favorable: Iterable[MetaMatchupResult],
    unfavorable: Iterable[MetaMatchupResult],
    coverage: float,
) -> RationaleCard:
    """Assemble a :class:`RationaleCard` from pre-computed pieces.

    This is the Sub-AC 3.4 headline function. It is the **single
    funnel** the rationale-card pipeline uses to go from the three
    structured pieces produced by the earlier Sub-AC 3 entries —

      * Sub-AC 3.1 :func:`select_favorable_matchups`   → ``favorable``
      * Sub-AC 3.2 :func:`select_unfavorable_matchups` → ``unfavorable``
      * Sub-AC 3.3 :func:`compute_meta_coverage`        → ``coverage``

    — to the seed-ontology ``rationale_card`` shape that downstream
    renderers consume. Centralizing the assembly here means:

    * Future renderers (CLI table, JSON exporter, eventual web UI)
      never see hand-rolled card dicts — they consume
      :class:`RationaleCard` instances whose invariants are already
      enforced.
    * The favorable / unfavorable / coverage drift the rationale
      module fights for elsewhere (see ``compute_meta_coverage``'s
      "Sibling-but-distinct…" docstring) collapses into one structural
      contract here. If a caller wires the wrong pieces in, the
      validator catches it.

    Parameters
    ----------
    team:
        The subject :class:`CandidateTeam` — the team this card
        explains. Held verbatim on the returned card.
    favorable:
        Iterable of :class:`MetaMatchupResult` — the team's "key wins"
        bullets. Materialized once into a tuple. **The function does
        not re-sort** the input: upstream is expected to have already
        applied :func:`select_favorable_matchups` (descending
        ``win_rate``). A caller who hand-picks records in custom order
        keeps that order in the card.
    unfavorable:
        Iterable of :class:`MetaMatchupResult` — the team's "key
        losses" bullets. Materialized once into a tuple. **Not
        re-sorted**: upstream is expected to have already applied
        :func:`select_unfavorable_matchups` (ascending ``win_rate``).
    coverage:
        Scalar in ``[0.0, 1.0]`` — the meta-coverage slice produced
        by :func:`compute_meta_coverage`. Passed through verbatim.

    Returns
    -------
    RationaleCard
        A frozen, structurally-validated rationale card with all three
        required fields populated.

    Raises
    ------
    TypeError
        ``team`` is not a :class:`CandidateTeam`, any entry of
        ``favorable`` / ``unfavorable`` is not a
        :class:`MetaMatchupResult`, or ``coverage`` is not a real
        number.
    ValueError
        ``coverage`` is NaN or outside ``[0.0, 1.0]``.

    Notes
    -----
    The function is **pure**: it never mutates its inputs and returns
    a fresh card. Materialization to ``tuple`` consumes each iterable
    exactly once (generators work).

    Why a function on top of the dataclass constructor?
        ``RationaleCard(team=..., favorable=tuple(it), ...)`` would
        also work, but every caller would have to remember to
        ``tuple()``-wrap the iterables. The function eliminates that
        ceremony and matches the rest of the rationale-module
        conventions (``select_favorable_matchups`` /
        ``select_unfavorable_matchups`` / ``compute_meta_coverage``
        are all functions over the data class types).

    Why no re-sorting?
        The function's contract is "assemble", not "rank". Re-sorting
        would silently override an upstream caller's intentional
        ordering — a future Sub-AC may produce custom-curated bullet
        lists, and a hidden re-sort would corrupt them. The validator
        instead enforces *type* correctness; *order* correctness is
        delegated to the upstream selectors.
    """

    return RationaleCard(
        team=team,
        favorable=tuple(favorable),
        unfavorable=tuple(unfavorable),
        coverage=coverage,
    )


__all__ = [
    "MetaMatchupResult",
    "RationaleCard",
    "build_rationale_card",
    "compute_meta_coverage",
    "select_favorable_matchups",
    "select_unfavorable_matchups",
]
