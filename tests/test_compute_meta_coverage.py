"""Unit tests for ``compute_meta_coverage`` (Sub-AC 3.3).

The headline contract this test suite enforces:

* ``compute_meta_coverage(team, meta_usage_table, matchup_results,
  win_threshold=...)`` returns a single scalar in ``[0, 1]`` — the
  usage-weighted **fraction of the meta** the team handles at or above
  ``win_threshold``.
* Coverage is a **mass** statistic (count, not rate); the threshold
  partitions opponents into "covered" vs "not covered" and the
  numerator is the realized usage share of the covered set.
* The denominator is the *realized* total usage of opponents matched in
  both the meta table and the results — matching the convention of the
  score-axis :func:`gbl_hacker.score.meta_coverage` (Sub-AC 2.3).
* The join key between meta entries and matchup results is the opponent
  species tuple (``lead, safe_swap, closer``).
* Unmatched-on-either-side opponents are silently dropped — symmetric
  to the score-axis's ``on_missing_build="skip"`` policy and the
  data-honesty principle (no-signal ≠ auto-loss).
* The function is **pure**: it does not mutate its inputs.
* ``MetaMatchupResult.usage_pct`` is *not* the canonical weight source
  — :class:`MetaSnapshot` is. The rationale-card pipeline and the
  Pareto-ranker pipeline must agree byte-for-byte on which weights they
  used, which is only possible if both pull from the snapshot.

This file's headline test — ``test_fixed_meta_and_results_fixture`` —
is the AC's explicit minimum deliverable: a fixed meta + fixed results
pair with a hand-computed coverage value. The remaining tests fence in
the documented contract so the rationale-card renderer can rely on it
without re-checking.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gbl_hacker.parse.taiman import (
    GREAT_LEAGUE_LABEL,
    MetaSnapshot,
    PokemonUsage,
    TeamUsage,
)
from gbl_hacker.score import (
    CandidateTeam,
    MetaMatchupResult,
    compute_meta_coverage,
)
from gbl_hacker.simulator import (
    ChargedMove,
    CombatantBuild,
    FastMove,
)

# --- fixture helpers ------------------------------------------------------
# Structurally aligned with test_meta_coverage.py / test_select_*.py so
# cross-axis fixture diffing stays a one-glance affair.


def _build(species: str, *, max_hp: int = 100, fast_damage: int = 2) -> CombatantBuild:
    return CombatantBuild(
        species=species,
        max_hp=max_hp,
        fast=FastMove(name="quick", damage=fast_damage, energy_gain=8),
        charged=ChargedMove(name="bomb", energy_cost=40, damage=70),
    )


def _candidate_team(*species_names: str) -> CandidateTeam:
    if len(species_names) != 3:
        raise AssertionError(
            f"test fixture needs 3 species, got {len(species_names)}"
        )
    return CandidateTeam.from_slots(_build(s) for s in species_names)


def _opp_team(prefix: str) -> CandidateTeam:
    """Build an opponent ``CandidateTeam`` keyed by a single prefix."""

    return _candidate_team(f"{prefix}-1", f"{prefix}-2", f"{prefix}-3")


def _make_meta(*team_specs: tuple[str, float]) -> MetaSnapshot:
    """Build a MetaSnapshot from ``(opp_prefix, usage_pct)`` tuples."""

    team_usage = tuple(
        TeamUsage(
            members=(f"{prefix}-1", f"{prefix}-2", f"{prefix}-3"),
            usage_pct=pct,
        )
        for prefix, pct in team_specs
    )
    return MetaSnapshot(
        league=GREAT_LEAGUE_LABEL,
        rating_bracket="upper",
        fetched_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        source_url="https://pokemongo-get.com/taimanparty",
        source_caveat=(
            "Taiman Party report density drops past upper bracket — "
            "top-500-specific slices are NOT reliable."
        ),
        pokemon_usage=(PokemonUsage(species="lead", usage_pct=15.0),),
        team_usage=team_usage,
    )


def _record(prefix: str, win_rate: float) -> MetaMatchupResult:
    """Build a ``MetaMatchupResult`` for the opponent at ``prefix``.

    Deliberately omits ``usage_pct`` — the function must pull weights
    from the meta snapshot, not from the record. Tests that set the
    record's ``usage_pct`` to a *contradictory* value below assert that
    the snapshot wins.
    """

    return MetaMatchupResult(opponent=_opp_team(prefix), win_rate=win_rate)


# ---------------------------------------------------------------------------
# The AC's explicit minimum: a fixed meta + fixed results fixture
# ---------------------------------------------------------------------------


def test_fixed_meta_and_results_fixture() -> None:
    """Fixed-fixture coverage matches the hand-computed value.

    Meta (4 opponents):
      * A: usage 40 %     · win 0.80  → covered  (≥ 0.5)
      * B: usage 30 %     · win 0.20  → NOT covered
      * C: usage 20 %     · win 0.60  → covered
      * D: usage 10 %     · win 0.40  → NOT covered

    Realized total usage      = 40 + 30 + 20 + 10 = 100
    Realized covered usage    = 40 + 20           =  60
    Expected coverage         = 60 / 100          =   0.60

    This is the AC's named fixed fixture. A bug that:
      * returned a weighted-mean win rate (∑ usage·win / ∑ usage)
        would yield 0.4·0.8 + 0.3·0.2 + 0.2·0.6 + 0.1·0.4 = 0.54
        (distinguishable from 0.60).
      * divided by 100 ignoring realized total would still yield 0.60
        on this fixture — covered by the truncation test below.
      * used strict ``>`` instead of ``>=`` against an exact-0.5 win
        would still yield 0.60 here — covered by the tie test below.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        ("opp-a", 40.0),
        ("opp-b", 30.0),
        ("opp-c", 20.0),
        ("opp-d", 10.0),
    )
    results = [
        _record("opp-a", 0.80),
        _record("opp-b", 0.20),
        _record("opp-c", 0.60),
        _record("opp-d", 0.40),
    ]
    score = compute_meta_coverage(
        team, meta, results, win_threshold=0.5
    )
    assert score == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Degenerate / empty inputs
# ---------------------------------------------------------------------------


def test_empty_meta_returns_zero() -> None:
    """No team_usage rows ⇒ no realized slice ⇒ 0.0 (matches sibling)."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta()
    score = compute_meta_coverage(team, meta, [])
    assert score == 0.0


def test_empty_results_returns_zero() -> None:
    """No matchup results ⇒ no realized slice ⇒ 0.0.

    A meta entry that has no corresponding result is "no signal"; with
    every entry un-simulated the denominator is 0 and the function
    returns 0.0 (data-honesty: no-signal ≠ auto-loss).
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 50.0), ("opp-b", 50.0))
    score = compute_meta_coverage(team, meta, [])
    assert score == 0.0


def test_zero_total_weight_returns_zero() -> None:
    """All matched usages == 0 ⇒ no realized slice ⇒ 0.0, no division-by-zero."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 0.0), ("opp-b", 0.0))
    results = [_record("opp-a", 1.0), _record("opp-b", 1.0)]
    score = compute_meta_coverage(team, meta, results)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Single-opponent semantics — binary above/below threshold
# ---------------------------------------------------------------------------


def test_single_opponent_above_threshold_returns_one() -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 42.0))
    results = [_record("opp-a", 0.75)]
    assert compute_meta_coverage(team, meta, results) == 1.0


def test_single_opponent_below_threshold_returns_zero() -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 42.0))
    results = [_record("opp-a", 0.25)]
    assert compute_meta_coverage(team, meta, results) == 0.0


def test_exact_tie_at_default_threshold_is_counted_as_covered() -> None:
    """``rate == win_threshold`` is "covered" (closed lower bound).

    Mirrors the score-axis sibling: at default 0.5 an exact 0.5 win
    rate counts toward coverage. GBL operators treat an even matchup
    as handleable, not as a loss.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 50.0))
    results = [_record("opp-a", 0.5)]
    assert compute_meta_coverage(team, meta, results, win_threshold=0.5) == 1.0


def test_strict_threshold_excludes_exact_tie() -> None:
    """A caller wanting strict edge passes ``win_threshold = 0.501``."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 50.0))
    results = [_record("opp-a", 0.5)]
    assert (
        compute_meta_coverage(team, meta, results, win_threshold=0.501) == 0.0
    )


# ---------------------------------------------------------------------------
# Coverage-mass arithmetic
# ---------------------------------------------------------------------------


def test_coverage_is_independent_of_covered_rate_magnitude() -> None:
    """A 0.51 win counts the same as a 0.99 win — coverage is mass, not rate.

    Same opponent weights, different magnitudes of the covered rate.
    Both must yield identical coverage, since the coverage statistic
    does not weight by how decisively you beat each covered opponent.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 60.0), ("opp-b", 40.0))
    barely = [_record("opp-a", 0.51), _record("opp-b", 0.0)]
    crushing = [_record("opp-a", 0.99), _record("opp-b", 0.0)]
    assert compute_meta_coverage(team, meta, barely) == pytest.approx(0.6)
    assert compute_meta_coverage(team, meta, crushing) == pytest.approx(0.6)


def test_coverage_is_monotone_decreasing_in_threshold() -> None:
    """Tightening the threshold can only shrink (or preserve) coverage.

    Setup (rates spanning 0.3, 0.5, 0.7, 0.9 at equal 25 % weights):
      * Opp A: 25 %, 0.3
      * Opp B: 25 %, 0.5
      * Opp C: 25 %, 0.7
      * Opp D: 25 %, 0.9
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        ("opp-a", 25.0),
        ("opp-b", 25.0),
        ("opp-c", 25.0),
        ("opp-d", 25.0),
    )
    results = [
        _record("opp-a", 0.3),
        _record("opp-b", 0.5),
        _record("opp-c", 0.7),
        _record("opp-d", 0.9),
    ]

    def _cov(t: float) -> float:
        return compute_meta_coverage(team, meta, results, win_threshold=t)

    # Threshold 0.0 — every in-range rate qualifies        → 1.00
    # Threshold 0.5 — B, C, D qualify (≥ 0.5)             → 0.75
    # Threshold 0.7 — C, D qualify                        → 0.50
    # Threshold 0.9 — D qualifies                         → 0.25
    # Threshold 0.91 — nothing qualifies                  → 0.00
    assert _cov(0.0) == pytest.approx(1.0)
    assert _cov(0.5) == pytest.approx(0.75)
    assert _cov(0.7) == pytest.approx(0.5)
    assert _cov(0.9) == pytest.approx(0.25)
    assert _cov(0.91) == pytest.approx(0.0)

    # Property check: a fine grid must be non-increasing in threshold.
    rates = [_cov(t / 100.0) for t in range(0, 101)]
    for lo, hi in zip(rates[1:], rates[:-1]):
        assert lo <= hi + 1e-12, (
            f"coverage rose with stricter threshold: {hi} → {lo}"
        )


def test_threshold_zero_returns_one_for_any_realized_meta() -> None:
    """``win_threshold = 0.0`` ⇒ every in-range rate qualifies ⇒ 1.0."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 10.0), ("opp-b", 90.0))
    results = [_record("opp-a", 0.0), _record("opp-b", 0.0)]
    assert (
        compute_meta_coverage(team, meta, results, win_threshold=0.0)
        == pytest.approx(1.0)
    )


def test_threshold_one_only_counts_perfect_wins() -> None:
    """``win_threshold = 1.0`` ⇒ only perfect-win opponents qualify."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        ("opp-a", 30.0),  # perfect win
        ("opp-b", 70.0),  # near-perfect, doesn't qualify
    )
    results = [_record("opp-a", 1.0), _record("opp-b", 0.99)]
    assert (
        compute_meta_coverage(team, meta, results, win_threshold=1.0)
        == pytest.approx(0.3)
    )


# ---------------------------------------------------------------------------
# Truncation / denominator-normalization contract
# ---------------------------------------------------------------------------


def test_truncated_meta_normalizes_against_realized_total() -> None:
    """A meta summing to < 100 is reported as fraction of *realized* slice.

    Setup:
      * Opp A: usage 20 %, win 0.8 (covered)
      * Opp B: usage 10 %, win 0.2 (not covered)

    Realized total = 30. Covered = 20. Expected = 20/30 ≈ 0.667.
    A bug that divided by 100 would return 0.20 (distinguishable).
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 20.0), ("opp-b", 10.0))
    results = [_record("opp-a", 0.8), _record("opp-b", 0.2)]
    assert compute_meta_coverage(team, meta, results) == pytest.approx(
        20.0 / 30.0
    )


# ---------------------------------------------------------------------------
# Join-key semantics — what counts as "in" the meta?
# ---------------------------------------------------------------------------


def test_unmatched_results_are_silently_dropped() -> None:
    """Results for off-meta opponents do not contribute to coverage.

    A rationale-card diagnostic might probe a non-meta opponent. That
    probe's record must not leak into the coverage scalar.

    Setup:
      * Meta: opp-a (60 %, covered), opp-b (40 %, not covered).
      * Results include a third record for opp-z (off-meta), win 1.0.
        opp-z is *not* in the meta table; it must be dropped entirely.

    Coverage = 60 / 100 = 0.60. A bug that included opp-z's record in
    the numerator would inflate the score above 0.60.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 60.0), ("opp-b", 40.0))
    results = [
        _record("opp-a", 0.8),
        _record("opp-b", 0.0),
        _record("opp-z", 1.0),  # off-meta — must not pollute the score
    ]
    assert compute_meta_coverage(team, meta, results) == pytest.approx(0.6)


def test_meta_entries_unmatched_by_results_drop_from_denominator() -> None:
    """An un-simulated meta entry contributes to neither numerator nor denominator.

    Setup:
      * Meta: opp-a (60 %, covered), opp-b (40 %, no result).
      * Results: only opp-a (win 0.8). opp-b has no signal.

    Realized denominator = 60 (only opp-a is scored).
    Covered numerator    = 60.
    Coverage             = 60 / 60 = 1.0.

    A bug that kept opp-b at the denominator anyway would yield
    60/100 = 0.6 (distinguishable).
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 60.0), ("opp-b", 40.0))
    results = [_record("opp-a", 0.8)]
    assert compute_meta_coverage(team, meta, results) == pytest.approx(1.0)


def test_meta_snapshot_is_canonical_weight_source() -> None:
    """Record ``usage_pct`` is ignored; snapshot weights win.

    Setup:
      * Meta: opp-a (70 %), opp-b (30 %). Snapshot weights.
      * Results: opp-a (win 1.0, usage_pct=10 *wrong*),
                 opp-b (win 0.0, usage_pct=90 *wrong*).

    If the snapshot is canonical, coverage = 70/100 = 0.7.
    If the record's usage_pct were canonical, coverage would be
    10/100 = 0.1 (clearly distinguishable). The test pins the
    snapshot-as-truth contract documented in the function docstring.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 70.0), ("opp-b", 30.0))
    results = [
        MetaMatchupResult(
            opponent=_opp_team("opp-a"), win_rate=1.0, usage_pct=10.0
        ),
        MetaMatchupResult(
            opponent=_opp_team("opp-b"), win_rate=0.0, usage_pct=90.0
        ),
    ]
    assert compute_meta_coverage(team, meta, results) == pytest.approx(0.7)


def test_duplicate_results_take_last() -> None:
    """Duplicate results for the same opponent — last write wins.

    Mirrors ``dict`` semantics and the most-recent-data convention.
    Two records for opp-a: first with win 0.0, second with win 1.0.
    The second must take precedence → opp-a covered → coverage = 1.0.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 100.0))
    results = [_record("opp-a", 0.0), _record("opp-a", 1.0)]
    assert compute_meta_coverage(team, meta, results) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Range / contract enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, -1.0, 2.0, 99.9])
def test_win_threshold_out_of_range_rejected(bad_threshold: float) -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 100.0))
    results = [_record("opp-a", 0.5)]
    with pytest.raises(ValueError, match="win_threshold"):
        compute_meta_coverage(
            team, meta, results, win_threshold=bad_threshold
        )


def test_non_record_input_raises_type_error() -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 100.0))
    with pytest.raises(TypeError, match="MetaMatchupResult"):
        compute_meta_coverage(team, meta, ["not-a-record"])  # type: ignore[list-item]


def test_record_with_bypassed_constructor_out_of_range_raises() -> None:
    """A caller bypassing the dataclass constructor still hits the validator.

    ``MetaMatchupResult.__post_init__`` checks ``win_rate ∈ [0, 1]``,
    but ``object.__new__`` skips it. ``compute_meta_coverage`` must
    re-validate so a bypass cannot silently corrupt the scalar.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 100.0))
    bypass = object.__new__(MetaMatchupResult)
    object.__setattr__(bypass, "opponent", _opp_team("opp-a"))
    object.__setattr__(bypass, "win_rate", 1.5)  # out of range
    object.__setattr__(bypass, "usage_pct", None)
    with pytest.raises(ValueError, match="out of range"):
        compute_meta_coverage(team, meta, [bypass])


def test_result_is_always_in_unit_interval() -> None:
    """For any in-range rates and thresholds the coverage is in [0, 1]."""

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        ("opp-a", 33.3),
        ("opp-b", 22.2),
        ("opp-c", 11.1),
    )
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        results = [
            _record("opp-a", rate),
            _record("opp-b", rate),
            _record("opp-c", rate),
        ]
        for threshold in (0.0, 0.25, 0.5, 0.75, 1.0):
            score = compute_meta_coverage(
                team, meta, results, win_threshold=threshold
            )
            assert 0.0 <= score <= 1.0
            expected = 1.0 if rate >= threshold else 0.0
            assert score == pytest.approx(expected), (
                f"rate={rate}, threshold={threshold}: "
                f"expected {expected}, got {score}"
            )


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_does_not_mutate_inputs() -> None:
    """Running the function leaves both inputs identical.

    The rationale-card pipeline calls
    ``compute_meta_coverage`` *and* ``select_{favorable,unfavorable}_
    matchups`` over the *same* results list. The order-independence of
    that flow depends on each function being pure.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 60.0), ("opp-b", 40.0))
    results = [_record("opp-a", 0.8), _record("opp-b", 0.2)]

    # Snapshot the records' identity and order.
    pre_order = list(results)
    pre_team_usage = tuple(meta.team_usage)

    _ = compute_meta_coverage(team, meta, results)

    # Records list (same object) is unchanged; team_usage tuple is
    # also unchanged.
    assert results == pre_order
    assert results is not pre_order  # identity check on our snapshot
    assert meta.team_usage == pre_team_usage


def test_accepts_generator_input() -> None:
    """The ``matchup_results`` parameter is documented as an Iterable.

    A generator must work — it gets consumed exactly once internally.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 60.0), ("opp-b", 40.0))

    def _gen():
        yield _record("opp-a", 0.8)
        yield _record("opp-b", 0.2)

    assert compute_meta_coverage(team, meta, _gen()) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Cross-axis sanity — parity with the score-axis sibling on a known fixture
# ---------------------------------------------------------------------------


def test_parity_with_score_axis_meta_coverage_on_shared_fixture() -> None:
    """``compute_meta_coverage`` and ``score.meta_coverage`` agree.

    Both functions express the same statistic — the difference is
    only the input layer (pre-computed results vs. simulator
    callable). On a shared fixture they must produce byte-identical
    scalars, otherwise the rationale card and the Pareto ranker would
    disagree on the headline coverage number.

    Setup:
      * Meta: opp-a (60 %), opp-b (40 %).
      * Win rates: opp-a → 0.8, opp-b → 0.2.

    Score-axis (with a constant-mapped ``set_win_rate_fn`` injection)
    and rationale-axis (with the corresponding records) must both
    return 0.6.
    """

    from gbl_hacker.score import meta_coverage as score_axis_meta_coverage

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(("opp-a", 60.0), ("opp-b", 40.0))
    results = [_record("opp-a", 0.8), _record("opp-b", 0.2)]

    # Build the registry / set_win_rate_fn pair needed by the
    # score-axis sibling. The mapping mirrors the per-record win
    # rates above.
    rationale_axis = compute_meta_coverage(team, meta, results)

    registry = {f"opp-a-{i}": _build(f"opp-a-{i}") for i in (1, 2, 3)}
    registry.update({f"opp-b-{i}": _build(f"opp-b-{i}") for i in (1, 2, 3)})

    rate_by_team: dict[tuple[str, str, str], float] = {
        ("opp-a-1", "opp-a-2", "opp-a-3"): 0.8,
        ("opp-b-1", "opp-b-2", "opp-b-3"): 0.2,
    }

    def _swr(a: CandidateTeam, b: CandidateTeam) -> float:
        return rate_by_team[b.species]

    score_axis = score_axis_meta_coverage(
        team,
        meta,
        build_registry=registry,
        set_win_rate_fn=_swr,
    )

    assert rationale_axis == pytest.approx(0.6)
    assert score_axis == pytest.approx(0.6)
    assert rationale_axis == pytest.approx(score_axis)
