"""Unit tests for ``build_rationale_card`` (Sub-AC 3.4).

The Sub-AC's explicit minimum deliverable:

    "Assemble the rationale card data structure (favorable list,
    unfavorable list, coverage slice) for one team, with a unit test
    asserting all three fields are present and well-formed."

The headline test ``test_all_three_fields_present_and_well_formed`` is
the AC's named minimum. The remaining tests fence in the documented
contract so downstream renderers (CLI table, JSON exporter, eventual
web UI) can rely on the structural-immutability + validation guarantees
without re-checking.

Test categories (in file order):

1. **Headline AC test** — the three required fields are present and
   well-formed on a realistic input.
2. **Pass-through fidelity** — values arrive on the card byte-identical
   to the inputs (no silent re-sorting / coercion / clamping).
3. **Iterable consumption** — generators and arbitrary iterables work;
   each is consumed exactly once and materialized to ``tuple``.
4. **Immutability** — the returned card is frozen; assignment raises.
5. **Purity** — the function does not mutate any of its inputs.
6. **Validation** — out-of-range / wrong-type inputs are rejected loudly
   at the boundary (the rationale-card layer is the last seam before
   rendering).
7. **End-to-end wiring** — the card composes correctly with the
   sibling Sub-AC 3.1 / 3.2 / 3.3 functions.

The fixture helpers mirror ``test_select_favorable_matchups.py`` and
``test_compute_meta_coverage.py`` line for line so cross-axis fixture
diffing stays a one-glance affair.
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
    RationaleCard,
    build_rationale_card,
    compute_meta_coverage,
    select_favorable_matchups,
    select_unfavorable_matchups,
)
from gbl_hacker.simulator import (
    ChargedMove,
    CombatantBuild,
    FastMove,
)


# --- fixture helpers ------------------------------------------------------
# Aligned with test_select_favorable_matchups.py / test_compute_meta_coverage.py
# so the end-to-end wiring test below ("compose with sibling selectors") is
# meaningful and the test files diff cleanly.


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
    """Build an opponent CandidateTeam keyed by a single prefix."""

    return _candidate_team(f"{prefix}-1", f"{prefix}-2", f"{prefix}-3")


def _record(
    prefix: str,
    win_rate: float,
    *,
    usage_pct: float | None = None,
) -> MetaMatchupResult:
    """Build a ``MetaMatchupResult`` for the opponent at ``prefix``."""

    return MetaMatchupResult(
        opponent=_opp_team(prefix),
        win_rate=win_rate,
        usage_pct=usage_pct,
    )


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


# ---------------------------------------------------------------------------
# Headline AC test — "all three fields present and well-formed"
# ---------------------------------------------------------------------------


def test_all_three_fields_present_and_well_formed() -> None:
    """The card carries the team, both bullet lists, and a valid coverage scalar.

    This is the Sub-AC 3.4 named minimum deliverable. It asserts on
    *one* call to ``build_rationale_card`` that:

      1. ``card.team``           — preserved and identifiable as the
                                    subject ``CandidateTeam`` (the
                                    rationale card's "for which team"
                                    line resolves to this object).
      2. ``card.favorable``      — present as a tuple of
                                    ``MetaMatchupResult``, non-empty
                                    on the realistic fixture, and in
                                    descending ``win_rate`` order.
      3. ``card.unfavorable``    — present as a tuple of
                                    ``MetaMatchupResult``, non-empty
                                    on the realistic fixture, and in
                                    ascending ``win_rate`` order.
      4. ``card.coverage``       — present as a float in [0, 1] and
                                    not NaN.

    "Well-formed" is asserted *structurally* (right types, right value
    ranges, right ordering convention) — not by exact-equality against
    a hand-computed expected value. Exact-value parity with the
    sibling sub-ACs is fenced in their own test suites
    (``test_select_favorable_matchups.py``, etc.); this test asserts
    the *assembly* contract.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    favorable = [
        _record("opp-strong-A", 0.90),
        _record("opp-strong-B", 0.75),
        _record("opp-strong-C", 0.62),
    ]  # already in descending order — what select_favorable_matchups returns
    unfavorable = [
        _record("opp-weak-X", 0.10),
        _record("opp-weak-Y", 0.25),
        _record("opp-weak-Z", 0.40),
    ]  # already in ascending order — what select_unfavorable_matchups returns
    coverage = 0.6

    card = build_rationale_card(team, favorable, unfavorable, coverage)

    # --- Field 1: team ---------------------------------------------------
    # The card's ``team`` is the subject team — held by identity, not a
    # silently-rebuilt copy.
    assert isinstance(card, RationaleCard), (
        "build_rationale_card must return a RationaleCard instance"
    )
    assert card.team is team, "card.team must preserve the subject team identity"
    assert card.team.species == ("y-1", "y-2", "y-3")

    # --- Field 2: favorable ---------------------------------------------
    assert isinstance(card.favorable, tuple), (
        f"card.favorable must be a tuple (immutable bullet list), "
        f"got {type(card.favorable).__name__}"
    )
    assert len(card.favorable) == 3, (
        f"card.favorable should carry the 3 records the caller supplied, "
        f"got {len(card.favorable)}"
    )
    for idx, rec in enumerate(card.favorable):
        assert isinstance(rec, MetaMatchupResult), (
            f"card.favorable[{idx}] is not a MetaMatchupResult"
        )
        assert 0.0 <= rec.win_rate <= 1.0
    # Favorable convention: descending by win_rate.
    favorable_rates = [r.win_rate for r in card.favorable]
    assert favorable_rates == sorted(favorable_rates, reverse=True), (
        f"card.favorable not in descending win_rate order: {favorable_rates}"
    )

    # --- Field 3: unfavorable -------------------------------------------
    assert isinstance(card.unfavorable, tuple), (
        f"card.unfavorable must be a tuple (immutable bullet list), "
        f"got {type(card.unfavorable).__name__}"
    )
    assert len(card.unfavorable) == 3, (
        f"card.unfavorable should carry the 3 records the caller supplied, "
        f"got {len(card.unfavorable)}"
    )
    for idx, rec in enumerate(card.unfavorable):
        assert isinstance(rec, MetaMatchupResult), (
            f"card.unfavorable[{idx}] is not a MetaMatchupResult"
        )
        assert 0.0 <= rec.win_rate <= 1.0
    # Unfavorable convention: ascending by win_rate.
    unfavorable_rates = [r.win_rate for r in card.unfavorable]
    assert unfavorable_rates == sorted(unfavorable_rates), (
        f"card.unfavorable not in ascending win_rate order: {unfavorable_rates}"
    )

    # --- Field 4: coverage ----------------------------------------------
    assert isinstance(card.coverage, float), (
        f"card.coverage must be a float, got {type(card.coverage).__name__}"
    )
    assert 0.0 <= card.coverage <= 1.0, (
        f"card.coverage out of [0, 1]: {card.coverage}"
    )
    # NaN check (NaN != NaN). Defensive: ``__post_init__`` rejects NaN
    # at construction time, but the headline test names this explicitly.
    assert card.coverage == card.coverage, "card.coverage is NaN"
    # Exact-value byte-equal pass-through (no clamping / coercion).
    assert card.coverage == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Pass-through fidelity — inputs land on the card byte-equal
# ---------------------------------------------------------------------------


def test_favorable_records_preserved_in_input_order() -> None:
    """The function does not re-sort ``favorable`` — input order wins.

    A future Sub-AC may hand-curate the bullet list. A silent re-sort
    here would corrupt that curation. The fixture below deliberately
    feeds a non-monotone order to prove the function passes through
    whatever the caller gave it.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    # Non-monotone on purpose. If the function silently re-sorted, the
    # order on the card would be 0.9, 0.7, 0.3 (descending).
    favorable = [
        _record("opp-A", 0.7),
        _record("opp-B", 0.9),
        _record("opp-C", 0.3),
    ]
    card = build_rationale_card(team, favorable, [], 0.5)

    assert [r.win_rate for r in card.favorable] == [0.7, 0.9, 0.3]
    # The species tuples on the records also survive verbatim.
    assert [r.opponent.species for r in card.favorable] == [
        ("opp-A-1", "opp-A-2", "opp-A-3"),
        ("opp-B-1", "opp-B-2", "opp-B-3"),
        ("opp-C-1", "opp-C-2", "opp-C-3"),
    ]


def test_unfavorable_records_preserved_in_input_order() -> None:
    """The function does not re-sort ``unfavorable`` — input order wins.

    Mirror of the favorable-order test. Same rationale: a hand-curated
    bullet list must survive the assembly step.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    unfavorable = [
        _record("opp-A", 0.4),
        _record("opp-B", 0.05),
        _record("opp-C", 0.25),
    ]
    card = build_rationale_card(team, [], unfavorable, 0.0)

    assert [r.win_rate for r in card.unfavorable] == [0.4, 0.05, 0.25]


def test_usage_pct_metadata_passes_through() -> None:
    """``MetaMatchupResult.usage_pct`` is preserved on the card records.

    A future renderer may print "opp X (usage 12.5 %)" alongside each
    bullet. The assembly step must not strip that metadata.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    favorable = [_record("opp-A", 0.8, usage_pct=25.0)]
    unfavorable = [_record("opp-Z", 0.1, usage_pct=None)]
    card = build_rationale_card(team, favorable, unfavorable, 0.5)

    assert card.favorable[0].usage_pct == 25.0
    assert card.unfavorable[0].usage_pct is None


def test_empty_bullet_lists_are_allowed() -> None:
    """Both bullet lists may legitimately be empty.

    An empty meta or a team with no clear strengths/weaknesses can
    produce empty selections upstream. The card must still assemble.
    The headline AC test exercises the non-empty path; this test is
    the boundary case.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    card = build_rationale_card(team, [], [], 0.0)

    assert card.favorable == ()
    assert card.unfavorable == ()
    assert card.coverage == 0.0


def test_coverage_boundary_values_are_accepted() -> None:
    """``coverage`` may legitimately be exactly 0.0 or 1.0.

    A team that handles nothing in the meta scores 0.0; a team that
    covers the entire realized meta scores 1.0. Both are valid.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    assert build_rationale_card(team, [], [], 0.0).coverage == 0.0
    assert build_rationale_card(team, [], [], 1.0).coverage == 1.0


def test_int_coverage_zero_and_one_are_accepted() -> None:
    """``coverage`` semantically wants a float but ``0`` / ``1`` are
    legitimate Python literals callers may pass. The dataclass stores
    the value verbatim; renderers can ``float(card.coverage)`` if they
    need a strict type.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    card_lo = build_rationale_card(team, [], [], 0)
    card_hi = build_rationale_card(team, [], [], 1)
    assert card_lo.coverage == 0
    assert card_hi.coverage == 1


# ---------------------------------------------------------------------------
# Iterable consumption — generators work; each iterable consumed once
# ---------------------------------------------------------------------------


def test_accepts_generator_inputs() -> None:
    """The bullet-list parameters are documented as Iterables.

    Generators must work — they get consumed exactly once internally
    and materialized to a tuple on the card.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    fav_gen = (_record(f"f-{i}", r) for i, r in enumerate([0.9, 0.7]))
    unfav_gen = (_record(f"u-{i}", r) for i, r in enumerate([0.1, 0.3]))

    card = build_rationale_card(team, fav_gen, unfav_gen, 0.5)
    assert [r.win_rate for r in card.favorable] == [0.9, 0.7]
    assert [r.win_rate for r in card.unfavorable] == [0.1, 0.3]


def test_input_iterables_materialized_to_tuples() -> None:
    """The bullet lists on the card are tuples regardless of input type.

    Important for the structural-immutability contract — a downstream
    renderer that holds a card cannot mutate its bullet ordering.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    favorable_list = [_record("a", 0.9)]
    unfavorable_list = [_record("b", 0.1)]
    card = build_rationale_card(team, favorable_list, unfavorable_list, 0.5)
    assert isinstance(card.favorable, tuple)
    assert isinstance(card.unfavorable, tuple)


# ---------------------------------------------------------------------------
# Immutability — the returned card is frozen
# ---------------------------------------------------------------------------


def test_card_is_frozen() -> None:
    """``RationaleCard`` is a frozen dataclass — assigning a field raises.

    Renderers downstream may pass cards through layers that should not
    be able to mutate them. Freezing the dataclass is the architectural
    guarantee that no such layer can.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    card = build_rationale_card(team, [], [], 0.5)
    with pytest.raises((AttributeError, TypeError)):
        card.coverage = 0.9  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        card.favorable = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Purity — the function does not mutate its inputs
# ---------------------------------------------------------------------------


def test_input_lists_not_mutated() -> None:
    """Caller-supplied lists survive the call unchanged.

    A common rationale-card flow calls ``select_favorable_matchups`` and
    ``select_unfavorable_matchups`` to produce the bullet lists, then
    feeds them into ``build_rationale_card`` while keeping the originals
    around for logging / debugging. The original lists must not be
    mutated by the assembly step.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    favorable = [_record("opp-A", 0.9), _record("opp-B", 0.7)]
    unfavorable = [_record("opp-X", 0.1), _record("opp-Y", 0.3)]
    favorable_before = list(favorable)
    unfavorable_before = list(unfavorable)

    _ = build_rationale_card(team, favorable, unfavorable, 0.5)

    assert favorable == favorable_before, "favorable list mutated"
    assert unfavorable == unfavorable_before, "unfavorable list mutated"


# ---------------------------------------------------------------------------
# Validation — wrong types / out-of-range values rejected loudly
# ---------------------------------------------------------------------------


def test_non_candidate_team_rejected() -> None:
    """``team`` must be a ``CandidateTeam`` — strings / dicts / None rejected."""

    with pytest.raises(TypeError, match="CandidateTeam"):
        build_rationale_card("not-a-team", [], [], 0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CandidateTeam"):
        build_rationale_card(None, [], [], 0.5)  # type: ignore[arg-type]


def test_non_record_in_favorable_rejected() -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    with pytest.raises(TypeError, match="MetaMatchupResult"):
        build_rationale_card(team, ["not-a-record"], [], 0.5)  # type: ignore[list-item]


def test_non_record_in_unfavorable_rejected() -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    with pytest.raises(TypeError, match="MetaMatchupResult"):
        build_rationale_card(team, [], ["not-a-record"], 0.5)  # type: ignore[list-item]


@pytest.mark.parametrize("bad_coverage", [-0.01, 1.01, -1.0, 2.0, 99.9])
def test_coverage_out_of_range_rejected(bad_coverage: float) -> None:
    team = _candidate_team("y-1", "y-2", "y-3")
    with pytest.raises(ValueError, match="coverage out of range"):
        build_rationale_card(team, [], [], bad_coverage)


def test_coverage_nan_rejected() -> None:
    """NaN ``coverage`` is rejected explicitly.

    All NaN comparisons return False, so ``0 <= nan <= 1`` would
    silently slip past a naive range check on some Python builds.
    The explicit NaN check on ``RationaleCard.__post_init__`` catches
    it. Matches the score-axis ``_validate_axis`` convention.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    with pytest.raises(ValueError, match="NaN"):
        build_rationale_card(team, [], [], float("nan"))


def test_coverage_non_number_rejected() -> None:
    """``coverage`` must be a real number — strings / None / lists rejected."""

    team = _candidate_team("y-1", "y-2", "y-3")
    with pytest.raises(TypeError, match="real number"):
        build_rationale_card(team, [], [], "0.5")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        build_rationale_card(team, [], [], None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        build_rationale_card(team, [], [], [0.5])  # type: ignore[arg-type]


def test_coverage_bool_rejected() -> None:
    """``bool`` is an ``int`` subclass in Python; reject it explicitly.

    A caller writing ``build_rationale_card(team, [], [], True)`` would
    otherwise silently slip in ``coverage = 1`` — an entirely different
    semantic meaning than "covered everything". Mirror of the bool-as-
    int rejection in the sibling selectors.
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    with pytest.raises(TypeError, match="real number"):
        build_rationale_card(team, [], [], True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        build_rationale_card(team, [], [], False)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end wiring — composes with the sibling Sub-AC 3.1 / 3.2 / 3.3 funcs
# ---------------------------------------------------------------------------


def test_composes_with_sibling_selectors_and_coverage() -> None:
    """The card assembles the *outputs* of the three sibling sub-AC funcs.

    The seed's rationale-card pipeline is:

        select_favorable_matchups   ─┐
        select_unfavorable_matchups ─┼─→ build_rationale_card → RationaleCard
        compute_meta_coverage       ─┘

    This test wires the three callers end-to-end so any contract drift
    between them shows up as a single failing test rather than three
    siblings disagreeing in subtle ways.

    Setup (4 opponents):
      * opp-a (40 %, win 0.80) — covered, favorable
      * opp-b (30 %, win 0.20) — uncovered, unfavorable
      * opp-c (20 %, win 0.60) — covered, favorable
      * opp-d (10 %, win 0.40) — uncovered, unfavorable

    Expected:
      * favorable (top-2, descending): opp-a (0.80), opp-c (0.60)
      * unfavorable (bottom-2, ascending): opp-b (0.20), opp-d (0.40)
      * coverage: (40 + 20) / 100 = 0.60
    """

    team = _candidate_team("y-1", "y-2", "y-3")
    meta = _make_meta(
        ("opp-a", 40.0),
        ("opp-b", 30.0),
        ("opp-c", 20.0),
        ("opp-d", 10.0),
    )
    matchup_results = [
        _record("opp-a", 0.80),
        _record("opp-b", 0.20),
        _record("opp-c", 0.60),
        _record("opp-d", 0.40),
    ]

    favorable = select_favorable_matchups(team, matchup_results, n=2)
    unfavorable = select_unfavorable_matchups(team, matchup_results, n=2)
    coverage = compute_meta_coverage(team, meta, matchup_results)

    card = build_rationale_card(team, favorable, unfavorable, coverage)

    # Favorable: opp-a, opp-c (descending win_rate).
    assert [r.win_rate for r in card.favorable] == [0.80, 0.60]
    assert [r.opponent.species[0] for r in card.favorable] == ["opp-a-1", "opp-c-1"]
    # Unfavorable: opp-b, opp-d (ascending win_rate).
    assert [r.win_rate for r in card.unfavorable] == [0.20, 0.40]
    assert [r.opponent.species[0] for r in card.unfavorable] == ["opp-b-1", "opp-d-1"]
    # Coverage: 60 / 100 = 0.60.
    assert card.coverage == pytest.approx(0.60)
    # Team identity preserved.
    assert card.team is team
