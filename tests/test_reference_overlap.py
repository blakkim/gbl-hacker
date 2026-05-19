"""Unit tests for ``gbl_hacker.reference.overlap`` (Sub-AC 5.2).

The headline contract:

    ``compute_overlap(recommendations, reference)`` returns an
    :class:`OverlapReport` whose Jaccard coefficients (and underlying
    shared / union sets) reflect the *non-trivial* overlap between the
    engine's recommended teams and an independent top-tier reference
    list — both at the team-level (unordered species triples) and the
    Pokémon-level (individual species).

The Sub-AC 5.2 minimum deliverable is "a unit test on hand-crafted
lists covering **zero-overlap, partial-overlap, and full-overlap**
cases". Those three tests are the spine of this file; the rest fence
in the documented contract (slot-order invariance, canonicalization
robustness, empty-recommendation degeneracy, input-purity).

All inputs are hand-built in-memory — no fixture file is loaded —
because the three overlap cases must be unambiguous, diff-stable, and
visible at a glance in the test source.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gbl_hacker.reference import (
    GREAT_LEAGUE_LABEL,
    OverlapReport,
    ReferenceBuild,
    ReferenceBuildDisplay,
    ReferenceTeam,
    ReferenceTeamList,
    compute_overlap,
)
from gbl_hacker.score import CandidateTeam
from gbl_hacker.simulator import ChargedMove, CombatantBuild, FastMove


# ---------------------------------------------------------------------------
# fixture helpers — structurally aligned with test_select_favorable_matchups.py
# so cross-axis fixture diffing stays a one-glance affair.
# ---------------------------------------------------------------------------


def _build(species: str, *, max_hp: int = 100, fast_damage: int = 2) -> CombatantBuild:
    """Construct a minimal :class:`CombatantBuild` keyed on ``species``.

    The combat stats are irrelevant to overlap — only ``species`` is
    consulted by :func:`compute_overlap` — but they must be valid for
    the dataclass constructors to accept them.
    """

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


def _reference_build(species: str) -> ReferenceBuild:
    """Construct a canonical :class:`ReferenceBuild` for ``species``.

    Charge moves are placeholders — overlap only looks at the species
    triple. The display fields mirror the canonical fields verbatim
    (since the test passes canonical ids directly).
    """

    display = ReferenceBuildDisplay(
        species=species,
        fast_move="quick",
        charge_moves=("bomb", "bomb"),
    )
    return ReferenceBuild(
        species=species,
        fast_move="quick",
        charge_moves=("bomb", "bomb"),
        display=display,
    )


def _reference_team(
    *species_names: str, name: str = "ref team", source_label: str = "pvpoke_meta"
) -> ReferenceTeam:
    if len(species_names) != 3:
        raise AssertionError(
            f"test fixture needs 3 species, got {len(species_names)}"
        )
    a, b, c = (_reference_build(s) for s in species_names)
    return ReferenceTeam(name=name, source_label=source_label, members=(a, b, c))


def _reference_list(*teams: ReferenceTeam) -> ReferenceTeamList:
    return ReferenceTeamList(
        source="hand_crafted_v1",
        source_url="",
        league=GREAT_LEAGUE_LABEL,
        captured_at=datetime(2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc),
        notes="hand-crafted overlap test fixture, not ground-truth",
        teams=teams,
    )


# ===========================================================================
# Sub-AC 5.2 spine — zero / partial / full overlap on hand-crafted lists
# ===========================================================================


def test_zero_overlap_yields_disjoint_metrics() -> None:
    """Disjoint species sets → both Jaccards are 0.0, shared sets empty.

    Hand-crafted setup:
      * Recommendations: 2 teams over species {alpha, bravo, charlie,
        delta, echo, foxtrot} — six distinct species, zero overlap
        with reference roster.
      * Reference: 2 teams over species {umbra, vega, wraith, xeno,
        yarrow, zenith} — six distinct species.

    Expected:
      * shared_team_keys is empty
      * shared_pokemon is empty
      * team_jaccard == 0.0 (0 / 4 unique teams)
      * pokemon_jaccard == 0.0 (0 / 12 unique species)
    """

    recommendations = [
        _candidate_team("alpha", "bravo", "charlie"),
        _candidate_team("delta", "echo", "foxtrot"),
    ]
    reference = _reference_list(
        _reference_team("umbra", "vega", "wraith"),
        _reference_team("xeno", "yarrow", "zenith"),
    )

    report = compute_overlap(recommendations, reference)

    assert report.shared_team_count == 0
    assert report.shared_team_keys == frozenset()
    assert report.shared_pokemon == frozenset()
    assert report.shared_pokemon_count == 0
    assert report.team_jaccard == 0.0
    assert report.pokemon_jaccard == 0.0
    # Sanity: the union sets are the sums of the two disjoint sides.
    assert report.union_team_count == 4
    assert report.union_pokemon_count == 12


def test_partial_overlap_yields_intermediate_metrics() -> None:
    """One full team match + extra single-species crossover → 0 < Jaccards < 1.

    Hand-crafted setup:
      * Recommendations: 2 teams.
          - rec[0] = (azumarill, annihilape, registeel)       ← matches ref[0]
          - rec[1] = (medicham_shadow, lickitung_shadow, charm) ← only
            medicham_shadow / lickitung_shadow are in ref.
      * Reference: 3 teams.
          - ref[0] = (azumarill, annihilape, registeel)        ← matches rec[0]
          - ref[1] = (medicham_shadow, lickitung_shadow, azumarill)
          - ref[2] = (galarian_stunfisk, azumarill, annihilape)

    Expected:
      * shared_team_count == 1 (rec[0] / ref[0] unordered set match)
      * Team Jaccard:
          rec_teams = {azu/anni/registeel, medi/lick/charm}             (size 2)
          ref_teams = {azu/anni/registeel, medi/lick/azu, gal/azu/anni} (size 3)
          shared = {azu/anni/registeel}     (size 1)
          union  = 2 + 3 - 1 = 4
          → 1/4 = 0.25
      * shared_pokemon = {azumarill, annihilape, registeel,
          medicham_shadow, lickitung_shadow}                            (size 5)
      * Pokémon Jaccard:
          rec_pokemon = {azu, anni, registeel, medi, lick, charm}       (size 6)
          ref_pokemon = {azu, anni, registeel, medi, lick, galarian}    (size 6)
          shared      = the first 5                                     (size 5)
          union       = 6 + 6 - 5 = 7
          → 5/7
    """

    recommendations = [
        _candidate_team("azumarill", "annihilape", "registeel"),
        _candidate_team("medicham_shadow", "lickitung_shadow", "charm"),
    ]
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel"),
        _reference_team("medicham_shadow", "lickitung_shadow", "azumarill"),
        _reference_team("galarian_stunfisk", "azumarill", "annihilape"),
    )

    report = compute_overlap(recommendations, reference)

    # Team-level
    assert report.shared_team_count == 1
    assert report.shared_team_keys == frozenset(
        {frozenset({"azumarill", "annihilape", "registeel"})}
    )
    assert report.union_team_count == 4
    assert report.team_jaccard == 0.25
    assert 0.0 < report.team_jaccard < 1.0  # the "partial" qualifier

    # Pokémon-level
    expected_shared = frozenset(
        {
            "azumarill",
            "annihilape",
            "registeel",
            "medicham_shadow",
            "lickitung_shadow",
        }
    )
    assert report.shared_pokemon == expected_shared
    assert report.shared_pokemon_count == 5
    assert report.union_pokemon_count == 7
    assert report.pokemon_jaccard == 5 / 7
    assert 0.0 < report.pokemon_jaccard < 1.0


def test_full_overlap_yields_unit_jaccard() -> None:
    """Recommendation team-set ≡ reference team-set → both Jaccards == 1.0.

    Hand-crafted setup (identical multisets at the team-identity level
    — i.e. same unordered species triples on both sides):
      * Recommendations: 2 teams, in (deliberately) different slot
        orders than the reference to also exercise the
        slot-order-invariance contract.
          - rec[0] = (registeel, annihilape, azumarill)  ← shuffle of ref[0]
          - rec[1] = (lickitung_shadow, azumarill, medicham_shadow)
                                                         ← shuffle of ref[1]
      * Reference: 2 teams.
          - ref[0] = (azumarill, annihilape, registeel)
          - ref[1] = (medicham_shadow, lickitung_shadow, azumarill)

    Expected:
      * shared_team_count == 2
      * Team Jaccard == 1.0 (2/2 — both sides cover the same set of
        unordered species triples)
      * shared_pokemon covers the full 5-species roster on both sides
      * Pokémon Jaccard == 1.0 (5/5)
    """

    recommendations = [
        _candidate_team("registeel", "annihilape", "azumarill"),
        _candidate_team("lickitung_shadow", "azumarill", "medicham_shadow"),
    ]
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel"),
        _reference_team("medicham_shadow", "lickitung_shadow", "azumarill"),
    )

    report = compute_overlap(recommendations, reference)

    # Team-level
    assert report.shared_team_count == 2
    assert report.shared_team_keys == frozenset(
        {
            frozenset({"azumarill", "annihilape", "registeel"}),
            frozenset({"medicham_shadow", "lickitung_shadow", "azumarill"}),
        }
    )
    assert report.union_team_count == 2
    assert report.team_jaccard == 1.0

    # Pokémon-level
    full_roster = frozenset(
        {
            "azumarill",
            "annihilape",
            "registeel",
            "medicham_shadow",
            "lickitung_shadow",
        }
    )
    assert report.shared_pokemon == full_roster
    assert report.recommendation_pokemon == full_roster
    assert report.reference_pokemon == full_roster
    assert report.shared_pokemon_count == 5
    assert report.union_pokemon_count == 5
    assert report.pokemon_jaccard == 1.0


# ===========================================================================
# documented-contract fences
# ===========================================================================


def test_slot_order_does_not_affect_team_overlap() -> None:
    """Two teams with identical species in different slot orders share a key.

    The team-identity key is the unordered species set — a recommended
    team ``(Registeel, Annihilape, Azumarill)`` and a reference team
    ``(Azumarill, Annihilape, Registeel)`` represent the same "core" and
    must contribute to the shared set.
    """

    recommendations = [_candidate_team("registeel", "annihilape", "azumarill")]
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel")
    )

    report = compute_overlap(recommendations, reference)
    assert report.shared_team_count == 1
    assert report.team_jaccard == 1.0


def test_canonicalization_normalizes_human_display_input() -> None:
    """Display-form species names on the rec side are canonicalized in-flight.

    Defense-in-depth: a future build-registry regression that stores
    ``"Medicham (Shadow)"`` instead of canonical ``"medicham_shadow"``
    must NOT silently zero out the overlap score. Both sides are run
    through :func:`canonical_id` so the comparison stays correct.
    """

    recommendations = [
        _candidate_team("Medicham (Shadow)", "Galarian Stunfisk", "Azumarill"),
    ]
    reference = _reference_list(
        _reference_team("medicham_shadow", "galarian_stunfisk", "azumarill"),
    )

    report = compute_overlap(recommendations, reference)
    assert report.shared_team_count == 1
    assert report.team_jaccard == 1.0
    assert report.pokemon_jaccard == 1.0


def test_empty_recommendations_yield_zero_overlap_with_nonempty_reference() -> None:
    """Empty rec list × non-empty reference → both Jaccards == 0.0.

    The mathematically correct value (∅ ∩ R = ∅, ∅ ∪ R = R, 0/|R|=0)
    AND the operator-facing intuition: "engine recommended nothing →
    it overlaps with nothing".
    """

    recommendations: list[CandidateTeam] = []
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel")
    )

    report = compute_overlap(recommendations, reference)
    assert report.shared_team_count == 0
    assert report.shared_pokemon == frozenset()
    assert report.team_jaccard == 0.0
    assert report.pokemon_jaccard == 0.0
    assert report.union_team_count == 1
    assert report.union_pokemon_count == 3


def test_pokemon_jaccard_can_be_nonzero_when_team_jaccard_is_zero() -> None:
    """The two axes are independent — partial-species, zero-team agreement.

    Setup: no team-identity match, but several species in common.
      * rec[0] = (azumarill, annihilape, charm)
        rec[1] = (registeel, swampert, medicham_shadow)
      * ref[0] = (azumarill, registeel, galarian_stunfisk)
        ref[1] = (annihilape, lickitung_shadow, swampert)

      rec_team_keys (2) and ref_team_keys (2) are fully disjoint —
      none of the unordered triples match. But the per-species rosters
      share {azumarill, annihilape, registeel, swampert} → 4 species.

    Expected: team_jaccard == 0.0, pokemon_jaccard > 0.0. Together
    these two scalars communicate "the engine and reference have no
    full-team agreement but share four core picks" — exactly the
    kind of nuance that would be lost if only one of the axes were
    reported.
    """

    recommendations = [
        _candidate_team("azumarill", "annihilape", "charm"),
        _candidate_team("registeel", "swampert", "medicham_shadow"),
    ]
    reference = _reference_list(
        _reference_team("azumarill", "registeel", "galarian_stunfisk"),
        _reference_team("annihilape", "lickitung_shadow", "swampert"),
    )

    report = compute_overlap(recommendations, reference)

    assert report.shared_team_count == 0
    assert report.team_jaccard == 0.0

    assert report.shared_pokemon == frozenset(
        {"azumarill", "annihilape", "registeel", "swampert"}
    )
    assert report.shared_pokemon_count == 4
    # rec_pokemon = {azu, anni, charm, registeel, swampert, medicham_shadow} = 6
    # ref_pokemon = {azu, registeel, galarian_stunfisk, anni, lickitung_shadow, swampert} = 6
    # union = 6 + 6 - 4 = 8
    assert report.union_pokemon_count == 8
    assert report.pokemon_jaccard == 4 / 8


def test_compute_overlap_accepts_generator_inputs() -> None:
    """Generator inputs are consumed exactly once.

    The contract docstring promises generator support; this test pins
    the consume-once contract so a future regression cannot silently
    promote the input to a tuple (which would change the iteration
    semantics observable to callers).
    """

    rec_list = [
        _candidate_team("azumarill", "annihilape", "registeel"),
        _candidate_team("medicham_shadow", "lickitung_shadow", "azumarill"),
    ]
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel"),
    )

    def _gen() -> object:
        for team in rec_list:
            yield team

    report = compute_overlap(_gen(), reference)
    assert report.shared_team_count == 1
    assert report.team_jaccard == 1 / 2  # rec has 2 teams, 1 shared, ref has 1


def test_duplicate_recommendation_teams_collapse_into_set() -> None:
    """A duplicated recommendation team appears once in the team-key set.

    The team-identity carrier is a ``frozenset`` of unordered species
    triples; duplicates collapse by construction. This matters for the
    overlap Jaccard — counting a duplicated rec twice would inflate
    the rec-side denominator and depress the Jaccard, which would
    misrepresent overlap quality.
    """

    same_team = _candidate_team("azumarill", "annihilape", "registeel")
    same_team_shuffled = _candidate_team("registeel", "azumarill", "annihilape")
    recommendations = [same_team, same_team_shuffled]
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel")
    )

    report = compute_overlap(recommendations, reference)
    # The two recs share the same unordered identity — they collapse.
    assert len(report.recommendation_team_keys) == 1
    assert report.shared_team_count == 1
    assert report.union_team_count == 1
    assert report.team_jaccard == 1.0


def test_compute_overlap_does_not_mutate_inputs() -> None:
    """Pure function — both inputs are byte-identical before and after.

    The contract doc says callers can reuse the recommendation list
    across multiple reference lists. Pinning this lets a later AC add
    a "compare against both PvPoke and a streamer fixture" pipeline
    without paranoid defensive copies.
    """

    recommendations = [
        _candidate_team("azumarill", "annihilape", "registeel"),
        _candidate_team("medicham_shadow", "lickitung_shadow", "azumarill"),
    ]
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel"),
    )

    rec_snapshot = list(recommendations)
    ref_teams_snapshot = tuple(reference.teams)

    _ = compute_overlap(recommendations, reference)

    assert recommendations == rec_snapshot
    assert reference.teams == ref_teams_snapshot


def test_report_jaccard_properties_are_consistent_with_set_sizes() -> None:
    """The derived properties match a hand-computed |∩| / |∪|.

    Defensive regression test against a bug where the properties
    return stale / divergent values from the underlying sets — e.g. a
    future refactor that stores a cached scalar and forgets to keep
    it in sync. Re-computing |∩| / |∪| from the *raw* set fields here
    pins the invariant.
    """

    recommendations = [
        _candidate_team("azumarill", "annihilape", "registeel"),
        _candidate_team("swampert", "annihilape", "registeel"),
    ]
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel"),
        _reference_team("galarian_stunfisk", "azumarill", "annihilape"),
    )

    report = compute_overlap(recommendations, reference)

    # Recompute from the raw set fields.
    rec_team_keys = report.recommendation_team_keys
    ref_team_keys = report.reference_team_keys
    expected_shared_team = rec_team_keys & ref_team_keys
    expected_union_team = rec_team_keys | ref_team_keys
    assert report.shared_team_keys == expected_shared_team
    assert report.shared_team_count == len(expected_shared_team)
    assert report.union_team_count == len(expected_union_team)
    assert report.team_jaccard == (
        len(expected_shared_team) / len(expected_union_team)
    )

    rec_pokemon = report.recommendation_pokemon
    ref_pokemon = report.reference_pokemon
    expected_shared_pokemon = rec_pokemon & ref_pokemon
    expected_union_pokemon = rec_pokemon | ref_pokemon
    assert report.shared_pokemon == expected_shared_pokemon
    assert report.shared_pokemon_count == len(expected_shared_pokemon)
    assert report.union_pokemon_count == len(expected_union_pokemon)
    assert report.pokemon_jaccard == (
        len(expected_shared_pokemon) / len(expected_union_pokemon)
    )


def test_overlap_report_equality_is_field_based() -> None:
    """Two reports built from identical inputs compare equal.

    Verifies the :class:`OverlapReport` dataclass equality contract
    — important because downstream tests / CLI rendering may dedupe
    or cache reports keyed on their identity.
    """

    recommendations = [_candidate_team("azumarill", "annihilape", "registeel")]
    reference = _reference_list(
        _reference_team("azumarill", "annihilape", "registeel")
    )

    report_a = compute_overlap(recommendations, reference)
    report_b = compute_overlap(recommendations, reference)
    assert report_a == report_b
    assert isinstance(report_a, OverlapReport)
