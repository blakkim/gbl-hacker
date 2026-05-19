"""Recommendation-vs-reference overlap metric (Sub-AC 5.2).

The seed's Acceptance Criterion 5 requires the engine's recommendation
list to demonstrate **non-trivial overlap with at least one independent
top-tier reference** — e.g. PvPoke's published meta team list or a top
streamer's published lineup set. Sub-AC 5.1 already owns *how* the
reference list arrives in memory (see :mod:`gbl_hacker.reference.loader`).
This module owns the comparison itself.

The contract surfaces *two* complementary overlap signals because the
seed wording explicitly mentions both — "count/Jaccard of shared teams
**or** shared core Pokémon":

1. **Team-level overlap** — does the engine propose the same 3-Pokémon
   cores the reference publishes? Team identity is the **unordered
   species set** of the three slots. A reference team
   ``(Azumarill, Annihilape, Registeel)`` and an engine recommendation
   ``(Registeel, Annihilape, Azumarill)`` are the *same team* for
   overlap purposes; only the slot ordering differs, which a top-rank
   player would not consider a different "core". Using the slot-ordered
   triple would over-penalize the engine for picking a viable
   permutation of a known reference team — exactly the failure mode
   the data-honesty principle warns against.

2. **Pokémon-level overlap** — does the engine pick the same individual
   species the reference relies on? This is the *finer-grained* signal:
   even when no full team matches, the engine and the reference can
   agree on (say) 4 of 5 unique core species. Reporting this separately
   means a partial alignment is visible instead of collapsed into a
   binary "no team match → score 0".

For each signal the module emits:

* the **shared set** (raw element set — used by the rationale card to
  render "the engine agrees with PvPoke on X / Y / Z"),
* the **union set** (so a UI can render "X of Y" without re-walking
  the inputs),
* the **Jaccard coefficient** ``|A ∩ B| / |A ∪ B|`` — symmetric,
  bounded in ``[0.0, 1.0]``, 1.0 iff the two sets are identical.

Conventions
-----------

* **Canonicalization is defense-in-depth.** ``ReferenceBuild.species``
  is already canonical by the loader's contract. The engine side's
  ``CandidateTeam.species`` comes from :class:`CombatantBuild.species`
  which is *expected* to be canonical too — but a future build
  registry that accidentally stores ``"Medicham (Shadow)"`` instead of
  ``"medicham_shadow"`` would silently drop the team out of the
  intersection. Running both sides through :func:`canonical_id` here
  makes the comparison robust to that class of regression.

* **The empty-recommendation case** returns ``team_jaccard = 0.0`` and
  ``pokemon_jaccard = 0.0`` (intersection is empty, union is the
  reference set). This is the mathematically correct value, and it
  matches the operator-facing intuition: "the engine recommended
  nothing → it overlaps with nothing".

* **The empty-everything case** (both sides empty — only reachable
  via the in-memory loader, since the on-disk loader rejects empty
  reference lists) returns ``team_jaccard = 1.0`` /
  ``pokemon_jaccard = 1.0`` by the standard ``0/0 = 1.0`` Jaccard
  convention. Callers that want to distinguish "trivially identical"
  from "non-trivially identical" can inspect
  :attr:`OverlapReport.union_team_count`.

* **The function is pure.** It does not mutate either input. Callers
  can reuse the recommendation list across multiple reference lists
  (PvPoke export *and* a streamer lineup) without defensive copies.

Boundary deferred to caller
---------------------------

This module emits *measurements* — raw shared/union sets and Jaccard
coefficients. It does **not** decide what counts as "non-trivial"
overlap; that threshold belongs to the engine's CLI output layer where
the data-honesty caveat about Taiman Party feed density also lives.
Keeping the math here unopinionated means a future audit AC can
re-threshold the same numbers without re-deriving the math.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gbl_hacker.reference.loader import ReferenceTeamList, canonical_id
from gbl_hacker.score.expected_win_rate import CandidateTeam


# A 3-Pokémon team's "identity" for overlap purposes — an unordered set
# of three canonical species ids. Two teams with the same three species
# in different slot orderings share this key.
TeamKey = frozenset[str]


def _team_key(species: tuple[str, ...]) -> TeamKey:
    """Normalize a slot-ordered species triple to its unordered identity key.

    Each species id is run through :func:`canonical_id` so the resulting
    frozenset is diff-stable across input forms (``"Medicham (Shadow)"``
    and ``"medicham_shadow"`` collapse to the same key element).
    """

    return frozenset(canonical_id(s) for s in species)


@dataclass(frozen=True, slots=True)
class OverlapReport:
    """Symmetric overlap report between a rec list and a reference list.

    Surfaces two complementary overlap axes — team-level (unordered
    species triples) and Pokémon-level (individual species set) — each
    with their shared / union element sets and a Jaccard coefficient.

    Attributes
    ----------
    shared_team_keys:
        Set of team identity keys present in *both* the recommendation
        and the reference. Each key is itself a frozenset of three
        canonical species ids. Useful for the rationale card line
        "engine and PvPoke agree on these cores".
    recommendation_team_keys:
        Set of unique team identity keys derived from the
        recommendation list (deduplicated — if the recommender emits
        the same core twice it appears once here).
    reference_team_keys:
        Set of unique team identity keys derived from the reference
        list. Already deduplicated by frozenset semantics.
    shared_pokemon:
        Set of canonical species ids present in *both* sides' rosters.
    recommendation_pokemon:
        Set of canonical species ids appearing in any recommendation
        team.
    reference_pokemon:
        Set of canonical species ids appearing in any reference team.

    Notes
    -----
    The properties (``shared_team_count``, ``team_jaccard``, etc.) are
    derived on access from the underlying sets — they are cheap (set
    sizes / ratios) and exposed as properties rather than stored fields
    so that the dataclass equality (which the unit tests rely on)
    compares only the raw set fields, not their derived values. Storing
    redundant fields would create the possibility of an inconsistent
    report and obscure equality semantics.
    """

    shared_team_keys: frozenset[TeamKey]
    recommendation_team_keys: frozenset[TeamKey]
    reference_team_keys: frozenset[TeamKey]
    shared_pokemon: frozenset[str]
    recommendation_pokemon: frozenset[str]
    reference_pokemon: frozenset[str]

    # --- team-level derived metrics ----------------------------------

    @property
    def shared_team_count(self) -> int:
        """Number of shared team identities."""

        return len(self.shared_team_keys)

    @property
    def union_team_count(self) -> int:
        """Size of the union of team identities from both sides."""

        return len(self.recommendation_team_keys | self.reference_team_keys)

    @property
    def team_jaccard(self) -> float:
        """Jaccard coefficient for team-identity sets.

        ``|shared| / |union|``. Returns ``1.0`` for the both-empty case
        (standard ``0/0 = 1`` Jaccard convention — both sides are
        trivially identical). ``0.0`` when one side is empty but the
        other is not.
        """

        union = self.union_team_count
        if union == 0:
            return 1.0
        return self.shared_team_count / union

    # --- pokémon-level derived metrics --------------------------------

    @property
    def shared_pokemon_count(self) -> int:
        """Number of shared individual species."""

        return len(self.shared_pokemon)

    @property
    def union_pokemon_count(self) -> int:
        """Size of the union of species sets from both sides."""

        return len(self.recommendation_pokemon | self.reference_pokemon)

    @property
    def pokemon_jaccard(self) -> float:
        """Jaccard coefficient for the per-side species rosters.

        See :attr:`team_jaccard` for the empty-case convention; same
        rules apply.
        """

        union = self.union_pokemon_count
        if union == 0:
            return 1.0
        return self.shared_pokemon_count / union


def compute_overlap(
    recommendations: Iterable[CandidateTeam],
    reference: ReferenceTeamList,
) -> OverlapReport:
    """Compute the overlap between a recommendation list and a reference list.

    Both sides' species identifiers are run through :func:`canonical_id`
    so the comparison is robust to display-form leakage (e.g. a build
    registry that stores ``"Medicham (Shadow)"`` instead of the
    canonical ``"medicham_shadow"``). Team identity is the **unordered
    species set** of the three slots — a recommended team and a
    reference team with the same three species in different slot
    orderings share the same identity key and contribute to the
    shared set.

    Parameters
    ----------
    recommendations:
        Iterable of :class:`CandidateTeam`. Consumed exactly once;
        works on generators. May be empty.
    reference:
        Loaded :class:`ReferenceTeamList`. The disk-loader rejects
        empty team lists by contract; the in-memory loader honors the
        same validation, so ``reference.teams`` is guaranteed
        non-empty when constructed via either entry point.

    Returns
    -------
    OverlapReport
        Symmetric measurement record. The function does not mutate
        either input; the report's sets are fresh frozensets safe to
        share across threads / further pipeline stages.

    Notes
    -----
    Computational complexity is linear in the total number of slots on
    both sides (``O(|recs| + |refs|)``), bounded above by 3 ×
    (rec_count + ref_count) since every team has 3 slots. For v0.1
    scale (single-digit recommendations, low-tens reference teams)
    this is instant.
    """

    rec_team_keys: set[TeamKey] = set()
    rec_pokemon: set[str] = set()
    for team in recommendations:
        canonical_species = tuple(canonical_id(s) for s in team.species)
        rec_team_keys.add(frozenset(canonical_species))
        rec_pokemon.update(canonical_species)

    ref_team_keys: set[TeamKey] = set()
    ref_pokemon: set[str] = set()
    for ref_team in reference.teams:
        # ref species are already canonical by loader contract; the
        # canonical_id pass here is defense-in-depth against a fixture
        # that bypassed the loader (e.g. constructed directly in a
        # test) and slipped a non-canonical form through.
        canonical_species = tuple(canonical_id(s) for s in ref_team.species)
        ref_team_keys.add(frozenset(canonical_species))
        ref_pokemon.update(canonical_species)

    shared_team_keys = rec_team_keys & ref_team_keys
    shared_pokemon = rec_pokemon & ref_pokemon

    return OverlapReport(
        shared_team_keys=frozenset(shared_team_keys),
        recommendation_team_keys=frozenset(rec_team_keys),
        reference_team_keys=frozenset(ref_team_keys),
        shared_pokemon=frozenset(shared_pokemon),
        recommendation_pokemon=frozenset(rec_pokemon),
        reference_pokemon=frozenset(ref_pokemon),
    )


__all__ = [
    "OverlapReport",
    "TeamKey",
    "compute_overlap",
]
