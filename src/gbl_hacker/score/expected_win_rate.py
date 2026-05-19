"""Expected-win-rate aggregator (Sub-AC 2.1).

``expected_win_rate(team, meta)`` collapses many per-matchup simulator
outcomes into a *single* scalar in ``[0.0, 1.0]`` — the team's win-rate
expectation over the report-density-weighted Taiman Party meta. This is
the first of the three score axes the seed pins down (the other two —
``worst_case_robustness`` and ``meta_coverage`` — land in 2.2 and 2.3).

The aggregation has two layers:

1. **Within an opponent team** — the v0.1 baseline pairs every one of
   your three slots against every one of their three slots and averages
   the nine 1v1 outcomes (3 × 3 = 9). This is the *maximum-entropy* slot
   policy: when neither side's switch policy is fixed, treating every
   slot pairing as equally likely is the least-biased default. The full
   set-state simulator with a real switch policy will eventually replace
   this baseline — and to make that swap cheap, the per-set aggregator
   is exposed as a pluggable ``set_win_rate_fn`` injection point. The
   aggregator on the *outside* (meta-level weighting) does not need to
   change when the inside is upgraded.

2. **Across the meta** — each opponent team contributes proportionally
   to its Taiman Party usage share. Concretely::

       expected_win_rate(T, M) =
           Σ_k  usage_pct_k · set_win_rate(T, O_k)
           ─────────────────────────────────────────
                       Σ_k  usage_pct_k

   The denominator is **the covered slice of the meta**, not 100% — if
   ``on_missing_build="skip"`` drops some opponents we still report the
   weighted mean of what we *can* score. This is consistent with how
   ``meta_coverage`` (Sub-AC 2.3) will surface the un-covered slice
   separately rather than silently penalizing the win rate.

Why not "just sum win × usage"? Taiman Party usage shares are reported
as percentages summing to (≤) 100, not to 1. Dividing by the realized
weight makes the function robust to (a) truncated meta lists that don't
sum to 100, and (b) opponents skipped via the missing-build policy. The
result is *always* the weighted mean win rate over the scored slice —
never a probability-mass that confuses "meta coverage" with "win rate".

PvPoke-bug-avoidance contract
-----------------------------

The whole reason this function exists is to compose set-state-aware
matchup outcomes. The injected ``set_win_rate_fn`` is **expected** to
honor entry energy, asymmetric shields, and switch-energy carry — the
set-state simulator already does. This module does not re-enforce those
properties on the per-matchup side; it would be a layering violation.
What it *does* enforce is that the per-set win-rate it receives is in
``[0.0, 1.0]``: an out-of-range return value from a future set
simulator raises immediately rather than silently corrupting the score.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from gbl_hacker.parse.taiman import MetaSnapshot, TeamUsage
from gbl_hacker.simulator import (
    MAX_SHIELDS,
    CombatantBuild,
    CombatantState,
    MatchupResult,
    resolve_matchup,
)

# A callable that estimates ``team_a``'s set-level win rate (in ``[0.0,
# 1.0]``) against ``team_b``. The Sub-AC 2.1 default is
# :func:`default_set_win_rate`; later ACs may inject a richer set-state
# simulator without touching the meta-aggregation layer.
SetWinRateFn = Callable[["CandidateTeam", "CandidateTeam"], float]


@dataclass(frozen=True, slots=True)
class CandidateTeam:
    """Ordered 3-slot GBL lineup: lead, safe_swap, closer.

    Mirrors the ``candidate_team`` concept in the seed ontology. Order
    is meaningful — ``lead`` enters the matchup first, ``safe_swap`` is
    the conventional second slot, ``closer`` is the third. The score
    module does not enforce a switch policy on top of the order; it
    simply preserves the upstream-reported slot identities so that
    downstream rationale cards can refer to "your lead" / "your closer"
    coherently.

    Attributes
    ----------
    lead, safe_swap, closer:
        The three :class:`~gbl_hacker.simulator.CombatantBuild` entries,
        in slot order.
    """

    lead: CombatantBuild
    safe_swap: CombatantBuild
    closer: CombatantBuild

    @classmethod
    def from_slots(cls, slots: Iterable[CombatantBuild]) -> "CandidateTeam":
        """Construct from an iterable of exactly 3 builds (slot order preserved)."""

        materialized = tuple(slots)
        if len(materialized) != 3:
            raise ValueError(
                f"CandidateTeam needs exactly 3 builds, got {len(materialized)}"
            )
        return cls(
            lead=materialized[0],
            safe_swap=materialized[1],
            closer=materialized[2],
        )

    @property
    def slots(self) -> tuple[CombatantBuild, CombatantBuild, CombatantBuild]:
        """The three builds in slot order (lead, safe_swap, closer)."""

        return (self.lead, self.safe_swap, self.closer)

    @property
    def species(self) -> tuple[str, str, str]:
        """Species identifiers in slot order (handy for logs / rationale cards)."""

        return (self.lead.species, self.safe_swap.species, self.closer.species)


class MissingBuildError(KeyError):
    """Raised when an opponent species is absent from the build registry.

    Carries the missing species as an attribute so callers can surface
    it in operator-facing diagnostics. Inherits from :class:`KeyError`
    so ``except KeyError`` callers stay compatible.
    """

    def __init__(self, species: str) -> None:
        super().__init__(species)
        self.species = species

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"no CombatantBuild registered for species {self.species!r}"


def materialize_opponent_team(
    team_usage: TeamUsage,
    build_registry: Mapping[str, CombatantBuild],
) -> CandidateTeam:
    """Resolve a :class:`TeamUsage`'s 3 species into a :class:`CandidateTeam`.

    Looks up each member species in ``build_registry`` using a form-aware
    key (``species#<form_id>`` when ``form_id != 0``) and falls back to
    the bare ``species`` key when the form-specific entry is missing.
    The fallback path keeps legacy callers — which construct registries
    keyed only by bare species names — working unchanged.

    Order matches ``team_usage.members`` (lead, safe_swap, closer).

    Raises
    ------
    MissingBuildError
        If any of the three member species is absent from the registry
        under both the form-aware and bare keys.
    """

    from gbl_hacker.build_registry import registry_key

    builds: list[CombatantBuild] = []
    for species, form_id in zip(team_usage.members, team_usage.member_forms):
        key = registry_key(species, form_id)
        build = build_registry.get(key)
        if build is None and form_id:
            # Legacy registry keyed by bare species — fall back to base.
            build = build_registry.get(species)
        if build is None:
            raise MissingBuildError(species)
        builds.append(build)
    return CandidateTeam.from_slots(builds)


def _result_to_win_fraction(result: MatchupResult, *, tie_value: float) -> float:
    """Map a :class:`MatchupResult` to a win-fraction in ``[0.0, 1.0]``.

    GBL has no draw outcome in real play, but the simulator can return
    ``winner=None`` on a double-KO or a turn-budget cutoff. We treat
    those as ``tie_value`` (default ``0.5``) — a neutral default that
    matches Elo-style scoring conventions. The caller can override.
    """

    if result.winner == "A":
        return 1.0
    if result.winner == "B":
        return 0.0
    return tie_value


def default_set_win_rate(
    team_a: CandidateTeam,
    team_b: CandidateTeam,
    *,
    starting_shields: int = MAX_SHIELDS,
    tie_value: float = 0.5,
) -> float:
    """Estimate ``team_a``'s set-level win rate vs ``team_b`` (v0.1 baseline).

    Algorithm: average the outcome of all 9 ``(your_slot_i, their_slot_j)``
    1v1 matchups, with each side starting at full HP, 0 energy, and
    ``starting_shields`` shields. Returns the mean win-fraction in
    ``[0.0, 1.0]``.

    Why 9-pairing average?
        Until the full set-state simulator (with a real switch policy)
        lands, slot-pairing uncertainty is irreducible. The maximum-
        entropy answer — assume every slot-vs-slot pairing is equally
        likely — is the least-biased default and avoids over-fitting
        to a particular switch policy that may not generalize.

    This baseline still benefits from the set-state simulator's
    correctness (entry energy can be ≠ 0 in a future variant, shield
    counts are honored, etc.) — it is not a re-implementation of
    PvPoke's isolated-matchup model. It is the *aggregation* over slot
    pairings; the per-matchup engine remains the set-state simulator.

    Parameters
    ----------
    team_a, team_b:
        The two lineups to compare. Order matters only insofar as the
        returned value is ``team_a``'s win rate.
    starting_shields:
        Initial shield count for both sides in every pairing; defaults
        to :data:`MAX_SHIELDS` (2 shields — the GBL start-of-set value).
        Set to ``0`` to evaluate post-shield-burn scenarios.
    tie_value:
        Win-fraction contributed by a draw (double-KO / turn-budget
        cutoff). Default ``0.5``.
    """

    if not (0 <= starting_shields <= MAX_SHIELDS):
        raise ValueError(
            f"starting_shields out of range: {starting_shields} "
            f"(must be in [0, {MAX_SHIELDS}])"
        )
    if not (0.0 <= tie_value <= 1.0):
        raise ValueError(f"tie_value out of range: {tie_value}")

    total = 0.0
    count = 0
    for build_a in team_a.slots:
        for build_b in team_b.slots:
            result = resolve_matchup(
                CombatantState.fresh(build_a, shields=starting_shields),
                CombatantState.fresh(build_b, shields=starting_shields),
            )
            total += _result_to_win_fraction(result, tie_value=tie_value)
            count += 1
    # count is always 9 — kept as a divisor explicitly so the formula is
    # readable and so a future variant (e.g. 6-pairing closer-vs-lead
    # focus) does not need to re-derive the divisor.
    return total / count


def expected_win_rate(
    team: CandidateTeam,
    meta: MetaSnapshot,
    *,
    build_registry: Mapping[str, CombatantBuild],
    starting_shields: int = MAX_SHIELDS,
    tie_value: float = 0.5,
    on_missing_build: Literal["skip", "raise"] = "raise",
    set_win_rate_fn: SetWinRateFn | None = None,
) -> float:
    """Weighted-mean win rate of ``team`` against the meta team distribution.

    This is the Sub-AC 2.1 headline function. It aggregates per-matchup
    simulator outputs into a *single* scalar by:

    1. Iterating ``meta.team_usage`` (each opponent team carries a usage
       percentage).
    2. Materializing each opponent team via ``build_registry``.
    3. Calling ``set_win_rate_fn(team, opp_team)`` — defaults to
       :func:`default_set_win_rate`'s 9-pairing average.
    4. Summing ``usage_pct · set_win_rate``, divided by the total
       covered ``usage_pct`` (the realized slice of the meta).

    Boundary behavior
    -----------------

    * **Empty meta** (``meta.team_usage`` empty): returns ``0.0``. There
      is no signal to score against. Callers that want to distinguish
      "no signal" from "0% win rate" should pre-check the meta.
    * **Missing species in registry**:

      - ``on_missing_build="raise"`` (default): re-raises
        :class:`MissingBuildError`. The strictest mode — keeps the
        operator honest about what they cannot score.
      - ``on_missing_build="skip"``: drops the un-materializable
        opponent team and continues; the realized weight in the
        denominator shrinks accordingly. If *all* teams are dropped
        the function returns ``0.0``.

    * **Out-of-range win rate** from ``set_win_rate_fn``: raises
      :class:`ValueError`. Defense-in-depth against a future set
      simulator regressing.

    The Taiman Party data-honesty caveat propagates by reference (the
    ``MetaSnapshot.source_caveat`` field is preserved on every snapshot
    rendering — this function does not strip it). The score is *of* the
    upper-bracket-weighted meta, so it inherits that caveat without
    needing to re-emit it here.

    Parameters
    ----------
    team:
        The candidate team to score.
    meta:
        Parsed Taiman Party snapshot. ``meta.team_usage`` drives the
        weighted sum; ``meta.pokemon_usage`` is *not* consulted in v0.1
        (pokemon-level usage is informative for ``meta_coverage`` /
        Sub-AC 2.3, not for win-rate aggregation).
    build_registry:
        Maps species names → :class:`CombatantBuild`. Required for
        materializing opponent teams.
    starting_shields:
        Forwarded to the default set-level aggregator. Ignored when a
        custom ``set_win_rate_fn`` is supplied.
    tie_value:
        Forwarded to the default set-level aggregator. Ignored when a
        custom ``set_win_rate_fn`` is supplied.
    on_missing_build:
        Strategy when ``build_registry`` is missing an opponent's
        species. ``"raise"`` (default) or ``"skip"``.
    set_win_rate_fn:
        Optional injection. ``None`` (default) → use
        :func:`default_set_win_rate` with the keyword args above.
        Supplying a custom callable lets later ACs swap in a richer
        set-state simulator without touching the meta-aggregation logic.

    Returns
    -------
    float
        Expected win rate in ``[0.0, 1.0]``.
    """

    if on_missing_build not in ("skip", "raise"):
        raise ValueError(
            f"on_missing_build must be 'skip' or 'raise', got {on_missing_build!r}"
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

    weighted_sum = 0.0
    total_weight = 0.0
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
        weighted_sum += opp_usage.usage_pct * rate
        total_weight += opp_usage.usage_pct

    if total_weight == 0.0:
        # Either the meta had zero-weight teams (degenerate upstream),
        # or every team was skipped under "skip" policy. Either way,
        # there's no scored slice — return 0.0.
        return 0.0

    score = weighted_sum / total_weight
    # Clamp defensively. Floating-point sum of products is in principle
    # bounded by [0, 1] when every rate is bounded by [0, 1] and weights
    # are non-negative, but defense-in-depth never hurts.
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def set_driver_win_rate(
    team_a: CandidateTeam,
    team_b: CandidateTeam,
    *,
    starting_shields: int = MAX_SHIELDS,
    tie_value: float = 0.5,
    stochastic_samples: int = 1,
    rng_seed: int | None = None,
    active_switch: bool = False,
) -> float:
    """Estimate ``team_a``'s set-level win rate via the full set driver.

    Honors mid-set HP / energy / shield / buff-stage carry-over, faint-
    driven switching, and the GBL switch-timer counter (instrumented
    but no active switch in v0.2). Buff stages reset on switch — mirrors
    the in-game rule.

    Parameters
    ----------
    stochastic_samples:
        Number of independent simulations to average. ``1`` (default)
        runs a single deterministic simulation; ``> 1`` runs N
        simulations with distinct ``random.Random`` seeds and averages
        the win fractions. Used to expose the shield-decision
        stochasticity baked into the resolver.
    rng_seed:
        Base seed for the per-sample RNG. ``None`` falls back to a
        pair-specific deterministic seed (hash of species tuples) so the
        same matchup always yields the same answer across runs without
        the operator having to pin it manually.
    """

    from gbl_hacker.simulator.set_driver import simulate_set

    if stochastic_samples <= 0:
        raise ValueError(
            f"stochastic_samples must be >= 1, got {stochastic_samples}"
        )

    if stochastic_samples == 1:
        result = simulate_set(
            team_a,
            team_b,
            starting_shields=starting_shields,
            active_switch=active_switch,
        )
        if result.winner == "A":
            return 1.0
        if result.winner == "B":
            return 0.0
        return tie_value

    import random as _random
    import hashlib as _hashlib

    if rng_seed is not None:
        base_seed = rng_seed
    else:
        # Deterministic across processes: Python's built-in hash() is
        # randomized per-process (PYTHONHASHSEED), so two runs of the
        # same command would otherwise produce different stochastic
        # samples. Use a stable digest of the team tuple instead.
        key = repr((team_a.species, team_b.species)).encode("utf-8")
        base_seed = int.from_bytes(_hashlib.md5(key).digest()[:4], "big")
    total = 0.0
    for k in range(stochastic_samples):
        sample_rng = _random.Random(base_seed + k)
        result = simulate_set(
            team_a,
            team_b,
            starting_shields=starting_shields,
            rng=sample_rng,
            active_switch=active_switch,
        )
        if result.winner == "A":
            total += 1.0
        elif result.winner is None:
            total += tie_value
    return total / stochastic_samples


__all__ = [
    "CandidateTeam",
    "MissingBuildError",
    "SetWinRateFn",
    "default_set_win_rate",
    "expected_win_rate",
    "materialize_opponent_team",
    "set_driver_win_rate",
]
