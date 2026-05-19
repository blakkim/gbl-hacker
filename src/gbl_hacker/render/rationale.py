"""Canonical renderer for recommended teams + per-team rationale cards.

Sub-AC 3.5 contract — the engine's final operator-facing output
attaches **one rationale card per recommended team** to the
recommendation table and renders both together. The rationale card is
the seed ontology's :class:`~gbl_hacker.score.RationaleCard` and is the
``interpretability`` evaluation principle made concrete:

    "Each recommendation includes a rationale card that persuades a
     top-rank player (does not read as a black-box ranking)."

This module owns the *display* side of that contract. The structural
guarantees this module enforces:

1. **Every recommended team gets a card.** The two input lists must
   align 1:1 — fewer or more cards than teams raises
   :class:`MismatchedRationaleError`. A silent zip-truncate would let
   the engine ship a recommendation with no rationale, which is exactly
   the failure mode AC 3.5 rejects.

2. **Card-to-team identity is verified.** Each
   :attr:`RationaleCard.team` is asserted to share the same species
   tuple as the corresponding :attr:`ScoredTeam.team`. A drift between
   the ranker's order and the rationale layer's order would silently
   attach the wrong card to the wrong team — equally bad for trust.

3. **Output is plain text, no ANSI / no colour.** Keeps the rendering
   friendly to:

   * redirection into a file (``gblh recommend > teams.txt``);
   * grepping (``grep '^team' teams.txt``);
   * future copy-paste into a markdown report.

4. **No suppression parameter.** The signature accepts the two inputs
   and an output stream — there is no ``skip_card=``, ``compact=False``
   that hides bullets, or environment override. Every recommended team
   shipped to the operator carries its card.

The renderer trusts the upstream pipeline for the *contents* of the
card (favorable / unfavorable lists in the documented sort order,
coverage in ``[0, 1]``); it asserts only the structural alignment
between the two lists. The rationale layer's own validators (in
:mod:`gbl_hacker.score.rationale`) keep the per-card invariants honest.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TextIO

from gbl_hacker.score import RationaleCard, ScoredTeam


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_BANNER_WIDTH: int = 72
"""Width of the ASCII frames around team headers.

Matches the snapshot renderer's :data:`_BANNER_WIDTH` so the two output
surfaces look uniform when an operator prints them back-to-back."""

_HORIZONTAL_RULE: str = "=" * _BANNER_WIDTH
"""Heavy rule between team blocks — easy to spot when scrolling."""

_SOFT_RULE: str = "-" * _BANNER_WIDTH
"""Light rule between subsections within a single team block."""

CARD_HEADER_PREFIX: str = "TEAM"
"""Banner-text prefix for each team's heading.

Pinned as a module-level constant so tests can grep the output for the
predictable header marker — and so a downstream tool (e.g. a markdown
post-processor) can split the rendering on it.
"""

EMPTY_RECOMMENDATIONS_NOTICE: str = (
    "(no recommended teams — nothing to render)"
)
"""One-line notice emitted when the input is empty.

A legitimate "rank but return nothing" pipeline (``k=0`` to
:func:`rank_top_k`) routes through this surface and must not silently
emit a blank stream — the operator should see "I asked, the answer was
none" rather than wonder if the command died."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MismatchedRationaleError(ValueError):
    """Raised when ``recommended_teams`` and ``rationale_cards`` don't align.

    Two failure modes:

    * **Length mismatch** — fewer or more cards than teams. A silent
      zip-truncate here would let the engine ship a recommendation with
      no attached rationale (Sub-AC 3.5's named failure mode).
    * **Identity mismatch** — the card at position ``i`` describes a
      different team than the recommended team at position ``i``. Lets a
      ranker-vs-rationale ordering bug land loudly rather than silently
      attaching the wrong card to the wrong team.

    Inherits from :class:`ValueError` because the inputs are
    structurally invalid (the contract is "1:1 alignment"); callers
    that ``except ValueError`` stay compatible.
    """


# ---------------------------------------------------------------------------
# Per-card formatters
# ---------------------------------------------------------------------------


def _format_team_header(rank: int, scored: ScoredTeam) -> str:
    """Frame the team's banner: ``=== TEAM #1: lead / safe_swap / closer ===``.

    The banner is the predictable anchor downstream tooling greps for —
    see :data:`CARD_HEADER_PREFIX`.
    """

    species = " / ".join(scored.team.species)
    return (
        f"{_HORIZONTAL_RULE}\n"
        f"{CARD_HEADER_PREFIX} #{rank}: {species}\n"
        f"{_HORIZONTAL_RULE}"
    )


def _format_score_line(scored: ScoredTeam) -> str:
    """Render the team's three score axes on one line.

    Each axis is printed as a percentage with one decimal place — the
    operator-facing precision used elsewhere in the engine's output.
    """

    s = scored.score
    return (
        f"  expected_win_rate:     {s.expected_win_rate * 100:>5.1f}%\n"
        f"  worst_case_robustness: {s.worst_case_robustness * 100:>5.1f}%\n"
        f"  meta_coverage:         {s.meta_coverage * 100:>5.1f}%"
    )


def _format_favorable_block(card: RationaleCard) -> str:
    """Render the rationale card's "key favorable matchups" subsection."""

    if not card.favorable:
        return "  favorable matchups: (none selected)"
    lines = ["  favorable matchups:"]
    for rec in card.favorable:
        opp = " / ".join(rec.opponent.species)
        usage = (
            f" (usage {rec.usage_pct:.1f}%)"
            if rec.usage_pct is not None
            else ""
        )
        lines.append(
            f"    + vs {opp}: win {rec.win_rate * 100:.1f}%{usage}"
        )
    return "\n".join(lines)


def _format_unfavorable_block(card: RationaleCard) -> str:
    """Render the rationale card's "key unfavorable matchups" subsection."""

    if not card.unfavorable:
        return "  unfavorable matchups: (none selected)"
    lines = ["  unfavorable matchups:"]
    for rec in card.unfavorable:
        opp = " / ".join(rec.opponent.species)
        usage = (
            f" (usage {rec.usage_pct:.1f}%)"
            if rec.usage_pct is not None
            else ""
        )
        lines.append(
            f"    - vs {opp}: win {rec.win_rate * 100:.1f}%{usage}"
        )
    return "\n".join(lines)


def _format_coverage_line(card: RationaleCard) -> str:
    """Render the rationale card's meta-coverage scalar."""

    return f"  meta_coverage_slice:  {card.coverage * 100:.1f}%"


def format_rationale_card(rank: int, scored: ScoredTeam, card: RationaleCard) -> str:
    """Render one (team, card) pair as a full multi-line block.

    Layout::

        ===========================================================
        TEAM #<rank>: <lead> / <safe_swap> / <closer>
        ===========================================================
          expected_win_rate:     XX.X%
          worst_case_robustness: XX.X%
          meta_coverage:         XX.X%
        -----------------------------------------------------------
          favorable matchups:
            + vs <opp>: win XX.X% (usage Y.Y%)
            …
          unfavorable matchups:
            - vs <opp>: win XX.X% (usage Y.Y%)
            …
          meta_coverage_slice:  XX.X%

    Exposed as a public helper so tests (and future downstream
    callers — e.g. a JSON-to-markdown converter) can format a single
    card without going through the multi-team driver.

    Parameters
    ----------
    rank:
        1-based rank of the team. Printed verbatim into the banner; not
        re-derived from ``scored``.
    scored:
        The :class:`~gbl_hacker.score.ScoredTeam` carrying the team and
        its three-axis score.
    card:
        The :class:`~gbl_hacker.score.RationaleCard` attached to
        ``scored.team``. Identity alignment is the caller's contract —
        ``format_rationale_card`` itself does not re-verify, so a
        renderer driver (e.g. :func:`render_rationale_cards`) is the
        right enforcement layer.
    """

    return "\n".join(
        (
            _format_team_header(rank, scored),
            _format_score_line(scored),
            _SOFT_RULE,
            _format_favorable_block(card),
            _format_unfavorable_block(card),
            _format_coverage_line(card),
        )
    )


# ---------------------------------------------------------------------------
# Multi-team driver — the Sub-AC 3.5 headline function
# ---------------------------------------------------------------------------


def render_rationale_cards(
    recommended_teams: Iterable[ScoredTeam],
    rationale_cards: Iterable[RationaleCard],
    *,
    stream: TextIO,
) -> None:
    """Render the recommended-team table with one rationale card per team.

    This is the Sub-AC 3.5 headline function. It is the **single
    sanctioned path** for displaying the engine's final
    recommendation output — every recommended team gets exactly one
    attached rationale card in the rendered table, and that attachment
    is structurally guaranteed by the validators below.

    Output layout (top to bottom):

    1. For each ``(scored_team, rationale_card)`` pair (in input order):
       a banner-framed "TEAM #N: ..." block followed by the team's
       score axes, then the card's favorable / unfavorable / coverage
       subsections (see :func:`format_rationale_card`).
    2. Trailing blank line between blocks for readability.

    Validation
    ----------

    Before any output is emitted:

    * Both iterables are materialized exactly once into lists.
    * Length parity is enforced — mismatched lengths raise
      :class:`MismatchedRationaleError`.
    * Per-position identity alignment is enforced — each
      ``rationale_cards[i].team.species`` must equal
      ``recommended_teams[i].team.species``. A drift here would silently
      attach the wrong card to the wrong team, which is the exact
      failure mode AC 3.5 forbids.

    The structural failure paths raise *before* any partial output is
    written — the operator never sees a half-rendered table that they
    might misread.

    Parameters
    ----------
    recommended_teams:
        Iterable of :class:`~gbl_hacker.score.ScoredTeam` instances —
        typically the return value of
        :func:`~gbl_hacker.score.rank.rank_top_k`. Consumed exactly
        once; works on generators.
    rationale_cards:
        Iterable of :class:`~gbl_hacker.score.RationaleCard` instances
        — one per recommended team, in the same order. Consumed
        exactly once; works on generators.
    stream:
        Output text stream (e.g. ``sys.stdout`` or a ``io.StringIO``).

    Raises
    ------
    MismatchedRationaleError
        If the two iterables have different lengths, or if any pair's
        team-species tuples disagree.
    TypeError
        If any element is not of the expected type.

    Notes
    -----
    The function is **pure** modulo the side-effecting write to
    ``stream``: it never mutates its inputs.

    Why no suppression parameter?
        The seed pins ``interpretability`` as a top-level evaluation
        principle. A flag that hides bullets would silently degrade the
        rationale card into a label-only score row — exactly the
        "black-box ranking" the principle rejects. Keeping the signature
        flag-free makes the suppression failure mode *not expressible*.

    Why is identity verified by species tuple, not object identity?
        The rationale layer may legitimately re-build the
        :class:`~gbl_hacker.score.CandidateTeam` (e.g. a JSON
        round-trip), so ``is`` comparison would over-reject. Species
        tuple is the seed-ontology-canonical identifier of a team
        composition and is what the rest of the engine joins on.
    """

    teams = list(recommended_teams)
    cards = list(rationale_cards)

    # Defense-in-depth type checks. The dataclasses' own __post_init__
    # already validates them at construction time, but a hand-built
    # mock that bypasses the constructor would silently slip through.
    for idx, st in enumerate(teams):
        if not isinstance(st, ScoredTeam):
            raise TypeError(
                f"recommended_teams[{idx}] is not a ScoredTeam: "
                f"got {type(st).__name__}"
            )
    for idx, card in enumerate(cards):
        if not isinstance(card, RationaleCard):
            raise TypeError(
                f"rationale_cards[{idx}] is not a RationaleCard: "
                f"got {type(card).__name__}"
            )

    if len(teams) != len(cards):
        raise MismatchedRationaleError(
            f"recommended_teams has {len(teams)} entries but "
            f"rationale_cards has {len(cards)} — must be 1:1. "
            "Every recommended team must have a rationale card "
            "(Sub-AC 3.5 interpretability invariant)."
        )

    # Identity alignment. Compare on species tuple — the canonical team
    # identifier elsewhere in the engine — instead of object identity so
    # that JSON-round-tripped cards still match.
    for idx, (scored, card) in enumerate(zip(teams, cards, strict=True)):
        if scored.team.species != card.team.species:
            raise MismatchedRationaleError(
                f"rationale_cards[{idx}].team.species "
                f"({card.team.species}) does not match "
                f"recommended_teams[{idx}].team.species "
                f"({scored.team.species}). The card-to-team binding "
                "must align positionally."
            )

    if not teams:
        # Legitimate "rank but return nothing" path (k=0 from
        # rank_top_k). Emit a single-line notice instead of a blank
        # stream so the operator does not wonder if the command crashed.
        stream.write(EMPTY_RECOMMENDATIONS_NOTICE)
        stream.write("\n")
        return

    for rank, (scored, card) in enumerate(
        zip(teams, cards, strict=True), start=1
    ):
        stream.write(format_rationale_card(rank, scored, card))
        stream.write("\n\n")


__all__ = [
    "CARD_HEADER_PREFIX",
    "EMPTY_RECOMMENDATIONS_NOTICE",
    "MismatchedRationaleError",
    "format_rationale_card",
    "render_rationale_cards",
]
