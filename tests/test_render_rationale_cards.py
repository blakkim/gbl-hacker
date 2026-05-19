"""Unit tests for ``render_rationale_cards`` (Sub-AC 3.5).

The Sub-AC's explicit minimum deliverable:

    "Implement a ``render_rationale_cards(recommended_teams, ...)``
    CLI/output function that attaches and renders one rationale card
    per recommended team in the output table, with a unit test
    verifying every recommended team in a fixture has an attached
    card in the rendered output."

The headline test ``test_every_recommended_team_has_an_attached_card``
is the AC's named minimum. The remaining tests fence in the documented
structural contract — length parity, identity alignment, empty-input
behavior, no-suppression invariant — so the rationale layer cannot
silently regress into a black-box ranking surface.

Fixture helpers mirror ``test_build_rationale_card.py`` line for line
so cross-file fixture diffing stays a one-glance affair.
"""

from __future__ import annotations

import inspect
import io

import pytest

from gbl_hacker.render.rationale import (
    CARD_HEADER_PREFIX,
    EMPTY_RECOMMENDATIONS_NOTICE,
    MismatchedRationaleError,
    format_rationale_card,
    render_rationale_cards,
)
from gbl_hacker.score import (
    CandidateTeam,
    MetaMatchupResult,
    RationaleCard,
    Score,
    ScoredTeam,
    build_rationale_card,
)
from gbl_hacker.simulator import (
    ChargedMove,
    CombatantBuild,
    FastMove,
)


# --- fixture helpers ------------------------------------------------------
# Aligned with test_build_rationale_card.py / test_select_favorable_matchups.py
# so cross-axis fixture diffing stays a one-glance affair.


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


def _scored_team(
    prefix: str,
    *,
    ev: float = 0.6,
    wcr: float = 0.4,
    coverage: float = 0.55,
) -> ScoredTeam:
    """Build a ``ScoredTeam`` for the team at ``prefix``."""

    team = _candidate_team(f"{prefix}-lead", f"{prefix}-safe", f"{prefix}-close")
    return ScoredTeam(
        team=team,
        score=Score(
            expected_win_rate=ev,
            worst_case_robustness=wcr,
            meta_coverage=coverage,
        ),
    )


def _card_for(scored: ScoredTeam, *, coverage: float = 0.6) -> RationaleCard:
    """Build a ``RationaleCard`` whose ``team`` matches ``scored.team``."""

    return build_rationale_card(
        scored.team,
        favorable=[_record("fav-A", 0.85, usage_pct=12.0)],
        unfavorable=[_record("unfav-Z", 0.20, usage_pct=8.5)],
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Headline AC test — every recommended team has an attached card in the output
# ---------------------------------------------------------------------------


def test_every_recommended_team_has_an_attached_card() -> None:
    """The Sub-AC 3.5 named minimum deliverable.

    Build a fixture of THREE recommended teams + their THREE rationale
    cards. Render them. Assert:

      1. The render emits one TEAM block per recommended team — no
         team is silently dropped.
      2. Each team block contains its team's identifying species tuple
         (lead / safe_swap / closer) — i.e. the right team header.
      3. Each team block also contains at least one bullet drawn from
         its **own** card (favorable opponent's species and bullet
         points), proving the card was attached to the *correct* team.
         A bug that emitted only headers — or that shuffled cards —
         would fail this assertion.

    These three assertions together pin "every recommended team in a
    fixture has an attached card in the rendered output" exactly as
    AC 3.5 names it.
    """

    # Fixture: three recommended teams, each with a card that pins a
    # distinct favorable opponent so we can identity-check the
    # attachment in the rendered output.
    team_a = _scored_team("alpha", ev=0.75, wcr=0.55, coverage=0.65)
    team_b = _scored_team("bravo", ev=0.68, wcr=0.40, coverage=0.60)
    team_c = _scored_team("charlie", ev=0.62, wcr=0.50, coverage=0.55)

    card_a = build_rationale_card(
        team_a.team,
        favorable=[_record("fav-alpha", 0.92, usage_pct=14.0)],
        unfavorable=[_record("unfav-alpha", 0.18, usage_pct=9.0)],
        coverage=0.65,
    )
    card_b = build_rationale_card(
        team_b.team,
        favorable=[_record("fav-bravo", 0.88, usage_pct=11.0)],
        unfavorable=[_record("unfav-bravo", 0.22, usage_pct=7.5)],
        coverage=0.60,
    )
    card_c = build_rationale_card(
        team_c.team,
        favorable=[_record("fav-charlie", 0.81, usage_pct=10.0)],
        unfavorable=[_record("unfav-charlie", 0.25, usage_pct=6.0)],
        coverage=0.55,
    )

    recommended_teams = [team_a, team_b, team_c]
    rationale_cards = [card_a, card_b, card_c]

    buf = io.StringIO()
    render_rationale_cards(
        recommended_teams,
        rationale_cards,
        stream=buf,
    )
    output = buf.getvalue()

    # --- Assertion 1: every team header is rendered exactly once ----
    # Count the team-banner sentinel to verify no team was silently
    # dropped. We expect exactly 3 headers for the 3 recommended teams.
    header_count = output.count(CARD_HEADER_PREFIX + " #")
    assert header_count == len(recommended_teams), (
        f"expected {len(recommended_teams)} {CARD_HEADER_PREFIX!r} headers, "
        f"got {header_count}.\n--- output ---\n{output}"
    )

    # --- Assertion 2: each team's species tuple appears in the output ---
    # If a team was dropped or replaced, its species would not be in
    # the rendering at all.
    for scored in recommended_teams:
        for slot_species in scored.team.species:
            assert slot_species in output, (
                f"recommended team {scored.team.species} slot "
                f"{slot_species!r} missing from rendered output:\n{output}"
            )

    # --- Assertion 3: each team's CARD content appears in the output ---
    # If the cards were silently shuffled (wrong card attached to wrong
    # team), the favorable/unfavorable opponent species would still
    # appear in the output as a whole — but they would appear *in the
    # wrong block*. We assert per-block: split the output at team
    # headers and check each block contains its OWN card's bullets.
    blocks = _split_into_team_blocks(output)
    assert len(blocks) == len(recommended_teams), (
        f"expected {len(recommended_teams)} team blocks, got {len(blocks)}.\n"
        f"--- output ---\n{output}"
    )
    for idx, (scored, card) in enumerate(
        zip(recommended_teams, rationale_cards, strict=True)
    ):
        block = blocks[idx]
        # The block must mention this team's species (it's the header
        # team, anchored above the block).
        for slot_species in scored.team.species:
            assert slot_species in block, (
                f"block #{idx} does not contain its team's "
                f"species {slot_species!r}:\n{block}"
            )
        # And the block must contain this CARD's own favorable + unfavorable
        # opponent identifiers — proving the card was attached to the
        # right team, not a sibling team's card.
        for rec in card.favorable:
            for slot in rec.opponent.species:
                assert slot in block, (
                    f"block #{idx} missing its card's favorable "
                    f"opponent slot {slot!r}:\n{block}"
                )
        for rec in card.unfavorable:
            for slot in rec.opponent.species:
                assert slot in block, (
                    f"block #{idx} missing its card's unfavorable "
                    f"opponent slot {slot!r}:\n{block}"
                )


def _split_into_team_blocks(rendered: str) -> list[str]:
    """Split a multi-team rendering into per-team text blocks.

    The renderer emits each block as ``=== TEAM #N: ... ===`` followed
    by score/rationale lines. We split on the TEAM-header marker and
    return one substring per team (banner included).
    """

    marker = CARD_HEADER_PREFIX + " #"
    # Find indexes of every banner-line occurrence.
    blocks: list[str] = []
    cursor = 0
    while True:
        idx = rendered.find(marker, cursor)
        if idx == -1:
            break
        # Next block starts at the *previous* horizontal rule above the
        # header, but for substring comparison the header alone is
        # sufficient.
        next_idx = rendered.find(marker, idx + len(marker))
        end = next_idx if next_idx != -1 else len(rendered)
        blocks.append(rendered[idx:end])
        cursor = end
    return blocks


# ---------------------------------------------------------------------------
# Structural alignment — length parity is enforced
# ---------------------------------------------------------------------------


def test_extra_card_raises_mismatched_error() -> None:
    """More cards than teams → ``MismatchedRationaleError``.

    A silent zip-truncate would let an extra card slip in unnoticed
    (and worse, would attach the wrong cards to the wrong teams in a
    long-tail meta where the engine produces N teams but the rationale
    layer overshoots).
    """

    teams = [_scored_team("alpha"), _scored_team("bravo")]
    cards = [_card_for(teams[0]), _card_for(teams[1]), _card_for(teams[1])]
    buf = io.StringIO()
    with pytest.raises(MismatchedRationaleError, match="1:1"):
        render_rationale_cards(teams, cards, stream=buf)


def test_missing_card_raises_mismatched_error() -> None:
    """Fewer cards than teams → ``MismatchedRationaleError``.

    The named failure mode from AC 3.5: a recommended team without a
    rationale would ship as a black-box ranking, exactly what the
    ``interpretability`` evaluation principle forbids.
    """

    teams = [
        _scored_team("alpha"),
        _scored_team("bravo"),
        _scored_team("charlie"),
    ]
    cards = [_card_for(teams[0]), _card_for(teams[1])]
    buf = io.StringIO()
    with pytest.raises(MismatchedRationaleError, match="1:1"):
        render_rationale_cards(teams, cards, stream=buf)


def test_no_partial_output_on_length_mismatch() -> None:
    """Validation runs before any bytes are written.

    A half-rendered table is worse than a clean failure — the operator
    might read it as authoritative. Assert the stream is untouched
    when the validator rejects the inputs.
    """

    teams = [_scored_team("alpha"), _scored_team("bravo")]
    cards = [_card_for(teams[0])]
    buf = io.StringIO()
    with pytest.raises(MismatchedRationaleError):
        render_rationale_cards(teams, cards, stream=buf)
    assert buf.getvalue() == "", (
        f"expected no partial output before validation, got:\n{buf.getvalue()}"
    )


# ---------------------------------------------------------------------------
# Structural alignment — per-position identity is enforced
# ---------------------------------------------------------------------------


def test_misaligned_cards_raise_mismatched_error() -> None:
    """A card attached to the wrong team's position raises.

    The two teams below have distinct species tuples; if we swap the
    cards, the card's ``.team.species`` will not match the recommended
    team at the same index. The renderer must catch this.
    """

    team_a = _scored_team("alpha")
    team_b = _scored_team("bravo")
    card_a = _card_for(team_a)
    card_b = _card_for(team_b)

    buf = io.StringIO()
    # Swap the cards on purpose — card_b describes team_b but is
    # placed at the position of team_a, and vice versa.
    with pytest.raises(MismatchedRationaleError, match="positionally"):
        render_rationale_cards(
            [team_a, team_b],
            [card_b, card_a],
            stream=buf,
        )


def test_no_partial_output_on_identity_mismatch() -> None:
    """Identity-mismatch failure path also writes nothing before raising."""

    team_a = _scored_team("alpha")
    team_b = _scored_team("bravo")
    buf = io.StringIO()
    with pytest.raises(MismatchedRationaleError):
        render_rationale_cards(
            [team_a, team_b],
            [_card_for(team_b), _card_for(team_a)],
            stream=buf,
        )
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Empty-input legitimacy — ``k=0`` from rank_top_k routes here
# ---------------------------------------------------------------------------


def test_empty_inputs_emit_explicit_notice() -> None:
    """``rank_top_k(..., k=0)`` is a legitimate empty result.

    The renderer should not silently produce a blank stream — the
    operator should see "(no recommended teams …)" rather than wonder
    if the command died.
    """

    buf = io.StringIO()
    render_rationale_cards([], [], stream=buf)
    output = buf.getvalue()
    assert EMPTY_RECOMMENDATIONS_NOTICE in output, (
        f"empty-input rendering missing the explicit notice. Output:\n{output}"
    )


# ---------------------------------------------------------------------------
# Type checks — defense-in-depth against hand-rolled mocks
# ---------------------------------------------------------------------------


def test_non_scored_team_rejected() -> None:
    team = _scored_team("alpha")
    buf = io.StringIO()
    with pytest.raises(TypeError, match="ScoredTeam"):
        render_rationale_cards(
            ["not-a-scored-team"],  # type: ignore[list-item]
            [_card_for(team)],
            stream=buf,
        )


def test_non_rationale_card_rejected() -> None:
    team = _scored_team("alpha")
    buf = io.StringIO()
    with pytest.raises(TypeError, match="RationaleCard"):
        render_rationale_cards(
            [team],
            ["not-a-card"],  # type: ignore[list-item]
            stream=buf,
        )


# ---------------------------------------------------------------------------
# No-suppression invariant — interpretability principle structurally enforced
# ---------------------------------------------------------------------------


def test_no_suppression_parameter_in_signature() -> None:
    """The renderer signature must NOT expose a "hide the card" flag.

    The seed pins ``interpretability`` as a top-level evaluation
    principle. A flag that hides bullets would silently degrade the
    rationale card into a label-only score row — exactly the
    "black-box ranking" the principle rejects. This test pins the
    signature so a future refactor cannot regress AC 3.5 by quietly
    adding a ``compact=`` / ``no_cards=`` kwarg.
    """

    sig = inspect.signature(render_rationale_cards)
    forbidden = {
        "compact",
        "no_cards",
        "skip_card",
        "skip_cards",
        "quiet",
        "include_cards",
        "include_card",
        "hide_card",
        "hide_cards",
        "suppress_rationale",
    }
    for name in sig.parameters:
        assert name not in forbidden, (
            f"render_rationale_cards must not expose a suppression "
            f"parameter {name!r} — AC 3.5 interpretability invariant."
        )


# ---------------------------------------------------------------------------
# Pass-through fidelity — score axes + card contents survive the rendering
# ---------------------------------------------------------------------------


def test_score_axes_are_rendered_for_each_team() -> None:
    """Every team block prints its three score axes verbatim.

    The score axes are the headline contract of ``ScoredTeam``; they
    must accompany the rationale card so the operator can see *why* a
    team was ranked where it is.
    """

    scored = _scored_team("alpha", ev=0.71, wcr=0.43, coverage=0.59)
    card = _card_for(scored)
    buf = io.StringIO()
    render_rationale_cards([scored], [card], stream=buf)
    output = buf.getvalue()

    # Each axis is rendered as a percentage with one decimal place.
    assert "expected_win_rate" in output
    assert "71.0" in output
    assert "worst_case_robustness" in output
    assert "43.0" in output
    assert "meta_coverage" in output
    assert "59.0" in output


def test_coverage_slice_rendered_for_each_card() -> None:
    """The card's coverage scalar shows up under its team's block."""

    scored = _scored_team("alpha")
    card = build_rationale_card(
        scored.team,
        favorable=[_record("fav-A", 0.85)],
        unfavorable=[_record("unfav-A", 0.20)],
        coverage=0.42,
    )
    buf = io.StringIO()
    render_rationale_cards([scored], [card], stream=buf)
    output = buf.getvalue()
    assert "42.0" in output, (
        f"card coverage 42.0% missing from rendering:\n{output}"
    )


def test_format_rationale_card_renders_single_block() -> None:
    """``format_rationale_card`` is the per-pair helper used by the driver.

    The driver test above covers the multi-team case; this test pins
    the single-pair helper so downstream tools that want a single card
    block (e.g. an interactive REPL) have a stable surface.
    """

    scored = _scored_team("alpha", ev=0.70, wcr=0.50, coverage=0.60)
    card = build_rationale_card(
        scored.team,
        favorable=[_record("fav-A", 0.95, usage_pct=15.0)],
        unfavorable=[_record("unfav-Z", 0.10, usage_pct=20.0)],
        coverage=0.60,
    )
    block = format_rationale_card(rank=1, scored=scored, card=card)
    assert "TEAM #1" in block
    # Team species shown in the banner.
    assert "alpha-lead" in block
    # Card's bullets present.
    assert "fav-A-1" in block
    assert "unfav-Z-1" in block
    # Score axes shown.
    assert "70.0" in block


# ---------------------------------------------------------------------------
# Generator inputs — iterables, not just lists
# ---------------------------------------------------------------------------


def test_accepts_generator_inputs() -> None:
    """Both inputs are documented as ``Iterable`` — generators must work."""

    team_a = _scored_team("alpha")
    team_b = _scored_team("bravo")
    card_a = _card_for(team_a)
    card_b = _card_for(team_b)

    teams_gen = (t for t in [team_a, team_b])
    cards_gen = (c for c in [card_a, card_b])
    buf = io.StringIO()
    render_rationale_cards(teams_gen, cards_gen, stream=buf)
    output = buf.getvalue()
    # Both teams should be present.
    assert "alpha-lead" in output
    assert "bravo-lead" in output


# ---------------------------------------------------------------------------
# Empty bullet lists — handled cleanly
# ---------------------------------------------------------------------------


def test_empty_bullet_lists_render_explicit_notice() -> None:
    """A card with empty favorable/unfavorable lists still renders.

    Upstream selectors are allowed to return empty lists (e.g. on a
    tiny meta or a degenerate fixture). The renderer must surface an
    explicit "none selected" notice rather than a blank section.
    """

    scored = _scored_team("alpha")
    card = build_rationale_card(scored.team, favorable=[], unfavorable=[], coverage=0.3)
    buf = io.StringIO()
    render_rationale_cards([scored], [card], stream=buf)
    output = buf.getvalue()
    # The team block is still emitted with the explicit "(none selected)"
    # notices for the two bullet lists.
    assert "favorable matchups: (none selected)" in output
    assert "unfavorable matchups: (none selected)" in output
