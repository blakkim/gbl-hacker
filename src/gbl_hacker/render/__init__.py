"""Human-readable renderers for ``MetaSnapshot`` and friends.

Every rendering of a meta_snapshot **must** surface the Taiman Party
report-density caveat (the ``data_honesty`` evaluation principle in
``seed.yaml``). This package is the single, canonical entry point for
that obligation: any code path that wants to display a snapshot to a
human goes through :func:`render_meta_snapshot`, which structurally
guarantees the caveat is emitted.

Why a dedicated package
-----------------------

Sub-AC 6 — "The data-quality caveat about Taiman Party report density is
visible in every meta snapshot output (not hidden behind a flag)" — is
an architectural constraint, not just a documentation one. The renderer
is designed so that hiding the caveat is *not expressible*:

* The function signature has **no** ``include_caveat`` / ``quiet`` /
  ``no_caveat`` parameter. There is nothing to set to ``False``.
* The caveat is emitted in a dedicated, prominent ``DATA HONESTY``
  block at the top of every rendering. It is also re-emitted in the
  footer summary line, so a reader scrolling past the header still
  sees it.
* If a snapshot somehow reaches the renderer with an empty
  ``source_caveat``, the renderer raises ``MissingCaveatError``
  rather than silently dropping the warning. (The
  :class:`~gbl_hacker.parse.taiman.MetaSnapshot` dataclass already
  rejects empty caveats in ``__post_init__``; this is the
  defence-in-depth check at the output layer.)

These constraints make a fence around the violation pattern Sub-AC 6
forbids: there is no flag, parameter, or environment override that can
suppress the caveat for any caller of this module.

Sub-AC 3.5 (this entry):
    :mod:`gbl_hacker.render.rationale` adds the
    :func:`~gbl_hacker.render.rationale.render_rationale_cards` driver
    that renders the engine's final recommendation table with one
    rationale card attached per recommended team. Same single-source-of
    -truth posture as the snapshot renderer: there is no flag that
    suppresses the rationale, so the ``interpretability`` evaluation
    principle is enforced structurally.
"""

from gbl_hacker.render.rationale import (
    CARD_HEADER_PREFIX,
    EMPTY_RECOMMENDATIONS_NOTICE,
    MismatchedRationaleError,
    format_rationale_card,
    render_rationale_cards,
)
from gbl_hacker.render.snapshot import (
    CAVEAT_HEADER_LABEL,
    MissingCaveatError,
    format_caveat_block,
    format_snapshot_header,
    render_meta_snapshot,
)

__all__ = [
    "CARD_HEADER_PREFIX",
    "CAVEAT_HEADER_LABEL",
    "EMPTY_RECOMMENDATIONS_NOTICE",
    "MismatchedRationaleError",
    "MissingCaveatError",
    "format_caveat_block",
    "format_rationale_card",
    "format_snapshot_header",
    "render_meta_snapshot",
    "render_rationale_cards",
]
