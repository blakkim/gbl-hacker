"""Set-state-aware GBL matchup simulator.

This package owns the *novelty* of GBL Hacker — the set-state machine that
PvPoke's isolated-matchup simulator does not model:

* entry energy != 0 (per-Pokémon energy persists across switches)
* asymmetric shield counts across the 0/1/2 x 0/1/2 grid
* mid-matchup switches with energy carry

Public surface so far:

Sub-AC 4.1 — switch-energy carry-over:
    * :class:`SlotState`, :class:`SetState`   — frozen state snapshots
    * :func:`apply_switch`, :func:`entry_energy`

Sub-AC 4.2 — asymmetric-shield matchup resolution:
    * :class:`FastMove`, :class:`ChargedMove`, :class:`CombatantBuild`
    * :class:`CombatantState` (carries per-side shields independently)
    * :class:`MatchupResult`, :class:`ChargedEvent`
    * :func:`resolve_matchup`

Sub-AC 4.3 — mid-matchup forced switch with energy preservation:
    * :func:`apply_forced_switch` — folds residual mid-matchup HP /
      energy / shields back onto the outgoing slot while bringing the
      incoming slot on field with its own preserved energy.
"""

from gbl_hacker.simulator.matchup import (
    MAX_TURNS,
    SHIELD_BLEED_DAMAGE,
    ChargedEvent,
    ChargedMove,
    CombatantBuild,
    CombatantState,
    FastMove,
    MatchupResult,
    Side,
    resolve_matchup,
)
from gbl_hacker.simulator.state import (
    ENERGY_CAP,
    MAX_HP,
    MAX_SHIELDS,
    SetState,
    SlotState,
)
from gbl_hacker.simulator.switch import (
    SwitchError,
    apply_forced_switch,
    apply_switch,
    entry_energy,
)

__all__ = [
    "ENERGY_CAP",
    "MAX_HP",
    "MAX_SHIELDS",
    "MAX_TURNS",
    "SHIELD_BLEED_DAMAGE",
    "ChargedEvent",
    "ChargedMove",
    "CombatantBuild",
    "CombatantState",
    "FastMove",
    "MatchupResult",
    "SetState",
    "Side",
    "SlotState",
    "SwitchError",
    "apply_forced_switch",
    "apply_switch",
    "entry_energy",
    "resolve_matchup",
]
