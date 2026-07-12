"""Trust tiers: the worst-case side effect of an action decides its gate.

    T0  read / compute, no persistent effect        -> runs freely
    T1  local write to our store or the sandbox      -> runs, reversible, audited
    T2  leaves the device / irreversible / spends    -> requires approval first

The tier is a property of the action, not the feature. A new capability becomes
gated simply by listing it here (or, if omitted, it defaults to T2 — fail
closed: an unclassified action is treated as the most dangerous).
"""

from __future__ import annotations

from brain.contracts import ActionType

TRUST_TIERS: dict[ActionType, str] = {
    # T0 — read / compute
    ActionType.THINK: "T0",
    ActionType.SEARCH: "T0",
    ActionType.CALCULATE: "T0",
    ActionType.MEMORY_SEARCH: "T0",
    ActionType.SUMMARIZE: "T0",
    ActionType.NOTE_READ: "T0",
    ActionType.WEB_FETCH: "T0",
    ActionType.FILE_PARSE: "T0",
    ActionType.CLARIFY: "T0",
    # T1 — reversible local writes (own store or sandbox)
    ActionType.MEMORY_WRITE: "T1",
    ActionType.REMINDER: "T1",
    ActionType.RESEARCH: "T1",
    ActionType.FILE_WRITE: "T1",
    ActionType.FILE_EDIT: "T1",
    # T2 — outbound / irreversible: added here as M4 (messaging, calendar) and
    # M5 (shell) land. Each new entry is gated automatically, no runner change.
}


def action_tier(action_type: ActionType) -> str:
    """The trust tier for an action. Unknown actions fail closed to T2."""
    return TRUST_TIERS.get(action_type, "T2")
