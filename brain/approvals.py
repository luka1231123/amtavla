"""Approval coordination: turn a human yes/no into (at most one) execution.

The ActionRunner writes a pending `approvals` row for any T2 action instead of
running it. This coordinator is the other half: it records the decision and, on
approval, executes the action exactly once (re-invoking the runner with
approved=True), then delivers the result. Denials and double-clicks never run
anything — a settled approval never flips.
"""

from __future__ import annotations

from typing import Any

from brain.contracts import Action, ActionType


class ApprovalCoordinator:
    def __init__(self, catalog: Any, action_runner: Any) -> None:
        self.catalog = catalog
        self.action_runner = action_runner

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.catalog.list_approvals(state="pending", limit=limit)

    def resolve(self, approval_id: int, approved: bool) -> dict[str, Any]:
        """Apply a decision. Returns {executed, state, result?}.

        Idempotent: only a still-pending approval is acted on. A denied or
        already-executed approval yields executed=False and runs nothing.
        """
        approval = self.catalog.decide_approval(approval_id, approved)

        if approval["state"] != "approved":
            # Denied, or already settled by an earlier call.
            self.catalog.record_action_audit(
                action_type=approval["action_type"],
                tier="T2",
                detail=str(approval.get("payload", {}).get("detail", "")),
                ok=True,
                approval_id=approval_id,
                outcome=f"decision:{approval['state']}",
                turn_id=approval.get("turn_id", ""),
            )
            return {"executed": False, "state": approval["state"], "summary": approval["summary"]}

        action_type = ActionType.parse(approval["action_type"])
        if action_type is None:
            self.catalog.mark_approval_executed(approval_id, ok=False, result={"error": "unknown action"})
            return {"executed": False, "state": "failed", "summary": approval["summary"]}

        detail = str(approval.get("payload", {}).get("detail", ""))
        action = Action(action_type=action_type, detail=detail)
        result = self.action_runner.run(
            action,
            user_input=detail,
            approved=True,
            turn_id=approval.get("turn_id", ""),
            session_id=approval.get("session_id", ""),
        )
        self.catalog.mark_approval_executed(
            approval_id, ok=result.ok, result=result.to_dict()
        )
        self.catalog.record_action_audit(
            action_type=approval["action_type"],
            tier="T2",
            detail=detail,
            ok=result.ok,
            approval_id=approval_id,
            outcome="executed" if result.ok else "execution_failed",
            turn_id=approval.get("turn_id", ""),
        )
        return {
            "executed": True,
            "state": "executed" if result.ok else "failed",
            "summary": approval["summary"],
            "result": result,
        }
