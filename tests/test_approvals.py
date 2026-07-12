import json

from brain.action_runner import ActionRunner
from brain.approvals import ApprovalCoordinator
from brain.contracts import Action, ActionType
from brain.memory.catalog import MemoryCatalog
from brain.trust import action_tier
from tools.localfiles import LocalFilesWriter


def _catalog(tmp_path):
    return MemoryCatalog(str(tmp_path / "catalog.db"))


def _t2_for(action_type):
    # Treat FILE_WRITE as T2 for the test so the gate can be exercised without a
    # shipped outbound action (the roadmap's "fake send action").
    if action_type == ActionType.FILE_WRITE:
        return "T2"
    return action_tier(action_type)


def _runner(tmp_path, catalog):
    return ActionRunner(
        files_writer=LocalFilesWriter(root=tmp_path),
        approvals=catalog,
        tier_for=_t2_for,
    )


def _write_action(tmp_path):
    return Action.create(
        ActionType.FILE_WRITE, json.dumps({"path": "out.md", "content": "sent!"})
    )


def test_default_tiers_have_no_t2_yet():
    # Every shipped action is currently T0 or T1; nothing outbound has landed.
    for action_type in ActionType:
        assert action_tier(action_type) in {"T0", "T1"}


def test_t2_action_is_blocked_and_not_executed(tmp_path):
    catalog = _catalog(tmp_path)
    runner = _runner(tmp_path, catalog)

    result = runner.run(_write_action(tmp_path), user_input="do it")

    assert result.ok is True
    assert result.output["status"] == "awaiting_approval"
    # The side effect did NOT happen.
    assert not (tmp_path / "out.md").exists()
    pending = catalog.list_approvals(state="pending")
    assert len(pending) == 1
    assert pending[0]["action_type"] == "FILE_WRITE"


def test_approve_executes_exactly_once(tmp_path):
    catalog = _catalog(tmp_path)
    runner = _runner(tmp_path, catalog)
    coordinator = ApprovalCoordinator(catalog, runner)

    runner.run(_write_action(tmp_path), user_input="do it")
    approval_id = catalog.list_approvals(state="pending")[0]["id"]

    first = coordinator.resolve(approval_id, approved=True)
    assert first["executed"] is True
    assert (tmp_path / "out.md").read_text() == "sent!"

    # A second decision must not run it again.
    (tmp_path / "out.md").unlink()
    second = coordinator.resolve(approval_id, approved=True)
    assert second["executed"] is False
    assert not (tmp_path / "out.md").exists()


def test_denied_action_never_runs(tmp_path):
    catalog = _catalog(tmp_path)
    runner = _runner(tmp_path, catalog)
    coordinator = ApprovalCoordinator(catalog, runner)

    runner.run(_write_action(tmp_path), user_input="do it")
    approval_id = catalog.list_approvals(state="pending")[0]["id"]

    outcome = coordinator.resolve(approval_id, approved=False)
    assert outcome["executed"] is False
    assert outcome["state"] == "denied"
    assert not (tmp_path / "out.md").exists()

    # Denied is final: a later approve does nothing.
    again = coordinator.resolve(approval_id, approved=True)
    assert again["executed"] is False
    assert not (tmp_path / "out.md").exists()


def test_fail_closed_without_approvals_store(tmp_path):
    # A T2 action with nowhere to record the request refuses rather than runs.
    runner = ActionRunner(
        files_writer=LocalFilesWriter(root=tmp_path), tier_for=_t2_for
    )
    result = runner.run(_write_action(tmp_path), user_input="do it")
    assert result.ok is False
    assert "needs your approval" in result.error
    assert not (tmp_path / "out.md").exists()


def test_audit_row_written_on_execution(tmp_path):
    catalog = _catalog(tmp_path)
    runner = _runner(tmp_path, catalog)
    coordinator = ApprovalCoordinator(catalog, runner)
    runner.run(_write_action(tmp_path), user_input="do it")
    approval_id = catalog.list_approvals(state="pending")[0]["id"]
    coordinator.resolve(approval_id, approved=True)

    with catalog._connect() as conn:
        outcomes = [
            row["outcome"]
            for row in conn.execute(
                "SELECT outcome FROM action_audit WHERE approval_id = ? ORDER BY id",
                (approval_id,),
            ).fetchall()
        ]
    assert "awaiting_approval" in outcomes
    assert "executed" in outcomes
