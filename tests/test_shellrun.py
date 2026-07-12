from brain.action_runner import ActionRunner
from brain.approvals import ApprovalCoordinator
from brain.contracts import Action, ActionType
from brain.memory.catalog import MemoryCatalog
from brain.trust import action_tier
from tools.shellrun import ShellRunner


def _enabled_runner(**over):
    cfg = {"tools": {"shell_run": {"enabled": True, "timeout_seconds": 10, **over}}}
    return ShellRunner(config=cfg)


def test_shell_run_is_classified_t2():
    assert action_tier(ActionType.SHELL_RUN) == "T2"


def test_shell_runner_captures_stdout_and_returncode():
    result = _enabled_runner().run("echo hello world")
    assert result["ok"] is True
    assert result["returncode"] == 0
    assert "hello world" in result["stdout"]


def test_shell_runner_nonzero_exit_is_reported():
    result = _enabled_runner().run("sh -c 'exit 3'")
    assert result["ok"] is False
    assert result["returncode"] == 3


def test_shell_runner_disabled_by_default():
    result = ShellRunner(config={"tools": {}}).run("echo nope")
    assert "disabled" in result["error"]


def test_shell_runner_output_is_capped():
    result = _enabled_runner(max_output_chars=20).run("python3 -c \"print('x' * 500)\"")
    assert result["truncated"] is True
    assert "truncated" in result["stdout"]


def test_shell_runner_times_out():
    result = _enabled_runner(timeout_seconds=1).run("sleep 5")
    assert result.get("timed_out") is True
    assert "timed out" in result["error"]


def test_shell_action_is_gated_and_not_run(tmp_path):
    catalog = MemoryCatalog(str(tmp_path / "c.db"))
    runner = ActionRunner(shell_runner=_enabled_runner(), approvals=catalog)
    action = Action.create(ActionType.SHELL_RUN, "touch should_not_exist.flag")

    result = runner.run(action, user_input="run it")

    assert result.output["status"] == "awaiting_approval"
    pending = catalog.list_approvals(state="pending")
    assert len(pending) == 1
    assert pending[0]["payload"]["detail"] == "touch should_not_exist.flag"


def test_shell_action_runs_only_after_approval(tmp_path):
    catalog = MemoryCatalog(str(tmp_path / "c.db"))
    runner = ActionRunner(shell_runner=_enabled_runner(), approvals=catalog)
    coordinator = ApprovalCoordinator(catalog, runner)

    runner.run(Action.create(ActionType.SHELL_RUN, "echo approved-run"), user_input="")
    approval_id = catalog.list_approvals(state="pending")[0]["id"]

    outcome = coordinator.resolve(approval_id, approved=True)
    assert outcome["executed"] is True
    assert "approved-run" in outcome["result"].output["stdout"]

    # A denied command never runs.
    runner.run(Action.create(ActionType.SHELL_RUN, "echo nope"), user_input="")
    deny_id = catalog.list_approvals(state="pending")[0]["id"]
    denied = coordinator.resolve(deny_id, approved=False)
    assert denied["executed"] is False
