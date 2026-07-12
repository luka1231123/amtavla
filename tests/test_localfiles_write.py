import json

from brain.action_runner import ActionRunner
from brain.contracts import Action, ActionType
from tools.localfiles import LocalFilesWriter


def _writer(tmp_path):
    return LocalFilesWriter(root=tmp_path)


def test_write_creates_file(tmp_path):
    writer = _writer(tmp_path)
    result = writer.write_file("notes.md", "# Todo\n- buy milk\n")
    assert result["created"] is True
    assert result["backup"] is None
    assert (tmp_path / "notes.md").read_text() == "# Todo\n- buy milk\n"


def test_overwrite_snapshots_backup(tmp_path):
    writer = _writer(tmp_path)
    writer.write_file("notes.md", "v1")
    result = writer.write_file("notes.md", "v2")
    assert result["overwrote"] is True
    assert (tmp_path / "notes.md").read_text() == "v2"
    # The previous contents are recoverable from the .bak snapshot.
    assert (tmp_path / "notes.md.bak").read_text() == "v1"


def test_path_escape_is_refused(tmp_path):
    writer = _writer(tmp_path)
    result = writer.write_file("../escape.md", "nope")
    assert "outside allowed root" in result["error"]
    assert not (tmp_path.parent / "escape.md").exists()


def test_binary_suffix_is_refused(tmp_path):
    result = _writer(tmp_path).write_file("thing.bin", "data")
    assert "Only text files" in result["error"]


def test_size_cap_enforced(tmp_path):
    result = _writer(tmp_path).write_file("big.txt", "x" * (256 * 1024 + 1))
    assert "write limit" in result["error"]
    assert not (tmp_path / "big.txt").exists()


def test_edit_replaces_and_backs_up(tmp_path):
    writer = _writer(tmp_path)
    writer.write_file("notes.md", "hello world")
    result = writer.edit_file("notes.md", "world", "there")
    assert result["replacements"] == 1
    assert (tmp_path / "notes.md").read_text() == "hello there"
    assert (tmp_path / "notes.md.bak").read_text() == "hello world"


def test_edit_missing_text_changes_nothing(tmp_path):
    writer = _writer(tmp_path)
    writer.write_file("notes.md", "hello")
    result = writer.edit_file("notes.md", "absent", "x")
    assert "not found" in result["error"]
    assert (tmp_path / "notes.md").read_text() == "hello"


def test_action_runner_file_write_json_detail(tmp_path):
    runner = ActionRunner(files_writer=_writer(tmp_path))
    action = Action.create(
        ActionType.FILE_WRITE,
        json.dumps({"path": "out.md", "content": "saved body"}),
    )
    result = runner.run(action, user_input="save that to out.md")
    assert result.ok is True
    assert (tmp_path / "out.md").read_text() == "saved body"
    assert result.sources[0].kind == "local_file"
    assert result.sources[0].metadata["tier"] == "T1"


def test_action_runner_file_write_missing_path_fails_structured(tmp_path):
    runner = ActionRunner(files_writer=_writer(tmp_path))
    action = Action.create(ActionType.FILE_WRITE, json.dumps({"content": "x"}))
    result = runner.run(action, user_input="write a file")
    assert result.ok is False
    assert "requires a target path" in result.error


def test_action_runner_file_edit(tmp_path):
    writer = _writer(tmp_path)
    writer.write_file("a.txt", "one two three")
    runner = ActionRunner(files_writer=writer)
    action = Action.create(
        ActionType.FILE_EDIT,
        json.dumps({"path": "a.txt", "find": "two", "replace": "TWO"}),
    )
    result = runner.run(action, user_input="edit a.txt")
    assert result.ok is True
    assert (tmp_path / "a.txt").read_text() == "one TWO three"
