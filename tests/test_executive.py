import time

import brain.memory.service as memory_service_module
from brain.memory.commitments import extract_commitments


def _config(vector_db):
    return {
        "memory": {
            "vector_db_path": str(vector_db),
            "embedding_dim": 8,
            "vector_top_k": 3,
            "reminder_gap_seconds": 3600,
        },
        "search": {"enabled": False},
    }


def _fake_embed(text: str) -> dict:
    values = [0.0] * 8
    values[0] = 1.0
    return {"embedding": values}


def _service(tmp_path, monkeypatch):
    monkeypatch.setattr(
        memory_service_module,
        "load_brain_config",
        lambda: _config(tmp_path / "vectors.db"),
    )
    monkeypatch.setattr(memory_service_module.llama_client, "embed", _fake_embed)
    return memory_service_module.MemoryService(db_dir=str(tmp_path / "db"))


def test_extract_commitments_detects_promise_and_deadline():
    now = time.time()
    found = extract_commitments(
        "I promised Anna to send the report by tomorrow", now
    )
    assert len(found) == 1
    assert "send the report" in found[0]["content"]
    assert found[0]["due_at"] is not None
    assert found[0]["due_at"] > now

    assert extract_commitments("Did I promise to send anything?", now) == []


def test_commitment_captured_from_conversation_turn(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.commit_turn(
        "remind me to renew the passport by friday",
        "Noted.",
        trace={"turn_id": "t1", "session_id": "cli", "intent": "remember"},
    )
    loops = service.open_commitments()
    assert any("renew the passport" in item["content"] for item in loops)
    assert loops[0]["metadata"]["due_at"] is not None

    done = service.complete_commitment(loops[0]["id"])
    assert done["metadata"]["status"] == "done"
    assert not any(
        item["id"] == loops[0]["id"] for item in service.open_commitments()
    )


def test_overdue_commitment_reminder_and_staleness(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    now = time.time()
    item = service.catalog.upsert_item(
        item_type="commitment",
        content="submit the tax forms",
        metadata={"status": "open", "due_at": now - 86400},
    )

    reminder = service._due_commitment_reminder(now)
    assert "overdue" in reminder
    assert "submit the tax forms" in reminder
    # Reminder gap prevents immediate repetition.
    assert service._due_commitment_reminder(now) == ""

    marked = service._apply_staleness_rules(now)
    assert marked == 1
    assert service.catalog.inspect_item(item["id"])["metadata"]["stale"] is True


def test_focus_suppresses_non_urgent_reminders(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    now = time.time()
    service.catalog.upsert_item(
        item_type="commitment",
        content="prepare slides",
        metadata={"status": "open", "due_at": now + 3600},
    )
    service.start_focus(30)
    assert service.in_focus()
    assert service._due_commitment_reminder(now) == ""

    service.end_focus()
    assert "due soon" in service._due_commitment_reminder(now)


def test_contradiction_detected_between_conflicting_facts(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.write_memory("my car is parked at level 3 of the garage")
    service.write_memory("my car is parked at the street near gate B")

    conflicted = [
        item
        for item in service.catalog.list_items(limit=100)
        if item["metadata"].get("contradicts")
    ]
    assert len(conflicted) == 2


def test_daily_brief_lists_overdue_conflict_and_pattern(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    now = time.time()
    service.catalog.upsert_item(
        item_type="commitment",
        content="pay the electricity bill",
        metadata={"status": "open", "due_at": now - 3600},
    )
    service.write_memory("my car is parked at level 3")
    service.write_memory("my car is parked at street level")
    service.catalog.record_context_snapshot({"active_project": "Amtavla"})

    brief = service.daily_brief()
    assert "OVERDUE: pay the electricity bill" in brief
    assert "vs" in brief
    assert "Amtavla" in brief
    assert "Open loops: 1" in brief


def test_style_profile_learned_and_injected(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.commit_turn(
        "please be concise and use bullet points",
        "Understood.",
        trace={"turn_id": "t1", "session_id": "cli", "intent": "smalltalk"},
    )
    rules = service.get_style_profile()
    assert "Keep answers short and concise." in rules
    assert "Prefer bullet points for lists." in rules

    recall = service.recall_context("anything", include_web=False)
    assert "bullet points" in recall["style_context"]


def test_review_drill_picks_and_marks_items(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.catalog.upsert_item(
        item_type="fact", content="python was created by guido", review_state="confirmed"
    )
    drill = service.review_drill(limit=2)
    assert "Review Drill" in drill
    assert "guido" in drill
    reviewed = service.catalog.list_items(item_type="fact")[0]
    assert reviewed["metadata"]["last_reviewed_at"] > 0


def test_capture_note_pipeline_tags_and_logs(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    result = service.capture_note(
        "Met Anna today about project Amtavla", capture_type="voice"
    )
    assert result["event"]["capture_type"] == "voice"
    assert any(tag["name"] == "Anna" for tag in result["tags"])
    assert service.catalog.list_capture_events()[0]["id"] == result["event"]["id"]


def test_recall_respects_time_window(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.catalog.upsert_item(item_type="fact", content="dentist appointment note")
    items = service.recall_memory_items("dentist appointment")
    assert items
    # "yesterday" window excludes items created today.
    assert service.recall_memory_items("dentist appointment yesterday") == []
