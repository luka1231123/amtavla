import datetime

import brain.memory.service as memory_service_module
from brain.contracts import SearchResult
from brain.memory.commitments import parse_reminder


def _config(vector_db):
    return {
        "memory": {
            "vector_db_path": str(vector_db),
            "embedding_dim": 8,
            "vector_top_k": 3,
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


# A Wednesday, 10:00 local time.
_NOW = datetime.datetime(2026, 7, 8, 10, 0).timestamp()


def test_parse_reminder_relative_offset():
    parsed = parse_reminder("remind me to stretch in 20 minutes", _NOW)
    assert parsed["content"] == "stretch in 20 minutes"
    assert parsed["due_at"] == _NOW + 20 * 60


def test_parse_reminder_tomorrow_morning():
    parsed = parse_reminder("remind me to call the dentist tomorrow morning", _NOW)
    due = datetime.datetime.fromtimestamp(parsed["due_at"])
    assert (due.date(), due.hour, due.minute) == (datetime.date(2026, 7, 9), 9, 0)


def test_parse_reminder_clock_time_and_meridiem():
    parsed = parse_reminder("set a reminder for the standup at 2pm", _NOW)
    due = datetime.datetime.fromtimestamp(parsed["due_at"])
    assert (due.date(), due.hour) == (datetime.date(2026, 7, 8), 14)


def test_parse_reminder_past_time_rolls_to_next_day():
    parsed = parse_reminder("remind me to journal at 8am", _NOW)
    due = datetime.datetime.fromtimestamp(parsed["due_at"])
    assert (due.date(), due.hour) == (datetime.date(2026, 7, 9), 8)


def test_parse_reminder_without_time_cue_has_no_due_at():
    parsed = parse_reminder("remind me about the visa paperwork", _NOW)
    assert parsed["content"] == "the visa paperwork"
    assert parsed["due_at"] is None


def test_parse_reminder_rejects_non_reminder_text():
    assert parse_reminder("what's the weather tomorrow", _NOW) is None


def test_due_reminder_fires_once_into_proactive_outbox(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    item = service.create_reminder("call the dentist", due_at=_NOW - 60)

    assert item["review_state"] == "confirmed"
    assert service._fire_due_reminders(now=_NOW) == 1
    due_str = datetime.datetime.fromtimestamp(_NOW - 60).strftime("%H:%M")
    assert service.drain_proactive_messages() == [
        f"Reminder ({due_str}): call the dentist"
    ]
    # Firing is idempotent: the same reminder never fires twice.
    assert service._fire_due_reminders(now=_NOW + 3600) == 0
    assert service.drain_proactive_messages() == []


def test_future_and_completed_reminders_do_not_fire(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.create_reminder("water the plants", due_at=_NOW + 3600)
    done = service.create_reminder("pay rent", due_at=_NOW - 60)
    service.catalog.update_item_metadata(done["id"], {"status": "done"})

    assert service._fire_due_reminders(now=_NOW) == 0
    assert service.drain_proactive_messages() == []


def test_explicit_reminder_and_captured_commitment_stay_one_item(
    tmp_path, monkeypatch
):
    service = _service(tmp_path, monkeypatch)
    reminder = service.create_reminder("send the invoice", due_at=_NOW + 3600)
    captured = service._capture_commitments("I need to send the invoice", {}, _NOW)

    assert len(captured) == 1
    assert captured[0]["id"] == reminder["id"]


def test_research_job_is_claimed_once_even_when_seen_twice(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    calls = []

    def fake_execute(topic):
        calls.append(topic)
        return "summary", []

    monkeypatch.setattr(service, "_execute_research", fake_execute)
    job = service.queue_research("claim race topic")

    # Simulate a second worker having already claimed the row.
    with service._connect(service._jobs_db) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running' WHERE id = ?", (job["job_id"],)
        )
    assert service._run_research_jobs() == 0
    assert calls == []

    with service._connect(service._jobs_db) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'pending' WHERE id = ?", (job["job_id"],)
        )
    assert service._run_research_jobs() == 1
    assert calls == ["claim race topic"]


def test_research_job_synthesizes_and_reports_proactively(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    def fake_search(query, cache=None):
        return [
            SearchResult.from_row(
                {
                    "title": f"Result for {query}",
                    "url": f"https://example.test/{query.replace(' ', '-')}",
                    "snippet": "Local-first sync keeps data on device.",
                },
                query=query,
                rank=1,
            )
        ]

    monkeypatch.setattr(memory_service_module, "search_web", fake_search)
    monkeypatch.setattr(
        memory_service_module.llama_client,
        "chat",
        lambda messages, **kwargs: {
            "message": {"content": "Local-first sync engines favor CRDTs."}
        },
    )

    job = service.queue_research("local-first sync engines")
    assert job["status"] == "pending"

    assert service._run_research_jobs() == 1
    messages = service.drain_proactive_messages()
    assert len(messages) == 1
    assert messages[0].startswith("Research done — local-first sync engines:")
    assert "CRDTs" in messages[0]

    insights = [
        item
        for item in service.catalog.list_items(item_type="insight")
        if item["metadata"].get("origin") == "research"
    ]
    assert len(insights) == 1
    assert insights[0]["metadata"]["topic"] == "local-first sync engines"

    with service._connect(service._jobs_db) as conn:
        row = conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (job["job_id"],)
        ).fetchone()
    assert row["status"] == "done"


def test_failed_research_job_reports_failure(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(memory_service_module, "search_web", lambda query, cache=None: [])

    job = service.queue_research("unfindable topic")
    assert service._run_research_jobs() == 0

    messages = service.drain_proactive_messages()
    assert len(messages) == 1
    assert messages[0].startswith("Research failed — unfindable topic:")

    with service._connect(service._jobs_db) as conn:
        row = conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (job["job_id"],)
        ).fetchone()
    assert row["status"] == "failed"


def test_idle_step_failure_does_not_starve_reminders_or_research(
    tmp_path, monkeypatch
):
    """A crashing synthesis step used to abort run_idle_jobs before the
    reminder and research steps, silently killing both features."""
    service = _service(tmp_path, monkeypatch)
    service.create_reminder("call the plumber", due_at=_NOW - 60)

    def boom():
        raise RuntimeError("unable to open database file")

    monkeypatch.setattr(service, "_run_synthesis_job", boom)
    monkeypatch.setattr(service, "_execute_research", lambda topic: ("ok", []))
    service.queue_research("resilience topic")

    metrics = service.run_idle_jobs()

    assert metrics["reminders_fired"] == 1
    assert metrics["research_jobs_done"] == 1
    assert any("synthesis" in err for err in metrics["errors"])


def test_fire_due_reminders_now_works_outside_idle_pipeline(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.create_reminder("stand up and stretch", due_at=_NOW - 1)

    assert service.fire_due_reminders_now() == 1
    messages = service.drain_proactive_messages()
    assert len(messages) == 1
    assert "stand up and stretch" in messages[0]


def test_queue_research_deduplicates_pending_topics(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    first = service.queue_research("sqlite vector search")
    second = service.queue_research("sqlite vector search")

    assert first["job_id"] == second["job_id"]
    with service._connect(service._jobs_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE kind = 'research'"
        ).fetchone()["c"]
    assert count == 1


def test_overdue_research_respects_min_age(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        service, "_execute_research", lambda topic: (calls.append(topic), ("s", []))[1]
    )
    service.queue_research("fresh topic")

    # Too fresh: a job queued moments ago must wait for its idle window.
    assert service.run_overdue_research(min_age_seconds=60.0) == 0
    assert calls == []

    # Age the job past the threshold, then it force-starts.
    with service._connect(service._jobs_db) as conn:
        conn.execute("UPDATE jobs SET created_at = created_at - 120")
    assert service.run_overdue_research(min_age_seconds=60.0) == 1
    assert calls == ["fresh topic"]
