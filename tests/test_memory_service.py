import json
import os
import queue
import sqlite3
import threading
import time

import brain.memory.service as memory_service_module
import brain.memory_controller as memory_controller_module
import pytest
from brain.memory.catalog import MemoryCatalog
from brain.memory.catalog import resolve_repo_path
from brain.memory.vector_store import SQLiteVecStore


def _config(vector_db):
    return {
        "memory": {
            "vector_db_path": str(vector_db),
            "embedding_dim": 8,
            "vector_top_k": 3,
            "proactive_max_asks": 2,
            "proactive_snooze_seconds": 60,
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


def _facts(service):
    with service._connect(service._semantic_db) as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT statement, confidence, provenance_json FROM facts ORDER BY id"
            ).fetchall()
        ]


def test_resolve_repo_path_anchors_relative_paths_to_repo_root():
    resolved = resolve_repo_path("brain/db")
    assert os.path.isabs(resolved)
    assert resolved.endswith(os.path.join("brain", "db"))
    # An absolute path is returned unchanged.
    assert resolve_repo_path("/tmp/amtavla-x.db") == "/tmp/amtavla-x.db"


def test_sqlite_contexts_close_connections(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    catalog = MemoryCatalog(str(tmp_path / "standalone-catalog.db"))
    vectors = SQLiteVecStore(str(tmp_path / "standalone-vectors.db"), embedding_dim=8)

    connections = []
    for manager in (
        service._connect(service._semantic_db),
        catalog._connect(),
        vectors._connect(),
    ):
        with manager as conn:
            conn.execute("SELECT 1").fetchone()
            connections.append(conn)

    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")


def test_recent_dialogue_returns_recent_turns_chronologically(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.commit_turn("to err is human", "That is a common idiom.")
    service.commit_turn("what is the continuation", "To persist is diabolical.")

    dialogue = service.recent_dialogue()

    assert [t["user_input"] for t in dialogue] == [
        "to err is human",
        "what is the continuation",
    ]
    # Recent dialogue feeds continuity regardless of keyword overlap — the whole
    # point is that a vague follow-up with no shared tokens still sees context.
    assert dialogue[-1]["response"] == "To persist is diabolical."


def test_recent_dialogue_excludes_turns_outside_the_window(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.commit_turn("an old exchange", "old reply")

    # A large gap means a new conversation, not a continuation: nothing recent.
    assert service.recent_dialogue(within_seconds=0) == []


def test_recall_context_grounds_on_user_words_not_assistant_prose(tmp_path, monkeypatch):
    # The recent-conversation block in combined_context must respect the same
    # firewall as episodic recall: the user's utterance grounds the next turn,
    # the assistant's own prior prose never does.
    service = _service(tmp_path, monkeypatch)
    service.commit_turn(
        "where is my bike",
        "Your bike is at level 3.",  # confabulation — must not re-enter grounding
    )

    combined = service.recall_context("anything", include_web=False)["combined_context"]

    assert "[Recent Conversation]" in combined
    assert "where is my bike" in combined
    assert "level 3" not in combined


def test_recall_drops_stopword_matches_and_zero_score_vectors(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.write_memory("Remember this: my bicycle is beside the garage")
    monkeypatch.setattr(
        memory_service_module.llama_client,
        "embed",
        lambda text: {"embedding": [0.0] * 8},
    )

    recall = service.recall_context(
        "what should you possess", include_web=False, top_k=5
    )

    assert recall["memory_items"] == []
    assert recall["semantic"] == []


class _RaiseOnceMemory:
    def __init__(self):
        self.committed = []
        self.calls = 0

    def commit_turn(self, user_input, response, trace=None):
        self.calls += 1
        if self.calls == 1:
            raise sqlite3.OperationalError("unable to open database file")
        self.committed.append((user_input, response))


class _FailingIdleMemory:
    def __init__(self):
        self.calls = 0

    def run_idle_jobs(self):
        self.calls += 1
        raise sqlite3.OperationalError("unable to open database file")


def test_worker_survives_a_raising_commit_turn():
    # A single failing commit must not kill the worker: the next good commit
    # still lands. Build the controller without __init__ to avoid a real
    # MemoryService, and drive its worker loop directly.
    controller = object.__new__(memory_controller_module.MemoryController)
    controller.memory = _RaiseOnceMemory()
    controller._turn_queue = queue.Queue()
    controller._stop_event = threading.Event()

    thread = threading.Thread(target=controller._worker_loop, daemon=True)
    thread.start()
    try:
        controller._turn_queue.put(("bad", "r1", {}))
        controller._turn_queue.put(("good", "r2", {}))
        controller._turn_queue.join()
    finally:
        controller._stop_event.set()
        controller._turn_queue.put(None)
        thread.join(timeout=1.0)

    assert controller.memory.calls == 2
    assert controller.memory.committed == [("good", "r2")]


def test_failed_idle_run_still_obeys_minimum_interval(monkeypatch):
    controller = object.__new__(memory_controller_module.MemoryController)
    controller.memory = _FailingIdleMemory()
    controller._turn_queue = queue.Queue()
    controller._foreground_lock = threading.Lock()
    controller._foreground_active = 0
    controller._last_activity_ts = 0.0
    controller._last_idle_run_ts = 0.0
    controller._idle_seconds = 4.0
    controller._idle_min_interval_seconds = 5.0
    monkeypatch.setattr(memory_controller_module.time, "time", lambda: 100.0)

    with pytest.raises(sqlite3.OperationalError):
        controller._run_idle_memory_maintenance()
    controller._run_idle_memory_maintenance()

    assert controller.memory.calls == 1
    assert controller._last_idle_run_ts == 100.0


def test_explicit_remember_stores_non_first_person_fact(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    service.commit_turn(
        "Remember this: Sarah's birthday is May 12.",
        "Noted.",
        {"intent": "remember"},
    )

    rows = _facts(service)
    assert len(rows) == 1
    assert rows[0]["statement"] == "Sarah's birthday is May 12"
    assert rows[0]["confidence"] == 0.75
    assert json.loads(rows[0]["provenance_json"])[0]["source"] == "explicit_remember"


def test_passive_non_first_person_command_is_not_stored(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    service.commit_turn(
        "List project Phoenix staging database details.",
        "I cannot access that.",
        {"intent": "default"},
    )

    assert _facts(service) == []


def test_polluted_memory_cleanup_deletes_bad_facts(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    now = time.time()
    with service._connect(service._semantic_db) as conn:
        conn.execute(
            """
            INSERT INTO facts(statement, canonical_key, confidence, first_seen, last_seen, provenance_json)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            ("**bad markdown fact**", "bad fact", 0.5, now, now, "[]"),
        )

    result = service.cleanup_polluted_memory()

    assert result["deleted_semantic_facts"] == 1
    assert _facts(service) == []


def test_proactive_feedback_confirm_promotes_candidate(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    now = time.time()
    with service._connect(service._insight_db) as conn:
        cursor = conn.execute(
            """
            INSERT INTO insights(thesis, rationale, evidence_json, novelty_score, confidence, status, feedback_state, ask_count, quality_score, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                "The user does focused work best after short planning.",
                "test",
                "{}",
                0.8,
                0.8,
                "candidate",
                "asked",
                0.8,
                now,
            ),
        )
        insight_id = int(cursor.lastrowid)
    service._active_asked_insight_id = insight_id

    service.commit_turn("yes keep it", "Saved.", {"intent": "default"})

    with service._connect(service._insight_db) as conn:
        row = conn.execute(
            "SELECT status, feedback_state FROM insights WHERE id = ?",
            (insight_id,),
        ).fetchone()
    assert row["status"] == "promoted"
    assert row["feedback_state"] == "confirmed"


def test_embedding_zero_vector_is_visible_in_status(tmp_path, monkeypatch):
    monkeypatch.setattr(
        memory_service_module,
        "load_brain_config",
        lambda: _config(tmp_path / "vectors.db"),
    )
    monkeypatch.setattr(
        memory_service_module.llama_client,
        "embed",
        lambda text: {"embedding": [0.0] * 8},
    )
    service = memory_service_module.MemoryService(db_dir=str(tmp_path / "db"))

    service._embed_text("anything")

    status = service.get_status()
    assert status["embedding_available"] is False
    assert "zero vector" in status["embedding_last_error"]


def test_explicit_memory_api_writes_with_provenance_and_can_be_searched(
    tmp_path, monkeypatch
):
    service = _service(tmp_path, monkeypatch)

    item = service.write_memory("The launch review is Friday.")
    result = service.search_memory("launch review")

    assert item["statement"] == "The launch review is Friday"
    assert item["confidence"] == 0.85
    assert item["provenance"][0]["source"] == "memory_write"
    assert result["semantic"][0]["id"] == item["id"]


def test_turn_creates_derived_episode_and_typed_preference(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    service.commit_turn(
        "I prefer concise answers with examples.",
        "Understood.",
        {"intent": "remember", "turn_id": "turn-123", "session_id": "test"},
    )

    items = service.list_memory_items(include_deleted=True)
    preference = next(item for item in items if item["item_type"] == "preference")
    episode = next(item for item in items if item["item_type"] == "episode")
    inspected = service.inspect_memory_item(preference["id"])

    assert preference["review_state"] == "candidate"
    assert episode["metadata"]["turn_id"] == "turn-123"
    assert inspected["sources"][0]["source_type"] == "event"
    assert inspected["sources"][0]["metadata"]["turn_id"] == "turn-123"


def test_corrected_and_deleted_catalog_memory_controls_recall(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    stored = service.write_memory("Project Phoenix review is Friday.")
    item_id = stored["memory_item_id"]

    corrected = service.correct_memory_item(
        item_id,
        "Project Phoenix review is Monday",
        "The date changed",
    )
    recalled = service.recall_memory_items("Phoenix Monday", top_k=10)

    assert corrected["review_state"] == "corrected"
    assert corrected["sources"][-1]["source_type"] == "user_correction"
    assert item_id in {item["id"] for item in recalled}

    service.set_memory_review_state(item_id, "deleted", "User asked to forget")
    recalled_after_delete = service.recall_memory_items("Phoenix Monday", top_k=10)
    assert item_id not in {item["id"] for item in recalled_after_delete}
    full_context = service.recall_context("Phoenix Monday", include_web=False)
    assert full_context["semantic"] == []


def test_unified_recall_exposes_inspectable_source_id(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    stored = service.write_memory("Sarah's birthday is May 12.")

    context = service.recall_context("Sarah birthday", include_web=False)

    assert context["memory_items"][0]["id"] == stored["memory_item_id"]
    assert context["memory_items"][0]["sources"]


def test_entity_named_in_query_recalls_linked_items(tmp_path, monkeypatch):
    # Phase 3 source-aware retrieval: naming a project surfaces items filed under
    # it even when their wording shares no token with the query.
    service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        memory_service_module.llama_client,
        "embed",
        lambda text: {"embedding": [0.0] * 8},
    )  # neutralize the degenerate fake vector so entity/tag recall is isolated
    cat = service.catalog
    entity = cat.upsert_entity("project", "Vake flat", review_state="confirmed")
    item = cat.upsert_item(
        item_type="fact",
        content="The viewing is scheduled on Saturday.",
        review_state="confirmed",
    )
    cat.link_item_entity(item["id"], entity["id"])

    recalled = service.recall_memory_items("what's happening with the Vake flat")

    assert item["id"] in {r["id"] for r in recalled}


def test_tag_named_in_query_recalls_tagged_items(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        memory_service_module.llama_client,
        "embed",
        lambda text: {"embedding": [0.0] * 8},
    )  # neutralize the degenerate fake vector so entity/tag recall is isolated
    cat = service.catalog
    item = cat.upsert_item(
        item_type="fact",
        content="Bring the signed lease and a deposit.",
        review_state="confirmed",
    )
    cat.assign_tag(item["id"], "project", "Housing", status="accepted")

    recalled = service.recall_memory_items("anything about Housing?")

    assert item["id"] in {r["id"] for r in recalled}


def test_unrelated_entity_is_not_pulled_in(tmp_path, monkeypatch):
    # A meaningful entity-name token must actually appear in the query, so an
    # unrelated project is not dragged into every recall.
    service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        memory_service_module.llama_client,
        "embed",
        lambda text: {"embedding": [0.0] * 8},
    )  # neutralize the degenerate fake vector so entity/tag recall is isolated
    cat = service.catalog
    entity = cat.upsert_entity("project", "Zephyr", review_state="confirmed")
    item = cat.upsert_item(
        item_type="fact", content="Kickoff is next month.", review_state="confirmed"
    )
    cat.link_item_entity(item["id"], entity["id"])

    recalled = service.recall_memory_items("where is my bike parked")

    assert item["id"] not in {r["id"] for r in recalled}


def test_broad_recall_sweeps_user_facts_without_lexical_or_vector_match(
    tmp_path, monkeypatch
):
    # The live "IDK about myself" failure: a broad recall query shares no tokens
    # with any stored fact, so the lexical gate returned nothing. The sweep must
    # surface the durable facts anyway. Zero-vector embeddings ensure only the
    # sweep — not vector similarity — can be responsible.
    service = _service(tmp_path, monkeypatch)
    service.commit_turn("my name is Mira", "ok", {"intent": "remember", "turn_id": "t1"})
    service.commit_turn(
        "I have a car parked in 10 B", "ok", {"intent": "remember", "turn_id": "t2"}
    )
    monkeypatch.setattr(
        memory_service_module.llama_client, "embed", lambda text: {"embedding": [0.0] * 8}
    )

    recall = service.recall_context("what do you know about me?", include_web=False)
    contents = " ".join(item["content"] for item in recall["memory_items"])

    assert "Mira" in contents
    assert "10 B" in contents


def test_typed_fact_stored_and_recalled(tmp_path, monkeypatch):
    # The live failure: "I have a car parked in 10 B" stored nothing structured.
    # Now it becomes a typed (car, location) fact that recall can surface.
    service = _service(tmp_path, monkeypatch)
    service.commit_turn(
        "remember that I have a car parked in 10 B of the parking lot",
        "Okay.",
        {"intent": "remember", "turn_id": "t1"},
    )
    loc = [
        item
        for item in service.list_memory_items()
        if item["item_type"] == "fact"
        and (item.get("metadata") or {}).get("attribute") == "location"
    ]
    assert len(loc) == 1
    assert loc[0]["external_key"] == "fact:car:location"
    assert "10 B" in loc[0]["content"]

    recall = service.recall_context("where is my car parked", include_web=False)
    assert "10 B" in recall["combined_context"]


def test_typed_fact_new_value_supersedes_without_duplicate(tmp_path, monkeypatch):
    # A new value for the same (entity, attribute) must update in place, not fork
    # into a contradictory second fact — the flaw behind the "10 B vs level 3"
    # phantom conflict in the live session.
    service = _service(tmp_path, monkeypatch)
    service.commit_turn(
        "my car is parked in 10 B",
        "Okay.",
        {"intent": "remember", "turn_id": "t1"},
    )
    service.commit_turn(
        "my car is now in 10 C",
        "Okay.",
        {"intent": "remember", "turn_id": "t2"},
    )
    loc = [
        item
        for item in service.list_memory_items()
        if item["item_type"] == "fact"
        and (item.get("metadata") or {}).get("attribute") == "location"
    ]
    assert len(loc) == 1  # superseded, not duplicated
    assert "10 C" in loc[0]["content"]
    assert "10 B" not in loc[0]["content"]
    assert loc[0]["version"] >= 2  # history preserved via version bump


def test_recall_firewall_excludes_assistant_prose(tmp_path, monkeypatch):
    # Reproduces the "level 3" hallucination loop: the assistant fabricates a
    # value in its reply, and on a later recall that fabrication must NOT come
    # back as memory. Only the user's own words are grounded observations.
    service = _service(tmp_path, monkeypatch)
    service.commit_turn(
        "can you remember that I have a car parked in 10 B of the parking lot",
        "Your car is at level 3.",  # assistant confabulation — must never re-enter recall
        {"intent": "remember", "turn_id": "turn-car"},
    )

    context = service.recall_context("where is my car parked", include_web=False)
    combined = context["combined_context"]

    # The user's grounded utterance survives recall...
    assert "10 B" in combined
    # ...but the assistant's fabricated value is firewalled out of the context
    # that grounds the next answer.
    assert "level 3" not in combined


def test_deleted_derived_episode_does_not_leak_from_raw_event_adapter(
    tmp_path, monkeypatch
):
    service = _service(tmp_path, monkeypatch)
    service.commit_turn(
        "Discuss lunar archive protocol.",
        "The protocol uses cold storage.",
        {"intent": "default", "turn_id": "turn-episode"},
    )
    episode = next(
        item
        for item in service.list_memory_items(include_deleted=True)
        if item["item_type"] == "episode"
    )
    service.set_memory_review_state(episode["id"], "deleted")

    context = service.recall_context("lunar archive", include_web=False)

    assert context["episodic"] == []
    assert episode["id"] not in {item["id"] for item in context["memory_items"]}


def test_transient_tool_sources_do_not_become_recallable_memory(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.commit_turn(
        "Find the current release date.",
        "The release is Monday.",
        {
            "intent": "web_factual",
            "turn_id": "turn-web",
            "actions": [
                {
                    "action_id": "action-1",
                    "action": "SEARCH",
                    "sources": [
                        {
                            "source_id": "web:abc",
                            "kind": "web",
                            "title": "Release notes",
                            "excerpt": "The release is Monday.",
                            "url": "https://example.test/release",
                            "metadata": {"rank": 1},
                        }
                    ],
                }
            ],
        },
    )

    excerpts = [
        item
        for item in service.list_memory_items(include_deleted=True)
        if item["item_type"] == "source_excerpt"
    ]
    assert excerpts == []
    assert service.recall_memory_items("release Monday") == []


def test_existing_facts_and_insights_migrate_into_catalog(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    now = time.time()
    with sqlite3.connect(db_dir / "semantic.db") as conn:
        conn.execute(
            """
            CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                statement TEXT NOT NULL,
                canonical_key TEXT NOT NULL UNIQUE,
                confidence REAL NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                provenance_json TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO facts VALUES(1, ?, ?, 0.8, ?, ?, ?)",
            (
                "I prefer keyboard shortcuts",
                "keyboard prefer shortcuts",
                now,
                now,
                '[{"source":"turn"}]',
            ),
        )
    with sqlite3.connect(db_dir / "insight_ltm.db") as conn:
        conn.execute(
            """
            CREATE TABLE insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thesis TEXT NOT NULL,
                rationale TEXT,
                evidence_json TEXT,
                novelty_score REAL NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                feedback_state TEXT NOT NULL,
                ask_count INTEGER NOT NULL DEFAULT 0,
                last_asked_at REAL,
                snoozed_until REAL,
                quality_score REAL NOT NULL DEFAULT 0.0,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO insights(
                id, thesis, rationale, evidence_json, novelty_score, confidence,
                status, feedback_state, quality_score, created_at
            ) VALUES(2, ?, ?, '{}', 0.8, 0.9, 'promoted', 'approved', 0.8, ?)
            """,
            ("Short planning improves focus", "Observed repeatedly", now),
        )

    monkeypatch.setattr(
        memory_service_module,
        "load_brain_config",
        lambda: _config(tmp_path / "vectors.db"),
    )
    monkeypatch.setattr(memory_service_module.llama_client, "embed", _fake_embed)
    service = memory_service_module.MemoryService(db_dir=str(db_dir))

    preference = service.catalog.get_by_external_key("semantic_fact:1")
    insight = service.catalog.get_by_external_key("insight:2")

    assert preference["item_type"] == "preference"
    assert preference["sources"][0]["source_type"] == "legacy_fact_provenance"
    assert insight["item_type"] == "insight"
    assert insight["review_state"] == "confirmed"


def _insert_candidate_insight(service, thesis="Focus improves after planning."):
    now = time.time()
    with service._connect(service._insight_db) as conn:
        cursor = conn.execute(
            """
            INSERT INTO insights(thesis, rationale, evidence_json, novelty_score, confidence, status, feedback_state, ask_count, quality_score, created_at)
            VALUES(?, 'test', '{}', 0.8, 0.8, 'candidate', 'asked', 0, 0.8, ?)
            """,
            (thesis, now),
        )
        return int(cursor.lastrowid)


def test_apply_insight_feedback_keep_and_discard(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    kept_id = _insert_candidate_insight(service, "Kept thesis about planning.")
    dropped_id = _insert_candidate_insight(service, "Dropped thesis about noise.")

    assert service.apply_insight_feedback(kept_id, True)["kept"] is True
    assert service.apply_insight_feedback(dropped_id, False)["kept"] is False

    with service._connect(service._insight_db) as conn:
        kept = conn.execute(
            "SELECT status, feedback_state FROM insights WHERE id = ?", (kept_id,)
        ).fetchone()
        dropped = conn.execute(
            "SELECT status, feedback_state FROM insights WHERE id = ?", (dropped_id,)
        ).fetchone()
    assert (kept["status"], kept["feedback_state"]) == ("promoted", "confirmed")
    assert (dropped["status"], dropped["feedback_state"]) == (
        "candidate",
        "rejected",
    )


def test_incidental_yes_no_words_do_not_resolve_memory_check(tmp_path, monkeypatch):
    """"no" or "right" buried in an unrelated sentence must not count as an
    answer to the pending keep/discard question."""
    service = _service(tmp_path, monkeypatch)
    insight_id = _insert_candidate_insight(service)
    service._active_asked_insight_id = insight_id

    service.commit_turn(
        "I parked the car on the right side of the street, there is no garage there",
        "Noted.",
        {"intent": "default"},
    )

    with service._connect(service._insight_db) as conn:
        row = conn.execute(
            "SELECT status, feedback_state FROM insights WHERE id = ?",
            (insight_id,),
        ).fetchone()
    # Neither promoted nor rejected — just snoozed for later.
    assert row["status"] == "candidate"
    assert row["feedback_state"] == "snoozed"


def test_recall_context_separates_memory_check_from_reminder_nudge(
    tmp_path, monkeypatch
):
    service = _service(tmp_path, monkeypatch)
    recall = service.recall_context("what should I do today", include_web=False)

    assert "reminder_nudge" in recall
    # The keep/discard question never rides inside the reminder nudge.
    assert "should I keep this insight" not in (recall["reminder_nudge"] or "")


def test_embedding_health_is_cheap_and_reflects_state(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service._embed_text("warm up")

    health = service.embedding_health()
    assert health["embedding_available"] is True
    assert health["embedding_last_error"] == ""
