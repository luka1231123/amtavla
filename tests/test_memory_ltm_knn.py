import time

import brain.memory.service as memory_service_module


def _fake_embed(text: str) -> dict:
    base = [0.0] * 8
    if "python" in text.lower():
        base[0] = 1.0
    elif "car" in text.lower():
        base[1] = 1.0
    return {"embedding": base}


def test_memory_service_recall_uses_vector_ltm(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    vector_db = tmp_path / "ltm_vectors.db"
    monkeypatch.setattr(
        memory_service_module,
        "load_brain_config",
        lambda: {
            "memory": {
                "vector_db_path": str(vector_db),
                "embedding_dim": 8,
                "vector_top_k": 3,
            }
        },
    )
    monkeypatch.setattr(memory_service_module.llama_client, "embed", _fake_embed)

    service = memory_service_module.MemoryService(db_dir=str(db_dir))
    now = time.time()
    with service._connect(service._insight_db) as conn:
        cursor = conn.execute(
            """
            INSERT INTO insights(thesis, rationale, evidence_json, novelty_score, confidence, status, feedback_state, ask_count, quality_score, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                "Python is useful for scripting tasks",
                "test",
                "{}",
                0.8,
                0.9,
                "promoted",
                "approved",
                0.8,
                now,
            ),
        )
        insight_id = int(cursor.lastrowid)

    service._sync_insight_vector(insight_id)
    recall = service.recall_context("python scripting", include_web=False, top_k=3)
    assert recall["insights"]
    assert recall["insights"][0]["id"] == insight_id
