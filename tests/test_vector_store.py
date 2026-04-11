from brain.memory.vector_store import SQLiteVecStore


def test_vector_store_upsert_and_query(tmp_path):
    db_path = tmp_path / "vectors.db"
    store = SQLiteVecStore(str(db_path), embedding_dim=4)

    store.upsert_node(
        node_id="n1",
        text_chunk="alpha signal",
        embedding=[1.0, 0.0, 0.0, 0.0],
        metadata={"kind": "insight", "status": "promoted", "insight_id": 1},
    )
    store.upsert_node(
        node_id="n2",
        text_chunk="beta signal",
        embedding=[0.0, 1.0, 0.0, 0.0],
        metadata={"kind": "insight", "status": "candidate", "insight_id": 2},
    )

    rows = store.query_knn(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=2,
        metadata_filter={"kind": "insight", "status": "promoted"},
    )
    assert rows
    assert rows[0]["node_id"] == "n1"

    store.clear()
    assert store.query_knn([1.0, 0.0, 0.0, 0.0], top_k=1) == []
