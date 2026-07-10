import server.phone_server as phone_server
from brain.memory.catalog import MemoryCatalog


def _client(tmp_path, monkeypatch):
    catalog = MemoryCatalog(str(tmp_path / "memory_catalog.db"))
    monkeypatch.setattr(phone_server, "MEMORY_CATALOG", catalog)
    monkeypatch.setattr(phone_server, "EXPORT_ROOT", tmp_path / "exports")
    phone_server.app.config.update(TESTING=True)
    return phone_server.app.test_client(), catalog


def test_memory_dashboard_and_review_api(tmp_path, monkeypatch):
    client, catalog = _client(tmp_path, monkeypatch)
    item = catalog.upsert_item(
        item_type="fact",
        content="The review is Friday",
        review_state="candidate",
        sources=[{"source_type": "event", "source_id": "1"}],
    )

    assert client.get("/memory").status_code == 200
    listing = client.get("/api/memory/items?q=review").get_json()
    assert listing["items"][0]["id"] == item["id"]

    corrected = client.post(
        f"/api/memory/items/{item['id']}/correct",
        json={"content": "The review is Monday", "reason": "date changed"},
    )
    assert corrected.status_code == 200
    assert corrected.get_json()["review_state"] == "corrected"

    confirmed = client.post(
        f"/api/memory/items/{item['id']}/state",
        json={"review_state": "confirmed"},
    )
    assert confirmed.get_json()["review_state"] == "confirmed"
    assert client.get("/api/memory/overview").get_json()["by_state"]["confirmed"] == 1


def test_memory_merge_entity_relation_and_export_api(tmp_path, monkeypatch):
    client, catalog = _client(tmp_path, monkeypatch)
    target = catalog.upsert_item(item_type="fact", content="Phoenix review")
    source = catalog.upsert_item(item_type="fact", content="Project review")

    merged = client.post(
        "/api/memory/merge",
        json={"target_id": target["id"], "source_ids": [source["id"]]},
    )
    assert merged.status_code == 200
    assert catalog.inspect_item(source["id"])["review_state"] == "archived"

    person = client.post(
        "/api/memory/entities",
        json={
            "entity_type": "person",
            "name": "Mira",
            "memory_item_id": target["id"],
            "role": "owner",
        },
    ).get_json()
    project = client.post(
        "/api/memory/entities",
        json={"entity_type": "project", "name": "Phoenix"},
    ).get_json()
    relation = client.post(
        "/api/memory/relations",
        json={
            "source_entity_id": person["id"],
            "relation_type": "owns",
            "target_entity_id": project["id"],
            "memory_item_id": target["id"],
        },
    )
    assert relation.status_code == 201
    assert client.get("/api/memory/relations").get_json()["count"] == 1

    exported = client.post("/api/memory/export")
    assert exported.status_code == 200
    assert exported.mimetype == "application/zip"
    assert int(exported.headers["X-Memory-Item-Count"]) == 2
    assert list((tmp_path / "exports").glob("*.zip"))


def test_memory_api_validates_operations(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)

    missing = client.get("/api/memory/items/999")
    invalid = client.post(
        "/api/memory/items/999/state",
        json={"review_state": "unknown"},
    )

    assert missing.status_code == 404
    assert invalid.status_code == 400
