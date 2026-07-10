import server.phone_server as phone_server
from brain.memory.catalog import MemoryCatalog


def _client(tmp_path, monkeypatch):
    catalog = MemoryCatalog(str(tmp_path / "memory_catalog.db"))
    monkeypatch.setattr(phone_server, "MEMORY_CATALOG", catalog)
    phone_server.app.config.update(TESTING=True)
    return phone_server.app.test_client(), catalog


def test_tag_review_endpoints(tmp_path, monkeypatch):
    client, catalog = _client(tmp_path, monkeypatch)
    item = catalog.upsert_item(item_type="fact", content="notes about the launch")
    suggested = catalog.assign_tag(item["id"], "project", "Launch", status="suggested")

    accepted = client.post(
        f"/api/memory/items/{item['id']}/tags/{suggested['tag_id']}",
        json={"action": "accept"},
    )
    assert accepted.status_code == 200
    assert accepted.get_json()["status"] == "accepted"

    corrected = client.post(
        f"/api/memory/items/{item['id']}/tags/{suggested['tag_id']}",
        json={"action": "correct", "tag_type": "project", "name": "Product Launch"},
    )
    assert corrected.status_code == 200
    assert corrected.get_json()["name"] == "Product Launch"

    bad = client.post(
        f"/api/memory/items/{item['id']}/tags/{suggested['tag_id']}",
        json={"action": "explode"},
    )
    assert bad.status_code == 400

    added = client.post(
        f"/api/memory/items/{item['id']}/tags",
        json={"tag_type": "person", "name": "Anna"},
    )
    assert added.status_code == 201
    assert added.get_json()["status"] == "accepted"

    listing = client.get("/api/memory/tags").get_json()
    assert listing["count"] >= 2

    filtered = client.get("/api/memory/items?tag=person:anna").get_json()
    assert [row["id"] for row in filtered["items"]] == [item["id"]]


def test_capture_endpoint_creates_item_event_and_tags(tmp_path, monkeypatch):
    client, catalog = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/memory/capture",
        json={"content": "Met Anna today about project Amtavla", "capture_type": "voice"},
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["event"]["capture_type"] == "voice"
    assert any(tag["name"] == "Anna" for tag in body["tags"])
    assert catalog.list_capture_events()[0]["memory_item_id"] == body["item"]["id"]

    empty = client.post("/api/memory/capture", json={"content": ""})
    assert empty.status_code == 400
