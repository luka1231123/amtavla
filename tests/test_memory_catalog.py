import json

import pytest

from brain.memory.catalog import MemoryCatalog


def _catalog(tmp_path):
    return MemoryCatalog(str(tmp_path / "memory_catalog.db"))


def test_item_provenance_correction_and_history(tmp_path):
    catalog = _catalog(tmp_path)
    item = catalog.upsert_item(
        item_type="fact",
        content="Sarah's birthday is May 12",
        review_state="confirmed",
        confidence=0.8,
        external_key="semantic_fact:1",
        sources=[
            {
                "source_type": "event",
                "source_id": "9",
                "excerpt": "Remember Sarah's birthday is May 12",
            }
        ],
    )

    corrected = catalog.correct_item(
        item["id"],
        "Sarah's birthday is May 13",
        "User corrected the date",
    )

    assert corrected["review_state"] == "corrected"
    assert corrected["version"] == 2
    assert corrected["sources"][0]["source_id"] == "9"
    assert corrected["sources"][1]["source_type"] == "user_correction"
    assert corrected["history"][0]["operation"] == "correct"
    assert corrected["history"][0]["before"]["content"].endswith("May 12")


def test_user_correction_is_not_overwritten_by_legacy_sync(tmp_path):
    catalog = _catalog(tmp_path)
    item = catalog.upsert_item(
        item_type="fact",
        content="Original value",
        external_key="semantic_fact:2",
    )
    catalog.correct_item(item["id"], "Corrected value")

    synced = catalog.upsert_item(
        item_type="fact",
        content="Original value reinforced",
        review_state="confirmed",
        external_key="semantic_fact:2",
        confidence=0.95,
    )

    assert synced["content"] == "Corrected value"
    assert synced["review_state"] == "corrected"
    assert synced["confidence"] == 0.95


def test_automated_sync_cannot_resurrect_or_downgrade_reviewed_memory(tmp_path):
    catalog = _catalog(tmp_path)
    confirmed = catalog.upsert_item(
        item_type="fact",
        content="Confirmed fact",
        review_state="confirmed",
        external_key="fact:confirmed",
    )
    deleted = catalog.upsert_item(
        item_type="fact",
        content="Deleted fact",
        review_state="confirmed",
        external_key="fact:deleted",
    )
    catalog.set_review_state(deleted["id"], "deleted")

    confirmed_sync = catalog.upsert_item(
        item_type="fact",
        content="Confirmed fact reinforced",
        review_state="candidate",
        external_key="fact:confirmed",
    )
    deleted_sync = catalog.upsert_item(
        item_type="fact",
        content="Deleted fact reinforced",
        review_state="confirmed",
        external_key="fact:deleted",
    )

    assert confirmed_sync["review_state"] == "confirmed"
    assert deleted_sync["review_state"] == "deleted"


def test_merge_archive_delete_and_search(tmp_path):
    catalog = _catalog(tmp_path)
    target = catalog.upsert_item(item_type="fact", content="Blue bicycle")
    source = catalog.upsert_item(item_type="fact", content="Bike color is blue")

    merged = catalog.merge_items(
        target["id"],
        [source["id"]],
        content="My bicycle is blue",
    )
    archived_source = catalog.inspect_item(source["id"])

    assert merged["content"] == "My bicycle is blue"
    assert archived_source["review_state"] == "archived"
    assert archived_source["merged_into_id"] == target["id"]
    assert catalog.list_items(query="bicycle")[0]["id"] == target["id"]

    deleted = catalog.set_review_state(target["id"], "deleted", "forget it")
    assert deleted["deleted_at"] is not None
    assert catalog.list_items(query="bicycle") == []
    assert catalog.list_items(include_deleted=True)


def test_entities_relations_and_item_links(tmp_path):
    catalog = _catalog(tmp_path)
    item = catalog.upsert_item(item_type="commitment", content="Review Phoenix Friday")
    person = catalog.upsert_entity("person", "Mira", review_state="confirmed")
    project = catalog.upsert_entity("project", "Phoenix", review_state="confirmed")
    catalog.link_item_entity(item["id"], project["id"], role="project", confidence=0.9)
    relation = catalog.create_relation(
        person["id"],
        "works on",
        project["id"],
        memory_item_id=item["id"],
        confidence=0.8,
    )

    inspected = catalog.inspect_item(item["id"])
    relations = catalog.list_relations()

    assert inspected["entities"][0]["name"] == "Phoenix"
    assert relation["relation_type"] == "works_on"
    assert relations[0]["source_name"] == "Mira"
    assert relations[0]["target_name"] == "Phoenix"


def test_plain_file_export_is_complete(tmp_path):
    catalog = _catalog(tmp_path)
    item = catalog.upsert_item(
        item_type="preference",
        content="I prefer concise answers",
        review_state="confirmed",
        sources=[
            {"source_type": "turn", "source_id": "turn-1", "excerpt": "concise"}
        ],
    )
    catalog.set_review_state(item["id"], "deleted")

    result = catalog.export(str(tmp_path / "export"))

    assert result["item_count"] == 1
    assert (tmp_path / "export" / "README.md").exists()
    assert "Preference" in (tmp_path / "export" / "memory.md").read_text()
    rows = (tmp_path / "export" / "memory_items.jsonl").read_text().splitlines()
    exported = json.loads(rows[0])
    assert exported["review_state"] == "deleted"
    assert exported["sources"][0]["source_id"] == "turn-1"


def test_catalog_rejects_invalid_types_and_empty_content(tmp_path):
    catalog = _catalog(tmp_path)
    with pytest.raises(ValueError):
        catalog.upsert_item(item_type="mystery", content="value")
    with pytest.raises(ValueError):
        catalog.upsert_item(item_type="fact", content="  ")
