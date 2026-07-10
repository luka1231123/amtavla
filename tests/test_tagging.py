import time

from brain.memory.catalog import MemoryCatalog
from brain.memory.context_engine import ContextEngine
from brain.memory.tagging import TagEngine


def _catalog(tmp_path):
    return MemoryCatalog(str(tmp_path / "memory_catalog.db"))


def test_tag_assignment_lifecycle_and_feedback(tmp_path):
    catalog = _catalog(tmp_path)
    item = catalog.upsert_item(item_type="fact", content="Car parked at the airport")

    assigned = catalog.assign_tag(
        item["id"], "location", "Airport", status="suggested", confidence=0.6
    )
    assert assigned["status"] == "suggested"

    accepted = catalog.set_tag_status(item["id"], assigned["tag_id"], "accepted")
    assert accepted["status"] == "accepted"
    assert accepted["source"] == "user"

    # An auto re-suggestion can never override a user decision.
    again = catalog.assign_tag(
        item["id"], "location", "Airport", status="suggested", confidence=0.9
    )
    assert again["status"] == "accepted"

    summary = catalog.tag_feedback_summary()
    assert summary["location:airport"]["accepted"] == 1


def test_tag_correction_replaces_and_records_feedback(tmp_path):
    catalog = _catalog(tmp_path)
    item = catalog.upsert_item(item_type="episode", content="Working on the launch")
    wrong = catalog.assign_tag(item["id"], "project", "Launch Party")

    fixed = catalog.correct_tag_assignment(
        item["id"], wrong["tag_id"], "project", "Product Launch"
    )
    assert fixed["status"] == "accepted"
    assert fixed["name"] == "Product Launch"

    active = catalog.list_item_tags(item["id"])
    assert [tag["name"] for tag in active] == ["Product Launch"]
    assert catalog.tag_feedback_summary()["project:launch party"]["corrected"] == 1


def test_list_items_filters_by_tag_entity_and_time(tmp_path):
    catalog = _catalog(tmp_path)
    tagged = catalog.upsert_item(item_type="fact", content="Meeting notes for amtavla")
    other = catalog.upsert_item(item_type="fact", content="Grocery list")
    catalog.assign_tag(tagged["id"], "project", "Amtavla", status="accepted")
    entity = catalog.upsert_entity("person", "Anna")
    catalog.link_item_entity(other["id"], entity["id"])

    by_tag = catalog.list_items(tag="project:amtavla")
    assert [item["id"] for item in by_tag] == [tagged["id"]]
    assert by_tag[0]["tags"][0]["name"] == "Amtavla"

    by_entity = catalog.list_items(entity_id=entity["id"])
    assert [item["id"] for item in by_entity] == [other["id"]]

    future = time.time() + 3600
    assert catalog.list_items(since=future) == []
    assert len(catalog.list_items(until=future)) == 2


def test_engine_extracts_project_person_location_time(tmp_path):
    engine = TagEngine(_catalog(tmp_path))
    now = time.time()
    suggestions = engine.suggest_tags(
        "Met Anna at the airport garage today to discuss project Amtavla", now=now
    )
    by_type = {(tag["tag_type"], tag["name"].lower()) for tag in suggestions}
    assert ("person", "anna") in by_type
    assert ("project", "amtavla") in by_type
    assert any(tag_type == "location" for tag_type, _ in by_type)
    assert any(tag_type == "time" for tag_type, _ in by_type)


def test_engine_learns_from_rejections(tmp_path):
    catalog = _catalog(tmp_path)
    engine = TagEngine(catalog)
    now = time.time()

    for content in ("Call Anna about the report", "Lunch with Anna tomorrow"):
        item = catalog.upsert_item(item_type="episode", content=content)
        assigned = engine.tag_item(item, now=now)
        anna = next(tag for tag in assigned if tag["name"] == "Anna")
        catalog.set_tag_status(item["id"], anna["tag_id"], "rejected")

    suggestions = engine.suggest_tags("Dinner with Anna next week", now=now)
    assert not any(tag["name"] == "Anna" for tag in suggestions)


def test_engine_matches_known_entities(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_entity("place", "Tbilisi Mall", review_state="confirmed")
    engine = TagEngine(catalog)
    suggestions = engine.suggest_tags("left my keys at tbilisi mall")
    assert any(
        tag["tag_type"] == "location" and tag["name"] == "Tbilisi Mall"
        for tag in suggestions
    )


def test_context_engine_tracks_active_project(tmp_path):
    catalog = _catalog(tmp_path)
    engine = ContextEngine(catalog)
    now = time.time()
    catalog.record_context_snapshot(
        {"active_project": "Amtavla"}, session_id="cli", turn_id="t1"
    )

    context = engine.current_context("cli", now=now)
    assert context["active_project"] == "Amtavla"

    # Active project flows into tagging suggestions for untagged content.
    tag_engine = TagEngine(catalog)
    suggestions = tag_engine.suggest_tags(
        "Fixed the retrieval bug this morning", session_context=context, now=now
    )
    assert any(
        tag["tag_type"] == "project" and tag["name"] == "Amtavla"
        for tag in suggestions
    )


def test_capture_events_and_snapshots_roundtrip(tmp_path):
    catalog = _catalog(tmp_path)
    item = catalog.upsert_item(item_type="episode", content="a captured note")
    event = catalog.record_capture_event(
        "pasted", "a captured note", session_id="cli", memory_item_id=item["id"]
    )
    assert event["capture_type"] == "pasted"
    assert catalog.list_capture_events(session_id="cli")[0]["id"] == event["id"]

    snapshot = catalog.record_context_snapshot({"active_project": "X"}, session_id="cli")
    assert snapshot["snapshot"]["active_project"] == "X"
    assert catalog.recent_context_snapshots(session_id="cli")[0]["id"] == snapshot["id"]


def test_contradiction_flags_both_items(tmp_path):
    catalog = _catalog(tmp_path)
    first = catalog.upsert_item(item_type="fact", content="parked at level 3")
    second = catalog.upsert_item(item_type="fact", content="parked at street level")
    catalog.flag_contradiction(first["id"], second["id"], note="conflict")

    a = catalog.inspect_item(first["id"])
    b = catalog.inspect_item(second["id"])
    assert a["metadata"]["contradicts"] == [second["id"]]
    assert b["metadata"]["contradicts"] == [first["id"]]
    assert any(h["operation"] == "contradiction_flag" for h in a["history"])
    assert catalog.overview()["contradiction_count"] == 2
