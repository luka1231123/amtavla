from __future__ import annotations

import time
from typing import Any

from brain.memory.catalog import MemoryCatalog
from brain.memory.commitments import extract_commitments
from brain.memory.context_engine import ContextEngine
from brain.memory.tagging import TagEngine


def capture_note(
    catalog: MemoryCatalog,
    tag_engine: TagEngine,
    context_engine: ContextEngine,
    content: str,
    *,
    item_type: str = "episode",
    capture_type: str = "pasted",
    session_id: str = "capture",
    now: float | None = None,
) -> dict[str, Any]:
    """Model-free capture pipeline: upsert, log the event, tag, extract commitments.

    Shared by the in-process MemoryService (which also syncs an LTM vector
    node) and the model-free dashboard server, so both entry points produce
    the same memory item shape instead of drifting apart.
    """
    now = now if now is not None else time.time()
    item = catalog.upsert_item(
        item_type=item_type,
        content=content,
        review_state="confirmed",
        confidence=0.7,
        importance=0.5,
        metadata={"captured": True, "session_id": session_id},
    )
    event = catalog.record_capture_event(
        capture_type,
        content,
        session_id=session_id,
        memory_item_id=item["id"],
    )
    session_context = context_engine.current_context(session_id, now=now)
    tags = tag_engine.tag_item(item, session_context=session_context, now=now)
    commitments = []
    for candidate in extract_commitments(content, now):
        metadata: dict[str, Any] = {"status": "open", "origin": "conversation"}
        if candidate["due_at"]:
            metadata["due_at"] = candidate["due_at"]
        commitment_item = catalog.upsert_item(
            item_type="commitment",
            content=candidate["content"],
            review_state="candidate",
            confidence=candidate["confidence"],
            importance=0.7,
            external_key=f"commitment:{candidate['canonical_key']}",
            metadata=metadata,
        )
        tag_engine.tag_item(commitment_item, session_context=session_context, now=now)
        commitments.append(commitment_item)
    return {
        "item": catalog.inspect_item(item["id"]),
        "event": event,
        "tags": tags,
        "commitments": commitments,
    }
