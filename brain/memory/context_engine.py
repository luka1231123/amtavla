from __future__ import annotations

import time
from collections import Counter
from typing import Any

from brain.memory.catalog import MemoryCatalog


class ContextEngine:
    """Infer the current working context from recent snapshots, tags, and captures.

    Combines recency, active project, correction-shaped tag history, and the
    current session so tagging and retrieval can lean on what the user is
    actually doing right now. Catalog-only: no models, no network.
    """

    def __init__(self, catalog: MemoryCatalog, config: dict[str, Any] | None = None):
        self.catalog = catalog
        config = config or {}
        self.snapshot_max_age_seconds = float(
            config.get("snapshot_max_age_seconds", 6 * 3600)
        )
        self.recent_item_window = int(config.get("recent_item_window", 25))

    def current_context(
        self, session_id: str = "", *, now: float | None = None
    ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        context: dict[str, Any] = {
            "session_id": session_id,
            "active_project": "",
            "recent_people": [],
            "recent_locations": [],
            "recent_topics": [],
        }

        # The freshest snapshot for this session wins for the active project.
        for snap in self.catalog.recent_context_snapshots(
            session_id=session_id or None, limit=5
        ):
            if now - float(snap["created_at"]) > self.snapshot_max_age_seconds:
                break
            active_project = (snap["snapshot"].get("active_project") or "").strip()
            if active_project:
                context["active_project"] = active_project
                break

        # Otherwise infer from recent items' tags; accepted tags count double.
        counts: dict[str, Counter] = {
            "project": Counter(),
            "person": Counter(),
            "location": Counter(),
            "topic": Counter(),
        }
        for item in self.catalog.list_items(limit=self.recent_item_window):
            for tag in item.get("tags", []):
                counter = counts.get(tag["tag_type"])
                if counter is not None:
                    weight = 2 if tag["status"] == "accepted" else 1
                    counter[tag["name"]] += weight

        if not context["active_project"] and counts["project"]:
            context["active_project"] = counts["project"].most_common(1)[0][0]
        context["recent_people"] = [name for name, _ in counts["person"].most_common(5)]
        context["recent_locations"] = [
            name for name, _ in counts["location"].most_common(5)
        ]
        context["recent_topics"] = [name for name, _ in counts["topic"].most_common(5)]
        return context

    def build_snapshot(
        self,
        *,
        session_id: str = "",
        intent: str = "",
        pathway: str = "",
        source_ids: list[str] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Compose the per-turn snapshot persisted alongside a committed turn."""
        context = self.current_context(session_id, now=now)
        return {
            "active_project": context["active_project"],
            "recent_people": context["recent_people"],
            "recent_locations": context["recent_locations"],
            "recent_topics": context["recent_topics"],
            "intent": intent,
            "pathway": pathway,
            "source_ids": list(source_ids or []),
        }
