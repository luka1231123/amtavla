from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


MEMORY_ITEM_TYPES = {
    "fact",
    "episode",
    "decision",
    "commitment",
    "preference",
    "idea",
    "insight",
    "source_excerpt",
}
REVIEW_STATES = {
    "candidate",
    "confirmed",
    "corrected",
    "rejected",
    "archived",
    "deleted",
}
ENTITY_TYPES = {
    "person",
    "place",
    "project",
    "idea",
    "document",
    "commitment",
    "organization",
    "other",
}
TAG_TYPES = {
    "project",
    "person",
    "location",
    "time",
    "topic",
}
TAG_STATUSES = {
    "suggested",
    "accepted",
    "corrected",
    "rejected",
}
CAPTURE_TYPES = {
    "typed",
    "pasted",
    "parsed",
    "voice",
    "ambient",
}


def _json_load(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _canonical_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


class MemoryCatalog:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_key TEXT UNIQUE,
                    item_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    importance REAL NOT NULL DEFAULT 0.5,
                    review_state TEXT NOT NULL DEFAULT 'candidate',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    archived_at REAL,
                    deleted_at REAL,
                    merged_into_id INTEGER,
                    FOREIGN KEY(merged_into_id) REFERENCES memory_items(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_type ON memory_items(item_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_state ON memory_items(review_state)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_updated ON memory_items(updated_at DESC)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_item_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    excerpt TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE,
                    UNIQUE(memory_item_id, source_type, source_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_sources_item ON memory_sources(memory_item_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_item_id INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    before_json TEXT NOT NULL DEFAULT '{}',
                    after_json TEXT NOT NULL DEFAULT '{}',
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    review_state TEXT NOT NULL DEFAULT 'candidate',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(entity_type, canonical_name)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_item_entities (
                    memory_item_id INTEGER NOT NULL,
                    entity_id INTEGER NOT NULL,
                    role TEXT NOT NULL DEFAULT 'related',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(memory_item_id, entity_id, role),
                    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE,
                    FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entity_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_entity_id INTEGER NOT NULL,
                    memory_item_id INTEGER,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE SET NULL,
                    UNIQUE(source_entity_id, relation_type, target_entity_id, memory_item_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(tag_type, canonical_name)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tag_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_item_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'suggested',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL DEFAULT 'auto',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE CASCADE,
                    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                    UNIQUE(memory_item_id, tag_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tag_assignments_item ON tag_assignments(memory_item_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tag_assignments_tag ON tag_assignments(tag_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tag_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_id INTEGER NOT NULL,
                    memory_item_id INTEGER,
                    action TEXT NOT NULL,
                    previous_status TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE SET NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS capture_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capture_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    turn_id TEXT NOT NULL DEFAULT '',
                    memory_item_id INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(memory_item_id) REFERENCES memory_items(id) ON DELETE SET NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_capture_events_created ON capture_events(created_at DESC)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS context_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT '',
                    turn_id TEXT NOT NULL DEFAULT '',
                    snapshot_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_context_snapshots_created ON context_snapshots(created_at DESC)"
            )

    @staticmethod
    def _validate_item_type(item_type: str) -> str:
        value = (item_type or "").strip().lower()
        if value not in MEMORY_ITEM_TYPES:
            raise ValueError(f"Unsupported memory item type: {item_type}")
        return value

    @staticmethod
    def _validate_review_state(review_state: str) -> str:
        value = (review_state or "").strip().lower()
        if value not in REVIEW_STATES:
            raise ValueError(f"Unsupported review state: {review_state}")
        return value

    @staticmethod
    def _validate_entity_type(entity_type: str) -> str:
        value = (entity_type or "").strip().lower()
        if value not in ENTITY_TYPES:
            raise ValueError(f"Unsupported entity type: {entity_type}")
        return value

    @staticmethod
    def _validate_tag_type(tag_type: str) -> str:
        value = (tag_type or "").strip().lower()
        if value not in TAG_TYPES:
            raise ValueError(f"Unsupported tag type: {tag_type}")
        return value

    @staticmethod
    def _validate_tag_status(status: str) -> str:
        value = (status or "").strip().lower()
        if value not in TAG_STATUSES:
            raise ValueError(f"Unsupported tag status: {status}")
        return value

    @staticmethod
    def _validate_capture_type(capture_type: str) -> str:
        value = (capture_type or "").strip().lower()
        if value not in CAPTURE_TYPES:
            raise ValueError(f"Unsupported capture type: {capture_type}")
        return value

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["metadata"] = _json_load(item.pop("metadata_json", "{}"), {})
        return item

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> dict[str, Any]:
        source = dict(row)
        source["metadata"] = _json_load(source.pop("metadata_json", "{}"), {})
        return source

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> dict[str, Any]:
        entity = dict(row)
        entity["metadata"] = _json_load(entity.pop("metadata_json", "{}"), {})
        return entity

    def get_meta(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM catalog_meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO catalog_meta(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def get_by_external_key(self, external_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM memory_items WHERE external_key = ?",
                (external_key,),
            ).fetchone()
        return self.inspect_item(row["id"]) if row else None

    def upsert_item(
        self,
        *,
        item_type: str,
        content: str,
        review_state: str = "candidate",
        confidence: float = 0.5,
        importance: float = 0.5,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        external_key: str | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        item_type = self._validate_item_type(item_type)
        review_state = self._validate_review_state(review_state)
        content = " ".join((content or "").split()).strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        now = time.time()
        metadata = metadata or {}

        with self._connect() as conn:
            existing = None
            if external_key:
                existing = conn.execute(
                    "SELECT * FROM memory_items WHERE external_key = ?",
                    (external_key,),
                ).fetchone()

            if existing is None:
                cursor = conn.execute(
                    """
                    INSERT INTO memory_items(
                        external_key, item_type, content, summary, confidence,
                        importance, review_state, metadata_json, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        external_key,
                        item_type,
                        content,
                        summary.strip(),
                        max(0.0, min(1.0, float(confidence))),
                        max(0.0, min(1.0, float(importance))),
                        review_state,
                        json.dumps(metadata),
                        now,
                        now,
                    ),
                )
                item_id = int(cursor.lastrowid)
                self._write_history(
                    conn,
                    item_id,
                    "create",
                    {},
                    {"content": content, "review_state": review_state},
                )
            else:
                item_id = int(existing["id"])
                existing_metadata = _json_load(existing["metadata_json"], {})
                existing_metadata.update(metadata)
                existing_state = existing["review_state"]
                locked_state = existing_state in {
                    "corrected",
                    "rejected",
                    "archived",
                    "deleted",
                }
                preserve_confirmation = (
                    existing_state == "confirmed" and review_state == "candidate"
                )
                next_content = (
                    existing["content"]
                    if existing_state == "corrected"
                    else content
                )
                next_state = (
                    existing_state
                    if locked_state or preserve_confirmation
                    else review_state
                )
                next_confidence = max(float(existing["confidence"]), float(confidence))
                content_changed = next_content != existing["content"]
                state_changed = next_state != existing_state
                next_version = int(existing["version"]) + (1 if content_changed else 0)
                conn.execute(
                    """
                    UPDATE memory_items
                    SET item_type = ?, content = ?, summary = ?, confidence = ?,
                        importance = ?, review_state = ?, metadata_json = ?,
                        version = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        item_type,
                        next_content,
                        summary.strip() or existing["summary"],
                        min(1.0, next_confidence),
                        max(float(existing["importance"]), float(importance)),
                        next_state,
                        json.dumps(existing_metadata),
                        next_version,
                        now,
                        item_id,
                    ),
                )
                if content_changed or state_changed:
                    self._write_history(
                        conn,
                        item_id,
                        "sync_update",
                        {
                            "content": existing["content"],
                            "review_state": existing_state,
                        },
                        {"content": next_content, "review_state": next_state},
                        external_key or "",
                    )

        for source in sources or []:
            self.add_source(item_id, **source)
        return self.inspect_item(item_id)

    def _write_history(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        operation: str,
        before: dict[str, Any],
        after: dict[str, Any],
        note: str = "",
    ):
        conn.execute(
            """
            INSERT INTO memory_history(
                memory_item_id, operation, before_json, after_json, note, created_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                operation,
                json.dumps(before),
                json.dumps(after),
                note,
                time.time(),
            ),
        )

    def add_source(
        self,
        item_id: int,
        source_type: str,
        source_id: str,
        excerpt: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_type = (source_type or "unknown").strip().lower()
        source_id = str(source_id or "").strip()
        if not source_id:
            raise ValueError("Source ID cannot be empty")
        with self._connect() as conn:
            if conn.execute(
                "SELECT id FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone() is None:
                raise KeyError(f"Memory item not found: {item_id}")
            conn.execute(
                """
                INSERT INTO memory_sources(
                    memory_item_id, source_type, source_id, excerpt, metadata_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_item_id, source_type, source_id) DO UPDATE SET
                    excerpt = excluded.excerpt,
                    metadata_json = excluded.metadata_json
                """,
                (
                    item_id,
                    source_type,
                    source_id,
                    excerpt.strip(),
                    json.dumps(metadata or {}),
                    time.time(),
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM memory_sources
                WHERE memory_item_id = ? AND source_type = ? AND source_id = ?
                """,
                (item_id, source_type, source_id),
            ).fetchone()
        return self._row_to_source(row)

    def list_items(
        self,
        *,
        query: str = "",
        item_type: str | None = None,
        review_state: str | None = None,
        include_deleted: bool = False,
        entity_id: int | None = None,
        tag: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if item_type:
            clauses.append("item_type = ?")
            params.append(self._validate_item_type(item_type))
        if review_state:
            clauses.append("review_state = ?")
            params.append(self._validate_review_state(review_state))
        elif not include_deleted:
            clauses.append("review_state != 'deleted'")
        if entity_id is not None:
            clauses.append(
                "id IN (SELECT memory_item_id FROM memory_item_entities WHERE entity_id = ?)"
            )
            params.append(int(entity_id))
        if tag:
            # Accepts "name" or "type:name"; rejected assignments never match.
            tag_type, _, tag_name = tag.partition(":")
            if not tag_name:
                tag_type, tag_name = "", tag
            tag_clause = (
                "id IN (SELECT ta.memory_item_id FROM tag_assignments ta "
                "JOIN tags t ON t.id = ta.tag_id "
                "WHERE t.canonical_name = ? AND ta.status != 'rejected'"
            )
            params.append(_canonical_name(tag_name))
            if tag_type:
                tag_clause += " AND t.tag_type = ?"
                params.append(self._validate_tag_type(tag_type))
            clauses.append(tag_clause + ")")
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("created_at <= ?")
            params.append(float(until))
        if query.strip():
            tokens = re.findall(r"[a-z0-9]+", query.lower())[:8]
            if tokens:
                token_clauses = []
                for token in tokens:
                    token_clauses.append("(LOWER(content) LIKE ? OR LOWER(summary) LIKE ?)")
                    params.extend([f"%{token}%", f"%{token}%"])
                clauses.append("(" + " OR ".join(token_clauses) + ")")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.extend([max(1, min(1000, int(limit))), max(0, int(offset))])
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_items{where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
            items = [self._row_to_item(row) for row in rows]
            self._attach_tags(conn, items)
        return items

    @staticmethod
    def _attach_tags(conn: sqlite3.Connection, items: list[dict[str, Any]]):
        if not items:
            return
        ids = [item["id"] for item in items]
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT ta.memory_item_id, t.id AS tag_id, t.tag_type, t.name,
                   ta.status, ta.confidence, ta.source
            FROM tag_assignments ta JOIN tags t ON t.id = ta.tag_id
            WHERE ta.memory_item_id IN ({placeholders})
                AND ta.status NOT IN ('rejected', 'corrected')
            ORDER BY ta.confidence DESC, t.name
            """,
            ids,
        ).fetchall()
        by_item: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_item.setdefault(row["memory_item_id"], []).append(dict(row))
        for item in items:
            item["tags"] = by_item.get(item["id"], [])

    def inspect_item(self, item_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?", (int(item_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"Memory item not found: {item_id}")
            sources = conn.execute(
                "SELECT * FROM memory_sources WHERE memory_item_id = ? ORDER BY created_at",
                (item_id,),
            ).fetchall()
            entities = conn.execute(
                """
                SELECT e.*, mie.role, mie.confidence AS link_confidence
                FROM memory_item_entities mie
                JOIN entities e ON e.id = mie.entity_id
                WHERE mie.memory_item_id = ?
                ORDER BY e.name
                """,
                (item_id,),
            ).fetchall()
            history = conn.execute(
                """
                SELECT id, operation, before_json, after_json, note, created_at
                FROM memory_history WHERE memory_item_id = ? ORDER BY id DESC
                """,
                (item_id,),
            ).fetchall()
            relations = conn.execute(
                """
                SELECT r.*, se.name AS source_name, te.name AS target_name
                FROM relations r
                JOIN entities se ON se.id = r.source_entity_id
                JOIN entities te ON te.id = r.target_entity_id
                WHERE r.memory_item_id = ? ORDER BY r.created_at DESC
                """,
                (item_id,),
            ).fetchall()
        item = self._row_to_item(row)
        item["sources"] = [self._row_to_source(source) for source in sources]
        item["entities"] = [self._row_to_entity(entity) for entity in entities]
        item["history"] = [
            {
                "id": entry["id"],
                "operation": entry["operation"],
                "before": _json_load(entry["before_json"], {}),
                "after": _json_load(entry["after_json"], {}),
                "note": entry["note"],
                "created_at": entry["created_at"],
            }
            for entry in history
        ]
        item["relations"] = []
        for relation in relations:
            value = dict(relation)
            value["metadata"] = _json_load(value.pop("metadata_json", "{}"), {})
            item["relations"].append(value)
        item["tags"] = self.list_item_tags(item_id, include_rejected=True)
        return item

    def correct_item(self, item_id: int, content: str, reason: str = "") -> dict:
        content = " ".join((content or "").split()).strip()
        if not content:
            raise ValueError("Corrected content cannot be empty")
        before = self.inspect_item(item_id)
        if before["review_state"] == "deleted":
            raise ValueError("Deleted memory must be restored before correction")
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_items
                SET content = ?, review_state = 'corrected', version = version + 1,
                    updated_at = ?, archived_at = NULL, deleted_at = NULL
                WHERE id = ?
                """,
                (content, now, item_id),
            )
            self._write_history(
                conn,
                item_id,
                "correct",
                {"content": before["content"], "review_state": before["review_state"]},
                {"content": content, "review_state": "corrected"},
                reason,
            )
        self.add_source(
            item_id,
            "user_correction",
            f"correction:{before['version'] + 1}",
            excerpt=content,
            metadata={"reason": reason},
        )
        return self.inspect_item(item_id)

    def set_review_state(
        self,
        item_id: int,
        review_state: str,
        note: str = "",
    ) -> dict:
        review_state = self._validate_review_state(review_state)
        before = self.inspect_item(item_id)
        now = time.time()
        archived_at = now if review_state == "archived" else None
        deleted_at = now if review_state == "deleted" else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_items
                SET review_state = ?, archived_at = ?, deleted_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (review_state, archived_at, deleted_at, now, item_id),
            )
            self._write_history(
                conn,
                item_id,
                "state_change",
                {"review_state": before["review_state"]},
                {"review_state": review_state},
                note,
            )
        return self.inspect_item(item_id)

    def merge_items(
        self,
        target_id: int,
        source_ids: list[int],
        *,
        content: str | None = None,
        note: str = "",
    ) -> dict:
        target_id = int(target_id)
        source_ids = sorted({int(item_id) for item_id in source_ids if int(item_id) != target_id})
        if not source_ids:
            raise ValueError("Merge requires at least one distinct source item")
        target = self.inspect_item(target_id)
        sources = [self.inspect_item(item_id) for item_id in source_ids]
        now = time.time()
        merged_content = " ".join((content or target["content"]).split()).strip()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_items
                SET content = ?, confidence = ?, importance = ?, review_state = 'corrected',
                    version = version + 1, updated_at = ? WHERE id = ?
                """,
                (
                    merged_content,
                    max([target["confidence"]] + [item["confidence"] for item in sources]),
                    max([target["importance"]] + [item["importance"] for item in sources]),
                    now,
                    target_id,
                ),
            )
            for source_id in source_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_sources(
                        memory_item_id, source_type, source_id, excerpt, metadata_json, created_at
                    )
                    SELECT ?, source_type, source_id, excerpt, metadata_json, created_at
                    FROM memory_sources WHERE memory_item_id = ?
                    """,
                    (target_id, source_id),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_item_entities(
                        memory_item_id, entity_id, role, confidence, created_at
                    )
                    SELECT ?, entity_id, role, confidence, created_at
                    FROM memory_item_entities WHERE memory_item_id = ?
                    """,
                    (target_id, source_id),
                )
                conn.execute(
                    """
                    UPDATE memory_items
                    SET review_state = 'archived', merged_into_id = ?, archived_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (target_id, now, now, source_id),
                )
                self._write_history(
                    conn,
                    source_id,
                    "merge_source",
                    {"review_state": "active"},
                    {"review_state": "archived", "merged_into_id": target_id},
                    note,
                )
            self._write_history(
                conn,
                target_id,
                "merge_target",
                {"content": target["content"]},
                {"content": merged_content, "merged_source_ids": source_ids},
                note,
            )
        return self.inspect_item(target_id)

    def upsert_entity(
        self,
        entity_type: str,
        name: str,
        *,
        description: str = "",
        review_state: str = "candidate",
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        entity_type = self._validate_entity_type(entity_type)
        review_state = self._validate_review_state(review_state)
        name = " ".join((name or "").split()).strip()
        canonical = _canonical_name(name)
        if not canonical:
            raise ValueError("Entity name cannot be empty")
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT review_state FROM entities WHERE entity_type = ? AND canonical_name = ?",
                (entity_type, canonical),
            ).fetchone()
            if existing and (
                existing["review_state"]
                in {"corrected", "rejected", "archived", "deleted"}
                or (
                    existing["review_state"] == "confirmed"
                    and review_state == "candidate"
                )
            ):
                review_state = existing["review_state"]
            conn.execute(
                """
                INSERT INTO entities(
                    entity_type, name, canonical_name, description, review_state,
                    metadata_json, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, canonical_name) DO UPDATE SET
                    name = excluded.name,
                    description = CASE WHEN excluded.description != '' THEN excluded.description ELSE entities.description END,
                    review_state = excluded.review_state,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    entity_type,
                    name,
                    canonical,
                    description.strip(),
                    review_state,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? AND canonical_name = ?",
                (entity_type, canonical),
            ).fetchone()
        return self._row_to_entity(row)

    def list_entities(self, query: str = "", limit: int = 200) -> list[dict]:
        params: list[Any] = []
        where = ""
        if query.strip():
            where = " WHERE canonical_name LIKE ? OR LOWER(description) LIKE ?"
            needle = f"%{_canonical_name(query)}%"
            params.extend([needle, needle])
        params.append(max(1, min(1000, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM entities{where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_entity(row) for row in rows]

    def link_item_entity(
        self,
        item_id: int,
        entity_id: int,
        *,
        role: str = "related",
        confidence: float = 0.5,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_item_entities(
                    memory_item_id, entity_id, role, confidence, created_at
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(memory_item_id, entity_id, role) DO UPDATE SET
                    confidence = excluded.confidence
                """,
                (
                    int(item_id),
                    int(entity_id),
                    (role or "related").strip().lower(),
                    max(0.0, min(1.0, float(confidence))),
                    time.time(),
                ),
            )

    def create_relation(
        self,
        source_entity_id: int,
        relation_type: str,
        target_entity_id: int,
        *,
        memory_item_id: int | None = None,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        relation_type = _canonical_name(relation_type).replace(" ", "_")
        if not relation_type:
            raise ValueError("Relation type cannot be empty")
        with self._connect() as conn:
            item_value = int(memory_item_id) if memory_item_id is not None else None
            row = conn.execute(
                """
                SELECT id FROM relations
                WHERE source_entity_id = ? AND relation_type = ?
                    AND target_entity_id = ? AND memory_item_id IS ?
                """,
                (
                    int(source_entity_id),
                    relation_type,
                    int(target_entity_id),
                    item_value,
                ),
            ).fetchone()
            if row is None:
                cursor = conn.execute(
                    """
                    INSERT INTO relations(
                        source_entity_id, relation_type, target_entity_id,
                        memory_item_id, confidence, metadata_json, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(source_entity_id),
                        relation_type,
                        int(target_entity_id),
                        item_value,
                        max(0.0, min(1.0, float(confidence))),
                        json.dumps(metadata or {}),
                        time.time(),
                    ),
                )
                relation_id = int(cursor.lastrowid)
            else:
                relation_id = int(row["id"])
                conn.execute(
                    "UPDATE relations SET confidence = ?, metadata_json = ? WHERE id = ?",
                    (
                        max(0.0, min(1.0, float(confidence))),
                        json.dumps(metadata or {}),
                        relation_id,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM relations WHERE id = ?",
                (relation_id,),
            ).fetchone()
        result = dict(row)
        result["metadata"] = _json_load(result.pop("metadata_json", "{}"), {})
        return result

    def list_relations(self, limit: int = 1000) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*, se.name AS source_name, se.entity_type AS source_type,
                       te.name AS target_name, te.entity_type AS target_type
                FROM relations r
                JOIN entities se ON se.id = r.source_entity_id
                JOIN entities te ON te.id = r.target_entity_id
                ORDER BY r.created_at DESC LIMIT ?
                """,
                (max(1, min(10000, int(limit))),),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _json_load(item.pop("metadata_json", "{}"), {})
            out.append(item)
        return out

    def upsert_tag(self, tag_type: str, name: str) -> dict[str, Any]:
        tag_type = self._validate_tag_type(tag_type)
        name = " ".join((name or "").split()).strip()
        canonical = _canonical_name(name)
        if not canonical:
            raise ValueError("Tag name cannot be empty")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tags(tag_type, name, canonical_name, created_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(tag_type, canonical_name) DO NOTHING
                """,
                (tag_type, name, canonical, time.time()),
            )
            row = conn.execute(
                "SELECT * FROM tags WHERE tag_type = ? AND canonical_name = ?",
                (tag_type, canonical),
            ).fetchone()
        return dict(row)

    def assign_tag(
        self,
        item_id: int,
        tag_type: str,
        name: str,
        *,
        status: str = "suggested",
        confidence: float = 0.5,
        source: str = "auto",
    ) -> dict[str, Any]:
        status = self._validate_tag_status(status)
        tag = self.upsert_tag(tag_type, name)
        now = time.time()
        with self._connect() as conn:
            if conn.execute(
                "SELECT id FROM memory_items WHERE id = ?", (int(item_id),)
            ).fetchone() is None:
                raise KeyError(f"Memory item not found: {item_id}")
            existing = conn.execute(
                "SELECT * FROM tag_assignments WHERE memory_item_id = ? AND tag_id = ?",
                (int(item_id), tag["id"]),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO tag_assignments(
                        memory_item_id, tag_id, status, confidence, source,
                        created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(item_id),
                        tag["id"],
                        status,
                        max(0.0, min(1.0, float(confidence))),
                        (source or "auto").strip().lower(),
                        now,
                        now,
                    ),
                )
            else:
                # An automated re-suggestion can never override a user decision.
                user_decided = existing["status"] in {"accepted", "corrected", "rejected"}
                if source == "auto" and user_decided:
                    return self._get_item_tag(conn, int(item_id), tag["id"])
                conn.execute(
                    """
                    UPDATE tag_assignments
                    SET status = ?, confidence = ?, source = ?, updated_at = ?
                    WHERE memory_item_id = ? AND tag_id = ?
                    """,
                    (
                        status,
                        max(float(existing["confidence"]), float(confidence)),
                        (source or "auto").strip().lower(),
                        now,
                        int(item_id),
                        tag["id"],
                    ),
                )
            return self._get_item_tag(conn, int(item_id), tag["id"])

    @staticmethod
    def _get_item_tag(conn: sqlite3.Connection, item_id: int, tag_id: int) -> dict[str, Any]:
        row = conn.execute(
            """
            SELECT t.id AS tag_id, t.tag_type, t.name, t.canonical_name,
                   ta.status, ta.confidence, ta.source,
                   ta.created_at, ta.updated_at, ta.memory_item_id
            FROM tag_assignments ta JOIN tags t ON t.id = ta.tag_id
            WHERE ta.memory_item_id = ? AND ta.tag_id = ?
            """,
            (item_id, tag_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Tag assignment not found: item {item_id}, tag {tag_id}")
        return dict(row)

    def set_tag_status(
        self,
        item_id: int,
        tag_id: int,
        status: str,
        note: str = "",
    ) -> dict[str, Any]:
        status = self._validate_tag_status(status)
        now = time.time()
        with self._connect() as conn:
            existing = self._get_item_tag(conn, int(item_id), int(tag_id))
            conn.execute(
                """
                UPDATE tag_assignments SET status = ?, source = 'user', updated_at = ?
                WHERE memory_item_id = ? AND tag_id = ?
                """,
                (status, now, int(item_id), int(tag_id)),
            )
            conn.execute(
                """
                INSERT INTO tag_feedback(
                    tag_id, memory_item_id, action, previous_status, note, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (int(tag_id), int(item_id), status, existing["status"], note, now),
            )
            self._write_history(
                conn,
                int(item_id),
                "tag_change",
                {"tag": existing["name"], "status": existing["status"]},
                {"tag": existing["name"], "status": status},
                note,
            )
            return self._get_item_tag(conn, int(item_id), int(tag_id))

    def correct_tag_assignment(
        self,
        item_id: int,
        tag_id: int,
        tag_type: str,
        name: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Replace a wrong tag with the right one; both sides feed future inference."""
        self.set_tag_status(item_id, tag_id, "corrected", note=note or f"replaced by {name}")
        return self.assign_tag(
            item_id, tag_type, name, status="accepted", confidence=1.0, source="user"
        )

    def list_item_tags(
        self, item_id: int, include_rejected: bool = False
    ) -> list[dict[str, Any]]:
        where = "" if include_rejected else " AND ta.status NOT IN ('rejected', 'corrected')"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT t.id AS tag_id, t.tag_type, t.name, t.canonical_name,
                       ta.status, ta.confidence, ta.source,
                       ta.created_at, ta.updated_at, ta.memory_item_id
                FROM tag_assignments ta JOIN tags t ON t.id = ta.tag_id
                WHERE ta.memory_item_id = ?{where}
                ORDER BY ta.confidence DESC, t.name
                """,
                (int(item_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_tags(self, tag_type: str | None = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if tag_type:
            where = " WHERE t.tag_type = ?"
            params.append(self._validate_tag_type(tag_type))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT t.*, COUNT(ta.id) AS assignment_count
                FROM tags t LEFT JOIN tag_assignments ta
                    ON ta.tag_id = t.id AND ta.status != 'rejected'
                {where}
                GROUP BY t.id ORDER BY assignment_count DESC, t.name
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def tag_feedback_summary(self) -> dict[str, dict[str, int]]:
        """Per-tag accept/reject/correct counts keyed 'tag_type:canonical_name'."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.tag_type, t.canonical_name, tf.action, COUNT(*) AS count
                FROM tag_feedback tf JOIN tags t ON t.id = tf.tag_id
                GROUP BY t.tag_type, t.canonical_name, tf.action
                """
            ).fetchall()
        summary: dict[str, dict[str, int]] = {}
        for row in rows:
            key = f"{row['tag_type']}:{row['canonical_name']}"
            summary.setdefault(key, {})[row["action"]] = row["count"]
        return summary

    def update_item_metadata(self, item_id: int, patch: dict[str, Any]) -> dict[str, Any]:
        """Merge keys into an item's metadata without touching content or review state."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM memory_items WHERE id = ?", (int(item_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"Memory item not found: {item_id}")
            metadata = _json_load(row["metadata_json"], {})
            before = {key: metadata.get(key) for key in patch}
            for key, value in patch.items():
                if value is None:
                    metadata.pop(key, None)
                else:
                    metadata[key] = value
            conn.execute(
                "UPDATE memory_items SET metadata_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata), time.time(), int(item_id)),
            )
            self._write_history(conn, int(item_id), "metadata_update", before, patch)
        return self.inspect_item(int(item_id))

    def flag_contradiction(self, item_id: int, other_id: int, note: str = "") -> None:
        """Mark two items as contradicting each other; neither is collapsed or hidden."""
        item_id, other_id = int(item_id), int(other_id)
        if item_id == other_id:
            raise ValueError("An item cannot contradict itself")
        now = time.time()
        with self._connect() as conn:
            for source, target in ((item_id, other_id), (other_id, item_id)):
                row = conn.execute(
                    "SELECT metadata_json FROM memory_items WHERE id = ?", (source,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Memory item not found: {source}")
                metadata = _json_load(row["metadata_json"], {})
                existing = metadata.get("contradicts") or []
                if target in existing:
                    continue
                metadata["contradicts"] = sorted(existing + [target])
                conn.execute(
                    "UPDATE memory_items SET metadata_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(metadata), now, source),
                )
                self._write_history(
                    conn,
                    source,
                    "contradiction_flag",
                    {"contradicts": existing},
                    {"contradicts": metadata["contradicts"]},
                    note,
                )

    def record_capture_event(
        self,
        capture_type: str,
        content: str,
        *,
        session_id: str = "",
        turn_id: str = "",
        memory_item_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        capture_type = self._validate_capture_type(capture_type)
        content = (content or "").strip()
        if not content:
            raise ValueError("Capture content cannot be empty")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO capture_events(
                    capture_type, content, session_id, turn_id,
                    memory_item_id, metadata_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_type,
                    content,
                    session_id or "",
                    turn_id or "",
                    int(memory_item_id) if memory_item_id is not None else None,
                    json.dumps(metadata or {}),
                    time.time(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM capture_events WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        event = dict(row)
        event["metadata"] = _json_load(event.pop("metadata_json", "{}"), {})
        return event

    def list_capture_events(
        self, *, session_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if session_id:
            where = " WHERE session_id = ?"
            params.append(session_id)
        params.append(max(1, min(1000, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM capture_events{where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["metadata"] = _json_load(event.pop("metadata_json", "{}"), {})
            events.append(event)
        return events

    def record_context_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        session_id: str = "",
        turn_id: str = "",
    ) -> dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO context_snapshots(session_id, turn_id, snapshot_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (session_id or "", turn_id or "", json.dumps(snapshot or {}), time.time()),
            )
            row = conn.execute(
                "SELECT * FROM context_snapshots WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        result = dict(row)
        result["snapshot"] = _json_load(result.pop("snapshot_json", "{}"), {})
        return result

    def recent_context_snapshots(
        self, *, session_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if session_id:
            where = " WHERE session_id = ?"
            params.append(session_id)
        params.append(max(1, min(100, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM context_snapshots{where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        snapshots = []
        for row in rows:
            result = dict(row)
            result["snapshot"] = _json_load(result.pop("snapshot_json", "{}"), {})
            snapshots.append(result)
        return snapshots

    def overview(self) -> dict[str, Any]:
        with self._connect() as conn:
            by_type = conn.execute(
                "SELECT item_type, COUNT(*) AS count FROM memory_items WHERE review_state != 'deleted' GROUP BY item_type"
            ).fetchall()
            by_state = conn.execute(
                "SELECT review_state, COUNT(*) AS count FROM memory_items GROUP BY review_state"
            ).fetchall()
            entity_count = conn.execute("SELECT COUNT(*) AS count FROM entities").fetchone()[
                "count"
            ]
            relation_count = conn.execute(
                "SELECT COUNT(*) AS count FROM relations"
            ).fetchone()["count"]
            tag_count = conn.execute("SELECT COUNT(*) AS count FROM tags").fetchone()["count"]
            suggested_tag_count = conn.execute(
                "SELECT COUNT(*) AS count FROM tag_assignments WHERE status = 'suggested'"
            ).fetchone()["count"]
            capture_count = conn.execute(
                "SELECT COUNT(*) AS count FROM capture_events"
            ).fetchone()["count"]
            contradiction_count = conn.execute(
                """
                SELECT COUNT(*) AS count FROM memory_items
                WHERE review_state != 'deleted' AND metadata_json LIKE '%"contradicts"%'
                """
            ).fetchone()["count"]
        return {
            "by_type": {row["item_type"]: row["count"] for row in by_type},
            "by_state": {row["review_state"]: row["count"] for row in by_state},
            "entity_count": entity_count,
            "relation_count": relation_count,
            "tag_count": tag_count,
            "suggested_tag_count": suggested_tag_count,
            "capture_count": capture_count,
            "contradiction_count": contradiction_count,
        }

    def export(self, export_dir: str) -> dict[str, Any]:
        target = Path(export_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        items = [
            self.inspect_item(item["id"])
            for item in self.list_items(include_deleted=True, limit=100000)
        ]
        entities = self.list_entities(limit=100000)
        relations = self.list_relations(limit=100000)

        def write_jsonl(path: Path, rows: list[dict]):
            with path.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

        write_jsonl(target / "memory_items.jsonl", items)
        write_jsonl(target / "entities.jsonl", entities)
        write_jsonl(target / "relations.jsonl", relations)

        with (target / "memory.md").open("w", encoding="utf-8") as file:
            file.write("# Amtavla Memory Export\n\n")
            for item_type in sorted(MEMORY_ITEM_TYPES):
                group = [item for item in items if item["item_type"] == item_type]
                if not group:
                    continue
                file.write(f"## {item_type.replace('_', ' ').title()}\n\n")
                for item in group:
                    file.write(
                        f"- [{item['review_state']}] #{item['id']}: {item['content']}\n"
                    )
                file.write("\n")
        with (target / "README.md").open("w", encoding="utf-8") as file:
            file.write(
                "# Amtavla Memory Export\n\n"
                "This directory contains newline-delimited JSON for lossless import "
                "and `memory.md` for human-readable review. Deleted records are retained "
                "in JSONL so the audit history remains complete.\n"
            )
        return {
            "directory": str(target),
            "item_count": len(items),
            "entity_count": len(entities),
            "relation_count": len(relations),
            "files": [
                str(target / "README.md"),
                str(target / "memory.md"),
                str(target / "memory_items.jsonl"),
                str(target / "entities.jsonl"),
                str(target / "relations.jsonl"),
            ],
        }

    def clear(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM tag_feedback")
            conn.execute("DELETE FROM tag_assignments")
            conn.execute("DELETE FROM tags")
            conn.execute("DELETE FROM capture_events")
            conn.execute("DELETE FROM context_snapshots")
            conn.execute("DELETE FROM relations")
            conn.execute("DELETE FROM memory_item_entities")
            conn.execute("DELETE FROM entities")
            conn.execute("DELETE FROM memory_history")
            conn.execute("DELETE FROM memory_sources")
            conn.execute("DELETE FROM memory_items")
