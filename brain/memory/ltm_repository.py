import time

from brain.memory.vector_store import SQLiteVecStore


class LtmRepository:
    def __init__(self, store: SQLiteVecStore, embed_fn):
        self._store = store
        self._embed = embed_fn

    def upsert_insight_node(
        self,
        insight_id: int,
        thesis: str,
        confidence: float,
        novelty_score: float,
        status: str,
        rationale: str = "",
    ):
        text = (thesis or "").strip()
        if not text:
            return
        vec = self._embed(text)
        metadata = {
            "kind": "insight",
            "insight_id": int(insight_id),
            "confidence": float(confidence),
            "novelty_score": float(novelty_score),
            "status": status,
            "rationale": rationale or "",
            "updated_at": time.time(),
        }
        self._store.upsert_node(
            node_id=self._node_id(insight_id),
            text_chunk=text,
            embedding=vec,
            metadata=metadata,
        )

    def search_insights(
        self, query: str, top_k: int = 5, status: str | None = "promoted"
    ) -> list[dict]:
        text = (query or "").strip()
        if not text:
            return []
        vec = self._embed(text)
        metadata_filter = {"kind": "insight"}
        if status:
            metadata_filter["status"] = status
        return self._store.query_knn(
            query_embedding=vec,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def upsert_memory_item_node(self, item: dict):
        content = (item.get("content") or "").strip()
        if not content:
            return
        item_id = int(item["id"])
        metadata = {
            "kind": "memory_item",
            "memory_item_id": item_id,
            "item_type": item.get("item_type", "fact"),
            "review_state": item.get("review_state", "candidate"),
            "confidence": float(item.get("confidence", 0.5)),
            "importance": float(item.get("importance", 0.5)),
            "updated_at": float(item.get("updated_at", time.time())),
        }
        self._store.upsert_node(
            node_id=f"memory_item:{item_id}",
            text_chunk=content,
            embedding=self._embed(content),
            metadata=metadata,
        )

    def delete_memory_item_node(self, item_id: int):
        self._store.delete_node(f"memory_item:{int(item_id)}")

    def search_memory_items(self, query: str, top_k: int = 8) -> list[dict]:
        text = (query or "").strip()
        if not text:
            return []
        return self._store.query_knn(
            query_embedding=self._embed(text),
            top_k=top_k,
            metadata_filter={"kind": "memory_item"},
        )

    def upsert_entity_node(self, entity: dict):
        name = (entity.get("name") or "").strip()
        if not name:
            return
        entity_id = int(entity["id"])
        description = (entity.get("description") or "").strip()
        text = f"{name}. {description}".strip()
        self._store.upsert_node(
            node_id=f"entity:{entity_id}",
            text_chunk=text,
            embedding=self._embed(text),
            metadata={
                "kind": "entity",
                "entity_id": entity_id,
                "entity_type": entity.get("entity_type", "other"),
                "review_state": entity.get("review_state", "candidate"),
                "updated_at": float(entity.get("updated_at", time.time())),
            },
        )

    def delete_entity_node(self, entity_id: int):
        self._store.delete_node(f"entity:{int(entity_id)}")

    def clear(self):
        self._store.clear()

    @staticmethod
    def _node_id(insight_id: int) -> str:
        return f"insight:{int(insight_id)}"
