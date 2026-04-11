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

    def clear(self):
        self._store.clear()

    @staticmethod
    def _node_id(insight_id: int) -> str:
        return f"insight:{int(insight_id)}"
