import json
import logging
import os
import sqlite3
import time

import numpy as np

logger = logging.getLogger("brain.memory.vector_store")


def _cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(query_norm * matrix_norm, 1e-12)
    return np.dot(matrix, query) / denom


class SQLiteVecStore:
    def __init__(self, db_path: str, embedding_dim: int = 768):
        self._db_path = db_path
        self._embedding_dim = int(embedding_dim)
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._sqlite_vec_enabled = False
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            import sqlite_vec

            sqlite_vec.load(conn)
            self._sqlite_vec_enabled = True
        except Exception:
            self._sqlite_vec_enabled = False
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ltm_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL UNIQUE,
                    text_chunk TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ltm_nodes_updated ON ltm_nodes(updated_at)"
            )

            if self._sqlite_vec_enabled:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS ltm_vectors USING vec0(embedding float[{self._embedding_dim}])"
                )

    @staticmethod
    def _sanitize_embedding(embedding: list[float], embedding_dim: int) -> list[float]:
        vec = [float(x) for x in (embedding or [])]
        if len(vec) >= embedding_dim:
            return vec[:embedding_dim]
        return vec + [0.0] * (embedding_dim - len(vec))

    def upsert_node(
        self,
        node_id: str,
        text_chunk: str,
        embedding: list[float],
        metadata: dict,
    ):
        now = time.time()
        vec = self._sanitize_embedding(embedding, self._embedding_dim)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, created_at FROM ltm_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO ltm_nodes(node_id, text_chunk, embedding_json, metadata_json, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        text_chunk,
                        json.dumps(vec),
                        json.dumps(metadata),
                        now,
                        now,
                    ),
                )
                doc_id = conn.execute(
                    "SELECT id FROM ltm_nodes WHERE node_id = ?", (node_id,)
                ).fetchone()["id"]
            else:
                doc_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE ltm_nodes
                    SET text_chunk = ?, embedding_json = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (text_chunk, json.dumps(vec), json.dumps(metadata), now, doc_id),
                )

            if self._sqlite_vec_enabled:
                try:
                    conn.execute("DELETE FROM ltm_vectors WHERE rowid = ?", (doc_id,))
                    conn.execute(
                        "INSERT INTO ltm_vectors(rowid, embedding) VALUES(?, ?)",
                        (doc_id, json.dumps(vec)),
                    )
                except Exception:
                    logger.debug(
                        "sqlite-vec insert failed; using scan fallback", exc_info=True
                    )

    def delete_node(self, node_id: str):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM ltm_nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row is None:
                return
            doc_id = int(row["id"])
            conn.execute("DELETE FROM ltm_nodes WHERE id = ?", (doc_id,))
            if self._sqlite_vec_enabled:
                conn.execute("DELETE FROM ltm_vectors WHERE rowid = ?", (doc_id,))

    def query_knn(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        top_k = max(1, int(top_k))
        qvec = self._sanitize_embedding(query_embedding, self._embedding_dim)
        metadata_filter = metadata_filter or {}

        with self._connect() as conn:
            if self._sqlite_vec_enabled:
                try:
                    vec_rows = conn.execute(
                        "SELECT rowid, distance FROM ltm_vectors WHERE embedding MATCH ? AND k = ?",
                        (json.dumps(qvec), top_k * 4),
                    ).fetchall()
                    if vec_rows:
                        candidates = []
                        for row in vec_rows:
                            doc = conn.execute(
                                "SELECT node_id, text_chunk, metadata_json FROM ltm_nodes WHERE id = ?",
                                (int(row["rowid"]),),
                            ).fetchone()
                            if doc is None:
                                continue
                            metadata = _json_load(doc["metadata_json"], {})
                            if not self._metadata_match(metadata, metadata_filter):
                                continue
                            distance = float(row["distance"])
                            score = max(0.0, 1.0 - distance)
                            candidates.append(
                                {
                                    "node_id": doc["node_id"],
                                    "text_chunk": doc["text_chunk"],
                                    "metadata": metadata,
                                    "score": round(score, 6),
                                }
                            )
                        if candidates:
                            return candidates[:top_k]
                except Exception:
                    logger.debug(
                        "sqlite-vec query failed; using scan fallback", exc_info=True
                    )

            rows = conn.execute(
                "SELECT node_id, text_chunk, embedding_json, metadata_json FROM ltm_nodes"
            ).fetchall()

        if not rows:
            return []

        embeddings = []
        docs = []
        for row in rows:
            metadata = _json_load(row["metadata_json"], {})
            if not self._metadata_match(metadata, metadata_filter):
                continue
            vec = np.asarray(
                self._sanitize_embedding(
                    _json_load(row["embedding_json"], []), self._embedding_dim
                ),
                dtype=np.float32,
            )
            embeddings.append(vec)
            docs.append((row["node_id"], row["text_chunk"], metadata))

        if not embeddings:
            return []

        matrix = np.vstack(embeddings)
        sims = _cosine_similarity_matrix(np.asarray(qvec, dtype=np.float32), matrix)
        order = np.argsort(-sims)[:top_k]
        out = []
        for idx in order:
            node_id, text_chunk, metadata = docs[int(idx)]
            out.append(
                {
                    "node_id": node_id,
                    "text_chunk": text_chunk,
                    "metadata": metadata,
                    "score": float(sims[int(idx)]),
                }
            )
        return out

    def clear(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM ltm_nodes")
            if self._sqlite_vec_enabled:
                conn.execute("DELETE FROM ltm_vectors")

    @staticmethod
    def _metadata_match(metadata: dict, metadata_filter: dict) -> bool:
        for key, expected in metadata_filter.items():
            if metadata.get(key) != expected:
                return False
        return True


def _json_load(value: str, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default
