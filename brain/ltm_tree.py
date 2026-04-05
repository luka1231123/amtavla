import json
import logging
import os
import uuid
from collections import deque
from functools import lru_cache

import numpy as np
import ollama
from filelock import FileLock

logger = logging.getLogger("brain.ltm_tree")

EMBEDDING_MODEL = "nomic-embed-text"
TREE_FILE = "brain/ltm_tree.json"
MAX_DEPTH = 4
RETRIEVAL_THRESHOLD = 0.3
MERGE_THRESHOLD = 0.85
FULL_TREE_MERGE_THRESHOLD = 0.75
MAX_CONTENT_PER_BRANCH = 50
EMBEDDING_DIM = 768


def _safe_embed(text: str) -> list[float]:
    return _embed_cached(text)


@lru_cache(maxsize=512)
def _embed_cached(text: str) -> tuple[float, ...]:
    try:
        resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return tuple(resp["embedding"])
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return tuple([0.0] * EMBEDDING_DIM)


def _safe_chat(messages: list[dict], model: str = "qwen2.5-coder:1.5b") -> str:
    try:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def _branch_embedding_text(branch: dict) -> str:
    return branch["topic"] + " " + " ".join(branch["content"])


class LtmTree:
    def __init__(self, tree_file: str | None = None):
        self.branches: list[dict] = []
        self._tree_file = tree_file or TREE_FILE
        self._index: dict[str, dict] = {}

    def _rebuild_index(self):
        self._index.clear()
        for node in self._collect_all_nodes():
            self._index[node["id"]] = node

    def load(self):
        if os.path.exists(self._tree_file):
            lock = FileLock(f"{self._tree_file}.lock")
            with lock:
                try:
                    with open(self._tree_file, "r") as f:
                        content = f.read().strip()
                        if not content:
                            self.branches = []
                            return
                        data = json.loads(content)
                        self.branches = data.get("branches", [])
                except (json.JSONDecodeError, ValueError):
                    self.branches = []
        self._rebuild_index()

    def save(self):
        lock = FileLock(f"{self._tree_file}.lock")
        with lock:
            with open(self._tree_file, "w") as f:
                json.dump({"branches": self.branches}, f, indent=2)

    def _find_branch(
        self, branch_id: str, nodes: list[dict] | None = None
    ) -> dict | None:
        return self._index.get(branch_id)

    def _collect_all_nodes(self, nodes: list[dict] | None = None) -> list[dict]:
        if nodes is None:
            nodes = self.branches
        result = []
        stack = list(nodes)
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(reversed(node["children"]))
        return result

    def _collect_subbranch_with_depths(
        self, branch: dict, depth: int = 0
    ) -> list[tuple[dict, int]]:
        result = []
        stack = [(branch, depth)]
        while stack:
            node, d = stack.pop()
            result.append((node, d))
            stack.extend((c, d + 1) for c in reversed(node["children"]))
        return result

    def _depth_of(self, branch_id: str) -> int:
        queue = deque()
        for node in self.branches:
            queue.append((node, 0))
        while queue:
            node, depth = queue.popleft()
            if node["id"] == branch_id:
                return depth
            for child in node["children"]:
                queue.append((child, depth + 1))
        return -1

    def _parent_of(self, branch_id: str) -> dict | None:
        for node_id, node in self._index.items():
            if node_id == branch_id:
                continue
            for child in node["children"]:
                if child["id"] == branch_id:
                    return node
        return None

    def _siblings_of(self, branch_id: str) -> list[dict]:
        parent = self._parent_of(branch_id)
        if parent is None:
            return list(self.branches)
        return list(parent["children"])

    def find_best_branch(
        self,
        embedding: list[float],
        threshold: float = RETRIEVAL_THRESHOLD,
        exclude_id: str | None = None,
    ) -> dict | None:
        best = None
        best_score = threshold
        for node in self._collect_all_nodes():
            if exclude_id and node["id"] == exclude_id:
                continue
            score = _cosine_similarity(embedding, node["embedding"])
            if score > best_score:
                best_score = score
                best = node
        return best

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        embedding = _safe_embed(query)
        candidates = []
        for node in self._collect_all_nodes():
            score = _cosine_similarity(embedding, node["embedding"])
            if score > RETRIEVAL_THRESHOLD:
                candidates.append((score, node))
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:top_k]

        if not candidates:
            return ""

        parts = []
        seen_ids = set()
        for score, branch in candidates:
            if branch["id"] in seen_ids:
                continue
            subbranch = self._collect_subbranch_with_depths(branch)
            lines = []
            for node, depth in subbranch:
                seen_ids.add(node["id"])
                indent = "  " * depth
                lines.append(f"{indent}## {node['topic']}")
                for c in node["content"]:
                    lines.append(f"{indent}  - {c}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def add_branch(
        self, topic: str, content: list[str], parent_id: str | None = None
    ) -> dict | None:
        content_text = " ".join(content)
        embedding = _safe_embed(topic + " " + content_text)
        branch = {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "embedding": list(embedding),
            "content": list(content),
            "children": [],
        }

        if parent_id is None:
            self.branches.append(branch)
            self._index[branch["id"]] = branch
            return branch

        parent = self._find_branch(parent_id)
        if parent is None:
            return None
        if self._depth_of(parent_id) >= MAX_DEPTH - 1:
            return None
        parent["children"].append(branch)
        self._index[branch["id"]] = branch
        return branch

    def append_to_branch(self, branch_id: str, content_items: list[str]) -> bool:
        branch = self._find_branch(branch_id)
        if branch is None:
            return False
        branch["content"].extend(content_items)
        if len(branch["content"]) > MAX_CONTENT_PER_BRANCH:
            branch["content"] = branch["content"][-MAX_CONTENT_PER_BRANCH:]
        branch["embedding"] = list(_safe_embed(_branch_embedding_text(branch)))
        return True

    def update_branch_topic(self, branch_id: str, new_topic: str) -> bool:
        branch = self._find_branch(branch_id)
        if branch is None:
            return False
        branch["topic"] = new_topic
        branch["embedding"] = list(_safe_embed(_branch_embedding_text(branch)))
        return True

    def merge_branches(self, keep_id: str, merge_id: str) -> bool:
        keep = self._find_branch(keep_id)
        merge = self._find_branch(merge_id)
        if keep is None or merge is None or keep_id == merge_id:
            return False

        keep["content"].extend(merge["content"])
        for child in merge["children"]:
            if child["id"] not in [c["id"] for c in keep["children"]]:
                keep["children"].append(child)
        keep["topic"] = _merge_topic_name(keep["topic"], merge["topic"])
        keep["embedding"] = list(_safe_embed(_branch_embedding_text(keep)))

        self._remove_branch(merge_id)
        return True

    def _remove_branch(self, branch_id: str) -> bool:
        self._index.pop(branch_id, None)
        for i, node in enumerate(self.branches):
            if node["id"] == branch_id:
                self.branches.pop(i)
                return True
            if self._remove_from_children(node, branch_id):
                return True
        return False

    def _remove_from_children(self, parent: dict, branch_id: str) -> bool:
        for i, child in enumerate(parent["children"]):
            if child["id"] == branch_id:
                self._index.pop(branch_id, None)
                parent["children"].pop(i)
                return True
            if self._remove_from_children(child, branch_id):
                return True
        return False

    def check_and_merge_siblings(self, branch_id: str):
        queue = deque([branch_id])
        visited = set()
        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            while True:
                siblings = self._siblings_of(current_id)
                if len(siblings) < 2:
                    break
                merged = False
                i = 0
                while i < len(siblings):
                    j = i + 1
                    while j < len(siblings):
                        sim = _cosine_similarity(
                            siblings[i]["embedding"], siblings[j]["embedding"]
                        )
                        if sim > MERGE_THRESHOLD:
                            self.merge_branches(siblings[i]["id"], siblings[j]["id"])
                            merged = True
                            break
                        j += 1
                    if merged:
                        break
                    i += 1
                if not merged:
                    break

            for sib in self._siblings_of(current_id):
                if sib["children"] and sib["id"] not in visited:
                    queue.append(sib["id"])

    def full_tree_merge(self):
        changed = True
        while changed:
            changed = False
            nodes = self._collect_all_nodes()
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    a, b = nodes[i], nodes[j]
                    if a["id"] not in self._index or b["id"] not in self._index:
                        continue
                    if self._parent_of(b["id"]) is a or self._parent_of(a["id"]) is b:
                        continue
                    sim = _cosine_similarity(a["embedding"], b["embedding"])
                    if sim > FULL_TREE_MERGE_THRESHOLD:
                        self.merge_branches(a["id"], b["id"])
                        changed = True
                        break
                if changed:
                    break

    def visualize(self) -> str:
        lines = []
        self._visualize_node(self.branches, 0, lines)
        if not lines:
            return "(empty)"
        return "\n".join(lines)

    def _visualize_node(self, nodes: list[dict], depth: int, lines: list[str]):
        for node in nodes:
            indent = "  " * depth
            content_preview = "; ".join(node["content"][:3])
            if len(node["content"]) > 3:
                content_preview += f" (+{len(node['content']) - 3} more)"
            lines.append(f"{indent}├── {node['topic']}")
            lines.append(f"{indent}│   [{content_preview}]")
            self._visualize_node(node["children"], depth + 1, lines)

    def get_most_active_branch(self) -> dict | None:
        best = None
        best_score = -1
        for node in self._collect_all_nodes():
            score = len(node["content"])
            if score > best_score:
                best_score = score
                best = node
        return best


def _merge_topic_name(a: str, b: str) -> str:
    if len(a) >= len(b):
        return a
    return b
