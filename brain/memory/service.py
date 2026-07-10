import json
import os
import re
import sqlite3
import time

import llama_client
from brain.config import load_brain_config
from brain.memory.catalog import MemoryCatalog
from brain.memory.commitments import extract_commitments
from brain.memory.context_engine import ContextEngine
from brain.memory.ltm_repository import LtmRepository
from brain.memory.tagging import TagEngine
from brain.memory.vector_store import SQLiteVecStore
from tools.websearch import render_search_results, search_web, tool_websearch

STYLE_INSTRUCTION_PATTERNS = [
    (re.compile(r"\b(?:be|keep it|make it|keep answers?|answer)\s+(?:shorter|brief|concise|short)\b", re.IGNORECASE), "Keep answers short and concise."),
    (re.compile(r"\buse bullet points\b", re.IGNORECASE), "Prefer bullet points for lists."),
    (re.compile(r"\b(?:no|don'?t use|stop using)\s+emoji\b", re.IGNORECASE), "Never use emoji."),
    (re.compile(r"\bmore formal\b", re.IGNORECASE), "Use a formal tone."),
    (re.compile(r"\bmore casual\b", re.IGNORECASE), "Use a casual tone."),
    (re.compile(r"\bno markdown\b", re.IGNORECASE), "Do not use markdown formatting."),
    (re.compile(r"^(?:please )?(always [^,.!?]{4,60})", re.IGNORECASE), None),
    (re.compile(r"^(?:please )?(never [^,.!?]{4,60})", re.IGNORECASE), None),
]

# Specific property patterns first: "my car is parked at X" must resolve to
# the "parked" property, not the generic "my car".
_CONTRADICTION_PATTERNS = [
    (
        re.compile(r"\b(?:i )?parked(?: the car| my car)? (?:at|in|on|near) (.+)", re.IGNORECASE),
        lambda m: ("parked", m.group(1)),
    ),
    (
        re.compile(r"\bi live (?:at|in) (.+)", re.IGNORECASE),
        lambda m: ("live", m.group(1)),
    ),
    (
        re.compile(r"\bi work (?:at|in|for) (.+)", re.IGNORECASE),
        lambda m: ("work", m.group(1)),
    ),
    (
        re.compile(r"\bi prefer (.+)", re.IGNORECASE),
        lambda m: ("prefer", m.group(1)),
    ),
    (
        re.compile(r"\bmy ([a-z]+(?: [a-z]+)?) (?:is|are|was|will be) (.+)", re.IGNORECASE),
        lambda m: (f"my {m.group(1).lower()}", m.group(2)),
    ),
]


def _extract_property(content: str) -> tuple[str, str] | None:
    for pattern, builder in _CONTRADICTION_PATTERNS:
        match = pattern.search(content or "")
        if match:
            prop, value = builder(match)
            return prop, value.strip().rstrip(".")
    return None


def _time_window_from_query(text: str, now: float) -> tuple[float | None, float | None]:
    import datetime as _dt

    lowered = (text or "").lower()
    today = _dt.date.fromtimestamp(now)

    def day_bounds(day):
        start = _dt.datetime.combine(day, _dt.time.min).timestamp()
        return start, start + 86400

    if "yesterday" in lowered:
        return day_bounds(today - _dt.timedelta(days=1))
    if "today" in lowered:
        return day_bounds(today)
    if re.search(r"\b(?:last|past|this) week\b", lowered):
        return now - 7 * 86400, None
    return None, None


NOISE_PATTERNS = [
    re.compile(r"^please confirm if it feels correct", re.IGNORECASE),
    re.compile(r"^i discovered a candidate insight", re.IGNORECASE),
    re.compile(r"^quick memory check", re.IGNORECASE),
    re.compile(r"^=== brain dump ===", re.IGNORECASE),
    re.compile(r"^status:\s*\{", re.IGNORECASE),
    re.compile(r"^how can i assist you today", re.IGNORECASE),
    re.compile(r"^not much, just here to help", re.IGNORECASE),
]

MEMORY_LIKE_PATTERNS = [
    re.compile(
        r"\b(my car|where is my|where did i park|i parked|parking lot)\b", re.IGNORECASE
    ),
    re.compile(r"\bremember (that|this)\b", re.IGNORECASE),
]

ASSISTANT_MEMORY_CONFIRM_PATTERNS = [
    re.compile(r"\bnoted\b", re.IGNORECASE),
    re.compile(r"\bremember(ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bgot it\b", re.IGNORECASE),
    re.compile(r"\byour\b.*\b(parked|deadline|preference|prefer)\b", re.IGNORECASE),
]

POLLUTED_FACT_PATTERNS = [
    re.compile(r"\*\*"),
    re.compile(r"`[^`]+`"),
    re.compile(r"\{[^}]*\}"),
    re.compile(r"^here('|’)s", re.IGNORECASE),
    re.compile(r"^for example", re.IGNORECASE),
    re.compile(r"^quick memory check", re.IGNORECASE),
    re.compile(r"compact memory notes that need to be rewritten", re.IGNORECASE),
    re.compile(r"how can i assist you today", re.IGNORECASE),
]

QUESTION_START_WORDS = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
}

IMPERATIVE_VERBS = {
    "make",
    "list",
    "show",
    "tell",
    "give",
    "ask",
    "set",
    "create",
    "explain",
}

FIRST_PERSON_MARKERS = {"i", "me", "my", "mine", "we", "our", "us"}

UNCERTAIN_LEADS = {"if", "maybe", "might", "perhaps", "possibly"}

MEMORY_DIRECTIVE_PREFIXES = (
    "remember this:",
    "remember this",
    "remember that:",
    "remember that",
    "don't forget:",
    "dont forget:",
    "don't forget",
    "dont forget",
    "save this:",
    "save this",
    "keep this in mind:",
    "keep this in mind",
    "memorize:",
    "memorize",
)


def _now_ts() -> float:
    return time.time()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _json_load(value: str, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _is_noise_text(text: str) -> bool:
    value = (text or "").strip()
    value = value.lstrip("-;:,. ")
    if not value:
        return True
    if value.startswith("/"):
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.search(value):
            return True
    return False


def _is_memory_like_text(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    return any(pattern.search(value) for pattern in MEMORY_LIKE_PATTERNS)


def _is_polluted_statement(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    for pattern in POLLUTED_FACT_PATTERNS:
        if pattern.search(value):
            return True
    if len(value) < 8:
        return True
    return False


def _normalize_fact_text(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip())
    value = value.strip(" -;:,.\n")
    return value


def _strip_memory_directive(text: str) -> str:
    value = _normalize_fact_text(text)
    lowered = value.lower()
    for prefix in MEMORY_DIRECTIVE_PREFIXES:
        if lowered.startswith(prefix):
            return _normalize_fact_text(value[len(prefix) :])
    return value


def _has_memory_directive(text: str) -> bool:
    lowered = _normalize_fact_text(text).lower()
    return any(lowered.startswith(prefix) for prefix in MEMORY_DIRECTIVE_PREFIXES)


def _infer_speech_act(text: str) -> str:
    value = _normalize_fact_text(text)
    lowered = value.lower()
    tokens = re.findall(r"[a-z0-9']+", lowered)
    if not tokens:
        return "other"

    first = tokens[0]
    if "?" in value:
        return "question"
    if first in QUESTION_START_WORDS and len(tokens) >= 3:
        return "question"
    if first in {"thanks", "thank", "hi", "hello", "hey", "yo"}:
        return "social"
    if first in IMPERATIVE_VERBS:
        return "command"
    if len(tokens) > 1 and tokens[0] == "please" and tokens[1] in IMPERATIVE_VERBS:
        return "command"
    if any(token in IMPERATIVE_VERBS for token in tokens[:3]) and not (
        "i" in tokens[:2] or "we" in tokens[:2]
    ):
        return "command"
    return "assertion"


def _should_store_user_fact(text: str) -> bool:
    value = _strip_memory_directive(text)
    if not value:
        return False
    if _infer_speech_act(value) != "assertion":
        return False
    tokens = re.findall(r"[a-z0-9']+", value.lower())
    if len(tokens) < 4 or len(tokens) > 28:
        return False
    if tokens and tokens[0] in UNCERTAIN_LEADS:
        return False
    if value.startswith("/"):
        return False
    if _is_noise_text(value) or _is_polluted_statement(value):
        return False
    if not any(token in FIRST_PERSON_MARKERS for token in tokens):
        return False
    return True


def _should_store_explicit_fact(text: str) -> bool:
    value = _normalize_fact_text(text)
    if not value:
        return False
    tokens = re.findall(r"[a-z0-9']+", value.lower())
    if len(tokens) < 3 or len(tokens) > 40:
        return False
    if tokens and tokens[0] in UNCERTAIN_LEADS:
        return False
    if value.startswith("/"):
        return False
    if _is_noise_text(value) or _is_polluted_statement(value):
        return False
    return True


class MemoryService:
    def __init__(self, db_dir: str = "brain/db"):
        self._config = load_brain_config()
        self._db_dir = db_dir
        os.makedirs(self._db_dir, exist_ok=True)

        self._episodic_db = os.path.join(self._db_dir, "episodic.db")
        self._semantic_db = os.path.join(self._db_dir, "semantic.db")
        self._insight_db = os.path.join(self._db_dir, "insight_ltm.db")
        self._jobs_db = os.path.join(self._db_dir, "jobs.db")

        memory_cfg = self._config.get("memory", {})
        self._episodic_ttl_seconds = float(
            memory_cfg.get("episodic_ttl_seconds", 7 * 86400)
        )
        self._recall_extension_seconds = float(
            memory_cfg.get("episodic_recall_extension_seconds", 2 * 86400)
        )
        self._strength_alpha = float(memory_cfg.get("episodic_strength_alpha", 0.25))
        self._strength_decay = float(memory_cfg.get("episodic_strength_decay", 0.02))
        self._proactive_turn_gap = int(memory_cfg.get("proactive_turn_gap", 8))
        self._proactive_seconds_gap = float(
            memory_cfg.get("proactive_seconds_gap", 600)
        )
        self._proactive_max_asks = int(memory_cfg.get("proactive_max_asks", 2))
        self._proactive_snooze_seconds = float(
            memory_cfg.get("proactive_snooze_seconds", 900)
        )
        self._embedding_dim = int(memory_cfg.get("embedding_dim", 768))
        vector_db_path = memory_cfg.get(
            "vector_db_path", os.path.join(self._db_dir, "ltm_vectors.db")
        )
        catalog_db_path = memory_cfg.get(
            "catalog_db_path", os.path.join(self._db_dir, "memory_catalog.db")
        )
        self._vector_top_k = int(memory_cfg.get("vector_top_k", 5))

        self._turn_counter = 0
        self._last_proactive_turn = -99999
        self._last_proactive_ts = 0.0
        self._active_asked_insight_id = None
        self._last_route_intent = ""
        self._last_route_pathway = ""
        self._embedding_available = None
        self._embedding_last_error = ""
        self._last_catalog_vector_sync = 0.0

        self.catalog = MemoryCatalog(catalog_db_path)
        self.tag_engine = TagEngine(self.catalog, memory_cfg.get("tagging", {}))
        self.context_engine = ContextEngine(self.catalog, memory_cfg.get("context", {}))
        self._focus_until = 0.0
        self._reminder_gap_seconds = float(
            memory_cfg.get("reminder_gap_seconds", 6 * 3600)
        )

        self._ltm_store = SQLiteVecStore(
            db_path=vector_db_path,
            embedding_dim=self._embedding_dim,
        )
        self._ltm_repo = LtmRepository(self._ltm_store, self._embed_text)

        self._init_schema()
        self.cleanup_polluted_memory()
        self._migrate_legacy_memory()
        self._reconcile_catalog_vectors(force=True)

    def _embed_text(self, text: str) -> list[float]:
        try:
            response = llama_client.embed(text)
            embedding = response.get("embedding", [])
            if isinstance(embedding, list) and embedding:
                vec = [float(x) for x in embedding]
                if not any(abs(x) > 1e-12 for x in vec):
                    self._embedding_available = False
                    self._embedding_last_error = (
                        "embedding provider returned a zero vector"
                    )
                else:
                    self._embedding_available = True
                    self._embedding_last_error = ""
                if len(vec) >= self._embedding_dim:
                    return vec[: self._embedding_dim]
                return vec + [0.0] * (self._embedding_dim - len(vec))
        except Exception as exc:
            self._embedding_available = False
            self._embedding_last_error = str(exc)
        return [0.0] * self._embedding_dim

    def _connect(self, db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect(self._episodic_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    session_id TEXT,
                    user_input TEXT NOT NULL,
                    response TEXT NOT NULL,
                    intent TEXT,
                    todo_json TEXT,
                    context_json TEXT,
                    error_text TEXT,
                    strength REAL NOT NULL DEFAULT 0.20,
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    last_recalled_at REAL,
                    expiry_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recall_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    query_text TEXT,
                    score_before REAL,
                    score_after REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_expiry ON events(expiry_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)"
            )

        with self._connect(self._semantic_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statement TEXT NOT NULL,
                    canonical_key TEXT NOT NULL UNIQUE,
                    confidence REAL NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    provenance_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_last_seen ON facts(last_seen)"
            )

        with self._connect(self._insight_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thesis TEXT NOT NULL,
                    rationale TEXT,
                    evidence_json TEXT,
                    novelty_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    feedback_state TEXT NOT NULL,
                    ask_count INTEGER NOT NULL DEFAULT 0,
                    last_asked_at REAL,
                    snoozed_until REAL,
                    quality_score REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_insights_status ON insights(status)"
            )

            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(insights)").fetchall()
            }
            if "ask_count" not in cols:
                conn.execute(
                    "ALTER TABLE insights ADD COLUMN ask_count INTEGER NOT NULL DEFAULT 0"
                )
            if "last_asked_at" not in cols:
                conn.execute("ALTER TABLE insights ADD COLUMN last_asked_at REAL")
            if "snoozed_until" not in cols:
                conn.execute("ALTER TABLE insights ADD COLUMN snoozed_until REAL")
            if "quality_score" not in cols:
                conn.execute(
                    "ALTER TABLE insights ADD COLUMN quality_score REAL NOT NULL DEFAULT 0.0"
                )

        with self._connect(self._jobs_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    payload_json TEXT,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    finished_at REAL,
                    error_text TEXT
                )
                """
            )

    @staticmethod
    def _classify_memory_item_type(content: str, fallback: str = "fact") -> str:
        if fallback in {"episode", "insight", "source_excerpt"}:
            return fallback
        text = (content or "").lower()
        if re.search(r"\b(i prefer|my preference|favorite|i like)\b", text):
            return "preference"
        if re.search(r"\b(i decided|we decided|decision|chose to)\b", text):
            return "decision"
        if re.search(
            r"\b(i need to|i must|i will|deadline|meeting|appointment|promise|due)\b",
            text,
        ):
            return "commitment"
        if re.search(r"\b(idea|concept|what if)\b", text):
            return "idea"
        return fallback

    def _migrate_legacy_memory(self):
        if self.catalog.get_meta("legacy_migration_v1") == "complete":
            return

        with self._connect(self._semantic_db) as conn:
            facts = conn.execute(
                "SELECT id, statement, confidence, provenance_json, first_seen, last_seen FROM facts"
            ).fetchall()
        for row in facts:
            provenance = _json_load(row["provenance_json"], [])
            sources = [
                {
                    "source_type": "legacy_fact_provenance",
                    "source_id": f"semantic:{row['id']}:{index}",
                    "excerpt": row["statement"],
                    "metadata": entry if isinstance(entry, dict) else {},
                }
                for index, entry in enumerate(provenance, 1)
            ]
            if not sources:
                sources = [
                    {
                        "source_type": "legacy_fact",
                        "source_id": str(row["id"]),
                        "excerpt": row["statement"],
                    }
                ]
            self.catalog.upsert_item(
                item_type=self._classify_memory_item_type(row["statement"]),
                content=row["statement"],
                review_state="confirmed",
                confidence=float(row["confidence"]),
                external_key=f"semantic_fact:{row['id']}",
                metadata={
                    "legacy_fact_id": row["id"],
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                },
                sources=sources,
            )

        with self._connect(self._episodic_db) as conn:
            episodes = conn.execute(
                "SELECT id, ts, session_id, user_input, response, intent FROM events"
            ).fetchall()
        for row in episodes:
            self.catalog.upsert_item(
                item_type="episode",
                content=f"User: {row['user_input']} Assistant: {row['response']}",
                review_state="confirmed",
                confidence=0.65,
                importance=0.4,
                external_key=f"episode:{row['id']}",
                metadata={
                    "event_id": row["id"],
                    "session_id": row["session_id"],
                    "intent": row["intent"],
                    "timestamp": row["ts"],
                },
                sources=[
                    {
                        "source_type": "event",
                        "source_id": str(row["id"]),
                        "excerpt": row["user_input"],
                    }
                ],
            )

        with self._connect(self._insight_db) as conn:
            insights = conn.execute(
                """
                SELECT id, thesis, rationale, evidence_json, confidence,
                       novelty_score, status, feedback_state, created_at
                FROM insights
                """
            ).fetchall()
        for row in insights:
            review_state = "confirmed" if row["status"] == "promoted" else "candidate"
            if row["feedback_state"] == "rejected":
                review_state = "rejected"
            self.catalog.upsert_item(
                item_type="insight",
                content=row["thesis"],
                summary=row["rationale"] or "",
                review_state=review_state,
                confidence=float(row["confidence"]),
                importance=float(row["novelty_score"]),
                external_key=f"insight:{row['id']}",
                metadata={
                    "legacy_insight_id": row["id"],
                    "feedback_state": row["feedback_state"],
                    "evidence": _json_load(row["evidence_json"], {}),
                    "created_at": row["created_at"],
                },
                sources=[
                    {
                        "source_type": "insight_synthesis",
                        "source_id": str(row["id"]),
                        "excerpt": row["rationale"] or row["thesis"],
                    }
                ],
            )
        self.catalog.set_meta("legacy_migration_v1", "complete")

    def _reconcile_catalog_vectors(self, force: bool = False):
        latest = self._last_catalog_vector_sync
        sync_started = time.time()
        items = self.catalog.list_items(include_deleted=True, limit=100000)
        for item in items:
            if not force and float(item["updated_at"]) <= latest:
                continue
            if item["review_state"] in {"candidate", "confirmed", "corrected"}:
                self._ltm_repo.upsert_memory_item_node(item)
            else:
                self._ltm_repo.delete_memory_item_node(item["id"])
        entities = self.catalog.list_entities(limit=100000)
        for entity in entities:
            if not force and float(entity["updated_at"]) <= latest:
                continue
            if entity["review_state"] in {"candidate", "confirmed", "corrected"}:
                self._ltm_repo.upsert_entity_node(entity)
            else:
                self._ltm_repo.delete_entity_node(entity["id"])
        self._last_catalog_vector_sync = sync_started

    def _link_detected_entities(self, item: dict):
        content = item["content"]
        candidates = []
        for match in re.finditer(r"\bproject\s+([A-Z][A-Za-z0-9_-]*)", content):
            candidates.append(("project", match.group(1), "project"))
        for match in re.finditer(r"\b([A-Z][a-z]+)'s\b", content):
            candidates.append(("person", match.group(1), "subject"))
        for match in re.finditer(r"\bmy name is\s+([A-Z][A-Za-z'-]+)", content):
            candidates.append(("person", match.group(1), "identity"))
        for match in re.finditer(r"\bdocument\s+([A-Z][A-Za-z0-9_-]*)", content):
            candidates.append(("document", match.group(1), "document"))

        seen = set()
        for entity_type, name, role in candidates:
            key = (entity_type, name.lower(), role)
            if key in seen:
                continue
            seen.add(key)
            entity = self.catalog.upsert_entity(
                entity_type,
                name,
                review_state="candidate",
                metadata={"detected_from_item_id": item["id"]},
            )
            self.catalog.link_item_entity(
                item["id"], entity["id"], role=role, confidence=0.65
            )
            self._ltm_repo.upsert_entity_node(entity)

    def _sync_trace_sources(self, trace: dict):
        turn_id = trace.get("turn_id", "")
        for action in trace.get("actions", []):
            if not isinstance(action, dict):
                continue
            for source in action.get("sources", []):
                if not isinstance(source, dict):
                    continue
                source_id = str(source.get("source_id") or "").strip()
                content = str(
                    source.get("excerpt")
                    or source.get("title")
                    or source.get("url")
                    or ""
                ).strip()
                if not source_id or not content:
                    continue
                item = self.catalog.upsert_item(
                    item_type="source_excerpt",
                    content=content,
                    summary=str(source.get("title") or ""),
                    review_state="candidate",
                    confidence=0.6,
                    importance=0.35,
                    external_key=f"turn_source:{turn_id}:{source_id}",
                    metadata={
                        "turn_id": turn_id,
                        "action_id": action.get("action_id", ""),
                        "action": action.get("action", ""),
                        "url": source.get("url", ""),
                    },
                    sources=[
                        {
                            "source_type": source.get("kind", "tool_run"),
                            "source_id": source_id,
                            "excerpt": content,
                            "metadata": source.get("metadata") or {},
                        }
                    ],
                )
                self._ltm_repo.upsert_memory_item_node(item)

    def prefetch_semantic_facts(self, query: str, limit: int = 6) -> list[dict]:
        q_tokens = _tokenize(query)
        with self._connect(self._semantic_db) as conn:
            rows = conn.execute(
                "SELECT id, statement, confidence, provenance_json, last_seen FROM facts"
            ).fetchall()

        ranked = []
        for row in rows:
            catalog_item = self.catalog.get_by_external_key(
                f"semantic_fact:{row['id']}"
            )
            if catalog_item and catalog_item["review_state"] not in {
                "candidate",
                "confirmed",
                "corrected",
            }:
                continue
            statement = (
                catalog_item["content"] if catalog_item else row["statement"]
            )
            overlap = len(_tokenize(statement) & q_tokens)
            if overlap == 0:
                continue
            confidence = (
                float(catalog_item["confidence"])
                if catalog_item
                else float(row["confidence"])
            )
            ranked.append(
                (
                    overlap + confidence * 0.5,
                    row,
                    statement,
                    confidence,
                    catalog_item,
                )
            )

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": row["id"],
                "memory_item_id": catalog_item["id"] if catalog_item else None,
                "statement": statement,
                "confidence": confidence,
                "score": round(score, 4),
                "provenance": _json_load(row["provenance_json"], []),
            }
            for score, row, statement, confidence, catalog_item in ranked[:limit]
        ]

    def _normalize_thesis(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        cleaned = re.sub(r"([;:,])\1+", r"\1", cleaned)
        cleaned = cleaned.strip(" ;,.-")
        if len(cleaned) > 260:
            cleaned = cleaned[:260].rstrip() + "..."
        return cleaned

    def _quality_score(self, text: str) -> float:
        normalized = self._normalize_thesis(text)
        if not normalized:
            return 0.0
        if _is_polluted_statement(normalized):
            return 0.0
        if _is_memory_like_text(normalized):
            return 0.0
        tokens = re.findall(r"[a-z0-9']+", normalized.lower())
        if len(tokens) < 6:
            return 0.1
        unique_ratio = len(set(tokens)) / len(tokens)
        repeated_penalty = (
            0.2
            if re.search(
                r"(please confirm|i discovered).*(please confirm|i discovered)",
                normalized.lower(),
            )
            else 0.0
        )
        return max(0.0, min(1.0, 0.35 + unique_ratio - repeated_penalty))

    def _should_keep_fact_origin(self, origin: str, text: str) -> bool:
        if origin == "user":
            return True
        if origin != "assistant":
            return False
        return any(p.search(text or "") for p in ASSISTANT_MEMORY_CONFIRM_PATTERNS)

    def _build_candidate_thesis(self, facts: list[sqlite3.Row]) -> str:
        statements = [
            self._normalize_thesis(x["statement"]) for x in facts if x["statement"]
        ]
        statements = [
            s
            for s in statements
            if len(s.split()) >= 5
            and not _is_memory_like_text(s)
            and not _is_polluted_statement(s)
        ]
        if not statements:
            return ""
        if len(statements) == 1:
            return statements[0]
        return f"{statements[0]}. {statements[1]}"

    def _pick_pending_insight(self):
        now = _now_ts()
        with self._connect(self._insight_db) as conn:
            rows = conn.execute(
                """
                SELECT id, thesis, ask_count, snoozed_until, quality_score, confidence, novelty_score
                FROM insights
                WHERE status = 'candidate' AND feedback_state IN ('pending', 'asked', 'snoozed')
                ORDER BY quality_score DESC, confidence DESC, created_at DESC
                LIMIT 12
                """
            ).fetchall()
        for row in rows:
            if int(row["ask_count"] or 0) >= self._proactive_max_asks:
                continue
            if float(row["snoozed_until"] or 0) > now:
                continue
            if float(row["quality_score"] or 0.0) < 0.45:
                continue
            if _is_memory_like_text(str(row["thesis"] or "")):
                continue
            return row
        return None

    def _build_proactive_prompt(self, thesis: str) -> str:
        return f'Quick memory check: should I keep this insight — "{self._normalize_thesis(thesis)}"?'

    def _maybe_get_proactive_prompt(
        self,
        force: bool = False,
        current_intent: str = "",
        current_pathway: str = "",
        query: str = "",
    ) -> dict:
        now = _now_ts()
        intent = (current_intent or "").strip().lower()
        pathway = (current_pathway or "").strip().lower()
        query_tokens = _tokenize(query)
        if not force and (
            intent in {"smalltalk", "greeting"}
            or pathway in {"direct_reply", "brain_dump_reply"}
            or len(query_tokens) <= 2
        ):
            return {"prompt": "", "insight_id": None}
        if not force:
            if (
                self._turn_counter - self._last_proactive_turn
            ) < self._proactive_turn_gap:
                return {"prompt": "", "insight_id": None}
            if (now - self._last_proactive_ts) < self._proactive_seconds_gap:
                return {"prompt": "", "insight_id": None}

        candidate = self._pick_pending_insight()
        if candidate is None:
            return {"prompt": "", "insight_id": None}

        insight_id = int(candidate["id"])
        ask_count = int(candidate["ask_count"] or 0) + 1
        prompt = self._build_proactive_prompt(candidate["thesis"])
        with self._connect(self._insight_db) as conn:
            conn.execute(
                """
                UPDATE insights
                SET ask_count = ?, last_asked_at = ?, snoozed_until = ?, feedback_state = 'asked'
                WHERE id = ?
                """,
                (ask_count, now, now + self._proactive_snooze_seconds, insight_id),
            )

        self._active_asked_insight_id = insight_id
        self._last_proactive_turn = self._turn_counter
        self._last_proactive_ts = now
        return {"prompt": prompt, "insight_id": insight_id}

    def force_proactive_ask(self) -> dict:
        return self._maybe_get_proactive_prompt(force=True)

    def _handle_feedback_if_any(self, user_input: str):
        if self._active_asked_insight_id is None:
            return
        text = (user_input or "").lower()
        if not text.strip():
            return
        confirm = any(
            w in text for w in ["yes", "correct", "right", "keep", "agree", "true"]
        )
        reject = any(
            w in text
            for w in ["no", "wrong", "reject", "discard", "false", "incorrect"]
        )
        if "not an insight" in text or "more of a memory" in text:
            reject = True

        now = _now_ts()
        new_state = "snoozed"
        new_status = "candidate"
        snoozed_until = now + self._proactive_snooze_seconds
        if confirm:
            new_state = "confirmed"
            new_status = "promoted"
            snoozed_until = now
        elif reject:
            new_state = "rejected"
            snoozed_until = now + (7 * self._proactive_snooze_seconds)

        with self._connect(self._insight_db) as conn:
            conn.execute(
                "UPDATE insights SET feedback_state = ?, status = ?, snoozed_until = ? WHERE id = ?",
                (new_state, new_status, snoozed_until, self._active_asked_insight_id),
            )
        self._sync_insight_vector(self._active_asked_insight_id)
        self._active_asked_insight_id = None

    def _sync_insight_vector(self, insight_id: int):
        with self._connect(self._insight_db) as conn:
            row = conn.execute(
                "SELECT id, thesis, rationale, confidence, novelty_score, status FROM insights WHERE id = ?",
                (insight_id,),
            ).fetchone()
        if row is None:
            return
        self._ltm_repo.upsert_insight_node(
            insight_id=int(row["id"]),
            thesis=row["thesis"],
            rationale=row["rationale"] or "",
            confidence=float(row["confidence"] or 0.0),
            novelty_score=float(row["novelty_score"] or 0.0),
            status=row["status"] or "candidate",
        )
        review_state = "confirmed" if row["status"] == "promoted" else "candidate"
        memory_item = self.catalog.upsert_item(
            item_type="insight",
            content=row["thesis"],
            summary=row["rationale"] or "",
            review_state=review_state,
            confidence=float(row["confidence"] or 0.0),
            importance=float(row["novelty_score"] or 0.0),
            external_key=f"insight:{row['id']}",
            metadata={"legacy_insight_id": row["id"]},
            sources=[
                {
                    "source_type": "insight_synthesis",
                    "source_id": str(row["id"]),
                    "excerpt": row["rationale"] or row["thesis"],
                }
            ],
        )
        if review_state in {"candidate", "confirmed", "corrected"}:
            self._ltm_repo.upsert_memory_item_node(memory_item)
        else:
            self._ltm_repo.delete_memory_item_node(memory_item["id"])

    def recall_memory_items(self, query: str, top_k: int = 8) -> list[dict]:
        text = (query or "").strip()
        if not text:
            return []
        self._reconcile_catalog_vectors()
        active_states = {"candidate", "confirmed", "corrected"}
        scores: dict[int, float] = {}
        items: dict[int, dict] = {}

        query_tokens = _tokenize(text)
        since, until = _time_window_from_query(text, _now_ts())
        keyword_items = self.catalog.list_items(
            query=text,
            include_deleted=False,
            since=since,
            until=until,
            limit=max(top_k * 4, 20),
        )
        for item in keyword_items:
            if item["review_state"] not in active_states:
                continue
            overlap = len(_tokenize(item["content"]) & query_tokens)
            phrase_bonus = 1.0 if text.lower() in item["content"].lower() else 0.0
            score = overlap + phrase_bonus + float(item["confidence"]) * 0.25
            scores[item["id"]] = max(scores.get(item["id"], 0.0), score)
            items[item["id"]] = item

        for hit in self._ltm_repo.search_memory_items(text, top_k=max(top_k * 3, 12)):
            item_id = int(hit.get("metadata", {}).get("memory_item_id", -1))
            if item_id <= 0:
                continue
            try:
                item = self.catalog.inspect_item(item_id)
            except KeyError:
                continue
            if item["review_state"] not in active_states:
                continue
            if since is not None and float(item["created_at"]) < since:
                continue
            if until is not None and float(item["created_at"]) > until:
                continue
            vector_score = max(0.0, float(hit.get("score", 0.0)))
            scores[item_id] = max(scores.get(item_id, 0.0), vector_score)
            items[item_id] = item

        ranked = sorted(
            items.values(),
            key=lambda item: (
                scores.get(item["id"], 0.0),
                float(item["importance"]),
                float(item["updated_at"]),
            ),
            reverse=True,
        )
        for item in ranked:
            item["score"] = round(scores.get(item["id"], 0.0), 6)
        return ranked[: max(1, int(top_k))]

    def recall_context(
        self,
        query: str,
        include_web: bool = True,
        top_k: int = 5,
        current_intent: str = "",
        current_pathway: str = "",
        search_cache: dict | None = None,
    ) -> dict:
        now = _now_ts()
        q_tokens = _tokenize(query)
        semantic = self.prefetch_semantic_facts(query, limit=top_k)
        memory_items = self.recall_memory_items(query, top_k=max(top_k, 8))

        with self._connect(self._episodic_db) as conn:
            rows = conn.execute(
                """
                SELECT id, ts, user_input, response, strength, recall_count, last_recalled_at, expiry_at
                FROM events
                WHERE status = 'active'
                ORDER BY ts DESC
                LIMIT 300
                """
            ).fetchall()

        episodic_ranked = []
        for row in rows:
            catalog_episode = self.catalog.get_by_external_key(f"episode:{row['id']}")
            if catalog_episode and catalog_episode["review_state"] not in {
                "candidate",
                "confirmed",
            }:
                continue
            overlap = len(
                _tokenize(f"{row['user_input']} {row['response']}") & q_tokens
            )
            if overlap == 0:
                continue
            recency = max(0.0, 1.0 - ((now - row["ts"]) / (7 * 86400)))
            strength = float(row["strength"])
            repetition = min(1.0, row["recall_count"] / 20.0)
            score = overlap + 0.8 * recency + 0.9 * strength + 0.5 * repetition
            episodic_ranked.append((score, row))

        episodic_ranked.sort(key=lambda x: x[0], reverse=True)
        episodic = []
        with self._connect(self._episodic_db) as conn:
            for score, row in episodic_ranked[:top_k]:
                before = float(row["strength"])
                after = min(1.0, before + self._strength_alpha * (1.0 - before))
                new_expiry = max(
                    float(row["expiry_at"]), now
                ) + self._recall_extension_seconds * (1.0 + after)
                conn.execute(
                    "UPDATE events SET strength = ?, recall_count = ?, last_recalled_at = ?, expiry_at = ? WHERE id = ?",
                    (after, int(row["recall_count"]) + 1, now, new_expiry, row["id"]),
                )
                conn.execute(
                    "INSERT INTO recall_log(event_id, ts, query_text, score_before, score_after) VALUES(?, ?, ?, ?, ?)",
                    (row["id"], now, query, before, after),
                )
                episodic.append(
                    {
                        "id": row["id"],
                        "user_input": row["user_input"],
                        "response": row["response"],
                        "strength": round(before, 4),
                        "recall_count": row["recall_count"],
                        "score": round(score, 4),
                        "ts": row["ts"],
                    }
                )

        insights = []
        vector_k = max(top_k, self._vector_top_k)
        if (query or "").strip():
            hits = self._ltm_repo.search_insights(
                query=query, top_k=vector_k, status="promoted"
            )
            insight_ids = [
                int(hit.get("metadata", {}).get("insight_id", -1))
                for hit in hits
                if int(hit.get("metadata", {}).get("insight_id", -1)) > 0
            ]

            insight_map = {}
            if insight_ids:
                placeholders = ",".join("?" for _ in insight_ids)
                with self._connect(self._insight_db) as conn:
                    rows = conn.execute(
                        f"SELECT id, thesis, rationale, confidence, novelty_score FROM insights WHERE status = 'promoted' AND id IN ({placeholders})",
                        insight_ids,
                    ).fetchall()
                insight_map = {int(row["id"]): row for row in rows}

            for hit in hits:
                metadata = hit.get("metadata", {})
                insight_id = int(metadata.get("insight_id", -1))
                row = insight_map.get(insight_id)
                if row is None:
                    continue
                catalog_insight = self.catalog.get_by_external_key(
                    f"insight:{insight_id}"
                )
                if catalog_insight and catalog_insight["review_state"] not in {
                    "candidate",
                    "confirmed",
                }:
                    continue
                insights.append(
                    {
                        "id": row["id"],
                        "thesis": row["thesis"],
                        "rationale": row["rationale"],
                        "confidence": row["confidence"],
                        "novelty_score": row["novelty_score"],
                        "score": round(float(hit.get("score", 0.0)), 4),
                    }
                )
                if len(insights) >= top_k:
                    break

        if not insights:
            with self._connect(self._insight_db) as conn:
                insight_rows = conn.execute(
                    "SELECT id, thesis, rationale, confidence, novelty_score FROM insights WHERE status = 'promoted' ORDER BY created_at DESC LIMIT ?",
                    (top_k,),
                ).fetchall()

            for row in insight_rows:
                catalog_insight = self.catalog.get_by_external_key(
                    f"insight:{row['id']}"
                )
                if catalog_insight and catalog_insight["review_state"] not in {
                    "candidate",
                    "confirmed",
                }:
                    continue
                insights.append(
                    {
                        "id": row["id"],
                        "thesis": row["thesis"],
                        "rationale": row["rationale"],
                        "confidence": row["confidence"],
                        "novelty_score": row["novelty_score"],
                    }
                )

        web_results = (
            search_web(query, cache=search_cache)
            if include_web and (query or "").strip()
            else []
        )
        max_search_chars = int(self._config.get("search", {}).get("max_chars", 2200))
        web_result = render_search_results(web_results, max_chars=max_search_chars)
        proactive = (
            {}
            if self.in_focus()
            else self._maybe_get_proactive_prompt(
                force=False,
                current_intent=current_intent,
                current_pathway=current_pathway,
                query=query,
            )
        )
        reminder = self._due_commitment_reminder(now)
        feedback_prompt = "\n\n".join(
            part for part in [proactive.get("prompt", ""), reminder] if part
        )

        semantic_lines = [x["statement"] for x in semantic]
        semantic_text = "\n".join(f"- {line}" for line in semantic_lines if line)

        def _flags(item: dict) -> str:
            flags = []
            if item["metadata"].get("stale"):
                flags.append("STALE")
            if item["metadata"].get("contradicts"):
                conflict_ids = ",".join(
                    f"#{other}" for other in item["metadata"]["contradicts"]
                )
                flags.append(f"CONFLICTS with {conflict_ids}")
            return f" [{'; '.join(flags)}]" if flags else ""

        memory_text = "\n".join(
            f"- [{item['item_type']}/{item['review_state']}]{_flags(item)} {item['content']}"
            for item in memory_items
        )
        episodic_text = "\n".join(
            f"- User: {x['user_input']} | Bot: {x['response']}" for x in episodic
        )
        insight_text = "\n".join(f"- {x['thesis']}" for x in insights)
        combined = "\n\n".join(
            x
            for x in [
                f"[Unified Memory]\n{memory_text}" if memory_text else "",
                (
                    f"[Semantic Facts]\n{semantic_text}"
                    if semantic_text and not memory_text
                    else ""
                ),
                (
                    f"[Episodic Recall]\n{episodic_text}"
                    if episodic_text and not memory_text
                    else ""
                ),
                (
                    f"[Promoted Insights]\n{insight_text}"
                    if insight_text and not memory_text
                    else ""
                ),
                f"[Web Grounding]\n{web_result}" if web_result else "",
            ]
            if x
        )

        return {
            "semantic": semantic,
            "episodic": episodic,
            "insights": insights,
            "memory_items": memory_items,
            "web": web_result,
            "web_results": [item.to_dict() for item in web_results],
            "combined_context": combined,
            "style_context": self._style_context(),
            "pending_feedback_prompt": feedback_prompt,
            "pending_feedback_id": proactive.get("insight_id"),
        }

    def commit_turn(self, user_input: str, response: str, trace: dict | None = None):
        trace = trace or {}
        self._turn_counter += 1
        self._handle_feedback_if_any(user_input)

        intent = (trace.get("intent") or "").strip().lower()
        self._last_route_intent = intent
        self._last_route_pathway = (trace.get("pathway") or "").strip().lower()
        if intent in {"brain_dump", "debug", "debug_ask", "debug_idle"}:
            return
        if _is_noise_text(user_input):
            return

        now = _now_ts()
        with self._connect(self._episodic_db) as conn:
            cursor = conn.execute(
                """
                INSERT INTO events(ts, session_id, user_input, response, intent, todo_json, context_json, error_text, expiry_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    trace.get("session_id", "default"),
                    user_input,
                    response,
                    trace.get("intent", ""),
                    json.dumps(trace.get("todo", [])),
                    json.dumps(trace.get("context", {})),
                    trace.get("error", ""),
                    now + self._episodic_ttl_seconds,
                ),
            )
            event_id = int(cursor.lastrowid)

        episode = self.catalog.upsert_item(
            item_type="episode",
            content=f"User: {user_input} Assistant: {response}",
            review_state="confirmed",
            confidence=0.65,
            importance=0.4,
            external_key=f"episode:{event_id}",
            metadata={
                "event_id": event_id,
                "turn_id": trace.get("turn_id", ""),
                "session_id": trace.get("session_id", "default"),
                "intent": trace.get("intent", ""),
                "timestamp": now,
            },
            sources=[
                {
                    "source_type": "event",
                    "source_id": str(event_id),
                    "excerpt": user_input,
                    "metadata": {"turn_id": trace.get("turn_id", "")},
                }
            ],
        )
        self._ltm_repo.upsert_memory_item_node(episode)
        self._sync_trace_sources(trace)
        self._upsert_semantic_facts(
            user_input,
            response,
            event_id=event_id,
            turn_id=trace.get("turn_id", ""),
        )

        session_id = trace.get("session_id", "default")
        capture_type = "parsed" if trace.get("input_source") == "phone" else "typed"
        self.catalog.record_capture_event(
            capture_type,
            user_input,
            session_id=session_id,
            turn_id=trace.get("turn_id", ""),
            memory_item_id=episode["id"],
            metadata={"intent": intent},
        )
        session_context = self.context_engine.current_context(session_id, now=now)
        self.tag_engine.tag_item(episode, session_context=session_context, now=now)
        self._capture_commitments(user_input, session_context, now)
        self._maybe_update_style_profile(user_input)
        self.catalog.record_context_snapshot(
            self.context_engine.build_snapshot(
                session_id=session_id,
                intent=intent,
                pathway=self._last_route_pathway,
                source_ids=trace.get("source_ids", []),
                now=now,
            ),
            session_id=session_id,
            turn_id=trace.get("turn_id", ""),
        )

    def _extract_fact_candidates(self, user_input: str, response: str) -> list[dict]:
        lines = []
        del response
        for part in re.split(r"[\n\.!?]", user_input or ""):
            explicit = _has_memory_directive(part)
            text = _strip_memory_directive(part)
            if explicit:
                should_store = _should_store_explicit_fact(text)
                source = "explicit_remember"
                confidence = 0.75
            else:
                should_store = _should_store_user_fact(text)
                source = "turn"
                confidence = 0.55
            if not should_store:
                continue
            normalized = _normalize_fact_text(text)
            if len(normalized.split()) < 3:
                continue
            if len(normalized) > 220:
                normalized = normalized[:220].strip() + "..."
            lines.append(
                {
                    "statement": normalized,
                    "source": source,
                    "confidence": confidence,
                }
            )
        out = []
        seen = set()
        for item in lines:
            key = item["statement"].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= 8:
                break
        return out

    def _upsert_semantic_facts(
        self,
        user_input: str,
        response: str,
        *,
        event_id: int | None = None,
        turn_id: str = "",
    ):
        candidates = self._extract_fact_candidates(user_input, response)
        self._upsert_fact_candidates(
            candidates,
            event_id=event_id,
            turn_id=turn_id,
        )

    def _upsert_fact_candidates(
        self,
        candidates: list[dict],
        *,
        event_id: int | None = None,
        turn_id: str = "",
    ) -> list[dict]:
        now = _now_ts()
        if not candidates:
            return []
        stored = []
        with self._connect(self._semantic_db) as conn:
            for candidate in candidates:
                statement = candidate["statement"]
                key = " ".join(sorted(_tokenize(statement)))
                if not key:
                    continue
                if _is_noise_text(statement) or _is_polluted_statement(statement):
                    continue
                source = candidate.get("source", "turn")
                confidence = float(candidate.get("confidence", 0.55))
                row = conn.execute(
                    "SELECT id, confidence, provenance_json FROM facts WHERE canonical_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    cursor = conn.execute(
                        "INSERT INTO facts(statement, canonical_key, confidence, first_seen, last_seen, provenance_json) VALUES(?, ?, ?, ?, ?, ?)",
                        (
                            statement,
                            key,
                            confidence,
                            now,
                            now,
                            json.dumps([{"ts": now, "source": source}]),
                        ),
                    )
                    fact_id = int(cursor.lastrowid)
                    stored_confidence = confidence
                    provenance = [{"ts": now, "source": source}]
                else:
                    provenance = _json_load(row["provenance_json"], [])
                    provenance.append({"ts": now, "source": source})
                    increment = 0.1 if source == "explicit_remember" else 0.05
                    stored_confidence = min(
                        0.99,
                        max(float(row["confidence"]), confidence) + increment,
                    )
                    conn.execute(
                        "UPDATE facts SET confidence = ?, last_seen = ?, provenance_json = ? WHERE id = ?",
                        (
                            stored_confidence,
                            now,
                            json.dumps(provenance[-15:]),
                            row["id"],
                        ),
                    )
                    fact_id = int(row["id"])
                stored.append(
                    {
                        "id": fact_id,
                        "statement": statement,
                        "confidence": stored_confidence,
                        "provenance": provenance[-15:],
                        "source": source,
                    }
                )
        for item in stored:
            if event_id is not None:
                source = {
                    "source_type": "event",
                    "source_id": str(event_id),
                    "excerpt": item["statement"],
                    "metadata": {
                        "turn_id": turn_id,
                        "extraction_source": item["source"],
                    },
                }
            else:
                source = {
                    "source_type": item["source"],
                    "source_id": f"semantic:{item['id']}:{len(item['provenance'])}",
                    "excerpt": item["statement"],
                }
            review_state = (
                "confirmed"
                if item["source"] in {"explicit_remember", "memory_write"}
                else "candidate"
            )
            memory_item = self.catalog.upsert_item(
                item_type=self._classify_memory_item_type(item["statement"]),
                content=item["statement"],
                review_state=review_state,
                confidence=item["confidence"],
                importance=0.6 if review_state == "confirmed" else 0.45,
                external_key=f"semantic_fact:{item['id']}",
                metadata={"legacy_fact_id": item["id"]},
                sources=[source],
            )
            self._link_detected_entities(memory_item)
            self._ltm_repo.upsert_memory_item_node(memory_item)
            self.tag_engine.tag_item(memory_item, now=now)
            self._detect_contradictions(memory_item)
            item["memory_item_id"] = memory_item["id"]
        return stored

    def write_memory(
        self,
        statement: str,
        *,
        source: str = "memory_write",
        confidence: float = 0.85,
    ) -> dict:
        normalized = _normalize_fact_text(_strip_memory_directive(statement or ""))
        if len(normalized.split()) < 2:
            raise ValueError("Memory statement is too short")
        if _is_noise_text(normalized) or _is_polluted_statement(normalized):
            raise ValueError("Memory statement was rejected by the quality filter")
        if len(normalized) > 220:
            normalized = normalized[:220].strip() + "..."
        stored = self._upsert_fact_candidates(
            [
                {
                    "statement": normalized,
                    "source": source,
                    "confidence": max(0.0, min(0.99, float(confidence))),
                }
            ]
        )
        if not stored:
            raise ValueError("Memory statement could not be stored")
        return stored[0]

    def _capture_commitments(
        self, text: str, session_context: dict, now: float
    ) -> list[dict]:
        stored = []
        for candidate in extract_commitments(text, now):
            metadata: dict = {"status": "open", "origin": "conversation"}
            if candidate["due_at"]:
                metadata["due_at"] = candidate["due_at"]
            item = self.catalog.upsert_item(
                item_type="commitment",
                content=candidate["content"],
                review_state="candidate",
                confidence=candidate["confidence"],
                importance=0.7,
                external_key=f"commitment:{candidate['canonical_key']}",
                metadata=metadata,
            )
            self._ltm_repo.upsert_memory_item_node(item)
            self.tag_engine.tag_item(item, session_context=session_context, now=now)
            stored.append(item)
        return stored

    def open_commitments(self) -> list[dict]:
        items = [
            item
            for item in self.catalog.list_items(item_type="commitment", limit=500)
            if item["review_state"] in {"candidate", "confirmed", "corrected"}
            and item["metadata"].get("status", "open") == "open"
        ]
        items.sort(key=lambda i: i["metadata"].get("due_at") or float("inf"))
        return items

    def complete_commitment(self, item_id: int) -> dict:
        return self.catalog.update_item_metadata(
            int(item_id), {"status": "done", "stale": None, "stale_reason": None}
        )

    def _due_commitment_reminder(self, now: float) -> str:
        active_project = ""
        for commitment in self.open_commitments():
            metadata = commitment["metadata"]
            due_at = metadata.get("due_at")
            overdue = bool(due_at) and due_at < now
            due_soon = bool(due_at) and now <= due_at <= now + 86400
            if not active_project and not due_at:
                active_project = self.context_engine.current_context().get(
                    "active_project", ""
                )
            context_match = (
                not due_at
                and active_project
                and any(
                    tag["tag_type"] == "project"
                    and tag["name"].lower() == active_project.lower()
                    for tag in commitment.get("tags", [])
                )
            )
            # Focus sessions suppress everything except overdue commitments.
            if self.in_focus() and not overdue:
                continue
            if not (overdue or due_soon or context_match):
                continue
            last_reminded = float(metadata.get("last_reminded_at") or 0)
            if now - last_reminded < self._reminder_gap_seconds:
                continue
            self.catalog.update_item_metadata(
                commitment["id"], {"last_reminded_at": now}
            )
            if overdue:
                label = "overdue"
            elif due_soon:
                label = "due soon"
            else:
                label = f"related to {active_project}"
            return f"Reminder ({label}): {commitment['content']}"
        return ""

    def _detect_contradictions(self, memory_item: dict) -> int:
        extracted = _extract_property(memory_item["content"])
        if extracted is None:
            return 0
        prop, value = extracted
        value_tokens = _tokenize(value)
        flagged = 0
        for other in self.catalog.list_items(limit=300):
            if other["id"] == memory_item["id"]:
                continue
            if other["item_type"] not in {"fact", "preference", "commitment"}:
                continue
            if other["review_state"] not in {"candidate", "confirmed", "corrected"}:
                continue
            other_extracted = _extract_property(other["content"])
            if other_extracted is None or other_extracted[0] != prop:
                continue
            other_tokens = _tokenize(other_extracted[1])
            union = value_tokens | other_tokens
            overlap = len(value_tokens & other_tokens) / len(union) if union else 1.0
            if overlap < 0.5:
                self.catalog.flag_contradiction(
                    memory_item["id"],
                    other["id"],
                    note=f"same property '{prop}', conflicting values",
                )
                flagged += 1
        return flagged

    def _apply_staleness_rules(self, now: float | None = None) -> int:
        import datetime as _dt

        now = now if now is not None else _now_ts()
        today = _dt.date.fromtimestamp(now)
        marked = 0
        for item in self.catalog.list_items(limit=1000):
            metadata = item["metadata"]
            if metadata.get("stale"):
                continue
            reason = ""
            if (
                item["item_type"] == "commitment"
                and metadata.get("status", "open") == "open"
                and metadata.get("due_at")
                and float(metadata["due_at"]) < now
            ):
                reason = "overdue commitment"
            else:
                for tag in item.get("tags", []):
                    if tag["tag_type"] != "time":
                        continue
                    if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", tag["name"]):
                        continue
                    if _dt.date.fromisoformat(tag["name"]) < today:
                        reason = f"references past date {tag['name']}"
                        break
            if reason:
                self.catalog.update_item_metadata(
                    item["id"], {"stale": True, "stale_reason": reason}
                )
                marked += 1
        return marked

    def start_focus(self, minutes: float) -> float:
        self._focus_until = _now_ts() + max(1.0, float(minutes)) * 60
        return self._focus_until

    def end_focus(self):
        self._focus_until = 0.0

    def in_focus(self) -> bool:
        return _now_ts() < self._focus_until

    def get_style_profile(self) -> list[str]:
        data = _json_load(self.catalog.get_meta("style_profile", ""), [])
        return data if isinstance(data, list) else []

    def _maybe_update_style_profile(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        rules = self.get_style_profile()
        lowered = {rule.lower() for rule in rules}
        for pattern, canned in STYLE_INSTRUCTION_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            rule = canned or match.group(1).strip().capitalize()
            if rule.lower() in lowered:
                continue
            rules.append(rule)
            lowered.add(rule.lower())
        if len(rules) > 12:
            rules = rules[-12:]
        self.catalog.set_meta("style_profile", json.dumps(rules))

    def _style_context(self) -> str:
        return "\n".join(f"- {rule}" for rule in self.get_style_profile())

    def daily_brief(self) -> str:
        now = _now_ts()
        commitments = self.open_commitments()
        overdue = [
            c
            for c in commitments
            if c["metadata"].get("due_at") and c["metadata"]["due_at"] < now
        ]
        commitment_line = "none"
        if overdue:
            commitment_line = f"OVERDUE: {overdue[0]['content']}"
        elif commitments:
            commitment_line = f"next: {commitments[0]['content']}"

        conflict_line = "none"
        for item in self.catalog.list_items(limit=500):
            others = item["metadata"].get("contradicts") or []
            if not others:
                continue
            try:
                other = self.catalog.inspect_item(others[0])
            except KeyError:
                continue
            if other["review_state"] in {"rejected", "deleted"}:
                continue
            conflict_line = (
                f"\"{item['content']}\" vs \"{other['content']}\" "
                f"(items #{item['id']}/#{other['id']} — review in /memory)"
            )
            break

        context = self.context_engine.current_context()
        pattern_line = "none"
        if context["active_project"]:
            pattern_line = f"active project: {context['active_project']}"
            if context["recent_topics"]:
                pattern_line += f"; recent focus: {', '.join(context['recent_topics'][:2])}"
        elif context["recent_topics"]:
            pattern_line = f"recent focus: {', '.join(context['recent_topics'][:3])}"

        return (
            "=== Daily Brief ===\n"
            f"Commitment: {commitment_line}\n"
            f"Conflict: {conflict_line}\n"
            f"Pattern: {pattern_line}\n"
            f"Open loops: {len(commitments)}"
        )

    def review_drill(self, limit: int = 3) -> str:
        now = _now_ts()
        candidates = [
            item
            for item in self.catalog.list_items(limit=500)
            if item["item_type"] in {"fact", "preference", "idea", "insight"}
            and item["review_state"] in {"confirmed", "corrected"}
        ]
        if not candidates:
            return "Nothing to review yet — confirm some memories first."
        candidates.sort(
            key=lambda i: (
                float(i["metadata"].get("last_reviewed_at") or 0),
                float(i["updated_at"]),
            )
        )
        picked = candidates[: max(1, int(limit))]
        lines = ["=== Review Drill ===", "Try to recall these before reading:"]
        for index, item in enumerate(picked, 1):
            lines.append(f"{index}. [{item['item_type']}] {item['content']}")
            self.catalog.update_item_metadata(item["id"], {"last_reviewed_at": now})
        return "\n".join(lines)

    def capture_note(
        self,
        content: str,
        *,
        capture_type: str = "pasted",
        session_id: str = "capture",
    ) -> dict:
        """Direct capture pipeline for typed/pasted/voice input outside a turn."""
        now = _now_ts()
        item = self.catalog.upsert_item(
            item_type="episode",
            content=content,
            review_state="confirmed",
            confidence=0.7,
            importance=0.5,
            metadata={"captured": True, "session_id": session_id},
        )
        self._ltm_repo.upsert_memory_item_node(item)
        event = self.catalog.record_capture_event(
            capture_type,
            content,
            session_id=session_id,
            memory_item_id=item["id"],
        )
        session_context = self.context_engine.current_context(session_id, now=now)
        tags = self.tag_engine.tag_item(item, session_context=session_context, now=now)
        self._capture_commitments(content, session_context, now)
        return {"item": self.catalog.inspect_item(item["id"]), "event": event, "tags": tags}

    def search_memory(self, query: str, top_k: int = 5) -> dict:
        return self.recall_context(query, include_web=False, top_k=top_k)

    def list_memory_items(self, **filters) -> list[dict]:
        return self.catalog.list_items(**filters)

    def inspect_memory_item(self, item_id: int) -> dict:
        return self.catalog.inspect_item(item_id)

    def correct_memory_item(
        self,
        item_id: int,
        content: str,
        reason: str = "",
    ) -> dict:
        item = self.catalog.correct_item(item_id, content, reason)
        self._ltm_repo.upsert_memory_item_node(item)
        self._link_detected_entities(item)
        return self.catalog.inspect_item(item_id)

    def set_memory_review_state(
        self,
        item_id: int,
        review_state: str,
        note: str = "",
    ) -> dict:
        item = self.catalog.set_review_state(item_id, review_state, note)
        if review_state in {"candidate", "confirmed", "corrected"}:
            self._ltm_repo.upsert_memory_item_node(item)
        else:
            self._ltm_repo.delete_memory_item_node(item_id)
        return item

    def merge_memory_items(
        self,
        target_id: int,
        source_ids: list[int],
        content: str | None = None,
        note: str = "",
    ) -> dict:
        item = self.catalog.merge_items(
            target_id,
            source_ids,
            content=content,
            note=note,
        )
        self._ltm_repo.upsert_memory_item_node(item)
        for source_id in source_ids:
            self._ltm_repo.delete_memory_item_node(source_id)
        return item

    def export_memory(self, export_dir: str) -> dict:
        return self.catalog.export(export_dir)

    def memory_overview(self) -> dict:
        return self.catalog.overview()

    def upsert_memory_entity(self, entity_type: str, name: str, **values) -> dict:
        entity = self.catalog.upsert_entity(entity_type, name, **values)
        if entity["review_state"] in {"candidate", "confirmed", "corrected"}:
            self._ltm_repo.upsert_entity_node(entity)
        return entity

    def link_memory_entity(self, item_id: int, entity_id: int, **values):
        self.catalog.link_item_entity(item_id, entity_id, **values)

    def create_memory_relation(
        self,
        source_entity_id: int,
        relation_type: str,
        target_entity_id: int,
        **values,
    ) -> dict:
        return self.catalog.create_relation(
            source_entity_id,
            relation_type,
            target_entity_id,
            **values,
        )

    def run_idle_jobs(self):
        created, promoted = self._run_synthesis_job()
        decayed, deleted = self._apply_episodic_decay()
        stale_marked = self._apply_staleness_rules()
        return {
            "insight_candidates_created": created,
            "insights_promoted": promoted,
            "episodic_decayed": decayed,
            "episodic_deleted": deleted,
            "stale_marked": stale_marked,
        }

    def _run_synthesis_job(self):
        with self._connect(self._semantic_db) as conn:
            facts = conn.execute(
                "SELECT id, statement, confidence FROM facts ORDER BY last_seen DESC LIMIT 40"
            ).fetchall()
        facts = [
            row
            for row in facts
            if not _is_polluted_statement(row["statement"])
            and len(_tokenize(row["statement"])) >= 4
        ]
        if len(facts) < 4:
            return 0, 0

        cluster = facts[:6]
        thesis = self._build_candidate_thesis(cluster)
        if not thesis:
            return 0, 0
        quality_score = self._quality_score(thesis)
        if quality_score < 0.6:
            return 0, 0

        evidence_query = " ".join(row["statement"] for row in cluster[:2])[:140]
        web_evidence = tool_websearch(evidence_query)
        has_web = bool(web_evidence and "failed" not in web_evidence.lower())
        avg_conf = sum(float(row["confidence"]) for row in cluster) / len(cluster)
        novelty_score = min(1.0, 0.45 + 0.05 * len(cluster))
        confidence = min(0.99, avg_conf + (0.08 if has_web else 0.0))

        status = (
            "promoted"
            if has_web and confidence >= 0.78 and novelty_score >= 0.7
            else "candidate"
        )
        feedback_state = "pending" if status == "candidate" else "approved"

        with self._connect(self._insight_db) as conn:
            existing = conn.execute(
                "SELECT id FROM insights WHERE thesis = ?", (thesis,)
            ).fetchone()
            if existing is not None:
                return 0, 0
            cursor = conn.execute(
                """
                INSERT INTO insights(thesis, rationale, evidence_json, novelty_score, confidence, status, feedback_state, ask_count, quality_score, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    thesis,
                    "Synthesized from reinforced semantic facts with web grounding.",
                    json.dumps(
                        {
                            "facts": [row["id"] for row in cluster],
                            "web": web_evidence[:2000],
                        }
                    ),
                    novelty_score,
                    confidence,
                    status,
                    feedback_state,
                    quality_score,
                    _now_ts(),
                ),
            )
            insight_id = int(cursor.lastrowid)

        self._ltm_repo.upsert_insight_node(
            insight_id=insight_id,
            thesis=thesis,
            rationale="Synthesized from reinforced semantic facts with web grounding.",
            confidence=confidence,
            novelty_score=novelty_score,
            status=status,
        )
        return 1, 1 if status == "promoted" else 0

    def _apply_episodic_decay(self):
        now = _now_ts()
        decayed = 0
        with self._connect(self._episodic_db) as conn:
            rows = conn.execute(
                "SELECT id, strength, last_recalled_at FROM events WHERE status = 'active'"
            ).fetchall()
            for row in rows:
                last_seen = float(row["last_recalled_at"] or 0)
                inactive_seconds = (now - last_seen) if last_seen else 86400
                decay_factor = self._strength_decay * max(1.0, inactive_seconds / 86400)
                new_strength = max(0.01, float(row["strength"]) - decay_factor)
                conn.execute(
                    "UPDATE events SET strength = ? WHERE id = ?",
                    (new_strength, row["id"]),
                )
                decayed += 1
            before = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
            conn.execute(
                "DELETE FROM events WHERE expiry_at <= ? AND strength < 0.15", (now,)
            )
            after = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        return decayed, max(0, before - after)

    def get_status(self) -> dict:
        with self._connect(self._episodic_db) as econn:
            episodic_count = econn.execute(
                "SELECT COUNT(*) AS c FROM events"
            ).fetchone()["c"]
        with self._connect(self._semantic_db) as sconn:
            semantic_count = sconn.execute(
                "SELECT COUNT(*) AS c FROM facts"
            ).fetchone()["c"]
        with self._connect(self._insight_db) as iconn:
            insight_count = iconn.execute(
                "SELECT COUNT(*) AS c FROM insights WHERE status = 'promoted'"
            ).fetchone()["c"]
            pending_count = iconn.execute(
                "SELECT COUNT(*) AS c FROM insights WHERE feedback_state IN ('pending','asked','snoozed')"
            ).fetchone()["c"]
        return {
            "episodic_count": episodic_count,
            "semantic_count": semantic_count,
            "insight_count": insight_count,
            "pending_feedback_count": pending_count,
            "memory_catalog": self.catalog.overview(),
            "embedding_available": self._embedding_available,
            "embedding_last_error": self._embedding_last_error,
        }

    def cleanup_polluted_memory(self) -> dict:
        deleted_facts = 0
        deleted_insights = 0
        deleted_catalog_items = 0
        with self._connect(self._semantic_db) as conn:
            rows = conn.execute("SELECT id, statement FROM facts").fetchall()
            bad_ids = [
                row["id"] for row in rows if _is_polluted_statement(row["statement"])
            ]
            if bad_ids:
                placeholders = ",".join("?" for _ in bad_ids)
                conn.execute(f"DELETE FROM facts WHERE id IN ({placeholders})", bad_ids)
                deleted_facts = len(bad_ids)

        with self._connect(self._insight_db) as conn:
            rows = conn.execute(
                "SELECT id, thesis FROM insights WHERE status = 'candidate'"
            ).fetchall()
            bad_ids = [
                row["id"] for row in rows if _is_polluted_statement(row["thesis"])
            ]
            if bad_ids:
                placeholders = ",".join("?" for _ in bad_ids)
                conn.execute(
                    f"DELETE FROM insights WHERE id IN ({placeholders})", bad_ids
                )
                deleted_insights = len(bad_ids)
        for item in self.catalog.list_items(include_deleted=False, limit=100000):
            if _is_polluted_statement(item["content"]):
                self.catalog.set_review_state(
                    item["id"],
                    "deleted",
                    "Automatic pollution cleanup",
                )
                self._ltm_repo.delete_memory_item_node(item["id"])
                deleted_catalog_items += 1
        return {
            "deleted_semantic_facts": deleted_facts,
            "deleted_candidate_insights": deleted_insights,
            "deleted_catalog_items": deleted_catalog_items,
        }

    def get_brain_dump(self, mode: str = "full", limit: int = 50) -> str:
        catalog_items = self.catalog.list_items(
            include_deleted=True,
            limit=limit,
        )
        entities = self.catalog.list_entities(limit=limit)
        with self._connect(self._episodic_db) as econn:
            episodic = econn.execute(
                "SELECT id, ts, user_input, response, strength, recall_count, expiry_at FROM events ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        with self._connect(self._semantic_db) as sconn:
            semantic = sconn.execute(
                "SELECT id, statement, confidence, last_seen, provenance_json FROM facts ORDER BY last_seen DESC LIMIT ?",
                (limit,),
            ).fetchall()
        with self._connect(self._insight_db) as iconn:
            insights = iconn.execute(
                "SELECT id, thesis, confidence, novelty_score, status, feedback_state, ask_count FROM insights ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        with self._connect(self._jobs_db) as jconn:
            jobs = jconn.execute(
                "SELECT id, kind, status, attempts, created_at, finished_at, error_text FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        lines = ["=== Brain Dump ===", f"Status: {self.get_status()}"]
        if mode in ("full", "memory", "catalog"):
            lines.append("\n--- Unified Memory Catalog ---")
            lines.extend(
                [
                    f"[{item['id']}] type={item['item_type']} state={item['review_state']} v={item['version']} conf={float(item['confidence']):.2f} :: {item['content']}"
                    for item in catalog_items
                ]
                or ["(empty)"]
            )
            lines.append("\n--- Entity Graph ---")
            lines.extend(
                [
                    f"[{entity['id']}] {entity['entity_type']} state={entity['review_state']} :: {entity['name']}"
                    for entity in entities
                ]
                or ["(empty)"]
            )
        if mode in ("full", "semantic"):
            lines.append("\n--- Semantic Memory ---")
            lines.extend(
                [
                    f"[{row['id']}] conf={float(row['confidence']):.2f} prov={len(_json_load(row['provenance_json'], []))} :: {row['statement']}"
                    for row in semantic
                ]
                or ["(empty)"]
            )
        if mode in ("full", "episodic"):
            lines.append("\n--- Episodic Memory ---")
            lines.extend(
                [
                    f"[{row['id']}] strength={float(row['strength']):.2f} recalls={row['recall_count']} exp={int(row['expiry_at'])} :: {row['user_input']} -> {row['response'][:110]}"
                    for row in episodic
                ]
                or ["(empty)"]
            )
        if mode in ("full", "insights", "ltm"):
            lines.append("\n--- Insight LTM ---")
            lines.extend(
                [
                    f"[{row['id']}] status={row['status']} feedback={row['feedback_state']} ask={row['ask_count']} conf={float(row['confidence']):.2f} nov={float(row['novelty_score']):.2f} :: {row['thesis']}"
                    for row in insights
                ]
                or ["(empty)"]
            )
        if mode in ("full", "jobs"):
            lines.append("\n--- Jobs ---")
            lines.extend(
                [
                    f"[{row['id']}] kind={row['kind']} status={row['status']} attempts={row['attempts']} err={(row['error_text'] or '')[:80]}"
                    for row in jobs
                ]
                or ["(empty)"]
            )
        return "\n".join(lines)

    def clear_all_memory(self):
        with self._connect(self._episodic_db) as conn:
            conn.execute("DELETE FROM recall_log")
            conn.execute("DELETE FROM events")
        with self._connect(self._semantic_db) as conn:
            conn.execute("DELETE FROM facts")
        with self._connect(self._insight_db) as conn:
            conn.execute("DELETE FROM insights")
        with self._connect(self._jobs_db) as conn:
            conn.execute("DELETE FROM jobs")
        self.catalog.clear()
        self._ltm_repo.clear()
        self._active_asked_insight_id = None
        self._turn_counter = 0
        self._last_proactive_turn = -99999
        self._last_proactive_ts = 0.0
