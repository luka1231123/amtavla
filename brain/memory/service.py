import json
import os
import re
import sqlite3
import time
from collections import Counter

import llama_client
from brain.config import load_brain_config
from brain.memory.ltm_repository import LtmRepository
from brain.memory.vector_store import SQLiteVecStore
from tools.websearch import tool_websearch


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
    prefixes = (
        "remember this:",
        "remember this",
        "remember that:",
        "remember that",
        "don't forget:",
        "dont forget:",
        "don't forget",
        "dont forget",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return _normalize_fact_text(value[len(prefix) :])
    return value


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
        self._vector_top_k = int(memory_cfg.get("vector_top_k", 5))

        self._turn_counter = 0
        self._last_proactive_turn = -99999
        self._last_proactive_ts = 0.0
        self._active_asked_insight_id = None
        self._last_route_intent = ""
        self._last_route_pathway = ""

        self._ltm_store = SQLiteVecStore(
            db_path=vector_db_path,
            embedding_dim=self._embedding_dim,
        )
        self._ltm_repo = LtmRepository(self._ltm_store, self._embed_text)

        self._init_schema()
        self.cleanup_polluted_memory()

    def _embed_text(self, text: str) -> list[float]:
        try:
            response = llama_client.embed(text)
            embedding = response.get("embedding", [])
            if isinstance(embedding, list) and embedding:
                vec = [float(x) for x in embedding]
                if len(vec) >= self._embedding_dim:
                    return vec[: self._embedding_dim]
                return vec + [0.0] * (self._embedding_dim - len(vec))
        except Exception:
            pass
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

    def prefetch_semantic_facts(self, query: str, limit: int = 6) -> list[dict]:
        q_tokens = _tokenize(query)
        with self._connect(self._semantic_db) as conn:
            rows = conn.execute(
                "SELECT id, statement, confidence, provenance_json, last_seen FROM facts"
            ).fetchall()

        ranked = []
        for row in rows:
            overlap = len(_tokenize(row["statement"]) & q_tokens)
            if overlap == 0:
                continue
            ranked.append((overlap + float(row["confidence"]) * 0.5, row))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": row["id"],
                "statement": row["statement"],
                "confidence": row["confidence"],
                "score": round(score, 4),
                "provenance": _json_load(row["provenance_json"], []),
            }
            for score, row in ranked[:limit]
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

            insights = [
                {
                    "id": row["id"],
                    "thesis": row["thesis"],
                    "rationale": row["rationale"],
                    "confidence": row["confidence"],
                    "novelty_score": row["novelty_score"],
                }
                for row in insight_rows
            ]

        web_result = (
            tool_websearch(query, cache=search_cache)
            if include_web and (query or "").strip()
            else ""
        )
        proactive = self._maybe_get_proactive_prompt(
            force=False,
            current_intent=current_intent,
            current_pathway=current_pathway,
            query=query,
        )

        semantic_lines = [x["statement"] for x in semantic]
        semantic_text = "\n".join(f"- {line}" for line in semantic_lines if line)
        episodic_text = "\n".join(
            f"- User: {x['user_input']} | Bot: {x['response']}" for x in episodic
        )
        insight_text = "\n".join(f"- {x['thesis']}" for x in insights)
        combined = "\n\n".join(
            x
            for x in [
                f"[Semantic Facts]\n{semantic_text}" if semantic_text else "",
                f"[Episodic Recall]\n{episodic_text}" if episodic_text else "",
                f"[Promoted Insights]\n{insight_text}" if insight_text else "",
                f"[Web Grounding]\n{web_result}" if web_result else "",
            ]
            if x
        )

        return {
            "semantic": semantic,
            "episodic": episodic,
            "insights": insights,
            "web": web_result,
            "combined_context": combined,
            "pending_feedback_prompt": proactive.get("prompt", ""),
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
            conn.execute(
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

        self._upsert_semantic_facts(user_input, response)

    def _extract_fact_candidates(self, user_input: str, response: str) -> list[str]:
        lines = []
        del response
        for part in re.split(r"[\n\.!?]", user_input or ""):
            text = _strip_memory_directive(part)
            if not _should_store_user_fact(text):
                continue
            normalized = _normalize_fact_text(text)
            if len(normalized.split()) < 3:
                continue
            if len(normalized) > 220:
                normalized = normalized[:220].strip() + "..."
            lines.append(normalized)
        out = []
        for text, _ in Counter(lines).most_common():
            out.append(text)
            if len(out) >= 8:
                break
        return out

    def _upsert_semantic_facts(self, user_input: str, response: str):
        now = _now_ts()
        candidates = self._extract_fact_candidates(user_input, response)
        if not candidates:
            return
        with self._connect(self._semantic_db) as conn:
            for statement in candidates:
                key = " ".join(sorted(_tokenize(statement)))
                if not key:
                    continue
                if _is_noise_text(statement) or _is_polluted_statement(statement):
                    continue
                row = conn.execute(
                    "SELECT id, confidence, provenance_json FROM facts WHERE canonical_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO facts(statement, canonical_key, confidence, first_seen, last_seen, provenance_json) VALUES(?, ?, ?, ?, ?, ?)",
                        (
                            statement,
                            key,
                            0.55,
                            now,
                            now,
                            json.dumps([{"ts": now, "source": "turn"}]),
                        ),
                    )
                else:
                    provenance = _json_load(row["provenance_json"], [])
                    provenance.append({"ts": now, "source": "turn"})
                    conn.execute(
                        "UPDATE facts SET confidence = ?, last_seen = ?, provenance_json = ? WHERE id = ?",
                        (
                            min(0.99, float(row["confidence"]) + 0.05),
                            now,
                            json.dumps(provenance[-15:]),
                            row["id"],
                        ),
                    )

    def run_idle_jobs(self):
        created, promoted = self._run_synthesis_job()
        decayed, deleted = self._apply_episodic_decay()
        return {
            "insight_candidates_created": created,
            "insights_promoted": promoted,
            "episodic_decayed": decayed,
            "episodic_deleted": deleted,
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
        }

    def cleanup_polluted_memory(self) -> dict:
        deleted_facts = 0
        deleted_insights = 0
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
        return {
            "deleted_semantic_facts": deleted_facts,
            "deleted_candidate_insights": deleted_insights,
        }

    def get_brain_dump(self, mode: str = "full", limit: int = 50) -> str:
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
        self._ltm_repo.clear()
        self._active_asked_insight_id = None
        self._turn_counter = 0
        self._last_proactive_turn = -99999
        self._last_proactive_ts = 0.0
