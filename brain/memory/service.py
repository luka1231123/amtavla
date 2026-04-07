import json
import os
import re
import sqlite3
import time
from collections import Counter

from brain.config import load_brain_config
from tools.websearch import deep_crawl_websearch, tool_websearch


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

        self._turn_counter = 0
        self._last_proactive_turn = -99999
        self._last_proactive_ts = 0.0
        self._active_asked_insight_id = None

        research_cfg = self._config.get("research", {})
        self._deep_crawl_max_results = int(
            research_cfg.get("deep_crawl_max_results", 8)
        )
        self._deep_crawl_max_jobs_per_idle = int(
            research_cfg.get("deep_crawl_max_jobs_per_idle", 1)
        )

        self._init_schema()

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

    def enqueue_research_job(self, query: str, source: str = "cli") -> int:
        payload = {"query": query, "source": source}
        with self._connect(self._jobs_db) as conn:
            row = conn.execute(
                """
                INSERT INTO jobs(kind, payload_json, status, attempts, created_at)
                VALUES(?, ?, 'queued', 0, ?)
                RETURNING id
                """,
                ("research_deep_crawl", json.dumps(payload), _now_ts()),
            ).fetchone()
            return int(row["id"])

    def _run_research_jobs(self, max_jobs: int | None = None) -> dict:
        limit = max_jobs if max_jobs is not None else self._deep_crawl_max_jobs_per_idle
        processed = 0
        completed = 0
        failed = 0
        with self._connect(self._jobs_db) as conn:
            rows = conn.execute(
                """
                SELECT id, payload_json, attempts
                FROM jobs
                WHERE kind = 'research_deep_crawl' AND status = 'queued'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            for row in rows:
                processed += 1
                job_id = int(row["id"])
                attempts = int(row["attempts"] or 0) + 1
                conn.execute(
                    "UPDATE jobs SET status = 'running', attempts = ? WHERE id = ?",
                    (attempts, job_id),
                )
                payload = _json_load(row["payload_json"], {})
                query = payload.get("query", "")
                try:
                    result = deep_crawl_websearch(
                        query,
                        max_results=self._deep_crawl_max_results,
                    )
                    payload["result"] = result
                    payload["completed_at"] = _now_ts()
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'completed', payload_json = ?, finished_at = ?, error_text = NULL
                        WHERE id = ?
                        """,
                        (json.dumps(payload), _now_ts(), job_id),
                    )
                    completed += 1
                except Exception as exc:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed', finished_at = ?, error_text = ?
                        WHERE id = ?
                        """,
                        (_now_ts(), str(exc), job_id),
                    )
                    failed += 1
        return {
            "research_jobs_processed": processed,
            "research_jobs_completed": completed,
            "research_jobs_failed": failed,
        }

    def latest_research_result(self) -> dict | None:
        with self._connect(self._jobs_db) as conn:
            row = conn.execute(
                """
                SELECT id, payload_json, finished_at
                FROM jobs
                WHERE kind = 'research_deep_crawl' AND status = 'completed'
                ORDER BY finished_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        payload = _json_load(row["payload_json"], {})
        return {
            "job_id": int(row["id"]),
            "query": payload.get("query", ""),
            "result": payload.get("result", ""),
            "finished_at": row["finished_at"],
        }

    def research_status(self, limit: int = 10) -> dict:
        with self._connect(self._jobs_db) as conn:
            rows = conn.execute(
                """
                SELECT id, status, attempts, created_at, finished_at, error_text, payload_json
                FROM jobs
                WHERE kind = 'research_deep_crawl'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            count_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS c
                FROM jobs
                WHERE kind = 'research_deep_crawl'
                GROUP BY status
                """
            ).fetchall()

        counts = {row["status"]: int(row["c"]) for row in count_rows}
        items = []
        for row in rows:
            payload = _json_load(row["payload_json"], {})
            items.append(
                {
                    "job_id": int(row["id"]),
                    "status": row["status"],
                    "attempts": int(row["attempts"] or 0),
                    "query": payload.get("query", ""),
                    "created_at": row["created_at"],
                    "finished_at": row["finished_at"],
                    "error": row["error_text"] or "",
                }
            )

        return {"counts": counts, "jobs": items}

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

    def _build_candidate_thesis(self, facts: list[sqlite3.Row]) -> str:
        statements = [
            self._normalize_thesis(x["statement"]) for x in facts if x["statement"]
        ]
        statements = [
            s for s in statements if len(s.split()) >= 5 and not _is_memory_like_text(s)
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

    def _maybe_get_proactive_prompt(self, force: bool = False) -> dict:
        now = _now_ts()
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
        self._active_asked_insight_id = None

    def recall_context(
        self, query: str, include_web: bool = True, top_k: int = 5
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

        web_result = tool_websearch(query) if include_web else ""
        proactive = self._maybe_get_proactive_prompt(force=False)

        semantic_text = "\n".join(f"- {x['statement']}" for x in semantic)
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
        for raw in [user_input, response]:
            for part in re.split(r"[\n\.!?]", raw or ""):
                text = part.strip()
                if len(text.split()) < 4:
                    continue
                if _is_noise_text(text):
                    continue
                if len(text) > 220:
                    text = text[:220].strip() + "..."
                lines.append(text)
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
                if _is_noise_text(statement):
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
        research = self._run_research_jobs()
        created, promoted = self._run_synthesis_job()
        decayed, deleted = self._apply_episodic_decay()
        out = {
            "insight_candidates_created": created,
            "insights_promoted": promoted,
            "episodic_decayed": decayed,
            "episodic_deleted": deleted,
        }
        out.update(research)
        return out

    def _run_synthesis_job(self):
        with self._connect(self._semantic_db) as conn:
            facts = conn.execute(
                "SELECT id, statement, confidence FROM facts ORDER BY last_seen DESC LIMIT 40"
            ).fetchall()
        if len(facts) < 3:
            return 0, 0

        cluster = facts[:6]
        thesis = self._build_candidate_thesis(cluster)
        if not thesis:
            return 0, 0
        quality_score = self._quality_score(thesis)
        if quality_score < 0.45:
            return 0, 0

        evidence_query = " ".join(row["statement"] for row in cluster[:2])[:140]
        web_evidence = tool_websearch(evidence_query)
        has_web = bool(web_evidence and "failed" not in web_evidence.lower())
        avg_conf = sum(float(row["confidence"]) for row in cluster) / len(cluster)
        novelty_score = min(1.0, 0.45 + 0.05 * len(cluster))
        confidence = min(0.99, avg_conf + (0.1 if has_web else 0.0))

        status = (
            "promoted"
            if has_web and confidence >= 0.72 and novelty_score >= 0.65
            else "candidate"
        )
        feedback_state = "pending" if status == "candidate" else "approved"

        with self._connect(self._insight_db) as conn:
            existing = conn.execute(
                "SELECT id FROM insights WHERE thesis = ?", (thesis,)
            ).fetchone()
            if existing is not None:
                return 0, 0
            conn.execute(
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
        self._active_asked_insight_id = None
        self._turn_counter = 0
        self._last_proactive_turn = -99999
        self._last_proactive_ts = 0.0
