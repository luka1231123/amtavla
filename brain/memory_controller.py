import atexit
import logging
import queue
import threading
import time

from brain.config import load_brain_config
from brain.memory import MemoryService

logger = logging.getLogger("brain.memory_controller")


class MemoryController:
    def __init__(self, stm_file: str | None = None, tree_file: str | None = None):
        del stm_file
        del tree_file
        self._config = load_brain_config()
        db_dir = self._config.get("memory", {}).get("db_dir", "brain/db")
        self.memory = MemoryService(db_dir=db_dir)
        idle_cfg = self._config.get("idle_memory", {})
        self._idle_enabled = bool(idle_cfg.get("enabled", True))
        self._idle_seconds = float(idle_cfg.get("idle_seconds", 4.0))
        self._idle_poll_seconds = float(idle_cfg.get("poll_seconds", 0.5))
        self._idle_min_interval_seconds = float(
            idle_cfg.get("min_interval_seconds", 5.0)
        )
        self._reminder_tick_seconds = float(
            idle_cfg.get("reminder_tick_seconds", 2.0)
        )

        self._last_activity_ts = time.time()
        self._last_idle_run_ts = 0.0
        self._turn_queue: queue.Queue[tuple[str, str, dict] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._closed = False
        self._debug_hook = None
        self._proactive_hook = None
        self._foreground_lock = threading.Lock()
        self._foreground_active = 0

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self._idle_thread.start()
        # Reminders get their own lightweight tick so a due time fires within
        # seconds even while the user is chatting or heavy idle work is failing.
        self._reminder_thread = threading.Thread(
            target=self._reminder_loop, daemon=True
        )
        self._reminder_thread.start()

        atexit.register(self._shutdown_on_exit)

    def set_debug_hook(self, debug_hook):
        self._debug_hook = debug_hook

    def set_proactive_hook(self, proactive_hook):
        """Called with each message the idle jobs queue for the user (reminders,
        finished research). The hook delivers it to whatever UI is attached."""
        self._proactive_hook = proactive_hook

    def _deliver_proactive_messages(self):
        messages = self.memory.drain_proactive_messages()
        if not messages:
            return
        if self._proactive_hook is None:
            # No UI attached yet: keep them queued instead of dropping.
            self.memory._proactive_outbox[:0] = messages
            return
        for message in messages:
            try:
                self._proactive_hook(message)
            except Exception:
                logger.debug("Proactive hook failed", exc_info=True)

    def _emit_debug(self, event_type: str, payload: dict):
        if self._debug_hook is None:
            return
        try:
            self._debug_hook(event_type, payload)
        except Exception:
            logger.debug("Memory debug hook failed", exc_info=True)

    def note_user_activity(self):
        self._last_activity_ts = time.time()

    def begin_foreground_turn(self):
        self.note_user_activity()
        with self._foreground_lock:
            self._foreground_active += 1

    def end_foreground_turn(self):
        with self._foreground_lock:
            if self._foreground_active > 0:
                self._foreground_active -= 1
        self.note_user_activity()

    def _shutdown_on_exit(self):
        if self._closed:
            return
        self._closed = True
        self.wait_for_idle(timeout=2.0)
        self._stop_event.set()
        self._turn_queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self._idle_thread.is_alive():
            self._idle_thread.join(timeout=1.0)
        if self._reminder_thread.is_alive():
            self._reminder_thread.join(timeout=1.0)

    def close(self):
        self._shutdown_on_exit()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._turn_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._turn_queue.task_done()
                break

            user_input, response, trace = item
            try:
                self.memory.commit_turn(user_input, response, trace=trace)
            except Exception:
                # A single bad commit must never kill the worker: if it did, all
                # future memory commits would silently stop for the rest of the
                # process's life. Log and drop this turn, keep consuming.
                logger.exception("commit_turn failed for a turn; worker continues")
            finally:
                self._turn_queue.task_done()

    def _reminder_loop(self):
        while not self._stop_event.is_set():
            try:
                fired = self.memory.fire_due_reminders_now()
                # Research normally waits for a quiet idle window; force-start
                # jobs that have waited over a minute so continuous chatting
                # can't delay them forever. Never while a turn is in flight —
                # research shares the model with the foreground reply.
                with self._foreground_lock:
                    foreground_busy = self._foreground_active > 0
                overdue = (
                    0
                    if foreground_busy
                    else self.memory.run_overdue_research(min_age_seconds=60.0)
                )
                if fired or overdue:
                    self._deliver_proactive_messages()
            except Exception:
                logger.debug("Reminder tick failed", exc_info=True)
            time.sleep(self._reminder_tick_seconds)

    def _idle_loop(self):
        while not self._stop_event.is_set():
            if not self._idle_enabled:
                time.sleep(0.5)
                continue
            try:
                self._run_idle_memory_maintenance()
            except Exception:
                logger.debug("Idle memory maintenance failed", exc_info=True)
            time.sleep(self._idle_poll_seconds)

    def _run_idle_memory_maintenance(self):
        if self._turn_queue.unfinished_tasks > 0:
            return
        with self._foreground_lock:
            if self._foreground_active > 0:
                return
        idle_for = time.time() - self._last_activity_ts
        if idle_for < self._idle_seconds:
            return
        now = time.time()
        if (now - self._last_idle_run_ts) < self._idle_min_interval_seconds:
            return

        self._last_idle_run_ts = now
        metrics = self.memory.run_idle_jobs()
        self._deliver_proactive_messages()
        logger.debug("Idle memory maintenance metrics: %s", metrics)

    def run_idle_now(self) -> dict:
        self.note_user_activity()
        metrics = self.memory.run_idle_jobs()
        self._deliver_proactive_messages()
        status = self.memory.get_status()
        return {"status": status, "metrics": metrics}

    def force_proactive_ask(self) -> dict:
        return self.memory.force_proactive_ask()

    def get_brain_dump(self, mode: str = "full", limit: int = 50) -> str:
        return self.memory.get_brain_dump(mode=mode, limit=limit)

    def clear_all_memory(self):
        self.memory.clear_all_memory()

    def wait_for_idle(self, timeout: float = 2.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self._turn_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._turn_queue.unfinished_tasks == 0

    def get_context_for_prompt(
        self,
        user_input,
        include_web: bool = True,
        intent: str = "",
        pathway: str = "",
        search_cache: dict | None = None,
    ):
        recall = self.memory.recall_context(
            user_input,
            include_web=include_web,
            current_intent=intent,
            current_pathway=pathway,
            search_cache=search_cache,
        )
        return {
            "memory_items": recall.get("memory_items", []),
            "semantic_facts": recall["semantic"],
            "episodic_context": recall["episodic"],
            "ltm_context": recall["insights"],
            "web_context": recall["web"],
            "web_results": recall.get("web_results", []),
            "conversation": recall.get("conversation", []),
            "combined_context": recall["combined_context"],
            "style_context": recall.get("style_context", ""),
            "pending_feedback_prompt": recall["pending_feedback_prompt"],
            "pending_feedback_id": recall.get("pending_feedback_id"),
            "reminder_nudge": recall.get("reminder_nudge", ""),
        }

    def apply_insight_feedback(self, insight_id: int, keep: bool) -> dict:
        self.note_user_activity()
        return self.memory.apply_insight_feedback(insight_id, keep)

    def search_memory(self, query: str, top_k: int = 5) -> dict:
        recall = self.memory.search_memory(query, top_k=top_k)
        return {
            "memory_items": recall.get("memory_items", []),
            "semantic_facts": recall["semantic"],
            "episodic_context": recall["episodic"],
            "ltm_context": recall["insights"],
            "web_context": recall["web"],
            "web_results": recall.get("web_results", []),
            "combined_context": recall["combined_context"],
            "pending_feedback_prompt": recall["pending_feedback_prompt"],
            "pending_feedback_id": recall.get("pending_feedback_id"),
        }

    def write_memory(self, statement: str) -> dict:
        self.note_user_activity()
        return self.memory.write_memory(statement)

    def recent_notes(self, limit: int = 20) -> list[dict]:
        return self.memory.recent_notes(limit=limit)

    def recent_dialogue(
        self, limit: int | None = None, within_seconds: float | None = None
    ) -> list[dict]:
        # Turns commit on a background worker; drain it first so the buffer
        # reflects the immediately-preceding turn even when the user fires two
        # messages in quick succession (otherwise a fast follow-up would resolve
        # against stale context, or none at all).
        self.wait_for_idle(timeout=2.0)
        return self.memory.recent_dialogue(limit=limit, within_seconds=within_seconds)

    def create_reminder(self, content: str, *, due_at: float | None = None) -> dict:
        self.note_user_activity()
        return self.memory.create_reminder(content, due_at=due_at)

    def queue_research(self, topic: str) -> dict:
        self.note_user_activity()
        return self.memory.queue_research(topic)

    def list_memory_items(self, **filters) -> list[dict]:
        return self.memory.list_memory_items(**filters)

    def inspect_memory_item(self, item_id: int) -> dict:
        return self.memory.inspect_memory_item(item_id)

    def correct_memory_item(self, item_id: int, content: str, reason: str = "") -> dict:
        return self.memory.correct_memory_item(item_id, content, reason)

    def set_memory_review_state(
        self, item_id: int, review_state: str, note: str = ""
    ) -> dict:
        return self.memory.set_memory_review_state(item_id, review_state, note)

    def merge_memory_items(
        self,
        target_id: int,
        source_ids: list[int],
        content: str | None = None,
        note: str = "",
    ) -> dict:
        return self.memory.merge_memory_items(
            target_id,
            source_ids,
            content=content,
            note=note,
        )

    def export_memory(self, export_dir: str) -> dict:
        return self.memory.export_memory(export_dir)

    def memory_overview(self) -> dict:
        return self.memory.memory_overview()

    def daily_brief(self) -> str:
        return self.memory.daily_brief()

    def review_drill(self, limit: int = 3) -> str:
        return self.memory.review_drill(limit=limit)

    def start_focus(self, minutes: float) -> float:
        return self.memory.start_focus(minutes)

    def end_focus(self):
        self.memory.end_focus()

    def in_focus(self) -> bool:
        return self.memory.in_focus()

    def open_commitments(self) -> list[dict]:
        return self.memory.open_commitments()

    def complete_commitment(self, item_id: int) -> dict:
        return self.memory.complete_commitment(item_id)

    def capture_note(self, content: str, **kwargs) -> dict:
        self.note_user_activity()
        return self.memory.capture_note(content, **kwargs)

    def process_turn_async(self, user_input, response, trace: dict | None = None):
        self.note_user_activity()
        self._turn_queue.put((user_input, response, trace or {}))

    def get_debug_info(self, mode="status"):
        status = self.memory.get_status()
        if mode == "status":
            return (
                "=== Memory Status ===\n"
                f"Episodic events: {status['episodic_count']}\n"
                f"Semantic facts: {status['semantic_count']}\n"
                f"Promoted insights: {status['insight_count']}\n"
                f"Pending insight feedback: {status['pending_feedback_count']}"
                f"\nMemory catalog: {sum(status['memory_catalog']['by_state'].values())}"
                f"\nEntities: {status['memory_catalog']['entity_count']}"
                f"\nEmbedding available: {status['embedding_available']}"
                f"\nEmbedding error: {status['embedding_last_error'] or '(none)'}"
            )
        if mode == "ltm":
            recall = self.memory.recall_context("", include_web=False)
            if not recall["insights"]:
                return "=== Long-Term Insights ===\n(empty)"
            lines = ["=== Long-Term Insights ==="]
            for item in recall["insights"]:
                lines.append(f"- {item['thesis']} (conf={item['confidence']:.2f})")
            return "\n".join(lines)
        if mode == "full":
            recall = self.memory.recall_context("", include_web=False)
            return (
                "=== Full Memory ===\n\n"
                f"Status: {status}\n\n"
                "--- Semantic ---\n"
                + "\n".join(f"- {x['statement']}" for x in recall["semantic"])
                + "\n\n--- Episodic ---\n"
                + "\n".join(
                    f"- {x['user_input']} -> {x['response']}"
                    for x in recall["episodic"]
                )
            )
        return "Unknown debug mode. Use: status, ltm, full"
