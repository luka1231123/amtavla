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

        self._last_activity_ts = time.time()
        self._last_idle_run_ts = 0.0
        self._turn_queue: queue.Queue[tuple[str, str, dict] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._debug_hook = None
        self._foreground_lock = threading.Lock()
        self._foreground_active = 0

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self._idle_thread.start()

        atexit.register(self._shutdown_on_exit)

    def set_debug_hook(self, debug_hook):
        self._debug_hook = debug_hook

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
        self.wait_for_idle(timeout=2.0)
        self._stop_event.set()
        self._turn_queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self._idle_thread.is_alive():
            self._idle_thread.join(timeout=1.0)

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
            finally:
                self._turn_queue.task_done()

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

        metrics = self.memory.run_idle_jobs()
        self._last_idle_run_ts = now
        logger.debug("Idle memory maintenance metrics: %s", metrics)

    def run_idle_now(self) -> dict:
        self.note_user_activity()
        metrics = self.memory.run_idle_jobs()
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
            "semantic_facts": recall["semantic"],
            "episodic_context": recall["episodic"],
            "ltm_context": recall["insights"],
            "web_context": recall["web"],
            "combined_context": recall["combined_context"],
            "pending_feedback_prompt": recall["pending_feedback_prompt"],
        }

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
