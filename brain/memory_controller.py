import atexit
import logging
import queue
import re
import threading
import time

from brain.stm import read_stm, append_stm, clear_stm
from brain.ltm_tree import LtmTree
from brain.consolidator import (
    consolidate_to_tree,
    detect_topic_shift,
    summarize_for_stm,
    _clean_topic_name,
)

logger = logging.getLogger("brain.memory_controller")
SAVE_INTERVAL = 5
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hey there|hi there|hello there|greetings|sup|yo|howdy|what'?s up)",
    re.IGNORECASE,
)


class MemoryController:
    def __init__(self, stm_file: str | None = None, tree_file: str | None = None):
        self._stm_file = stm_file
        self.tree = LtmTree(tree_file=tree_file)
        self.tree.load()
        self.current_branch_id = None
        self._has_meaningful_turn = False
        self._turns_since_save = 0
        self._lock = threading.RLock()
        self._turn_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        active = self.tree.get_most_active_branch()
        if active and len(active.get("content", [])) > 0:
            self.current_branch_id = active["id"]
            self._has_meaningful_turn = True

        atexit.register(self._shutdown_on_exit)

    def _shutdown_on_exit(self):
        self.wait_for_idle(timeout=2.0)
        self._stop_event.set()
        self._turn_queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self.tree.save()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._turn_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._turn_queue.task_done()
                break

            user_input, response = item
            try:
                self._process_turn(user_input, response)
            finally:
                self._turn_queue.task_done()

    def wait_for_idle(self, timeout: float = 2.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self._turn_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._turn_queue.unfinished_tasks == 0

    def _save_on_exit(self):
        self.tree.save()

    def _get_current_branch(self):
        if self.current_branch_id is None:
            return None
        return self.tree._find_branch(self.current_branch_id)

    def get_context_for_prompt(self, user_input):
        stm = read_stm(stm_file=self._stm_file)
        ltm_context = self.tree.retrieve_context(user_input)
        return {
            "working_memory": stm,
            "ltm_context": ltm_context,
        }

    def process_turn_async(self, user_input, response):
        self._turn_queue.put((user_input, response))

    def _is_greeting(self, text: str) -> bool:
        return bool(GREETING_PATTERNS.match(text.strip()))

    def _process_turn(self, user_input, response):
        try:
            snippet = summarize_for_stm(user_input, response)

            if not snippet or not snippet.strip():
                logger.debug("Empty snippet, skipping turn")
                return

            is_trivial_greeting = self._is_greeting(user_input) and len(snippet) < 15
            if is_trivial_greeting and not self._has_meaningful_turn:
                logger.debug("Greeting turn, skipping branch creation")
                return

            if not self._has_meaningful_turn:
                append_stm(user_input, response, stm_file=self._stm_file)

                if snippet.lower() == user_input.lower().strip():
                    logger.debug("Snippet echoes input, skipping branch")
                    return

                topic = _clean_topic_name(user_input[:40], "Conversation")
                new_branch = self.tree.add_branch(topic, [snippet])
                if new_branch:
                    self.current_branch_id = new_branch["id"]
                    self._has_meaningful_turn = True
                    logger.debug("Initial branch: %s", new_branch["topic"])
                self._maybe_save_tree()
                return

            current_branch = self._get_current_branch()
            shifted, new_topic = detect_topic_shift(user_input, current_branch)
            if shifted:
                stm_before_shift = read_stm(stm_file=self._stm_file)
                self._consolidate_and_reset(stm_before_shift)
                if new_topic:
                    clean_topic = _clean_topic_name(new_topic, "New Topic")
                else:
                    clean_topic = _clean_topic_name(user_input[:40], "New Topic")

                new_branch = self.tree.add_branch(clean_topic, [snippet])
                if new_branch:
                    self.current_branch_id = new_branch["id"]
                    logger.debug("New branch: %s", new_branch["topic"])

                append_stm(user_input, response, stm_file=self._stm_file)
            else:
                append_stm(user_input, response, stm_file=self._stm_file)

            self._maybe_save_tree()
        except Exception as e:
            logger.error("Error in _process_turn: %s", e)

    def _maybe_save_tree(self):
        with self._lock:
            self._turns_since_save += 1
            if self._turns_since_save >= SAVE_INTERVAL:
                self.tree.save()
                self._turns_since_save = 0

    def _consolidate_and_reset(self, stm_snapshot: str | None = None):
        logger.debug("Consolidating STM to tree")
        stm = (
            stm_snapshot
            if stm_snapshot is not None
            else read_stm(stm_file=self._stm_file)
        )
        if stm.strip():
            consolidate_to_tree(stm, self.tree, self.current_branch_id)
            self.tree.save()
        clear_stm(stm_file=self._stm_file)
        self.current_branch_id = None
        with self._lock:
            self._turns_since_save = 0

    def get_debug_info(self, mode="status"):
        if mode == "status":
            stm = read_stm(stm_file=self._stm_file)
            branch = self._get_current_branch()
            topic_str = branch["topic"] if branch else "None"
            return f"=== Short-Term Memory ===\n{stm if stm else '(empty)'}\n\nCurrent Branch: {topic_str}"
        elif mode == "ltm":
            return "=== Long-Term Memory ===\n" + self.tree.visualize()
        elif mode == "full":
            stm = read_stm(stm_file=self._stm_file)
            return (
                f"=== Full Memory ===\n\n--- STM ---\n{stm or '(empty)'}\n\n--- LTM ---\n"
                + self.tree.visualize()
            )
        return "Unknown debug mode. Use: status, ltm, full"
