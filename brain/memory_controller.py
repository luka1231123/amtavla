import atexit
import logging
import re
import threading

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
        self.turn_count = 0
        self._cached_ltm_context = ""
        self._turns_since_save = 0
        self._lock = threading.Lock()

        active = self.tree.get_most_active_branch()
        if active and len(active.get("content", [])) > 0:
            self.current_branch_id = active["id"]

        atexit.register(self._save_on_exit)

    def _save_on_exit(self):
        self.tree.save()

    def _get_current_branch(self):
        if self.current_branch_id is None:
            return None
        return self.tree._find_branch(self.current_branch_id)

    def get_context_for_prompt(self, user_input):
        stm = read_stm(stm_file=self._stm_file)
        if self._cached_ltm_context:
            ltm_context = self._cached_ltm_context
        else:
            ltm_context = self.tree.retrieve_context(user_input)
        return {
            "working_memory": stm,
            "ltm_context": ltm_context,
        }

    def process_turn_async(self, user_input, response):
        t = threading.Thread(
            target=self._process_turn, args=(user_input, response), daemon=True
        )
        t.start()

    def _is_greeting(self, text: str) -> bool:
        return bool(GREETING_PATTERNS.match(text.strip()))

    def _process_turn(self, user_input, response):
        try:
            with self._lock:
                self.turn_count += 1
            snippet = summarize_for_stm(user_input, response)

            if not snippet or not snippet.strip():
                logger.debug("Empty snippet, skipping turn")
                return

            append_stm(user_input, response, stm_file=self._stm_file)

            current_branch = self._get_current_branch()

            if self.turn_count == 1:
                if self._is_greeting(user_input) and len(snippet) < 15:
                    logger.debug("Greeting turn, skipping branch creation")
                    return
                if snippet.lower() == user_input.lower().strip():
                    logger.debug("Snippet echoes input, skipping branch")
                    return

                topic = _clean_topic_name(user_input[:40], "Conversation")
                new_branch = self.tree.add_branch(topic, [snippet])
                if new_branch:
                    self.current_branch_id = new_branch["id"]
                    logger.debug("Initial branch: %s", new_branch["topic"])
                    self._cached_ltm_context = self.tree.retrieve_context(user_input)
                return

            shifted, new_topic = detect_topic_shift(user_input, current_branch)
            if shifted:
                self._consolidate_and_reset()
                if new_topic:
                    clean_topic = _clean_topic_name(new_topic, "New Topic")
                    new_branch = self.tree.add_branch(clean_topic, [snippet])
                    if new_branch:
                        self.current_branch_id = new_branch["id"]
                        logger.debug("New branch: %s", new_branch["topic"])
                        self._cached_ltm_context = self.tree.retrieve_context(
                            user_input
                        )
                else:
                    self._cached_ltm_context = ""
            else:
                self._cached_ltm_context = self.tree.retrieve_context(user_input)

            with self._lock:
                self._turns_since_save += 1
                if self._turns_since_save >= SAVE_INTERVAL:
                    self.tree.save()
                    self._turns_since_save = 0
        except Exception as e:
            logger.error("Error in _process_turn: %s", e)

    def _consolidate_and_reset(self):
        logger.debug("Consolidating STM to tree")
        stm = read_stm(stm_file=self._stm_file)
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
