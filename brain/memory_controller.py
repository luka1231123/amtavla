import threading
from brain.stm import read_stm, append_stm, clear_stm
from brain.ltm import get_relevant_lines, append_ltm
from brain.consolidator import (
    promote_stm_to_ltm,
    detect_topic_shift,
    summarize_for_stm,
)


class MemoryController:
    def __init__(self):
        self.current_topic = None
        self.turn_count = 0
        self._memory_thread = None

    def get_context_for_prompt(self, user_input):
        stm = read_stm()
        relevant_ltm = get_relevant_lines(user_input)
        ltm_context = ""
        if relevant_ltm:
            print(
                f"   [DEBUG-BRAIN] -> Retrieved {len(relevant_ltm)} relevant LTM lines for: {user_input[:40]}"
            )
            ltm_lines = []
            for score, line in relevant_ltm:
                ltm_lines.append(line)
            ltm_context = "\nLong-Term Memory:\n" + "\n".join(ltm_lines)
        return {
            "working_memory": stm,
            "ltm_context": ltm_context,
        }

    def process_turn_async(self, user_input, response):
        if self._memory_thread and self._memory_thread.is_alive():
            print("   [DEBUG-BRAIN] -> Waiting for previous memory op to finish...")
            self._memory_thread.join()

        self._memory_thread = threading.Thread(
            target=self._process_turn, args=(user_input, response), daemon=True
        )
        self._memory_thread.start()

    def _process_turn(self, user_input, response):
        self.turn_count += 1
        snippet = summarize_for_stm(user_input, response)
        append_stm(snippet)

        if self.turn_count == 1:
            self.current_topic = user_input[:50]
            print(f"   [DEBUG-BRAIN] -> Initial topic: {self.current_topic}")
            return

        shifted, new_topic = detect_topic_shift(user_input, self.current_topic)
        if shifted:
            self._consolidate_and_reset(new_topic)
        else:
            self.current_topic = new_topic

    def _consolidate_and_reset(self, new_topic):
        print(f"   [DEBUG-BRAIN] -> Consolidating, shifting topic to: {new_topic}")
        stm = read_stm()
        if stm.strip():
            promoted = promote_stm_to_ltm(stm, self.current_topic or "general")
            if promoted:
                append_ltm(promoted)
        clear_stm()
        self.current_topic = new_topic

    def get_debug_info(self, mode="status"):
        if mode == "status":
            stm = read_stm()
            return f"=== Short-Term Memory ===\n{stm if stm else '(empty)'}\n\nCurrent Topic: {self.current_topic or 'None'}"
        elif mode == "ltm":
            from brain.ltm import read_ltm

            lines = read_ltm()
            if not lines:
                return "=== Long-Term Memory ===\n(empty)"
            return "=== Long-Term Memory ===\n" + "\n".join(lines)
        elif mode == "full":
            from brain.ltm import read_ltm

            lines = read_ltm()
            stm = read_stm()
            return (
                f"=== Full Memory ===\n\n--- STM ---\n{stm or '(empty)'}\n\n--- LTM ({len(lines)} lines) ---\n"
                + "\n".join(lines)
                if lines
                else "(empty)"
            )
        return "Unknown debug mode. Use: status, ltm, full"
