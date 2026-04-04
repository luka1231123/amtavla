from brain.stm import (
    read_working_memory,
    append_to_working_memory,
    clear_working_memory,
)
from brain.ltm import KnowledgeGraph
from brain.consolidator import (
    consolidate_stm_to_ltm,
    detect_topic_shift,
    summarize_for_working_memory,
)


class MemoryController:
    def __init__(self):
        self.graph = KnowledgeGraph()
        self.current_topic = None
        self.turn_count = 0

    def get_context_for_prompt(self, user_input):
        working_memory = read_working_memory()
        keywords = user_input.lower().split()
        relevant_ltm = self.graph.get_relevant_context(keywords, top_n=5)
        ltm_context = ""
        if relevant_ltm:
            print(
                f"   [DEBUG-BRAIN] -> Retrieved {len(relevant_ltm)} relevant LTM nodes for: {user_input[:40]}"
            )
            ltm_parts = []
            for node in relevant_ltm:
                self.graph.reinforce_node(node["id"], amount=0.1)
                conn_str = ", ".join(
                    f"{c['relation']} -> {c['other']}" for c in node["connections"]
                )
                ltm_parts.append(
                    f"  - {node['label']} ({node['category']}): {conn_str}"
                )
            ltm_context = "\nRelevant Knowledge:\n" + "\n".join(ltm_parts)
        return {
            "working_memory": working_memory,
            "ltm_context": ltm_context,
        }

    def process_turn(self, user_input, response):
        self.turn_count += 1
        snippet = summarize_for_working_memory(user_input, response)
        append_to_working_memory(snippet)

        if self.turn_count == 1:
            self.current_topic = user_input[:50]
            print(f"   [DEBUG-BRAIN] -> Initial topic set: {self.current_topic}")
            return

        shifted, new_topic = detect_topic_shift(user_input, self.current_topic)
        if shifted:
            self.consolidate_and_reset(new_topic)
        else:
            self.current_topic = new_topic

    def consolidate_and_reset(self, new_topic):
        print(
            f"   [DEBUG-BRAIN] -> Consolidating memory, shifting topic to: {new_topic}"
        )
        working_memory = read_working_memory()
        if working_memory.strip():
            facts = consolidate_stm_to_ltm(
                working_memory, self.current_topic or "general"
            )
            for node in facts.get("nodes", []):
                self.graph.add_node(
                    node_id=node["id"],
                    label=node["label"],
                    category=node.get("category", "general"),
                    strength=node.get("strength", 1.0),
                )
            for edge in facts.get("edges", []):
                self.graph.add_edge(
                    source=edge["source"],
                    target=edge["target"],
                    relation=edge.get("relation", "related"),
                )
        clear_working_memory()
        self.current_topic = new_topic

    def get_debug_info(self, mode="status"):
        if mode == "status":
            wm = read_working_memory()
            return f"=== Working Memory ===\n{wm if wm else '(empty)'}\n\nCurrent Topic: {self.current_topic or 'None'}"
        elif mode == "graph":
            top = self.graph.get_top_nodes(top_n=5)
            if not top:
                return "=== LTM Graph ===\n(no nodes)"
            lines = ["=== LTM Graph (Top 5 Nodes) ==="]
            for node in top:
                lines.append(
                    f"  [{node['strength']:.1f}] {node['label']} ({node['category']})"
                )
            return "\n".join(lines)
        elif mode == "full":
            all_data = self.graph.get_all()
            return f"=== Full LTM Graph ===\nNodes: {len(all_data['nodes'])}\nEdges: {len(all_data['edges'])}\n\n{all_data}"
        return "Unknown debug mode. Use: status, graph, full"
