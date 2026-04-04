import json
import os
import time

LTM_GRAPH_FILE = "brain/ltm_graph.json"


class KnowledgeGraph:
    def __init__(self):
        self.graph = self._load()

    def _load(self):
        if os.path.exists(LTM_GRAPH_FILE):
            with open(LTM_GRAPH_FILE, "r") as f:
                return json.load(f)
        return {"nodes": {}, "edges": []}

    def _save(self):
        with open(LTM_GRAPH_FILE, "w") as f:
            json.dump(self.graph, f, indent=2)

    def add_node(self, node_id, label, category="general", strength=1.0):
        if node_id not in self.graph["nodes"]:
            self.graph["nodes"][node_id] = {
                "label": label,
                "category": category,
                "strength": strength,
                "created_at": time.time(),
                "accessed_at": time.time(),
            }
            self._save()
            print(
                f"   [DEBUG-LTM] -> New node: '{label}' ({category}, strength={strength})"
            )
            return True
        return False

    def update_node(self, node_id, label=None, strength=None, category=None):
        if node_id in self.graph["nodes"]:
            node = self.graph["nodes"][node_id]
            if label is not None:
                node["label"] = label
            if strength is not None:
                node["strength"] = strength
            if category is not None:
                node["category"] = category
            node["accessed_at"] = time.time()
            self._save()
            return True
        return False

    def reinforce_node(self, node_id, amount=0.5):
        if node_id in self.graph["nodes"]:
            node = self.graph["nodes"][node_id]
            node["strength"] = min(node["strength"] + amount, 10.0)
            node["accessed_at"] = time.time()
            self._save()
            print(
                f"   [DEBUG-LTM] -> Reinforced '{node['label']}' to strength {node['strength']:.1f}"
            )
            return node["strength"]
        return 0

    def add_edge(self, source, target, relation="related"):
        for edge in self.graph["edges"]:
            if edge["source"] == source and edge["target"] == target:
                edge["strength"] = min(edge.get("strength", 1.0) + 0.3, 5.0)
                self._save()
                print(
                    f"   [DEBUG-LTM] -> Reinforced edge: {source} -[{relation}]-> {target}"
                )
                return False
        self.graph["edges"].append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "strength": 1.0,
            }
        )
        self._save()
        print(f"   [DEBUG-LTM] -> New edge: {source} -[{relation}]-> {target}")
        return True

    def get_relevant_context(self, keywords, top_n=5):
        scored_nodes = []
        for node_id, node in self.graph["nodes"].items():
            score = 0
            label_lower = node["label"].lower()
            for kw in keywords:
                if kw.lower() in label_lower:
                    score += 2
            score += node["strength"]
            if score > 0:
                scored_nodes.append((node_id, node, score))
        scored_nodes.sort(key=lambda x: x[2], reverse=True)
        result = []
        for node_id, node, score in scored_nodes[:top_n]:
            connected_edges = [
                e
                for e in self.graph["edges"]
                if e["source"] == node_id or e["target"] == node_id
            ]
            result.append(
                {
                    "id": node_id,
                    "label": node["label"],
                    "category": node["category"],
                    "strength": node["strength"],
                    "connections": [
                        {
                            "relation": e["relation"],
                            "other": e["target"]
                            if e["source"] == node_id
                            else e["source"],
                        }
                        for e in connected_edges[:3]
                    ],
                }
            )
        return result

    def decay_unused_nodes(self, decay_rate=0.1, min_strength=0.5):
        now = time.time()
        one_day = 86400
        removed = []
        for node_id, node in list(self.graph["nodes"].items()):
            age_days = (now - node["accessed_at"]) / one_day
            if age_days > 0:
                node["strength"] -= decay_rate * age_days
            if node["strength"] < min_strength:
                removed.append(node_id)
                del self.graph["nodes"][node_id]
                self.graph["edges"] = [
                    e
                    for e in self.graph["edges"]
                    if e["source"] != node_id and e["target"] != node_id
                ]
        if removed:
            self._save()
            print(
                f"   [DEBUG-LTM] -> Decayed and removed {len(removed)} nodes: {removed}"
            )
        return removed

    def get_top_nodes(self, top_n=5):
        nodes = sorted(
            self.graph["nodes"].items(),
            key=lambda x: x[1]["strength"],
            reverse=True,
        )
        return [{"id": nid, **data} for nid, data in nodes[:top_n]]

    def get_all(self):
        return self.graph
