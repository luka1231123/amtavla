import json
import sys

GRAPH_FILE = "brain/ltm_graph.json"


def load_graph():
    try:
        with open(GRAPH_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Graph file not found: {GRAPH_FILE}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Invalid JSON in graph file")
        sys.exit(1)


def display_tree(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    if not nodes:
        print("Graph is empty.")
        return

    children = {}
    has_parent = set()
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        children.setdefault(src, []).append(tgt)
        has_parent.add(tgt)

    roots = [nid for nid in nodes if nid not in has_parent]
    if not roots:
        roots = list(nodes.keys())

    print(f"=== Knowledge Graph ({len(nodes)} nodes, {len(edges)} edges) ===\n")

    def _print_node(node_id, indent=0, visited=None):
        if visited is None:
            visited = set()
        node = nodes.get(node_id)
        if not node:
            return
        prefix = "  " * indent + ("├─ " if indent > 0 else "")
        strength_bar = "█" * int(node["strength"]) + "░" * max(
            0, 5 - int(node["strength"])
        )
        print(
            f"{prefix}{node['label']} [{strength_bar} {node['strength']:.1f}] ({node['category']})"
        )
        if node_id in visited:
            print("  " * (indent + 1) + "└─ (circular ref)")
            return
        visited.add(node_id)
        for child in children.get(node_id, []):
            _print_node(child, indent + 1, visited.copy())

    for root in roots:
        _print_node(root)
        print()


def display_table(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    print(f"=== Knowledge Graph ({len(nodes)} nodes, {len(edges)} edges) ===\n")

    if nodes:
        print(f"{'ID':<25} {'Label':<25} {'Cat':<12} {'Str':>5} {'Created':<12}")
        print("-" * 80)
        for nid, data in sorted(
            nodes.items(), key=lambda x: x[1]["strength"], reverse=True
        ):
            print(
                f"{nid:<25} {data['label']:<25} {data['category']:<12} {data['strength']:>5.1f} {data['created_at']:<12.0f}"
            )

    if edges:
        print(f"\n{'Source':<25} {'Relation':<15} {'Target':<25} {'Str':>5}")
        print("-" * 70)
        for edge in edges:
            print(
                f"{edge['source']:<25} {edge['relation']:<15} {edge['target']:<25} {edge['strength']:>5.1f}"
            )


def display_ascii_graph(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    if not nodes:
        print("Graph is empty.")
        return

    node_list = list(nodes.keys())
    node_idx = {nid: i for i, nid in enumerate(node_list)}
    n = len(node_list)

    width = 80
    height = 25
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    import math

    cx, cy = width // 2, height // 2
    radius_x = min(width // 2 - 5, 30)
    radius_y = min(height // 2 - 2, 10)

    positions = {}
    for i, nid in enumerate(node_list):
        angle = 2 * math.pi * i / n - math.pi / 2
        x = int(cx + radius_x * math.cos(angle))
        y = int(cy + radius_y * math.sin(angle))
        positions[nid] = (x, y)

    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in positions and tgt in positions:
            x1, y1 = positions[src]
            x2, y2 = positions[tgt]
            steps = max(abs(x2 - x1), abs(y2 - y1))
            for s in range(steps + 1):
                t = s / max(steps, 1)
                x = int(x1 + (x2 - x1) * t)
                y = int(y1 + (y2 - y1) * t)
                if 0 <= y < height and 0 <= x < width:
                    canvas[y][x] = "─" if abs(x2 - x1) > abs(y2 - y1) else "│"

    for nid, (x, y) in positions.items():
        label = nodes[nid]["label"][:8]
        label_x = max(0, min(x - len(label) // 2, width - len(label)))
        label_y = max(0, min(y, height - 1))
        for i, ch in enumerate(label):
            if label_x + i < width:
                canvas[label_y][label_x + i] = ch

    print(f"=== Knowledge Graph (ASCII) [{len(nodes)} nodes, {len(edges)} edges] ===\n")
    for row in canvas:
        print("".join(row))


def main():
    graph = load_graph()
    mode = sys.argv[1] if len(sys.argv) > 1 else "tree"

    if mode == "tree":
        display_tree(graph)
    elif mode == "table":
        display_table(graph)
    elif mode == "ascii":
        display_ascii_graph(graph)
    else:
        print(f"Usage: {sys.argv[0]} [tree|table|ascii]")
        sys.exit(1)


if __name__ == "__main__":
    main()
