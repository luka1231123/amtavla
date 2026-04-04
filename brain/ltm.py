import os
import re

LTM_FILE = "brain/ltm.txt"
JACCARD_THRESHOLD = 0.05


def _tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return set(tokens)


def _jaccard(a, b):
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _parse_importance(line):
    match = re.search(r"importance:\s*([\d.]+)", line)
    return float(match.group(1)) if match else 1.0


def read_ltm():
    if not os.path.exists(LTM_FILE):
        return []
    with open(LTM_FILE, "r") as f:
        return [l.strip() for l in f if l.strip()]


def append_ltm(lines):
    if isinstance(lines, str):
        lines = [lines]
    with open(LTM_FILE, "a") as f:
        for line in lines:
            if line.strip():
                f.write(line.strip() + "\n")


def get_relevant_lines(query, threshold=JACCARD_THRESHOLD):
    all_lines = read_ltm()
    scored = []
    for line in all_lines:
        j = _jaccard(query, line)
        importance = _parse_importance(line)
        score = j * importance
        if score > threshold:
            scored.append((score, line))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(score, line) for score, line in scored]


def clear_ltm():
    if os.path.exists(LTM_FILE):
        os.remove(LTM_FILE)
