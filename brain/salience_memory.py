import re

from brain.config import load_brain_config


PREFERENCE_RE = re.compile(
    r"\b(i (like|love|prefer)|my favorite|call me)\b", re.IGNORECASE
)
TASK_RE = re.compile(
    r"\b(i need|remind me|don't forget|todo|to do|must|deadline)\b", re.IGNORECASE
)
CONSTRAINT_RE = re.compile(
    r"\b(budget|under \$?\d+|must|should|cannot|can't|location|city|timezone|version)\b",
    re.IGNORECASE,
)
ENTITY_RE = re.compile(
    r"(\b\d{1,2}:\d{2}\b|\b\d+(?:\.\d+)?\b|\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b|https?://\S+)",
    re.IGNORECASE,
)
STOP_FALLBACK = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "is",
    "are",
    "i",
    "you",
    "it",
    "we",
    "they",
}


def _normalize_for_similarity(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in STOP_FALLBACK and len(w) > 2}


def _too_similar(
    new_text: str, existing_lines: list[str], threshold: float = 0.8
) -> bool:
    new_tokens = _normalize_for_similarity(new_text)
    if not new_tokens:
        return False
    for line in existing_lines:
        old_tokens = _normalize_for_similarity(line)
        if not old_tokens:
            continue
        inter = len(new_tokens & old_tokens)
        union = len(new_tokens | old_tokens)
        if union > 0 and (inter / union) >= threshold:
            return True
    return False


def _score_line(text: str) -> int:
    score = 0
    if PREFERENCE_RE.search(text):
        score += 3
    if TASK_RE.search(text):
        score += 3
    if CONSTRAINT_RE.search(text):
        score += 2
    if ENTITY_RE.search(text):
        score += 2
    if len(text.split()) >= 8:
        score += 1
    return score


def extract_salient_facts(
    user_input: str,
    response: str,
    existing_stm_lines: list[str] | None = None,
    config: dict | None = None,
) -> tuple[list[str], list[dict]]:
    cfg = config or load_brain_config()
    salience_cfg = cfg.get("salience", {})
    max_facts = int(salience_cfg.get("max_facts_per_turn", 3))
    min_score = int(salience_cfg.get("min_score", 3))
    max_fact_length = int(salience_cfg.get("max_fact_length", 140))
    ignore_patterns = [
        re.compile(p, re.IGNORECASE) for p in salience_cfg.get("ignore_regex", [])
    ]

    existing = existing_stm_lines or []
    candidates = [
        ("user", user_input.strip()),
        ("assistant", response.strip()),
    ]

    kept = []
    debug = []
    for origin, raw in candidates:
        if not raw:
            continue

        ignored = any(p.search(raw) for p in ignore_patterns)
        score = _score_line(raw)
        reason = []
        if PREFERENCE_RE.search(raw):
            reason.append("preference")
        if TASK_RE.search(raw):
            reason.append("task")
        if CONSTRAINT_RE.search(raw):
            reason.append("constraint")
        if ENTITY_RE.search(raw):
            reason.append("entity")
        if ignored:
            reason.append("ignored_pattern")

        text = raw[:max_fact_length]
        if len(raw) > max_fact_length:
            text += "..."
        if origin == "user":
            text = f"user: {text}"
        else:
            text = f"assistant: {text}"

        is_similar = _too_similar(text, existing + kept)
        should_keep = (not ignored) and score >= min_score and (not is_similar)

        debug.append(
            {
                "origin": origin,
                "raw": raw,
                "score": score,
                "reason": reason,
                "similar": is_similar,
                "kept": should_keep,
            }
        )

        if should_keep:
            kept.append(text)
            if len(kept) >= max_facts:
                break

    return kept, debug
