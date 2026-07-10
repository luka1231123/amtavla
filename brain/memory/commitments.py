from __future__ import annotations

import datetime
import re

_WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
)

# (pattern, confidence) — ordered from strongest to weakest signal.
_COMMITMENT_PATTERNS = (
    (re.compile(r"\bremind me to ([^,.!?]+)", re.IGNORECASE), 0.9),
    (re.compile(r"\bdon'?t let me forget (?:to )?([^,.!?]+)", re.IGNORECASE), 0.9),
    (re.compile(r"\bi promised (?:\w+ )?(?:to )?([^,.!?]+)", re.IGNORECASE), 0.8),
    (re.compile(r"\bi (?:need|have|got) to ([^,.!?]+)", re.IGNORECASE), 0.65),
    (re.compile(r"\bi(?:'ll| will) ([^,.!?]+)", re.IGNORECASE), 0.5),
)

_DEADLINE_RE = re.compile(
    r"\b(?:by|before|until|due)\s+(tonight|today|tomorrow|next week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"20\d{2}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def _resolve_deadline(phrase: str, now: float) -> float | None:
    phrase = phrase.lower().strip()
    today = datetime.date.fromtimestamp(now)

    def end_of(day: datetime.date) -> float:
        return datetime.datetime.combine(day, datetime.time(23, 59)).timestamp()

    if phrase in ("today", "tonight"):
        return end_of(today)
    if phrase == "tomorrow":
        return end_of(today + datetime.timedelta(days=1))
    if phrase == "next week":
        return end_of(today + datetime.timedelta(days=7))
    if phrase in _WEEKDAYS:
        ahead = (_WEEKDAYS.index(phrase) - today.weekday()) % 7 or 7
        return end_of(today + datetime.timedelta(days=ahead))
    try:
        return end_of(datetime.date.fromisoformat(phrase))
    except ValueError:
        return None


def extract_commitments(text: str, now: float) -> list[dict]:
    """Detect commitments made in normal conversation: promise, deadline, confidence."""
    text = (text or "").strip()
    if not text or text.endswith("?"):
        return []
    found = []
    seen = set()
    for pattern, confidence in _COMMITMENT_PATTERNS:
        for match in pattern.finditer(text):
            task = " ".join(match.group(1).split()).strip()
            if len(task.split()) < 2:
                continue
            key = " ".join(sorted(re.findall(r"[a-z0-9]+", task.lower())))
            if not key or key in seen:
                continue
            seen.add(key)
            due_at = None
            deadline_match = _DEADLINE_RE.search(task) or _DEADLINE_RE.search(text)
            if deadline_match:
                due_at = _resolve_deadline(deadline_match.group(1), now)
            found.append(
                {
                    "content": task,
                    "canonical_key": key,
                    "due_at": due_at,
                    "confidence": confidence,
                }
            )
    return found[:4]
