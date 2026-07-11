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


_TIME_OF_DAY = {
    "morning": datetime.time(9, 0),
    "noon": datetime.time(12, 0),
    "afternoon": datetime.time(15, 0),
    "evening": datetime.time(19, 0),
    "night": datetime.time(21, 0),
}

_REMINDER_TASK_RE = re.compile(
    r"\b(?:remind me|set a reminder|don'?t let me forget)\s*(?:to|about|of)?\s*(.*)",
    re.IGNORECASE,
)
_QUESTION_REMINDER_RE = re.compile(
    r"\b(?:would|could|can|will)\s+you\s+remind\s+me\s+(?:to|about|of)?\s*(.*)",
    re.IGNORECASE,
)
_WHEN_RE = re.compile(
    r"\b(today|tonight|tomorrow|next week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"20\d{2}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_TOD_RE = re.compile(r"\b(morning|noon|afternoon|evening|night)\b", re.IGNORECASE)
_CLOCK_RE = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_IN_RE = re.compile(r"\bin\s+(\d+)\s+(minute|min|hour|day|week)s?\b", re.IGNORECASE)


def parse_reminder(text: str, now: float) -> dict | None:
    """Parse an explicit reminder request into task text and a resolved due time.

    Handles day words plus a time of day ("tomorrow morning"), clock times
    ("at 14:00", "at 2pm"), and relative offsets ("in 20 minutes"). Returns
    None when the text doesn't look like a reminder request at all.
    """
    text = (text or "").strip()
    match = _REMINDER_TASK_RE.search(text)
    if not match:
        match = _QUESTION_REMINDER_RE.search(text)
    if not match:
        return None
    task = " ".join(match.group(1).split()).strip(" .,!?")

    relative = _IN_RE.search(text)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).lower()
        seconds = {"minute": 60, "min": 60, "hour": 3600, "day": 86400, "week": 7 * 86400}[unit]
        return {"content": task or text, "due_at": now + amount * seconds}

    day_match = _WHEN_RE.search(text)
    base_day = datetime.date.fromtimestamp(now)
    due_day = base_day
    if day_match:
        resolved = _resolve_deadline(day_match.group(1), now)
        if resolved is not None:
            due_day = datetime.date.fromtimestamp(resolved)

    tod_match = _TOD_RE.search(text)
    clock_match = None
    for candidate in _CLOCK_RE.finditer(text):
        hour = int(candidate.group(1))
        # Bare small numbers without minutes or am/pm are usually not times.
        if candidate.group(2) or candidate.group(3) or text[: candidate.start()].rstrip().lower().endswith("at"):
            if 0 <= hour <= 23:
                clock_match = candidate
                break

    if clock_match:
        hour = int(clock_match.group(1))
        minute = int(clock_match.group(2) or 0)
        meridiem = (clock_match.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        due_time = datetime.time(min(hour, 23), min(minute, 59))
    elif tod_match:
        due_time = _TIME_OF_DAY[tod_match.group(1).lower()]
    elif day_match:
        due_time = datetime.time(9, 0)
    else:
        return {"content": task or text, "due_at": None}

    due_at = datetime.datetime.combine(due_day, due_time).timestamp()
    if due_at <= now and not day_match:
        due_at += 86400
    return {"content": task or text, "due_at": due_at}


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
