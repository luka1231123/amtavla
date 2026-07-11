from __future__ import annotations

import datetime
import re
import time
from typing import Any

from brain.memory.catalog import MemoryCatalog, _canonical_name
from brain.memory.commitments import _resolve_deadline

# Capitalized words that look like names but never are.
_NAME_STOPWORDS = {
    "i", "the", "a", "an", "my", "me", "we", "you", "it", "this", "that",
    "these", "those", "project", "document", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "today", "tomorrow", "yesterday", "remember",
    "don", "please", "tell", "note",
}

_TIME_PERIOD_WORDS = {
    "morning", "afternoon", "evening", "night", "week", "weekend", "month",
    "today", "tomorrow", "yesterday", "tonight",
}

_WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
)


def _date_tag(now: float, day_offset: int = 0) -> str:
    day = datetime.date.fromtimestamp(now) + datetime.timedelta(days=day_offset)
    return day.isoformat()


def _date_tag_from_ts(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.date.fromtimestamp(ts).isoformat()


class TagEngine:
    """Fast heuristic auto-tagger: regex + known entities + feedback history.

    No model or network dependencies, so suggestions land within milliseconds
    and the same engine works in the model-free dashboard server.
    """

    def __init__(self, catalog: MemoryCatalog, config: dict[str, Any] | None = None):
        self.catalog = catalog
        config = config or {}
        self.min_confidence = float(config.get("min_confidence", 0.3))
        self.max_tags_per_item = int(config.get("max_tags_per_item", 6))
        self.reject_penalty = float(config.get("feedback_reject_penalty", 0.15))
        self.accept_boost = float(config.get("feedback_accept_boost", 0.1))

    def suggest_tags(
        self,
        content: str,
        *,
        session_context: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidate tags [{tag_type, name, confidence}], best first."""
        content = (content or "").strip()
        if not content:
            return []
        now = now if now is not None else time.time()
        session_context = session_context or {}

        candidates: dict[tuple[str, str], dict[str, Any]] = {}

        def add(tag_type: str, name: str, confidence: float):
            name = " ".join(name.split()).strip(" .,!?")
            canonical = _canonical_name(name)
            if not canonical or canonical in _NAME_STOPWORDS:
                return
            key = (tag_type, canonical)
            existing = candidates.get(key)
            if existing is None or confidence > existing["confidence"]:
                candidates[key] = {
                    "tag_type": tag_type,
                    "name": name,
                    "confidence": confidence,
                }

        self._extract_projects(content, add)
        self._extract_people(content, add)
        self._extract_locations(content, add)
        self._extract_time(content, now, add)
        self._match_known_entities(content, add)
        self._apply_session_context(content, session_context, candidates, add)
        self._apply_feedback(candidates)

        ranked = sorted(
            (c for c in candidates.values() if c["confidence"] >= self.min_confidence),
            key=lambda c: c["confidence"],
            reverse=True,
        )
        return ranked[: self.max_tags_per_item]

    def tag_item(
        self,
        item: dict[str, Any],
        *,
        session_context: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Suggest and persist tag assignments for a catalog item dict."""
        suggestions = self.suggest_tags(
            item.get("content", ""), session_context=session_context, now=now
        )
        assigned = []
        for suggestion in suggestions:
            assigned.append(
                self.catalog.assign_tag(
                    item["id"],
                    suggestion["tag_type"],
                    suggestion["name"],
                    status="suggested",
                    confidence=suggestion["confidence"],
                    source="auto",
                )
            )
        return assigned

    @staticmethod
    def _extract_projects(content: str, add):
        for match in re.finditer(r"\bproject\s+([A-Z][A-Za-z0-9_-]*)", content):
            add("project", match.group(1), 0.75)
        for match in re.finditer(r"\b([A-Z][A-Za-z0-9_-]+)\s+project\b", content):
            add("project", match.group(1), 0.7)

    @staticmethod
    def _extract_people(content: str, add):
        for match in re.finditer(r"\b([A-Z][a-z]+)'s\b", content):
            add("person", match.group(1), 0.6)
        # Verb may start the sentence, so allow a leading capital — but the
        # name capture stays strictly capitalized to avoid common nouns.
        for match in re.finditer(
            r"\b(?:[Ww]ith|[Mm]et|[Mm]eet(?:ing)?|[Cc]all(?:ed)?|[Tt]old|[Tt]ell"
            r"|[Aa]sk(?:ed)?|[Ff]rom|[Ee]mail|[Ll]unch with|[Dd]inner with)"
            r"\s+([A-Z][a-z]+)\b",
            content,
        ):
            add("person", match.group(1), 0.65)
        for match in re.finditer(r"\bmy name is\s+([A-Z][A-Za-z'-]+)", content):
            add("person", match.group(1), 0.8)

    @staticmethod
    def _extract_locations(content: str, add):
        for match in re.finditer(
            r"\b(?:at|in|near)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b", content
        ):
            add("location", match.group(1), 0.6)
        stop_words = _TIME_PERIOD_WORDS | {"to", "for", "about", "and", "with", "on"}
        for match in re.finditer(
            r"\b(?:at|near|parked at|left at)\s+the\s+([a-z][a-z0-9 ]{2,40})", content
        ):
            phrase = re.split(r"[.,;!?]", match.group(1))[0]
            words = []
            for word in phrase.split():
                if word in stop_words or len(words) == 4:
                    break
                words.append(word)
            if words:
                add("location", " ".join(words), 0.5)

    @staticmethod
    def _extract_time(content: str, now: float, add):
        lowered = content.lower()
        if re.search(r"\btoday\b|\btonight\b", lowered):
            add("time", _date_tag(now), 0.7)
        if re.search(r"\btomorrow\b", lowered):
            add("time", _date_tag(now, 1), 0.7)
        if re.search(r"\byesterday\b", lowered):
            add("time", _date_tag(now, -1), 0.7)
        for match in re.finditer(r"\b(20\d{2}-\d{2}-\d{2})\b", lowered):
            add("time", match.group(1), 0.85)
        for match in re.finditer(
            r"\b(?:this|in the)\s+(morning|afternoon|evening|weekend)\b", lowered
        ):
            add("time", match.group(1), 0.5)
        if re.search(r"\bnext week\b", lowered):
            resolved = _resolve_deadline("next week", now)
            add("time", _date_tag_from_ts(resolved) or "next week", 0.55)
        for weekday in _WEEKDAYS:
            if re.search(rf"\b(?:on|next|this)\s+{weekday}\b", lowered):
                resolved = _resolve_deadline(weekday, now)
                add("time", _date_tag_from_ts(resolved) or weekday, 0.55)

    def _match_known_entities(self, content: str, add):
        """Confirmed knowledge improves tagging: known entities matched by name."""
        canonical_content = f" {_canonical_name(content)} "
        type_map = {"person": "person", "place": "location", "project": "project"}
        for entity in self.catalog.list_entities(limit=500):
            tag_type = type_map.get(entity["entity_type"])
            if tag_type is None:
                continue
            if entity["review_state"] in {"rejected", "deleted", "archived"}:
                continue
            if f" {entity['canonical_name']} " in canonical_content:
                confidence = 0.75 if entity["review_state"] == "confirmed" else 0.6
                add(tag_type, entity["name"], confidence)

    @staticmethod
    def _apply_session_context(content, session_context, candidates, add):
        active_project = (session_context.get("active_project") or "").strip()
        if active_project:
            has_project = any(key[0] == "project" for key in candidates)
            if not has_project:
                add("project", active_project, 0.4)
        canonical_content = f" {_canonical_name(content)} "
        for person in session_context.get("recent_people", []):
            key = ("person", _canonical_name(person))
            if key in candidates:
                candidates[key]["confidence"] = min(
                    0.95, candidates[key]["confidence"] + 0.1
                )
        for location in session_context.get("recent_locations", []):
            if f" {_canonical_name(location)} " in canonical_content:
                add("location", location, 0.55)

    def _apply_feedback(self, candidates: dict):
        """User corrections steer later suggestions: rejects suppress, accepts boost."""
        feedback = self.catalog.tag_feedback_summary()
        if not feedback:
            return
        for (tag_type, canonical), candidate in list(candidates.items()):
            stats = feedback.get(f"{tag_type}:{canonical}", {})
            rejected = stats.get("rejected", 0) + stats.get("corrected", 0)
            accepted = stats.get("accepted", 0)
            if rejected >= 2 and accepted == 0:
                del candidates[(tag_type, canonical)]
                continue
            adjusted = (
                candidate["confidence"]
                + accepted * self.accept_boost
                - rejected * self.reject_penalty
            )
            candidate["confidence"] = max(0.0, min(0.95, adjusted))
