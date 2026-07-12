"""Structured fact extraction: turn a user utterance into typed knowledge.

The memory system's durable knowledge is stored as (entity, attribute, value)
tuples rather than whole-sentence blobs, so that a new value for an existing
(entity, attribute) *supersedes* the old one instead of piling up as a
contradiction. This module is the front door that produces those tuples.

Two strategies, one interface:

- ``extract_rule_based`` — deterministic, offline, no model. Covers the
  high-value patterns (identity, vehicle location, residence, employer,
  possessive "my X is Y"). Always available; used directly in tests and as the
  fallback when the model is disabled or fails.
- ``extract_model_based`` — a grammar-constrained LLM pass (malformed JSON is
  impossible at the sampler) that handles arbitrary phrasing. Enabled in
  production via ``extraction.model_enabled`` in brain_config.json.

``FactExtractor.extract`` runs the model when configured and falls back to
rules, so callers get a single ``list[dict]`` of claims regardless of strategy.
Each claim: ``{"entity", "attribute", "value", "confidence"}``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# Grammar-constrained output schema for the model-based extractor. llama.cpp
# converts this to a GBNF grammar, so the model physically cannot emit anything
# but a list of well-formed claims.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "attribute": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["entity", "attribute", "value"],
            },
        }
    },
    "required": ["facts"],
}

_EXTRACTION_SYSTEM = (
    "You extract durable personal facts from one user message. Return only facts "
    "the user asserts about themselves or their world that are worth remembering "
    "long-term (identity, belongings and their location/attributes, home, work, "
    "relationships, stable preferences). Model each as (entity, attribute, value):\n"
    "- entity: the thing the fact is about ('user' for the user themselves, else "
    "the noun, e.g. 'car', 'bike').\n"
    "- attribute: the property ('name', 'location', 'color', 'residence', "
    "'employer', 'preference', ...).\n"
    "- value: the concrete value, copied from the user's words, not paraphrased.\n"
    "Rules: extract nothing from questions, chit-chat, or commands. Never invent a "
    "value the user did not state. Copy values verbatim (keep '10 B', not 'level "
    "3'). If there is nothing durable, return an empty list."
)

# --- Deterministic rule-based extraction -----------------------------------

# Each rule maps a regex match to (entity, attribute, value-group). Ordered:
# the first matching rule per clause wins, so specific patterns precede the
# generic possessive catch-all.
_VEHICLE = r"(car|bike|bicycle|motorcycle|motorbike|scooter|van|truck)"

_RULES: list[tuple[re.Pattern[str], Callable[[re.Match[str]], tuple[str, str, str]]]] = [
    # Identity
    (
        re.compile(r"\bmy name(?:'s| is| =)\s+(.+)", re.IGNORECASE),
        lambda m: ("user", "name", m.group(1)),
    ),
    (
        re.compile(r"\b(?:i am|i'm)\s+called\s+(.+)", re.IGNORECASE),
        lambda m: ("user", "name", m.group(1)),
    ),
    (
        re.compile(r"\bcall me\s+(.+)", re.IGNORECASE),
        lambda m: ("user", "name", m.group(1)),
    ),
    # Vehicle location — the class that failed live ("parked in 10 B").
    (
        re.compile(
            rf"\b(?:my |the |a )?{_VEHICLE}\b.*?\bparked\s+(?:in|at|on)\s+(.+)",
            re.IGNORECASE,
        ),
        lambda m: (m.group(1).lower(), "location", m.group(2)),
    ),
    (
        re.compile(
            rf"\bi (?:have |)parked\s+(?:my |the |a )?{_VEHICLE}\s+(?:in|at|on)\s+(.+)",
            re.IGNORECASE,
        ),
        lambda m: (m.group(1).lower(), "location", m.group(2)),
    ),
    (
        re.compile(
            rf"\bi have\s+(?:my |a |an )?{_VEHICLE}\s+(?:in|at|on)\s+(.+)",
            re.IGNORECASE,
        ),
        lambda m: (m.group(1).lower(), "location", m.group(2)),
    ),
    # Vehicle location without "parked": "my car is (now/currently) in 10 C".
    # Must precede the generic possessive rule so it keys as
    # (car, location) — the same key as the "parked" forms — so a new location
    # supersedes the old instead of forking into (user, car).
    (
        re.compile(
            rf"\b(?:my |the )?{_VEHICLE}\s+(?:is|'s|are)\s+(?:now\s+|currently\s+|still\s+)?"
            r"(?:in|at|on)\s+(.+)",
            re.IGNORECASE,
        ),
        lambda m: (m.group(1).lower(), "location", m.group(2)),
    ),
    # Residence / employer
    (
        re.compile(r"\bi live (?:at|in)\s+(.+)", re.IGNORECASE),
        lambda m: ("user", "residence", m.group(1)),
    ),
    (
        re.compile(r"\bi work (?:at|for)\s+(.+)", re.IGNORECASE),
        lambda m: ("user", "employer", m.group(1)),
    ),
    # Generic possessive catch-all: "my <noun> is/are <value>".
    (
        re.compile(
            r"\bmy ([a-z][a-z ]{0,30}?) (?:is|are|was|will be)\s+(.+)",
            re.IGNORECASE,
        ),
        lambda m: (_possessive_entity(m.group(1)), _possessive_attribute(m.group(1)), m.group(2)),
    ),
]

# Nouns that name a belonging: "my bike is blue" -> entity=bike, attribute=color
# is unknowable from the surface, so the possessive rule keeps entity='user' and
# attribute=<noun> ("The user's bike is blue"). Vehicles are handled above for
# location; here the possessive form still captures colour/state cleanly.


def _possessive_entity(noun: str) -> str:
    return "user"


def _possessive_attribute(noun: str) -> str:
    return " ".join(noun.lower().split())


def _clean_value(value: str) -> str:
    value = " ".join((value or "").split())
    # Trim a trailing conjunction clause the regex greedily swallowed only when
    # it clearly starts a new statement ("... and parked in B2").
    value = re.sub(r"\s+and\s+(?:it |they |).*$", "", value, flags=re.IGNORECASE) \
        if re.search(r"\band\s+(?:parked|is|are|located)\b", value, re.IGNORECASE) else value
    return value.strip().strip(".,;:!?").strip()


def extract_rule_based(text: str) -> list[dict[str, Any]]:
    """Deterministic (entity, attribute, value) extraction. No model, no network."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    # Split on sentence/clause boundaries so "my name is Mira. my bike is blue"
    # yields two facts.
    for clause in re.split(r"[\n.!?;]", text or ""):
        clause = clause.strip()
        if not clause:
            continue
        for pattern, builder in _RULES:
            match = pattern.search(clause)
            if not match:
                continue
            entity, attribute, raw_value = builder(match)
            value = _clean_value(raw_value)
            entity = (entity or "").strip().lower()
            attribute = " ".join((attribute or "").split()).lower()
            if not (entity and attribute and value):
                continue
            key = (f"{entity}:{attribute}", value.lower())
            dedup = (entity, attribute)
            if dedup in seen:
                continue
            seen.add(dedup)
            out.append(
                {
                    "entity": entity,
                    "attribute": attribute,
                    "value": value,
                    "confidence": 0.8,
                }
            )
            break  # first matching rule per clause
    return out


def extract_model_based(
    text: str, chat_fn: Callable[..., dict], *, profile: str = "default"
) -> list[dict[str, Any]]:
    """Grammar-constrained LLM extraction. Raises on transport error so the
    caller can fall back to rules; returns [] when the model finds nothing."""
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {"role": "user", "content": (text or "").strip()},
    ]
    response = chat_fn(messages, schema=EXTRACTION_SCHEMA, profile=profile)
    content = (response or {}).get("message", {}).get("content", "") or ""
    if content.startswith("Error:"):
        raise RuntimeError(content)
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    facts = data.get("facts", []) if isinstance(data, dict) else []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        entity = " ".join(str(fact.get("entity", "")).split()).lower()
        attribute = " ".join(str(fact.get("attribute", "")).split()).lower()
        value = " ".join(str(fact.get("value", "")).split()).strip().strip(".,;:!?")
        if not (entity and attribute and value):
            continue
        dedup = (entity, attribute)
        if dedup in seen:
            continue
        seen.add(dedup)
        try:
            confidence = float(fact.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        out.append(
            {
                "entity": entity,
                "attribute": attribute,
                "value": value,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    return out


class FactExtractor:
    """Strategy wrapper: model-first (when enabled) with a rules fallback.

    ``model_enabled`` mirrors the ``intent_model_enabled`` config switch — off in
    tests (deterministic, offline) and on in production for arbitrary phrasing.
    """

    def __init__(
        self,
        *,
        model_enabled: bool = False,
        chat_fn: Callable[..., dict] | None = None,
        profile: str = "default",
    ) -> None:
        self.model_enabled = bool(model_enabled)
        self._chat_fn = chat_fn
        self._profile = profile

    def extract(self, text: str) -> list[dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return []
        if self.model_enabled and self._chat_fn is not None:
            try:
                facts = extract_model_based(text, self._chat_fn, profile=self._profile)
                if facts:
                    return facts
            except Exception:
                pass  # fall through to deterministic rules
        return extract_rule_based(text)
