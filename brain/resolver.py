"""Turn resolution: rewrite a context-dependent utterance into a standalone one.

The local model is small. The way to stop it fumbling is not to make it smarter
mid-answer but to hand every stage an *unambiguous* input — to plan ahead. A
follow-up like "look it up", "no bro I want them now", or "what is the
continuation of that phrase" means nothing on its own: the router mislabels it,
recall retrieves nothing, and SEARCH queries the literal string. This module
turns such fragments into self-contained requests ("search the web for who said
'to err is human'") *before* any of those decisions are made.

Design constraints (why this is safe with a dumb model):

- **Deterministic gate first.** Resolution only fires for utterances that are
  clearly context-dependent (deictic pronouns, follow-up leads, terse
  fragments) AND only when there is a live conversation to resolve against. A
  self-contained question passes through untouched — the model never gets a
  chance to corrupt a query that was already fine.
- **Explicit commands are never rewritten.** "remember ...", "remind me ...",
  "delete ..." carry their own intent and route deterministically; rewriting
  them could turn a memory-write into a web search. They are skipped.
- **The rewrite only ever feeds routing/recall/search.** Permission gates
  (MEMORY_WRITE, REMINDER) and memory commit always use the original words, and
  the generator still sees the original utterance plus the real dialogue. So a
  bad rewrite can at worst mis-route a fragment that was already ambiguous —
  never downgrade a turn that would otherwise have worked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Pronouns/deixis that only resolve against prior turns. Deliberately limited to
# genuine anaphora: broader "the origin / the rest" style phrases were dropped
# because they also open self-contained questions ("what is the origin of the
# universe") and a false positive here hands a fine query to the small model to
# mangle.
_DEICTIC_RE = re.compile(
    r"\b(it|its|it's|that|those|these|this|them|they|their|"
    r"the same|the one|the ones)\b",
    re.IGNORECASE,
)
# Imperatives that mean nothing without a referent — they carry a deictic object
# or an explicitly relative word. These are unambiguously dependent.
_DEPENDENT_IMPERATIVE_RE = re.compile(
    r"^\s*(please\s+)?("
    r"(look|search|check|google)\s+(it|that|them|those|this)\b|"
    r"(look|search|find)\s+(it|that|them)\s+up\b|"
    r"find\s+(it|them|that)\s+out\b|"
    r"do\s+(it|that)\b|"
    r"tell\s+me\s+more\b|"
    r"go\s+on\b|carry\s+on\b|keep\s+going\b|try\s+again\b)",
    re.IGNORECASE,
)
# Bare continuation fragments — dependent only when they ARE the whole utterance
# (no self-contained content of their own). Anchored end-to-end so "continue"
# flags but "continue the Python loop" does not.
_CONTINUATION_FRAGMENT_RE = re.compile(
    r"^\s*(yeah|yep|yes|ok|okay|sure|go ahead|go on|continue|carry on|more|"
    r"again|why not|and then|the rest|the same)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

# Minimal stopword set for the rewrite-sanity overlap check (below).
_STOPWORDS = frozenset(
    "a an the is are was were be been being do does did to of in on at for and or "
    "but so it its that this these those them they their there here what who whom "
    "when where why how which up out me my you your i we he she his her look "
    "search find tell more again please can could would will now bro man".split()
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def _content_tokens(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS
    }
# Explicit-intent commands that must route on their own literal words. Rewriting
# these is never allowed — it could change the intent the router keys on.
_EXPLICIT_COMMAND_RE = re.compile(
    r"^\s*(please\s+)?("
    r"remember\b|memori[sz]e\b|note\b|save this\b|don'?t forget\b|keep this\b|"
    r"remind me\b|set an? reminder\b|don'?t let me forget\b|"
    r"delete\b|forget\b|clear\b|wipe\b|/)"
    ,
    re.IGNORECASE,
)

# Grammar-constrained shape: the small model physically cannot emit anything but
# this, so parsing never fails (llama.cpp turns the schema into a GBNF grammar).
RESOLVER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone_query": {"type": "string"},
        "is_followup": {"type": "boolean"},
    },
    "required": ["standalone_query", "is_followup"],
    "additionalProperties": False,
}

_MAX_DEPENDENT_WORDS = 12


@dataclass
class ResolvedInput:
    original: str
    text: str
    is_followup: bool = False
    source: str = "passthrough"  # passthrough | rules | model

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "text": self.text,
            "is_followup": self.is_followup,
            "source": self.source,
            "rewritten": self.text.strip() != self.original.strip(),
        }


def looks_dependent(text: str) -> bool:
    """True when an utterance cannot be understood without the prior turns.

    Biased toward false negatives: a missed follow-up merely passes through
    (routing no worse than before), whereas a false positive hands an already-fine
    query to the small model to rewrite. So only genuinely dependent shapes count.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _EXPLICIT_COMMAND_RE.search(stripped):
        return False
    if _DEPENDENT_IMPERATIVE_RE.search(stripped):
        return True
    if _CONTINUATION_FRAGMENT_RE.match(stripped):
        return True
    # A deictic reference in a short utterance has no antecedent of its own.
    if len(stripped.split()) <= _MAX_DEPENDENT_WORDS and _DEICTIC_RE.search(stripped):
        return True
    return False


class TurnResolver:
    """Resolve context-dependent utterances ahead of routing.

    The model call is optional and defensive: if it is disabled, fails, or
    returns something degenerate, the original text is used unchanged.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        enabled: bool = True,
        model: str = "default",
        max_context_turns: int = 6,
    ) -> None:
        self.client = client
        self.enabled = enabled
        self.model = model
        self.max_context_turns = max_context_turns

    @classmethod
    def from_config(
        cls, config: dict[str, Any] | None, client: Any | None = None
    ) -> "TurnResolver":
        routing = (config or {}).get("routing", {})
        return cls(
            client=client,
            enabled=bool(routing.get("resolve_followups_enabled", True)),
            model=str(routing.get("intent_model", "default")),
        )

    def _model(self):
        if self.client is not None:
            return self.client
        import llama_client

        return llama_client

    def resolve(
        self, user_input: str, conversation: list[dict] | None
    ) -> ResolvedInput:
        original = (user_input or "").strip()
        conversation = conversation or []
        # No live conversation → we are starting, not continuing: nothing to
        # resolve against, so the utterance is taken at face value.
        if not original or not conversation:
            return ResolvedInput(original, original, is_followup=False)
        if not looks_dependent(original):
            return ResolvedInput(original, original, is_followup=False)
        if not self.enabled:
            # Flagged as a follow-up so downstream/observability knows context is
            # in play, but left verbatim without a model to rewrite it.
            return ResolvedInput(original, original, is_followup=True, source="rules")

        transcript = _render_transcript(conversation[-self.max_context_turns :])
        rewritten = self._resolve_with_model(original, transcript)
        if rewritten and _is_reasonable_rewrite(original, rewritten, transcript):
            return ResolvedInput(
                original, rewritten, is_followup=True, source="model"
            )
        return ResolvedInput(original, original, is_followup=True, source="rules")

    def _resolve_with_model(self, original: str, transcript: str) -> str | None:
        from brain.json_utils import extract_first_json_object
        system = (
            "You rewrite a user's latest message into a single self-contained "
            "request, resolving pronouns and vague references ('it', 'that', "
            "'them', 'the continuation') using the recent conversation. Rules:\n"
            "- Preserve the user's actual intent and their action verb (if they "
            "said to search, keep it a search request).\n"
            "- Do NOT answer, explain, or add facts. Only rewrite.\n"
            "- If the message already stands on its own, return it unchanged and "
            "set is_followup to false.\n"
            "- Keep it short and literal.\n\n"
            f"Recent conversation:\n{transcript}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Latest message: {original}"},
        ]
        try:
            response = self._model().chat(
                messages, model=self.model, schema=RESOLVER_SCHEMA
            )
            content = (response.get("message", {}) or {}).get("content", "")
            if content.startswith("Error:"):
                return None
            data = extract_first_json_object(content)
            if not data:
                return None
            query = str(data.get("standalone_query", "")).strip()
            return query or None
        except Exception:
            return None


def _render_transcript(conversation: list[dict]) -> str:
    lines = []
    for turn in conversation:
        user_said = (turn.get("user_input") or "").strip()
        assistant_said = (turn.get("response") or "").strip()
        if user_said:
            lines.append(f"User: {user_said}")
        if assistant_said:
            lines.append(f"Assistant: {assistant_said}")
    return "\n".join(lines) if lines else "(none)"


def _is_reasonable_rewrite(original: str, rewritten: str, transcript: str = "") -> bool:
    """Reject degenerate or hallucinated model output so a bad rewrite can't
    make a turn worse than passthrough.

    A rewrite must be non-empty, not absurdly long, and grounded: it has to share
    at least one content word with the original utterance or the conversation it
    was resolved against. A rewrite that invents wholly unrelated content (the
    classic small-model failure) is discarded in favour of the original words.
    """
    rewritten = rewritten.strip()
    if not rewritten:
        return False
    if len(rewritten) > max(160, len(original) * 8):
        return False
    rewrite_tokens = _content_tokens(rewritten)
    if not rewrite_tokens:
        return False
    grounding = _content_tokens(original) | _content_tokens(transcript)
    if grounding and not (rewrite_tokens & grounding):
        return False
    return True
