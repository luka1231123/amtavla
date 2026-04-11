import re
import json
import math

from brain.config import load_brain_config
from brain.json_utils import extract_first_json_object
from brain.prompt_builder import PromptBuilder


class IntentRouter:
    def __init__(self, config: dict | None = None):
        self.config = config or load_brain_config()
        self.prompt_builder = PromptBuilder()
        routing = self.config.get("routing", {})
        self.default_pathway = routing.get("default_pathway", "planner_full")
        self.intent_model_enabled = bool(routing.get("intent_model_enabled", True))
        self.intent_model_threshold = int(routing.get("intent_model_threshold", 1))
        self.intent_model = routing.get("intent_model", "default")
        self.intent_embedding_enabled = bool(
            routing.get("intent_embedding_enabled", True)
        )
        self.low_confidence_fallback = routing.get(
            "intent_low_confidence_fallback", self.default_pathway
        )

        intents = self.config.get("intents", [])
        self._intents = []
        for intent in intents:
            compiled_regex = [
                re.compile(p, re.IGNORECASE) for p in intent.get("regex", []) if p
            ]
            self._intents.append(
                {
                    "name": intent.get("name", "unknown"),
                    "priority": int(intent.get("priority", 0)),
                    "pathway": intent.get("pathway", self.default_pathway),
                    "min_score": int(intent.get("min_score", 1)),
                    "keywords": [k.lower() for k in intent.get("keywords", [])],
                    "regex": compiled_regex,
                }
            )
        self._intents.sort(key=lambda i: i["priority"], reverse=True)
        self._intent_vectors = self._build_intent_vectors()

    def _embed(self, text: str) -> list[float] | None:
        try:
            import llama_client

            result = llama_client.embed(text)
            vec = result.get("embedding")
            if isinstance(vec, list) and vec:
                return vec
        except Exception:
            return None
        return None

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _build_intent_vectors(self) -> dict:
        if not self.intent_embedding_enabled:
            return {}
        vectors = {}
        for intent in self._intents:
            seed = [intent["name"]] + intent["keywords"][:12]
            text = " | ".join(seed)
            vec = self._embed(text)
            if vec is not None:
                vectors[intent["name"]] = vec
        return vectors

    def _embedding_route(self, text: str) -> dict | None:
        if not self.intent_embedding_enabled or not self._intent_vectors:
            return None
        qv = self._embed(text)
        if qv is None:
            return None
        best = None
        for intent in self._intents:
            vec = self._intent_vectors.get(intent["name"])
            if vec is None:
                continue
            sim = self._cosine(qv, vec)
            cand = {
                "intent": intent["name"],
                "pathway": intent["pathway"],
                "score": max(1, int(sim * 10)),
                "confidence": max(0.0, min(1.0, (sim + 1.0) / 2.0)),
                "source": "embedding",
            }
            if best is None or cand["confidence"] > best["confidence"]:
                best = cand
        if best and best["confidence"] >= 0.55:
            return best
        return None

    def _llm_route(self, text: str) -> dict | None:
        if not self.intent_model_enabled:
            return None
        intents = [
            {
                "intent": i["name"],
                "pathway": i["pathway"],
                "keywords": i["keywords"][:8],
            }
            for i in self._intents
        ]
        prompt = self.prompt_builder.build_intent_router_prompt(
            user_input=text,
            intents_json=json.dumps(intents),
        )
        try:
            import llama_client

            resp = llama_client.chat(
                [{"role": "user", "content": prompt}], model=self.intent_model
            )
            raw = resp.get("message", {}).get("content", "")
            data = extract_first_json_object(raw)
            if not data:
                return None
            intent = str(data.get("intent", "")).strip()
            pathway = str(data.get("pathway", "")).strip()
            confidence = float(data.get("confidence", 0.0))
            if not intent or not pathway:
                return None
            valid = next((i for i in self._intents if i["name"] == intent), None)
            if valid is None:
                return None
            if pathway != valid["pathway"]:
                pathway = valid["pathway"]
            return {
                "intent": intent,
                "pathway": pathway,
                "score": max(1, int(confidence * 5)),
                "confidence": max(0.0, min(1.0, confidence)),
                "source": "llm",
            }
        except Exception:
            return None

    def route(self, text: str) -> dict:
        prompt = (text or "").strip()
        lowered = prompt.lower()
        word_set = set(re.findall(r"[a-z0-9']+", lowered))

        best = None
        for intent in self._intents:
            score = 0
            for kw in intent["keywords"]:
                if not kw:
                    continue
                if " " in kw:
                    if kw in lowered:
                        score += 1
                else:
                    if kw in word_set:
                        score += 1
            for pattern in intent["regex"]:
                if pattern.search(prompt):
                    score += 2

            if score < intent["min_score"]:
                continue

            candidate = {
                "intent": intent["name"],
                "pathway": intent["pathway"],
                "score": score,
                "confidence": min(1.0, 0.35 + score * 0.2),
                "source": "rules",
            }

            if best is None:
                best = candidate
            elif candidate["score"] > best["score"]:
                best = candidate

        if best is None:
            fallback = {
                "intent": "default",
                "pathway": self.low_confidence_fallback,
                "score": 0,
                "confidence": 0.0,
                "source": "fallback",
            }
            emb = self._embedding_route(prompt)
            if emb is not None:
                best = emb
            else:
                llm = self._llm_route(prompt)
                return llm or fallback

        if best and best.get("confidence", 0.0) < 0.65:
            emb = self._embedding_route(prompt)
            if emb and emb.get("confidence", 0.0) >= best.get("confidence", 0.0):
                best = emb

        if (
            best["score"] <= self.intent_model_threshold
            or best.get("confidence", 0) < 0.75
        ):
            llm = self._llm_route(prompt)
            if llm and llm.get("confidence", 0.0) >= best.get("confidence", 0.0):
                best = llm

        if best.get("confidence", 0.0) < 0.5:
            return {
                "intent": "default",
                "pathway": self.low_confidence_fallback,
                "score": 0,
                "confidence": 0.0,
                "source": "fallback",
            }

        return best
