from copy import deepcopy

from brain.config import load_brain_config
from brain.intent_router import IntentRouter


def _config():
    return {
        "routing": {
            "default_pathway": "planner_full",
            "intent_model_enabled": False,
            "intent_embedding_enabled": False,
            "intent_low_confidence_fallback": "planner_full",
        },
        "intents": [
            {
                "name": "remember",
                "priority": 10,
                "pathway": "remember_reply",
                "min_score": 1,
                "keywords": ["remember this"],
                "regex": [],
            }
        ],
    }


def test_rule_route_is_deterministic_without_model_or_embeddings():
    route = IntentRouter(_config()).route("Remember this launch detail")

    assert route["intent"] == "remember"
    assert route["pathway"] == "remember_reply"
    assert route["source"] == "rules"


def test_unknown_route_uses_configured_fallback():
    route = IntentRouter(_config()).route("something unrelated")

    assert route == {
        "intent": "default",
        "pathway": "planner_full",
        "score": 0,
        "confidence": 0.0,
        "source": "fallback",
    }


def test_what_do_you_know_routes_to_memory_recall():
    config = deepcopy(load_brain_config())
    config["routing"]["intent_model_enabled"] = False
    config["routing"]["intent_embedding_enabled"] = False

    route = IntentRouter(config).route("What do you know about Project Phoenix?")

    assert route["intent"] == "memory_recall"
    assert route["pathway"] == "memory_recall_reply"


def test_external_knowledge_question_prefers_search_without_model():
    config = deepcopy(load_brain_config())
    config["routing"]["intent_model_enabled"] = False
    config["routing"]["intent_embedding_enabled"] = False

    route = IntentRouter(config).route("How does a heat pump work?")

    assert route["intent"] == "web_factual"
    assert route["pathway"] == "search_then_reply"
    assert route["source"] == "knowledge_question"


def test_ambiguous_assistant_question_does_not_force_web_search():
    config = deepcopy(load_brain_config())
    config["routing"]["intent_model_enabled"] = False
    config["routing"]["intent_embedding_enabled"] = False

    route = IntentRouter(config).route("What should you possess?")

    assert route["pathway"] == "planner_full"
