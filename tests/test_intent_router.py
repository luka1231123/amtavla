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


def test_brain_dump_is_never_reached_by_fuzzy_model_routing(monkeypatch):
    # Dumping the whole memory store must be an explicit request. Here the small
    # model tries to classify a vague memory-ish utterance as a brain dump; the
    # router must refuse it and fall back rather than dump unprompted.
    import llama_client

    monkeypatch.setattr(
        llama_client,
        "chat",
        lambda messages, model="default": {
            "message": {
                "content": (
                    '{"intent": "brain_dump", "pathway": "brain_dump_reply", '
                    '"confidence": 0.95}'
                )
            }
        },
    )
    config = {
        "routing": {
            "default_pathway": "planner_full",
            "intent_model_enabled": True,
            "intent_model_threshold": 1,
            "intent_embedding_enabled": False,
            "intent_low_confidence_fallback": "planner_full",
        },
        "intents": [
            {
                "name": "brain_dump",
                "priority": 120,
                "pathway": "brain_dump_reply",
                "min_score": 1,
                "keywords": ["brain dump"],
                "regex": ["^brain dump$"],
            }
        ],
    }

    route = IntentRouter(config).route("what do you have in your memory")

    assert route["intent"] != "brain_dump"
    assert route["pathway"] != "brain_dump_reply"

    # ...but the exact phrase still triggers it deterministically via rules.
    explicit = IntentRouter(config).route("brain dump")
    assert explicit["intent"] == "brain_dump"
    assert explicit["source"] == "rules"
