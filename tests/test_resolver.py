from brain.resolver import ResolvedInput, TurnResolver, looks_dependent


class _FakeModel:
    """Records the resolution call and returns a scripted rewrite."""

    def __init__(self, query="who said to err is human", is_followup=True):
        self.query = query
        self.is_followup = is_followup
        self.calls = []

    def chat(self, messages, model=None, schema=None):
        self.calls.append({"messages": messages, "model": model, "schema": schema})
        return {
            "message": {
                "content": (
                    '{"standalone_query": "%s", "is_followup": %s}'
                    % (self.query, "true" if self.is_followup else "false")
                )
            }
        }


_CONVO = [
    {"user_input": "to err is human", "response": "A common idiom about mistakes."},
]


def test_no_conversation_passes_through_untouched():
    resolver = TurnResolver(client=_FakeModel())
    out = resolver.resolve("look it up", conversation=[])
    assert out.text == "look it up"
    assert out.is_followup is False
    assert out.source == "passthrough"


def test_self_contained_question_is_not_rewritten():
    model = _FakeModel()
    resolver = TurnResolver(client=model)
    out = resolver.resolve("Who wrote Pride and Prejudice?", conversation=_CONVO)
    assert out.text == "Who wrote Pride and Prejudice?"
    assert out.source == "passthrough"
    # The model must never be consulted for a query that already stands alone.
    assert model.calls == []


def test_explicit_command_is_never_rewritten_even_with_context():
    # Rewriting "remember this ..." could turn a memory-write into a web search.
    model = _FakeModel()
    resolver = TurnResolver(client=model)
    out = resolver.resolve("remember this: my bike is blue", conversation=_CONVO)
    assert out.text == "remember this: my bike is blue"
    assert out.source == "passthrough"
    assert model.calls == []


def test_dependent_followup_is_rewritten_by_model():
    model = _FakeModel(query="who said to err is human")
    resolver = TurnResolver(client=model)
    out = resolver.resolve("look it up in the web", conversation=_CONVO)
    assert out.text == "who said to err is human"
    assert out.is_followup is True
    assert out.source == "model"
    # The conversation was handed to the model so it could resolve the referent.
    assert "to err is human" in model.calls[0]["messages"][0]["content"]


def test_model_failure_falls_back_to_original_words():
    class _Broken:
        def chat(self, messages, model=None, schema=None):
            return {"message": {"content": "Error: model down"}}

    resolver = TurnResolver(client=_Broken())
    out = resolver.resolve("what is the continuation of that phrase", _CONVO)
    # A follow-up we could not rewrite still routes on its literal words — never
    # worse than before — but is flagged so continuity handling stays on.
    assert out.text == "what is the continuation of that phrase"
    assert out.is_followup is True
    assert out.source == "rules"


def test_disabled_resolver_flags_followup_without_calling_model():
    model = _FakeModel()
    resolver = TurnResolver(client=model, enabled=False)
    out = resolver.resolve("look it up", conversation=_CONVO)
    assert out.text == "look it up"
    assert out.source == "rules"
    assert out.is_followup is True
    assert model.calls == []


def test_degenerate_rewrite_is_rejected():
    resolver = TurnResolver(client=_FakeModel(query="   "))
    out = resolver.resolve("look it up", conversation=_CONVO)
    assert out.text == "look it up"  # blank rewrite rejected
    assert out.source == "rules"


def test_looks_dependent_classification():
    assert looks_dependent("look it up") is True
    assert looks_dependent("no bro I want them now") is True
    assert looks_dependent("what is the continuation of that phrase") is True
    assert looks_dependent("yeah look it up") is True
    assert looks_dependent("do it") is True
    assert looks_dependent("tell me more") is True
    assert looks_dependent("continue") is True
    # Self-contained / explicit-command inputs are not dependent.
    assert looks_dependent("What is 17 * 19?") is False
    assert looks_dependent("Who wrote Pride and Prejudice?") is False
    assert looks_dependent("remember this: my name is Mira") is False
    assert looks_dependent("") is False


def test_self_contained_questions_with_common_leads_are_not_dependent():
    # Regression: "why/so/how about" open plenty of standalone questions. They
    # must NOT be flagged, or the small model gets handed a fine query to mangle.
    assert looks_dependent("Why do birds migrate?") is False
    assert looks_dependent("So what is the capital of France?") is False
    assert looks_dependent("How about Python versus Java?") is False
    assert looks_dependent("continue the Python loop over the list") is False


def test_hallucinated_rewrite_unrelated_to_context_is_rejected():
    # A drifted rewrite that shares no content word with the utterance or the
    # conversation is discarded in favour of the original words.
    resolver = TurnResolver(client=_FakeModel(query="the weather in Tokyo tomorrow"))
    out = resolver.resolve("look it up", conversation=_CONVO)
    assert out.text == "look it up"
    assert out.source == "rules"


def test_grounded_rewrite_sharing_context_tokens_is_accepted():
    resolver = TurnResolver(client=_FakeModel(query="origin of to err is human"))
    out = resolver.resolve("look it up", conversation=_CONVO)
    assert out.text == "origin of to err is human"
    assert out.source == "model"


def test_from_config_reads_flag():
    resolver = TurnResolver.from_config(
        {"routing": {"resolve_followups_enabled": False, "intent_model": "fast"}}
    )
    assert resolver.enabled is False
    assert resolver.model == "fast"


def test_resolved_input_to_dict_marks_rewrites():
    assert ResolvedInput("look it up", "search X", True, "model").to_dict()[
        "rewritten"
    ] is True
    assert ResolvedInput("hi", "hi").to_dict()["rewritten"] is False
