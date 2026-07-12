import json
import os

import llama_client
from brain import schemas
from brain.contracts import ActionType


# ----------------------------------------------------------------- model resolve
def test_resolve_model_path_prefers_configured_filename(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "old-model.gguf").write_text("x")
    wanted = models / "new-model.gguf"
    wanted.write_text("x")

    monkeypatch.setattr(llama_client, "_MODELS_DIR", str(models))
    path = llama_client.resolve_model_path({"model_filename": "new-model.gguf"})
    assert path == str(wanted)


def test_resolve_model_path_missing_configured_file_is_none(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "present.gguf").write_text("x")
    monkeypatch.setattr(llama_client, "_MODELS_DIR", str(models))

    assert llama_client.resolve_model_path({"model_filename": "absent.gguf"}) is None


def test_resolve_model_path_fallback_is_deterministic_and_skips_vocab(
    tmp_path, monkeypatch
):
    models = tmp_path / "models"
    models.mkdir()
    (models / "ggml-vocab-llama.gguf").write_text("x")
    (models / "b-model.gguf").write_text("x")
    (models / "a-model.gguf").write_text("x")
    monkeypatch.setattr(llama_client, "_MODELS_DIR", str(models))

    # No configured filename that exists -> deterministic sorted, vocab skipped.
    path = llama_client.resolve_model_path({"model_filename": "missing.gguf"})
    assert path is None  # configured-but-missing returns None, never guesses

    path = llama_client.resolve_model_path({})
    assert path == str(models / "a-model.gguf")


# ----------------------------------------------------------------- server command
def test_build_server_cmd_reflects_config():
    cmd = llama_client._build_server_cmd(
        "/models/m.gguf",
        {
            "gpu_layers": 50,
            "context_size": 16384,
            "use_jinja": True,
            "reasoning_mode": "off",
        },
    )
    assert "--jinja" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"
    assert cmd[cmd.index("-c") + 1] == "16384"
    assert cmd[cmd.index("-ngl") + 1] == "50"
    assert cmd[cmd.index("-m") + 1] == "/models/m.gguf"


def test_build_server_cmd_omits_jinja_by_default():
    cmd = llama_client._build_server_cmd("/models/m.gguf", {})
    assert "--jinja" not in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "auto"
    assert cmd[cmd.index("-c") + 1] == "4096"


# ----------------------------------------------------------------- payload / schema
def test_build_payload_without_schema_has_no_response_format():
    payload = llama_client._build_payload(
        [{"role": "user", "content": "hi"}], config={}
    )
    assert "response_format" not in payload
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0.7


def test_build_payload_embeds_json_schema_response_format():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    payload = llama_client._build_payload(
        [{"role": "user", "content": "hi"}], schema=schema, config={}
    )
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema


def test_build_payload_applies_named_sampling_profile():
    cfg = {
        "max_tokens": 256,
        "sampling": {
            "default": {"temperature": 0.7, "top_p": 0.8},
            "thinking": {"temperature": 0.6, "top_p": 0.95},
        },
    }
    payload = llama_client._build_payload(
        [{"role": "user", "content": "hi"}], profile="thinking", config=cfg
    )
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 256


def test_cache_key_distinguishes_schema_and_profile():
    msgs = [{"role": "user", "content": "hi"}]
    plain = llama_client._make_cache_key(msgs)
    with_schema = llama_client._make_cache_key(msgs, schema={"type": "object"})
    with_profile = llama_client._make_cache_key(msgs, profile="thinking")
    assert len({plain, with_schema, with_profile}) == 3


# ----------------------------------------------------------------- schemas
def _assert_json_serializable(obj):
    json.dumps(obj)  # raises if not serializable


def test_schemas_are_serializable_and_well_formed():
    for schema in (
        schemas.route_schema(),
        schemas.plan_schema(),
        schemas.EXTRACTION_SCHEMA,
        schemas.INSIGHT_SCHEMA,
    ):
        _assert_json_serializable(schema)
        assert schema["type"] == "object"
        assert schema["required"]
        for field in schema["required"]:
            assert field in schema["properties"]


def test_plan_schema_action_enum_matches_contract():
    action_enum = schemas.plan_schema()["properties"]["actions"]["items"][
        "properties"
    ]["action"]["enum"]
    assert set(action_enum) == {a.value for a in ActionType}


def test_plan_schema_respects_max_steps():
    assert schemas.plan_schema(max_steps=3)["properties"]["actions"]["maxItems"] == 3


def test_route_schema_pathway_enum_includes_planner_full_and_known():
    enum = schemas.route_schema()["properties"]["pathway"]["enum"]
    assert "planner_full" in enum
    assert "reminder_reply" in enum
    assert "research_reply" in enum
