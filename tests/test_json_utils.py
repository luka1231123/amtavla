from brain.json_utils import extract_first_json_object


def test_extract_first_json_object_direct():
    payload = extract_first_json_object('{"intent":"smalltalk","confidence":0.9}')
    assert payload == {"intent": "smalltalk", "confidence": 0.9}


def test_extract_first_json_object_from_mixed_text():
    raw = 'Result:\n```json\n{"steps":[{"action":"THINK","detail":""}]}\n```'
    payload = extract_first_json_object(raw)
    assert payload is not None
    assert payload["steps"][0]["action"] == "THINK"
