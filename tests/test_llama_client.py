#!/usr/bin/env python3

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llama_client


class LlamaClientTests(unittest.TestCase):
    def setUp(self):
        with llama_client._cache_lock:
            llama_client._response_cache.clear()

    def test_returns_error_when_llama_unavailable(self):
        messages = [{"role": "user", "content": "hello"}]
        with patch("llama_client._call_llama", side_effect=RuntimeError("down")):
            resp = llama_client.chat(messages)

        self.assertIn("Error:", resp["message"]["content"])

    def test_llama_choices_payload_is_parsed(self):
        messages = [{"role": "user", "content": "hello"}]
        llama_payload = {"choices": [{"message": {"content": "from llama"}}]}
        with patch("llama_client._call_llama", return_value=llama_payload):
            resp = llama_client.chat(messages)

        self.assertEqual(resp["message"]["content"], "from llama")

    def test_response_cache_avoids_repeated_calls(self):
        messages = [{"role": "user", "content": "cache me"}]
        with patch(
            "llama_client._call_llama",
            return_value={"choices": [{"message": {"content": "cached"}}]},
        ) as llama_mock:
            resp_1 = llama_client.chat(messages)
            resp_2 = llama_client.chat(messages)

        self.assertEqual(resp_1["message"]["content"], "cached")
        self.assertEqual(resp_2["message"]["content"], "cached")
        self.assertEqual(llama_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
