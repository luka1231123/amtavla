#!/usr/bin/env python3

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import server.phone_server as phone_server
except ModuleNotFoundError:
    phone_server = None


@unittest.skipIf(phone_server is None, "Flask is not installed in current environment")
class PhoneServerApiTests(unittest.TestCase):
    def setUp(self):
        phone_server.app.config["TESTING"] = True
        self.client = phone_server.app.test_client()
        phone_server.command_store = None
        phone_server.response_store = None
        phone_server.command_updated_at = None
        phone_server.response_updated_at = None
        phone_server.server_started_at = time.time()

    def test_command_lifecycle(self):
        get_empty = self.client.get("/command")
        self.assertEqual(get_empty.status_code, 200)
        self.assertEqual(get_empty.get_json(), {"command": None})

        set_resp = self.client.post("/command", json={"text": "hello from web"})
        self.assertEqual(set_resp.status_code, 200)
        self.assertEqual(set_resp.get_json().get("status"), "ok")

        get_set = self.client.get("/command")
        self.assertEqual(get_set.status_code, 200)
        self.assertEqual(get_set.get_json(), {"command": "hello from web"})

        ack_resp = self.client.post("/command/ack")
        self.assertEqual(ack_resp.status_code, 200)
        self.assertEqual(ack_resp.get_json().get("status"), "ok")

        get_cleared = self.client.get("/command")
        self.assertEqual(get_cleared.status_code, 200)
        self.assertEqual(get_cleared.get_json(), {"command": None})

    def test_response_lifecycle(self):
        get_empty = self.client.get("/response")
        self.assertEqual(get_empty.status_code, 200)
        self.assertEqual(get_empty.get_json(), {"response": None})

        set_resp = self.client.post("/response", json={"text": "assistant output"})
        self.assertEqual(set_resp.status_code, 200)
        self.assertEqual(set_resp.get_json().get("status"), "ok")

        get_set = self.client.get("/response")
        self.assertEqual(get_set.status_code, 200)
        self.assertEqual(get_set.get_json(), {"response": "assistant output"})

        ack_resp = self.client.post("/response/ack")
        self.assertEqual(ack_resp.status_code, 200)
        self.assertEqual(ack_resp.get_json().get("status"), "ok")

        get_cleared = self.client.get("/response")
        self.assertEqual(get_cleared.status_code, 200)
        self.assertEqual(get_cleared.get_json(), {"response": None})

    def test_debug_state_json(self):
        self.client.post("/command", json={"text": "pending command"})
        self.client.post("/response", json={"text": "pending response"})

        resp = self.client.get("/debug/state")
        self.assertEqual(resp.status_code, 200)

        payload = resp.get_json()
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("command"), "pending command")
        self.assertEqual(payload.get("response"), "pending response")
        self.assertTrue(payload.get("pending", {}).get("command"))
        self.assertTrue(payload.get("pending", {}).get("response"))
        self.assertIsNotNone(payload.get("command_updated_at"))
        self.assertIsNotNone(payload.get("response_updated_at"))
        self.assertIsNotNone(payload.get("uptime_seconds"))

    def test_debug_page_renders(self):
        resp = self.client.get("/debug")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("amtavla Debug", body)
        self.assertIn("Overall Health", body)
        self.assertIn("/debug/state", body)


if __name__ == "__main__":
    unittest.main()
