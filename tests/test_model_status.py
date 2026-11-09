import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web_ui"))
import model_status  # noqa: E402

BASE = "http://router:18080/v1"


def fake_urlopen(responses):
    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(url, timeout):
        for key, body in responses.items():
            if key in url:
                if isinstance(body, Exception):
                    raise body
                return Resp(body)
        raise AssertionError(url)
    return opener


def models(state):
    return {"data": [{"id": "other", "status": {"value": "unloaded"}},
                     {"id": "swift", "status": {"value": state}}]}


class ModelStatusTests(unittest.TestCase):
    def check(self, responses):
        with patch.object(model_status, "urlopen", fake_urlopen(responses)):
            return model_status.model_status(BASE, "openai/swift")[0]

    def test_idle(self):
        self.assertEqual(self.check({"/v1/models": models("loaded"),
                                     "/slots?model=swift": [{"is_processing": False}]}), "idle")

    def test_busy_when_any_slot_processing(self):
        self.assertEqual(self.check({"/v1/models": models("loaded"),
                                     "/slots?model=swift": [{"is_processing": True}]}), "busy")

    def test_model_not_loaded(self):
        self.assertEqual(self.check({"/v1/models": models("unloaded")}), "not_loaded")
        self.assertEqual(self.check({"/v1/models": models("loading")}), "loading")

    def test_unreachable(self):
        self.assertEqual(self.check({"/v1/models": OSError("refused")}), "down")

    def test_slots_url_drops_v1_suffix(self):
        seen = []
        opener = fake_urlopen({"/v1/models": models("loaded"), "/slots": [{"is_processing": False}]})

        def spy(url, timeout):
            seen.append(url)
            return opener(url, timeout)
        with patch.object(model_status, "urlopen", spy):
            model_status.model_status(BASE, "openai/swift")
        self.assertEqual(seen, ["http://router:18080/v1/models", "http://router:18080/slots?model=swift"])


if __name__ == "__main__":
    unittest.main()
