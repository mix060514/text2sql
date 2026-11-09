import os
import subprocess
import sys
import unittest
from pathlib import Path


class BackendSelectionTests(unittest.TestCase):
    def selected(self, backend):
        src = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, TEXT2SQL_BACKEND=backend,
                   PYTHONPATH=os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
        code = "from text2sql.agents.planned_agent.agent import root_agent; print(type(root_agent).__name__)"
        return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)

    def test_default_backend(self):
        self.assertEqual(self.selected("qwen27b").stdout.strip(), "Text2SQLWorkflow")

    def test_removed_and_unknown_backends_are_rejected(self):
        for backend in ("codex", "gpt"):
            result = self.selected(backend)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("TEXT2SQL_BACKEND", result.stderr)


if __name__ == "__main__":
    unittest.main()
