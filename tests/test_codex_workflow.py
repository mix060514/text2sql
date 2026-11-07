import asyncio
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from google.adk.runners import InMemoryRunner
from google.genai import types

from text2sql.agents.planned_agent.codex_workflow import root_agent
from text2sql.agents.planned_agent.codex_backend import stop_process
from text2sql.agents.planned_agent.contracts import AnswerSubmission
from text2sql.agents.planned_agent.mcp_server import QueryRecorder
from text2sql.agents.planned_agent.tools import execute_query
from text2sql.runtime import run_question, stream_turn


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def runner(self):
        return InMemoryRunner(agent=root_agent.clone(), app_name="strong_test")

    def evidence(self, sql='SELECT COUNT(*) AS Orders FROM sales_data'):
        state = {}
        execute_query(sql, state)
        return state

    async def test_commits_only_executed_data(self):
        spec = AnswerSubmission(answer="共 30,000 筆。")
        with patch("text2sql.agents.planned_agent.codex_workflow.run_codex", return_value=(spec, self.evidence(), {})):
            state = await run_question(self.runner(), "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["raw_query_result"], [[30000]])

    async def test_missing_query_cannot_publish(self):
        spec = AnswerSubmission(answer="invented")
        with patch("text2sql.agents.planned_agent.codex_workflow.run_codex", return_value=(spec, {"query_success": False}, {})):
            state = await run_question(self.runner(), "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertNotEqual(state["final_answer"], "invented")
        self.assertIsNone(state["chart_json"])

    async def test_invalid_chart_cannot_publish(self):
        sql = 'SELECT Region, SUM("Total Revenue") AS Revenue FROM sales_data GROUP BY Region'
        spec = AnswerSubmission(answer="invalid", chart_type="bar", x_column="invented", y_column="Revenue")
        with patch("text2sql.agents.planned_agent.codex_workflow.run_codex", return_value=(spec, self.evidence(sql), {})):
            state = await run_question(self.runner(), "各區域营收")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertNotEqual(state["final_answer"], "invalid")

    async def test_harness_error_is_not_retried_by_application(self):
        with patch("text2sql.agents.planned_agent.codex_workflow.run_codex", side_effect=RuntimeError("harness failed")) as call:
            state = await run_question(self.runner(), "訂單數是多少？")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(state["workflow_status"], "failed")

    async def test_second_turn_does_not_reuse_chart(self):
        query = 'SELECT Region, SUM("Total Revenue") AS Revenue FROM sales_data GROUP BY Region'
        replies = [
            (AnswerSubmission(answer="regional data", chart_type="bar", x_column="Region", y_column="Revenue"), self.evidence(query), {}),
            (AnswerSubmission(answer="共 30,000 筆。"), self.evidence(), {}),
        ]
        runner = self.runner()
        await runner.session_service.create_session(app_name=runner.app_name, user_id="test", session_id="same")
        with patch("text2sql.agents.planned_agent.codex_workflow.run_codex", side_effect=replies):
            for question in ("各區域营收", "訂單數是多少？"):
                async for _ in stream_turn(runner, user_id="test", session_id="same", new_message=types.Content(role="user", parts=[types.Part(text=question)])):
                    pass
        session = await runner.session_service.get_session(app_name=runner.app_name, user_id="test", session_id="same")
        self.assertIsNone(session.state["chart_json"])
        self.assertEqual(session.state["raw_query_result"], [[30000]])

    async def test_query_receipt_records_actual_last_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            recorder = QueryRecorder(path)
            self.assertTrue(recorder.query('SELECT COUNT(*) AS Orders FROM sales_data')["success"])
            self.assertFalse(recorder.query('DELETE FROM sales_data')["success"])
            receipt = json.loads(path.read_text())
            self.assertFalse(receipt["state"]["query_success"])
            self.assertEqual(receipt["state"]["raw_query_result"], [])
            self.assertEqual(receipt["query_calls"], 2)

    async def test_query_budget_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = QueryRecorder(Path(directory) / "receipt.json")
            with patch("text2sql.agents.planned_agent.mcp_server.MAX_QUERY_CALLS", 1):
                self.assertTrue(recorder.query('SELECT COUNT(*) AS n FROM sales_data')["success"])
                self.assertEqual(recorder.query('SELECT COUNT(*) AS n FROM sales_data')["error"]["code"], "QUERY_BUDGET")

    async def test_invocation_timeout_cannot_publish_answer(self):
        async def delayed(*args):
            await asyncio.sleep(1)
        with patch("text2sql.agents.planned_agent.codex_workflow.run_codex", side_effect=delayed), patch("text2sql.runtime.AGENT_TIMEOUT_SECONDS", 0.05):
            state = await run_question(self.runner(), "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual(state["workflow_error"]["code"], "TimeoutError")

    async def test_stops_owned_subprocess_on_interruption(self):
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(30)", start_new_session=True)
        await stop_process(process)
        self.assertIsNotNone(process.returncode)


class BackendSelectionTests(unittest.TestCase):
    def selected(self, backend):
        import os, subprocess
        src = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, TEXT2SQL_BACKEND=backend,
                   PYTHONPATH=os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
        code = "from text2sql.agents.planned_agent.agent import root_agent; print(type(root_agent).__name__)"
        return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)

    def test_default_and_codex_backends(self):
        self.assertEqual(self.selected("qwen27b").stdout.strip(), "Text2SQLWorkflow")
        self.assertEqual(self.selected("codex").stdout.strip(), "CodexSalesAgent")

    def test_unknown_backend_is_rejected(self):
        result = self.selected("gpt")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TEXT2SQL_BACKEND", result.stderr)


if __name__ == "__main__":
    unittest.main()
