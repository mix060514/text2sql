import asyncio
import json
import unittest
from unittest.mock import patch

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from text2sql.agents.planned_agent.agent import root_agent
from text2sql.runtime import run_question, stream_turn


class FakeModel(BaseLlm):
    replies: list[str]
    calls: int = 0
    delay: float = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.replies.pop(0))]))


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, sql_replies, report_replies):
        agent = root_agent.clone()
        agent.sub_agents[0].model = FakeModel(model="fake_sql", replies=sql_replies)
        agent.sub_agents[1].model = FakeModel(model="fake_report", replies=report_replies)
        return InMemoryRunner(agent=agent, app_name="workflow_test")

    def sql(self, query='SELECT COUNT(*) AS Orders FROM sales_data'):
        return json.dumps({"sql": query})

    def report(self, **kwargs):
        return json.dumps({"answer": "實際資料共有 30,000 筆。", **kwargs}, ensure_ascii=False)

    async def test_consumes_both_stages_after_first_agent_final_event(self):
        runner = self.runner([self.sql()], [self.report()])
        state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["raw_query_result"], [[30000]])
        self.assertEqual(runner.agent.sub_agents[1].model.calls, 1)

    async def test_invalid_sql_repairs_only_planning_stage(self):
        runner = self.runner([self.sql('SELECT missing FROM sales_data'), self.sql()], [self.report()])
        state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["sql_attempts"], 2)
        self.assertEqual(runner.agent.sub_agents[1].model.calls, 1)

    async def test_structural_failure_has_bounded_repair(self):
        runner = self.runner(['not JSON', self.sql()], [self.report()])
        state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["sql_attempts"], 2)

    async def test_sql_exhaustion_never_runs_answer_stage(self):
        runner = self.runner([self.sql('DELETE FROM sales_data')] * 3, [])
        state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertFalse(state["query_success"])
        self.assertEqual(runner.agent.sub_agents[1].model.calls, 0)
        self.assertEqual(state["workflow_error"]["code"], "SQL_ATTEMPTS_EXHAUSTED")

    async def test_invalid_chart_repairs_report_without_requery(self):
        query = 'SELECT Region, SUM("Total Revenue") AS Revenue FROM sales_data GROUP BY Region ORDER BY Region'
        runner = self.runner([self.sql(query)], [self.report(), self.report(chart_type="bar", x_column="Region", y_column="Revenue")])
        state = await run_question(runner, "各區域營收")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["report_attempts"], 2)
        self.assertEqual(runner.agent.sub_agents[0].model.calls, 1)
        self.assertTrue(state["chart_json"])

    async def test_report_exhaustion_fails_closed(self):
        runner = self.runner([self.sql()], ['not JSON', 'not JSON'])
        state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual(state["workflow_error"]["code"], "REPORT_ATTEMPTS_EXHAUSTED")

    async def test_empty_result_does_not_trigger_sql_retry(self):
        runner = self.runner([self.sql("SELECT Region FROM sales_data WHERE Region = 'missing'")], [self.report(answer="沒有符合條件的資料。")])
        state = await run_question(runner, "missing 資料")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["raw_query_result"], [])
        self.assertEqual(state["sql_attempts"], 1)

    async def test_adk_total_call_budget_is_enforced(self):
        runner = self.runner([self.sql()], [self.report()])
        with patch("text2sql.runtime.MAX_LLM_CALLS", 1):
            state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertIn("LlmCalls", state["workflow_error"]["code"])

    async def test_second_turn_does_not_reuse_first_chart(self):
        query = 'SELECT Region, SUM("Total Revenue") AS Revenue FROM sales_data GROUP BY Region ORDER BY Region'
        runner = self.runner([self.sql(query), self.sql()], [self.report(chart_type="bar", x_column="Region", y_column="Revenue"), self.report()])
        await runner.session_service.create_session(app_name=runner.app_name, user_id="test", session_id="same")
        for question in ("各區域營收", "訂單數是多少？"):
            async for _ in stream_turn(runner, user_id="test", session_id="same", new_message=types.Content(role="user", parts=[types.Part(text=question)])):
                pass
            session = await runner.session_service.get_session(app_name=runner.app_name, user_id="test", session_id="same")
            if question == "各區域營收":
                self.assertTrue(session.state["chart_json"])
        self.assertIsNone(session.state["chart_json"])
        self.assertEqual(session.state["raw_query_result"], [[30000]])

    async def test_invocation_timeout_cannot_publish_answer(self):
        runner = self.runner([self.sql()], [self.report()])
        runner.agent.sub_agents[1].model.delay = 1
        with patch("text2sql.runtime.AGENT_TIMEOUT_SECONDS", 0.05):
            state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual(state["workflow_error"]["code"], "TimeoutError")
        self.assertIsNone(state["chart_json"])


if __name__ == "__main__":
    unittest.main()
