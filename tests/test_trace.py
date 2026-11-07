import json
import unittest
from unittest.mock import patch

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from text2sql.agents.planned_agent.codex_workflow import root_agent as codex_root
from text2sql.agents.planned_agent.contracts import AnswerSubmission
from text2sql.agents.planned_agent.qwen_workflow import root_agent as qwen_root
from text2sql.agents.planned_agent.tools import step_run_sql
from text2sql.runtime import run_question

REGION_SQL = 'SELECT Region, SUM("Total Revenue") AS Revenue FROM sales_data GROUP BY Region ORDER BY Region'
ACCEPT = json.dumps({"accepted": True, "reason": "ok"})
REJECT = json.dumps({"accepted": False, "reason": "wrong metric"})


def steps(state):
    return [(entry["name"], entry["attempt"], entry["status"]) for entry in state["workflow_trace"]]


class FakeModel(BaseLlm):
    replies: list[str]

    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.replies.pop(0))]))


class QwenTraceTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, sql_replies, report_replies, review_replies=None):
        agent = qwen_root.clone()
        agent.sub_agents[0].model = FakeModel(model="fake_sql", replies=sql_replies)
        agent.sub_agents[1].model = FakeModel(model="fake_review",
                                              replies=review_replies or [ACCEPT] * len(sql_replies))
        agent.sub_agents[2].model = FakeModel(model="fake_report", replies=report_replies)
        return InMemoryRunner(agent=agent, app_name="trace_test")

    def sql(self, query="SELECT COUNT(*) AS Orders FROM sales_data"):
        return json.dumps({"sql": query})

    def report(self, **kwargs):
        return json.dumps({"answer": "實際資料。", **kwargs}, ensure_ascii=False)

    async def test_happy_path_records_every_step_with_io(self):
        runner = self.runner([self.sql(REGION_SQL)],
                             [self.report(chart_type="bar", x_column="Region", y_column="Revenue")])
        state = await run_question(runner, "各區域營收")
        self.assertEqual(steps(state), [
            ("agent_sql", 1, "ok"), ("step_run_sql", 1, "ok"), ("agent_review", 1, "ok"),
            ("agent_report", 1, "ok"), ("step_publish", 1, "ok"), ("step_draw_chart", 1, "ok"),
        ])
        by_name = {entry["name"]: entry for entry in state["workflow_trace"]}
        self.assertEqual(by_name["agent_sql"]["input"]["user_question"], "各區域營收")
        self.assertEqual(by_name["agent_sql"]["output"]["sql"], REGION_SQL)
        self.assertEqual(by_name["step_run_sql"]["output"]["columns"], ["Region", "Revenue"])
        self.assertEqual(by_name["step_run_sql"]["output"]["row_count"], 4)
        self.assertEqual(by_name["agent_report"]["output"]["chart_type"], "bar")
        self.assertEqual(by_name["step_draw_chart"]["output"]["plotly_traces"][0]["type"], "bar")
        self.assertEqual(by_name["step_draw_chart"]["output"]["plotly_traces"][0]["x"],
                         ["APAC", "EMEA", "LATAM", "North America"])
        self.assertEqual(by_name["agent_sql"]["kind"], "agent")
        self.assertEqual(by_name["step_run_sql"]["kind"], "step")
        self.assertIsInstance(by_name["agent_sql"]["ms"], int)

    async def test_retries_are_visible(self):
        runner = self.runner([self.sql("SELECT missing FROM sales_data"), self.sql(REGION_SQL), self.sql(REGION_SQL)],
                             [self.report(), self.report(chart_type="bar", x_column="Region", y_column="Revenue")],
                             review_replies=[REJECT, ACCEPT])
        state = await run_question(runner, "各區域營收")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(steps(state), [
            ("agent_sql", 1, "ok"), ("step_run_sql", 1, "error"),
            ("agent_sql", 2, "ok"), ("step_run_sql", 2, "ok"), ("agent_review", 2, "rejected"),
            ("agent_sql", 3, "ok"), ("step_run_sql", 3, "ok"), ("agent_review", 3, "ok"),
            ("agent_report", 1, "ok"), ("step_publish", 1, "error"),
            ("agent_report", 2, "ok"), ("step_publish", 2, "ok"), ("step_draw_chart", 2, "ok"),
        ])
        trace = state["workflow_trace"]
        self.assertIn("no such column", trace[1]["output"]["error"]["message"])
        self.assertIn("no such column", trace[2]["input"]["sql_feedback"])
        self.assertEqual(trace[4]["output"]["reason"], "wrong metric")
        self.assertIn("require a chart", trace[10]["input"]["report_feedback"])

    async def test_failure_keeps_trace(self):
        runner = self.runner([self.sql("DELETE FROM sales_data")] * 3, [])
        state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual([name for name, _, _ in steps(state)], ["agent_sql", "step_run_sql"] * 3)

    async def test_new_turn_starts_empty_trace(self):
        runner = self.runner([self.sql()], [self.report()])
        state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(steps(state)[-1], ("step_draw_chart", 1, "skipped"))
        json.dumps(state["workflow_trace"], ensure_ascii=False)  # serializable for the UI


class CodexTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_trace(self):
        evidence = {}
        step_run_sql(REGION_SQL, evidence)
        spec = AnswerSubmission(answer="ok", chart_type="pie", x_column="Region", y_column="Revenue")
        with patch("text2sql.agents.planned_agent.codex_workflow.run_codex",
                   return_value=(spec, evidence, {"query_calls": 2})):
            state = await run_question(InMemoryRunner(agent=codex_root.clone(), app_name="t"), "佔比")
        self.assertEqual(steps(state), [("agent_codex", 1, "ok"), ("tool_query_sales", 2, "ok"),
                                        ("step_publish", 1, "ok"), ("step_draw_chart", 1, "ok")])
        self.assertEqual(state["workflow_trace"][1]["kind"], "tool")
        self.assertEqual(state["workflow_trace"][3]["output"]["plotly_traces"][0]["type"], "pie")

    async def test_codex_failure_keeps_trace(self):
        with patch("text2sql.agents.planned_agent.codex_workflow.run_codex", side_effect=RuntimeError("boom")):
            state = await run_question(InMemoryRunner(agent=codex_root.clone(), app_name="t"), "x")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual(steps(state), [("agent_codex", 1, "error")])
