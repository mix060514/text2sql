import asyncio
import json
import unittest
from unittest.mock import patch

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from text2sql.agents.planned_agent.qwen_workflow import root_agent
from text2sql.agents.planned_agent.tools import read_sales
from text2sql.runtime import run_question, stream_turn


class FakeModel(BaseLlm):
    replies: list[str]
    calls: int = 0
    delay: float = 0
    prompts: list[str] = []

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        self.prompts.append(str(llm_request))
        if self.delay:
            await asyncio.sleep(self.delay)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.replies.pop(0))]))


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, sql_replies, report_replies, review_replies=None):
        agent = root_agent.clone()
        agent.sub_agents[0].model = FakeModel(model="fake_sql", replies=sql_replies)
        agent.sub_agents[1].model = FakeModel(model="fake_review", replies=review_replies if review_replies is not None else [json.dumps({"accepted": True, "reason": "Matches requested metric and scope"})] * len(sql_replies))
        agent.sub_agents[2].model = FakeModel(model="fake_report", replies=report_replies)
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
        self.assertEqual(runner.agent.sub_agents[2].model.calls, 1)

    async def test_invalid_sql_repairs_only_planning_stage(self):
        runner = self.runner([self.sql('SELECT missing FROM sales_data'), self.sql()], [self.report()])
        state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["sql_attempts"], 2)
        self.assertEqual(runner.agent.sub_agents[2].model.calls, 1)

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
        self.assertEqual(runner.agent.sub_agents[2].model.calls, 0)
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

    async def test_executable_wrong_metric_rewrites_and_executes_quantity(self):
        wrong = 'SELECT COUNT(DISTINCT "Order ID") AS Units FROM sales_data'
        right = 'SELECT SUM("Quantity") AS Units FROM sales_data'
        reason = "銷售量必須 SUM(Quantity)，不是訂單數"
        runner = self.runner([self.sql(wrong), self.sql(right)], [self.report()],
                             [json.dumps({"accepted": False, "reason": reason}),
                              json.dumps({"accepted": True, "reason": "Correct quantity"})])
        state = await run_question(runner, "總銷售量是多少？")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual(state["raw_query_result"], read_sales(right)["rows"])
        self.assertNotEqual(state["raw_query_result"], read_sales(wrong)["rows"])
        self.assertEqual(state["sql_attempts"], 2)
        self.assertIn(reason, runner.agent.sub_agents[0].model.prompts[1])
        reviewer_prompt = runner.agent.sub_agents[1].model.prompts[0]
        for value in ("總銷售量是多少", wrong, "Units", "30000", "追問意圖", "SUM"):
            self.assertIn(value, reviewer_prompt)
        self.assertEqual(runner.agent.sub_agents[1].model.calls, 2)
        self.assertEqual(runner.agent.sub_agents[2].model.calls, 1)

    async def test_all_semantic_rejections_never_publish_or_report(self):
        runner = self.runner([self.sql()] * 3, [],
                             [json.dumps({"accepted": False, "reason": "Wrong metric"})] * 3)
        state = await run_question(runner, "總銷售量是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual(state["workflow_error"]["code"], "SQL_ATTEMPTS_EXHAUSTED")
        self.assertEqual(runner.agent.sub_agents[2].model.calls, 0)
        self.assertEqual(runner.agent.sub_agents[1].model.calls, 3)
        self.assertFalse(state["query_success"])
        self.assertIsNone(state["semantic_review"])
        self.assertIsNone(state["reviewed_query"])
        self.assertEqual(state["raw_query_result"], [])
        self.assertNotIn("30,000", state["final_answer"])
        self.assertIsNone(state["chart_json"])

    async def test_rejected_result_cannot_survive_later_schema_or_sql_failure(self):
        runner = self.runner([self.sql(), 'not JSON', self.sql('SELECT missing FROM sales_data')], [],
                             [json.dumps({"accepted": False, "reason": "Wrong metric"})])
        state = await run_question(runner, "總銷售量是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual(state["sql_attempts"], 3)
        self.assertEqual(runner.agent.sub_agents[1].model.calls, 1)
        self.assertEqual(runner.agent.sub_agents[2].model.calls, 0)
        self.assertFalse(state["query_success"])
        self.assertEqual(state["raw_query_result"], [])
        self.assertIsNone(state["reviewed_query"])
        self.assertIsNone(state["chart_json"])


    async def test_malformed_review_is_never_acceptance(self):
        for reply in ('not JSON', '{"accepted": "true", "reason": "ok"}',
                      '{"accepted": true}', '{"accepted": true, "reason": " "}'):
            with self.subTest(reply=reply):
                runner = self.runner([self.sql()] * 3, [], [reply] * 3)
                state = await run_question(runner, "訂單數是多少？")
                self.assertEqual(state["workflow_status"], "failed")
                self.assertEqual(runner.agent.sub_agents[2].model.calls, 0)
                self.assertEqual(state["raw_query_result"], [])

    async def test_query_and_report_repairs_fit_original_eight_call_budget(self):
        query = 'SELECT Region, SUM("Total Revenue") AS Revenue FROM sales_data GROUP BY Region ORDER BY Region'
        runner = self.runner([self.sql(query)] * 3,
                             [self.report(), self.report(chart_type="bar", x_column="Region", y_column="Revenue")],
                             [json.dumps({"accepted": False, "reason": "Repair scope"})] * 2 +
                             [json.dumps({"accepted": True, "reason": "Scope matches"})])
        state = await run_question(runner, "各區域營收")
        self.assertEqual(state["workflow_status"], "completed")
        self.assertEqual([agent.model.calls for agent in runner.agent.sub_agents], [3, 3, 2])
        self.assertEqual(state["sql_attempts"], 3)
        self.assertEqual(state["report_attempts"], 2)


    async def test_previous_turn_acceptance_cannot_pass_current_review(self):
        runner = self.runner([self.sql()] * 4, [self.report()],
                             [json.dumps({"accepted": True, "reason": "Orders requested"})] +
                             [json.dumps({"accepted": False, "reason": "Quantity requested"})] * 3)
        await runner.session_service.create_session(app_name=runner.app_name, user_id="test", session_id="isolation")
        for question in ("訂單數是多少？", "那銷售量呢？"):
            async for _ in stream_turn(runner, user_id="test", session_id="isolation",
                                       new_message=types.Content(role="user", parts=[types.Part(text=question)])):
                pass
            session = await runner.session_service.get_session(app_name=runner.app_name, user_id="test", session_id="isolation")
            if question == "訂單數是多少？":
                self.assertEqual(session.state["workflow_status"], "completed")
                self.assertTrue(session.state["semantic_review"]["accepted"])
        self.assertEqual(session.state["workflow_status"], "failed")
        self.assertIsNone(session.state["reviewed_query"])
        self.assertEqual(session.state["raw_query_result"], [])
        self.assertEqual(runner.agent.sub_agents[2].model.calls, 1)
        self.assertIn("訂單數是多少", runner.agent.sub_agents[1].model.prompts[1])


    async def test_invocation_timeout_cannot_publish_answer(self):
        runner = self.runner([self.sql()], [self.report()])
        runner.agent.sub_agents[2].model.delay = 1
        with patch("text2sql.runtime.AGENT_TIMEOUT_SECONDS", 0.05):
            state = await run_question(runner, "訂單數是多少？")
        self.assertEqual(state["workflow_status"], "failed")
        self.assertEqual(state["workflow_error"]["code"], "TimeoutError")
        self.assertIsNone(state["chart_json"])


if __name__ == "__main__":
    unittest.main()


class ThinkingConfigTests(unittest.TestCase):
    def test_only_report_writer_disables_thinking(self):
        from text2sql.agents.planned_agent import qwen_workflow as qw

        off = {"chat_template_kwargs": {"enable_thinking": False}}
        self.assertEqual(qw.agent_report.model._additional_args.get("extra_body"), off)
        for agent in (qw.agent_sql, qw.agent_review):
            self.assertNotIn("extra_body", agent.model._additional_args)
        self.assertNotIn("extra_body", qw.model(thinking=True)._additional_args)
