import json
import unittest
from unittest.mock import patch

from google.adk.runners import InMemoryRunner
from google.genai import types

from text2sql.agents.planned_agent.agent import root_agent
from text2sql.runtime import run_question, stream_turn


class GreetingTests(unittest.IsolatedAsyncioTestCase):
    def runner(self):
        return InMemoryRunner(agent=root_agent.clone(), app_name="greeting_test")

    async def test_bare_greetings_skip_entire_query_workflow(self):
        runner = self.runner()
        with patch.object(type(runner.agent), "_run_async_impl", side_effect=AssertionError("must not run")) as body:
            for question in ("hello", " HELLO!! ", "你好！", "嗨", "ＨＥＬＬＯ"):
                with self.subTest(question=question):
                    state = await run_question(runner, question)
                    self.assertEqual(state["workflow_status"], "completed")
                    self.assertEqual(state["interaction_kind"], "greeting")
                    self.assertIn("你好", state["final_answer"])
                    self.assertNotIn("30,000", state["final_answer"])
                    self.assertFalse(state["query_success"])
                    self.assertEqual(state["sql_query"], "")
                    self.assertEqual(state["sql_attempts"], 0)
            body.assert_not_called()

    async def test_greeting_with_actual_question_enters_query_workflow(self):
        runner = self.runner()
        with patch.object(type(runner.agent), "_run_async_impl", side_effect=RuntimeError("test query path")) as body:
            state = await run_question(runner, "hello，請查南美的總銷售額")
        body.assert_called_once()
        self.assertEqual(state["interaction_kind"], "query")

    async def test_greetings_clear_artifacts_and_preserve_followup_intent(self):
        runner = self.runner()
        previous = {"question": "南美總銷售額", "sql": 'SELECT SUM("Total Revenue") FROM sales_data WHERE Region = \'LATAM\''}
        await runner.session_service.create_session(
            app_name=runner.app_name, user_id="test", session_id="same",
            state={"workflow_status": "completed", "query_success": True,
                   "user_question": previous["question"], "sql_query": previous["sql"],
                   "chart_json": "old chart", "raw_query_result": [[123]],
                   "final_answer": "old answer", "sql_attempts": 3,
                   "semantic_review": {"accepted": True, "reason": "old"}, "reviewed_query": {"old": True}},
        )
        with patch.object(type(runner.agent), "_run_async_impl", side_effect=AssertionError("must not run")) as body:
            for question in ("hello", "你好"):
                async for _ in stream_turn(runner, user_id="test", session_id="same",
                                           new_message=types.Content(role="user", parts=[types.Part(text=question)])):
                    pass
                session = await runner.session_service.get_session(app_name=runner.app_name, user_id="test", session_id="same")
                self.assertIsNone(session.state["chart_json"])
                self.assertIsNone(session.state["semantic_review"])
                self.assertIsNone(session.state["reviewed_query"])
                self.assertEqual(session.state["raw_query_result"], [])
                self.assertEqual(json.loads(session.state["previous_query_context"]), previous)
                self.assertEqual(session.state["sql_attempts"], 0)
            body.assert_not_called()
        with patch.object(type(runner.agent), "_run_async_impl", side_effect=RuntimeError("test followup")) as body:
            async for _ in stream_turn(runner, user_id="test", session_id="same",
                                       new_message=types.Content(role="user", parts=[types.Part(text="那北美呢？")])):
                pass
        body.assert_called_once()
        session = await runner.session_service.get_session(app_name=runner.app_name, user_id="test", session_id="same")
        self.assertEqual(json.loads(session.state["previous_query_context"]), previous)
