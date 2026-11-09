import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import eval_e2e  # noqa: E402

from text2sql import config  # noqa: E402
from text2sql.agents.judge_agent.agent import judge_agent  # noqa: E402


def row(question, reasoning, correct=False, **extra):
    base = dict(question=question, eval_type="number", difficulty="simple", sql_query="SELECT 1",
                sql_result=[[1]], answer="1", sql_correct=correct, sql_score=10 if correct else 0,
                sql_reasoning=reasoning, has_expected_plot=False, expected_should_plot=False,
                actual_should_plot=False, expected_chart_type=None, actual_chart_type=None,
                plot_decision_correct=True, chart_json_valid=True, trace_type_correct=True,
                x_order_correct=True, trace_count_correct=True, e2e_success=correct)
    base.update(extra)
    return base


class RejudgeTests(unittest.IsolatedAsyncioTestCase):
    def test_judge_uses_bounded_streaming_model_with_thinking_budget(self):
        from text2sql.agents.planned_agent.qwen_workflow import BoundedStreamingLiteLlm
        self.assertIsInstance(judge_agent.model, BoundedStreamingLiteLlm)
        self.assertEqual(judge_agent.model._additional_args["extra_body"]["thinking_budget_tokens"],
                         config.JUDGE_THINKING_BUDGET)

    async def test_rejudge_only_retries_missing_judgements_and_recomputes_success(self):
        rows = [
            row("ok", "fine", correct=True),
            row("missing", "Judge Agent did not return result"),
            row("timeout", "{'code': 'Timeout'}", sql_query="", answer=""),
            row("plot", "Judge error: boom", has_expected_plot=True, expected_should_plot=True,
                x_order_correct=False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "e2e_results.json").write_text(json.dumps(rows), encoding="utf-8")
            (run / "e2e_summary.json").write_text(json.dumps({"run_metadata": {"model": "m"}}), encoding="utf-8")
            calls = []

            async def fake_judge(question, *args, **kwargs):
                calls.append(question)
                return True, 9, "re-judged"

            with patch.object(eval_e2e, "judge_sql_result", fake_judge):
                summary = await eval_e2e.rejudge_run(run)
            saved = {r["question"]: r for r in json.loads((run / "e2e_results.json").read_text())}
            self.assertTrue((run / "e2e_results.before_rejudge.json").exists())

        self.assertEqual(calls, ["missing", "plot"])
        self.assertTrue(saved["missing"]["e2e_success"])
        self.assertEqual(saved["missing"]["sql_reasoning"], "re-judged")
        self.assertTrue(saved["plot"]["sql_correct"])
        self.assertFalse(saved["plot"]["e2e_success"])  # chart still wrong
        self.assertFalse(saved["timeout"]["e2e_success"])
        self.assertEqual(summary["run_metadata"], {"model": "m"})
        self.assertAlmostEqual(summary["e2e_success_rate"], 2 / 4)


if __name__ == "__main__":
    unittest.main()


class XOrderTests(unittest.TestCase):
    """Q101-Q105: chart x was 2023-Q1.. (project convention YYYY-Qn), eval set wrote 2023 Q1.."""

    def check(self, x, expected):
        chart = {"data": [{"type": "scatter", "x": x, "y": [1] * len(x)}]}
        plot = {"should_plot": True, "chart_type": "line", "expected_x_order": expected, "expected_traces": 1}
        return eval_e2e.validate_plotly_json(json.dumps(chart), plot)["x_order_correct"]

    def test_quarter_label_separator_is_ignored(self):
        self.assertTrue(self.check(["2023-Q1", "2023-Q2", "2024-Q1"], ["2023 Q1", "2023 Q2", "2024 Q1"]))

    def test_wrong_order_still_fails(self):
        self.assertFalse(self.check(["2023-Q2", "2023-Q1"], ["2023 Q1", "2023 Q2"]))

    def test_different_labels_still_fail(self):
        self.assertFalse(self.check(["2023-01", "2023-02"], ["2023 Q1", "2023 Q2"]))


class JudgePromptTests(unittest.IsolatedAsyncioTestCase):
    """Q110: a pie chart was drawn, but the judge only sees the text and docked points for 'no pie'."""

    async def test_judge_is_told_charts_are_checked_separately(self):
        prompts = []

        async def fake_run(runner, prompt, user_id="eval_user"):
            prompts.append(prompt)
            return {"judge_result": {"is_correct": True, "confidence_score": 10, "reasoning": "ok"}}

        with patch.object(eval_e2e, "run_agent_and_get_state", fake_run):
            await eval_e2e.judge_sql_result("q", 1, "a", "SELECT 1", [[1]], chart_type="pie")
        self.assertIn("圖表由系統另外繪製", prompts[0])
        self.assertIn("pie", prompts[0])


class SyntheticSubsetTests(unittest.TestCase):
    def test_subsets_split_template_complex_and_block(self):
        import synthetic
        items = [{"label": "allow", "seed_id": "A-1"}, {"label": "allow", "seed_id": "X-region"},
                 {"label": "block", "seed_id": "B01"}]
        pick = lambda subset: [i["seed_id"] for i in synthetic.select_items(items, subset)]
        self.assertEqual(pick("complex"), ["X-region"])
        self.assertEqual(pick("template"), ["A-1"])
        self.assertEqual(pick("block"), ["B01"])
        self.assertEqual(pick(None), ["A-1", "X-region", "B01"])


class EffortFieldTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_seconds_and_sql_attempts(self):
        state = {"workflow_status": "completed", "sql_query": "SELECT 1", "raw_query_result": [[1]],
                 "final_answer": "1", "workflow_trace": [{"name": "agent_sql"}, {"name": "step_run_sql"},
                                                          {"name": "step_run_sql"}]}

        async def fake_run(runner, prompt, user_id="eval_user"):
            return state

        async def fake_judge(*args, **kwargs):
            return True, 10, "ok"

        with patch.object(eval_e2e, "run_agent_and_get_state", fake_run), \
                patch.object(eval_e2e, "judge_sql_result", fake_judge):
            r = await eval_e2e.evaluate_single({"question": "q", "ground_truth": 1})
        self.assertEqual(r.sql_attempts, 2)
        self.assertGreaterEqual(r.seconds, 0.0)

    def test_old_rows_without_effort_fields_still_load(self):
        old = row("q", "fine", correct=True)
        self.assertEqual(eval_e2e.E2EResult(**old).sql_attempts, 0)
