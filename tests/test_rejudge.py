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

            async def fake_judge(question, *args):
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
