import importlib
import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from text2sql.agents.planned_agent import prompts, tools

ROOT = Path(__file__).resolve().parents[1]


def eval_questions():
    qs = {json.loads(l)["question"] for l in (ROOT / "data/eval_set_v3.jsonl").open(encoding="utf-8") if l.strip()}
    for path in (ROOT / "data/synthetic/runs").glob("*/dataset.jsonl"):
        qs |= {json.loads(l)["question"] for l in path.open(encoding="utf-8") if l.strip()}
    return qs


class FewShotTests(unittest.TestCase):
    def test_every_example_sql_runs_on_the_real_database(self):
        self.assertGreater(len(prompts.SQL_EXAMPLES), 0)
        for ex in prompts.SQL_EXAMPLES:
            result = tools.read_sales(ex["sql"])
            self.assertTrue(result["rows"], ex["question"])

    def test_examples_are_not_evaluation_questions(self):
        asked = eval_questions()
        for ex in prompts.SQL_EXAMPLES:
            self.assertNotIn(ex["question"], asked)

    def test_examples_do_not_reproduce_any_oracle_answer(self):
        """Same type, different content: no example returns the rows of an evaluation answer."""
        answers = []
        for path in (ROOT / "data/synthetic/runs").glob("*/dataset.jsonl"):
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    rows = (json.loads(line).get("ground_truth") or {}).get("rows")
                    if rows:
                        answers.append([list(r) for r in rows])
        self.assertTrue(answers)
        for ex in prompts.SQL_EXAMPLES:
            rows = [list(r) for r in tools.read_sales(ex["sql"])["rows"]]
            self.assertNotIn(rows, answers, ex["question"])

    def test_flag_controls_the_planner_prompt(self):
        from text2sql.agents.planned_agent import qwen_workflow as qw
        sql = prompts.SQL_EXAMPLES[0]["sql"]
        self.assertNotIn(sql, qw.sql_instruction(False))
        self.assertIn(sql, qw.sql_instruction(True))
        self.assertTrue(qw.sql_instruction(False).startswith(prompts.DOMAIN))

    def test_few_shot_is_on_by_default_and_can_be_turned_off(self):
        from text2sql import config
        env = {k: v for k, v in os.environ.items() if k != "SQL_FEW_SHOT"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(importlib.reload(config).SQL_FEW_SHOT)
        with patch.dict(os.environ, {"SQL_FEW_SHOT": "false"}):
            self.assertFalse(importlib.reload(config).SQL_FEW_SHOT)
        importlib.reload(config)

if __name__ == "__main__":
    unittest.main()
