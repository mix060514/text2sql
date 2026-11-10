import json
import re
import unittest
from pathlib import Path

from text2sql.agents.planned_agent import tools
from text2sql.agents.planned_agent.prompts import DOMAIN

ROOT = Path(__file__).resolve().parents[1]


class QuarterRuleTests(unittest.TestCase):
    """Q94 (2026-11-07 eval): the planner mapped month 1/2/3 to Q1/Q2/Q3 and every other month
    to Q4; the reviewer accepted it. The prompt must spell out the month->quarter rule."""

    def quarter_expr(self):
        match = re.search(r"季度編號\s*=\s*(.+?)；", DOMAIN)
        self.assertIsNotNone(match, "DOMAIN must define the quarter expression")
        return match.group(1)

    def test_quarter_expression_matches_ground_truth(self):
        expr = self.quarter_expr()
        truth = next(json.loads(l) for l in (ROOT / "data/seeds/golden_seeds.jsonl").open(encoding="utf-8")
                     if json.loads(l)["seed_id"] == "N12-2023")["reference_answer"]
        sql = (f"SELECT 'Q' || ({expr}) AS q, ROUND(SUM(\"Total Revenue\"), 2) AS revenue FROM sales_data "
               f"WHERE substr(\"Order Date\", 1, 4) = '2023' GROUP BY q ORDER BY q")
        rows = tools.read_sales(sql)["rows"]
        self.assertEqual({q: v for q, v in rows}, truth)


if __name__ == "__main__":
    unittest.main()
