"""SQL-first complex seeds: double oracle (SQL vs pandas on the CSV) and the SQL-match layer."""
import json
from pathlib import Path
import tempfile
import unittest

from text2sql.agents.planned_agent import tools
from text2sql.datagen import complex as cx
from text2sql.datagen.oracle import compute_oracle

PROJECT = Path(__file__).resolve().parents[1]
CHECKED_IN = PROJECT / "data" / "seeds" / "complex_seeds.jsonl"
SEEDS, PROBLEMS = cx.build_complex_seeds()


def complex_seed(seed_id):
    return json.loads(json.dumps(next(s for s in SEEDS if s["seed_id"] == seed_id)))


class ComplexOracleTests(unittest.TestCase):
    def test_every_complex_sql_matches_independent_pandas(self):
        self.assertEqual(PROBLEMS, [])
        self.assertEqual(len(SEEDS), 34)
        self.assertEqual(len({s["template"] for s in SEEDS}), 34)

    def test_checked_in_seeds_are_the_built_seeds(self):
        checked = [json.loads(l) for l in CHECKED_IN.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(checked, json.loads(json.dumps(SEEDS)))

    def test_double_oracle_catches_wrong_sql(self):
        # Each mutation is a plausible text2sql mistake; the pandas implementation must disagree.
        mutations = {
            "region_yoy": ("DESC, region", "ASC, region"),
            "h2_vs_h1": ("<= 6", "< 6"),
            "nth_customer": ("rn = {n}", "rn = {n} + 1"),
            "aov_region": ('COUNT("Order ID")', 'COUNT(DISTINCT "Customer Name")'),
            "retained": ("> 0 AND", "> 0 OR"),
            "large_order_share": (">= {qty}", "> {qty}"),
        }
        df = cx.frame()
        for key, params, *_ in ((s[0], s[4][0][0]) for s in cx.SPECS):
            if key not in mutations:
                continue
            old, new = mutations[key]
            self.assertIn(old, cx.SQL[key], key)
            rows = tools.read_sales(cx.SQL[key].replace(old, new).format(**params))["rows"]
            self.assertTrue(cx.compare_rows(rows, cx.REFERENCE[key](df, **params)), key)

    def test_month_over_month_january_uses_previous_december(self):
        rows = complex_seed("X-region_mom-APAC-2024")["reference_rows"]
        self.assertEqual(rows[0], ["2024-01", -34.01])  # was None before run 20261107-153104's finding
        self.assertEqual(len(rows), 12)

    def test_rules_catch_leaks_found_in_review_of_run_161201(self):
        from text2sql.datagen.checks import rule_check
        cases = {
            ("X-high_price_orders-500", "單價超過五百的訂單，各產品類別分別有几單？"): "簡體字 几",
            ("X-asp_change-3", "2024 年比 2023 年，哪三個產品的平均單價漲幅最大？各漲了多少錢？"): "不得出現「幅」",
            ("X-product_growth_amount-3", "2024年較2023年銷售額增加幅度最大的前三個產品，金額各增多少？"): "不得出現「幅」",
            ("X-region_top_customers_orders-EMEA-2024-5", "2024 年 EMEA 地區哪五家客戶賣得最多？順便告訴我有幾張訂單。"): "賣得最多",
        }
        for (seed_id, question), reason in cases.items():
            self.assertIn(reason, " ".join(rule_check(question, complex_seed(seed_id), set())), question)
        # Run 161943: ranking direction dropped -> must fail; the requested order-count column -> must pass.
        dropped = "2024年比2023年營收金額多出來的三個產品是哪幾個？各自多了多少錢？"
        self.assertIn("排名方向", " ".join(rule_check(dropped, complex_seed("X-product_growth_amount-3"), set())))
        extra = "2024 年 EMEA 地區銷售金額最高的五個客戶是誰？請列出他們各自的訂單筆數。"
        self.assertEqual(rule_check(extra, complex_seed("X-region_top_customers_orders-EMEA-2024-5"), set()), [])
        fine = "查 2024 年 EMEA 銷售額前 5 的客戶及訂單數。"
        self.assertEqual(rule_check(fine, complex_seed("X-region_top_customers_orders-EMEA-2024-5"), set()), [])

    def test_pipeline_oracle_refuses_disagreement(self):
        bad = complex_seed("X-pareto-80")
        bad["reference_rows"] = [[4]]
        with self.assertRaisesRegex(ValueError, "SQL and pandas disagree"):
            compute_oracle(bad)
        self.assertEqual(compute_oracle(complex_seed("X-pareto-80"))["rows"], [[5]])


if __name__ == "__main__":
    unittest.main()
