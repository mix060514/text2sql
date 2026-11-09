import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import opt_report  # noqa: E402


class OptReportTests(unittest.TestCase):
    def test_stats_counts_timeouts_first_try_and_seconds(self):
        rows = [
            {"type": "a", "passed": True, "timeout": False, "seconds": 10, "attempts": 1},
            {"type": "a", "passed": True, "timeout": False, "seconds": 30, "attempts": 2},
            {"type": "a", "passed": False, "timeout": True, "seconds": 300, "attempts": 0},
            {"type": "b", "passed": True, "timeout": False, "seconds": 20, "attempts": 1},
        ]
        s = opt_report.by_type(rows)
        self.assertAlmostEqual(s["a"]["success"], 2 / 3)
        self.assertAlmostEqual(s["a"]["timeout"], 1 / 3)
        self.assertAlmostEqual(s["a"]["first_try"], 1 / 3)
        self.assertEqual(s["a"]["avg_seconds"], 113.3)
        self.assertEqual(opt_report.hard_types(rows), ["a"])

    def test_template_rows_from_real_eval_set(self):
        rows = [{"question": "請列出 EMEA 區域在 2023 至 2024 年每個月的銷售趨勢數據。", "e2e_success": False,
                 "sql_query": "TIMEOUT", "sql_reasoning": "Timeout exceeded"}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(rows, f, ensure_ascii=False)
        r = opt_report.load_template(f.name)[0]
        self.assertEqual(r["type"], "跨年逐月趨勢")
        self.assertTrue(r["timeout"])
        self.assertFalse(r["passed"])


    def test_review_accepts_only_the_declared_transformations(self):
        item = {"ground_truth": {"columns": ["category", "rank", "product", "qty"],
                                 "rows": [["A", 1, "x", 10], ["A", 2, "y", 5]]}}
        review = {"S": {"drop": ["rank"], "normalize": [], "reason": ""}}
        record = {"seed_id": "S", "workflow_status": "completed", "row_count": 2,
                  "rows": [["A", "x", 10], ["A", "y", 5]]}
        self.assertTrue(opt_report.review_passes(record, item, review))
        wrong = dict(record, rows=[["A", "x", 10], ["A", "y", 6]])
        self.assertFalse(opt_report.review_passes(wrong, item, review))
        self.assertFalse(opt_report.review_passes(dict(record, seed_id="other"), item, review))
        self.assertFalse(opt_report.review_passes(dict(record, workflow_status="failed"), item, review))

    def test_review_label_and_sign_normalization(self):
        item = {"ground_truth": {"columns": ["month", "pct"], "rows": [["01", -3.5], ["04", -1.0]]}}
        review = {"S": {"drop": [], "normalize": ["month", "abs"], "reason": ""}}
        ok = {"seed_id": "S", "workflow_status": "completed", "row_count": 2,
              "rows": [["2024-01", 3.5], ["2024-04", 1.0]]}
        self.assertTrue(opt_report.review_passes(ok, item, review))
        self.assertFalse(opt_report.review_passes(dict(ok, rows=[["2024-02", 3.5], ["2024-04", 1.0]]), item, review))

    def test_regression_reason_prefers_manual_note_and_states_the_real_limit(self):
        self.assertEqual(opt_report.timeout_reason(600.0), "逾時（超過每題 600 秒）")
        notes = {"q1": "SQL 先篩選再排名，名次永遠是 1"}
        self.assertEqual(opt_report.regression_reason("q1", "自動原因", notes), "SQL 先篩選再排名，名次永遠是 1")
        self.assertEqual(opt_report.regression_reason("q2", "自動原因", notes), "自動原因")

    def test_review_weekday_accepts_english_names_but_not_wrong_days(self):
        item = {"ground_truth": {"columns": ["weekday", "v"], "rows": [["星期一", 1.0], ["星期日", 7.0]]}}
        review = {"S": {"drop": [], "normalize": ["weekday"], "reason": ""}}
        ok = {"seed_id": "S", "workflow_status": "completed", "row_count": 2,
              "rows": [["Sunday", 7.0], ["Monday", 1.0]]}
        self.assertTrue(opt_report.review_passes(ok, item, review))
        self.assertTrue(opt_report.review_passes(dict(ok, rows=[["週一", 1.0], ["週日", 7.0]]), item, review))
        swapped = dict(ok, rows=[["Monday", 7.0], ["Sunday", 1.0]])
        self.assertFalse(opt_report.review_passes(swapped, item, review))


if __name__ == "__main__":
    unittest.main()
