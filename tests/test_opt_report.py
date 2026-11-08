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


if __name__ == "__main__":
    unittest.main()
