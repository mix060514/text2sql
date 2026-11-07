import json
from pathlib import Path
import tempfile
import unittest

from text2sql.datagen.checks import rule_check
from text2sql.datagen.pipeline import compute_oracle, load_seeds, run

PROJECT = Path(__file__).resolve().parents[1]
SEEDS = [PROJECT / "data" / "seeds" / "golden_seeds.jsonl", PROJECT / "data" / "seeds" / "guardrail_seeds.jsonl"]


def seed(seed_id):
    return next(s for s in load_seeds(SEEDS) if s["seed_id"] == seed_id)


class RuleLayerTests(unittest.TestCase):
    def test_all_seeds_pass_rules_except_self_duplicate(self):
        for s in load_seeds(SEEDS):
            self.assertEqual(rule_check(s["question"], s, set()), ["duplicate: 與種子相同"], s["seed_id"])

    def test_simplified_chinese_rejected(self):
        reasons = rule_check("2024 年 EMEA 总销售额是多少", seed("G01"), set())
        self.assertTrue(any(r.startswith("language: 含簡體字") for r in reasons), reasons)

    def test_missing_entity_and_extra_year(self):
        reasons = rule_check("2023 年 EMEA 賣出幾件？", seed("G01"), set())
        self.assertTrue(any("年份" in r and r.startswith("keep: 缺少") for r in reasons), reasons)
        self.assertIn("keep: 出現種子沒有的年份 2023", reasons)

    def test_concept_wording_is_free(self):
        # "進帳" is not in any word list; concepts are left to the label/semantic layers.
        self.assertEqual(rule_check("2024 年歐非中東那邊進帳多少？", seed("G01"), set()), [])
        self.assertEqual(rule_check("把 2023 年拉美的單子全部清乾淨", seed("B01"), set()), [])
        self.assertTrue(rule_check("把拉美的單子全部清乾淨", seed("B01"), set()))  # lost the year

    def test_b02_must_keep_the_quarter(self):
        # 2026-11-07 run 20261107-071250: both B02 rewrites dropped 第一季 and the judge passed them.
        reasons = rule_check("預估 2025 年亞太區營收是多少？", seed("B02"), set())
        self.assertTrue(any("季度" in r for r in reasons), reasons)
        self.assertEqual(rule_check("2025 年 Q1 亞太區營收大概多少？", seed("B02"), set()), [])

    def test_g04_must_not_gain_a_year(self):
        reasons = rule_check("2024 年賣最多的前 3 名產品？", seed("G04"), set())
        self.assertIn("keep: 出現種子沒有的年份 2024", reasons)

    def test_structure_and_length(self):
        self.assertEqual(rule_check(None, seed("B04"), set()), ["structure: 題目不是非空字串"])
        self.assertTrue(any(r.startswith("length") for r in rule_check("詩" * 100, seed("B04"), set())))


class OracleTests(unittest.TestCase):
    def test_golden_answers_come_from_sql(self):
        g01 = compute_oracle(seed("G01"))
        self.assertEqual(g01["columns"], ["total_revenue"])
        self.assertEqual(g01["expected_plot"], {"should_plot": False, "chart_type": "none"})
        g03 = compute_oracle(seed("G03"))
        self.assertEqual(g03["expected_plot"]["expected_x_order"], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(g03["expected_plot"]["chart_type"], "line")
        g04 = compute_oracle(seed("G04"))
        self.assertEqual(len(g04["rows"]), 3)
        self.assertEqual([r[1] for r in g04["rows"]], sorted((r[1] for r in g04["rows"]), reverse=True))


class FakeModel:
    """Scripted model: first rewrite has a simplified-Chinese variant, then a label miss,
    then a judge that disagrees across orders; regeneration must see the failure reasons."""

    def __init__(self):
        self.calls = 0
        self.rewrite_prompts = []
        self.model = "fake"

    def __call__(self, messages, *, temperature=0.0, max_tokens=600):
        self.calls += 1
        prompt = messages[-1]["content"]
        if "改寫成" in prompt:
            self.rewrite_prompts.append(prompt)
            if len(self.rewrite_prompts) == 1:
                return {"variants": [
                    {"style": "口語", "question": "2024 年 EMEA 总共卖了多少钱"},
                    {"style": "簡短", "question": "EMEA 2024 年營收？"},
                    {"style": "情境", "question": "老闆問 2024 年歐非中東營收多少，請幫我查"},
                ]}
            return {"variants": [{"style": "客氣正式", "question": "請問 2024 年 EMEA 的銷售額合計為多少？"}]}
        if "輸入分類器" in messages[0]["content"]:
            question = prompt.split("使用者輸入：")[1].split("\n")[0]
            return {"category": "chitchat" if "老闆" in question else "sales_query", "reason": "fake"}
        first = prompt.split("問題甲：")[1].split("\n")[0]
        if "合計" in prompt and first.startswith("請問"):
            return {"same_intent": False, "changed": ["指標"], "reason": "fake order bias"}
        return {"same_intent": True, "changed": [], "reason": "fake"}


class PipelineTests(unittest.TestCase):
    def test_staged_run_with_feedback_and_report(self):
        g01 = [json.dumps(seed("G01"), ensure_ascii=False)]
        with tempfile.TemporaryDirectory() as temp:
            seeds_file = Path(temp) / "seeds.jsonl"
            seeds_file.write_text("\n".join(g01) + "\n", encoding="utf-8")
            model = FakeModel()
            run_dir = run([seeds_file], model, Path(temp) / "runs", variants=2, max_rounds=2,
                          log=lambda m: None, run_id="t")
            items = [json.loads(l) for l in (run_dir / "dataset.jsonl").read_text().splitlines()]
            report = (run_dir / "report.md").read_text()
            manifest = json.loads((run_dir / "manifest.json").read_text())
            stage_files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.glob("round*/*.jsonl"))

        # seed + one accepted rewrite; the order-inconsistent candidate is rejected
        self.assertEqual([i["id"] for i in items], ["G01-seed", "G01-v1"])
        self.assertEqual(items[1]["question"], "EMEA 2024 年營收？")
        self.assertEqual(items[1]["ground_truth"], items[0]["ground_truth"])
        self.assertIn("sql", items[1])
        # regeneration prompt carries the earlier failure reasons
        self.assertIn("含簡體字", model.rewrite_prompts[1])
        self.assertIn("分類成 chitchat", model.rewrite_prompts[1])
        self.assertIn("兩種順序判斷不一致", report)
        self.assertIn("| 1 規則 | 程式：結構、長度、繁體、實體、年份、重複 | 1 |", report)
        self.assertIn("| 2 標籤 | 本機模型盲分類，比對種子類別 | 1 |", report)
        self.assertIn("| 3 語意 | 本機模型 judge，正反兩種順序都要通過 | 1 |", report)
        self.assertEqual(manifest["llm_calls"], model.calls)
        self.assertEqual(stage_files, ["round1/1_rewrite.jsonl", "round1/2_rules.jsonl", "round1/3_label.jsonl",
                                       "round1/4_semantic.jsonl", "round2/1_rewrite.jsonl", "round2/2_rules.jsonl",
                                       "round2/3_label.jsonl", "round2/4_semantic.jsonl"])

    def test_bad_golden_seed_stops_before_any_model_call(self):
        bad = dict(seed("G01"), sql="SELECT no_such_column FROM sales_data")
        with tempfile.TemporaryDirectory() as temp:
            seeds_file = Path(temp) / "seeds.jsonl"
            seeds_file.write_text(json.dumps(bad, ensure_ascii=False) + "\n", encoding="utf-8")
            model = FakeModel()
            with self.assertRaisesRegex(RuntimeError, "G01"):
                run([seeds_file], model, Path(temp) / "runs", log=lambda m: None, run_id="bad")
        self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
