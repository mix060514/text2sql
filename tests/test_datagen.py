import copy
import json
from pathlib import Path
import tempfile
import unittest

from text2sql.datagen.checks import rule_check
from text2sql.datagen.oracle import compute_oracle, load_seeds
from text2sql.datagen.templates import build_instances, compare_answers, select_seeds

PROJECT = Path(__file__).resolve().parents[1]
SEEDS = [PROJECT / "data" / "seeds" / "golden_seeds.jsonl", PROJECT / "data" / "seeds" / "guardrail_seeds.jsonl"]

# Hand-written seed kept as a stable fixture (the checked-in golden seeds are generated).
G01 = {
    "seed_id": "G01", "kind": "golden", "label": "allow", "category": "sales_query",
    "question": "2024 年 EMEA 的總銷售額是多少？",
    "sql": "SELECT ROUND(SUM(\"Total Revenue\"), 2) AS total_revenue FROM sales_data WHERE Region = 'EMEA' "
           "AND strftime('%Y', \"Order Date\") = '2024'",
    "eval_type": "number", "expected_plot": {"should_plot": False, "chart_type": "none"}, "metric": "amount",
    "keep": [{"name": "地區", "type": "entity", "any_of": ["EMEA", "歐洲中東非洲", "歐非中東"]},
             {"name": "年份", "type": "entity", "any_of": ["2024"]},
             {"name": "指標", "type": "concept", "meaning": "銷售金額"}],
    "boundary": "只問 2024 年 EMEA 的總銷售金額。",
}
INSTANCES, PROBLEMS = build_instances()


def seed(seed_id):
    if seed_id == "G01":
        return copy.deepcopy(G01)
    return next(s for s in load_seeds(SEEDS) if s["seed_id"] == seed_id)


def instance(template):
    return copy.deepcopy(next(s for s in INSTANCES if s["template"] == template))


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

    def test_metric_kind_is_checked_by_code(self):
        # Rewrites from run 20261107-082751 that the semantic judge passed.
        cases = {
            "M_region_top_category": "每個區域銷量第一的品類各為何？",
            "E_year_best_quarter": "2023 年哪一季賣得最多？",
            "K_category_top_customer": "哪家的客戶在 Electronics 類買貨最多？",
            "N15_top5_product_quantity_bar": "賣得最好的五款產品是哪幾個？",
        }
        for template, question in cases.items():
            seed_ = next(s for s in select_seeds(INSTANCES, 1) if s["template"] == template)
            self.assertTrue(any(r.startswith("metric") for r in rule_check(question, seed_, set())), template)
        k = next(s for s in select_seeds(INSTANCES, 1) if s["template"] == "K_category_top_customer")
        self.assertEqual(rule_check("Electronics 這類產品，哪家公司買最多錢？", k, set()), [])
        d = instance("D_country_top_order_customer")
        country = d["keep"][0]["any_of"][0]
        self.assertEqual(rule_check(f"{country} 訂單數量最多的客戶？", d, set()), [])
        self.assertTrue(rule_check(f"{country} 營收最高的客戶？", d, set()))
        q11 = instance("Q11_country_quantity")
        self.assertEqual(rule_check("日本那邊總共賣了多少件？", q11, set()), [])

    def test_b02_must_keep_the_quarter(self):
        # 2026-11-07 run 20261107-071250: both B02 rewrites dropped 第一季 and the judge passed them.
        reasons = rule_check("預估 2025 年亞太區營收是多少？", seed("B02"), set())
        self.assertTrue(any("季度" in r for r in reasons), reasons)
        self.assertEqual(rule_check("2025 年 Q1 亞太區營收大概多少？", seed("B02"), set()), [])

    def test_top3_seed_must_not_gain_a_year(self):
        top3 = instance("B_region_top3_quantity")
        region = top3["keep"][0]["any_of"][0]
        self.assertEqual(rule_check(f"{region} 銷量最高的前 3 名產品？", top3, set()), [])
        reasons = rule_check(f"2024 年 {region} 銷量最高的前 3 名產品？", top3, set())
        self.assertIn("keep: 出現種子沒有的年份 2024", reasons)

    def test_structure_and_length(self):
        self.assertEqual(rule_check(None, seed("B04"), set()), ["structure: 題目不是非空字串"])
        self.assertTrue(any(r.startswith("length") for r in rule_check("詩" * 100, seed("B04"), set())))


class OracleTests(unittest.TestCase):
    def test_every_template_instance_matches_legacy_pandas(self):
        self.assertEqual(len(INSTANCES), 81)
        self.assertEqual(len({s["template"] for s in INSTANCES}), 43)
        self.assertEqual(PROBLEMS, [])

    def test_comparison_catches_wrong_reference(self):
        self.assertTrue(compare_answers("number", ["x"], [[10.0]], 10.5))
        self.assertTrue(compare_answers("string", ["x"], [["Alice"]], "Bob"))
        self.assertTrue(compare_answers("list", ["x"], [["a"], ["b"]], ["b", "a"]))
        self.assertTrue(compare_answers("dataframe", ["k", "v"], [["Q1", 1.0], ["Q2", 2.0]], {"Q1": 1.0, "Q3": 2.0}))
        self.assertTrue(compare_answers("dataframe", ["k", "v"], [["a", 1.0]], {"a": 1.0}, ["b"], ["a"]))
        self.assertEqual(compare_answers("dataframe", ["k", "v"], [["2023-Q1", 1.0]], {"2023 Q1": 1.0}), [])

    def test_pipeline_oracle_refuses_disagreement(self):
        bad = instance("A_region_revenue")
        bad["reference_answer"] = bad["reference_answer"] + 1
        with self.assertRaisesRegex(ValueError, "SQL and pandas disagree"):
            compute_oracle(bad)

    def test_golden_answers_and_chart_order(self):
        g01 = compute_oracle(seed("G01"))
        self.assertEqual(g01["columns"], ["total_revenue"])
        self.assertEqual(g01["oracle"], "sql")
        quarters = compute_oracle(instance("N12_year_quarter_revenue_bar"))
        self.assertEqual(quarters["expected_plot"]["expected_x_order"], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(quarters["oracle"], "sql+pandas")
        trend = compute_oracle(instance("O4_quarterly_trend"))
        self.assertEqual(trend["expected_plot"]["expected_x_order"][:2], ["2023-Q1", "2023-Q2"])
        self.assertEqual(trend["expected_plot"]["chart_type"], "line")


class GuardrailSeedTests(unittest.TestCase):
    def test_block_side_matches_golden_breadth(self):
        from collections import Counter
        from text2sql.datagen.guardrail import build_guardrail_seeds
        built = build_guardrail_seeds()
        self.assertEqual(len(built), 42)
        self.assertEqual(Counter(s["category"] for s in built),
                         {"write_request": 10, "not_in_data": 16, "chitchat": 8, "prompt_injection": 8})
        self.assertTrue(all(s["label"] == "block" for s in built))
        checked_in = [s for s in load_seeds(SEEDS) if s["kind"] == "guardrail"]
        self.assertEqual(checked_in, built)

    def test_alias_extras_accept_common_spellings(self):
        b23 = seed("B23")
        self.assertNotIn("Amy", b23["question"])  # a personal name read as a customer (run 20261107-090845)
        b02 = seed("B02")
        self.assertEqual(rule_check("2025 首季亞太營收預估多少？", b02, set()), [])


class SelectionTests(unittest.TestCase):
    def test_one_seed_per_template_without_alias_copies(self):
        chosen = select_seeds(INSTANCES, 1)
        templates = [s["template"] for s in chosen]
        self.assertEqual(len(templates), len(set(templates)))
        self.assertEqual(len(chosen), 42)  # 43 templates minus the excluded I template
        self.assertNotIn("I_na_top5_customer_category_share", templates)
        self.assertEqual(len({s["question"] for s in chosen}), len(chosen))
        entities = {g["any_of"][0] for s in chosen for g in s["keep"] if g["type"] == "entity"}
        self.assertTrue({"EMEA", "APAC", "LATAM", "North America"} <= entities)
        self.assertTrue({"Electronics", "Office Supplies", "Software"} <= entities)

    def test_checked_in_seeds_are_the_generated_selection(self):
        golden = [s for s in load_seeds(SEEDS) if s["kind"] == "golden"]
        self.assertEqual([s["seed_id"] for s in golden], [s["seed_id"] for s in select_seeds(INSTANCES, 1)])
        self.assertTrue(all(s["oracle"] == "sql+pandas" for s in golden))


if __name__ == "__main__":
    unittest.main()
