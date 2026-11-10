#!/usr/bin/env python3
"""Build the exam: one set of under 100 questions, each asked once, each testing a different point.

uv run python scripts/build_exam.py

The exam replaces the earlier question sets (16 template, 27 alias, 354 synthetic), where every
complex and refusal question was asked three times in different words. Here a question appears
once; about half use a colloquial wording taken from the accepted model rewrites
(data/synthetic/runs/<RUN>/dataset.jsonl, same id), the rest the original sentence.

PICKS is the blueprint: (group, point tested, source question id, wording). Add a row to test
something new. wording 0 is the original sentence, 1 or 2 one of its two accepted rewrites.
Reference answers are never written by hand: the question's SQL is run on the database now.

Written as data/exam/dataset.jsonl + manifest.json in the format eval/synthetic.load_run reads;
run with `eval/eval_e2e.py --exam`. No model is used to build it.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from text2sql.agents.planned_agent import prompts, tools  # noqa: E402
from text2sql.datagen.pipeline import compute_oracle  # noqa: E402

SEED_FILES = [PROJECT / "data" / "seeds" / name for name in
              ("golden_seeds.jsonl", "complex_seeds.jsonl", "guardrail_seeds.jsonl")]
REWRITES = PROJECT / "data" / "synthetic" / "runs" / "20261107-162208" / "dataset.jsonl"
EXAM = PROJECT / "data" / "exam"
GROUPS = ("基本查詢", "畫圖", "多步驟查詢", "不同叫法", "該拒答")

PICKS = [
    # 基本查詢：一條查詢就能回答
    ("基本查詢", "篩選地區後加總", "A-EMEA", 1),
    ("基本查詢", "依年份加總", "Q1-2024", 0),
    ("基本查詢", "訂單數", "Q10-UnitedStates", 1),
    ("基本查詢", "銷售量", "Q11-Japan", 0),
    ("基本查詢", "單一產品的平均單價", "Q9-EnterpriseLaptopX1", 1),
    ("基本查詢", "篩選後取銷售量前三名", "B-NorthAmerica", 1),
    ("基本查詢", "依訂單數取第一名", "D-Canada", 0),
    ("基本查詢", "平均單價＝總額÷總量，取最高", "L", 1),
    ("基本查詢", "不篩選，取銷售額最高", "Q8", 0),
    ("基本查詢", "依季度分組取最高", "E-2023", 2),
    ("基本查詢", "每個區域各取第一名", "M", 2),
    ("基本查詢", "兩年變動的絕對值最大", "J", 0),
    ("基本查詢", "年增率", "F-OfficeSupplies", 2),
    ("基本查詢", "同一季跨年相減", "H", 1),
    # 畫圖：除了數字，還要畫對圖
    ("畫圖", "長條圖：各年份", "C-Electronics", 2),
    ("畫圖", "長條圖：各類別訂單數", "N4", 0),
    ("畫圖", "長條圖：區域內各國家", "N5-LATAM", 1),
    ("畫圖", "長條圖：一年內各季", "N12-2023", 0),
    ("畫圖", "長條圖：前五名，由高到低", "N14", 1),
    ("畫圖", "折線圖：逐月，標籤 YYYY-MM", "O1-OfficeSupplies", 1),
    ("畫圖", "折線圖：跨年逐季，標籤 YYYY-Qn", "O4", 0),
    ("畫圖", "圓餅圖：銷售額占比", "P1", 1),
    ("畫圖", "圓餅圖：訂單數占比", "P3", 0),
    ("畫圖", "圓餅圖：先篩選年份再算占比", "P4-2023", 0),
    # 多步驟查詢：要分幾步算（子查詢、視窗函數、跨年比較）
    ("多步驟查詢", "每個區域各取第一名並附金額", "X-region_top_product-2024", 0),
    ("多步驟查詢", "每個類別各取前三名", "X-category_top3_quantity", 1),
    ("多步驟查詢", "月增率", "X-region_mom-APAC-2024", 2),
    ("多步驟查詢", "逐月累計＋折線圖", "X-cumulative-2023", 2),
    ("多步驟查詢", "三個類別都買過的客戶數", "X-all_categories-2024", 1),
    ("多步驟查詢", "兩年都下單的客戶數", "X-retained", 0),
    ("多步驟查詢", "當年才第一次下單的客戶", "X-new_customers-2024", 1),
    ("多步驟查詢", "平均每筆訂單金額，由高到低", "X-aov_region", 0),
    ("多步驟查詢", "類別內各區域占比", "X-region_share_of_category-2024-Software", 1),
    ("多步驟查詢", "各區域年增率＋長條圖", "X-region_yoy", 2),
    ("多步驟查詢", "前十大客戶占全部的比例", "X-top_customer_concentration-10", 0),
    ("多步驟查詢", "大訂單占銷售額的比例", "X-large_order_share-10", 2),
    ("多步驟查詢", "每一年各自最高的月份", "X-best_month_each_year", 2),
    ("多步驟查詢", "平均單價漲幅前三名", "X-asp_change-3", 0),
    ("多步驟查詢", "累計到 80% 需要幾個產品", "X-pareto-80", 0),
    ("多步驟查詢", "在四個區域都下過單的客戶數", "X-multi_region_customers-4", 1),
    ("多步驟查詢", "區域 × 類別交叉表", "X-region_category_matrix-2023", 0),
    ("多步驟查詢", "每一季與去年同季比較", "X-category_quarter_yoy-Electronics", 0),
    ("多步驟查詢", "前五大客戶與各自的訂單數", "X-region_top_customers_orders-EMEA-2024-5", 2),
    ("多步驟查詢", "平均每筆訂單的件數", "X-avg_qty_per_order", 1),
    ("多步驟查詢", "高於平均值的國家", "X-countries_above_avg", 0),
    ("多步驟查詢", "下半年對上半年的成長率", "X-h2_vs_h1-2024", 1),
    ("多步驟查詢", "在同區域國家中的名次", "X-country_rank_in_region-Japan", 2),
    ("多步驟查詢", "第二名", "X-nth_customer-2023-2", 1),
    ("多步驟查詢", "超過門檻的訂單，分組計數", "X-high_price_orders-500", 2),
    ("多步驟查詢", "平均每位客戶的銷售額", "X-revenue_per_customer_region", 1),
    ("多步驟查詢", "區域內每個國家各取第一名", "X-country_top_quantity_product-LATAM", 0),
    ("多步驟查詢", "增加金額前三名", "X-product_growth_amount-3", 1),
    ("多步驟查詢", "單一客戶的類別占比", "X-customer_category_mix-GlobalSummitSolutions", 2),
    ("多步驟查詢", "季增率最高的一季", "X-quarter_max_qoq", 0),
    # 該拒答：不該去查資料的輸入
    ("該拒答", "要求改資料：刪除", "B01", 1),
    ("該拒答", "要求改資料：修改數值", "B06", 0),
    ("該拒答", "要求改資料：新增一筆", "B07", 1),
    ("該拒答", "要求改資料：清空資料表", "B09", 0),
    ("該拒答", "要求改資料：建立新資料表", "B10", 2),
    ("該拒答", "資料庫沒有：預測未來", "B02", 0),
    ("該拒答", "資料庫沒有：毛利", "B03", 1),
    ("該拒答", "資料庫沒有：庫存", "B21", 0),
    ("該拒答", "資料庫沒有：客戶聯絡方式", "B22", 1),
    ("該拒答", "資料庫沒有：競爭對手", "B26", 0),
    ("該拒答", "閒聊：寫詩", "B04", 0),
    ("該拒答", "閒聊：打招呼", "B29", 1),
    ("該拒答", "閒聊：翻譯", "B31", 0),
    ("該拒答", "閒聊：天氣", "B32", 2),
    ("該拒答", "閒聊：寫程式", "B33", 1),
    ("該拒答", "提示注入：要求列出系統提示詞", "B05", 1),
    ("該拒答", "提示注入：要求讀系統資料表", "B36", 0),
    ("該拒答", "提示注入：要求改變角色", "B37", 1),
    ("該拒答", "提示注入：假冒管理員", "B39", 2),
    ("該拒答", "提示注入：要求金鑰與連線設定", "B41", 0),
]

# The reference answer keeps only what the question asks for; these SQL results carry a
# supporting column the question does not ask for (docs/optimization/complex_review.json).
DROP_COLUMNS = {
    "X-region_top_customers_orders-EMEA-2024-5": ["total_revenue"],
    "X-countries_above_avg": ["total_revenue"],
    "X-best_month_each_year": ["year"],
}

# 不同叫法：the name in the question is not the value stored in the database.
# (alias, sentence kind); the alias must be in the question-answering prompt's glossary.
ALIAS_PICKS = [("歐非中東", "revenue"), ("亞太", "orders"), ("拉美", "quantity"), ("北美", "revenue"),
               ("大陸", "orders"), ("US", "quantity"), ("UK", "revenue"), ("德國", "orders"),
               ("澳洲", "quantity"), ("巴西", "revenue")]
ALIAS_KINDS = {
    "revenue": ("{name} 2024 年的總銷售額是多少？",
                'SELECT ROUND(SUM("Total Revenue"), 2) AS total_revenue FROM sales_data '
                "WHERE {column} = '{value}' AND strftime('%Y', \"Order Date\") = '2024'"),
    "orders": ("{name} 總共有多少筆訂單？",
               'SELECT COUNT("Order ID") AS order_count FROM sales_data WHERE {column} = \'{value}\''),
    "quantity": ("{name} 2023 年總共賣出多少件？",
                 "SELECT SUM(Quantity) AS total_quantity FROM sales_data "
                 "WHERE {column} = '{value}' AND strftime('%Y', \"Order Date\") = '2023'"),
}


def glossary(domain=prompts.DOMAIN):
    """[(column, alias, canonical)] from the '地區別名：' and '國家別名：' lines of the prompt."""
    out = []
    for label, column in (("地區別名", "Region"), ("國家別名", "Country")):
        match = re.search(label + r"：(.*?)(?:\n\s*\n|\n[^\n]*別名：|$)", domain, re.S)
        if not match:
            raise ValueError(f"{label} not found in prompts.DOMAIN")
        for entry in re.split(r"[；。\n]", match.group(1)):
            if "=" in entry:
                names, canonical = entry.strip().split("=", 1)
                out += [(column, name.strip(), canonical.strip()) for name in names.split("/") if name.strip()]
    return out


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def picked_items():
    seeds = {s["seed_id"]: s for path in SEED_FILES for s in load_jsonl(path)}
    wordings = {}
    for row in load_jsonl(REWRITES):
        wordings.setdefault(row["seed_id"], {})[row["id"]] = row
    items = []
    for group, point, seed_id, wording in PICKS:
        seed = seeds[seed_id]
        row = wordings[seed_id][f"{seed_id}-seed" if wording == 0 else f"{seed_id}-v{wording}"]
        if wording == 0 and row["question"] != seed["question"]:
            raise ValueError(f"{seed_id}: the accepted original sentence differs from the seed file")
        item = {"id": row["id"], "question": row["question"], "label": seed["label"], "subset": "exam",
                "group": group, "point": point, "wording": "原句" if wording == 0 else "口語",
                "category": seed["category"], "source": row["source"], "seed_id": seed_id}
        if seed["label"] == "allow":
            oracle = compute_oracle(seed)
            columns, rows = oracle["columns"], oracle["rows"]
            for name in DROP_COLUMNS.get(seed_id, []):
                j = columns.index(name)
                columns = columns[:j] + columns[j + 1:]
                rows = [r[:j] + r[j + 1:] for r in rows]
            item.update({"difficulty": seed.get("difficulty"), "sql": seed["sql"], "eval_type": seed["eval_type"],
                         "ground_truth": {"columns": columns, "rows": rows},
                         "expected_plot": oracle["expected_plot"]})
        else:
            item["expected_behavior"] = seed["expected_behavior"]
        items.append(item)
    return items


def alias_items():
    known = {name: (column, canonical) for column, name, canonical in glossary()}
    items = []
    for name, kind in ALIAS_PICKS:
        column, canonical = known[name]
        sentence, sql = ALIAS_KINDS[kind]
        sql = sql.format(column=column, value=canonical)
        result = tools.read_sales(sql)
        if not result["rows"] or result["rows"][0][0] is None:
            raise ValueError(f"{name}={canonical}: no rows for {kind}")
        items.append({"id": f"alias-{column}-{name}" + ("" if kind == "revenue" else f"-{kind}"),
                      "question": sentence.format(name=name),
                      "label": "allow", "subset": "exam", "group": "不同叫法",
                      "point": f"{name} → {canonical}", "wording": "原句", "category": "sales_query",
                      "source": "seed", "seed_id": f"alias-{canonical}", "alias": name, "canonical": canonical,
                      "column": column, "difficulty": "simple", "sql": sql, "eval_type": "number",
                      "ground_truth": {"columns": result["columns"], "rows": result["rows"]},
                      "expected_plot": None})
    return items


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    items = picked_items() + alias_items()
    items.sort(key=lambda item: GROUPS.index(item["group"]))  # stable: blueprint order within a group
    for field in ("id", "question"):
        repeated = [value for value, n in Counter(item[field] for item in items).items() if n > 1]
        if repeated:
            raise ValueError(f"repeated {field}: {repeated}")
    if len(items) > 100:
        raise ValueError(f"the exam has {len(items)} questions; keep it at 100 or fewer")
    return items


def main():
    items = build()
    EXAM.mkdir(parents=True, exist_ok=True)
    dataset = EXAM / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    manifest = {"status": "complete", "count": len(items), "groups": dict(Counter(i["group"] for i in items)),
                "dataset_sha256": sha256(dataset), "database_sha256": sha256(tools.DB_PATH),
                "built_at": datetime.now().isoformat(timespec="seconds"), "builder": "scripts/build_exam.py"}
    (EXAM / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"exam: {len(items)} questions -> {dataset.relative_to(PROJECT)}  {manifest['groups']}")
    for item in items:
        print(f"  {item['group']:<6} {item['wording']} {item['point']:<28} {item['question']}")


if __name__ == "__main__":
    main()
