#!/usr/bin/env python3
"""Build the two small, program-graded question banks that replace eval_set_v3 (132 questions).

uv run python scripts/build_eval_banks.py

- template: 16 golden seeds picked by hand, one ability each (filter+sum, order count, ASP,
  Top-N, YoY, same-quarter change, per-group max, bar/line/pie). Question, SQL and expected
  chart come from data/seeds/golden_seeds.jsonl; the oracle is the seed SQL run now.
- alias: one question per alias in the M4 prompt's glossary (prompts.DOMAIN), so a wrong
  answer points at one name. The sentence and the SQL are fixed; only the name changes.

Each bank is written as data/banks/<name>/dataset.jsonl + manifest.json in the format
eval/synthetic.load_run reads, and run with `eval/eval_e2e.py --bank <name>`. No model is used.
"""
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from text2sql.agents.planned_agent import prompts, tools  # noqa: E402
from text2sql.datagen.pipeline import compute_oracle  # noqa: E402

SEEDS = PROJECT / "data" / "seeds" / "golden_seeds.jsonl"
BANKS = PROJECT / "data" / "banks"

# One seed per ability; the last three failed in the 2026-11-09 eval_set_v3 run.
TEMPLATE_PICKS = [
    ("A-EMEA", "篩選＋加總"),
    ("Q10-UnitedStates", "訂單數（不是列數）"),
    ("L", "平均單價＝總額÷總量"),
    ("B-NorthAmerica", "篩選＋Top-N＋數量"),
    ("D-Canada", "依訂單數取第一"),
    ("E-2023", "季度分組取最大"),
    ("F-OfficeSupplies", "年增率"),
    ("H", "同季比較"),
    ("J", "絕對值變動最大"),
    ("M", "組內取最大"),
    ("N5-LATAM", "區域內細分＋長條圖"),
    ("N14", "Top-N 排序＋長條圖"),
    ("O1-OfficeSupplies", "月份標籤 YYYY-MM＋折線圖"),
    ("O4", "跨年季度標籤 YYYY-Qn＋折線圖"),
    ("P1", "占比＋圓餅圖"),
    ("P3", "訂單數占比＋圓餅圖"),
]

ALIAS_QUESTION = "{name} 2024 年的總銷售額是多少？"
ALIAS_SQL = ('SELECT ROUND(SUM("Total Revenue"), 2) AS total_revenue FROM sales_data '
             "WHERE {column} = '{value}' AND strftime('%Y', \"Order Date\") = '2024'")


def glossary(domain=prompts.DOMAIN):
    """[(column, alias, canonical)] from the '地區別名：' and '國家別名：' lines of the M4 prompt."""
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


def template_items():
    seeds = {s["seed_id"]: s for s in map(json.loads, SEEDS.read_text(encoding="utf-8").splitlines()) if s}
    items = []
    for seed_id, ability in TEMPLATE_PICKS:
        seed = seeds[seed_id]
        oracle = compute_oracle(seed)
        items.append({"id": seed_id, "question": seed["question"], "label": "allow", "subset": "template",
                      "category": seed["category"], "source": "seed", "seed_id": seed_id, "ability": ability,
                      "template": seed["template"], "difficulty": seed.get("difficulty"),
                      "sql": seed["sql"], "eval_type": seed["eval_type"],
                      "ground_truth": {"columns": oracle["columns"], "rows": oracle["rows"]},
                      "expected_plot": oracle["expected_plot"]})
    return items


def alias_items():
    items = []
    for column, name, canonical in glossary():
        sql = ALIAS_SQL.format(column=column, value=canonical)
        result = tools.read_sales(sql)
        if not result["rows"] or result["rows"][0][0] is None:
            raise ValueError(f"{name}={canonical}: no 2024 sales in {column}")
        items.append({"id": f"alias-{column}-{name}", "question": ALIAS_QUESTION.format(name=name),
                      "label": "allow", "subset": "alias", "category": "sales_query", "source": "seed",
                      "seed_id": f"alias-{canonical}", "alias": name, "canonical": canonical, "column": column,
                      "difficulty": "simple", "sql": sql, "eval_type": "number",
                      "ground_truth": {"columns": result["columns"], "rows": result["rows"]},
                      "expected_plot": None})
    return items


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bank(name, items):
    folder = BANKS / name
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    manifest = {"status": "complete", "bank": name, "count": len(items), "dataset_sha256": sha256(dataset),
                "database_sha256": sha256(tools.DB_PATH), "built_at": datetime.now().isoformat(timespec="seconds"),
                "builder": "scripts/build_eval_banks.py"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dataset


def main():
    for name, items in (("template", template_items()), ("alias", alias_items())):
        path = write_bank(name, items)
        print(f"{name}: {len(items)} items -> {path.relative_to(PROJECT)}")
        for item in items:
            plot = (item["expected_plot"] or {}).get("chart_type") if (item["expected_plot"] or {}).get("should_plot") else "-"
            print(f"  {item['id']:<28} {plot:<5} {item['question']}")


if __name__ == "__main__":
    main()
