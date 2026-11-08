#!/usr/bin/env python3
"""Build docs/optimization/report.json for the 優化成果 page from evaluation evidence files.

Every number on the page is computed here from result files copied into docs/optimization/
(eval/results is git-ignored). Missing evidence leaves that part empty instead of guessing.

    uv run python scripts/build_optimization_report.py \
        [--template-baseline eval/results/e2e_<ts>] [--complex-baseline eval/results/synthetic_complex_<ts>] \
        [--template-fewshot <run>] [--complex-fewshot <run>]
"""

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import opt_report as o  # noqa: E402

OUT = ROOT / "docs" / "optimization"
ORIGINAL_RUN = ROOT / "eval" / "results" / "e2e_2026-11-07_17-23-39"


def copy(src: Path, name: str) -> Path | None:
    """Copy an evidence file into docs/optimization (tracked) and return the copy."""
    if src and src.exists():
        dst = OUT / name
        shutil.copyfile(src, dst)
        return dst
    existing = OUT / name
    return existing if existing.exists() else None


def load_jsonl_runs(path: Path) -> list[dict]:
    """Rows written by the one-question-at-a-time verification (scratchpad one.py)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({"question": r["q"], "passed": bool(r["e2e"]), "timeout": "Timeout" in r.get("why", ""),
                         "seconds": r.get("sec"), "attempts": None})
    return rows


def pick(rows, questions):
    return [r for r in rows if r["question"] in questions]


def cte_timing() -> dict:
    """Measure the LIMIT-CTE join with and without MATERIALIZED on the real database."""
    db = ROOT / "data" / "global_sales_data.sqlite"
    cte = ('top_customers AS {m}(SELECT "Customer Name", SUM("Total Revenue") AS customer_total FROM sales_data '
           "WHERE \"Region\"='EMEA' AND substr(\"Order Date\",1,4)='2024' GROUP BY \"Customer Name\" "
           'ORDER BY customer_total DESC, "Customer Name" LIMIT 3)')
    body = ('SELECT s."Customer Name", s."Product Category", SUM(s."Total Revenue")*100.0/t.customer_total '
            'FROM sales_data AS s JOIN top_customers AS t ON s."Customer Name"=t."Customer Name" '
            "WHERE s.\"Region\"='EMEA' AND substr(s.\"Order Date\",1,4)='2024' "
            'GROUP BY s."Customer Name", s."Product Category", t.customer_total')
    out, cap = {}, 20.0
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for name, m in (("plain", ""), ("materialized", "MATERIALIZED ")):
            start = time.monotonic()
            conn.set_progress_handler(lambda: int(time.monotonic() - start > cap), 1000)
            try:
                conn.execute(f"WITH {cte.format(m=m)} {body}").fetchall()
                out[name] = {"seconds": round(time.monotonic() - start, 3), "finished": True}
            except sqlite3.OperationalError:
                out[name] = {"seconds": cap, "finished": False}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-baseline", type=Path)
    ap.add_argument("--complex-baseline", type=Path)
    ap.add_argument("--template-fewshot", type=Path)
    ap.add_argument("--complex-fewshot", type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    original = copy(ORIGINAL_RUN / "e2e_results.before_rejudge.json", "template_2026-11-07_original.json")
    rejudged = copy(ORIGINAL_RUN / "e2e_results.json", "template_2026-11-07_rejudged.json")
    t_base = copy(args.template_baseline and args.template_baseline / "e2e_results.json", "template_baseline.json")
    c_base = copy(args.complex_baseline and args.complex_baseline / "synthetic_results.json", "complex_baseline.json")
    t_few = copy(args.template_fewshot and args.template_fewshot / "e2e_results.json", "template_fewshot.json")
    c_few = copy(args.complex_fewshot and args.complex_fewshot / "synthetic_results.json", "complex_fewshot.json")
    streaming = OUT / "timeout_questions_after_streaming.jsonl"

    orig_rows = o.load_template(original)
    timeout_qs = {r["question"] for r in orig_rows if r["timeout"]}
    judge_missing = [r for r in json.loads(original.read_text(encoding="utf-8"))
                     if str(r.get("sql_reasoning", "")).startswith("Judge Agent did not return result")]
    rejudged_map = {r["question"]: r for r in json.loads(rejudged.read_text(encoding="utf-8"))}
    t_base_rows = o.load_template(t_base) if t_base else []
    t_few_rows = o.load_template(t_few) if t_few else []
    review_path = OUT / "complex_review.json"
    review = {k: v for k, v in json.loads(review_path.read_text(encoding="utf-8")).items()
              if not k.startswith("_")} if review_path.exists() else {}
    c_base_rows = o.load_synthetic(c_base, "complex", review) if c_base else []
    c_few_rows = o.load_synthetic(c_few, "complex", review) if c_few else []
    reviewed = lambda rows: [dict(r, passed=r["reviewed"]) for r in rows]

    def compare(before, after):
        return {"before": o.stats(before) if before else None, "after": o.stats(after) if after else None}

    q = lambda *texts: {r["question"] for r in orig_rows if any(t in r["question"] for t in texts)}
    cases = [
        {
            "id": "streaming",
            "title": "模型請求逾時連鎖失敗",
            "problem": "兩年逐月趨勢、前五大客戶等 18 題在 120 秒內等不到模型回覆，全部失敗；"
                       "失敗還會一題接一題連續發生。",
            "cause": "模型思考沒有上限，常想到 4,096 tokens 才停；用一般（非串流）請求時，程式放棄後 llama.cpp "
                     "仍把那個沒人要的回答生完，佔住唯一的推論槽，後面排隊的請求跟著逾時。",
            "fix": "寫 SQL 與審查加上思考上限（2,048／1,024 tokens），寫報告關閉思考；"
                   "請求改用串流並設總時限，逾時就關閉連線，llama.cpp 立即釋放推論槽。",
            "metrics": compare(pick(orig_rows, timeout_qs),
                               load_jsonl_runs(streaming) if streaming.exists() else []),
            "scope": "2026-11-07 逾時的 18 題，修正後逐題重跑",
        },
        {
            "id": "judge",
            "title": "LLM 評審連鎖失敗",
            "problem": f"{len(judge_missing)} 題的回答其實正確，卻因評審沒有回傳結果被記為 0 分。",
            "cause": "評審用的是沒有時限、不串流的模型呼叫，一次逾時就留下幽靈請求，"
                     "後面的評審請求連續失敗（兩串，各 8 題與 12 題）。",
            "fix": "評審改用同一套有思考上限的串流模型，並新增 --rejudge 只重評失敗題、保留原答案。",
            "metrics": compare(
                [{"passed": False, "timeout": False, "seconds": None, "attempts": None} for _ in judge_missing],
                [{"passed": bool(rejudged_map[r["question"]]["sql_correct"]), "timeout": False, "seconds": None,
                  "attempts": None} for r in judge_missing]),
            "scope": "評審沒有回傳結果的題目",
            "labels": ["成功率"],
        },
        {
            "id": "materialized",
            "title": "前 N 名 CTE 查詢被 5 秒上限擋下",
            "problem": "「前五大客戶各類別占比」第一次寫出的 SQL 常被資料庫 5 秒上限中斷，只能整個重寫，多花 1 到 2 分鐘。",
            "cause": "帶 LIMIT 的 CTE 再 JOIN 回明細時，SQLite 以 co-routine 執行，明細每一列都重算一次前 N 名。",
            "fix": "在 few-shot 範例示範 WITH ... AS MATERIALIZED，前 N 名只算一次。",
            "timing": cte_timing(),
            "scope": "同一句查詢在正式資料庫上實測",
        },
        {
            "id": "quarter",
            "title": "月份被當成季度",
            "problem": "「2023 年各季度的銷售額」把 1、2、3 月當成 Q1、Q2、Q3，其餘月份全算 Q4；審查也沒抓到。",
            "cause": "提示只規定季度標籤格式，沒有寫月份怎麼對應季度。",
            "fix": "提示寫明季度公式 (月份 + 2) / 3，並有測試用標準答案驗證公式。",
            "metrics": compare(pick(orig_rows, q("2023 年各季度的銷售額")),
                               pick(t_base_rows, q("2023 年各季度的銷售額"))),
            "scope": "Q94",
        },
        {
            "id": "grading",
            "title": "評分規則誤判正確答案",
            "problem": "跨年季度趨勢 5 題與指定圓餅圖 1 題，答案正確卻被判錯。",
            "cause": "X 軸 2023-Q1（專案規範）與題庫的 2023 Q1 只差分隔符號就判順序錯；"
                     "評審看不到圖，因此以「沒附圓餅圖」扣分。",
            "fix": "比對 X 軸順序時忽略分隔符號（順序仍嚴格比對）；評審提示註明圖表另由程式檢查。",
            "metrics": compare(pick(orig_rows, q("各季度的總銷售額趨勢", "區域 2023 至 2024 年各季度", "請用圓餅圖表示")),
                               pick(t_base_rows, q("各季度的總銷售額趨勢", "區域 2023 至 2024 年各季度", "請用圓餅圖表示"))),
            "scope": "Q101–Q105、Q110",
        },
        {
            "id": "complex_grading",
            "title": "複雜題標準答案過嚴",
            "problem": "複雜題用 SQLite 標準答案逐列比對，有些答案內容正確，卻因格式不同被判錯。",
            "cause": "標準答案多帶題目沒問的欄位（名次、金額），標籤寫法不同（週一／星期一、01／2024-01），"
                     "「下滑多少百分比」的正負號表示法不同。",
            "fix": "逐題人工複核，並寫下該題允許的轉換（省略哪些欄、怎麼統一標籤）；程式只套用這些轉換重新比對，"
                   "結果仍須完全一致才算對，數值錯的題目不會因複核變成正確。",
            "metrics": compare(c_base_rows, reviewed(c_base_rows)) if c_base_rows else None,
            "scope": "複雜題基準 102 題（嚴格比對 → 複核後）",
            "labels": ["成功率"],
        },
    ]

    hard_template = o.hard_types(t_base_rows) if t_base_rows else []
    # Hard complex types are chosen on reviewed correctness: a grader artefact is not a
    # reason to teach the model anything.
    hard_complex = o.hard_types(reviewed(c_base_rows)) if c_base_rows else []
    few_shot = {
        "template": {t: compare([r for r in t_base_rows if r["type"] == t], [r for r in t_few_rows if r["type"] == t])
                     for t in hard_template},
        "complex": {t: compare([r for r in reviewed(c_base_rows) if r["type"] == t],
                               [r for r in reviewed(c_few_rows) if r["type"] == t])
                    for t in hard_complex},
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall": {
            "template": {"2026-11-07": o.stats(orig_rows), "baseline": o.stats(t_base_rows) if t_base_rows else None,
                         "few_shot": o.stats(t_few_rows) if t_few_rows else None},
            "complex": {"baseline": o.stats(c_base_rows) if c_base_rows else None,
                        "baseline_reviewed": o.stats(reviewed(c_base_rows)) if c_base_rows else None,
                        "few_shot": o.stats(c_few_rows) if c_few_rows else None,
                        "few_shot_reviewed": o.stats(reviewed(c_few_rows)) if c_few_rows else None},
        },
        "by_type": {
            "template": {"2026-11-07": o.by_type(orig_rows), "baseline": o.by_type(t_base_rows),
                         "few_shot": o.by_type(t_few_rows)},
            "complex": {"baseline": o.by_type(c_base_rows), "few_shot": o.by_type(c_few_rows)},
        },
        "concurrency": [{k: v for k, v in row.items() if k != "requests_detail"}
                        for row in json.loads((OUT / "concurrency.json").read_text(encoding="utf-8"))]
                       if (OUT / "concurrency.json").exists() else [],
        "cases": cases,
        "complex_review": [
            {"seed_id": r["seed_id"], "question": r["question"], "reason": review[r["seed_id"]]["reason"],
             "accepted": r["reviewed"]}
            for r in c_base_rows if not r["passed"] and r["seed_id"] in review
        ],
        "few_shot": few_shot,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'report.json'}: {len(cases)} cases, hard template types {hard_template}, "
          f"hard complex types {len(hard_complex)}")


if __name__ == "__main__":
    main()
