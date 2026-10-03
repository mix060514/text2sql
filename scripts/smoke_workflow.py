#!/usr/bin/env python3
"""Live, database-grounded checks shared by both branches; no LLM judge."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from google.adk.runners import InMemoryRunner
from text2sql.agents.planned_agent.agent import root_agent
from text2sql.agents.planned_agent.tools import DB_PATH, read_sales
from text2sql.config import ANALYSIS_MODEL, WORKFLOW_VARIANT
from text2sql.runtime import run_question

CASES = [
    ("LATAM revenue", "南美 的總銷售額是多少？", "SELECT SUM(\"Total Revenue\") AS Revenue FROM sales_data WHERE Region = 'LATAM'", "none"),
    ("EMEA quantity ranking", "歐非中東 銷售量最高的前三個產品是什麼？", 'SELECT "Product Name", SUM(Quantity) AS Units FROM sales_data WHERE Region = \'EMEA\' GROUP BY "Product Name" ORDER BY Units DESC, "Product Name" ASC LIMIT 3', "bar"),
    ("Order count", "Canada 訂單數量最多的客戶是誰？", 'SELECT "Customer Name", COUNT(DISTINCT "Order ID") AS Orders FROM sales_data WHERE Country = \'Canada\' GROUP BY "Customer Name" ORDER BY Orders DESC, "Customer Name" ASC LIMIT 1', "none"),
    ("Year comparison", "請列出 Electronics 類別在各年份的銷售額總計。", 'SELECT strftime(\'%Y\', "Order Date") AS Year, SUM("Total Revenue") AS Revenue FROM sales_data WHERE "Product Category" = \'Electronics\' GROUP BY Year ORDER BY Year', "bar"),
    ("Single-year quarter labels", "請給我 2023 年每季的銷售額趨勢。", 'SELECT \'Q\' || CAST((CAST(strftime(\'%m\', "Order Date") AS INTEGER)+2)/3 AS INTEGER) AS Quarter, SUM("Total Revenue") AS Revenue FROM sales_data WHERE strftime(\'%Y\', "Order Date") = \'2023\' GROUP BY Quarter ORDER BY Quarter', "line"),
    ("Monthly trend", "請列出 North America 區域在 2023 至 2024 年每個月的銷售趨勢數據。", 'SELECT strftime(\'%Y-%m\', "Order Date") AS Period, SUM("Total Revenue") AS Revenue FROM sales_data WHERE Region = \'North America\' AND strftime(\'%Y\', "Order Date") IN (\'2023\',\'2024\') GROUP BY Period ORDER BY Period', "line"),
    ("Region proportions", "各區域的銷售額佔比是多少？", 'SELECT Region, SUM("Total Revenue") AS Revenue FROM sales_data GROUP BY Region ORDER BY Region', "pie"),
    ("Empty result", "請列出 Country 是 Atlantis 的所有不同客戶名稱。", 'SELECT DISTINCT "Customer Name" FROM sales_data WHERE Country = \'Atlantis\' ORDER BY "Customer Name"', "none"),
]


def cell_key(value):
    try:
        return ("number", round(float(value), 5))
    except (TypeError, ValueError):
        return ("text", str(value))


def normalized_rows(rows):
    # Column aliases/order may differ. Compare the actual values in every row.
    return sorted(tuple(sorted(cell_key(cell) for cell in row)) for row in rows)


def pie_matches(state, expected):
    columns = state.get("column_names", [])
    spec = state.get("plot_decision", {})
    try:
        x, y = columns.index(spec["x_column"]), columns.index(spec["y_column"])
        actual = {row[x]: float(row[y]) for row in state["raw_query_result"]}
        truth = dict(expected)
        return actual.keys() == truth.keys() and all(
            math.isclose(actual[key] / sum(actual.values()), truth[key] / sum(truth.values()), abs_tol=0.0001)
            for key in truth
        )
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return False


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runner = InMemoryRunner(agent=root_agent, app_name="smoke_workflow")
    results = []
    cases = CASES[:args.limit] if args.limit else CASES
    for name, question, oracle_sql, expected_chart in cases:
        started = time.monotonic()
        state = await run_question(runner, question)
        expected = read_sales(oracle_sql)["rows"]
        data_ok = pie_matches(state, expected) if expected_chart == "pie" else normalized_rows(state.get("raw_query_result", [])) == normalized_rows(expected)
        if name == "Order count":
            # "Who?" permits returning just the customer; the count need not
            # be selected. Still verify that the SQL ranks distinct orders.
            actual = state.get("raw_query_result", [])
            data_ok = (data_ok or actual == [[expected[0][0]]]) and bool(
                re.search(r'COUNT\s*\(\s*DISTINCT\s+"Order ID"', state.get("sql_query", ""), re.IGNORECASE)
            )
        actual_chart = state.get("plot_decision", {}).get("chart_type") or "none"
        chart_ok = actual_chart == expected_chart and (expected_chart == "none" or bool(state.get("chart_json")))
        passed = state.get("workflow_status") == "completed" and data_ok and chart_ok
        result = {"case": name, "question": question, "passed": passed,
                  "data_correct": data_ok, "chart_correct": chart_ok,
                  "seconds": round(time.monotonic() - started, 2),
                  "sql": state.get("sql_query"), "columns": state.get("column_names"),
                  "rows": state.get("raw_query_result"), "answer": state.get("final_answer"),
                  "chart_type": actual_chart, "sql_attempts": state.get("sql_attempts"),
                  "report_attempts": state.get("report_attempts"),
                  "error": state.get("workflow_error"), "harness": state.get("harness_stats")}
        results.append(result)
        print(f"{'PASS' if passed else 'FAIL'} {name}: {result['seconds']}s", flush=True)
        if not passed:
            print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {"workflow": WORKFLOW_VARIANT, "model": ANALYSIS_MODEL,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "database_sha256": hashlib.sha256(DB_PATH.read_bytes()).hexdigest(),
              "passed": sum(r["passed"] for r in results), "total": len(results),
              "scope": "Actual SQL result values and chart generation; does not prove every natural-language claim.",
              "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['passed']}/{report['total']} passed", flush=True)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
