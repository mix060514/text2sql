#!/usr/bin/env python3
"""
Test script for failed evaluation cases.
Run this BEFORE full e2e evaluation to verify fixes.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Setup path to import local src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from google.adk.runners import InMemoryRunner
from google.genai import types

from text2sql.agents.planned_agent.agent import root_agent
from text2sql.agents.planned_agent.tools import read_sales
from text2sql.runtime import run_question

# Compute against the current database, rather than retaining a seed-42 value.
LATAM_REVENUE = read_sales('SELECT SUM("Total Revenue") FROM sales_data WHERE Region = \'LATAM\'')["rows"][0][0]

# Failed cases extracted from e2e_results.json
FAILED_CASES = [
    # Region mapping issues
    {
        "question": "南美 的總銷售額是多少？",
        "expected_contains": [str(LATAM_REVENUE), f"{LATAM_REVENUE:,.2f}"],
        "check_type": "contains_any",
        "issue": "南美 should map to LATAM",
    },
    {
        "question": "歐非中東 銷售量最高的前三個產品是什麼？",
        "expected_sql_not_contain": "APAC",  # Should NOT use APAC
        "expected_sql_contain": "EMEA",  # Should use EMEA
        "check_type": "sql_region",
        "issue": "歐非中東 should map to EMEA, not APAC",
    },
    # Order Count vs Quantity confusion
    {
        "question": "Canada 訂單數量最多的客戶是誰？",
        "expected_sql_contain": "COUNT(DISTINCT",
        "check_type": "sql_contains",
        "issue": "訂單數量 should use COUNT(DISTINCT Order ID)",
    },
    {
        "question": "India 訂單數量最多的客戶是誰？",
        "expected_sql_contain": "COUNT(DISTINCT",
        "check_type": "sql_contains",
        "issue": "訂單數量 should use COUNT(DISTINCT Order ID)",
    },
    # Plot decision issues - should plot but didn't
    {
        "question": "請列出 Electronics 類別在各年份的銷售額總計。",
        "expected_plot": True,
        "check_type": "should_plot",
        "issue": "Multi-year comparison should generate chart",
    },
    {
        "question": "請列出 Software 類別在各年份的銷售額總計。",
        "expected_plot": True,
        "check_type": "should_plot",
        "issue": "Multi-year comparison should generate chart",
    },
]


async def run_agent_and_get_state(runner: InMemoryRunner, question: str) -> dict:
    """Execute agent and return session state, following eval_e2e.py pattern."""
    return await run_question(runner, question, user_id="test_user")


async def run_single_test(runner: InMemoryRunner, case: dict) -> dict:
    """Run a single test case and return results."""
    question = case["question"]

    try:
        state = await asyncio.wait_for(
            run_agent_and_get_state(runner, question), timeout=120
        )

        # Extract results
        sql_query = state.get("sql_query", "")
        answer = state.get("answer", "")
        plot_decision = state.get("plot_decision", {})

        # Check based on type
        check_type = case.get("check_type", "")
        passed = False
        reason = ""

        if check_type == "contains_any":
            expected = case.get("expected_contains", [])
            passed = any(exp in answer for exp in expected)
            reason = f"Answer: {answer[:200]}..." if answer else "No answer"

        elif check_type == "sql_contains":
            expected = case.get("expected_sql_contain", "")
            passed = expected.upper() in sql_query.upper() if sql_query else False
            reason = f"SQL: {sql_query[:300]}..." if sql_query else "No SQL"

        elif check_type == "sql_region":
            should_contain = case.get("expected_sql_contain", "")
            should_not_contain = case.get("expected_sql_not_contain", "")
            sql_upper = sql_query.upper()
            passes_contain = (
                should_contain.upper() in sql_upper if should_contain else True
            )
            passes_not_contain = (
                should_not_contain.upper() not in sql_upper
                if should_not_contain
                else True
            )
            passed = passes_contain and passes_not_contain
            reason = f"SQL: {sql_query[:300]}..." if sql_query else "No SQL"

        elif check_type == "should_plot":
            should_plot = (
                plot_decision.get("should_plot", False)
                if isinstance(plot_decision, dict)
                else False
            )
            passed = should_plot == case.get("expected_plot", False) and bool(state.get("chart_json"))
            reason = f"Plot decision: {plot_decision}"

        passed = passed and state.get("workflow_status") == "completed"
        return {
            "question": question,
            "issue": case.get("issue", ""),
            "passed": passed,
            "reason": reason,
            "sql_query": sql_query[:500] if sql_query else "",
        }

    except asyncio.TimeoutError:
        return {
            "question": question,
            "issue": case.get("issue", ""),
            "passed": False,
            "reason": "Timeout (120s)",
            "sql_query": "",
        }
    except Exception as e:
        import traceback

        return {
            "question": question,
            "issue": case.get("issue", ""),
            "passed": False,
            "reason": f"Error: {str(e)}\n{traceback.format_exc()[:200]}",
            "sql_query": "",
        }


async def main():
    """Run all failed case tests."""
    print("=" * 60)
    print("Testing Failed Cases")
    print("=" * 60)

    runner = InMemoryRunner(agent=root_agent, app_name="failed_cases_test")

    results = []
    passed_count = 0

    for i, case in enumerate(FAILED_CASES):
        print(f"\n[{i+1}/{len(FAILED_CASES)}] Testing: {case['question'][:50]}...")

        result = await run_single_test(runner, case)
        results.append(result)

        if result["passed"]:
            passed_count += 1
            print(f"  ✅ PASSED")
        else:
            print(f"  ❌ FAILED - {result['issue']}")
            print(f"     Reason: {result['reason'][:100]}...")

    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {passed_count}/{len(FAILED_CASES)} tests passed")
    print("=" * 60)

    # Save results
    output_path = Path(__file__).parent / "failed_cases_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    # Return exit code
    return 0 if passed_count == len(FAILED_CASES) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
