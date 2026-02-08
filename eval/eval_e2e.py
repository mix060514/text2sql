#!/usr/bin/env python3
"""
End-to-End Evaluation Script for Text-to-SQL + Visualization Pipeline.

Evaluates the complete workflow:
1. SQL generation and execution correctness
2. Plot decision accuracy
3. Chart data correctness (x_values order, data integrity)
4. Chart image generation

Usage:
    uv run python eval/eval_e2e.py
"""

import json
import pathlib
import sys
import os
import asyncio
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List

# Setup path to import local src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from google.adk.runners import InMemoryRunner
from google.genai import types

# Import Agents
from text2sql.agents.planned_agent.agent import root_agent
from text2sql.agents.judge_agent.agent import judge_agent


PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
EVAL_SET_PATH = DATA_DIR / "eval_set_v3.jsonl"

# Shared runner instances
eval_runner = InMemoryRunner(agent=root_agent, app_name="e2e_eval")
judge_runner = InMemoryRunner(agent=judge_agent, app_name="judge_eval")


@dataclass
class E2EResult:
    """End-to-end evaluation result for a single question."""

    question: str
    eval_type: str
    difficulty: str

    # SQL Results
    sql_query: str
    sql_result: Any
    sql_correct: bool
    sql_score: int  # 0-10 from judge
    sql_reasoning: str

    # Plot Decision Results
    expected_should_plot: bool
    actual_should_plot: bool
    expected_chart_type: str | None
    actual_chart_type: str | None
    plot_decision_correct: bool

    # Chart Data Results (only if should_plot)
    chart_data: Dict | None
    chart_image_generated: bool
    x_order_correct: bool  # True if x_values match expected order

    # Overall
    e2e_success: bool  # True if all components passed


@dataclass
class E2EMetrics:
    """Aggregated metrics for the evaluation."""

    total_cases: int = 0
    sql_correct_count: int = 0
    plot_decision_correct_count: int = 0
    chart_generated_count: int = 0
    x_order_correct_count: int = 0
    e2e_success_count: int = 0

    # Cases with expected plots
    cases_with_expected_plot: int = 0

    def add_result(self, result: E2EResult):
        self.total_cases += 1
        if result.sql_correct:
            self.sql_correct_count += 1
        if result.plot_decision_correct:
            self.plot_decision_correct_count += 1
        if result.expected_should_plot:
            self.cases_with_expected_plot += 1
            if result.chart_image_generated:
                self.chart_generated_count += 1
            if result.x_order_correct:
                self.x_order_correct_count += 1
        if result.e2e_success:
            self.e2e_success_count += 1

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "sql_accuracy": (
                self.sql_correct_count / self.total_cases if self.total_cases > 0 else 0
            ),
            "plot_decision_accuracy": (
                self.plot_decision_correct_count / self.total_cases
                if self.total_cases > 0
                else 0
            ),
            "chart_generation_rate": (
                self.chart_generated_count / self.cases_with_expected_plot
                if self.cases_with_expected_plot > 0
                else None
            ),
            "x_order_accuracy": (
                self.x_order_correct_count / self.cases_with_expected_plot
                if self.cases_with_expected_plot > 0
                else None
            ),
            "e2e_success_rate": (
                self.e2e_success_count / self.total_cases if self.total_cases > 0 else 0
            ),
            "cases_with_expected_plot": self.cases_with_expected_plot,
        }


async def run_agent_and_get_state(
    runner: InMemoryRunner, prompt: str, user_id: str = "eval_user"
) -> dict:
    """Execute agent and return session state."""
    session_id = f"{runner.app_name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for _ in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        pass

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    return session.state if session else {}


async def judge_sql_result(
    question: str, expected: Any, actual_answer: str, sql_query: str, sql_result: Any
) -> tuple[bool, int, str]:
    """Use Judge Agent to evaluate SQL result correctness."""
    try:
        prompt = f"""請評估以下 Text-to-SQL Agent 的回答品質：

**使用者問題**: {question}

**預期正確答案**: {json.dumps(expected, ensure_ascii=False)}

**SQL 查詢**: {sql_query if sql_query else "無法取得"}

**SQL 結果**: {json.dumps(sql_result, ensure_ascii=False) if sql_result else "無法取得"}

**Agent 回答**: {actual_answer}
"""
        state = await run_agent_and_get_state(judge_runner, prompt)
        judge_result = state.get("judge_result", {})

        if not judge_result:
            return False, 0, "Judge Agent did not return result"

        is_correct = judge_result.get("is_correct", False)
        score = min(10, max(0, int(judge_result.get("confidence_score", 0))))
        reasoning = judge_result.get("reasoning", "")

        return is_correct, score, reasoning

    except Exception as e:
        return False, 0, f"Judge error: {str(e)}"


def check_x_order(actual_x_values: List, expected_x_order: List) -> bool:
    """Check if actual x_values match expected order."""
    if not expected_x_order:
        return True  # No expected order to check

    if not actual_x_values:
        return False

    # Compare lists directly
    return list(actual_x_values) == list(expected_x_order)


async def evaluate_single(entry: Dict, timeout: int = 300) -> E2EResult:
    """Evaluate a single test case end-to-end."""
    question = entry["question"]
    ground_truth = entry.get("ground_truth")
    eval_type = entry.get("eval_type", "unknown")
    difficulty = entry.get("difficulty", "unknown")
    expected_plot = entry.get("expected_plot", {})

    expected_should_plot = expected_plot.get("should_plot", False)
    expected_chart_type = expected_plot.get("chart_type")
    expected_x_order = expected_plot.get("expected_x_order", [])

    try:
        # Run the full workflow
        state = await asyncio.wait_for(
            run_agent_and_get_state(eval_runner, question), timeout=timeout
        )

        # Extract SQL results
        sql_query = state.get("sql_query", "")
        sql_result = state.get("raw_query_result")
        final_answer = state.get("answer", "")

        # Judge SQL result
        sql_correct, sql_score, sql_reasoning = await judge_sql_result(
            question, ground_truth, final_answer, sql_query, sql_result
        )

        # Extract plot decision
        plot_decision = state.get("plot_decision", {})
        actual_should_plot = plot_decision.get("should_plot", False)
        actual_chart_type = plot_decision.get("chart_type")

        # Check plot decision correctness
        plot_decision_correct = actual_should_plot == expected_should_plot
        if expected_should_plot and actual_should_plot:
            plot_decision_correct = actual_chart_type == expected_chart_type

        # Extract chart data
        chart_data = state.get("chart_data")
        chart_image = state.get("chart_image")
        chart_image_generated = bool(chart_image)

        # Check x_order
        x_order_correct = True
        if expected_should_plot and chart_data:
            actual_x_values = chart_data.get("x_values", [])
            x_order_correct = check_x_order(actual_x_values, expected_x_order)

        # Determine e2e success
        e2e_success = sql_correct and plot_decision_correct
        if expected_should_plot:
            e2e_success = e2e_success and chart_image_generated and x_order_correct

        return E2EResult(
            question=question,
            eval_type=eval_type,
            difficulty=difficulty,
            sql_query=sql_query[:500] if sql_query else "",
            sql_result=(
                sql_result[:10]
                if isinstance(sql_result, list) and len(sql_result) > 10
                else sql_result
            ),
            sql_correct=sql_correct,
            sql_score=sql_score,
            sql_reasoning=sql_reasoning[:200],
            expected_should_plot=expected_should_plot,
            actual_should_plot=actual_should_plot,
            expected_chart_type=expected_chart_type,
            actual_chart_type=actual_chart_type,
            plot_decision_correct=plot_decision_correct,
            chart_data=chart_data,
            chart_image_generated=chart_image_generated,
            x_order_correct=x_order_correct,
            e2e_success=e2e_success,
        )

    except asyncio.TimeoutError:
        return E2EResult(
            question=question,
            eval_type=eval_type,
            difficulty=difficulty,
            sql_query="TIMEOUT",
            sql_result=None,
            sql_correct=False,
            sql_score=0,
            sql_reasoning="Timeout exceeded",
            expected_should_plot=expected_should_plot,
            actual_should_plot=False,
            expected_chart_type=expected_chart_type,
            actual_chart_type=None,
            plot_decision_correct=False,
            chart_data=None,
            chart_image_generated=False,
            x_order_correct=False,
            e2e_success=False,
        )

    except Exception as e:
        return E2EResult(
            question=question,
            eval_type=eval_type,
            difficulty=difficulty,
            sql_query="ERROR",
            sql_result=None,
            sql_correct=False,
            sql_score=0,
            sql_reasoning=str(e)[:200],
            expected_should_plot=expected_should_plot,
            actual_should_plot=False,
            expected_chart_type=expected_chart_type,
            actual_chart_type=None,
            plot_decision_correct=False,
            chart_data=None,
            chart_image_generated=False,
            x_order_correct=False,
            e2e_success=False,
        )


async def run_evaluation(limit: int = None):
    """Run the full end-to-end evaluation."""
    # Load eval set
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = [json.loads(line) for line in f if line.strip()]

    if limit:
        eval_set = eval_set[:limit]

    output_dir = (
        PROJECT_DIR
        / "eval"
        / "results"
        / f"e2e_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = E2EMetrics()
    results: List[E2EResult] = []

    print(f"\n{'='*70}")
    print(f"End-to-End Evaluation | Cases: {len(eval_set)}")
    print(f"{'='*70}\n")

    for i, entry in enumerate(eval_set):
        question = entry["question"]
        has_plot = "expected_plot" in entry

        print(f"[{i+1}/{len(eval_set)}] {question[:50]}... ", end="", flush=True)

        result = await evaluate_single(entry)
        results.append(result)
        metrics.add_result(result)

        # Status indicators
        sql_status = "✓" if result.sql_correct else "✗"
        plot_status = ""
        if has_plot:
            p_ok = "✓" if result.plot_decision_correct else "✗"
            c_ok = "✓" if result.chart_image_generated else "✗"
            x_ok = "✓" if result.x_order_correct else "✗"
            plot_status = f" [Plot:{p_ok} Chart:{c_ok} Order:{x_ok}]"

        e2e_status = "🟢" if result.e2e_success else "🔴"

        print(f"{sql_status} SQL={result.sql_score}/10{plot_status} {e2e_status}")

        # Save intermediate results
        with open(output_dir / "e2e_results.json", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # Final summary
    summary = metrics.get_summary()
    summary["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*70}")
    print(f"SQL Accuracy:           {summary['sql_accuracy']:.1%}")
    print(f"Plot Decision Accuracy: {summary['plot_decision_accuracy']:.1%}")
    if summary["chart_generation_rate"] is not None:
        print(f"Chart Generation Rate:  {summary['chart_generation_rate']:.1%}")
        print(f"X-Order Accuracy:       {summary['x_order_accuracy']:.1%}")
    print(f"E2E Success Rate:       {summary['e2e_success_rate']:.1%}")
    print(f"{'='*70}\n")

    # Save summary
    with open(output_dir / "e2e_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_dir}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run end-to-end evaluation")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of test cases"
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(limit=args.limit))
