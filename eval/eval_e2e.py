#!/usr/bin/env python3
"""
End-to-End Evaluation Script for Text-to-SQL + Visualization Pipeline.

Evaluates the complete workflow:
1. SQL generation and execution correctness
2. Plot decision accuracy (should_plot + chart_type)
3. Plotly JSON validation (trace type, x-axis order, trace count)

Usage:
    uv run python eval/eval_e2e.py
    uv run python eval/eval_e2e.py --plot-only        # Only plot questions
    uv run python eval/eval_e2e.py --plot-only --limit 5
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
    answer: str  # Agent's final natural language answer
    sql_correct: bool
    sql_score: int  # 0-10 from judge
    sql_reasoning: str

    # Plot Decision Results
    has_expected_plot: bool  # Whether this question has plot ground truth
    expected_should_plot: bool
    actual_should_plot: bool
    expected_chart_type: str | None
    actual_chart_type: str | None
    plot_decision_correct: bool

    # Chart Validation Results (only if should_plot)
    chart_json_valid: bool  # Plotly JSON parseable and has traces
    trace_type_correct: bool  # trace type matches expected
    x_order_correct: bool  # x values match expected order
    trace_count_correct: bool  # trace count matches expected

    # Overall
    e2e_success: bool  # True if all components passed


@dataclass
class E2EMetrics:
    """Aggregated metrics for the evaluation."""

    total_cases: int = 0
    sql_correct_count: int = 0
    plot_decision_correct_count: int = 0
    chart_json_valid_count: int = 0
    trace_type_correct_count: int = 0
    x_order_correct_count: int = 0
    trace_count_correct_count: int = 0
    e2e_success_count: int = 0

    # Cases with plot ground truth (has_expected_plot=True)
    cases_with_plot_gt: int = 0
    # Subset: cases expecting should_plot=True
    cases_with_expected_plot: int = 0

    def add_result(self, result: E2EResult):
        self.total_cases += 1
        if result.sql_correct:
            self.sql_correct_count += 1
        if result.has_expected_plot:
            self.cases_with_plot_gt += 1
            if result.plot_decision_correct:
                self.plot_decision_correct_count += 1
            if result.expected_should_plot:
                self.cases_with_expected_plot += 1
                if result.chart_json_valid:
                    self.chart_json_valid_count += 1
                if result.trace_type_correct:
                    self.trace_type_correct_count += 1
                if result.x_order_correct:
                    self.x_order_correct_count += 1
                if result.trace_count_correct:
                    self.trace_count_correct_count += 1
        if result.e2e_success:
            self.e2e_success_count += 1

    def get_summary(self) -> Dict[str, Any]:
        plot_gt = self.cases_with_plot_gt
        plot_yes = self.cases_with_expected_plot
        return {
            "total_cases": self.total_cases,
            "sql_accuracy": (
                self.sql_correct_count / self.total_cases if self.total_cases > 0 else 0
            ),
            "plot_decision_accuracy": (
                self.plot_decision_correct_count / plot_gt if plot_gt > 0 else None
            ),
            "chart_json_valid_rate": (
                self.chart_json_valid_count / plot_yes if plot_yes > 0 else None
            ),
            "trace_type_accuracy": (
                self.trace_type_correct_count / plot_yes if plot_yes > 0 else None
            ),
            "x_order_accuracy": (
                self.x_order_correct_count / plot_yes if plot_yes > 0 else None
            ),
            "trace_count_accuracy": (
                self.trace_count_correct_count / plot_yes if plot_yes > 0 else None
            ),
            "e2e_success_rate": (
                self.e2e_success_count / self.total_cases if self.total_cases > 0 else 0
            ),
            "cases_with_plot_gt": plot_gt,
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


def validate_plotly_json(chart_json_str: str, expected_plot: dict) -> dict:
    """Parse Plotly JSON and validate against expected plot config."""
    try:
        chart = json.loads(chart_json_str) if isinstance(chart_json_str, str) else chart_json_str
    except (json.JSONDecodeError, TypeError):
        return {
            "chart_generated": False,
            "trace_type_correct": False,
            "x_order_correct": True,
            "trace_count_correct": False,
            "actual_trace_types": [],
            "actual_x_values": [],
            "actual_trace_count": 0,
        }

    traces = chart.get("data", [])

    # 1. Trace type validation
    type_map = {"bar": "bar", "line": "scatter", "pie": "pie"}
    expected_type = type_map.get(expected_plot.get("chart_type"))
    actual_types = [t.get("type") for t in traces]
    trace_type_correct = bool(actual_types) and all(
        t == expected_type for t in actual_types
    )

    # 2. X axis order validation (from trace data, not state metadata)
    expected_x = expected_plot.get("expected_x_order", [])
    if expected_type == "pie":
        actual_x = list(traces[0].get("labels", [])) if traces else []
    else:
        actual_x = [str(v) for v in traces[0].get("x", [])] if traces else []

    if expected_x:
        x_order_correct = actual_x == [str(v) for v in expected_x]
    else:
        x_order_correct = True

    # 3. Trace count validation
    expected_traces = expected_plot.get("expected_traces")
    trace_count_correct = (
        len(traces) == expected_traces if expected_traces is not None else True
    )

    return {
        "chart_generated": len(traces) > 0,
        "trace_type_correct": trace_type_correct,
        "x_order_correct": x_order_correct,
        "trace_count_correct": trace_count_correct,
        "actual_trace_types": actual_types,
        "actual_x_values": actual_x[:30],
        "actual_trace_count": len(traces),
    }


async def evaluate_single(entry: Dict, timeout: int = 300) -> E2EResult:
    """Evaluate a single test case end-to-end."""
    question = entry["question"]
    ground_truth = entry.get("ground_truth")
    eval_type = entry.get("eval_type", "unknown")
    difficulty = entry.get("difficulty", "unknown")
    has_expected_plot = "expected_plot" in entry
    expected_plot = entry.get("expected_plot", {})

    expected_should_plot = expected_plot.get("should_plot", False)
    expected_chart_type = expected_plot.get("chart_type")

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

        # Plot evaluation: only when expected_plot is defined
        if has_expected_plot:
            # Check plot decision correctness
            plot_decision_correct = actual_should_plot == expected_should_plot
            if expected_should_plot and actual_should_plot:
                plot_decision_correct = actual_chart_type == expected_chart_type

            # Validate chart via Plotly JSON
            chart_json_valid = False
            trace_type_correct = False
            x_order_correct = True
            trace_count_correct = True

            chart_json = state.get("chart_json", "")
            if expected_should_plot and chart_json:
                validation = validate_plotly_json(chart_json, expected_plot)
                chart_json_valid = validation["chart_generated"]
                trace_type_correct = validation["trace_type_correct"]
                x_order_correct = validation["x_order_correct"]
                trace_count_correct = validation["trace_count_correct"]
            elif not expected_should_plot:
                # No plot expected — these fields are N/A, default True
                chart_json_valid = True
                trace_type_correct = True
                trace_count_correct = True

            # Determine e2e success (plot-aware)
            e2e_success = sql_correct and plot_decision_correct
            if expected_should_plot:
                e2e_success = (
                    e2e_success
                    and chart_json_valid
                    and trace_type_correct
                    and x_order_correct
                )
                # trace_count_correct is a soft metric, not part of e2e_success
        else:
            # No expected_plot → skip plot evaluation entirely
            plot_decision_correct = True
            chart_json_valid = True
            trace_type_correct = True
            x_order_correct = True
            trace_count_correct = True
            e2e_success = sql_correct

        return E2EResult(
            question=question,
            eval_type=eval_type,
            difficulty=difficulty,
            sql_query=sql_query[:500] if sql_query else "",
            sql_result=(
                sql_result[:50]
                if isinstance(sql_result, list) and len(sql_result) > 50
                else sql_result
            ),
            answer=final_answer,
            sql_correct=sql_correct,
            sql_score=sql_score,
            sql_reasoning=sql_reasoning[:200],
            has_expected_plot=has_expected_plot,
            expected_should_plot=expected_should_plot,
            actual_should_plot=actual_should_plot,
            expected_chart_type=expected_chart_type,
            actual_chart_type=actual_chart_type,
            plot_decision_correct=plot_decision_correct,
            chart_json_valid=chart_json_valid,
            trace_type_correct=trace_type_correct,
            x_order_correct=x_order_correct,
            trace_count_correct=trace_count_correct,
            e2e_success=e2e_success,
        )

    except asyncio.TimeoutError:
        return E2EResult(
            question=question,
            eval_type=eval_type,
            difficulty=difficulty,
            sql_query="TIMEOUT",
            sql_result=None,
            answer="",
            sql_correct=False,
            sql_score=0,
            sql_reasoning="Timeout exceeded",
            has_expected_plot=has_expected_plot,
            expected_should_plot=expected_should_plot,
            actual_should_plot=False,
            expected_chart_type=expected_chart_type,
            actual_chart_type=None,
            plot_decision_correct=False,
            chart_json_valid=False,
            trace_type_correct=False,
            x_order_correct=False,
            trace_count_correct=False,
            e2e_success=False,
        )

    except Exception as e:
        return E2EResult(
            question=question,
            eval_type=eval_type,
            difficulty=difficulty,
            sql_query="ERROR",
            sql_result=None,
            answer="",
            sql_correct=False,
            sql_score=0,
            sql_reasoning=str(e)[:200],
            has_expected_plot=has_expected_plot,
            expected_should_plot=expected_should_plot,
            actual_should_plot=False,
            expected_chart_type=expected_chart_type,
            actual_chart_type=None,
            plot_decision_correct=False,
            chart_json_valid=False,
            trace_type_correct=False,
            x_order_correct=False,
            trace_count_correct=False,
            e2e_success=False,
        )


async def get_latest_failed_results() -> pathlib.Path | None:
    """Find the most recent e2e_results.json file that has FAILED cases."""
    results_dir = PROJECT_DIR / "eval" / "results"
    if not results_dir.exists():
        return None

    # Find all json files in any subdirectory of eval/results
    # Matching pattern: e2e_*/e2e_results*.json to catch retries too
    result_files = sorted(
        results_dir.glob("**/e2e_results*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for file_path in result_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Check if there are any failed cases
                if any(not r.get("e2e_success", False) for r in data):
                    return file_path
        except Exception:
            continue

    return None


async def run_evaluation(
    limit: int = None, retry_failed: bool = False, plot_only: bool = False
):
    """Run the full end-to-end evaluation."""
    # Load eval set
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = [json.loads(line) for line in f if line.strip()]

    # Filter to plot-only questions if requested
    if plot_only:
        eval_set = [e for e in eval_set if "expected_plot" in e]
        print(f"Plot-only mode: {len(eval_set)} questions with expected_plot")

    previous_results = []
    output_dir = None
    results_filename = "e2e_results.json"
    summary_filename = "e2e_summary.json"

    if retry_failed:
        latest_results_path = await get_latest_failed_results()
        if not latest_results_path:
            print("No previous results with failures found to retry. 🎉")
            return

        print(f"Loading previous results from: {latest_results_path}")
        with open(latest_results_path, "r", encoding="utf-8") as f:
            previous_results = json.load(f)

        # Identify failed cases
        failed_questions = {
            r["question"] for r in previous_results if not r.get("e2e_success", False)
        }

        if not failed_questions:
            print("No failed cases found in the selected run! 🎉")
            # Should correspond to get_latest_failed_results filtering, but double check
            return

        # Filter eval set to only include failed cases
        original_count = len(eval_set)
        eval_set = [
            entry for entry in eval_set if entry["question"] in failed_questions
        ]

        print(
            f"Retrying {len(eval_set)} failed cases (out of {original_count} total cases)."
        )

        # Output config for retry
        output_dir = latest_results_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"e2e_results_retry_{timestamp}.json"
        summary_filename = f"e2e_summary_retry_{timestamp}.json"

    if limit:
        eval_set = eval_set[:limit]

    # If NOT retry mode, create new directory
    if output_dir is None:
        output_dir = (
            PROJECT_DIR
            / "eval"
            / "results"
            / f"e2e_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    metrics = E2EMetrics()
    new_results: List[E2EResult] = []

    print(f"\n{'='*70}")
    print(f"End-to-End Evaluation | Cases: {len(eval_set)}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*70}\n")

    for i, entry in enumerate(eval_set):
        question = entry["question"]
        has_plot = "expected_plot" in entry

        print(f"[{i+1}/{len(eval_set)}] {question[:50]}... ", end="", flush=True)

        result = await evaluate_single(entry)
        new_results.append(result)
        metrics.add_result(result)

        # Status indicators
        sql_status = "✓" if result.sql_correct else "✗"
        plot_status = ""
        if has_plot:
            ep = entry["expected_plot"]
            if ep.get("should_plot"):
                p_ok = "✓" if result.plot_decision_correct else "✗"
                t_ok = "✓" if result.trace_type_correct else "✗"
                x_ok = "✓" if result.x_order_correct else "✗"
                c_ok = "✓" if result.trace_count_correct else "✗"
                plot_status = f" [Plot:{p_ok} Trace:{t_ok} X:{x_ok} Count:{c_ok}]"
            else:
                p_ok = "✓" if result.plot_decision_correct else "✗"
                plot_status = f" [Plot:{p_ok}(no-plot)]"

        e2e_status = "🟢" if result.e2e_success else "🔴"

        print(f"{sql_status} SQL={result.sql_score}/10{plot_status} {e2e_status}")

        # --- Merge and Save Step ---
        # 1. Start with previous results (if any)
        # 2. Update with new results

        final_results_list = []

        if retry_failed:
            # Map new results by question for easy lookup
            new_results_map = {r.question: asdict(r) for r in new_results}

            # Iterate through OLD results to preserve order
            for old_r in previous_results:
                q = old_r.get("question")
                if q in new_results_map:
                    # Replace with new result
                    final_results_list.append(new_results_map[q])
                else:
                    # Keep old result
                    final_results_list.append(old_r)
        else:
            final_results_list = [asdict(r) for r in new_results]

        # Save merged results
        with open(output_dir / results_filename, "w", encoding="utf-8") as f:
            json.dump(final_results_list, f, indent=2, ensure_ascii=False)

    # --- Final Summary Calculation ---
    # We need to re-calculate metrics based on the FULL merged list
    final_metrics = E2EMetrics()

    for r_dict in final_results_list:
        try:
            # Reconstruct object to use logic in add_result
            r_obj = E2EResult(**r_dict)
            final_metrics.add_result(r_obj)
        except Exception:
            pass

    summary = final_metrics.get_summary()
    summary["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*70}")
    print(f"FINAL STATS (Merged):")
    print(f"SQL Accuracy:           {summary['sql_accuracy']:.1%}")
    if summary["plot_decision_accuracy"] is not None:
        print(f"Plot Decision Accuracy: {summary['plot_decision_accuracy']:.1%} ({summary['cases_with_plot_gt']} questions)")
    if summary["chart_json_valid_rate"] is not None:
        print(f"Chart JSON Valid Rate:  {summary['chart_json_valid_rate']:.1%}")
        print(f"Trace Type Accuracy:    {summary['trace_type_accuracy']:.1%}")
        print(f"X-Order Accuracy:       {summary['x_order_accuracy']:.1%}")
        print(f"Trace Count Accuracy:   {summary['trace_count_accuracy']:.1%}")
    print(f"E2E Success Rate:       {summary['e2e_success_rate']:.1%}")
    print(f"{'='*70}\n")

    # Save summary
    with open(output_dir / summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_dir}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run end-to-end evaluation")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of test cases"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only failed cases from the last run",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only run questions with expected_plot",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            limit=args.limit,
            retry_failed=args.retry_failed,
            plot_only=args.plot_only,
        )
    )
