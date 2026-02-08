#!/usr/bin/env python3
"""
Evaluation script for plot decision agent.

Tests whether the agent correctly decides:
1. should_plot: Whether to visualize the data
2. chart_type: Which chart type to use
3. x_column, y_column, color_column: Whether columns are correctly mapped
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

# Import the plot_decision_agent for standalone testing
from text2sql.agents.planned_agent.agent import plot_decision_agent


PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
PLOT_EVAL_SET = DATA_DIR / "plot_eval_set.jsonl"


# Load evaluation set
with open(PLOT_EVAL_SET, "r", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]

print(f"Loaded {len(eval_set)} plot evaluation cases")


# Create runner for plot_decision_agent
plot_runner = InMemoryRunner(agent=plot_decision_agent, app_name="plot_decision_eval")


@dataclass
class PlotEvalResult:
    """Single evaluation result"""

    question: str
    expected_should_plot: bool
    expected_chart_type: str | None
    expected_x: str | None
    expected_y: str | None
    expected_color: str | None

    actual_should_plot: bool | None
    actual_chart_type: str | None
    actual_x: str | None
    actual_y: str | None
    actual_color: str | None

    should_plot_correct: bool
    chart_type_correct: bool
    columns_correct: bool

    raw_response: str


async def run_plot_decision(question: str, mock_columns: List[str] = None) -> Dict:
    """Run plot_decision_agent and get result"""
    session_id = f"plot_eval_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    # Mock query result values (placeholder)
    mock_query_result = [("Sample", 100)]

    state = {
        "user_question": question,
        "raw_query_result": mock_query_result,
    }

    if mock_columns:
        state["column_names"] = mock_columns

    await plot_runner.session_service.create_session(
        app_name=plot_runner.app_name,
        user_id="eval_user",
        session_id=session_id,
        state=state,
    )

    # The prompt triggers the agent to make a decision based on state
    content = types.Content(
        role="user",
        parts=[
            types.Part(text=f"Decide if this question needs visualization: {question}")
        ],
    )

    # Collect events
    response_text = ""
    async for event in plot_runner.run_async(
        user_id="eval_user",
        session_id=session_id,
        new_message=content,
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    response_text += part.text

    # Get final state
    session = await plot_runner.session_service.get_session(
        app_name=plot_runner.app_name,
        user_id="eval_user",
        session_id=session_id,
    )

    plot_decision = session.state.get("plot_decision", {}) if session else {}

    return {
        "should_plot": plot_decision.get("should_plot"),
        "chart_type": plot_decision.get("chart_type"),
        "x_column": plot_decision.get("x_column"),
        "y_column": plot_decision.get("y_column"),
        "color_column": plot_decision.get("color_column"),
        "raw_response": response_text,
    }


async def evaluate_single(entry: Dict) -> PlotEvalResult:
    """Evaluate a single test case"""
    question = entry["question"]
    mock_columns = entry.get("mock_columns", [])

    expected_should_plot = entry["should_plot"]
    expected_chart_type = entry.get("chart_type")
    expected_x = entry.get("x_column")
    expected_y = entry.get("y_column")
    expected_color = entry.get("color_column")

    try:
        result = await asyncio.wait_for(
            run_plot_decision(question, mock_columns), timeout=60
        )

        actual_should_plot = result.get("should_plot")
        actual_chart_type = result.get("chart_type")
        actual_x = result.get("x_column")
        actual_y = result.get("y_column")
        actual_color = result.get("color_column")

        should_plot_correct = actual_should_plot == expected_should_plot

        chart_type_correct = False
        columns_correct = False

        if expected_should_plot and actual_should_plot:
            chart_type_correct = actual_chart_type == expected_chart_type

            # Check columns if should_plot is correct
            # We allow approximate matching or exact matching? Exact for now.
            x_match = actual_x == expected_x
            y_match = actual_y == expected_y
            # Color is tricky: if expected is None, actual should be None or not present
            color_match = (actual_color == expected_color) or (
                not expected_color and not actual_color
            )

            columns_correct = x_match and y_match and color_match

        elif not expected_should_plot and not actual_should_plot:
            chart_type_correct = True
            columns_correct = True  # Valid since no plot needed
        else:
            chart_type_correct = False
            columns_correct = False

        return PlotEvalResult(
            question=question,
            expected_should_plot=expected_should_plot,
            expected_chart_type=expected_chart_type,
            expected_x=expected_x,
            expected_y=expected_y,
            expected_color=expected_color,
            actual_should_plot=actual_should_plot,
            actual_chart_type=actual_chart_type,
            actual_x=actual_x,
            actual_y=actual_y,
            actual_color=actual_color,
            should_plot_correct=should_plot_correct,
            chart_type_correct=chart_type_correct,
            columns_correct=columns_correct,
            raw_response=result.get("raw_response", "")[:200],
        )

    except Exception as e:
        return PlotEvalResult(
            question=question,
            expected_should_plot=expected_should_plot,
            expected_chart_type=expected_chart_type,
            expected_x=expected_x,
            expected_y=expected_y,
            expected_color=expected_color,
            actual_should_plot=None,
            actual_chart_type=None,
            actual_x=None,
            actual_y=None,
            actual_color=None,
            should_plot_correct=False,
            chart_type_correct=False,
            columns_correct=False,
            raw_response=f"Error: {e}",
        )


async def run_evaluation():
    """Run full evaluation"""
    output_dir = (
        PROJECT_DIR
        / "eval"
        / "results"
        / f"plot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[PlotEvalResult] = []

    print(f"\n{'='*60}")
    print(f"Plot Decision Eval | Cases: {len(eval_set)}")
    print(f"{'='*60}\n")

    should_plot_correct = 0
    chart_type_correct = 0
    columns_correct = 0
    plot_cases_total = 0  # Count cases where should_plot is True

    for i, entry in enumerate(eval_set):
        print(
            f"[{i+1}/{len(eval_set)}] {entry['question'][:40]}... ", end="", flush=True
        )

        result = await evaluate_single(entry)
        results.append(result)

        if result.should_plot_correct:
            should_plot_correct += 1

        if entry["should_plot"]:
            plot_cases_total += 1
            if result.chart_type_correct and result.should_plot_correct:
                chart_type_correct += 1
            if result.columns_correct and result.should_plot_correct:
                columns_correct += 1

        status = "✓" if result.should_plot_correct else "✗"
        chart_status = ""
        if result.actual_should_plot:
            chart_status = f" [{result.actual_chart_type}]"
            if not result.columns_correct:
                chart_status += f" (Cols ✗ A:{result.actual_x}/{result.actual_y}/{result.actual_color} E:{result.expected_x}/{result.expected_y}/{result.expected_color})"

        print(f"{status}{chart_status}")

    # Calculate metrics
    should_plot_accuracy = should_plot_correct / len(eval_set) if eval_set else 0
    chart_type_accuracy = (
        chart_type_correct / plot_cases_total if plot_cases_total > 0 else 0
    )
    columns_accuracy = columns_correct / plot_cases_total if plot_cases_total > 0 else 0

    print(f"\n{'='*60}")
    print(
        f"should_plot (有無畫圖): {should_plot_accuracy:.1%} ({should_plot_correct}/{len(eval_set)})"
    )
    print(
        f"chart_type  (圖表類型): {chart_type_accuracy:.1%} ({chart_type_correct}/{plot_cases_total})"
    )
    print(
        f"columns     (欄位對應): {columns_accuracy:.1%} ({columns_correct}/{plot_cases_total})"
    )
    print(f"{'='*60}\n")

    # Save results
    with open(output_dir / "plot_eval_results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": len(eval_set),
        "should_plot_accuracy": should_plot_accuracy,
        "chart_type_accuracy": chart_type_accuracy,
        "columns_accuracy": columns_accuracy,
    }

    with open(output_dir / "plot_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"結果已儲存至 {output_dir}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
