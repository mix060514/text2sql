#!/usr/bin/env python3
"""
Generate ground truth dataset for plot decision evaluation.

This script creates plot_eval_set.jsonl containing:
- Questions that should be visualized
- Mock SQL columns for context
- Expected PlotDecision (chart_type, x, y, color)
"""

import json
import pathlib

OUTPUT_FILE = pathlib.Path(__file__).parent.parent / "data" / "plot_eval_set.jsonl"


def generate_plot_eval_set():
    """Generate evaluation dataset for plot decision testing."""

    eval_set = []

    # =========================================================================
    # PIE CHART cases - 佔比、分佈
    # =========================================================================
    # Expectation: x=Category, y=Value, color=None
    pie_cases = [
        {
            "question": "各區域銷售額佔比是多少？",
            "mock_columns": ["Region", "Total Revenue"],
            "expected_x": "Region",
            "expected_y": "Total Revenue",
        },
        {
            "question": "各產品類別的銷售額比例為何？",
            "mock_columns": ["Product Category", "Total Revenue"],
            "expected_x": "Product Category",
            "expected_y": "Total Revenue",
        },
        {
            "question": "顯示各區域的訂單數量分佈",
            "mock_columns": ["Region", "Order Count"],
            "expected_x": "Region",
            "expected_y": "Order Count",
        },
        {
            "question": "North America 各產品類別的銷售比例",
            "mock_columns": ["Product Category", "Total Revenue"],
            "expected_x": "Product Category",
            "expected_y": "Total Revenue",
        },
        {
            "question": "APAC 區域的客戶訂單佔比",
            "mock_columns": ["Customer Name", "Order Count"],
            "expected_x": "Customer Name",
            "expected_y": "Order Count",
        },
    ]

    for case in pie_cases:
        eval_set.append(
            {
                "question": case["question"],
                "mock_columns": case["mock_columns"],
                "should_plot": True,
                "chart_type": "pie",
                "x_column": case["expected_x"],
                "y_column": case["expected_y"],
                "color_column": None,
                "eval_type": "plot_decision",
            }
        )

    # =========================================================================
    # LINE CHART cases - 趨勢
    # =========================================================================
    line_cases = [
        {
            "question": "顯示北美 2023-2024 每月銷售趨勢",
            "mock_columns": ["Month", "Total Revenue", "Year"],
            "expected_x": "Month",
            "expected_y": "Total Revenue",
            "expected_color": "Year",
        },
        {
            "question": "畫出 EMEA 區域過去兩年的月銷售額走勢",
            "mock_columns": ["Year", "Month", "Total Revenue"],
            "expected_x": "Month",
            "expected_y": "Total Revenue",
            "expected_color": "Year",
        },
        {
            "question": "Electronics 類別的年度銷售趨勢圖",
            "mock_columns": ["Year", "Total Revenue"],
            "expected_x": "Year",
            "expected_y": "Total Revenue",
            "expected_color": None,
        },
        {
            "question": "請畫出各季度的銷售額變化",
            "mock_columns": ["Quarter", "Year", "Total Revenue"],
            "expected_x": "Quarter",
            "expected_y": "Total Revenue",
            "expected_color": "Year",
        },
        {
            "question": "2023 年各月的訂單數量趨勢",
            "mock_columns": ["Month", "Order Count"],
            "expected_x": "Month",
            "expected_y": "Order Count",
            "expected_color": None,
        },
    ]

    for case in line_cases:
        eval_set.append(
            {
                "question": case["question"],
                "mock_columns": case["mock_columns"],
                "should_plot": True,
                "chart_type": "line",
                "x_column": case["expected_x"],
                "y_column": case["expected_y"],
                "color_column": case.get("expected_color"),
                "eval_type": "plot_decision",
            }
        )

    # =========================================================================
    # BAR CHART cases - 比較
    # =========================================================================
    bar_cases = [
        {
            "question": "各區域的銷售額比較",
            "mock_columns": ["Region", "Total Revenue"],
            "expected_x": "Region",
            "expected_y": "Total Revenue",
            "expected_color": None,
        },
        {
            "question": "顯示前五大客戶的購買金額",
            "mock_columns": ["Customer Name", "Total Revenue"],
            "expected_x": "Customer Name",
            "expected_y": "Total Revenue",
            "expected_color": None,
        },
        {
            "question": "比較 2023 和 2024 年各區域的銷售額",
            "mock_columns": ["Region", "Year", "Total Revenue"],
            "expected_x": "Region",
            "expected_y": "Total Revenue",
            "expected_color": "Year",
        },
        {
            "question": "APAC 和 EMEA 的季度銷售比較",
            "mock_columns": ["Quarter", "Region", "Total Revenue"],
            "expected_x": "Quarter",
            "expected_y": "Total Revenue",
            "expected_color": "Region",
        },
    ]

    for case in bar_cases:
        eval_set.append(
            {
                "question": case["question"],
                "mock_columns": case["mock_columns"],
                "should_plot": True,
                "chart_type": "bar",
                "x_column": case["expected_x"],
                "y_column": case["expected_y"],
                "color_column": case.get("expected_color"),
                "eval_type": "plot_decision",
            }
        )

    # =========================================================================
    # NO PLOT cases
    # =========================================================================
    no_plot_cases = [
        {
            "question": "北美的總銷售額是多少？",
            "mock_columns": ["Total Revenue"],
        },
        {
            "question": "銷售量最高的前 10 個產品",  # ambiguous, technically a table.
            "mock_columns": ["Product Name", "Sales Volume"],
        },
        {
            "question": "2023 年哪個季度銷售最好？",
            "mock_columns": ["Quarter", "Total Revenue"],
            # This returns one row "Q4", or list of quarters.
            # If explicit "which quarter", usually 1 row result.
        },
        {
            "question": "Electronics 在 2024 年的 YoY 是多少？",
            "mock_columns": ["YoY Growth"],
        },
        {
            "question": "總共有多少筆訂單？",
            "mock_columns": ["Order Count"],
        },
    ]

    for case in no_plot_cases:
        eval_set.append(
            {
                "question": case["question"],
                "mock_columns": case["mock_columns"],
                "should_plot": False,
                "chart_type": None,
                "x_column": None,
                "y_column": None,
                "color_column": None,
                "eval_type": "plot_decision",
            }
        )

    # Write to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in eval_set:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Generated {len(eval_set)} evaluation cases.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_plot_eval_set()
