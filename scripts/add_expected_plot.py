#!/usr/bin/env python3
"""
Script to add expected_plot field to eval_set_v2.jsonl for cases that should have visualization.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set_v2.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_set_v3.jsonl"


def determine_expected_plot(entry: dict) -> dict | None:
    """Determine if this entry should have an expected_plot field."""
    question = entry.get("question", "")
    ground_truth = entry.get("ground_truth")
    eval_type = entry.get("eval_type", "")
    tags = entry.get("tags", [])

    # Case 1: Monthly trend data (lines 57-73 equivalent)
    # These are time series that should be visualized with line charts
    if "月" in question and "趨勢" in question and eval_type == "dataframe":
        # Extract the x values from ground truth keys
        if isinstance(ground_truth, dict):
            x_order = list(ground_truth.keys())
            return {
                "should_plot": True,
                "chart_type": "line",
                "x_column": "Period",
                "y_column": "Revenue",
                "color_column": None,
                "expected_x_order": x_order,
            }

    # Case 2: Year comparison (lines 35-37 equivalent)
    # Year-over-year comparison by category
    if "各年份" in question and "銷售額總計" in question and eval_type == "dataframe":
        if isinstance(ground_truth, dict):
            return {
                "should_plot": True,
                "chart_type": "bar",
                "x_column": "Year",
                "y_column": "Total Revenue",
                "color_column": None,
                "expected_x_order": list(ground_truth.keys()),
            }

    # Case 3: Regional preference (line 82 equivalent)
    if "各個區域" in question and "最偏好" in question:
        return {
            "should_plot": True,
            "chart_type": "bar",
            "x_column": "Region",
            "y_column": "Product Category",
            "color_column": None,
            "expected_x_order": None,
        }

    # Default: No plot expected for single value or specific lookup questions
    return None


def main():
    # Read original eval set
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(entries)} entries from {EVAL_SET_PATH}")

    updated_count = 0
    updated_entries = []

    for entry in entries:
        expected_plot = determine_expected_plot(entry)
        if expected_plot:
            entry["expected_plot"] = expected_plot
            updated_count += 1
            print(f"  + Added expected_plot to: {entry['question'][:50]}...")
        updated_entries.append(entry)

    # Write updated eval set
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nUpdated {updated_count} entries")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
