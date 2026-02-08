import sys
import os
from unittest.mock import MagicMock

# Adjust path to find modules
sys.path.append("/home/mix060514/pj/text2sql/src")

from text2sql.agents.planned_agent.agent import render_chart_tool
from google.adk.tools import ToolContext


def test_render_chart_tool():
    # Mock ToolContext
    tool_context = MagicMock(spec=ToolContext)
    tool_context.state = {}

    print("Test 1: Normal Data")
    tool_context.state["plot_decision"] = {"chart_type": "bar"}
    tool_context.state["raw_query_result"] = [("Category A", 100), ("Category B", 200)]

    # We mock render_chart in agent.py to avoid actual matplotlib usage if possible?
    # Or strict test: let it run (might fail if no display, but we use Agg backend)

    # The agent.py imports render_chart from tools.py.
    # We can rely on render_chart returning a dict with "success" or "error".

    result = render_chart_tool(tool_context)
    print(f"Result: {result.keys()}")
    if result.get("success"):
        print("  Success: True")
        if "image" in result:
            print("  Image generated (base64 length):", len(result["image"]))
    else:
        print("  Error:", result.get("error"))

    print("\nTest 2: String Values (should cast)")
    tool_context.state["raw_query_result"] = [("Cat A", "50"), ("Cat B", "60.5")]
    result = render_chart_tool(tool_context)
    if result.get("success"):
        print("  Success: True")
    else:
        print("  Error:", result.get("error"))

    print("\nTest 3: Invalid Data (should fail/empty)")
    tool_context.state["raw_query_result"] = [("A", "bad"), ("B", "worse")]
    result = render_chart_tool(tool_context)
    print("  Result success:", result.get("success"))
    print("  Error:", result.get("error"))

    print("\nTest 4: Empty Data")
    tool_context.state["raw_query_result"] = []
    result = render_chart_tool(tool_context)
    print("  Result success:", result.get("success"))
    print("  Error:", result.get("error"))

    print("\nTest 7: Plotly - Grouped Bar Chart")
    tool_context.state["raw_query_result"] = [
        ("2023", "Q1", 100),
        ("2023", "Q2", 120),
        ("2024", "Q1", 110),
        ("2024", "Q2", 130),
    ]
    tool_context.state["column_names"] = ["Year", "Quarter", "Sales"]

    tool_context.state["plot_decision"] = {
        "chart_type": "bar",
        "x_column": "Quarter",
        "y_column": "Sales",
        "color_column": "Year",
    }

    result = render_chart_tool(tool_context)
    print("  Success:", result.get("success"))
    if result.get("success"):
        print("  Image len:", len(result.get("image", "")))
    else:
        print("  Error:", result.get("error"))

    print("\nTest 8: Plotly - Simple Line Chart")
    tool_context.state["raw_query_result"] = [("Jan", 10), ("Feb", 20), ("Mar", 15)]
    tool_context.state["column_names"] = ["Month", "Revenue"]

    tool_context.state["plot_decision"] = {
        "chart_type": "line",
        "x_column": "Month",
        "y_column": "Revenue",
    }

    result = render_chart_tool(tool_context)
    print("  Success:", result.get("success"))
    if not result.get("success"):
        print("  Error:", result.get("error"))


if __name__ == "__main__":
    try:
        test_render_chart_tool()
    except Exception as e:
        print(f"Test crashed: {e}")
        import traceback

        traceback.print_exc()
