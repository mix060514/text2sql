import sys
import os
import unittest
import sqlite3
import pandas as pd
import plotly.express as px
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from text2sql.agents.planned_agent.tools import execute_sql
from text2sql.agents.planned_agent.agent import ensure_chart_context, render_chart_tool


class MockContext:
    def __init__(self, state):
        self.state = state


class TestFixesAndFeatures(unittest.TestCase):

    def test_04_end_to_end_chart_summary(self):
        """Test end-to-end chart generation returns SUMMARY, not image (Bug Fix)."""
        print("\nTest 4: End-to-End Chart Generation Summary Return")

        # User's complex query
        sql_query = """
SELECT 
    strftime('%Y', "Order Date") AS Year,
    CASE 
        WHEN strftime('%m', "Order Date") IN ('01', '02', '03') THEN 'Q1'
        WHEN strftime('%m', "Order Date") IN ('04', '05', '06') THEN 'Q2'
        WHEN strftime('%m', 'Order Date') IN ('07', '08', '09') THEN 'Q3'
        WHEN strftime('%m', 'Order Date') IN ('10', '11', '12') THEN 'Q4'
        ELSE 'Unknown'
    END AS Quarter,
    "Region",
    SUM("Total Revenue") AS Revenue
FROM sales_data
GROUP BY strftime('%Y', "Order Date"), 
         Quarter,
         "Region"
ORDER BY Year, Quarter;
        """

        state = {
            "sql_query": sql_query,
            "plot_decision": {
                "chart_type": "bar",
                "x_column": "Quarter",
                "y_column": "Revenue",
                "color_column": "Region",
            },
        }
        ctx = MockContext(state)

        # 1. Execute SQL
        print("  Executing SQL...")
        res_str = execute_sql(sql_query, ctx)
        if res_str.startswith("Execution Error"):
            self.fail(f"SQL execution failed: {res_str}")

        # 2. Render Chart
        print("  Rendering Chart...")
        res = render_chart_tool(ctx)

        # New check: result should NOT contain image key, but state SHOULD.
        self.assertTrue(res["success"])
        self.assertNotIn("image", res)  # KEY FIX VERIFICATION
        self.assertIn("message", res)
        self.assertEqual(res["message"], "Chart generated and stored in state.")

        # Verify state has image
        self.assertIn("chart_image", ctx.state)
        self.assertTrue(len(ctx.state["chart_image"]) > 1000)

        print(
            f"  Chart generated successfully. Return value is safe (no image). State has image ({len(ctx.state['chart_image'])} bytes)."
        )


if __name__ == "__main__":
    unittest.main()
