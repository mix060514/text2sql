#!/usr/bin/env python3
"""Verify the chart image is logically correct."""
import sys
import os
import asyncio
import base64
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from google.adk.runners import InMemoryRunner
from google.genai import types

from text2sql.agents.planned_agent.agent import root_agent


async def test_and_save_chart():
    runner = InMemoryRunner(agent=root_agent, app_name="test_plot")
    test_question = "請給我 2023 到 2024 年各區域每季度的銷售額趨勢"

    print(f"Question: {test_question}\n")
    print("=" * 60)

    session_id = "test_session_verify"

    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="test_user",
        session_id=session_id,
    )

    content = types.Content(role="user", parts=[types.Part(text=test_question)])

    async for event in runner.run_async(
        user_id="test_user",
        session_id=session_id,
        new_message=content,
    ):
        pass  # Just run to completion

    # Get session state
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id="test_user",
        session_id=session_id,
    )

    # Verify data
    raw_result = session.state.get("raw_query_result", [])
    column_names = session.state.get("column_names", [])
    plot_decision = session.state.get("plot_decision", {})
    chart_json = session.state.get("chart_json", "")

    print("\n" + "=" * 60)
    print("DATA VERIFICATION:")
    print(f"  Columns: {column_names}")
    print(f"  Rows: {len(raw_result)}")
    print("\n  Raw Data:")
    for row in raw_result:
        print(f"    {row}")

    print("\n" + "=" * 60)
    print("PLOT DECISION:")
    print(f"  should_plot: {plot_decision.get('should_plot')}")
    print(f"  chart_type: {plot_decision.get('chart_type')}")
    print(f"  x_column: {plot_decision.get('x_column')}")
    print(f"  y_column: {plot_decision.get('y_column')}")
    print(f"  color_column: {plot_decision.get('color_column')}")

    # Verify chart JSON
    if chart_json:
        import json

        try:
            chart_dict = json.loads(chart_json)
            print("\n✅ Chart JSON is valid.")
            print(f"   JSON length: {len(chart_json)} chars")

            # Check for data/layout keys
            if "data" in chart_dict and "layout" in chart_dict:
                print(
                    "   ✅ JSON contains 'data' and 'layout' keys (valid Plotly structure)."
                )
            else:
                print("   ⚠️ JSON missing standard Plotly keys.")

        except json.JSONDecodeError as e:
            print(f"\n❌ Invalid JSON format: {e}")
            return

        # Simple logic check
        print("\n" + "=" * 60)
        print("LOGIC CHECK:")

        x_col = plot_decision.get("x_column")
        y_col = plot_decision.get("y_column")

        if x_col not in column_names:
            print(f"  ❌ x_column '{x_col}' not in data columns!")
        else:
            print(f"  ✅ x_column '{x_col}' exists in data")

        if y_col not in column_names:
            print(f"  ❌ y_column '{y_col}' not in data columns!")
        else:
            print(f"  ✅ y_column '{y_col}' exists in data")

    else:
        print("\n❌ No chart JSON generated!")


if __name__ == "__main__":
    asyncio.run(test_and_save_chart())
