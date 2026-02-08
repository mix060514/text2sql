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
    chart_image_b64 = session.state.get("chart_image", "")

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

    # Save chart image
    if chart_image_b64:
        output_path = (
            Path(__file__).parent.parent / "eval" / "results" / "test_chart.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image_bytes = base64.b64decode(chart_image_b64)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        print(f"\n✅ Chart saved to: {output_path}")
        print(f"   Image size: {len(image_bytes)} bytes")

        # Verify logic: X-axis should be Period, Y-axis should be Revenue
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

        # Check if data order is correct (for time series)
        if raw_result:
            periods = [row[0] for row in raw_result]
            expected_order = [
                "2023-Q1",
                "2023-Q2",
                "2023-Q3",
                "2023-Q4",
                "2024-Q1",
                "2024-Q2",
                "2024-Q3",
                "2024-Q4",
            ]
            if periods == expected_order:
                print("  ✅ Data is in correct chronological order")
            else:
                print(f"  ⚠️  Data order: {periods}")
                print(f"      Expected: {expected_order}")
    else:
        print("\n❌ No chart image generated!")


if __name__ == "__main__":
    asyncio.run(test_and_save_chart())
