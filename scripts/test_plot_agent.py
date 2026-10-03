#!/usr/bin/env python3
"""Test script to send the error prompt to root_agent."""
import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from google.adk.runners import InMemoryRunner
from google.genai import types

from text2sql.agents.planned_agent.agent import root_agent
from text2sql.runtime import stream_turn


async def test_with_prompt():
    runner = InMemoryRunner(agent=root_agent, app_name="test_plot")

    # The problematic query - testing Year+Quarter combined column
    test_question = "請給我 2023 到 2024 年各季度的銷售額趨勢"

    print(f"Question: {test_question}\n")
    print("=" * 60)

    session_id = "test_session_001"

    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="test_user",
        session_id=session_id,
    )

    content = types.Content(role="user", parts=[types.Part(text=test_question)])

    async for event in stream_turn(
        runner,
        user_id="test_user",
        session_id=session_id,
        new_message=content,
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(f"[{event.author}]: {part.text[:500]}")

    # Check final state
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id="test_user",
        session_id=session_id,
    )

    print("\n" + "=" * 60)
    print("Final State:")
    print(
        f"  user_question: {session.state.get('user_question', 'NOT SET')[:100] if session.state.get('user_question') else 'NOT SET'}"
    )
    print(
        f"  sql_query: {session.state.get('sql_query', 'NOT SET')[:200] if session.state.get('sql_query') else 'NOT SET'}..."
    )
    print(
        f"  query_result: {str(session.state.get('query_result', 'NOT SET'))[:200]}..."
    )
    print(f"  plot_decision: {session.state.get('plot_decision', 'NOT SET')}")
    print(
        f"  chart_json: {'YES (' + str(len(session.state.get('chart_image', ''))) + ' bytes)' if session.state.get('chart_image') else 'NOT SET'}"
    )


if __name__ == "__main__":
    asyncio.run(test_with_prompt())
