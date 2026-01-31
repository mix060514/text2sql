#!/usr/bin/env python3
"""Quick test script for Agent V6"""
import asyncio
import sys

sys.path.insert(0, "/home/mix060514/pj/text2sql/src")

from text2sql.agents.planned_agent.agent_v6 import planned_agent_v6
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types


async def test_agent_v6():
    # Create session
    session_service = InMemorySessionService()
    session_id = "test_session"
    user_id = "test_user"
    app_name = "test_app"

    # Create session properly
    session = await session_service.create_session(session_id=session_id)

    # Create runner
    runner = Runner(
        agent=planned_agent_v6,
        app_name=app_name,
        session_service=session_service,
    )

    # Test question
    question = "North America 的總銷售額是多少？"
    content = types.Content(role="user", parts=[types.Part(text=question)])

    print(f"=== Testing Agent V6 ===")
    print(f"Question: {question}\n")

    event_count = 0
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        event_count += 1
        print(f"\n--- Event {event_count} ---")

        if event.content and event.content.parts:
            for i, part in enumerate(event.content.parts):
                if part.text:
                    print(f"Text Part {i}: {part.text[:200]}")
                if part.function_call:
                    print(f"Function Call: {part.function_call.name}")
                    print(f"  Args: {part.function_call.args}")
                if part.function_response:
                    print(f"Function Response: {part.function_response.name}")
                    print(f"  Response: {str(part.function_response.response)[:200]}")

        if event.is_final_response():
            print(f"\n=== FINAL RESPONSE ===")
            if event.content and event.content.parts:
                print(event.content.parts[0].text)
            break

        if event_count > 30:
            print("\n!!! Too many events, stopping...")
            break


if __name__ == "__main__":
    asyncio.run(test_agent_v6())
