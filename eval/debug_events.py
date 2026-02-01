"""
Debug script to examine event stream structure from SequentialAgent + LoopAgent
"""

import json
import time
import asyncio
import uuid
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from text2sql.agents.planned_agent.agent import root_agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

session_service = InMemorySessionService()
APP_NAME = "debug_eval"
USER_ID = "debug_user"


async def debug_run():
    question = "亞太區 的總銷售額是多少？"
    session_id = f"debug_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=question)])

    print(f"Question: {question}")
    print("=" * 60)
    print("Event Stream:")
    print("=" * 60)

    event_count = 0
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        event_count += 1
        author = getattr(event, "author", "unknown")
        is_final = (
            event.is_final_response() if hasattr(event, "is_final_response") else False
        )

        print(f"\n--- Event #{event_count} ---")
        print(f"Author: {author}")
        print(f"Is Final: {is_final}")

        if event.content and event.content.parts:
            for i, part in enumerate(event.content.parts):
                print(f"  Part {i}:")

                # Text
                if hasattr(part, "text") and part.text:
                    text_preview = (
                        part.text[:200] + "..." if len(part.text) > 200 else part.text
                    )
                    print(f"    Text: {text_preview}")

                # Function call
                if part.function_call:
                    print(f"    Function Call: {part.function_call.name}")
                    print(f"    Args: {part.function_call.args}")

                # Function response
                if part.function_response:
                    print(f"    Function Response: {part.function_response.name}")
                    resp = part.function_response.response
                    resp_preview = (
                        str(resp)[:300] + "..." if len(str(resp)) > 300 else str(resp)
                    )
                    print(f"    Response: {resp_preview}")

        if is_final:
            print("\n*** FINAL RESPONSE ***")
            break

    print("\n" + "=" * 60)
    print(f"Total events: {event_count}")


if __name__ == "__main__":
    asyncio.run(debug_run())
