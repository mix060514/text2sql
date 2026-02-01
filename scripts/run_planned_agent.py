# @title Import necessary libraries
import os
import asyncio
import logging
import warnings
import sys

import litellm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Ignore all warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.ERROR)

# Setup path to import local src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import the NEW planned agent
from text2sql.agents.planned_agent.agent import root_agent

print("Libraries imported.")
session_service = InMemorySessionService()

litellm._turn_on_debug()

# Define constants
APP_NAME = "text2sql_app"
USER_ID = "user_1"
SESSION_ID = "session_002"


async def init_session(
    app_name: str, user_id: str, session_id: str
) -> InMemorySessionService:
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    print(
        f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'"
    )
    return session


session = asyncio.run(init_session(APP_NAME, USER_ID, SESSION_ID))
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)
print(f"Runner created for agent '{runner.agent.name}'.")


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        # print(
        #     f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Content: {event.content}"
        # )  # Debug print

        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )
            break

    print(f"<<< Agent Response: {final_response_text}")


async def run_conversation():
    # Test a simple query to trigger the workflow
    await call_agent_async(
        "Show me the top 3 sales records by amount.",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )


if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")
