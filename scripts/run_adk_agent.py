# @title Import necessary libraries
import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  # For creating message Content/Parts

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.ERROR)
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from text2sql.agents.direct_agent.agent import root_agent as direct_agent

print("Libraries imported.")
session_service = InMemorySessionService()

# Define constants for identifying the interaction context
APP_NAME = "text2sql_app"
USER_ID = "user_1"
SESSION_ID = "session_001"  # Using a fixed ID for simplicity


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
    agent=direct_agent,  # The agent we want to run
    app_name=APP_NAME,  # Associates runs with our app
    session_service=session_service,  # Uses our session manager
)
print(f"Runner created for agent '{runner.agent.name}'.")


from google.genai import types  # For creating message Content/Parts


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        # You can uncomment the line below to see *all* events during execution
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif (
                event.actions and event.actions.escalate
            ):  # Handle potential errors/escalations
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )
            # Add more checks here if needed (e.g., specific error codes)
            break  # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")


async def run_conversation():
    await call_agent_async(
        "What is the weather like in London?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    await call_agent_async(
        "How about Paris?", runner=runner, user_id=USER_ID, session_id=SESSION_ID
    )  # Expecting the tool's error message

    await call_agent_async(
        "Tell me the weather in New York",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )


import asyncio

if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")
