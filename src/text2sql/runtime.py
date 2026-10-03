"""One invocation boundary: ADK call budget, deadline, and full event stream."""

import asyncio
import uuid

from google.adk.agents.run_config import RunConfig
from google.adk.events import Event, EventActions
from google.genai import types

from text2sql.config import AGENT_TIMEOUT_SECONDS, MAX_LLM_CALLS


async def stream_turn(runner, *, user_id: str, session_id: str, new_message):
    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            async for event in runner.run_async(
                user_id=user_id, session_id=session_id, new_message=new_message,
                run_config=RunConfig(max_llm_calls=MAX_LLM_CALLS),
            ):
                yield event
    except Exception as exc:
        message = "這次分析未能完成，請稍後重試。"
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id,
        )
        event = Event(
            author=runner.agent.name,
            content=types.Content(role="model", parts=[types.Part(text=message)]),
            actions=EventActions(state_delta={
                "workflow_status": "failed",
                "workflow_error": {"code": type(exc).__name__, "message": str(exc)[:300]},
                "answer": message, "final_answer": message, "chart_json": None, "chart_data": None,
                "plot_decision": {"should_plot": False, "chart_type": None},
            }),
        )
        if session:
            await runner.session_service.append_event(session=session, event=event)
        yield event


async def run_question(runner, prompt: str, user_id: str = "eval_user") -> dict:
    session_id = uuid.uuid4().hex
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id,
    )
    async for _ in stream_turn(
        runner, user_id=user_id, session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        pass
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id,
    )
    return session.state if session else {}
