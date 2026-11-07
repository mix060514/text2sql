"""Strong-model path: one Codex Harness invocation, with hard application gates."""

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types

from .context import start_turn
from .codex_backend import run_codex
from .tools import complete_answer


class CodexSalesAgent(BaseAgent):
    """ADK is only the UI/session adapter; it does not orchestrate reasoning."""

    async def _run_async_impl(self, ctx):
        spec, evidence, stats = await run_codex(
            ctx.session.state["user_question"], ctx.session.state["previous_query_context"],
        )
        state = dict(ctx.session.state)
        state.update(evidence)
        state["harness_stats"] = stats
        yield Event(invocation_id=ctx.invocation_id, author=self.name,
                    actions=EventActions(state_delta=state))
        complete_answer(spec, state, require_review=False)  # receipt is the evidence; no reviewer on this path
        yield Event(
            invocation_id=ctx.invocation_id, author=self.name,
            actions=EventActions(state_delta=state),
            content=types.Content(role="model", parts=[types.Part(text=state["final_answer"])]),
        )


root_agent = CodexSalesAgent(
    name="root_agent", description="One Codex reasoning agent with bounded sales-data access.",
    before_agent_callback=start_turn,
)
